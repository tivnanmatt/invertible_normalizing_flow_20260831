#!/usr/bin/env python
"""step4_tarflow_ablation.py -- Which sequence transform helps TarFlow most?

Ablation over the TarFlow permutation slot: the official Identity/Flip patch
ordering is replaced, block by block, with verified orthogonal transforms from
invlib (step 3), so the causal transformer autoregresses over transform
COEFFICIENTS (Haar detail bands, DCT/Hartley frequencies, random rotations,
learned rotations, ...) instead of raw patch order. All candidates are
orthogonal, so the metablock likelihood accounting and the N(0,I) prior are
untouched -- variants differ ONLY in the feature basis of the autoregression.

Protocol: unconditional MNIST (zero-padded to 32x32, patch 4 -> T=64 patches),
uniform dequantization (fp32), identical model init / data order / budget for
every variant; the metric of merit is validation bits/dim after a FIXED small
number of epochs (finite-epoch generative performance), plus final test bpd,
wall time and parameter overhead.

Budget profiles: 'full' (GPU) and 'reduced' (CPU); auto-selected by CUDA
availability, or forced via --profile.

Outputs (outputs/step4_tarflow_ablation/): per-variant metrics/results,
ranking table + bpd-curve overlay + sample grids, LaTeX fragments.

Usage:
  python step4_tarflow_ablation.py [--variants haar dct] [--profile reduced]
  python step4_tarflow_ablation.py --collect
"""

import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision as tv
import yaml
from torch.utils.data import DataLoader, Dataset, Subset

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "step4_tarflow_ablation.yml"

import invlib  # noqa: E402
import step1_datasets  # noqa: E402
import step2_tarflow as s2  # noqa: E402


def load_config(path=DEFAULT_CONFIG):
    with open(path) as f:
        return yaml.safe_load(f)


class PadTo(Dataset):
    """Zero-pad square images (28->32 keeps 8-bit values valid for
    dequantization, unlike interpolation)."""

    def __init__(self, base, size):
        self.base, self.size = base, size

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i]
        p = self.size - x.shape[-1]
        x = F.pad(x, (p // 2, p - p // 2, p // 2, p - p // 2))
        return x, y


# --------------------------------------------------------------------------
# variant construction: one SeqTransform per metablock
# --------------------------------------------------------------------------

def _with_flip(t, block_idx, T):
    """Match the official alternation: reverse the coefficient order on odd
    blocks so consecutive blocks autoregress in opposite directions."""
    if block_idx % 2 == 1:
        return invlib.SeqCompose([t, invlib.SeqFlip(T)])
    return t


def build_variant(name, T, num_blocks):
    ts = []
    for i in range(num_blocks):
        if name == "baseline_flip":
            t = invlib.SeqIdentity(T) if i % 2 == 0 else invlib.SeqFlip(T)
        elif name == "identity_only":
            t = invlib.SeqIdentity(T)
        elif name == "random_perm":
            t = invlib.SeqRandomPermutation(T, seed=i)
        elif name == "haar":
            t = _with_flip(invlib.SeqHaar(T), i, T)
        elif name == "hadamard":
            t = _with_flip(invlib.SeqHadamard(T), i, T)
        elif name == "dct":
            t = _with_flip(invlib.SeqDCT(T), i, T)
        elif name == "hartley":
            t = _with_flip(invlib.SeqHartley(T), i, T)
        elif name == "rand_ortho":
            t = _with_flip(invlib.SeqRandomOrthogonal(T, seed=i), i, T)
        elif name == "householder":
            t = _with_flip(invlib.SeqHouseholder(T, k=16, seed=i), i, T)
        elif name == "cayley":
            t = _with_flip(invlib.SeqCayley(T, seed=i), i, T)
        else:
            raise KeyError(f"unknown variant {name!r}")
        ts.append(t)
    return ts


# --------------------------------------------------------------------------
# training one variant
# --------------------------------------------------------------------------

def make_loaders(cfg, bud, device):
    bundle = step1_datasets.build_dataset(cfg["dataset"], step1_datasets.load_config())
    g = torch.Generator().manual_seed(cfg["seed"])

    def subset(base, n):
        base = s2.TarFlowInput(PadTo(base, cfg["img_size"]))
        if n and n < len(base):
            idx = torch.randperm(len(base), generator=g)[:n].tolist()
            base = Subset(base, idx)
        return base

    kw = dict(batch_size=bud["batch"], num_workers=2, pin_memory=device != "cpu")
    train = DataLoader(subset(bundle.train, bud["train_subset"]), shuffle=True,
                       drop_last=True, generator=torch.Generator().manual_seed(cfg["seed"]), **kw)
    val = DataLoader(subset(bundle.val, bud["val_subset"]), shuffle=False, **kw)
    test = DataLoader(subset(bundle.test, bud["test_subset"]), shuffle=False, **kw)
    return train, val, test


def run_variant(name, cfg, bud, device, out_data, out_figs):
    transformer_flow = s2.import_tarflow(cfg)
    torch.manual_seed(cfg["seed"])  # identical init for every variant
    np.random.seed(cfg["seed"])

    mc = cfg["model"]
    T = (cfg["img_size"] // cfg["patch_size"]) ** 2
    model = transformer_flow.Model(
        in_channels=cfg["channel_size"], img_size=cfg["img_size"],
        patch_size=cfg["patch_size"], channels=mc["channels"],
        num_blocks=mc["blocks"], layers_per_block=mc["layers_per_block"],
        nvp=True, num_classes=0)
    base_params = sum(p.numel() for p in model.parameters())
    for i, t in enumerate(build_variant(name, T, mc["blocks"])):
        model.blocks[i].permutation = t
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())

    train_loader, val_loader, test_loader = make_loaders(cfg, bud, device)
    optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.95),
                                  lr=bud["lr"], weight_decay=1e-4)
    steps_per_epoch = len(train_loader)
    total_steps = bud["epochs"] * steps_per_epoch
    warmup = steps_per_epoch

    def lr_at(step):
        min_lr, max_lr = 1e-6, bud["lr"]
        if step <= warmup:
            return min_lr + step / warmup * (max_lr - min_lr)
        t = (step - warmup) / max(1, total_steps - warmup)
        return min_lr + 0.5 * (1 + math.cos(math.pi * t)) * (max_lr - min_lr)

    if device != "cpu":
        torch.cuda.reset_peak_memory_stats()
    rc = {"noise_type": cfg["noise_type"]}
    curve = []
    step = 0
    t0 = time.time()
    metrics_path = out_data / f"{name}_metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "loss", "val_bpd", "epoch_seconds"])
    for epoch in range(bud["epochs"]):
        te = time.time()
        tot, nb = 0.0, 0
        for x, y in train_loader:
            x = s2.apply_noise(x.to(device), rc)
            step += 1
            for pg in optimizer.param_groups:
                pg["lr"] = lr_at(step)
            optimizer.zero_grad()
            z, outputs, logdets = model(x, None)
            loss = model.get_loss(z, logdets)
            loss.backward()
            if cfg.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()
            tot += loss.item()
            nb += 1
        val_bpd = s2.evaluate_bpd(model, val_loader, device, False)
        curve.append((epoch + 1, val_bpd))
        dt = time.time() - te
        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, tot / nb, val_bpd, round(dt, 1)])
        print(f"[{name}] epoch {epoch+1}/{bud['epochs']} loss {tot/nb:.4f} "
              f"val_bpd {val_bpd:.4f} ({dt:.0f}s)", flush=True)

    test_bpd = s2.evaluate_bpd(model, test_loader, device, False)
    result = {
        "name": name, "n_params": n_params,
        "transform_params": n_params - base_params,
        "val_bpd_curve": curve, "final_val_bpd": curve[-1][1],
        "test_bpd": test_bpd,
        "train_minutes": round((time.time() - t0) / 60, 2),
        "peak_mem_mb": (round(torch.cuda.max_memory_allocated() / 2**20)
                        if device != "cpu" else None),
        "device": device, "budget": bud,
    }
    print(f"[{name}] TEST BPD {test_bpd:.4f} ({result['train_minutes']} min)")

    if bud.get("sample", True):
        n = cfg["sample_nrow"] ** 2
        noise = torch.randn(n, T, cfg["channel_size"] * cfg["patch_size"] ** 2,
                            generator=torch.Generator().manual_seed(cfg["seed"]))
        with torch.no_grad():
            samples = model.reverse(noise.to(device))
        tv.utils.save_image(samples.float(), out_figs / f"{name}_samples.png",
                            normalize=True, nrow=cfg["sample_nrow"])

    with open(out_data / f"{name}_result.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


# --------------------------------------------------------------------------
# collect: ranking table, overlay curves, latex fragments
# --------------------------------------------------------------------------

def _tex(s):
    s = (str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
         .replace("#", r"\#"))
    return s.replace("<", r"\textless{}").replace(">", r"\textgreater{}")


def collect(cfg, out_data, out_figs):
    results = []
    for name in cfg["variants"]:
        p = out_data / f"{name}_result.json"
        if p.exists():
            results.append(json.load(open(p)))
    results.sort(key=lambda r: r["final_val_bpd"])

    lines = [r"\begin{tabular}{rlrrrrr}", r"\hline",
             r"rank & variant & val bpd & test bpd & xform params & min & mem MB \\",
             r"\hline"]
    for i, r in enumerate(results):
        mem = r["peak_mem_mb"] if r["peak_mem_mb"] is not None else "--"
        lines.append(
            f"{i+1} & {_tex(r['name'])} & {r['final_val_bpd']:.4f} & "
            f"{r['test_bpd']:.4f} & {r['transform_params']} & "
            f"{r['train_minutes']:.1f} & {mem} \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    (out_data / "ranking_table.tex").write_text("\n".join(lines) + "\n")

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for r in results:
        ep, bpd = zip(*r["val_bpd_curve"])
        ax.plot(ep, bpd, marker="o", ms=3, label=f"{r['name']} ({r['final_val_bpd']:.3f})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation bits/dim")
    ax.set_title("TarFlow permutation-slot ablation: val bpd vs epoch")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_figs / "bpd_overlay.png", dpi=120)
    plt.close(fig)

    figs = [r"\begin{figure}[htbp]", r"\centering",
            r"\includegraphics[width=0.8\textwidth]"
            r"{outputs/step4_tarflow_ablation/figures/bpd_overlay.png}",
            r"\caption{Validation bits/dim per epoch for every permutation-slot "
            r"variant (legend sorted by final val bpd). File: "
            r"\texttt{outputs/step4\_tarflow\_ablation/figures/bpd\_overlay.png}.}",
            r"\end{figure}", ""]
    for r in results:
        f = out_figs / f"{r['name']}_samples.png"
        if f.exists():
            figs += [r"\begin{figure}[htbp]", r"\centering",
                     rf"\includegraphics[width=0.42\textwidth]{{outputs/step4_tarflow_ablation/figures/{f.name}}}",
                     rf"\caption{{\textbf{{{_tex(r['name'])}}}: samples after the fixed budget "
                     rf"(val bpd {r['final_val_bpd']:.4f}). File: "
                     rf"\texttt{{{_tex('outputs/step4_tarflow_ablation/figures/' + f.name)}}}.}}",
                     r"\end{figure}", ""]
    figs.append(r"\clearpage")
    (out_data / "figures_step4.tex").write_text("\n".join(figs) + "\n")
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--profile", choices=["full", "reduced"], default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--collect", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    out_root = REPO_ROOT / cfg["output_root"]
    out_data, out_figs = out_root / "data", out_root / "figures"
    out_data.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    profile = args.profile or ("full" if device != "cpu" else "reduced")
    bud = cfg["budget"][profile]

    if not args.collect:
        names = args.variants or cfg["variants"]
        print(f"step4: {len(names)} variants, profile={profile}, device={device}")
        for name in names:
            run_variant(name, cfg, bud, device, out_data, out_figs)

    results = collect(cfg, out_data, out_figs)
    prov = dict(step="step4_tarflow_ablation", config=cfg, profile=profile,
                device=device,
                command=" ".join(["python"] + (argv if argv is not None else sys.argv)),
                tarflow_repo="https://github.com/apple/ml-tarflow",
                tarflow_commit=s2.tarflow_commit(cfg),
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                versions=dict(python=sys.version.split()[0], torch=torch.__version__),
                ranking=[r["name"] for r in results])
    with open(out_data / "provenance.json", "w") as f:
        json.dump(prov, f, indent=2)
    if results:
        print("ranking (best first):",
              ", ".join(f"{r['name']}={r['final_val_bpd']:.4f}" for r in results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
