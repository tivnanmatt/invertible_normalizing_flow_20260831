#!/usr/bin/env python
"""step6_haar_patchify.py -- a Haar pyramid AS the patchifier.

TarFlow patchifies an image into a systematic non-overlapping grid of
patch x patch blocks in raster order, so token 0 is the top-left corner patch:
the autoregression starts from an arbitrary corner and sweeps left-to-right.

This step replaces that choice. A depth-L Mallat 2D Haar pyramid is applied to
the folded image before patchify and undone after, with L chosen so the LL_L
band exactly fills one patch (32 -> 16 -> 8 -> 4 for patch 4, so L = 3). The
subband layout then hands patchify a token 0 that is a 4x4 THUMBNAIL OF THE
WHOLE IMAGE -- an orthonormal rescaling of average-pooling the image down to a
single patch -- with later tokens carrying progressively finer detail bands.
The model autoregresses coarse-to-fine over the image instead of left-to-right
over corner patches, at identical cost (T is unchanged at 64).

The pyramid is orthogonal (Q Q^T = I, |det| = 1, logdet 0), so the metablock
likelihood accounting and the N(0,I) prior are untouched; variants differ ONLY
in the basis of the autoregression. Both properties are verified at startup and
the check is recorded in provenance.

Variants are per-block slot schedules (see the config):
  I  = standard patchify order        F  = standard patchify reversed
  H  = Haar-pyramid domain            Hf = Haar-pyramid domain, reversed
so baseline_flip = [I,F,I,F] (official) and haar_alt = [H,I,H,F] alternates a
Haar-domain round with a pixel-domain round taken in the opposite order.

Dataset is CIFAR-10: the step-4 analysis predicts the pixel basis loses its
MNIST-specific advantage on natural images, and this is the test of that.

Usage:
  python step6_haar_patchify.py [--variants haar_alt] [--profile reduced]
  python step6_haar_patchify.py --collect
"""

import argparse
import csv
import json
import math
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
DEFAULT_CONFIG = REPO_ROOT / "configs" / "step6_haar_patchify.yml"

import invlib  # noqa: E402
import step1_datasets  # noqa: E402
import step2_tarflow as s2  # noqa: E402
from step4_tarflow_ablation import PadTo  # noqa: E402


def load_config(path=DEFAULT_CONFIG):
    with open(path) as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------
# startup verification -- the scientific claim rests on these
# --------------------------------------------------------------------------

def verify_basis(cfg, verbose=True):
    """Confirm (a) patchify is a systematic grid, (b) the pyramid is orthogonal
    with |det| = 1, (c) it round-trips, and (d) token 0 is the thumbnail."""
    N, p, ch = cfg["img_size"], cfg["patch_size"], cfg["channel_size"]
    L = cfg["haar_levels"]
    T = (N // p) ** 2
    checks = {}

    # (a) patchify == non-overlapping raster grid of p x p blocks
    idx = torch.arange(N * N, dtype=torch.float64).reshape(1, 1, N, N)
    tok = F.unfold(idx, p, stride=p).transpose(1, 2)
    checks["patchify_is_grid"] = bool(
        torch.equal(tok[0, 0].reshape(p, p), idx[0, 0, :p, :p])
        and torch.equal(tok[0, 1].reshape(p, p), idx[0, 0, :p, p:2 * p]))

    H = invlib.HaarPyramid2D(N, p, ch, levels=L)

    # (b) orthogonality of the induced (N*N, N*N) map, per channel
    eye = torch.eye(N * N, dtype=torch.float64).reshape(N * N, 1, N, N)
    Q = H._dwt(eye, L).reshape(N * N, N * N)
    orth_err = (Q @ Q.T - torch.eye(N * N, dtype=torch.float64)).abs().max().item()
    checks["orthogonality_max_err"] = orth_err
    checks["abs_logdet"] = abs(torch.linalg.slogdet(Q)[1].item())

    # (c) round-trip through the SeqTransform interface, on token layout
    x = torch.randn(4, T, ch * p * p, dtype=torch.float64)
    rt = H.forward(H.forward(x, dim=1, inverse=False), dim=1, inverse=True)
    checks["roundtrip_max_err"] = (x - rt).abs().max().item()

    # (d) token 0 is the whole-image thumbnail (== average-pool * 2**L)
    img = torch.randn(4, ch, N, N, dtype=torch.float64)
    toks = H.forward(F.unfold(img, p, stride=p).transpose(1, 2), dim=1)
    tok0 = toks[:, 0].reshape(4, ch, p, p)
    pooled = F.avg_pool2d(img, 1 << L) * float(1 << L)
    checks["token0_is_thumbnail"] = bool(torch.allclose(tok0, pooled, atol=1e-9))

    ok = (checks["patchify_is_grid"] and orth_err < 1e-9
          and checks["abs_logdet"] < 1e-9 and checks["roundtrip_max_err"] < 1e-9
          and checks["token0_is_thumbnail"])
    checks["all_passed"] = ok
    if verbose:
        print("basis verification:")
        print(f"  patchify is a systematic p x p raster grid : {checks['patchify_is_grid']}")
        print(f"  Q Q^T = I, max err                         : {orth_err:.3e}")
        print(f"  |logdet|                                   : {checks['abs_logdet']:.3e}")
        print(f"  round-trip max err                         : {checks['roundtrip_max_err']:.3e}")
        print(f"  token 0 == whole-image thumbnail           : {checks['token0_is_thumbnail']}")
        print(f"  ALL PASSED                                 : {ok}", flush=True)
    if not ok:
        raise SystemExit("basis verification FAILED -- refusing to train")
    return checks


# --------------------------------------------------------------------------
# variant construction: one slot transform per metablock
# --------------------------------------------------------------------------

def build_variant(schedule, cfg, num_blocks):
    """Map a slot schedule like [H, I, H, F] to SeqTransforms."""
    N, p, ch = cfg["img_size"], cfg["patch_size"], cfg["channel_size"]
    L, T = cfg["haar_levels"], (cfg["img_size"] // cfg["patch_size"]) ** 2
    if len(schedule) != num_blocks:
        raise ValueError(f"schedule {schedule} has {len(schedule)} entries, "
                         f"model has {num_blocks} blocks")
    ts = []
    for code in schedule:
        if code == "I":
            t = invlib.SeqIdentity(T)
        elif code == "F":
            t = invlib.SeqFlip(T)
        elif code == "H":
            t = invlib.HaarPyramid2D(N, p, ch, levels=L)
        elif code == "Hf":
            t = invlib.SeqCompose([invlib.HaarPyramid2D(N, p, ch, levels=L),
                                   invlib.SeqFlip(T)])
        else:
            raise KeyError(f"unknown slot code {code!r}")
        ts.append(t)
    return ts


# --------------------------------------------------------------------------
# training one variant  (protocol identical to step 4)
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


def run_variant(name, schedule, cfg, bud, device, out_data, out_figs):
    transformer_flow = s2.import_tarflow(cfg)
    torch.manual_seed(cfg["seed"])  # identical init for every variant
    np.random.seed(cfg["seed"])

    mc = cfg["model"]
    model = transformer_flow.Model(
        in_channels=cfg["channel_size"], img_size=cfg["img_size"],
        patch_size=cfg["patch_size"], channels=mc["channels"],
        num_blocks=mc["blocks"], layers_per_block=mc["layers_per_block"],
        nvp=True, num_classes=0)
    base_params = sum(p.numel() for p in model.parameters())
    for i, t in enumerate(build_variant(schedule, cfg, mc["blocks"])):
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
    curve, step, t0 = [], 0, time.time()
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
        "name": name, "schedule": schedule, "n_params": n_params,
        "transform_params": n_params - base_params,
        "val_bpd_curve": curve, "final_val_bpd": curve[-1][1],
        "test_bpd": test_bpd,
        "train_minutes": round((time.time() - t0) / 60, 2),
        "peak_mem_mb": (round(torch.cuda.max_memory_allocated() / 2**20)
                        if device != "cpu" else None),
        "device": device, "budget": bud,
    }
    print(f"[{name}] TEST BPD {test_bpd:.4f} ({result['train_minutes']} min)", flush=True)

    # checkpoint weights so sample grids can be re-rendered without retraining
    torch.save({"model": model.state_dict(), "schedule": schedule,
                "cfg": cfg, "result": result}, out_data / f"{name}_model.pth")

    if bud.get("sample", True):
        n = cfg["sample_nrow"] ** 2
        T = (cfg["img_size"] // cfg["patch_size"]) ** 2
        noise = torch.randn(n, T, cfg["channel_size"] * cfg["patch_size"] ** 2,
                            generator=torch.Generator().manual_seed(cfg["seed"]))
        with torch.no_grad():
            samples = model.reverse(noise.to(device))
        # explicit value_range: without it save_image rescales by each grid's own
        # min/max, making brightness/contrast incomparable across variants.
        tv.utils.save_image(samples.float().clamp(-1, 1),
                            out_figs / f"{name}_samples.png",
                            normalize=True, value_range=(-1, 1),
                            nrow=cfg["sample_nrow"])

    with open(out_data / f"{name}_result.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


# --------------------------------------------------------------------------
# collect
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
    if not results:
        print("nothing to collect")
        return results
    results.sort(key=lambda r: r["final_val_bpd"])

    lines = [r"\begin{tabular}{rllrrrr}", r"\hline",
             r"rank & variant & schedule & val bpd & test bpd & min & mem MB \\",
             r"\hline"]
    for i, r in enumerate(results):
        mem = r["peak_mem_mb"] if r["peak_mem_mb"] is not None else "--"
        lines.append(
            f"{i+1} & {_tex(r['name'])} & {_tex(','.join(r['schedule']))} & "
            f"{r['final_val_bpd']:.4f} & {r['test_bpd']:.4f} & "
            f"{r['train_minutes']:.1f} & {mem} \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    (out_data / "ranking_table.tex").write_text("\n".join(lines) + "\n")

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for r in results:
        ep, bpd = zip(*r["val_bpd_curve"])
        ax.plot(ep, bpd, label=f"{r['name']} ({r['final_val_bpd']:.3f})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation bits/dim")
    ax.set_title("Haar pyramid as patchifier: val bpd vs epoch (CIFAR-10)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_figs / "bpd_overlay.png", dpi=120)
    plt.close(fig)

    figs = [r"\begin{figure}[htbp]", r"\centering",
            r"\includegraphics[width=0.8\textwidth]"
            r"{outputs/step6_haar_patchify/figures/bpd_overlay.png}",
            r"\caption{Validation bits/dim per epoch, Haar-pyramid patchifier "
            r"vs the official ordering (CIFAR-10).}",
            r"\end{figure}", ""]
    for r in results:
        f = out_figs / f"{r['name']}_samples.png"
        if f.exists():
            figs += [r"\begin{figure}[htbp]", r"\centering",
                     rf"\includegraphics[width=0.42\textwidth]{{outputs/step6_haar_patchify/figures/{f.name}}}",
                     rf"\caption{{\textbf{{{_tex(r['name'])}}} ({_tex(','.join(r['schedule']))}): "
                     rf"samples after the budget (val bpd {r['final_val_bpd']:.4f}).}}",
                     r"\end{figure}", ""]
    figs.append(r"\clearpage")
    (out_data / "figures_step6.tex").write_text("\n".join(figs) + "\n")
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--profile", choices=["full", "reduced"], default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_root = REPO_ROOT / cfg["output_root"]
    out_data, out_figs = out_root / "data", out_root / "figures"
    out_data.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

    if args.collect:
        collect(cfg, out_data, out_figs)
        return

    checks = verify_basis(cfg)
    if args.verify_only:
        return

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    profile = args.profile or ("full" if device != "cpu" else "reduced")
    bud = cfg["budget"][profile]
    names = args.variants or list(cfg["variants"])
    print(f"step6: {len(names)} variants, profile={profile}, device={device}", flush=True)

    prov = dict(step="step6_haar_patchify", config=cfg, profile=profile,
                device=device, verification=checks,
                started=datetime.now(timezone.utc).isoformat())
    with open(out_data / "provenance.json", "w") as f:
        json.dump(prov, f, indent=2)

    for name in names:
        run_variant(name, cfg["variants"][name], cfg, bud, device, out_data, out_figs)

    results = collect(cfg, out_data, out_figs)
    print("ranking (best first):",
          ", ".join(f"{r['name']}={r['final_val_bpd']:.4f}" for r in results), flush=True)


if __name__ == "__main__":
    main()
