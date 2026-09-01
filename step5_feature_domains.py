#!/usr/bin/env python
"""step5_feature_domains.py -- Replace TarFlow's FEATURE EXTRACTION per block.

Step 4 mixed token vectors across the sequence position only. This step does
what was actually intended: each metablock analyzes the IMAGE into a 2D
orthogonal coefficient domain (full 2D Haar or 2D DCT, coefficients ordered
coarse-to-fine and chunked into tokens), runs the causal AR update over those
coefficients, and synthesizes back to the image domain immediately after --
so every block still maps image -> image, and blocks in different domains can
be stacked (pixel block, Haar block, ...). Token 0 is the coarse
approximation (containing the DC coefficient); finer detail tokens condition
on it.

Variants (per-block domain assignment, 4 blocks):
  pixels_flip     [I, F, I, F]                  official baseline anchor
  haar2d_alt      [I, H, F, H∘F]                alternate pixel / Haar blocks
  dct2d_alt       [I, D, F, D∘F]                alternate pixel / DCT blocks
  haar2d_flip     [H, H∘F, H, H∘F]              all Haar, direction alternates
  dct2d_flip      [D, D∘F, D, D∘F]              all DCT, direction alternates
  haar2d_pure_dc  [H, H, H, H]                  all Haar, NO flips: the coarse
                  token is invariant through the whole flow, and its 16
                  coefficients are modeled by a separate per-dimension
                  Gaussian-KDE density (fit after flow training). Exact
                  factorization p(c) = p(c0) p(rest | c0): the flow is the
                  second factor, the KDE the first. Sampling draws c0 from
                  the KDE and Gaussian noise for the rest.

Same protocol as step 4 (identical init/data/budget per variant, uniform
dequantization, grad clipping, val bpd per epoch, final test bpd, samples).

Usage:
  python step5_feature_domains.py [--variants ...] [--profile full|reduced]
  python step5_feature_domains.py --collect
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
import torchvision as tv
import yaml

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "step5_feature_domains.yml"

import invlib  # noqa: E402
import step2_tarflow as s2  # noqa: E402
import step4_tarflow_ablation as s4  # noqa: E402


def load_config(path=DEFAULT_CONFIG):
    with open(path) as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------
# per-dimension Gaussian KDE for the invariant coarse-token coefficients
# --------------------------------------------------------------------------

class KDE1D:
    """Gaussian KDE per dimension: fit(values), log_prob(x), sample(n).
    Silverman bandwidth; points subsampled for evaluation cost."""

    def __init__(self, values, max_points=4000, seed=0):
        g = torch.Generator().manual_seed(seed)
        v = values.flatten().float()
        if v.numel() > max_points:
            v = v[torch.randperm(v.numel(), generator=g)[:max_points]]
        self.v = v
        n = v.numel()
        iqr = (v.quantile(0.75) - v.quantile(0.25)).item()
        sigma = min(v.std().item(), iqr / 1.34 + 1e-12)
        self.h = max(0.9 * sigma * n ** (-0.2), 1e-3)

    def log_prob(self, x):
        x = x.flatten().float()
        d = (x.unsqueeze(1) - self.v.unsqueeze(0)) / self.h  # (m, n)
        return (torch.logsumexp(-0.5 * d * d, dim=1)
                - math.log(self.v.numel() * self.h * math.sqrt(2 * math.pi)))

    def sample(self, n, generator=None):
        idx = torch.randint(self.v.numel(), (n,), generator=generator)
        return self.v[idx] + self.h * torch.randn(n, generator=generator)


class DCModel:
    """Independent per-dimension KDEs over the invariant coarse-token
    coefficients c0 (B, C). Approximation: dims treated independently."""

    def __init__(self, c0_train, seed=0):
        self.kdes = [KDE1D(c0_train[:, d], seed=seed + d)
                     for d in range(c0_train.shape[1])]

    def log_prob(self, c0):  # (B, C) -> (B,)
        return torch.stack([self.kdes[d].log_prob(c0[:, d])
                            for d in range(c0.shape[1])], dim=1).sum(dim=1)

    def sample(self, n, seed=0):
        g = torch.Generator().manual_seed(seed)
        return torch.stack([k.sample(n, generator=g) for k in self.kdes], dim=1)


# --------------------------------------------------------------------------
# variants
# --------------------------------------------------------------------------

def build_variant(name, cfg):
    N, p = cfg["img_size"], cfg["patch_size"]
    T = (N // p) ** 2
    I, Fp = invlib.SeqIdentity, invlib.SeqFlip
    H = lambda: invlib.Haar2DFeatures(N, p, cfg["channel_size"])  # noqa: E731
    D = lambda: invlib.DCT2DFeatures(N, p, cfg["channel_size"])  # noqa: E731
    HF = lambda: invlib.SeqCompose([H(), Fp(T)])  # noqa: E731
    DF = lambda: invlib.SeqCompose([D(), Fp(T)])  # noqa: E731
    table = {
        "pixels_flip": [I(T), Fp(T), I(T), Fp(T)],
        "haar2d_alt": [I(T), H(), Fp(T), HF()],
        "dct2d_alt": [I(T), D(), Fp(T), DF()],
        "haar2d_flip": [H(), HF(), H(), HF()],
        "dct2d_flip": [D(), DF(), D(), DF()],
        "haar2d_pure_dc": [H(), H(), H(), H()],
    }
    if name not in table:
        raise KeyError(f"unknown variant {name!r}")
    ts = table[name]
    assert len(ts) == cfg["model"]["blocks"]
    return ts


# --------------------------------------------------------------------------
# bpd evaluation with an optional DC model on the invariant coarse token
# --------------------------------------------------------------------------

@torch.no_grad()
def evaluate_bpd_dc(model, loader, device, feature, dc_model):
    """bits/dim with token-0 (in the feature domain) scored by the DC model
    and all remaining coefficients by the N(0,1) prior. Exact factorization:
    p(x) = p(c0) p(rest | c0) since c0 is invariant through the flow and
    |det| = 1 for the feature transform."""
    model.eval()
    total, count = 0.0, 0
    for x, y in loader:
        x = x.to(device)
        x_int = (x + 1) * (255 / 2)
        x = ((x_int + torch.rand_like(x_int)) / 256) * 2 - 1
        z, _, logdets = model(x, None)  # z in patch-token domain
        zc = feature(z, dim=1)          # feature domain: token 0 == c0(data)
        D_total = float(np.prod(x.shape[1:]))
        c0 = zc[:, 0, :]
        rest = zc[:, 1:, :]
        log_prior = (-0.5 * (rest**2 + math.log(2 * math.pi))).flatten(1).sum(-1)
        log_dc = dc_model.log_prob(c0.cpu()).to(x.device)
        log_p = log_prior + log_dc - math.log(128) * D_total
        nll = -log_p / D_total - logdets
        total += (nll / math.log(2)).sum().item()
        count += x.size(0)
    model.train()
    return total / count


@torch.no_grad()
def collect_c0(model, loader, device, feature, max_batches=100):
    """Invariant coarse-token coefficients over (dequantized) training data."""
    outs = []
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x = x.to(device)
        x_int = (x + 1) * (255 / 2)
        x = ((x_int + torch.rand_like(x_int)) / 256) * 2 - 1
        z, _, _ = model(x, None)
        outs.append(feature(z, dim=1)[:, 0, :].cpu())
    return torch.cat(outs)


# --------------------------------------------------------------------------
# training (same protocol as step 4)
# --------------------------------------------------------------------------

def run_variant(name, cfg, bud, device, out_data, out_figs):
    transformer_flow = s2.import_tarflow(cfg)
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    mc = cfg["model"]
    T = (cfg["img_size"] // cfg["patch_size"]) ** 2
    model = transformer_flow.Model(
        in_channels=cfg["channel_size"], img_size=cfg["img_size"],
        patch_size=cfg["patch_size"], channels=mc["channels"],
        num_blocks=mc["blocks"], layers_per_block=mc["layers_per_block"],
        nvp=True, num_classes=0)
    for i, t in enumerate(build_variant(name, cfg)):
        model.blocks[i].permutation = t
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())

    train_loader, val_loader, test_loader = s4.make_loaders(cfg, bud, device)
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

    use_dc = name.endswith("_pure_dc")
    feature = invlib.Haar2DFeatures(cfg["img_size"], cfg["patch_size"],
                                    cfg["channel_size"]).to(device) if use_dc else None
    rc = {"noise_type": cfg["noise_type"]}
    curve, step = [], 0
    t0 = time.time()
    metrics_path = out_data / f"{name}_metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "loss", "val_bpd", "epoch_seconds"])
    dc_model = None
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
        if use_dc:
            dc_model = DCModel(collect_c0(model, train_loader, device, feature),
                               seed=cfg["seed"])
            val_bpd = evaluate_bpd_dc(model, val_loader, device, feature, dc_model)
        else:
            val_bpd = s2.evaluate_bpd(model, val_loader, device, False)
        curve.append((epoch + 1, val_bpd))
        dt = time.time() - te
        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, tot / nb, val_bpd, round(dt, 1)])
        print(f"[{name}] epoch {epoch+1}/{bud['epochs']} loss {tot/nb:.4f} "
              f"val_bpd {val_bpd:.4f} ({dt:.0f}s)", flush=True)

    if use_dc:
        test_bpd = evaluate_bpd_dc(model, test_loader, device, feature, dc_model)
    else:
        test_bpd = s2.evaluate_bpd(model, test_loader, device, False)
    result = {"name": name, "n_params": n_params,
              "val_bpd_curve": curve, "final_val_bpd": curve[-1][1],
              "test_bpd": test_bpd, "dc_model": use_dc,
              "train_minutes": round((time.time() - t0) / 60, 2),
              "device": device, "budget": bud}
    print(f"[{name}] TEST BPD {test_bpd:.4f} ({result['train_minutes']} min)")

    if bud.get("sample", True):
        n = cfg["sample_nrow"] ** 2
        g = torch.Generator().manual_seed(cfg["seed"])
        noise = torch.randn(n, T, cfg["channel_size"] * cfg["patch_size"] ** 2,
                            generator=g)
        if use_dc:
            # feature-domain noise: coarse token from the DC model, rest N(0,1)
            zc = feature(noise.to(device), dim=1)
            zc[:, 0, :] = dc_model.sample(n, seed=cfg["seed"]).to(device)
            noise = feature(zc, dim=1, inverse=True).cpu()
        with torch.no_grad():
            samples = model.reverse(noise.to(device))
        # clip like the official FID path: undertrained AR reverses can emit
        # rare huge values that would blank the normalized grid
        tv.utils.save_image(samples.float().clamp(-1, 1),
                            out_figs / f"{name}_samples.png",
                            normalize=True, nrow=cfg["sample_nrow"])

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
    results.sort(key=lambda r: r["final_val_bpd"])

    lines = [r"\begin{tabular}{rlrrrl}", r"\hline",
             r"rank & variant & val bpd & test bpd & min & DC model \\", r"\hline"]
    for i, r in enumerate(results):
        lines.append(f"{i+1} & {_tex(r['name'])} & {r['final_val_bpd']:.4f} & "
                     f"{r['test_bpd']:.4f} & {r['train_minutes']:.1f} & "
                     f"{'KDE' if r['dc_model'] else '--'} \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    (out_data / "ranking_table.tex").write_text("\n".join(lines) + "\n")

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for r in results:
        ep, bpd = zip(*r["val_bpd_curve"])
        ax.plot(ep, bpd, marker="o", ms=3,
                label=f"{r['name']} ({r['final_val_bpd']:.3f})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation bits/dim")
    ax.set_title("feature-domain ablation: val bpd vs epoch")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_figs / "bpd_overlay.png", dpi=120)
    plt.close(fig)

    figs = [r"\begin{figure}[htbp]", r"\centering",
            r"\includegraphics[width=0.8\textwidth]"
            r"{outputs/step5_feature_domains/figures/bpd_overlay.png}",
            r"\caption{Validation bits/dim per epoch, per-block feature-domain "
            r"variants. File: "
            r"\texttt{outputs/step5\_feature\_domains/figures/bpd\_overlay.png}.}",
            r"\end{figure}", ""]
    for r in results:
        f = out_figs / f"{r['name']}_samples.png"
        if f.exists():
            figs += [r"\begin{figure}[htbp]", r"\centering",
                     rf"\includegraphics[width=0.42\textwidth]{{outputs/step5_feature_domains/figures/{f.name}}}",
                     rf"\caption{{\textbf{{{_tex(r['name'])}}}: samples after the fixed "
                     rf"budget (val bpd {r['final_val_bpd']:.4f}"
                     + (r", coarse token drawn from the per-dim KDE" if r["dc_model"] else "")
                     + rf"). File: \texttt{{{_tex('outputs/step5_feature_domains/figures/' + f.name)}}}.}}",
                     r"\end{figure}", ""]
    figs.append(r"\clearpage")
    (out_data / "figures_step5.tex").write_text("\n".join(figs) + "\n")
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
        print(f"step5: {len(names)} variants, profile={profile}, device={device}")
        for name in names:
            run_variant(name, cfg, bud, device, out_data, out_figs)

    results = collect(cfg, out_data, out_figs)
    prov = dict(step="step5_feature_domains", config=cfg, profile=profile,
                device=device,
                command=" ".join(["python"] + (argv if argv is not None else sys.argv)),
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
