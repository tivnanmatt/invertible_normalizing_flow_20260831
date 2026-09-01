#!/usr/bin/env python
"""step7_system_bridge.py -- a TarFlow bridge from the pseudo-inverse to the image.

The flow analogue of System-embedded Diffusion Bridge Models
(Sobieski, Tivnan et al., arXiv:2506.23726). Instead of a time-indexed SDE whose
coefficients embed the measurement system, we embed the SAME system into the
BASE DENSITY of an exact-likelihood normalizing flow.

Standard TarFlow trains  log p(x) = log N(f(x); 0, I) + log|det J_f|.
Here, for a linear system y = A x + sigma*eps, we replace the standard normal by
the system's own terminal distribution -- the SDB bridge endpoint at t=1:

    log p(x|y) = log N( f(x) ; A^+y , Sigma_1 ) + log|det J_f|
    Sigma_1    = gamma A^+ Sigma A^+T + beta (I - A^+A)

This is a proper conditional density in x for every fixed y (the change of
variables integrates to 1), so it trains by exact maximum likelihood on paired
(x, y) and reports an honest conditional bits/dim.

WHY NO CONDITIONING NETWORK IS NEEDED. f is UNCONDITIONAL. At the optimum it
must act on the range space like the projector A^+A -- then f(x) - A^+y is
exactly the measurement noise -A^+n, whose law is the range block of Sigma_1 --
while on the null space it must Gaussianise the unmeasured content to
N(0, beta). The measurement enters only through the base's mean and covariance.
The conditioning that inpainting needs (what goes in the hole depends on the
surroundings) is supplied by TarFlow's own autoregression: later tokens attend
to earlier ones, so generated null content is conditioned on observed range
content. Nothing in the official model is modified.

Sampling  z ~ N(A^+y, Sigma_1),  x = f^-1(z)  is then a POSTERIOR SAMPLE: the
range space is pinned to the measurement (data consistency) and the null space
is drawn from the learned prior conditioned on it.

Note the identity map is already a sensible initialisation: with f = id the
range residual x - A^+y is exactly -A^+n, which matches Sigma_1's range block by
construction, so training starts from a well-scaled point and only has to learn
the null-space prior.

Usage:
  python step7_system_bridge.py [--systems inpaint_box sr_2x] [--profile reduced]
  python step7_system_bridge.py --collect
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
import torchvision as tv
import yaml
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "step7_system_bridge.yml"

import measlib  # noqa: E402
import step1_datasets  # noqa: E402
import step2_tarflow as s2  # noqa: E402
from step4_tarflow_ablation import PadTo  # noqa: E402

LOG_K = math.log(128.0)   # official [-1,1] 8-bit dequantization constant


def load_config(path=DEFAULT_CONFIG):
    with open(path) as f:
        return yaml.safe_load(f)


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


def build_system(cfg, sys_name, device):
    sc = dict(cfg["systems"][sys_name])
    kind = sc.pop("kind")
    shape = (cfg["channel_size"], cfg["img_size"], cfg["img_size"])
    op = measlib.build(kind, shape, **sc)
    op._s = op._s.to(device)
    return op


def cond_bpd_terms(model, op, x):
    """Return (log p(x|y) per dim, in nats) for a batch, plus the pinv recon."""
    xhat = op.pinv_recon(x)
    z_tok, _, logdets = model(x, None)
    z_img = model.unpatchify(z_tok)
    D = float(np.prod(x.shape[1:]))
    lp = op.base_logprob(z_img, xhat) / D - LOG_K      # per-dim, dequantized
    return lp + logdets, xhat


@torch.no_grad()
def eval_bpd(model, op, loader, device, seed=0):
    """Conditional bits/dim only -- no sampling, cheap enough to run each epoch."""
    model.eval()
    torch.manual_seed(seed)                       # deterministic measurement noise
    tot, n = 0.0, 0
    for x, _ in loader:
        x = s2.apply_noise(x.to(device), {"noise_type": "uniform"})
        lp, _ = cond_bpd_terms(model, op, x)
        tot += (-lp / math.log(2)).sum().item()
        n += x.size(0)
    model.train()
    return tot / n


@torch.no_grad()
def evaluate(model, op, loader, device, seed=0, n_post=4):
    """Full metrics: conditional bpd, PSNR, and DATA CONSISTENCY.

    Data consistency asks whether f^-1 actually respects the measurement: the
    range-space component of a posterior sample should reproduce A^+y. Nothing
    enforces this architecturally -- the flow has to learn to act like the
    projector A^+A on the range -- so it is an empirical check of the central
    claim, not a given.
    """
    model.eval()
    torch.manual_seed(seed)
    tot_bpd, n = 0.0, 0
    acc = dict(psnr_pinv=0.0, psnr_samp=0.0, psnr_mean=0.0, dc_rel=0.0)
    m = 0
    for bi, (x, _) in enumerate(loader):
        x = s2.apply_noise(x.to(device), {"noise_type": "uniform"})
        lp, xhat = cond_bpd_terms(model, op, x)
        tot_bpd += (-lp / math.log(2)).sum().item()
        n += x.size(0)
        if bi == 0:                                # sampling only on one batch
            k = min(16, x.size(0))
            xs, xh = x[:k], xhat[:k]
            recs = [model.reverse(model.patchify(op.sample_base(xh))).clamp(-1, 1)
                    for _ in range(n_post)]
            pr, pm = recs[0], torch.stack(recs).mean(0)
            acc["psnr_pinv"] += _psnr(xh, xs) * k
            acc["psnr_samp"] += _psnr(pr, xs) * k
            acc["psnr_mean"] += _psnr(pm, xs) * k
            # relative range-space disagreement between sample and measurement
            num = (op.project(pr) - op.project(xh)).flatten(1).norm(dim=-1)
            den = op.project(xh).flatten(1).norm(dim=-1).clamp_min(1e-8)
            acc["dc_rel"] += (num / den).mean().item() * k
            m += k
    model.train()
    m = max(m, 1)
    return (tot_bpd / n, acc["psnr_pinv"] / m, acc["psnr_samp"] / m,
            acc["psnr_mean"] / m, acc["dc_rel"] / m)


def _psnr(a, b):
    """PSNR on the [0,1] scale, averaged over the batch."""
    a01, b01 = (a.clamp(-1, 1) + 1) / 2, (b.clamp(-1, 1) + 1) / 2
    mse = ((a01 - b01) ** 2).flatten(1).mean(-1).clamp_min(1e-12)
    return (10 * torch.log10(1.0 / mse)).mean().item()


def run_system(sys_name, cfg, bud, device, out_data, out_figs):
    transformer_flow = s2.import_tarflow(cfg)
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    op = build_system(cfg, sys_name, device)
    print(f"[{sys_name}] system: {op}", flush=True)

    mc = cfg["model"]
    model = transformer_flow.Model(
        in_channels=cfg["channel_size"], img_size=cfg["img_size"],
        patch_size=cfg["patch_size"], channels=mc["channels"],
        num_blocks=mc["blocks"], layers_per_block=mc["layers_per_block"],
        nvp=True, num_classes=0).to(device)
    # model.reverse() rescales the latent by model.var; we supply our own base
    # covariance, so var must stay at its identity init (update_prior unused).
    assert torch.allclose(model.var, torch.ones_like(model.var)), \
        "model.var must be identity: the base covariance is Sigma_1, not var"

    train_loader, val_loader, test_loader = make_loaders(cfg, bud, device)
    optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.95),
                                  lr=bud["lr"], weight_decay=1e-4)
    steps_per_epoch = len(train_loader)
    total_steps, warmup = bud["epochs"] * steps_per_epoch, steps_per_epoch

    def lr_at(step):
        min_lr, max_lr = 1e-6, bud["lr"]
        if step <= warmup:
            return min_lr + step / warmup * (max_lr - min_lr)
        t = (step - warmup) / max(1, total_steps - warmup)
        return min_lr + 0.5 * (1 + math.cos(math.pi * t)) * (max_lr - min_lr)

    if device != "cpu":
        torch.cuda.reset_peak_memory_stats()
    curve, step, t0 = [], 0, time.time()
    metrics_path = out_data / f"{sys_name}_metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_bpd", "val_bpd", "epoch_seconds"])
    for epoch in range(bud["epochs"]):
        te, tot, nb = time.time(), 0.0, 0
        for x, _ in train_loader:
            x = s2.apply_noise(x.to(device), {"noise_type": "uniform"})
            step += 1
            for pg in optimizer.param_groups:
                pg["lr"] = lr_at(step)
            optimizer.zero_grad()
            lp, _ = cond_bpd_terms(model, op, x)
            loss = -lp.mean()                       # nats/dim
            loss.backward()
            if cfg.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()
            tot += loss.item() / math.log(2)        # report in bits/dim
            nb += 1
        val_bpd = eval_bpd(model, op, val_loader, device, seed=cfg["seed"])
        curve.append((epoch + 1, val_bpd))
        dt = time.time() - te
        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, tot / nb, val_bpd, round(dt, 1)])
        print(f"[{sys_name}] epoch {epoch+1}/{bud['epochs']} train_bpd {tot/nb:.4f} "
              f"val_bpd {val_bpd:.4f} ({dt:.0f}s)", flush=True)

    test_bpd, p_pinv, p_samp, p_mean, dc = evaluate(model, op, test_loader, device,
                                                    seed=cfg["seed"])
    result = {
        "name": sys_name, "system": repr(op),
        "kind": cfg["systems"][sys_name]["kind"],
        "sigma": op.sigma, "gamma": op.gamma, "beta": op.beta,
        "rank": int((op.sing() > 0).sum()), "n_null": op.n_null(),
        "logdet_sigma": op.logdet_sigma(),
        "val_bpd_curve": curve, "final_val_bpd": curve[-1][1],
        "test_cond_bpd": test_bpd,
        "psnr_pinv": p_pinv, "psnr_sample": p_samp, "psnr_post_mean": p_mean,
        "data_consistency_rel_err": dc,
        "n_params": sum(p.numel() for p in model.parameters()),
        "train_minutes": round((time.time() - t0) / 60, 2),
        "peak_mem_mb": (round(torch.cuda.max_memory_allocated() / 2**20)
                        if device != "cpu" else None),
        "device": device, "budget": bud,
    }
    print(f"[{sys_name}] TEST COND BPD {test_bpd:.4f} | PSNR pinv {p_pinv:.2f} "
          f"sample {p_samp:.2f} post-mean {p_mean:.2f} | data-consist {dc:.4f} "
          f"({result['train_minutes']} min)", flush=True)

    torch.save({"model": model.state_dict(), "cfg": cfg, "system": sys_name,
                "result": result}, out_data / f"{sys_name}_model.pth")
    _figure(model, op, test_loader, device, cfg, sys_name, out_figs)
    with open(out_data / f"{sys_name}_result.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


@torch.no_grad()
def _figure(model, op, loader, device, cfg, sys_name, out_figs):
    """Rows: ground truth | pseudo-inverse | 3 posterior samples | posterior mean."""
    model.eval()
    torch.manual_seed(cfg["seed"])
    x, _ = next(iter(loader))
    k = min(8, x.size(0))
    x = s2.apply_noise(x[:k].to(device), {"noise_type": "uniform"})
    xhat = op.pinv_recon(x)
    samples = [model.reverse(model.patchify(op.sample_base(xhat))).clamp(-1, 1)
               for _ in range(8)]
    rows = [x, xhat.clamp(-1, 1)] + samples[:3] + [torch.stack(samples).mean(0)]
    labels = ["truth", "pinv A+y", "post 1", "post 2", "post 3", "post mean"]
    grid = torch.cat(rows, 0)
    tv.utils.save_image(grid, out_figs / f"{sys_name}_posterior.png",
                        normalize=True, value_range=(-1, 1), nrow=k)
    with open(out_figs / f"{sys_name}_posterior_rows.txt", "w") as f:
        f.write("\n".join(f"row {i}: {l}" for i, l in enumerate(labels)) + "\n")
    model.train()


def _tex(s):
    return (str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&"))


def collect(cfg, out_data, out_figs):
    results = []
    for name in cfg["systems"]:
        p = out_data / f"{name}_result.json"
        if p.exists():
            results.append(json.load(open(p)))
    if not results:
        print("nothing to collect")
        return results
    results.sort(key=lambda r: r["test_cond_bpd"])
    lines = [r"\begin{tabular}{llrrrrrr}", r"\hline",
             r"system & kind & null dims & cond bpd & PSNR $A^+y$ & "
             r"PSNR sample & PSNR post-mean & DC err \\", r"\hline"]
    for r in results:
        lines.append(
            f"{_tex(r['name'])} & {_tex(r['kind'])} & {r['n_null']} & "
            f"{r['test_cond_bpd']:.4f} & {r['psnr_pinv']:.2f} & "
            f"{r['psnr_sample']:.2f} & {r['psnr_post_mean']:.2f} & "
            f"{r.get('data_consistency_rel_err', float('nan')):.4f} \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    (out_data / "ranking_table.tex").write_text("\n".join(lines) + "\n")

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for r in results:
        ep, bpd = zip(*r["val_bpd_curve"])
        ax.plot(ep, bpd, label=f"{r['name']} ({r['test_cond_bpd']:.3f})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("conditional bits/dim")
    ax.set_title("System-embedded flow bridge: conditional bpd")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_figs / "bpd_overlay.png", dpi=120)
    plt.close(fig)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--systems", nargs="*", default=None)
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

    print("measurement-system self-tests:")
    if not all(measlib.self_test().values()):
        raise SystemExit("measlib self-test FAILED -- refusing to train")
    if args.verify_only:
        return

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    profile = args.profile or ("full" if device != "cpu" else "reduced")
    bud = cfg["budget"][profile]
    names = args.systems or list(cfg["systems"])
    print(f"step7: {len(names)} systems, profile={profile}, device={device}", flush=True)

    with open(out_data / "provenance.json", "w") as f:
        json.dump(dict(step="step7_system_bridge", config=cfg, profile=profile,
                       device=device,
                       started=datetime.now(timezone.utc).isoformat()), f, indent=2)

    for name in names:
        run_system(name, cfg, bud, device, out_data, out_figs)

    results = collect(cfg, out_data, out_figs)
    print("systems (best cond bpd first):",
          ", ".join(f"{r['name']}={r['test_cond_bpd']:.4f}" for r in results), flush=True)


if __name__ == "__main__":
    main()
