#!/usr/bin/env python
"""step8_conditional_flow.py -- conditional TarFlow on the pseudo-inverse.

The simple alternative to the system-embedded bridge of step 7. Nothing is
embedded in the base density: the flow is an official TarFlow with the
standard-normal base and the official bits/dim, and the measurement enters
as an EXTRA INPUT CHANNEL of the coupling networks,

    log p(x | y) = log N( f(x; A^+y) ; 0, I ) + log|det J_f(.; A^+y)|,

where A^+y is the pseudo-inverse reconstruction of y = A x + sigma eps. The
conditioning is therefore always image-shaped no matter what shape y has --
a masked image, a thumbnail, or a sinogram: A is rectangular in general and y
could not be concatenated to x, but A^+y can. Training is exact maximum
likelihood on (x, y) pairs drawn fresh every step; a posterior sample is
x = f^-1(z; A^+y) with z ~ N(0, I). There is no stiff base, no whitening,
no closed-form target: the measurement noise lives in the conditioning input,
where the network can learn to ignore it, not in the density's target.

HOW THE CONDITIONING ENTERS (invlib.condition_on_image). A^+y is patchified
like the image, embedded by its own linear projection and positional table,
and prepended to the token sequence as a prefix. Every image token attends to
the whole prefix and, causally, to earlier image tokens; the output at the end
of the prefix supplies the affine parameters of image token 0 (zeros in the
official model) and the output at image token i those of token i+1, exactly
as in the official model. Coupling, permutations, log-determinant and the
KV-cached sampling loop are the official ones (the prefix is pushed through
the cache first), so the map is exactly invertible for every conditioning
image. The official repository is not modified: blocks are re-classed at
runtime and gain one linear layer and one positional table each.

The reported number is the conditional bits/dim of the test images given
their (freshly drawn) measurement, directly comparable with the unconditional
MNIST numbers of steps 4-6 (a conditional density can only be sharper) and
with the conditional bits/dim of step 7. Data consistency, PSNR of A^+y, of
one posterior sample and of the posterior mean, and failure counts are
reported as in step 7 so the two approaches share one table format.

Usage:
  python step8_conditional_flow.py [--systems ct_sparse sr_4x] [--profile reduced]
  python step8_conditional_flow.py --verify-only      # invertibility + logdet checks
  python step8_conditional_flow.py --collect
"""

import argparse
import copy
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
DEFAULT_CONFIG = REPO_ROOT / "configs" / "step8_conditional_flow.yml"

import invlib  # noqa: E402
import measlib  # noqa: E402
import step1_datasets  # noqa: E402
import step2_tarflow as s2  # noqa: E402
from step4_tarflow_ablation import PadTo  # noqa: E402
from step7_system_bridge import FAIL_BPD, _psnr, _tex, bpd_stats, isolated_rng  # noqa: E402

# Keys of a `systems:` entry that configure the run rather than the operator.
RUN_KEYS = ("noise_type", "noise_std", "scale_bound", "max_rewinds")


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
    sc = {k: v for k, v in cfg["systems"][sys_name].items() if k not in RUN_KEYS}
    kind = sc.pop("kind")
    shape = (cfg["channel_size"], cfg["img_size"], cfg["img_size"])
    op = measlib.build(kind, shape, **sc)
    op._s = op._s.to(device)
    return op


def run_opts(cfg, sys_name):
    sc = cfg["systems"][sys_name]
    get = lambda k, d=None: sc.get(k, cfg.get(k, d))  # noqa: E731
    noise = {"noise_type": get("noise_type", "uniform")}
    if noise["noise_type"] == "gaussian":
        noise["noise_std"] = float(get("noise_std"))
    return dict(noise=noise, scale_bound=get("scale_bound") or None,
                max_rewinds=int(get("max_rewinds", 0)))


def build_model(cfg, tf, device, scale_bound=None):
    mc = cfg["model"]
    model = tf.Model(in_channels=cfg["channel_size"], img_size=cfg["img_size"],
                     patch_size=cfg["patch_size"], channels=mc["channels"],
                     num_blocks=mc["blocks"], layers_per_block=mc["layers_per_block"],
                     nvp=True, num_classes=0).to(device)
    invlib.condition_on_image(model, cfg["channel_size"], bound=scale_bound)
    return model


def n_meas(op):
    """Number of scalar measurements: rows of A (rank for the implicit systems)."""
    return getattr(op, "n_meas", int((op.sing() > 0).sum()))


def cond_bpd_terms(model, op, x, xhat=None):
    """log p(x | A^+y) per dim in nats, with the official base and constant.

    A fresh measurement is drawn unless `xhat` is given, so training scores
    each image against a new noise realisation every step (exact conditional
    maximum likelihood) and evaluation scores real (x, y) pairs.
    """
    if xhat is None:
        xhat = op.pinv_recon(x)
    z, _, logdets = model(x, model.patchify(xhat))
    D = float(np.prod(x.shape[1:]))
    lp_base = s2.gaussian_log_prob(z) / D         # includes the -log 128 term
    return lp_base + logdets, xhat, lp_base, logdets


@torch.no_grad()
def posterior_sample(model, op, xhat, generator=None):
    """x = f^-1(z; A^+y), z ~ N(0, I)."""
    c = model.patchify(xhat)
    z = torch.randn(c.shape, device=c.device, dtype=c.dtype, generator=generator)
    return model.reverse(z, c)


@torch.no_grad()
def eval_bpd(model, op, loader, device, seed=0, noise=None):
    model.eval()
    noise = noise or {"noise_type": "uniform"}
    vals = []
    with isolated_rng(device, seed):
        for x, _ in loader:
            x = s2.apply_noise(x.to(device), noise)
            lp, *_ = cond_bpd_terms(model, op, x)
            vals.append(-lp / math.log(2))
    model.train()
    return bpd_stats(torch.cat(vals))


@torch.no_grad()
def evaluate(model, op, loader, device, seed=0, n_post=4, n_sample_img=64, noise=None):
    """Conditional bpd on the whole loader; PSNR and data consistency on the
    first `n_sample_img` images (autoregressive sampling is the slow part)."""
    model.eval()
    noise = noise or {"noise_type": "uniform"}
    vals, psnr, dc = [], dict(pinv=[], sample=[], post_mean=[]), []
    n_bad, m = 0, 0
    with isolated_rng(device, seed):
        for x, _ in loader:
            x = s2.apply_noise(x.to(device), noise)
            lp, xhat, _, _ = cond_bpd_terms(model, op, x)
            vals.append(-lp / math.log(2))
            if m < n_sample_img:
                k = min(n_sample_img - m, x.size(0))
                xs, xh = x[:k], xhat[:k]
                recs = [posterior_sample(model, op, xh) for _ in range(n_post)]
                ok = torch.stack([torch.isfinite(r.flatten(1)).all(1) for r in recs]).all(0)
                n_bad += int((~ok).sum())
                recs = [r.clamp(-1, 1) for r in recs]
                pr, pm = recs[0], torch.stack(recs).mean(0)
                psnr["pinv"].append(_psnr(xh, xs)[ok])
                psnr["sample"].append(_psnr(pr, xs)[ok])
                psnr["post_mean"].append(_psnr(pm, xs)[ok])
                num = (op.project(pr) - op.project(xh)).flatten(1).norm(dim=-1)
                den = op.project(xh).flatten(1).norm(dim=-1).clamp_min(1e-8)
                dc.append((num / den)[ok])
                m += k
    model.train()

    def avg(parts):
        t = torch.cat(parts) if parts else torch.zeros(0)
        return t.mean().item() if t.numel() else float("nan")

    return dict(bpd=bpd_stats(torch.cat(vals)),
                psnr_pinv=avg(psnr["pinv"]), psnr_sample=avg(psnr["sample"]),
                psnr_post_mean=avg(psnr["post_mean"]), dc_rel=avg(dc),
                n_sample_img=m, n_bad_samples=n_bad)


def run_system(sys_name, cfg, bud, device, out_data, out_figs):
    tf = s2.import_tarflow(cfg)
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    op = build_system(cfg, sys_name, device)
    opts = run_opts(cfg, sys_name)
    noise, scale_bound, max_rewinds = opts["noise"], opts["scale_bound"], opts["max_rewinds"]
    print(f"[{sys_name}] system: {op} | measurements {n_meas(op)} | input noise {noise} | "
          f"scale bound {scale_bound} | max rewinds {max_rewinds}", flush=True)

    model = build_model(cfg, tf, device, scale_bound)
    train_loader, val_loader, test_loader = make_loaders(cfg, bud, device)
    optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.95),
                                  lr=bud["lr"], weight_decay=1e-4)
    steps_per_epoch = len(train_loader)
    total_steps, warmup = bud["epochs"] * steps_per_epoch, steps_per_epoch

    def lr_at(step, scale=1.0):
        min_lr, max_lr = 1e-6, bud["lr"] * scale
        if step <= warmup:
            return min_lr + step / warmup * (max_lr - min_lr)
        t = (step - warmup) / max(1, total_steps - warmup)
        return min_lr + 0.5 * (1 + math.cos(math.pi * t)) * (max_lr - min_lr)

    if device != "cpu":
        torch.cuda.reset_peak_memory_stats()
    curve, step, t0 = [], 0, time.time()
    metrics_path = out_data / f"{sys_name}_metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_bpd", "val_bpd", "val_bpd_ok", "val_n_fail",
                                "base_bpd", "logdet_bpd", "n_skipped", "epoch_seconds",
                                "rewinds"])
    n_skipped, aborted, loss_ema = 0, None, None
    reject_mult = float(cfg.get("loss_reject_mult", 5.0))
    reject_abs = float(cfg.get("loss_reject_abs", 5.0))
    # Same safeguards as step 7 (batch rejection, rewind-on-collapse); every
    # rejected batch and every rewind is counted and reported.
    healthy, n_rewinds, lr_scale, best_ok, rewind_log = None, 0, 1.0, float("inf"), []
    epoch = 0
    while epoch < bud["epochs"]:
        te, tot, nb, nskip_epoch = time.time(), 0.0, 0, 0
        acc_base, acc_ld = 0.0, 0.0
        for x, _ in train_loader:
            x = s2.apply_noise(x.to(device), noise)
            step += 1
            for pg in optimizer.param_groups:
                pg["lr"] = lr_at(step, lr_scale)
            optimizer.zero_grad()
            lp, _, lp_base, ld = cond_bpd_terms(model, op, x)
            loss = -lp.mean()
            lv = loss.item()
            reject = (not math.isfinite(lv)) or (
                loss_ema is not None and lv > reject_mult * loss_ema + reject_abs)
            if reject:
                n_skipped += 1
                nskip_epoch += 1
                optimizer.zero_grad(set_to_none=True)
                continue
            loss.backward()
            if cfg.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            if not all(torch.isfinite(p.grad).all()
                       for p in model.parameters() if p.grad is not None):
                n_skipped += 1
                nskip_epoch += 1
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.step()
            loss_ema = lv if loss_ema is None else 0.99 * loss_ema + 0.01 * lv
            tot += lv / math.log(2)
            acc_base += (-lp_base.mean().item()) / math.log(2)
            acc_ld += (-ld.mean().item()) / math.log(2)
            nb += 1
        collapse = None
        if nskip_epoch > (nb + nskip_epoch) // 2:
            collapse = f"{nskip_epoch}/{nb + nskip_epoch} batches rejected at epoch {epoch+1}"
            vs = dict(mean=float("nan"), mean_ok=float("nan"), n_fail=-1)
        else:
            vs = eval_bpd(model, op, val_loader, device, seed=cfg["seed"], noise=noise)
            if not math.isfinite(vs["mean_ok"]) or vs["mean_ok"] > best_ok + 1.0:
                collapse = (f"val bpd {vs['mean_ok']:.3f} vs best {best_ok:.3f} "
                            f"at epoch {epoch+1}")
        dt = time.time() - te
        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, tot / max(nb, 1), vs["mean"], vs["mean_ok"],
                                    vs["n_fail"], acc_base / max(nb, 1), acc_ld / max(nb, 1),
                                    n_skipped, round(dt, 1), n_rewinds])
        if collapse:
            if healthy is None or n_rewinds >= max_rewinds:
                aborted = collapse + (f"; evaluated at epoch {healthy['epoch']} weights"
                                      if healthy else "")
                print(f"[{sys_name}] ABORT: {aborted}", flush=True)
                if healthy:
                    model.load_state_dict(healthy["model"])
                break
            n_rewinds += 1
            lr_scale *= 0.5
            model.load_state_dict(healthy["model"])
            optimizer.load_state_dict(healthy["opt"])
            step, loss_ema = healthy["step"], healthy["loss_ema"]
            msg = (f"REWIND {n_rewinds}/{max_rewinds}: {collapse} -> epoch "
                   f"{healthy['epoch']} weights restored, peak lr x{lr_scale:g}")
            rewind_log.append(msg)
            print(f"[{sys_name}] {msg}", flush=True)
            epoch = healthy["epoch"]
            continue
        best_ok = min(best_ok, vs["mean_ok"])
        healthy = dict(model=copy.deepcopy(model.state_dict()),
                       opt=copy.deepcopy(optimizer.state_dict()),
                       epoch=epoch + 1, step=step, loss_ema=loss_ema)
        curve.append((epoch + 1, vs["mean_ok"]))
        fail = f" val_fail {vs['n_fail']}" if vs["n_fail"] else ""
        print(f"[{sys_name}] epoch {epoch+1}/{bud['epochs']} train_bpd {tot/nb:.4f} "
              f"val_bpd {vs['mean_ok']:.4f}{fail} [base {acc_base/nb:.3f} "
              f"logdet {acc_ld/nb:.3f} skip {n_skipped}] ({dt:.0f}s)", flush=True)
        epoch += 1

    if not curve:
        curve.append((0, float("nan")))
    ev = evaluate(model, op, test_loader, device, seed=cfg["seed"], noise=noise)
    tb = ev["bpd"]
    result = {
        "name": sys_name, "system": repr(op),
        "kind": cfg["systems"][sys_name]["kind"],
        "sigma": op.sigma, "n_meas": n_meas(op),
        "rank": int((op.sing() > 0).sum()), "n_null": op.n_null(),
        "scale_bound": scale_bound, "input_noise": noise,
        "max_rewinds": max_rewinds, "n_rewinds": n_rewinds, "rewind_log": rewind_log,
        "lr_scale_final": lr_scale,
        "val_bpd_curve": curve, "final_val_bpd": curve[-1][1],
        "test_cond_bpd": tb["mean_ok"], "test_cond_bpd_all": tb["mean"],
        "test_cond_bpd_median": tb["median"], "test_n_fail": tb["n_fail"],
        "test_n": tb["n"], "fail_bpd_threshold": FAIL_BPD,
        "psnr_pinv": ev["psnr_pinv"], "psnr_sample": ev["psnr_sample"],
        "psnr_post_mean": ev["psnr_post_mean"],
        "data_consistency_rel_err": ev["dc_rel"],
        "n_sample_img": ev["n_sample_img"], "n_bad_samples": ev["n_bad_samples"],
        "n_params": sum(p.numel() for p in model.parameters()),
        "train_minutes": round((time.time() - t0) / 60, 2),
        "peak_mem_mb": (round(torch.cuda.max_memory_allocated() / 2**20)
                        if device != "cpu" else None),
        "device": device, "budget": bud,
        "n_skipped_steps": n_skipped, "aborted": aborted,
    }
    print(f"[{sys_name}] TEST COND BPD {tb['mean_ok']:.4f} (all {tb['mean']:.4g}, "
          f"median {tb['median']:.4f}, fail {tb['n_fail']}/{tb['n']}) | "
          f"PSNR pinv {ev['psnr_pinv']:.2f} sample {ev['psnr_sample']:.2f} "
          f"post-mean {ev['psnr_post_mean']:.2f} | data-consist {ev['dc_rel']:.4f} | "
          f"bad samples {ev['n_bad_samples']}/{ev['n_sample_img']} | "
          f"rewinds {n_rewinds} ({result['train_minutes']} min)", flush=True)

    torch.save({"model": model.state_dict(), "cfg": cfg, "system": sys_name,
                "result": result}, out_data / f"{sys_name}_model.pth")
    _figure(model, op, test_loader, device, cfg, sys_name, out_figs, noise)
    with open(out_data / f"{sys_name}_result.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


@torch.no_grad()
def _figure(model, op, loader, device, cfg, sys_name, out_figs, noise=None):
    """Rows: ground truth | A^+y | 3 posterior samples | posterior mean; and,
    for a system whose measurement is not an image, the measurements too."""
    model.eval()
    x, _ = next(iter(loader))
    k = min(8, x.size(0))
    with isolated_rng(device, cfg["seed"]):
        x = s2.apply_noise(x[:k].to(device), noise or {"noise_type": "uniform"})
        xhat = op.pinv_recon(x)
        samples = [posterior_sample(model, op, xhat).nan_to_num(0.0).clamp(-1, 1)
                   for _ in range(8)]
        y = op.measure(x) if hasattr(op, "n_angles") else None
    rows = [x, xhat.clamp(-1, 1)] + samples[:3] + [torch.stack(samples).mean(0)]
    labels = ["truth", "pinv A+y", "post 1", "post 2", "post 3", "post mean"]
    tv.utils.save_image(torch.cat(rows, 0), out_figs / f"{sys_name}_posterior.png",
                        normalize=True, value_range=(-1, 1), nrow=k)
    with open(out_figs / f"{sys_name}_posterior_rows.txt", "w") as f:
        f.write("\n".join(f"row {i}: {l}" for i, l in enumerate(labels)) + "\n")
    if y is not None:
        # the measurement itself, in its own shape (e.g. a sinogram: views x bins)
        y = y.reshape(k, op.n_angles, op.n_det).cpu()
        fig, axes = plt.subplots(1, k, figsize=(1.6 * k, 1.6))
        for i, ax in enumerate(axes):
            ax.imshow(y[i], cmap="gray", aspect="auto")
            ax.set_xticks([]), ax.set_yticks([])
        axes[0].set_ylabel(f"{op.n_angles} views")
        fig.suptitle(f"{sys_name}: measured sinograms ({op.n_angles} x {op.n_det}), sigma={op.sigma}",
                     fontsize=8)
        fig.tight_layout()
        fig.savefig(out_figs / f"{sys_name}_measurements.png", dpi=120)
        plt.close(fig)
    model.train()


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
    lines = [r"\begin{tabular}{llrrrrrrrrr}", r"\hline",
             r"system & kind & meas & null dims & cond bpd & fail & rewinds & PSNR $A^+y$ & "
             r"PSNR sample & PSNR post-mean & DC err \\", r"\hline"]
    for r in results:
        n_fail, n = r.get("test_n_fail", 0), r.get("test_n", 0)
        n_bad, n_img = r.get("n_bad_samples", 0), r.get("n_sample_img", 0)
        kind = r["kind"]
        noise = r.get("input_noise", {"noise_type": "uniform"})
        if noise.get("noise_type") == "gaussian":
            kind += f", gauss {noise['noise_std']:g}"
        if not r.get("scale_bound"):
            kind += ", unbounded"
        rw = f"{r.get('n_rewinds', 0)}/{r.get('max_rewinds', 0)}"
        if r.get("aborted"):
            rw += " abort"
        lines.append(
            f"{_tex(r['name'])} & {_tex(kind)} & {r['n_meas']} & {r['n_null']} & "
            f"{r['test_cond_bpd']:.4f} & {n_fail}/{n} & {rw} & {r['psnr_pinv']:.2f} & "
            f"{r['psnr_sample']:.2f} & {r['psnr_post_mean']:.2f} & "
            f"{r.get('data_consistency_rel_err', float('nan')):.4f} "
            + (f"({n_bad}/{n_img} bad) " if n_bad else "") + r"\\")
    lines += [r"\hline", r"\end{tabular}"]
    (out_data / "ranking_table.tex").write_text("\n".join(lines) + "\n")

    # left: the whole run; right: the second half, where the curves separate
    fig, (ax, axz) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [3, 2]})
    for r in results:
        ep, bpd = zip(*r["val_bpd_curve"])
        (line,) = ax.plot(ep, bpd, label=f"{r['name']} ({r['test_cond_bpd']:.3f})")
        half = [(e, b) for e, b in zip(ep, bpd) if e > len(ep) // 2]
        axz.plot(*zip(*half), color=line.get_color())
    # the unconditional value of the same model/budget (step 4 baseline_flip):
    # a conditional density can only be sharper, so every curve should end
    # below this line
    uncond = REPO_ROOT / "outputs" / "step4_tarflow_ablation" / "data" / "baseline_flip_result.json"
    if uncond.exists():
        u = json.load(open(uncond)).get("test_bpd")
        if u is not None:
            ax.axhline(u, color="k", ls="--", lw=1, label=f"unconditional, step 4 ({u:.3f})")
            axz.axhline(u, color="k", ls="--", lw=1)
    ax.set_xlabel("epoch")
    ax.set_ylabel("conditional bits/dim")
    ax.set_title("Conditional TarFlow on $A^+y$: validation bits/dim")
    ax.legend(fontsize=8)
    axz.set_xlabel("epoch")
    axz.set_title("second half", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_figs / "bpd_overlay.png", dpi=120)
    plt.close(fig)
    return results


def verify(cfg, verbose=True):
    """Checks that the conditioned model is what it claims to be.

    (1) exact inversion given the conditioning image, (2) the reported
    log-determinant against the autograd Jacobian, (3) z depends on the
    conditioning image, (4) the autoregressive structure is intact in a
    single block: z token i does not depend on image tokens > i, and does
    depend on every conditioning token.
    """
    tf = s2.import_tarflow(cfg)
    torch.manual_seed(0)
    img, patch, C = 8, 4, 1
    ok = True

    def small(blocks, bound=None):
        m = tf.Model(in_channels=C, img_size=img, patch_size=patch, channels=64,
                     num_blocks=blocks, layers_per_block=1, nvp=True, num_classes=0)
        invlib.condition_on_image(m, C, bound=bound)
        for name, p in m.named_parameters():   # non-trivial couplings (proj_out is zero-init)
            p.data.add_((0.02 if "proj_out" in name else 0.1) * torch.randn_like(p))
        return m.eval()      # float32: the official LayerNorm casts to float

    x = torch.randn(3, C, img, img)
    cimg = torch.randn(3, C, img, img)
    for blocks, bound in ((2, None), (2, 8.0), (1, None)):
        m = small(blocks, bound)
        c = m.patchify(cimg)
        z, _, ld = m(x, c)
        xr = m.reverse(z, c)
        inv_err = (xr - x).abs().max().item()

        D = C * img * img
        jac_err = 0.0
        for b in range(x.size(0)):
            f = lambda v: m(v.reshape(1, C, img, img), c[b:b + 1])[0].flatten()  # noqa: E731
            J = torch.autograd.functional.jacobian(f, x[b].flatten())
            ld_true = torch.linalg.slogdet(J)[1] / D
            jac_err = max(jac_err, abs(ld_true.item() - ld[b].item()))

        z2, _, _ = m(x, m.patchify(cimg + 0.5))
        cond_dep = (z2 - z).abs().max().item()

        ar_ok = True
        if blocks == 1:                   # identity permutation: token order = sequence order
            T = z.size(1)
            for j in range(T):
                xt = m.patchify(x).clone()
                xt[:, j] += 1.0
                zt, _, _ = m(m.unpatchify(xt), c)
                d = (zt - z).abs().amax(dim=(0, 2))
                ar_ok &= bool((d[:j] < 1e-12).all()) and bool(d[j] > 1e-6)
            for j in range(T):
                ct = c.clone()
                ct[:, j] += 1.0
                zt, _, _ = m(x, ct)
                d = (zt - z).abs().amax(dim=(0, 2))
                ar_ok &= bool((d > 1e-6).all())    # every image token sees every cond token
        this = inv_err < 1e-4 and jac_err < 1e-4 and cond_dep > 1e-3 and ar_ok
        ok &= this
        if verbose:
            print(f"blocks {blocks} bound {bound}: inversion {inv_err:.1e} | logdet vs jacobian "
                  f"{jac_err:.1e} | cond dependence {cond_dep:.2e} | AR structure "
                  f"{'ok' if ar_ok else 'BROKEN'} | {'OK' if this else 'FAIL'}")
    if verbose:
        print("conditional TarFlow verified:", ok)
    return ok


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
    print("conditional-flow checks:")
    if not verify(cfg):
        raise SystemExit("conditional flow verification FAILED -- refusing to train")
    if args.verify_only:
        return

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    profile = args.profile or ("full" if device != "cpu" else "reduced")
    bud = cfg["budget"][profile]
    names = args.systems or list(cfg["systems"])
    print(f"step8: {len(names)} systems, profile={profile}, device={device}", flush=True)

    with open(out_data / "provenance.json", "w") as f:
        json.dump(dict(step="step8_conditional_flow", config=cfg, profile=profile,
                       device=device, tarflow_commit=s2.tarflow_commit(cfg),
                       started=datetime.now(timezone.utc).isoformat()), f, indent=2)

    for name in names:
        run_system(name, cfg, bud, device, out_data, out_figs)

    results = collect(cfg, out_data, out_figs)
    print("systems (best cond bpd first):",
          ", ".join(f"{r['name']}={r['test_cond_bpd']:.4f}" for r in results), flush=True)


if __name__ == "__main__":
    main()
