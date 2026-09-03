#!/usr/bin/env python
"""step10_variational_posterior.py -- Bayesian inpainting by fine-tuning the prior.

One chest radiograph x* (ChestMNIST-64 test split), one measurement
    y = M (x* + sigma eps),   M = diag(mask) with a centred square hole,
and the step-9 unconditional TarFlow p_theta as the prior. Following Feng et
al. (ICCV 2023, arXiv 2304.11751, Sec. 3.2) the posterior p(x|y) is
approximated by a normalizing flow q_phi fitted by the reverse KL

    phi* = argmin E_{x ~ q_phi}[ -log p(y|x) - log p_theta(x) + log q_phi(x) ],

estimated with Monte-Carlo batches x = g_phi(z), z ~ N(0, I). The objective
equals KL(q_phi || p(x|y)) - log p(y), so the reported value is a negative
ELBO: an upper bound on -log p(y) that is tight iff q_phi is the posterior.

Adaptation to TarFlow (see the config header): q_phi has the prior's
architecture and STARTS AT the prior's weights, so step 0 is the prior and
the fit is a fine-tuning; samples and log q_phi(x) come from
invlib.differentiable_reverse, an autograd-safe, per-block checkpointed
version of TarFlow's sequential inverse. Nothing in apple/ml-tarflow is
modified.

Outputs (outputs/step10_variational_posterior/):
  data/     problem.pt, verify.json, metrics.csv (per step), eval.csv
            (periodic posterior statistics), ckpt.pth (resume), q_final.pth,
            samples.pt (final posterior samples), result.json, summary.tex
  figures/  problem.png, posterior_stepNNNNN.png, posterior_samples.png,
            posterior_summary.png, prior_vs_posterior.png, calibration.png,
            curves.png, walk_frames.png, walk.mp4

Usage:
  python step10_variational_posterior.py            # fit (auto-resumes), eval, walk
  python step10_variational_posterior.py --verify   # self-tests only
  python step10_variational_posterior.py --collect  # figures/fragments only
"""

import argparse
import copy
import csv
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from skimage.metrics import structural_similarity

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "step10_variational_posterior.yml"

import invlib  # noqa: E402
import measlib  # noqa: E402
import step2_tarflow as s2  # noqa: E402
import step9_chestmnist_tarflow as s9  # noqa: E402


def load_config(path=DEFAULT_CONFIG):
    with open(path) as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------
# the inverse problem
# --------------------------------------------------------------------------

def build_problem(cfg, cfg9, device):
    """One test image, one noisy masked measurement, all seeded."""
    bundle = s9.build_bundle(cfg9)
    test = s2.TarFlowInput(bundle.test)
    g = torch.Generator().manual_seed(cfg["seed"])
    idx = cfg.get("test_index")
    if idx is None:
        idx = int(torch.randint(len(test), (1,), generator=g))
    x, label = test[idx]
    x = x.unsqueeze(0).to(device)                                  # (1,C,N,N) in [-1,1]
    m = dict(cfg["measurement"])
    name, sigma = m.pop("name"), float(m.pop("sigma"))
    op = measlib.build(name, tuple(x.shape[1:]), sigma=sigma, **m)
    mask = op.sing(device, x.dtype).unsqueeze(0)                   # (1,C,N,N) in {0,1}
    eps = torch.randn(x.shape, generator=g).to(device)
    y = mask * (x + sigma * eps)
    return dict(x=x, y=y, mask=mask, sigma=sigma, op=op, index=idx, label=int(np.asarray(label).reshape(-1)[0]),
                n_test=len(test), n_obs=int(mask.sum()), n_hole=int((mask == 0).sum()))


def neg_log_lik(x, prob):
    """-log p(y | x) per sample, Gaussian on the observed pixels (with constant)."""
    r = prob["mask"] * (x - prob["y"])
    s2_ = prob["sigma"] ** 2
    return (r ** 2).flatten(1).sum(-1) / (2 * s2_) + 0.5 * prob["n_obs"] * math.log(2 * math.pi * s2_)


def prior_log_prob(prior, x):
    """log p_theta(x) with per-block activation checkpointing (the prior's
    parallel forward at batch 32 would otherwise hold ~6 GB of activations
    next to the reverse pass); same arithmetic as step9.log_prob."""
    h = prior.patchify(x)
    logdet = 0.0
    for blk in prior.blocks:
        h, ld = torch.utils.checkpoint.checkpoint(blk, h, None, use_reentrant=False)
        logdet = logdet + ld
    n_dims = float(np.prod(x.shape[1:]))
    return -0.5 * (h ** 2 + math.log(2 * math.pi)).flatten(1).sum(-1) + logdet * n_dims


def objective(q, prior, z, prob):
    """x = g_phi(z) and the three per-sample terms of the reverse KL."""
    x, log_q = invlib.differentiable_reverse(q, z, checkpoint=True)
    return x, neg_log_lik(x, prob), -prior_log_prob(prior, x), log_q


@torch.no_grad()
def draw_samples(model, z, batch=32):
    """x = g(z) with the official (in-place, no-grad) TarFlow reverse."""
    return torch.cat([model.reverse(z[i:i + batch].clone()) for i in range(0, z.size(0), batch)])


def latent_shape(model):
    return tuple(model.var.shape)                                   # (tokens, dims per token)


# --------------------------------------------------------------------------
# posterior statistics
# --------------------------------------------------------------------------

def _to01(a):
    return (a.clamp(-1, 1) + 1) / 2


def sample_stats(S, prob, thresh):
    """Mean/std maps and the paper's metrics from a set of posterior samples.

    Applies the paper's outlier rule first (drop samples with any |pixel| >
    thresh) and reports how many were dropped. PSNR/SSIM are on the [0,1]
    scale; 'hole' restricts the error to the unobserved pixels.
    """
    x = prob["x"]
    keep = S.abs().flatten(1).max(-1).values <= thresh
    n_out = int((~keep).sum())
    S = S[keep] if keep.any() else S
    mean, std = S.mean(0, keepdim=True), S.std(0, keepdim=True)
    hole = prob["mask"] == 0
    err01 = _to01(mean) - _to01(x)
    mse_all, mse_hole = (err01 ** 2).mean(), (err01[hole] ** 2).mean()
    ssim = structural_similarity(_to01(mean)[0, 0].cpu().numpy(), _to01(x)[0, 0].cpu().numpy(), data_range=1.0)
    zs = ((x - mean) / std.clamp_min(1e-6))[hole]                  # standardised truth in the hole
    return dict(mean=mean, std=std, n_samples=int(S.size(0)), n_outlier=n_out,
                psnr=float(10 * torch.log10(1 / mse_all.clamp_min(1e-12))),
                psnr_hole=float(10 * torch.log10(1 / mse_hole.clamp_min(1e-12))),
                ssim=float(ssim), std_hole=float(std[hole].mean()), std_obs=float(std[~hole].mean()),
                z_rms=float(zs.pow(2).mean().sqrt()), z_mean=float(zs.mean()),
                frac_within_1=float((zs.abs() < 1).float().mean()),
                frac_within_2=float((zs.abs() < 2).float().mean()))


def baseline_stats(prob):
    """Zero-fill (A^+ y) reference: PSNR of the masked measurement itself."""
    err01 = _to01(prob["y"]) - _to01(prob["x"])
    hole = prob["mask"] == 0
    return dict(psnr=float(10 * torch.log10(1 / (err01 ** 2).mean())),
                psnr_hole=float(10 * torch.log10(1 / (err01[hole] ** 2).mean())),
                ssim=float(structural_similarity(_to01(prob["y"])[0, 0].cpu().numpy(),
                                                 _to01(prob["x"])[0, 0].cpu().numpy(), data_range=1.0)))


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

def _img(ax, a, title=None, vmin=-1, vmax=1, cmap="gray"):
    im = ax.imshow(a.detach().float().cpu().squeeze(), cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=7)
    return im


def _measurement_rgb(prob):
    """The measurement with the hole tinted red (so a black hole is not confused with lung)."""
    y01 = _to01(prob["y"])[0, 0].cpu().numpy()
    hole = (prob["mask"][0, 0] == 0).cpu().numpy()
    rgb = np.stack([y01, y01, y01], -1)
    rgb[hole] = 0.75 * rgb[hole] + 0.25 * np.array([1.0, 0.0, 0.0])
    rgb[hole, 0] = np.maximum(rgb[hole, 0], 0.35)
    return rgb


def fig_problem(prob, out_figs):
    fig, axes = plt.subplots(1, 3, figsize=(5.4, 2.0))
    _img(axes[0], prob["x"], f"truth (test #{prob['index']})")
    _img(axes[1], prob["mask"], f"mask ({prob['n_obs']} observed / {prob['n_hole']} hole)", vmin=0, vmax=1)
    axes[2].imshow(_measurement_rgb(prob), interpolation="nearest")
    axes[2].axis("off")
    axes[2].set_title(f"y = M(x + {prob['sigma']}$\\,\\epsilon$)", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_figs / "problem.png", dpi=180)
    plt.close(fig)


def fig_posterior(S, st, prob, path, title, n_show=8):
    """Truth / y / mean / std / |error| (top) and n_show samples (bottom)."""
    fig, axes = plt.subplots(2, n_show, figsize=(1.35 * n_show, 3.0))
    for a in axes.flat:
        a.axis("off")
    _img(axes[0, 0], prob["x"], "truth")
    axes[0, 1].imshow(_measurement_rgb(prob), interpolation="nearest")
    axes[0, 1].set_title("measurement", fontsize=7)
    _img(axes[0, 2], st["mean"], f"post. mean  {st['psnr']:.1f} dB")
    _img(axes[0, 3], st["std"], f"post. std (hole {st['std_hole']:.3f})", vmin=0, vmax=max(0.05, float(st["std"].max())), cmap="magma")
    _img(axes[0, 4], (st["mean"] - prob["x"]).abs(), f"|mean - truth|  hole {st['psnr_hole']:.1f} dB", vmin=0, vmax=0.5, cmap="magma")
    for k in range(min(n_show, S.size(0))):
        _img(axes[1, k], S[k], "sample" if k == 0 else None)
    fig.suptitle(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_grid(S, path, nrow, title):
    n = (S.size(0) // nrow) * nrow
    fig, axes = plt.subplots(n // nrow, nrow, figsize=(1.1 * nrow, 1.1 * n / nrow + 0.3))
    for k, a in enumerate(axes.flat):
        _img(a, S[k])
    fig.suptitle(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_prior_vs_posterior(prior, q, prob, device, seed, out_figs, n=8):
    """The same latents through the prior and the fitted posterior flow."""
    T, D = latent_shape(prior)
    z = torch.randn(n, T, D, device=device, generator=torch.Generator(device=device).manual_seed(seed))
    xp, xq = draw_samples(prior, z), draw_samples(q, z)
    fig, axes = plt.subplots(2, n, figsize=(1.2 * n, 2.7))
    for k in range(n):
        _img(axes[0, k], xp[k], "prior $g_\\theta(z)$" if k == 0 else None)
        _img(axes[1, k], xq[k], "posterior $g_\\phi(z)$" if k == 0 else None)
    fig.suptitle("identical latents z through the prior (top) and the fine-tuned posterior flow (bottom)", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_figs / "prior_vs_posterior.png", dpi=160)
    plt.close(fig)
    return xp, xq


def fig_calibration(S, st, prob, out_figs):
    hole = prob["mask"] == 0
    zs = ((prob["x"] - st["mean"]) / st["std"].clamp_min(1e-6))[hole].cpu().numpy()
    err = (prob["x"] - st["mean"]).abs()[hole].cpu().numpy()
    sd = st["std"][hole].cpu().numpy()
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.5))
    lo, hi = -5, 5
    axes[0].hist(np.clip(zs, lo, hi), bins=50, range=(lo, hi), density=True, alpha=0.7, label="hole pixels")
    t = np.linspace(lo, hi, 200)
    axes[0].plot(t, np.exp(-t ** 2 / 2) / math.sqrt(2 * math.pi), "k-", lw=1, label="N(0,1)")
    axes[0].set_xlabel("(truth - posterior mean) / posterior std")
    axes[0].set_title(f"standardised truth in the hole: mean {st.get('z_mean', float('nan')):.2f}, RMS {st['z_rms']:.2f}\n"
                      f"|z|<1: {100*st['frac_within_1']:.0f}% (68% if calibrated), |z|<2: {100*st['frac_within_2']:.0f}% (95%)", fontsize=7)
    axes[0].legend(fontsize=7)
    axes[1].scatter(sd, err, s=4, alpha=0.4)
    m = max(sd.max(), err.max())
    axes[1].plot([0, m], [0, m], "k--", lw=0.8, label="|error| = std")
    axes[1].set_xlabel("posterior std")
    axes[1].set_ylabel("|truth - posterior mean|")
    axes[1].set_title("per-pixel spread vs. error (hole)", fontsize=7)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_figs / "calibration.png", dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------

def verify(q, prior, prob, batch, device):
    """The identities the whole step rests on, checked on the actual prior."""
    T, D = latent_shape(prior)
    z = torch.randn(2, T, D, device=device)
    with torch.no_grad():
        x_off = prior.reverse(z.clone())
        x_diff, log_q = invlib.differentiable_reverse(prior, z, checkpoint=False)
        log_p = s9.log_prob(prior, x_diff)
    log_p_ckpt = prior_log_prob(prior, x_diff.clone().requires_grad_(True))
    out = dict(max_abs_x_diff=float((x_diff - x_off).abs().max()),
               max_abs_logq_vs_logp=float((log_q - log_p).abs().max()),
               max_abs_ckpt_logp_diff=float((log_p_ckpt.detach() - log_p).abs().max()),
               mean_abs_logp=float(log_p.abs().mean()))
    # one full objective step at the configured batch: finiteness, time, memory
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.time()
    z = torch.randn(batch, T, D, device=device)
    x, data, nlp, lq = objective(q, prior, z, prob)
    loss = (data + nlp + lq).mean()
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
        out["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
    out["step_seconds"] = time.time() - t0
    gn = torch.nn.utils.clip_grad_norm_(q.parameters(), float("inf"))
    out.update(loss_finite=bool(torch.isfinite(loss)), grad_norm=float(gn), grad_finite=bool(torch.isfinite(gn)),
               init_data=float(data.detach().mean()), init_neg_logp=float(nlp.detach().mean()), init_log_q=float(lq.detach().mean()),
               sample_mode_left_on=any(a.attention.sample for blk in q.blocks for a in blk.attn_blocks))
    q.zero_grad(set_to_none=True)
    return out


# --------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------

def _append_csv(path, row):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def periodic_eval(q, z_eval, prob, cfg, step, out_data, out_figs, elapsed):
    S = draw_samples(q, z_eval, cfg["fit"]["batch"])
    st = sample_stats(S, prob, cfg["eval"]["outlier_thresh"])
    row = dict(step=step, psnr=st["psnr"], psnr_hole=st["psnr_hole"], ssim=st["ssim"],
               std_hole=st["std_hole"], std_obs=st["std_obs"], z_rms=st["z_rms"],
               frac_within_1=st["frac_within_1"], frac_within_2=st["frac_within_2"],
               n_outlier=st["n_outlier"], elapsed_min=elapsed / 60)
    _append_csv(out_data / "eval.csv", row)
    fig_posterior(S, st, prob, out_figs / f"posterior_step{step:05d}.png",
                  f"step {step}: {z_eval.size(0)} posterior samples (fixed noise), {st['n_outlier']} outliers dropped")
    print(f"[step10]   eval @ {step}: PSNR {st['psnr']:.2f} (hole {st['psnr_hole']:.2f}) SSIM {st['ssim']:.3f} "
          f"std hole {st['std_hole']:.3f} obs {st['std_obs']:.3f} z_rms {st['z_rms']:.2f} outliers {st['n_outlier']}", flush=True)
    return row


def fit(cfg, cfg9, device, out_data, out_figs, verify_only=False):
    torch.manual_seed(cfg["seed"])
    fc = cfg["fit"]
    prior, prior_path = s9.load_prior(cfg9, device, cfg["prior_checkpoint"])
    for p in prior.parameters():
        p.requires_grad_(False)
    q = copy.deepcopy(prior)
    for p in q.parameters():
        p.requires_grad_(True)
    prob = build_problem(cfg, cfg9, device)
    torch.save({k: v for k, v in prob.items() if k != "op"}, out_data / "problem.pt")
    fig_problem(prob, out_figs)
    print(f"[step10] prior {prior_path.name}; {prob['op']}; test image #{prob['index']}/{prob['n_test']} "
          f"(label {prob['label']}); {sum(p.numel() for p in q.parameters())/1e6:.1f}M trainable params", flush=True)

    ver = verify(q, prior, prob, fc["batch"], device)
    with open(out_data / "verify.json", "w") as f:
        json.dump(ver, f, indent=2)
    print("[step10] verify: " + ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in ver.items()), flush=True)
    assert ver["max_abs_x_diff"] < 1e-4 and ver["max_abs_logq_vs_logp"] < 1e-2 * ver["mean_abs_logp"] and ver["grad_finite"]
    if verify_only:
        return None, None, prob

    T, D = latent_shape(q)
    z_eval = torch.randn(fc["n_eval"], T, D, device=device,
                         generator=torch.Generator(device=device).manual_seed(cfg["seed"] + 1))
    opt = torch.optim.Adam(q.parameters(), lr=fc["lr"], betas=tuple(fc["betas"]))
    step, n_nonfinite, elapsed = 0, 0, 0.0
    ckpt_path = out_data / "ckpt.pth"
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device)
        q.load_state_dict(ck["q"])
        opt.load_state_dict(ck["opt"])
        step, n_nonfinite, elapsed = ck["step"], ck["n_nonfinite"], ck["elapsed"]
        torch.set_rng_state(ck["rng_cpu"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(ck["rng_cuda"], device)
        print(f"[step10] resumed at step {step} ({elapsed/60:.1f} min so far)", flush=True)
    else:
        periodic_eval(q, z_eval, prob, cfg, 0, out_data, out_figs, 0.0)   # the prior as q_phi

    def save_ckpt():
        torch.save(dict(q=q.state_dict(), opt=opt.state_dict(), step=step, n_nonfinite=n_nonfinite,
                        elapsed=elapsed, rng_cpu=torch.get_rng_state(),
                        rng_cuda=torch.cuda.get_rng_state(device) if device.type == "cuda" else None),
                   ckpt_path)

    t_start = time.time() - elapsed
    while step < fc["steps"]:
        t0 = time.time()
        z = torch.randn(fc["batch"], T, D, device=device)
        x, data, nlp, lq = objective(q, prior, z, prob)
        loss = (data + nlp + lq).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(q.parameters(), fc["grad_clip"])
        if torch.isfinite(loss) and torch.isfinite(gn):
            opt.step()
        else:
            n_nonfinite += 1
        step += 1
        elapsed = time.time() - t_start
        n_out = int((x.detach().abs().flatten(1).max(-1).values > cfg["eval"]["outlier_thresh"]).sum())
        _append_csv(out_data / "metrics.csv",
                    dict(step=step, loss=float(loss), data=float(data.mean()), neg_logp=float(nlp.mean()),
                         log_q=float(lq.mean()), grad_norm=float(gn), n_outlier=n_out,
                         step_seconds=time.time() - t0, n_nonfinite=n_nonfinite))
        if step % fc["log_every"] == 0 or step == 1:
            mem = f", peak {torch.cuda.max_memory_allocated()/2**30:.1f} GB" if device.type == "cuda" else ""
            print(f"[step10] step {step}/{fc['steps']}  -ELBO {float(loss):.1f}  data {float(data.mean()):.1f}  "
                  f"-logp {float(nlp.mean()):.1f}  logq {float(lq.mean()):.1f}  |g| {float(gn):.2f}  "
                  f"outliers {n_out}/{fc['batch']}  {time.time()-t0:.1f}s/step{mem}", flush=True)
        del x, data, nlp, lq, loss
        if step % fc["eval_every"] == 0 or step == fc["steps"]:
            periodic_eval(q, z_eval, prob, cfg, step, out_data, out_figs, elapsed)
        if step % fc["ckpt_every"] == 0 or step == fc["steps"]:
            save_ckpt()
    torch.save(q.state_dict(), out_data / "q_final.pth")
    return q, prior, prob


# --------------------------------------------------------------------------
# final evaluation and the latent random walk
# --------------------------------------------------------------------------

@torch.no_grad()
def evaluate(cfg, q, prior, prob, device, out_data, out_figs):
    ev = cfg["eval"]
    T, D = latent_shape(q)
    g = torch.Generator(device=device).manual_seed(cfg["seed"] + 2)
    z = torch.randn(ev["n_samples"], T, D, device=device, generator=g)
    S = draw_samples(q, z, cfg["fit"]["batch"])
    torch.save(dict(z=z.cpu(), samples=S.cpu()), out_data / "samples.pt")
    st = sample_stats(S, prob, ev["outlier_thresh"])
    Sp = draw_samples(prior, z, cfg["fit"]["batch"])                # the prior on the same latents
    stp = sample_stats(Sp, prob, ev["outlier_thresh"])
    base = baseline_stats(prob)

    keep = S.abs().flatten(1).max(-1).values <= ev["outlier_thresh"]
    fig_grid(S[keep][:ev["n_grid"]], out_figs / "posterior_samples.png", 8,
             f"{ev['n_grid']} posterior samples x ~ q_phi (of {ev['n_samples']}, {st['n_outlier']} outliers dropped)")
    fig_posterior(S[keep], st, prob, out_figs / "posterior_summary.png",
                  f"final: {st['n_samples']} posterior samples; PSNR {st['psnr']:.2f} dB (hole {st['psnr_hole']:.2f}), "
                  f"SSIM {st['ssim']:.3f}; zero-fill {base['psnr']:.2f} dB (hole {base['psnr_hole']:.2f})")
    fig_calibration(S[keep], st, prob, out_figs)
    fig_prior_vs_posterior(prior, q, prob, device, cfg["seed"] + 3, out_figs)

    def pack(s):
        return {k: v for k, v in s.items() if k not in ("mean", "std")}
    return dict(posterior=pack(st), prior_on_same_latents=pack(stp), zero_fill=base)


@torch.no_grad()
def latent_walk(cfg, q, prior, prob, device, out_figs):
    """z_{t+1} = cos(a) z_t + sin(a) eps keeps z_t ~ N(0,I) exactly, so every
    frame is a posterior sample and consecutive frames are correlated cos(a)."""
    w = cfg["walk"]
    T, D = latent_shape(q)
    g = torch.Generator(device=device).manual_seed(cfg["seed"] + 4)
    z = torch.randn(1, T, D, device=device, generator=g)
    zs = [z]
    ca, sa = math.cos(w["angle"]), math.sin(w["angle"])
    for _ in range(w["n_frames"] - 1):
        z = ca * z + sa * torch.randn(z.shape, device=device, generator=g)
        zs.append(z)
    Z = torch.cat(zs)
    Xq, Xp = draw_samples(q, Z, w["batch"]), draw_samples(prior, Z, w["batch"])

    # frame strip for the paper
    idx = np.linspace(0, w["n_frames"] - 1, 8).round().astype(int)
    fig, axes = plt.subplots(2, len(idx), figsize=(1.2 * len(idx), 2.7))
    for k, t in enumerate(idx):
        _img(axes[0, k], Xp[t], f"t = {t}")
        _img(axes[1, k], Xq[t])
    axes[0, 0].set_ylabel("prior"); axes[1, 0].set_ylabel("posterior")
    fig.suptitle(f"latent random walk (angle {w['angle']} rad/frame): prior (top) vs posterior flow (bottom)", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_figs / "walk_frames.png", dpi=160)
    plt.close(fig)

    # the animation
    fig, axes = plt.subplots(1, 4, figsize=(8.2, 2.35))
    axes[0].imshow(_measurement_rgb(prob), interpolation="nearest"); axes[0].set_title("measurement y", fontsize=8)
    im_p = _img(axes[1], Xp[0], "prior $g_\\theta(z_t)$")
    im_q = _img(axes[2], Xq[0], "posterior $g_\\phi(z_t)$")
    _img(axes[3], prob["x"], "truth")
    axes[0].axis("off")
    txt = fig.text(0.5, 0.02, "", ha="center", fontsize=7)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    writer = animation.FFMpegWriter(fps=w["fps"], codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    path = out_figs / "walk.mp4"
    with writer.saving(fig, path, dpi=150):
        for t in range(w["n_frames"]):
            im_p.set_data(Xp[t, 0].cpu()); im_q.set_data(Xq[t, 0].cpu())
            txt.set_text(f"latent random walk, frame {t+1}/{w['n_frames']}  (same $z_t$ through both flows)")
            writer.grab_frame()
    plt.close(fig)
    outl = int((Xq.abs().flatten(1).max(-1).values > cfg["eval"]["outlier_thresh"]).sum())
    return dict(n_frames=w["n_frames"], fps=w["fps"], angle=w["angle"], mp4=str(path),
                walk_outlier_frames=outl)


# --------------------------------------------------------------------------
# collection: curves + LaTeX fragment
# --------------------------------------------------------------------------

def _read_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]} if rows else {}


def _smooth(v, k=25):
    if len(v) < 2 * k:
        return v
    return np.convolve(v, np.ones(k) / k, mode="valid")


def collect(cfg, out_data, out_figs):
    m, e = _read_csv(out_data / "metrics.csv"), _read_csv(out_data / "eval.csv")
    res = json.load(open(out_data / "result.json")) if (out_data / "result.json").exists() else {}
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.6))
    ax = axes[0, 0]
    ax.plot(m["step"], m["loss"], color="C0", alpha=0.25, lw=0.6)
    ax.plot(m["step"][len(m["step"]) - len(_smooth(m["loss"])):], _smooth(m["loss"]), color="C0", lw=1.5)
    ax.set_title("negative ELBO  E_q[-log p(y|x) - log p(x) + log q(x)]", fontsize=8); ax.set_yscale("symlog")
    for ax, key, title in [(axes[0, 1], "data", "-log p(y|x)"), (axes[0, 2], "neg_logp", "-log p_theta(x)"),
                           (axes[1, 0], "log_q", "log q_phi(x)")]:
        ax.plot(m["step"], m[key], lw=0.6, alpha=0.4, color="C1")
        ax.plot(m["step"][len(m["step"]) - len(_smooth(m[key])):], _smooth(m[key]), lw=1.5, color="C1")
        ax.set_title(title + " (nats, batch mean)", fontsize=8)
    # the data term includes (n_obs/2) log 2 pi sigma^2, so its noise-level value is negative
    n_obs = res["problem"]["n_obs"] if res else None
    if n_obs:
        floor = 0.5 * n_obs * (1 + math.log(2 * math.pi * cfg["measurement"]["sigma"] ** 2))
        axes[0, 1].axhline(floor, color="k", ls="--", lw=0.8, label=f"noise level ({floor:,.0f})")
        axes[0, 1].legend(fontsize=6)
    axes[0, 1].set_yscale("symlog")
    ax = axes[1, 1]
    ax.plot(e["step"], e["psnr"], "o-", ms=3, label="posterior mean, all pixels")
    ax.plot(e["step"], e["psnr_hole"], "s-", ms=3, label="posterior mean, hole only")
    if res.get("zero_fill"):
        ax.axhline(res["zero_fill"]["psnr_hole"], color="k", ls="--", lw=0.8, label="zero-fill, hole")
    ax.set_title("PSNR of the posterior mean (dB)", fontsize=8); ax.legend(fontsize=6)
    ax = axes[1, 2]
    ax.plot(e["step"], e["std_hole"], "o-", ms=3, label="posterior std, hole")
    ax.plot(e["step"], e["std_obs"], "s-", ms=3, label="posterior std, observed")
    ax.axhline(cfg["measurement"]["sigma"], color="k", ls="--", lw=0.8, label="noise sigma")
    ax.set_yscale("log"); ax.set_title("posterior spread", fontsize=8); ax.legend(fontsize=6)
    for ax in axes.flat:
        ax.set_xlabel("SGD step", fontsize=8); ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out_figs / "curves.png", dpi=160)
    plt.close(fig)

    if not res:
        return
    p, pr, b, v = res["posterior"], res["prior_on_same_latents"], res["zero_fill"], res["verify"]
    # the calibration figure and the hole bias are recomputed from the saved samples (CPU)
    if (out_data / "samples.pt").exists() and (out_data / "problem.pt").exists():
        S = torch.load(out_data / "samples.pt", map_location="cpu")["samples"]
        prob = torch.load(out_data / "problem.pt", map_location="cpu")
        st = sample_stats(S, prob, cfg["eval"]["outlier_thresh"])
        fig_calibration(S, st, prob, out_figs)
        p = dict(p, z_mean=st["z_mean"])
    fc = cfg["fit"]
    first = m["loss"][:fc["log_every"]].mean(); last = m["loss"][-100:].mean()
    tex = (f"Test image \\#{res['problem']['index']} of the ChestMNIST-64 test split, centred "
           f"{cfg['measurement']['box']}$\\times${cfg['measurement']['box']} hole, noise $\\sigma={cfg['measurement']['sigma']}$ "
           f"on the $[-1,1]$ scale ({res['problem']['n_hole']} of {res['problem']['n_obs']+res['problem']['n_hole']} pixels unobserved). "
           f"Reverse-KL fine-tuning for {int(m['step'][-1])} steps of Adam (lr {fc['lr']:g}, batch {fc['batch']}, clip {fc['grad_clip']}), "
           f"{m['step_seconds'].mean():.1f}\\,s/step, {res['elapsed_min']:.0f}\\,min: the negative ELBO fell from "
           f"{first:,.0f} (mean of the first {fc['log_every']} steps; $q_\\phi=p_\\theta$) to {last:,.0f} nats (mean of the last 100); "
           f"$-\\log p(y|x)$ from {m['data'][:fc['log_every']].mean():,.0f} to {m['data'][-100:].mean():,.0f} "
           f"(noise level $\\tfrac{{n_{{\\rm obs}}}}{{2}}(1+\\log 2\\pi\\sigma^2) = {floor:,.0f}$, i.e. observed-pixel RMS residual "
           f"{cfg['measurement']['sigma'] * math.sqrt(max(1 + 2 * (m['data'][-100:].mean() - floor) / res['problem']['n_obs'], 0)):.4f} vs. $\\sigma={cfg['measurement']['sigma']}$); "
           f"{int(m['n_nonfinite'][-1])} non-finite steps skipped. "
           f"Over {cfg['eval']['n_samples']} posterior samples ({p['n_outlier']} outliers by the paper's rule): "
           f"posterior-mean PSNR {p['psnr']:.2f}\\,dB (hole {p['psnr_hole']:.2f}), SSIM {p['ssim']:.3f}, "
           f"versus zero-fill {b['psnr']:.2f}\\,dB (hole {b['psnr_hole']:.2f}), SSIM {b['ssim']:.3f}, and the prior's mean on the same "
           f"latents {pr['psnr']:.2f}\\,dB (hole {pr['psnr_hole']:.2f}); posterior std {p['std_hole']:.3f} in the hole vs. "
           f"{p['std_obs']:.3f} on observed pixels ($\\sigma={cfg['measurement']['sigma']}$); in the hole the standardised truth "
           f"has RMS {p['z_rms']:.2f}" + (f" (mean {p['z_mean']:+.2f})" if "z_mean" in p else "")
           + f" with {100*p['frac_within_1']:.0f}\\% within $\\pm1$ (68\\% if calibrated) and "
           f"{100*p['frac_within_2']:.0f}\\% within $\\pm2$ (95\\%). "
           f"Self-test on the prior: differentiable reverse vs. official reverse max $|\\Delta x|={v['max_abs_x_diff']:.1e}$, "
           f"$\\log q$ from the reverse vs. forward $\\log p$ max $|\\Delta|={v['max_abs_logq_vs_logp']:.1e}$ "
           f"(on $|\\log p|\\approx{v['mean_abs_logp']:.0f}$), one objective step {v['step_seconds']:.1f}\\,s"
           + (f" at {v['peak_mem_gb']:.1f}\\,GB peak" if "peak_mem_gb" in v else "") + ".")
    (out_data / "summary.tex").write_text(tex + "\n")
    print(tex)


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--collect", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    cfg9 = s9.load_config(REPO_ROOT / cfg["prior_config"])
    out_root = Path(cfg["output_root"])
    out_root = out_root if out_root.is_absolute() else REPO_ROOT / out_root
    out_data, out_figs = out_root / "data", out_root / "figures"
    out_data.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)
    if args.collect:
        collect(cfg, out_data, out_figs)
        return
    device = torch.device(args.device)
    t0 = time.time()
    q, prior, prob = fit(cfg, cfg9, device, out_data, out_figs, verify_only=args.verify)
    if args.verify:
        return
    res = dict(problem={k: prob[k] for k in ("index", "label", "n_test", "n_obs", "n_hole", "sigma")},
               verify=json.load(open(out_data / "verify.json")))
    res.update(evaluate(cfg, q, prior, prob, device, out_data, out_figs))
    res["walk"] = latent_walk(cfg, q, prior, prob, device, out_figs)
    res["elapsed_min"] = torch.load(out_data / "ckpt.pth", map_location="cpu")["elapsed"] / 60
    res["total_wall_min"] = (time.time() - t0) / 60
    with open(out_data / "result.json", "w") as f:
        json.dump(res, f, indent=2)
    with open(out_data / "provenance.json", "w") as f:
        json.dump(dict(config=cfg, prior_config=cfg9, tarflow_commit=s2.tarflow_commit(cfg9),
                       torch=torch.__version__, timestamp=datetime.now(timezone.utc).isoformat(),
                       argv=sys.argv), f, indent=2)
    collect(cfg, out_data, out_figs)
    print("step10 done")


if __name__ == "__main__":
    main()
