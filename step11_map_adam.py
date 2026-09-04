#!/usr/bin/env python
"""step11_map_adam.py -- MAP estimation with Adam for the step-10 inpainting problem.

Same measurement as step 10 (rebuilt from the step-10 config; identical y by
construction and cross-checked against step 10's saved problem.pt when it is
present), same step-9 TarFlow prior p_theta, and two maximum-a-posteriori
point estimates that differ only in the coordinates in which the posterior
density is maximised:

  latent MAP  (noise-space MAP: Asim et al. 2020 for Glow; the Adam warm start
               of Consistency Posterior Sampling, Purohit et al. CVPR 2025)
      z* = argmin_z  J_z(z) = -log p(y | g_theta(z)) - log N(z; 0, I),   x = g_theta(z*)
  image MAP   (the posterior mode in pixel coordinates; needs the prior density)
      x* = argmin_x  J_x(x) = -log p(y | x) - log p_theta(x).

With log p_theta(x) = log N(f_theta(x)) + log|det df_theta/dx| the two are
related by  J_x(g(z)) = J_z(z) - log|det df/dx|_{g(z)},  so their minimisers
coincide only where the flow's log-Jacobian is flat: a MAP estimate is not
invariant under reparametrisation, and a flow prior makes both versions
computable (the latent one costs the sequential TarFlow inverse per step, the
pixel one a parallel forward pass). Both are run from the same n_restarts
initialisations in one batch (restart 0 is deterministic, z = 0, the prior's
latent mode; the others z ~ N(0, I); the image MAP starts at the same images
x = g(z)), so that the spread of the solutions over restarts -- the only
"diversity" a diversified-MAP sampler has -- can be compared with the
posterior spread of step 10 and, in step 12, with the exact posterior.

Outputs (outputs/step11_map_adam/):
  data/     problem.pt, verify.json, metrics_{latent,image}.csv (batch means),
            trace_{latent,image}.npz (per-restart objectives per step),
            snapshots_{latent,image}.pt (images every frame_every steps),
            solutions_{latent,image}.pt (final z, x and per-restart metrics),
            refs.pt (truth / zero-fill / inits / step-10 samples under both
            objectives), result.json, summary.tex, provenance.json
  figures/  curves.png, solutions.png, restarts.png, spread.png,
            objectives.png, trajectory_frames.png, trajectory.mp4

Usage:
  python step11_map_adam.py            # verify, optimise both variants, evaluate
  python step11_map_adam.py --verify   # self-tests only
  python step11_map_adam.py --collect  # figures/fragments from saved data (CPU)
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
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from skimage.metrics import structural_similarity

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "step11_map_adam.yml"

import invlib  # noqa: E402
import step2_tarflow as s2  # noqa: E402
import step9_chestmnist_tarflow as s9  # noqa: E402
import step10_variational_posterior as s10  # noqa: E402

KINDS = ("latent", "image")
LABEL = dict(latent="latent MAP  $\\min_z -\\log p(y|g_\\theta(z)) - \\log N(z)$",
             image="image MAP  $\\min_x -\\log p(y|x) - \\log p_\\theta(x)$")


def load_config(path=DEFAULT_CONFIG):
    with open(path) as f:
        return yaml.safe_load(f)


def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


# --------------------------------------------------------------------------
# the two objectives (per restart, in nats, with all constants)
# --------------------------------------------------------------------------

def neg_log_normal(z):
    """-log N(z; 0, I) per sample."""
    return 0.5 * (z ** 2 + math.log(2 * math.pi)).flatten(1).sum(-1)


def objective_latent(prior, z, prob):
    """x = g_theta(z) through the differentiable sequential inverse; J_z = data - log N(z)."""
    x, _ = invlib.differentiable_reverse(prior, z, checkpoint=True)
    return x, s10.neg_log_lik(x, prob), neg_log_normal(z)


def objective_image(prior, x, prob):
    """J_x = data - log p_theta(x), the prior density from one (checkpointed) parallel forward."""
    return x, s10.neg_log_lik(x, prob), -s10.prior_log_prob(prior, x)


OBJECTIVE = dict(latent=objective_latent, image=objective_image)


# --------------------------------------------------------------------------
# assessing any set of images under both objectives
# --------------------------------------------------------------------------

@torch.no_grad()
def assess(prior, X, prob, Z=None, batch=32):
    """Per image: both objectives and their terms from ONE parallel forward
    pass of the prior (z = f(x), log-Jacobian), the reconstruction metrics
    against the truth, and the observed-pixel residual. If Z (the latent the
    image was generated from) is given, the round-trip error f(g(z)) - z is
    recorded too."""
    keys = ("data", "neg_log_pz", "logdet", "neg_log_p", "J_z", "J_x", "psnr", "psnr_hole", "ssim",
            "bias_hole", "obs_rms", "max_abs", "z_norm", "roundtrip")
    out = {k: [] for k in keys}
    hole = prob["mask"] == 0
    n_dims = float(np.prod(X.shape[1:]))
    x01_true = s10._to01(prob["x"])
    for i in range(0, X.size(0), batch):
        x = X[i:i + batch]
        zf, _, ld_mean = prior(x)
        logdet = ld_mean * n_dims
        nlpz = neg_log_normal(zf)
        data = s10.neg_log_lik(x, prob)
        err01 = s10._to01(x) - x01_true
        mse_all = (err01 ** 2).flatten(1).mean(-1)
        mse_hole = (err01 ** 2)[hole.expand_as(err01)].view(x.size(0), -1).mean(-1)
        out["data"].append(data)
        out["neg_log_pz"].append(nlpz)
        out["logdet"].append(logdet)
        out["neg_log_p"].append(nlpz - logdet)
        out["J_z"].append(data + nlpz)
        out["J_x"].append(data + nlpz - logdet)
        out["psnr"].append(10 * torch.log10(1 / mse_all.clamp_min(1e-12)))
        out["psnr_hole"].append(10 * torch.log10(1 / mse_hole.clamp_min(1e-12)))
        out["ssim"].append(torch.tensor([structural_similarity(s10._to01(x[k])[0].cpu().numpy(), x01_true[0, 0].cpu().numpy(),
                                                               data_range=1.0) for k in range(x.size(0))], device=x.device))
        out["bias_hole"].append((x - prob["x"])[hole.expand_as(x)].view(x.size(0), -1).mean(-1))
        out["obs_rms"].append(((prob["mask"] * (x - prob["y"])) ** 2).flatten(1).sum(-1).div(prob["n_obs"]).sqrt())
        out["max_abs"].append(x.abs().flatten(1).max(-1).values)
        out["z_norm"].append(zf.flatten(1).norm(dim=-1))
        out["roundtrip"].append((zf - Z[i:i + batch]).flatten(1).abs().max(-1).values if Z is not None
                                else torch.full((x.size(0),), float("nan"), device=x.device))
    return {k: torch.cat(v).float().cpu() for k, v in out.items()}


def summarise(a):
    return {k: dict(mean=float(v.mean()), std=float(v.std()) if v.numel() > 1 else 0.0, min=float(v.min()), max=float(v.max()))
            for k, v in a.items()}


def spread(X, prob):
    """How different the restarts' solutions are: std map and pairwise RMS in the hole."""
    hole = (prob["mask"][0] == 0)
    std = X.std(0)
    Xh = X[:, hole].flatten(1)
    pd = torch.cdist(Xh, Xh) / math.sqrt(Xh.size(1))
    iu = torch.triu_indices(X.size(0), X.size(0), 1)
    pr = pd[iu[0], iu[1]]
    return dict(std_hole=float(std[hole].mean()), std_obs=float(std[~hole].mean()),
                pair_rms_hole=float(pr.mean()), pair_rms_hole_min=float(pr.min()), pair_rms_hole_max=float(pr.max()))


# --------------------------------------------------------------------------
# the optimiser
# --------------------------------------------------------------------------

def _append_csv(path, row):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def lr_at(mc, k):
    if mc.get("lr_schedule", "cosine") == "cosine":
        return mc["lr"] * 0.5 * (1 + math.cos(math.pi * k / mc["steps"]))
    return mc["lr"]


def optimise(kind, prior, prob, init, mc, device, out_data):
    """Adam on the free variable (z for 'latent', x for 'image'); all restarts
    in one batch. The per-restart objectives are independent and Adam is
    per-coordinate, so minimising their sum runs n_restarts independent
    optimisations. Non-finite gradients are zeroed and counted."""
    v = init.clone().requires_grad_(True)
    opt = torch.optim.Adam([v], lr=mc["lr"], betas=tuple(mc["betas"]))
    K, R = mc["steps"], v.size(0)
    trace = dict(J=np.zeros((K + 1, R)), data=np.zeros((K + 1, R)), prior=np.zeros((K + 1, R)),
                 grad_norm=np.zeros((K, R)), lr=np.zeros(K), step_seconds=np.zeros(K))
    frames, frame_steps, n_nonfinite = [], [], 0
    csv_path = out_data / f"metrics_{kind}.csv"
    if csv_path.exists():
        csv_path.unlink()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    for k in range(K):
        t0 = time.time()
        for grp in opt.param_groups:
            grp["lr"] = lr_at(mc, k)
        x, data, pri = OBJECTIVE[kind](prior, v, prob)
        J = data + pri
        opt.zero_grad(set_to_none=True)
        J.sum().backward()
        bad = ~torch.isfinite(v.grad.flatten(1)).all(-1)
        if bad.any():
            n_nonfinite += int(bad.sum())
            v.grad[bad] = 0
        gn = v.grad.flatten(1).norm(dim=-1)
        opt.step()
        trace["J"][k], trace["data"][k], trace["prior"][k] = J.detach().cpu().numpy(), data.detach().cpu().numpy(), pri.detach().cpu().numpy()
        trace["grad_norm"][k], trace["lr"][k], trace["step_seconds"][k] = gn.cpu().numpy(), lr_at(mc, k), time.time() - t0
        if k % mc["frame_every"] == 0:                                 # the state BEFORE update k (k = 0: the init)
            frames.append(x.detach().cpu().clone()); frame_steps.append(k)
        if k % mc["log_every"] == 0 or k == K - 1:
            _append_csv(csv_path, dict(step=k, J_mean=float(J.mean()), J_min=float(J.min()), data=float(data.mean()),
                                       prior=float(pri.mean()), grad_norm=float(gn.mean()), lr=lr_at(mc, k),
                                       step_seconds=time.time() - t0, n_nonfinite=n_nonfinite))
            mem = f", peak {torch.cuda.max_memory_allocated()/2**30:.1f} GB" if device.type == "cuda" else ""
            print(f"[step11] {kind} step {k}/{K}  J mean {float(J.mean()):.1f} min {float(J.min()):.1f}  data {float(data.mean()):.1f}  "
                  f"prior {float(pri.mean()):.1f}  |g| {float(gn.mean()):.2f}  lr {lr_at(mc, k):.2e}  {time.time()-t0:.2f}s/step{mem}", flush=True)
        del x, data, pri, J
    # the final state (after the last update), evaluated without gradients
    with torch.no_grad():
        x, data, pri = OBJECTIVE[kind](prior, v, prob)
        trace["J"][K], trace["data"][K], trace["prior"][K] = (data + pri).cpu().numpy(), data.cpu().numpy(), pri.cpu().numpy()
        frames.append(x.cpu().clone()); frame_steps.append(K)
    elapsed = time.time() - t_start
    np.savez(out_data / f"trace_{kind}.npz", **trace)
    torch.save(dict(steps=torch.tensor(frame_steps), x=torch.stack(frames)), out_data / f"snapshots_{kind}.pt")
    print(f"[step11] {kind}: {K} steps in {elapsed/60:.1f} min, {n_nonfinite} non-finite restart-gradients zeroed", flush=True)
    return v.detach(), x.detach(), dict(elapsed_min=elapsed / 60, n_nonfinite=n_nonfinite,
                                        step_seconds=float(trace["step_seconds"].mean()),
                                        peak_mem_gb=torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else None)


# --------------------------------------------------------------------------
# self-tests
# --------------------------------------------------------------------------

def _fd_check(prior, kind, v0, prob, steps):
    """Directional derivative of the objective along its own gradient direction:
    autograd (= |grad|) vs central finite differences at several step sizes.
    fp32 throughout (the official TarFlow code casts to fp32 inside its
    norms, so an fp64 copy of the model cannot run); the gradient direction
    gives the largest signal-to-noise, and the best step size is reported."""
    v = v0.clone().requires_grad_(True)
    _, data, pri = OBJECTIVE[kind](prior, v, prob)
    (data + pri).sum().backward()
    gnorm = float(v.grad.norm())
    d = v.grad / v.grad.norm()
    with torch.no_grad():
        J = lambda u: float(sum(OBJECTIVE[kind](prior, u, prob)[1:]).sum())
        fd = {h: (J(v0 + h * d) - J(v0 - h * d)) / (2 * h) for h in steps}
    return dict(autograd=gnorm, finite_difference={str(h): f for h, f in fd.items()},
                rel_err=min(abs(gnorm - f) / max(abs(f), 1e-12) for f in fd.values()))


def verify(prior, prob, cfg, device, prob10_path):
    out = {}
    if prob10_path.exists():                                            # (a) it is the step-10 measurement
        saved = torch.load(prob10_path, map_location=device)
        out["problem_matches_step10"] = bool(saved["index"] == prob["index"] and torch.equal(saved["y"], prob["y"])
                                             and torch.equal(saved["mask"], prob["mask"]) and saved["sigma"] == prob["sigma"])
    T, D = s10.latent_shape(prior)
    g = torch.Generator(device=device).manual_seed(cfg["seed"] + 100)
    z = torch.randn(2, T, D, device=device, generator=g)
    with torch.no_grad():                                               # (b) J_x(g(z)) = J_z(z) - logdet: both code paths at the same points
        x = s10.draw_samples(prior, z)
        a = assess(prior, x, prob, Z=z)
    x_d, data, pri = objective_latent(prior, z.clone().requires_grad_(True), prob)
    out["max_abs_x_diff"] = float((x_d.detach() - x).abs().max())
    out["max_abs_Jz_diff"] = float(((data + pri).detach().cpu() - a["J_z"]).abs().max())
    out["max_abs_roundtrip_z"] = float(a["roundtrip"].max())
    out["mean_abs_Jz"] = float(a["J_z"].abs().mean())
    xi = x[:1].clone().requires_grad_(True)
    _, data_i, pri_i = objective_image(prior, xi, prob)
    out["max_abs_Jx_diff"] = float(((data_i + pri_i).detach().cpu() - a["J_x"][:1]).abs().max())
    del x_d, data, pri, data_i, pri_i
    # (c) directional finite differences of both objectives (fp32, along the gradient)
    out["fd_latent"] = _fd_check(prior, "latent", z[:1], prob, cfg["verify"]["fd_steps"])
    out["fd_image"] = _fd_check(prior, "image", x[:1], prob, cfg["verify"]["fd_steps"])
    # (d) one full step of each variant at the configured batch: time, memory, finiteness
    R = cfg["map"]["n_restarts"]
    zR = torch.randn(R, T, D, device=device, generator=g)
    for kind, v0 in (("latent", zR), ("image", s10.draw_samples(prior, zR, cfg["map"]["batch"]))):
        v = v0.clone().requires_grad_(True)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        t0 = time.time()
        _, data, pri = OBJECTIVE[kind](prior, v, prob)
        (data + pri).sum().backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        out[f"step_seconds_{kind}"] = time.time() - t0
        out[f"peak_mem_gb_{kind}"] = torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else None
        out[f"grad_finite_{kind}"] = bool(torch.isfinite(v.grad).all())
        out[f"init_J_{kind}"] = float((data + pri).detach().mean())
        del v, data, pri
    out["sample_mode_left_on"] = any(a_.attention.sample for blk in prior.blocks for a_ in blk.attn_blocks)
    return out


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

def _img(ax, a, title=None, vmin=-1, vmax=1, cmap="gray"):
    return s10._img(ax, a, title, vmin, vmax, cmap)


def fig_curves(traces, refs, prob, out_figs):
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.4))
    floor = 0.5 * prob["n_obs"] * (1 + math.log(2 * math.pi * prob["sigma"] ** 2))
    for r, kind in enumerate(KINDS):
        tr = traces[kind]
        steps = np.arange(tr["J"].shape[0])
        for c, (key, title) in enumerate((("J", "objective J (nats)"), ("data", "-log p(y|x)"),
                                           ("prior", "-log N(z)" if kind == "latent" else "-log p_theta(x)"))):
            ax = axes[r, c]
            ax.plot(steps, tr[key], color="C0" if kind == "latent" else "C1", lw=0.5, alpha=0.25)
            ax.plot(steps, np.median(tr[key], 1), color="k", lw=1.2, label="median of restarts")
            ax.set_yscale("symlog", linthresh=100)
            ax.set_title(f"{kind}: {title}", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.set_xlabel("Adam step", fontsize=8)
            ref_key = key if key != "prior" else ("neg_log_pz" if kind == "latent" else "neg_log_p")
            ref_key = {"J": "J_z" if kind == "latent" else "J_x"}.get(key, ref_key)
            if "truth" in refs:
                ax.axhline(float(refs["truth"][ref_key][0]), color="g", ls="--", lw=0.9, label="truth")
            if "step10" in refs:
                ax.axhline(float(refs["step10"][ref_key].mean()), color="m", ls=":", lw=0.9, label="step-10 samples (mean)")
            if key == "data":
                ax.axhline(floor, color="k", ls="-.", lw=0.8, label=f"noise level {floor:,.0f}")
            ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out_figs / "curves.png", dpi=160)
    plt.close(fig)


def fig_solutions(sol, refs, prob, out_figs, step10_mean=None):
    zL = sol["latent"]; xI = sol["image"]
    bL = int(zL["a"]["J_z"].argmin()); bI = int(xI["a"]["J_x"].argmin())
    ests = [("zero-fill $A^+y$", prob["y"], refs["zero_fill"], 0),
            (f"latent MAP (best of {zL['x'].size(0)})", zL["x"][bL:bL + 1], zL["a"], bL),
            (f"image MAP (best of {xI['x'].size(0)})", xI["x"][bI:bI + 1], xI["a"], bI)]
    if step10_mean is not None:
        ests.append(("step-10 posterior mean", step10_mean, refs["step10_mean"], 0))
    n = 2 + len(ests)
    fig, axes = plt.subplots(2, n, figsize=(1.45 * n, 3.1))
    for a in axes.flat:
        a.axis("off")
    _img(axes[0, 0], prob["x"], "truth")
    axes[0, 1].imshow(s10._measurement_rgb(prob), interpolation="nearest"); axes[0, 1].set_title("measurement", fontsize=7)
    _img(axes[1, 0], zL["x"].mean(0), f"mean of {zL['x'].size(0)} latent MAPs\n{float(refs['latent_mean']['psnr_hole'][0]):.1f} dB hole")
    _img(axes[1, 1], xI["x"].mean(0), f"mean of {xI['x'].size(0)} image MAPs\n{float(refs['image_mean']['psnr_hole'][0]):.1f} dB hole")
    for c, (title, x, a, i) in enumerate(ests, start=2):
        _img(axes[0, c], x, f"{title}\n{float(a['psnr'][i]):.1f} dB (hole {float(a['psnr_hole'][i]):.1f})")
        _img(axes[1, c], (x - prob["x"]).abs(), f"|error|  bias hole {float(a['bias_hole'][i]):+.3f}", vmin=0, vmax=0.5, cmap="magma")
    fig.suptitle("point estimates of the same posterior in two coordinate systems (and the step-10 variational mean)", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_figs / "solutions.png", dpi=160)
    plt.close(fig)


def fig_restarts(sol, inits, out_figs, n=8):
    n = min(n, inits.size(0))
    fig, axes = plt.subplots(3, n, figsize=(1.2 * n, 3.9))
    for k in range(n):
        _img(axes[0, k], inits[k], "init $g_\\theta(0)$" if k == 0 else f"init {k}: $g_\\theta(z)$, z~N(0,I)" if k == 1 else f"init {k}")
        _img(axes[1, k], sol["latent"]["x"][k], f"J_z {float(sol['latent']['a']['J_z'][k]):,.0f}")
        _img(axes[2, k], sol["image"]["x"][k], f"J_x {float(sol['image']['a']['J_x'][k]):,.0f}")
    axes[0, 0].set_ylabel("init"); axes[1, 0].set_ylabel("latent MAP"); axes[2, 0].set_ylabel("image MAP")
    fig.suptitle(f"the first {n} restarts: initial prior sample (top), latent MAP (middle) and image MAP (bottom) from it", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_figs / "restarts.png", dpi=160)
    plt.close(fig)


def fig_spread(sol, prob, out_figs, step10_std=None):
    panels = [(f"std over {sol['latent']['x'].size(0)} latent MAPs", sol["latent"]["x"].std(0)),
              (f"std over {sol['image']['x'].size(0)} image MAPs", sol["image"]["x"].std(0)),
              ("|latent - image MAP| (paired mean)", (sol["latent"]["x"] - sol["image"]["x"]).abs().mean(0))]
    if step10_std is not None:
        panels.append(("step-10 posterior std", step10_std))
    vmax = max(0.05, max(float(p[1].max()) for p in panels))
    fig, axes = plt.subplots(1, len(panels), figsize=(1.9 * len(panels), 2.1))
    for ax, (t, m) in zip(axes, panels):
        hole = prob["mask"][0] == 0
        _img(ax, m, f"{t}\nhole {float(m[hole].mean()):.3f} / obs {float(m[~hole].mean()):.3f}", vmin=0, vmax=vmax, cmap="magma")
    fig.tight_layout()
    fig.savefig(out_figs / "spread.png", dpi=160)
    plt.close(fig)


def fig_objectives(sol, refs, out_figs):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    ax = axes[0]
    groups = [("inits (prior samples)", refs["inits"], "0.6", "o"), ("latent MAP", sol["latent"]["a"], "C0", "o"),
              ("image MAP", sol["image"]["a"], "C1", "s"), ("truth", refs["truth"], "g", "*"), ("zero-fill", refs["zero_fill"], "k", "x")]
    if "step10" in refs:
        groups.insert(3, ("step-10 samples", refs["step10"], "m", "."))
    for name, a, col, mk in groups:
        ax.scatter(a["J_z"].numpy(), a["J_x"].numpy(), s=28 if mk == "*" else 9, c=col, marker=mk, label=name, alpha=0.8)
    ax.set_xscale("symlog", linthresh=1000); ax.set_yscale("symlog", linthresh=1000)
    ax.set_xlabel("$J_z$ = -log p(y|x) - log N(f(x))  (latent coordinates)", fontsize=7)
    ax.set_ylabel("$J_x$ = -log p(y|x) - log $p_\\theta$(x)  (pixel coordinates)", fontsize=7)
    ax.set_title("every point under both objectives", fontsize=8); ax.legend(fontsize=6); ax.tick_params(labelsize=7)
    for ax, kind, key, col in ((axes[1], "latent", "J_z", "C0"), (axes[2], "image", "J_x", "C1")):
        a = sol[kind]["a"]
        ax.scatter(a[key].numpy(), a["psnr_hole"].numpy(), s=12, c=col)
        ax.scatter(a[key].numpy()[:1], a["psnr_hole"].numpy()[:1], s=40, facecolors="none", edgecolors="k", label="restart 0 (z = 0)")
        ax.axhline(float(refs["zero_fill"]["psnr_hole"][0]), color="k", ls="--", lw=0.8, label="zero-fill")
        if "step10_mean" in refs:
            ax.axhline(float(refs["step10_mean"]["psnr_hole"][0]), color="m", ls=":", lw=0.9, label="step-10 posterior mean")
        ax.set_xlabel(f"final {key} of the restart (nats)", fontsize=7); ax.set_ylabel("hole PSNR of the restart (dB)", fontsize=7)
        ax.set_title(f"{kind} MAP: does a lower objective mean a better hole?", fontsize=8); ax.legend(fontsize=6); ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out_figs / "objectives.png", dpi=160)
    plt.close(fig)


def fig_trajectory(snaps, traces, prob, cfg, out_figs):
    """Frames of restarts 0 (z = 0) and 1 (random) for both variants, as a strip and an mp4."""
    steps = snaps["latent"]["steps"]
    assert np.array_equal(steps, snaps["image"]["steps"])
    panels = [("latent, restart 0 (z=0)", "latent", 0), ("latent, restart 1", "latent", 1),
              ("image, restart 0", "image", 0), ("image, restart 1", "image", 1)]
    idx = np.linspace(0, len(steps) - 1, 8).round().astype(int)
    fig, axes = plt.subplots(len(panels), len(idx), figsize=(1.2 * len(idx), 1.25 * len(panels) + 0.3))
    for r, (t, kind, j) in enumerate(panels):
        for c, i in enumerate(idx):
            _img(axes[r, c], snaps[kind]["x"][i, j], f"step {int(steps[i])}" if r == 0 else None)
        axes[r, 0].set_ylabel(t, fontsize=6)
    fig.suptitle("Adam trajectories (the image at each step)", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_figs / "trajectory_frames.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(panels) + 2, figsize=(2.05 * (len(panels) + 2), 2.45))
    axes[0].imshow(s10._measurement_rgb(prob), interpolation="nearest"); axes[0].set_title("measurement y", fontsize=8); axes[0].axis("off")
    ims = [_img(axes[1 + r], snaps[kind]["x"][0, j], t) for r, (t, kind, j) in enumerate(panels)]
    _img(axes[-1], prob["x"], "truth")
    txt = fig.text(0.5, 0.02, "", ha="center", fontsize=7)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    writer = animation.FFMpegWriter(fps=cfg["anim"]["fps"], codec="libx264",
                                    extra_args=["-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"])   # libx264 needs even sizes
    path = out_figs / "trajectory.mp4"
    with writer.saving(fig, path, dpi=150):
        for i, s in enumerate(steps):
            for im, (_, kind, j) in zip(ims, panels):
                im.set_data(snaps[kind]["x"][i, j, 0])
            JL = traces["latent"]["J"][int(s), :2]; JI = traces["image"]["J"][int(s), :2]
            txt.set_text(f"Adam step {int(s)}   J_z: {JL[0]:,.0f} / {JL[1]:,.0f}   J_x: {JI[0]:,.0f} / {JI[1]:,.0f}")
            writer.grab_frame()
    plt.close(fig)
    return dict(n_frames=len(steps), fps=cfg["anim"]["fps"], mp4=str(path))


# --------------------------------------------------------------------------
# collection: figures + LaTeX fragment from the saved data (CPU)
# --------------------------------------------------------------------------

def _load_all(out_data):
    prob = torch.load(out_data / "problem.pt", map_location="cpu")
    sol = {k: torch.load(out_data / f"solutions_{k}.pt", map_location="cpu") for k in KINDS}
    refs = torch.load(out_data / "refs.pt", map_location="cpu")
    traces = {k: dict(np.load(out_data / f"trace_{k}.npz")) for k in KINDS}
    snaps = {k: torch.load(out_data / f"snapshots_{k}.pt", map_location="cpu") for k in KINDS}
    for s in snaps.values():
        s["steps"] = s["steps"].numpy()
    return prob, sol, refs, traces, snaps


def collect(cfg, out_data, out_figs):
    prob, sol, refs, traces, snaps = _load_all(out_data)
    res = json.load(open(out_data / "result.json")) if (out_data / "result.json").exists() else {}
    s10_dir = resolve(cfg["step10_output"]) / "data"
    step10_mean = step10_std = None
    if (s10_dir / "samples.pt").exists():
        S = torch.load(s10_dir / "samples.pt", map_location="cpu")["samples"]
        step10_mean, step10_std = S.mean(0, keepdim=True), S.std(0)
    fig_curves(traces, refs, prob, out_figs)
    fig_solutions(sol, refs, prob, out_figs, step10_mean if "step10_mean" in refs else None)
    fig_restarts(sol, refs["inits_x"], out_figs)
    fig_spread(sol, prob, out_figs, step10_std)
    fig_objectives(sol, refs, out_figs)
    fig_trajectory(snaps, traces, prob, cfg, out_figs)
    if not res:
        return
    mc, v = cfg["map"], res["verify"]
    L, I = res["latent"], res["image"]
    sL, sI = L["summary"], I["summary"]
    t, z0, s10r = res["refs"]["truth"], res["refs"]["zero_fill"], res["refs"].get("step10")
    floor = 0.5 * prob["n_obs"] * (1 + math.log(2 * math.pi * prob["sigma"] ** 2))
    gb = lambda r: f", {r['peak_mem_gb']:.1f}\\,GB peak" if r.get("peak_mem_gb") is not None else ""
    tex = (f"Same measurement as step 10 (test image \\#{prob['index']}, {prob['n_hole']} hole pixels, $\\sigma={prob['sigma']}$; "
           f"identity with the saved step-10 problem {'verified' if v.get('problem_matches_step10') else 'NOT checked'}). "
           f"{mc['n_restarts']} restarts (restart 0 at $z=0$) of {mc['steps']} Adam steps (lr {mc['lr']:g}, {mc.get('lr_schedule', 'cosine')} decay, "
           f"$\\beta=({mc['betas'][0]},{mc['betas'][1]})$): latent MAP {L['run']['step_seconds']:.2f}\\,s/step, {L['run']['elapsed_min']:.0f}\\,min{gb(L['run'])}; "
           f"image MAP {I['run']['step_seconds']:.2f}\\,s/step, {I['run']['elapsed_min']:.1f}\\,min{gb(I['run'])}; "
           f"{L['run']['n_nonfinite'] + I['run']['n_nonfinite']} non-finite gradients. "
           f"Final objectives over restarts: $J_z$ {sL['J_z']['mean']:,.0f}$\\pm${sL['J_z']['std']:,.0f} (min {sL['J_z']['min']:,.0f}) for the latent MAP "
           f"vs. $J_z$ {t['J_z']['mean']:,.0f} at the truth" + (f" and {s10r['J_z']['mean']:,.0f}$\\pm${s10r['J_z']['std']:,.0f} for the step-10 samples" if s10r else "") + "; "
           f"$J_x$ {sI['J_x']['mean']:,.0f}$\\pm${sI['J_x']['std']:,.0f} (min {sI['J_x']['min']:,.0f}) for the image MAP vs. {t['J_x']['mean']:,.0f} at the truth"
           + (f" and {s10r['J_x']['mean']:,.0f}$\\pm${s10r['J_x']['std']:,.0f} for the step-10 samples" if s10r else "") + ". "
           f"Cross-evaluated, the latent MAPs have $J_x$ {sL['J_x']['mean']:,.0f}$\\pm${sL['J_x']['std']:,.0f} and the image MAPs $J_z$ "
           f"{sI['J_z']['mean']:,.0f}$\\pm${sI['J_z']['std']:,.0f}; the log-Jacobian $\\log|\\det \\partial f/\\partial x|$ is "
           f"{-sL['logdet']['mean']:,.0f} at the latent MAPs, {-sI['logdet']['mean']:,.0f} at the image MAPs and {-t['logdet']['mean']:,.0f} at the truth (sign as $-$logdet). "
           f"Data term: latent MAP {sL['data']['mean']:,.0f} (observed-pixel RMS residual {sL['obs_rms']['mean']:.4f}), image MAP {sI['data']['mean']:,.0f} "
           f"({sI['obs_rms']['mean']:.4f}), noise level {floor:,.0f} ($\\sigma={prob['sigma']}$); $\\|z\\|$ {sL['z_norm']['mean']:.1f} at the latent MAPs, "
           f"{sI['z_norm']['mean']:.1f} at the image MAPs, {t['z_norm']['mean']:.1f} at the truth ($\\sqrt{{n}}={math.sqrt(prob['x'].numel()):.0f}$). "
           f"Reconstruction: latent MAP PSNR {sL['psnr']['mean']:.2f}$\\pm${sL['psnr']['std']:.2f}\\,dB (hole {sL['psnr_hole']['mean']:.2f}$\\pm${sL['psnr_hole']['std']:.2f}, "
           f"best-$J_z$ restart {L['best']['psnr_hole']:.2f}, mean of restarts {res['refs']['latent_mean']['psnr_hole']['mean']:.2f}), SSIM {sL['ssim']['mean']:.3f}, hole bias {sL['bias_hole']['mean']:+.3f}; "
           f"image MAP {sI['psnr']['mean']:.2f}$\\pm${sI['psnr']['std']:.2f}\\,dB (hole {sI['psnr_hole']['mean']:.2f}$\\pm${sI['psnr_hole']['std']:.2f}, "
           f"best-$J_x$ restart {I['best']['psnr_hole']:.2f}, mean of restarts {res['refs']['image_mean']['psnr_hole']['mean']:.2f}), SSIM {sI['ssim']['mean']:.3f}, hole bias {sI['bias_hole']['mean']:+.3f}; "
           f"zero-fill {z0['psnr']['mean']:.2f}\\,dB (hole {z0['psnr_hole']['mean']:.2f})"
           + (f"; step-10 posterior mean {res['refs']['step10_mean']['psnr']['mean']:.2f}\\,dB (hole {res['refs']['step10_mean']['psnr_hole']['mean']:.2f})" if "step10_mean" in res["refs"] else "") + ". "
           f"Spread over restarts (hole std / mean pairwise RMS): latent {L['spread']['std_hole']:.3f} / {L['spread']['pair_rms_hole']:.3f}, "
           f"image {I['spread']['std_hole']:.3f} / {I['spread']['pair_rms_hole']:.3f}; paired latent-vs-image distance in the hole {res['paired']['rms_hole_mean']:.3f} RMS; "
           f"$J_z$(latent MAP) $<$ $J_z$(image MAP) in {res['paired']['n_Jz_latent_better']}/{mc['n_restarts']} pairs and $J_x$(image MAP) $<$ $J_x$(latent MAP) in "
           f"{res['paired']['n_Jx_image_better']}/{mc['n_restarts']}. Max $|$pixel$|$ {sL['max_abs']['max']:.2f} (latent) / {sI['max_abs']['max']:.2f} (image). "
           f"Self-test: $J_z$ from the differentiable reverse vs. the forward pass max $|\\Delta|={v['max_abs_Jz_diff']:.1e}$ (on $|J_z|\\approx{v['mean_abs_Jz']:.0f}$), "
           f"$J_x$ path max $|\\Delta|={v['max_abs_Jx_diff']:.1e}$, round trip $f(g(z))-z$ max {v['max_abs_roundtrip_z']:.1e}, directional finite differences (fp32, along the gradient) "
           f"rel. err. {v['fd_latent']['rel_err']:.1e} (latent) / {v['fd_image']['rel_err']:.1e} (image).")
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
    cfg10 = s10.load_config(resolve(cfg["step10_config"]))
    cfg9 = s9.load_config(resolve(cfg10["prior_config"]))
    out_root = resolve(cfg["output_root"])
    out_data, out_figs = out_root / "data", out_root / "figures"
    out_data.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)
    if args.collect:
        collect(cfg, out_data, out_figs)
        return
    device = torch.device(args.device)
    torch.manual_seed(cfg["seed"])
    t_all = time.time()
    prior, prior_path = s9.load_prior(cfg9, device, cfg10["prior_checkpoint"])
    for p in prior.parameters():
        p.requires_grad_(False)
    prob = s10.build_problem(cfg10, cfg9, device)                     # the step-10 measurement, rebuilt
    torch.save({k: v for k, v in prob.items() if k != "op"}, out_data / "problem.pt")
    s10_dir = resolve(cfg["step10_output"]) / "data"
    print(f"[step11] prior {prior_path.name}; {prob['op']}; test image #{prob['index']}/{prob['n_test']}", flush=True)

    ver = verify(prior, prob, cfg, device, s10_dir / "problem.pt")
    with open(out_data / "verify.json", "w") as f:
        json.dump(ver, f, indent=2)
    print("[step11] verify: " + json.dumps(ver), flush=True)
    assert ver["max_abs_x_diff"] < 1e-4 and ver["max_abs_Jz_diff"] < 1e-2 * ver["mean_abs_Jz"] and ver["max_abs_Jx_diff"] < 1e-2 * ver["mean_abs_Jz"]
    assert ver["fd_latent"]["rel_err"] < 1e-2 and ver["fd_image"]["rel_err"] < 1e-2
    assert ver["grad_finite_latent"] and ver["grad_finite_image"] and not ver["sample_mode_left_on"]
    assert ver.get("problem_matches_step10", True)
    if args.verify:
        return

    mc = cfg["map"]
    T, D = s10.latent_shape(prior)
    g = torch.Generator(device=device).manual_seed(cfg["seed"] + 1)
    z_init = torch.randn(mc["n_restarts"], T, D, device=device, generator=g)
    z_init[0] = 0                                                       # restart 0: the prior's latent mode
    x_init = s10.draw_samples(prior, z_init, mc["batch"])
    init_of = dict(latent=z_init, image=x_init)
    res = dict(problem={k: prob[k] for k in ("index", "label", "n_test", "n_obs", "n_hole", "sigma")}, verify=ver)
    sol = {}
    for kind in cfg["variants"]:
        path = out_data / f"solutions_{kind}.pt"
        if path.exists():                                               # variant-level resume
            sol[kind] = torch.load(path, map_location=device)
            print(f"[step11] {kind}: loaded {path.name}", flush=True)
            continue
        v, x, run = optimise(kind, prior, prob, init_of[kind], mc, device, out_data)
        if kind == "latent":
            x_off = s10.draw_samples(prior, v, mc["batch"])              # the official reverse at the solution
            run["max_abs_x_vs_official"] = float((x - x_off).abs().max())
            z, x = v, x_off
        else:
            z = None
        a = assess(prior, x, prob, Z=z, batch=mc["batch"])
        if z is None:
            z = torch.cat([prior(x[i:i + mc["batch"]])[0] for i in range(0, x.size(0), mc["batch"])]).detach()
        sol[kind] = dict(z=z.cpu(), x=x.cpu(), a=a, run=run)
        torch.save(sol[kind], path)
    prob_cpu = {k: v.cpu() if torch.is_tensor(v) else v for k, v in prob.items() if k != "op"}
    for kind in KINDS:
        a = sol[kind]["a"]
        key = "J_z" if kind == "latent" else "J_x"
        b = int(a[key].argmin())
        res[kind] = dict(run=sol[kind]["run"], summary=summarise(a), spread=spread(sol[kind]["x"], prob_cpu),
                         best=dict(restart=b, **{k: float(v[b]) for k, v in a.items()}),
                         restart0={k: float(v[0]) for k, v in a.items()})

    # reference points under both objectives
    refs = dict(truth=assess(prior, prob["x"], prob), zero_fill=assess(prior, prob["y"], prob),
                inits=assess(prior, x_init, prob, Z=z_init, batch=mc["batch"]), inits_x=x_init.cpu(),
                latent_mean=assess(prior, sol["latent"]["x"].mean(0, keepdim=True).to(device), prob),
                image_mean=assess(prior, sol["image"]["x"].mean(0, keepdim=True).to(device), prob))
    if (s10_dir / "samples.pt").exists():
        S = torch.load(s10_dir / "samples.pt", map_location=device)["samples"]
        refs["step10"] = assess(prior, S, prob, batch=mc["batch"])
        refs["step10_mean"] = assess(prior, S.mean(0, keepdim=True), prob)
    torch.save(refs, out_data / "refs.pt")
    res["refs"] = {k: summarise(v) for k, v in refs.items() if k != "inits_x"}
    xL, xI = sol["latent"]["x"], sol["image"]["x"]
    hole = (prob["mask"][0] == 0).cpu()
    d = ((xL - xI)[:, hole] ** 2).flatten(1).mean(-1).sqrt()
    res["paired"] = dict(rms_hole_mean=float(d.mean()), rms_hole_min=float(d.min()), rms_hole_max=float(d.max()),
                         n_Jz_latent_better=int((sol["latent"]["a"]["J_z"] < sol["image"]["a"]["J_z"]).sum()),
                         n_Jx_image_better=int((sol["image"]["a"]["J_x"] < sol["latent"]["a"]["J_x"]).sum()))
    res["total_wall_min"] = (time.time() - t_all) / 60
    with open(out_data / "result.json", "w") as f:
        json.dump(res, f, indent=2)
    with open(out_data / "provenance.json", "w") as f:
        json.dump(dict(config=cfg, step10_config=cfg10, prior_config=cfg9, prior_checkpoint=str(prior_path),
                       tarflow_commit=s2.tarflow_commit(cfg9), torch=torch.__version__,
                       timestamp=datetime.now(timezone.utc).isoformat(), argv=sys.argv), f, indent=2)
    collect(cfg, out_data, out_figs)
    print("step11 done")


if __name__ == "__main__":
    main()
