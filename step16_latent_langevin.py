#!/usr/bin/env python
"""step16_latent_langevin.py -- Random walks on the posterior, in latent space only.

Not the mode but the distribution. The posterior in latent coordinates is
p(z | y) \\propto p(y | g(z)) N(z; 0, I) -- exactly, because x = g(z) is a bijection and
the Jacobian of the change of variables cancels between the prior density and the
volume element -- so a Markov chain with that target, pushed through g, is a posterior
sampler for x in which no log-determinant appears. The chain is Hamiltonian Monte
Carlo on z with the energy
    U(z) = [ -log p(y | g(z)) + ||z||^2 / (2 tau^2) ] / T,
T = tau = 1 the posterior itself, tau < 1 a cooled prior (latent truncation), T < 1 a
tempered posterior: `leapfrog` steps of size eps per iteration, momentum refreshed
every iteration, Metropolis-corrected; eps is adapted per chain during the burn-in
towards the target acceptance and then frozen (`leapfrog: 1` is MALA). Chains start
from the step-14 latent MAP (lowest-J_z restart) or from a latent maximum-likelihood
point (Adam on the data term alone, from the MAP). All systems x variants x chains
run as ONE batch through invlib.FastReverse; x = g(z) is recorded after every
iteration and rendered as one mp4 per system.

Outputs (outputs/step16_latent_langevin/):
  data/     langevin_{res}.json, langevin_{res}_frames.npz (recorded states, traces),
            langevin_{res}_solutions.pt, summary.tex, provenance.json
  figures/  langevin_{res}.png (posterior from the MAP: samples, mean, std),
            langevin_{res}_variants.png, langevin_{res}_traces.png,
            langevin_{res}_{system}.mp4 (one per system)

Usage:
  python step16_latent_langevin.py --res 32 [--device cuda:0] [--iterations K --burn-in B]
  python step16_latent_langevin.py --res 32 --figures --video    # re-render from the saved files
  python step16_latent_langevin.py --collect
"""

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "step16_latent_langevin.yml"

import invlib  # noqa: E402
import step10_variational_posterior as s10  # noqa: E402
import step13_lidc_tarflow as s13  # noqa: E402
import step14_lidc_gallery as s14  # noqa: E402


def load_config(path=DEFAULT_CONFIG):
    with open(path) as f:
        return yaml.safe_load(f)


def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


INIT_LABEL = dict(map="latent MAP", mle="latent MLE", prior="prior sample")


def stem(cfg, res):
    """File stem: langevin_{res}{tag}."""
    return f"langevin_{res}{cfg.get('tag', '')}"


def pick_systems(cfg, cfg14):
    """The step-14 systems, optionally restricted to cfg['systems'] (names), with their gallery row indices."""
    sys_all = cfg14["systems"][:cfg14.get("n_rows", len(cfg14["systems"]))]
    keep = [i for i, sp in enumerate(sys_all) if not cfg.get("systems") or sp["name"] in cfg["systems"]]
    return [sys_all[i] for i in keep], keep


def window_mean(a, w=10):
    """Trailing mean over the last `w` entries (the acceptance-rate window)."""
    c = np.cumsum(np.concatenate([[0.0], np.asarray(a, dtype=np.float64)]))
    k = np.arange(1, len(a) + 1)
    lo = np.maximum(k - w, 0)
    return (c[k] - c[lo]) / (k - lo)


def residual_ratio(data, noise_level, M):
    """||y - Ax||^2 / (M sigma^2) from the data term: data = ||r||^2 / 2 sigma^2 + const and
    noise_level = const + M / 2."""
    return (data - noise_level) / (0.5 * M) + 1


# --------------------------------------------------------------------------
# the sampler
# --------------------------------------------------------------------------

def run(cfg, res, device, out_data, out_figs, iterations=None, burn_in=None):
    s13.set_tf32(False)
    cfg14 = s14.load_config(resolve(cfg["step14_config"]))
    out14 = REPO_ROOT / cfg14["output_root"] / "data"
    gal = json.load(open(out14 / f"gallery_{res}.json"))
    # the solutions file holds numpy traces next to the tensors (our own file, not a checkpoint)
    sol = torch.load(out14 / f"gallery_{res}_solutions.pt", map_location="cpu", weights_only=False)
    cfg13 = s13.load_config(resolve(cfg14["priors"][res]))
    prior, ckpt = s13.load_prior(cfg13, device, cfg14["prior_checkpoint"])
    for p in prior.parameters():
        p.requires_grad_(False)
    T_, D = s10.latent_shape(prior)
    n = T_ * D
    systems, keep_idx = pick_systems(cfg, cfg14)
    gal_rows = [gal["rows"][i] for i in keep_idx]
    variants, C = cfg["variants"], int(cfg["chains"])
    S, V = len(systems), len(variants)
    B = S * V * C
    K = int(iterations or cfg["iterations"])
    K_burn = int(burn_in or cfg["burn_in"])
    K_adapt = min(int(cfg["adapt_iters"]), K_burn)
    L = int(cfg["leapfrog"])
    tag = f"[step16:{res}]"
    rows_of = lambda s, v: slice((s * V + v) * C, (s * V + v + 1) * C)        # chains of (system, variant)
    print(f"{tag} prior {ckpt} (T={T_}, D={D}); {S} systems x {V} variants x {C} chains = batch {B}; "
          f"{K} HMC iterations x {L} leapfrog steps, burn-in {K_burn}, adapt {K_adapt}", flush=True)

    # the stored step-14 problems (measurements re-used, so the generator is untouched)
    gen = torch.Generator().manual_seed(cfg14["seed"])
    probs, x_trues, z_map = [], [], []
    for spec, row in zip(systems, gal_rows):
        st = sol[spec["name"]]
        x_true = st["x_true"].to(device)
        probs.append(s14.make_problem(spec, x_true, gen, y=st["y"]))
        x_trues.append(x_true)
        z_map.append(st["z"][int(np.argmin(row["J"]))].to(device))          # the method's pick

    def data_terms(x, per_system):
        """The data terms of a batch made of `per_system` consecutive rows per system."""
        return torch.cat([p["nll"](x[s * per_system:(s + 1) * per_system]) for s, p in enumerate(probs)])

    x_true_rows = torch.cat([x_trues[s].expand(V * C, -1, -1, -1) for s in range(S)])
    temp = torch.tensor([float(v["temperature"]) for s in range(S) for v in variants for c in range(C)], device=device)
    tau = torch.tensor([float(v["prior_scale"]) for s in range(S) for v in variants for c in range(C)], device=device)
    M_rows = torch.tensor([float(p["M"]) for p in probs for _ in range(V * C)], device=device)
    noise_level = torch.tensor([p["noise_level"] for p in probs for _ in range(V * C)], device=device)

    # initial states: the latent MAP, or the latent maximum-likelihood point continued from it
    z_init = {"map": torch.stack(z_map)}                                        # (S, T, D)
    if any(v["init"] == "mle" for v in variants):
        torch.cuda.synchronize()
        t1 = time.time()
        fr_mle = invlib.FastReverse(prior, S, device, mode="compile")
        zm = z_init["map"].clone().requires_grad_(True)
        opt = torch.optim.Adam([zm], lr=float(cfg["mle"]["lr"]))
        for k in range(int(cfg["mle"]["steps"])):
            x, _ = fr_mle(zm)
            d = data_terms(x, 1)
            opt.zero_grad(set_to_none=True)
            d.sum().backward()
            opt.step()
        with torch.no_grad():
            x, _ = fr_mle(zm)
            d = data_terms(x, 1)
        z_init["mle"] = zm.detach()
        del fr_mle
        mle_stats = dict(steps=int(cfg["mle"]["steps"]), lr=float(cfg["mle"]["lr"]), seconds=time.time() - t1,
                         residual_ratio=[float(residual_ratio(d[s], probs[s]["noise_level"], probs[s]["M"])) for s in range(S)],
                         z_norm=[float(v_) for v_ in zm.detach().flatten(1).norm(dim=-1)],
                         psnr=[float(v_) for v_ in s14.psnr01(x, torch.cat(x_trues))])
        print(f"{tag} latent MLE init ({mle_stats['seconds']:.0f} s): residual/(M sigma^2) "
              f"{['%.2f' % v_ for v_ in mle_stats['residual_ratio']]}, |z| {['%.1f' % v_ for v_ in mle_stats['z_norm']]}, "
              f"PSNR {['%.1f' % v_ for v_ in mle_stats['psnr']]}", flush=True)
    else:
        mle_stats = None
    z0 = torch.empty(B, T_, D, device=device)
    for s in range(S):
        for v, var in enumerate(variants):
            if var["init"] not in z_init:
                raise ValueError(var["init"])
            z0[rows_of(s, v)] = z_init[var["init"]][s]

    torch.cuda.synchronize()
    t0 = time.time()
    if cfg.get("reverse") == "fast2":    # in-place-cache reverse (step 14's choice at T = 1024; same (x, log_q) interface)
        fr = invlib.FastReverse2(prior, B, device, chunk=int(cfg.get("chunk", 128)), mode="graph")
    else:
        fr = invlib.FastReverse(prior, B, device, mode="compile")
    torch.cuda.synchronize()
    build_s = time.time() - t0
    print(f"{tag} graph build {build_s:.1f} s (batch {B}, {cfg.get('reverse', 'fast')})", flush=True)

    def energy(z):
        """U, grad U, x = g(z) and the data term, per row."""
        z = z.detach().requires_grad_(True)
        x, _ = fr(z)
        d = data_terms(x, V * C)
        U = (d + 0.5 * (z ** 2).flatten(1).sum(-1) / tau ** 2) / temp
        (g,) = torch.autograd.grad(U.sum(), z)
        return U.detach(), g, x.detach(), d.detach()

    with torch.no_grad():
        x0 = torch.cat([s10.draw_samples(prior, z0[i:i + 32]) for i in range(0, B, 32)])
    z = z0.clone()
    U, g, x, d = energy(z)
    log_eps = torch.full((B,), math.log(float(cfg["eps0"])), device=device)
    rate, target, jit = float(cfg["adapt_rate"]), float(cfg["target_accept"]), float(cfg.get("eps_jitter", 0.0))
    noise_gen = torch.Generator(device=device).manual_seed(cfg["seed"])
    tr = {k: np.zeros((K, B), dtype=np.float32) for k in ("U", "data", "z_norm", "psnr", "accept", "accept_prob", "eps")}
    frames = np.zeros((K, B, x.shape[-2], x.shape[-1]), dtype=np.float16)
    ts = []
    torch.cuda.synchronize()
    t_all = time.time()
    for k in range(K):
        t1 = time.time()
        eps = log_eps.exp()
        if jit > 0:
            eps = eps * (1 + jit * (2 * torch.rand(B, generator=noise_gen, device=device) - 1))
        e3 = eps.view(B, 1, 1)
        p = torch.randn(z.shape, generator=noise_gen, device=device)
        H0 = U + 0.5 * (p ** 2).flatten(1).sum(-1)
        zc, gc = z, g
        p = p - 0.5 * e3 * gc                                                   # leapfrog: half kick,
        for l in range(L):
            zc = zc + e3 * p                                                    # drift,
            Uc, gc, xc, dc = energy(zc)
            p = p - (e3 if l < L - 1 else 0.5 * e3) * gc                        # kick (half at the end)
        H1 = Uc + 0.5 * (p ** 2).flatten(1).sum(-1)
        log_alpha = H0 - H1
        finite = torch.isfinite(log_alpha) & torch.isfinite(gc).flatten(1).all(-1)
        log_alpha = torch.where(finite, log_alpha, torch.full_like(log_alpha, -float("inf")))
        u = torch.rand(B, generator=noise_gen, device=device)
        acc = u.log() < log_alpha
        a_prob = log_alpha.clamp(max=0).exp()
        a3 = acc.view(B, 1, 1)
        z = torch.where(a3, zc, z)
        g = torch.where(a3, gc, g)
        x = torch.where(acc.view(B, 1, 1, 1), xc, x)
        U = torch.where(acc, Uc, U)
        d = torch.where(acc, dc, d)
        if k < K_adapt:                                                        # rate decays linearly to a quarter
            log_eps = log_eps + rate * (1 - 0.75 * k / K_adapt) * (a_prob - target)
        torch.cuda.synchronize()
        ts.append(time.time() - t1)
        tr["U"][k] = U.cpu().numpy()
        tr["data"][k] = d.cpu().numpy()
        tr["z_norm"][k] = z.flatten(1).norm(dim=-1).cpu().numpy()
        tr["psnr"][k] = s14.psnr01(x, x_true_rows).cpu().numpy()
        tr["accept"][k] = acc.float().cpu().numpy()
        tr["accept_prob"][k] = a_prob.cpu().numpy()
        tr["eps"][k] = eps.cpu().numpy()
        frames[k] = x[:, 0].half().cpu().numpy()
        if k % 10 == 0 or k == K - 1:
            lo = max(0, k - 9)
            acc_w = tr["accept"][lo:k + 1].mean(0)
            rr = residual_ratio(tr["data"][k], noise_level.cpu().numpy(), M_rows.cpu().numpy())
            print(f"{tag} iter {k:4d}: " + " ".join(
                f"{p_['name']}[acc {acc_w[rows_of(s, 0)].mean():.2f} eps {float(eps[rows_of(s, 0)].mean()):.1e} "
                f"|z| {tr['z_norm'][k, rows_of(s, 0)].mean():.1f} res {rr[rows_of(s, 0)].mean():.2f}]"
                for s, p_ in enumerate(probs)) + f"  {ts[-1]:.2f} s/iter ({ts[-1] / L:.2f} s/grad)", flush=True)
    total = time.time() - t_all
    keep = np.arange(K) >= K_burn                                              # the sampling phase

    rows = []
    for s, (spec, prob, x_true) in enumerate(zip(systems, probs, x_trues)):
        row_true = gal_rows[s]
        with torch.no_grad():
            zt, _, _ = prior(x_true)
        for v, var in enumerate(variants):
            r = rows_of(s, v)
            F_ = frames[keep][:, r].astype(np.float32)                          # (K', C, N, N)
            mean = torch.from_numpy(F_.mean((0, 1)))[None, None]
            std = torch.from_numpy(F_.std((0, 1)))[None, None]
            psnr_mean = float(s14.psnr01(mean, x_true.cpu())[0])
            xs = x[r]
            U_true = float((row_true["data_true"] + 0.5 * float((zt ** 2).sum()) / var["prior_scale"] ** 2) / var["temperature"])
            rows.append(dict(
                model=spec["name"], variant=var["name"], init=var["init"], temperature=var["temperature"],
                prior_scale=var["prior_scale"], index=row_true["index"], M=prob["M"], sigma=prob["sigma"],
                noise_level=float(prob["noise_level"]), data_true=row_true["data_true"], U_true=U_true,
                accept_burn=float(tr["accept"][:K_burn, r].mean()), accept_sample=float(tr["accept"][K_burn:, r].mean()),
                eps=[float(v_) for v_ in np.exp(log_eps.cpu().numpy()[r])],
                U=[float(v_) for v_ in tr["U"][-1, r]], data=[float(v_) for v_ in tr["data"][-1, r]],
                residual_ratio=[float(residual_ratio(v_, prob["noise_level"], prob["M"])) for v_ in tr["data"][-1, r]],
                z_norm=[float(v_) for v_ in tr["z_norm"][-1, r]], psnr=[float(v_) for v_ in tr["psnr"][-1, r]],
                psnr_sample_mean=float(tr["psnr"][K_burn:, r].mean()), psnr_posterior_mean=psnr_mean,
                n_samples=int(F_.shape[0] * F_.shape[1]), std_mean=float(std.mean()),
                z_norm_init=[float(v_) for v_ in z0[r].flatten(1).norm(dim=-1)],
                dist_from_init=[float(v_) for v_ in (z[r] - z0[r]).flatten(1).norm(dim=-1)],
                pairwise_rms=float(((xs[:, None] - xs[None]) ** 2).mean((2, 3, 4)).sqrt().sum() / (C * (C - 1))),
                max_abs=[float(v_) for v_ in xs.flatten(1).abs().max(-1).values]))
            rr = rows[-1]
            print(f"{tag} {spec['name']:15s} {var['name']:20s}: acc {rr['accept_burn']:.2f}/{rr['accept_sample']:.2f} "
                  f"eps {['%.1e' % v_ for v_ in rr['eps']]} U {['%.0f' % v_ for v_ in rr['U']]} (truth {U_true:.0f}) "
                  f"res/(M sigma^2) {['%.2f' % v_ for v_ in rr['residual_ratio']]} "
                  f"|z| {['%.1f' % v_ for v_ in rr['z_norm']]} PSNR {['%.1f' % v_ for v_ in rr['psnr']]} "
                  f"(samples {rr['psnr_sample_mean']:.1f}, posterior mean {psnr_mean:.1f}) spread {rr['pairwise_rms']:.3f} "
                  f"moved {['%.1f' % v_ for v_ in rr['dist_from_init']]}", flush=True)
    res_json = dict(res=res, prior=str(ckpt), latent=dict(T=T_, D=D), n=n, sqrt_n=math.sqrt(n), batch=B, chains=C,
                    iterations=K, leapfrog=L, burn_in=K_burn, adapt_iters=K_adapt, eps0=cfg["eps0"], target_accept=target,
                    eps_jitter=jit, variants=variants, mle=mle_stats, seconds=total, iter_seconds=float(np.mean(ts)),
                    grad_seconds=float(np.mean(ts)) / L, graph_build_seconds=build_s, reverse=cfg.get("reverse", "fast"),
                    indices=[gal["indices"][i] for i in keep_idx], systems=[sp["name"] for sp in systems], rows=rows)
    json.dump(res_json, open(out_data / f"{stem(cfg, res)}.json", "w"), indent=1)
    np.savez_compressed(out_data / f"{stem(cfg, res)}_frames.npz", frames=frames, **{f"trace_{k}": v_ for k, v_ in tr.items()})
    torch.save(dict(x_true=torch.cat(x_trues).cpu(), y={p_["name"]: p_["y"].cpu() for p_ in probs},
                    z_init=z0.cpu(), x_init=x0.cpu(), z_final=z.cpu(), x_final=x.cpu()),
               out_data / f"{stem(cfg, res)}_solutions.pt")
    print(f"{tag} done: {total:.0f} s ({np.mean(ts):.2f} s/iteration, {np.mean(ts) / L:.2f} s/gradient)", flush=True)
    return res_json


# --------------------------------------------------------------------------
# figures and videos, from the saved files
# --------------------------------------------------------------------------

def _load(cfg, res, out_data):
    cfg14 = s14.load_config(resolve(cfg["step14_config"]))
    J = json.load(open(out_data / f"{stem(cfg, res)}.json"))
    sys_all = cfg14["systems"][:cfg14.get("n_rows", len(cfg14["systems"]))]
    systems = [sp for sp in sys_all if sp["name"] in J.get("systems", [sp["name"] for sp in sys_all])]
    Fz = np.load(out_data / f"{stem(cfg, res)}_frames.npz")
    sol = torch.load(out_data / f"{stem(cfg, res)}_solutions.pt", map_location="cpu", weights_only=False)
    # the measurement display needs the operators: rebuild the problems on the CPU with the stored y
    gen = torch.Generator().manual_seed(cfg14["seed"])
    probs = [s14.make_problem(spec, sol["x_true"][s:s + 1], gen, y=sol["y"][spec["name"]]) for s, spec in enumerate(systems)]
    return cfg14, systems, J, Fz, sol, probs


def figures(cfg, res, out_data, out_figs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cfg14, systems, J, Fz, sol, probs = _load(cfg, res, out_data)
    variants, C, S, V = J["variants"], J["chains"], len(systems), len(J["variants"])
    K, K_burn, L = J["iterations"], J["burn_in"], J["leapfrog"]
    rows_of = lambda s, v: slice((s * V + v) * C, (s * V + v + 1) * C)
    frames = Fz["frames"]
    keep = np.arange(K) >= K_burn
    row = lambda s, v: J["rows"][s * V + v]

    # 1. the posterior from the MAP: samples, mean, std
    v0 = 0
    ncol = 3 + C + 2
    fig, axes = plt.subplots(S, ncol, figsize=(1.9 * ncol + 1.2, 1.95 * S))
    for s, spec in enumerate(systems):
        r = row(s, v0)
        s10._img(axes[s, 0], sol["x_true"][s, 0], f"truth, test #{r['index']}")
        s14.show_measurement(axes[s, 1], probs[s]["op"], probs[s]["y"], r["M"])
        xi = sol["x_init"][rows_of(s, v0)][:1]
        s10._img(axes[s, 2], xi[0, 0], f"latent MAP (init)\n{float(s14.psnr01(xi, sol['x_true'][s:s + 1])[0]):.1f} dB")
        for c in range(C):
            s10._img(axes[s, 3 + c], sol["x_final"][rows_of(s, v0)][c, 0], f"chain {c + 1}, iteration {K}\n{r['psnr'][c]:.1f} dB")
        F_ = frames[keep][:, rows_of(s, v0)].astype(np.float32)
        s10._img(axes[s, 3 + C], torch.from_numpy(F_.mean((0, 1))), f"posterior mean ({r['n_samples']})\n{r['psnr_posterior_mean']:.1f} dB")
        s10._img(axes[s, 4 + C], torch.from_numpy(F_.std((0, 1))), f"posterior std (mean {r['std_mean']:.3f})", vmin=0, vmax=0.5, cmap="magma")
        axes[s, 0].text(-0.12, 0.5, s14.describe(spec, res), transform=axes[s, 0].transAxes, rotation=90, va="center",
                        ha="center", fontsize=7)
    fig.suptitle(f"LIDC {res}$\\times${res}: HMC on the latent posterior $p(z\\,|\\,y) \\propto p(y\\,|\\,g(z))\\,N(z;0,I)$ from the "
                 f"latent MAP ({K} iterations $\\times$ {L} leapfrog steps, burn-in {K_burn}, {C} chains); posterior mean and std "
                 f"over the recorded samples", fontsize=8)
    fig.tight_layout(rect=(0.02, 0, 1, 0.97))
    fig.savefig(out_figs / f"{stem(cfg, res)}.png", dpi=150)
    plt.close(fig)

    # 2. the variants: one final sample and the posterior mean of each
    ncol = 2 + 2 * V
    fig, axes = plt.subplots(S, ncol, figsize=(1.75 * ncol + 1.2, 1.95 * S))
    for s, spec in enumerate(systems):
        s10._img(axes[s, 0], sol["x_true"][s, 0], f"truth, test #{row(s, 0)['index']}")
        s14.show_measurement(axes[s, 1], probs[s]["op"], probs[s]["y"], row(s, 0)["M"])
        for v, var in enumerate(variants):
            r = row(s, v)
            F_ = frames[keep][:, rows_of(s, v)].astype(np.float32)
            s10._img(axes[s, 2 + 2 * v], sol["x_final"][rows_of(s, v)][0, 0],
                     f"{var['name']}\nsample: {r['psnr'][0]:.1f} dB, acc {r['accept_sample']:.2f}")
            s10._img(axes[s, 3 + 2 * v], torch.from_numpy(F_.mean((0, 1))), f"mean: {r['psnr_posterior_mean']:.1f} dB\n$\\|z\\|$ {np.mean(r['z_norm']):.0f}")
        axes[s, 0].text(-0.12, 0.5, s14.describe(spec, res), transform=axes[s, 0].transAxes, rotation=90, va="center",
                        ha="center", fontsize=7)
    fig.suptitle(f"LIDC {res}$\\times${res}: the operating points -- init (latent MAP / latent MLE) and the prior temperature "
                 f"$\\tau$; last state of chain 1 and the posterior mean of each", fontsize=8)
    fig.tight_layout(rect=(0.02, 0, 1, 0.97))
    fig.savefig(out_figs / f"{stem(cfg, res)}_variants.png", dpi=150)
    plt.close(fig)

    # 3. traces: U - U_truth, residual / (M sigma^2), |z|, PSNR, acceptance; chains averaged
    names = ["$U - U(z_{true})$", "$\\|y - Ax\\|^2 / (M\\sigma^2)$", "$\\|z\\|$", "PSNR (dB)", "acceptance (10-iteration window)"]
    fig, axes = plt.subplots(len(names), S, figsize=(2.6 * S + 0.5, 1.9 * len(names) + 0.6), sharex=True)
    kk = np.arange(K)
    win = np.ones(10) / 10
    for s, spec in enumerate(systems):
        for v, var in enumerate(variants):
            r = row(s, v)
            sl_ = rows_of(s, v)
            U = Fz["trace_U"][:, sl_].mean(1) - r["U_true"]
            resid = residual_ratio(Fz["trace_data"][:, sl_].mean(1), r["noise_level"], r["M"])
            accw = window_mean(Fz["trace_accept"][:, sl_].mean(1))
            for i, y in enumerate((U, resid, Fz["trace_z_norm"][:, sl_].mean(1), Fz["trace_psnr"][:, sl_].mean(1), accw)):
                axes[i, s].plot(kk, y, lw=0.8, label=var["name"] if s == 0 else None)
        axes[0, s].set_title(spec["label"], fontsize=8)
        axes[0, s].set_yscale("symlog", linthresh=10)
        axes[1, s].set_yscale("log")
        axes[1, s].axhline(1.0, color="k", ls="--", lw=0.6)
        axes[2, s].axhline(J["sqrt_n"], color="k", ls="--", lw=0.6)
        axes[4, s].axhline(J["target_accept"], color="k", ls="--", lw=0.6)
        for i in range(len(names)):
            axes[i, s].axvline(K_burn, color="gray", lw=0.5)
            axes[i, s].tick_params(labelsize=6)
        axes[-1, s].set_xlabel("HMC iteration", fontsize=7)
    for i, nm in enumerate(names):
        axes[i, 0].set_ylabel(nm, fontsize=7)
    axes[0, 0].legend(fontsize=5.5, loc="upper right")
    fig.suptitle(f"LIDC {res}$\\times${res}: HMC traces (mean over the {C} chains); dashed: the truth's residual, $\\sqrt{{n}}$, "
                 f"target acceptance; grey: end of burn-in", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_figs / f"{stem(cfg, res)}_traces.png", dpi=150)
    plt.close(fig)
    print(f"[step16:{res}] figures done", flush=True)


def video(cfg, res, out_data, out_figs, only=None):
    """One mp4 per system: row 0 the references, then one row per variant with the chains,
    the running posterior mean (sampling phase) and the running posterior std."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    cfg14, systems, J, Fz, sol, probs = _load(cfg, res, out_data)
    variants, C, S, V = J["variants"], J["chains"], len(systems), len(J["variants"])
    K, K_burn, L = J["iterations"], J["burn_in"], J["leapfrog"]
    rows_of = lambda s, v: slice((s * V + v) * C, (s * V + v + 1) * C)
    frames = Fz["frames"]
    fps, dpi = int(cfg["video"]["fps"]), int(cfg["video"]["dpi"])
    k_start = K_burn if cfg["video"].get("sampling_only", False) else 0     # frames = the recorded samples only
    ncol = C + 2
    to01 = lambda a: (np.clip(a, -1, 1) + 1) / 2
    for s, spec in enumerate(systems):
        if only and spec["name"] not in only:
            continue
        t0 = time.time()
        fig, axes = plt.subplots(V + 1, ncol, figsize=(1.7 * ncol + 1.3, 1.75 * (V + 1) + 0.5))
        for ax in axes.flat:
            ax.axis("off")
        # references
        x_true = sol["x_true"][s, 0].numpy()
        s10._img(axes[0, 0], sol["x_true"][s, 0], f"truth, test #{J['rows'][s * V]['index']}")
        s14.show_measurement(axes[0, 1], probs[s]["op"], probs[s]["y"], J["rows"][s * V]["M"])
        inits = {}
        for v, var in enumerate(variants):
            inits.setdefault(var["init"], sol["x_init"][rows_of(s, v)][:1])
        for j, (nm, im) in enumerate(inits.items()):
            if 2 + j < ncol:
                s10._img(axes[0, 2 + j], im[0, 0], f"init: {INIT_LABEL[nm]}\n{float(s14.psnr01(im, sol['x_true'][s:s + 1])[0]):.1f} dB")
        ims, titles, side = {}, {}, {}
        for v, var in enumerate(variants):
            for c in range(C):
                ims[v, c] = axes[v + 1, c].imshow(to01(frames[0, rows_of(s, v)][c].astype(np.float32)), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                titles[v, c] = axes[v + 1, c].set_title("", fontsize=7)
            ims[v, "mean"] = axes[v + 1, C].imshow(to01(frames[0, rows_of(s, v)][0].astype(np.float32)), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            titles[v, "mean"] = axes[v + 1, C].set_title("", fontsize=7)
            ims[v, "std"] = axes[v + 1, C + 1].imshow(np.zeros_like(x_true), cmap="magma", vmin=0, vmax=0.5, interpolation="nearest")
            titles[v, "std"] = axes[v + 1, C + 1].set_title("running std", fontsize=7)
            side[v] = axes[v + 1, 0].text(-0.08, 0.5, "", transform=axes[v + 1, 0].transAxes, rotation=90, va="center",
                                          ha="center", fontsize=6)
        sup = fig.suptitle("", fontsize=8)
        fig.tight_layout(rect=(0.03, 0, 1, 0.95))
        # yuv420p needs even frame dimensions: pad by one pixel where the figure size is odd
        writer = FFMpegWriter(fps=fps, codec="libx264",
                              extra_args=["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p", "-crf", "20"])
        path = out_figs / f"{stem(cfg, res)}_{spec['name']}.mp4"
        # running mean/std of the sampling phase, over the chains of each variant
        acc_sum = {v: np.zeros_like(x_true, dtype=np.float64) for v in range(V)}
        acc_sq = {v: np.zeros_like(x_true, dtype=np.float64) for v in range(V)}
        n_acc = {v: 0 for v in range(V)}
        with writer.saving(fig, str(path), dpi):
            for k in range(k_start, K):
                phase = "burn-in" if k < K_burn else f"sample {k - K_burn + 1} of {K - K_burn}"
                sup.set_text(f"LIDC {res}$\\times${res}, {spec['label']}: HMC on the latent posterior, iteration {k} of {K} "
                             f"({L} leapfrog steps each; {phase}, burn-in {K_burn})\nrows = variants, columns = chains, "
                             f"running posterior mean, running posterior std")
                for v, var in enumerate(variants):
                    r = rows_of(s, v)
                    xs = frames[k, r].astype(np.float32)
                    if k >= K_burn:
                        acc_sum[v] += xs.sum(0)
                        acc_sq[v] += (xs ** 2).sum(0)
                        n_acc[v] += C
                    for c in range(C):
                        ims[v, c].set_data(to01(xs[c]))
                        titles[v, c].set_text(f"chain {c + 1}: {Fz['trace_psnr'][k, r][c]:.1f} dB")
                    if n_acc[v] > 0:
                        m = acc_sum[v] / n_acc[v]
                        sd = np.sqrt(np.maximum(acc_sq[v] / n_acc[v] - m ** 2, 0))
                        ims[v, "mean"].set_data(to01(m))
                        ims[v, "std"].set_data(sd)
                        pm = 10 * np.log10(1 / max(float(((to01(m) - to01(x_true)) ** 2).mean()), 1e-12))
                        titles[v, "mean"].set_text(f"running mean ({n_acc[v]}): {pm:.1f} dB")
                        titles[v, "std"].set_text(f"running std (mean {sd.mean():.3f})")
                    else:
                        ims[v, "mean"].set_data(to01(xs.mean(0)))
                        titles[v, "mean"].set_text("(burn-in: mean of the chains)")
                    lo = max(0, k - 9)
                    accw = Fz["trace_accept"][lo:k + 1, r].mean()
                    rj = J["rows"][s * V + v]
                    resid = residual_ratio(Fz["trace_data"][k, r].mean(), rj["noise_level"], rj["M"])
                    side[v].set_text(f"{var['name']}  T={var['temperature']:g} $\\tau$={var['prior_scale']:g}\n"
                                     f"acc {accw:.2f}  $\\epsilon$ {Fz['trace_eps'][k, r].mean():.1e}  $\\|z\\|$ {Fz['trace_z_norm'][k, r].mean():.0f}  "
                                     f"res {resid:.2f}$M\\sigma^2$")
                writer.grab_frame()
        plt.close(fig)
        print(f"[step16:{res}] {path.name}: {K - k_start} frames, {time.time() - t0:.0f} s", flush=True)


def _tex(s):
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")


def collect(cfg, out_data):
    lines = []
    for p in sorted(out_data.glob("langevin_*.json")):
        J = json.load(open(p))
        V = len(J["variants"])
        parts = []
        for s in range(len(J["rows"]) // V):
            r0 = J["rows"][s * V]
            vs = "; ".join(f"{_tex(r['variant'])}: acc {r['accept_sample']:.2f}, $\\|z\\|$ {np.mean(r['z_norm']):.0f}, "
                           f"samples {r['psnr_sample_mean']:.1f} dB, mean {r['psnr_posterior_mean']:.1f} dB"
                           for r in J["rows"][s * V:(s + 1) * V])
            parts.append(f"{_tex(r0['model'])} (\\#{r0['index']}): {vs}")
        lines.append(f"\\noindent\\texttt{{{_tex(p.stem)}}}: {J['res']}$\\times${J['res']} ($n={J['n']}$, $\\sqrt{{n}}={J['sqrt_n']:.0f}$; "
                     f"batch {J['batch']}, {J['chains']} chains, {J['iterations']} HMC iterations $\\times$ {J['leapfrog']} leapfrog steps, "
                     f"burn-in {J['burn_in']}, {J['grad_seconds']:.2f} s per gradient, {J['seconds'] / 60:.0f} min).\\\\\n"
                     + " \\\\\n".join(parts) + ".")
    (out_data / "summary.tex").write_text("\n\n".join(lines) + "\n")
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--res", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--iterations", type=int, default=0)
    ap.add_argument("--burn-in", type=int, default=0)
    ap.add_argument("--figures", action="store_true", help="re-render the figures from the saved files")
    ap.add_argument("--video", action="store_true", help="re-render the videos from the saved files")
    ap.add_argument("--only", nargs="*", help="systems to render videos for")
    ap.add_argument("--collect", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    out_root = REPO_ROOT / cfg["output_root"]
    out_data, out_figs = out_root / "data", out_root / "figures"
    out_data.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)
    if args.res and not (args.figures or args.video):
        run(cfg, args.res, torch.device(args.device), out_data, out_figs, iterations=args.iterations, burn_in=args.burn_in)
        figures(cfg, args.res, out_data, out_figs)
        video(cfg, args.res, out_data, out_figs)
    if args.res and args.figures:
        figures(cfg, args.res, out_data, out_figs)
    if args.res and args.video:
        video(cfg, args.res, out_data, out_figs, only=args.only)
    collect(cfg, out_data)
    prov = dict(step="step16_latent_langevin", command=" ".join(["python"] + (argv or sys.argv[1:])),
                finished=datetime.now(timezone.utc).isoformat(), config=cfg)
    with open(out_data / f"provenance{cfg.get('tag', '')}.json", "w") as f:
        json.dump(prov, f, indent=2)


if __name__ == "__main__":
    main()
