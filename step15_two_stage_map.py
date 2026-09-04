#!/usr/bin/env python
"""step15_two_stage_map.py -- Latent initialisation, pixel refinement.

Two-stage MAP for the step-14 problems. Stage 1 is step 14 itself: Adam on z with
x = g_theta(z), objective J_z(z) = -log p(y | g(z)) - log N(z; 0, I), three random
restarts per system. Stage 2 takes each restart's image x_1 = g(z_1) and runs Adam on
the pixels of x under step 11's image-space objective
    J_x(x) = -log p(y | x) - log p_theta(x)
           = -log p(y | x) + 1/2 |f(x)|^2 + const - log |det df/dx|,
the exact posterior density in x (J_z lacks the log-determinant). f is the parallel
transformer forward, so a step costs one forward/backward pass and 1000 steps take
seconds; no sequential reverse is needed.

Control: the same pixel-domain optimisation started from the prior samples g(z_0) that
seeded the latent restarts (step 11's pixel-only image MAP). Both variants use the same
schedule, so the value of the latent initialisation is read off directly; the truth's
J_x is the reference. Per row at initialisation and at the end: J_x, J_z, data term
(against its noise level), log-determinant, |f(x)|, PSNR, max|x|; restart spread.

Outputs (outputs/step15_two_stage_map/):
  data/     two_stage_{res}.json, two_stage_{res}_solutions.pt, two_stage_{res}_traces.npz,
            summary.tex, provenance.json
  figures/  two_stage_{res}.png (rows = systems: truth | measurement | latent MAP -> refined |
            prior sample -> pixel-only), two_stage_{res}_curves.png (J_x and data term per step)

Usage:
  python step15_two_stage_map.py --res 32 [--device cuda:0]
  python step15_two_stage_map.py --collect          # summary.tex from the json files
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
DEFAULT_CONFIG = REPO_ROOT / "configs" / "step15_two_stage_map.yml"

import step10_variational_posterior as s10  # noqa: E402
import step11_map_adam as s11  # noqa: E402
import step13_lidc_tarflow as s13  # noqa: E402
import step14_lidc_gallery as s14  # noqa: E402

METRICS = ("J_x", "J_z", "data", "logdet", "z_norm", "psnr", "max_abs")


def load_config(path=DEFAULT_CONFIG):
    with open(path) as f:
        return yaml.safe_load(f)


def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


# --------------------------------------------------------------------------
# the image-space objective
# --------------------------------------------------------------------------

def forward_terms(prior, x):
    """z = f(x) and the two prior terms of J_x: -log N(z; 0, I) and log|det df/dx| (all
    dims; the model's logdet is the per-dimension mean). Same arithmetic as
    s10.prior_log_prob, without the activation checkpointing (a 32-px batch is small)."""
    z, _, ld = prior(x)
    return z, s11.neg_log_normal(z), ld * float(np.prod(x.shape[1:]))


@torch.no_grad()
def assess(prior, x, data_terms, x_true):
    z, nlpz, logdet = forward_terms(prior, x)
    data = data_terms(x)
    return dict(J_x=data + nlpz - logdet, J_z=data + nlpz, data=data, logdet=logdet,
                z_norm=z.flatten(1).norm(dim=-1), psnr=s14.psnr01(x, x_true),
                max_abs=x.flatten(1).abs().max(-1).values)


def optimise_pixels(prior, x_init, data_terms, pc, tag, sl, names):
    """Adam on the pixels of all rows at once (independent objectives, elementwise
    optimiser: exactly the separate optimisations); non-finite gradients are zeroed
    and counted, as in step 11."""
    x = x_init.clone().requires_grad_(True)
    K = int(pc["iters"])
    opt = torch.optim.Adam([x], lr=pc["lr"], betas=tuple(pc.get("betas", (0.9, 0.999))))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=K, eta_min=pc.get("lr_min", 0.0))
    tr = np.zeros((K, 4, x.size(0)))                                   # J_x, data, nlpz, logdet per step
    n_nonfinite = 0
    torch.cuda.synchronize()
    t0 = time.time()
    for k in range(K):
        z, nlpz, logdet = forward_terms(prior, x)
        data = data_terms(x)
        J = data + nlpz - logdet
        opt.zero_grad(set_to_none=True)
        J.sum().backward()
        bad = ~torch.isfinite(x.grad)
        if bad.any():
            n_nonfinite += int(bad.any(dim=(1, 2, 3)).sum())
            x.grad[bad] = 0.0
        opt.step()
        sched.step()
        tr[k] = torch.stack([J, data, nlpz, logdet]).detach().cpu().numpy()
        if k % int(pc.get("log_every", 100)) == 0 or k == K - 1:
            print(f"{tag} step {k:4d}: best-restart J_x per system "
                  + " ".join(f"{n} {float(J[sl(s)].min()):9.1f}" for s, n in enumerate(names))
                  + f"  lr {sched.get_last_lr()[0]:.5f}", flush=True)
    torch.cuda.synchronize()
    return x.detach(), tr, time.time() - t0, n_nonfinite


# --------------------------------------------------------------------------
# the experiment
# --------------------------------------------------------------------------

def run(cfg, res, device, out_data, out_figs):
    s13.set_tf32(False)
    cfg14 = s14.load_config(resolve(cfg["step14_config"]))
    out14 = REPO_ROOT / cfg14["output_root"] / "data"
    g14 = json.load(open(out14 / f"gallery_{res}.json"))
    sol = torch.load(out14 / f"gallery_{res}_solutions.pt", map_location="cpu", weights_only=False)   # holds a numpy J_trace
    cfg13 = s13.load_config(resolve(cfg14["priors"][res]))
    prior, ckpt = s13.load_prior(cfg13, device, cfg14["prior_checkpoint"])
    for p in prior.parameters():
        p.requires_grad_(False)
    T, D = s10.latent_shape(prior)
    systems = cfg14["systems"][:cfg14.get("n_rows", len(cfg14["systems"]))]
    names = [spec["name"] for spec in systems]
    S, R = len(systems), int(g14["restarts"])
    sl = lambda s: slice(s * R, (s + 1) * R)
    tag = f"[step15:{res}]"
    print(f"{tag} prior {ckpt} (T={T}, D={D}); stage-1 solutions {out14 / f'gallery_{res}_solutions.pt'} "
          f"({S} systems x {R} restarts, {g14['iters']} latent steps); test slices {g14['indices']}", flush=True)

    # the stage-1 problems, rebuilt from the step-14 spec with the stored measurements
    probs, x_trues, x_lat, z0 = [], [], [], []
    for s, (spec, row) in enumerate(zip(systems, g14["rows"])):
        st = sol[spec["name"]]
        x_true = st["x_true"].to(device)
        probs.append(s14.make_problem(spec, x_true, None, y=st["y"]))
        x_trues.append(x_true.expand(R, -1, -1, -1))
        x_lat.append(st["x"].to(device))
        torch.manual_seed(cfg14["seed"] + 100 * s)                         # the step-14 restart seeds
        z0.append(torch.randn(R, T, D, device=device))
    x_true_rows = torch.cat(x_trues)
    x_lat = torch.cat(x_lat)

    def data_terms(x):
        return torch.cat([p["nll"](x[sl(s)]) for s, p in enumerate(probs)])

    # consistency with the stored stage-1 numbers: the data term of the stored images and
    # J_z with z = f(x) (the round trip f(g(z)) is exact to fp32) against gallery_{res}.json
    a_lat = assess(prior, x_lat, data_terms, x_true_rows)
    d_data = max(abs(float(a_lat["data"][s * R + j]) - row["data"][j]) for s, row in enumerate(g14["rows"]) for j in range(R))
    d_J = max(abs(float(a_lat["J_z"][s * R + j]) - row["J"][j]) for s, row in enumerate(g14["rows"]) for j in range(R))
    z_lat = torch.cat([sol[n]["z"] for n in names]).to(device)
    with torch.no_grad():
        d_z = float((forward_terms(prior, x_lat)[0] - z_lat).abs().max())
    print(f"{tag} stage-1 check: max |data - stored| {d_data:.3g} nats, max |J_z - stored| {d_J:.3g} nats, "
          f"max |f(x) - z| {d_z:.2e}", flush=True)

    # the control's initialisation: the prior samples the latent restarts started from
    with torch.no_grad():
        x_prior = s10.draw_samples(prior, torch.cat(z0))
    inits = dict(latent_then_pixel=x_lat, pixel_only=x_prior)
    x_true1 = x_true_rows[::R]                                                 # one row per system
    truth = assess(prior, x_true1, lambda x: torch.cat([p["nll"](x[s:s + 1]) for s, p in enumerate(probs)]), x_true1)

    pc = dict(cfg["pixel"])
    results, store, traces = {}, {}, {}
    for var in cfg["variants"]:
        print(f"{tag} variant {var}: Adam on pixels, {pc['iters']} steps, lr {pc['lr']} -> {pc['lr_min']} cosine", flush=True)
        a0 = assess(prior, inits[var], data_terms, x_true_rows)
        x_out, tr, secs, n_bad = optimise_pixels(prior, inits[var], data_terms, pc, f"{tag} {var}", sl, names)
        a1 = assess(prior, x_out, data_terms, x_true_rows)
        results[var] = dict(init=a0, final=a1, seconds=secs, n_nonfinite=n_bad)
        store[var] = dict(init=inits[var].cpu(), final=x_out.cpu())
        traces[var] = tr
        print(f"{tag} variant {var}: {secs:.0f} s ({secs / pc['iters'] * 1e3:.0f} ms/step), {n_bad} non-finite gradient rows", flush=True)

    sweep = []
    for lr in cfg.get("lr_sweep", []):
        pcs = dict(pc, lr=lr)
        x_out, tr, secs, n_bad = optimise_pixels(prior, x_lat, data_terms, pcs, f"{tag} sweep lr {lr}", sl, names)
        a1 = assess(prior, x_out, data_terms, x_true_rows)
        sweep.append(dict(lr=lr, J_x=[float(v) for v in a1["J_x"]], psnr=[float(v) for v in a1["psnr"]],
                          data=[float(v) for v in a1["data"]], z_norm=[float(v) for v in a1["z_norm"]], n_nonfinite=n_bad))
        print(f"{tag} sweep lr {lr}: best-restart J_x per system "
              + " ".join(f"{n} {min(sweep[-1]['J_x'][sl(s)]):9.1f}" for s, n in enumerate(names)), flush=True)

    rows = []
    for s, (spec, row) in enumerate(zip(systems, g14["rows"])):
        out = dict(model=spec["name"], params=probs[s]["params"], index=row["index"], M=probs[s]["M"], sigma=probs[s]["sigma"],
                   noise_level=float(probs[s]["noise_level"]),
                   truth={m: float(truth[m][s]) for m in METRICS})
        for var, rv in results.items():
            xs = store[var]["final"][sl(s)]
            out[var] = dict(init={m: [float(v) for v in rv["init"][m][sl(s)]] for m in METRICS},
                            final={m: [float(v) for v in rv["final"][m][sl(s)]] for m in METRICS},
                            pairwise_rms=float(((xs[:, None] - xs[None]) ** 2).mean((2, 3, 4)).sqrt().sum() / (R * (R - 1))),
                            J_min_over_steps=[float(v) for v in traces[var][:, 0, sl(s)].min(0)])
        rows.append(out)
        a, b = out["latent_then_pixel"], out["pixel_only"]
        print(f"{tag} {spec['name']:15s} #{row['index']}: truth J_x {out['truth']['J_x']:.0f} (J_z {out['truth']['J_z']:.0f}, data "
              f"{out['truth']['data']:.0f}, noise level {out['noise_level']:.0f}, logdet {out['truth']['logdet']:.0f})\n"
              f"{tag}    latent MAP     J_x {['%.0f' % v for v in a['init']['J_x']]} J_z {['%.0f' % v for v in a['init']['J_z']]} "
              f"PSNR {['%.1f' % v for v in a['init']['psnr']]}\n"
              f"{tag}    -> refined     J_x {['%.0f' % v for v in a['final']['J_x']]} data {['%.0f' % v for v in a['final']['data']]} "
              f"logdet {['%.0f' % v for v in a['final']['logdet']]} PSNR {['%.1f' % v for v in a['final']['psnr']]} "
              f"|f(x)| {['%.1f' % v for v in a['final']['z_norm']]} max|x| {['%.2f' % v for v in a['final']['max_abs']]} spread {a['pairwise_rms']:.3f}\n"
              f"{tag}    pixel-only     J_x {['%.0f' % v for v in b['final']['J_x']]} data {['%.0f' % v for v in b['final']['data']]} "
              f"logdet {['%.0f' % v for v in b['final']['logdet']]} PSNR {['%.1f' % v for v in b['final']['psnr']]} "
              f"|f(x)| {['%.1f' % v for v in b['final']['z_norm']]} max|x| {['%.2f' % v for v in b['final']['max_abs']]} spread {b['pairwise_rms']:.3f}",
              flush=True)

    res_json = dict(res=res, prior=str(ckpt), latent=dict(T=T, D=D), restarts=R, stage1=dict(iters=g14["iters"], lr=g14["lr"], lr_min=g14["lr_min"]),
                    pixel=pc, indices=g14["indices"], check=dict(data=d_data, J_z=d_J, z_roundtrip=d_z),
                    seconds={v: r["seconds"] for v, r in results.items()}, n_nonfinite={v: r["n_nonfinite"] for v, r in results.items()},
                    sqrt_D=math.sqrt(T * D), rows=rows, lr_sweep=sweep)
    json.dump(res_json, open(out_data / f"two_stage_{res}.json", "w"), indent=1)
    torch.save(dict(x_true=x_true_rows[::R].cpu(), **{v: st for v, st in store.items()}), out_data / f"two_stage_{res}_solutions.pt")
    np.savez_compressed(out_data / f"two_stage_{res}_traces.npz", **{v: tr for v, tr in traces.items()}, names=np.array(names))

    figures(cfg, res, out_data, out_figs)
    print(f"{tag} done", flush=True)
    return res_json


def figures(cfg, res, out_data, out_figs):
    """Both figures from the saved json / solutions / traces (re-runnable, CPU)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = json.load(open(out_data / f"two_stage_{res}.json"))
    rows, R, pc = g["rows"], g["restarts"], g["pixel"]
    store = torch.load(out_data / f"two_stage_{res}_solutions.pt", map_location="cpu")
    traces = dict(np.load(out_data / f"two_stage_{res}_traces.npz"))
    cfg14 = s14.load_config(resolve(cfg["step14_config"]))
    sol = torch.load(REPO_ROOT / cfg14["output_root"] / "data" / f"gallery_{res}_solutions.pt", map_location="cpu", weights_only=False)
    systems = cfg14["systems"][:cfg14.get("n_rows", len(cfg14["systems"]))]
    probs = [s14.make_problem(spec, sol[spec["name"]]["x_true"], None, y=sol[spec["name"]]["y"]) for spec in systems]

    # gallery: the restart with the lowest final J_x of each variant, before and after stage 2
    cols = ["truth", "measurement", "latent MAP (stage 1)", "+ pixel refinement", "prior sample", "pixel-only MAP"]
    fig, axes = plt.subplots(len(systems), len(cols), figsize=(1.9 * len(cols) + 1.2, 2.05 * len(systems)))
    for s, (spec, row) in enumerate(zip(systems, rows)):
        st = sol[spec["name"]]
        s10._img(axes[s, 0], st["x_true"][0, 0], f"truth, test #{row['index']}\n$J_x$ {row['truth']['J_x']:.0f}")
        s14.show_measurement(axes[s, 1], probs[s]["op"], probs[s]["y"], row["M"])
        for c, var in ((2, "latent_then_pixel"), (4, "pixel_only")):
            b = int(np.argmin(row[var]["final"]["J_x"]))
            for cc, stage in ((c, "init"), (c + 1, "final")):
                m = row[var][stage]
                s10._img(axes[s, cc], store[var][stage][s * R + b, 0],
                         f"{cols[cc]} (r{b + 1})\n{m['psnr'][b]:.1f} dB, $J_x$ {m['J_x'][b]:.0f}")
        axes[s, 0].text(-0.12, 0.5, s14.describe(spec, res), transform=axes[s, 0].transAxes, rotation=90, va="center",
                        ha="center", fontsize=7)
        for ax in axes[s]:
            ax.title.set_fontsize(6.5)
    fig.suptitle(f"LIDC {res}$\\times${res}: latent MAP (step 14) $\\to$ pixel-domain refinement of $J_x$ (Adam on pixels, "
                 f"{pc['iters']} steps, lr {pc['lr']} cosine), and the pixel-only MAP from the same prior samples; "
                 f"lowest-$J_x$ restart of each", fontsize=8)
    fig.tight_layout(rect=(0.02, 0, 1, 0.97))
    fig.savefig(out_figs / f"two_stage_{res}.png", dpi=150)
    plt.close(fig)

    # curves: J_x relative to the truth's, and the squared residual in units of its expectation
    # M sigma^2 (data term = |r|^2 / 2 sigma^2 + const, noise level = const + M/2)
    colours = dict(latent_then_pixel="tab:blue", pixel_only="tab:orange")
    fig, axes = plt.subplots(2, len(systems), figsize=(2.6 * len(systems), 4.6), squeeze=False)
    for s, (spec, row) in enumerate(zip(systems, rows)):
        half_M = 0.5 * row["M"]
        for var in ("latent_then_pixel", "pixel_only"):
            tr = traces[var]
            for j in range(R):
                axes[0, s].plot(tr[:, 0, s * R + j] - row["truth"]["J_x"], color=colours[var], lw=0.8, alpha=0.8,
                                label=var.replace("_", " ") if j == 0 else None)
                axes[1, s].plot((tr[:, 1, s * R + j] - row["noise_level"] + half_M) / half_M, color=colours[var], lw=0.8, alpha=0.8)
        axes[0, s].axhline(0, color="k", ls="--", lw=0.8)
        axes[1, s].axhline((row["truth"]["data"] - row["noise_level"] + half_M) / half_M, color="k", ls="--", lw=0.8)
        axes[0, s].set_yscale("symlog", linthresh=100)
        axes[1, s].set_yscale("log")
        axes[0, s].set_title(s14.describe(spec, res).replace("\n", ", "), fontsize=7)
        axes[1, s].set_xlabel("pixel step", fontsize=7)
        for ax in axes[:, s]:
            ax.tick_params(labelsize=6)
    axes[0, 0].set_ylabel("$J_x - J_x(\\mathrm{truth})$ [nats]", fontsize=7)
    axes[1, 0].set_ylabel("$\\|y - Ax\\|^2 / (M \\sigma^2)$", fontsize=7)
    axes[0, 0].legend(fontsize=6, loc="upper right")
    fig.suptitle(f"LIDC {res}$\\times${res}: pixel-domain stage, all restarts (truth dashed)", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_figs / f"two_stage_{res}_curves.png", dpi=150)
    plt.close(fig)


def _tex(s):
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")


def collect(cfg, out_data):
    lines = []
    for p in sorted(out_data.glob("two_stage_*.json")):
        g = json.load(open(p))
        parts = []
        for row in g["rows"]:
            a, b, t = row["latent_then_pixel"], row["pixel_only"], row["truth"]
            ba, bb = int(np.argmin(a["final"]["J_x"])), int(np.argmin(b["final"]["J_x"]))
            parts.append(f"{_tex(row['model'])}: $J_x$ {a['init']['J_x'][ba]:.0f} $\\to$ {a['final']['J_x'][ba]:.0f} "
                         f"(truth {t['J_x']:.0f}; pixel-only {b['final']['J_x'][bb]:.0f}), data {a['final']['data'][ba]:.0f} "
                         f"(noise level {row['noise_level']:.0f}; pixel-only {b['final']['data'][bb]:.0f}), "
                         f"PSNR {a['init']['psnr'][ba]:.1f} $\\to$ {a['final']['psnr'][ba]:.1f} dB (pixel-only {b['final']['psnr'][bb]:.1f}), "
                         f"$\\|f(x)\\|$ {a['final']['z_norm'][ba]:.0f} (truth {t['z_norm']:.0f}; pixel-only {b['final']['z_norm'][bb]:.0f}), "
                         f"restart spread {a['pairwise_rms']:.3f} (pixel-only {b['pairwise_rms']:.3f})")
        secs = g["seconds"]["latent_then_pixel"]
        lines.append(f"{g['res']}$\\times${g['res']} ({g['restarts']} restarts; stage 1 {g['stage1']['iters']} latent steps, "
                     f"stage 2 {g['pixel']['iters']} pixel steps at lr {g['pixel']['lr']} cosine, {secs:.0f} s): " + "; ".join(parts) + ".")
    (out_data / "summary.tex").write_text("\n".join(lines) + "\n")
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--res", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--figures", action="store_true", help="remake the figures of --res from the saved outputs")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    out_root = REPO_ROOT / cfg["output_root"]
    out_data, out_figs = out_root / "data", out_root / "figures"
    out_data.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)
    if args.res and args.figures:
        figures(cfg, args.res, out_data, out_figs)
    elif args.res:
        run(cfg, args.res, torch.device(args.device), out_data, out_figs)
    collect(cfg, out_data)
    prov = dict(step="step15_two_stage_map", command=" ".join(["python"] + (argv or sys.argv[1:])),
                finished=datetime.now(timezone.utc).isoformat(), config=cfg)
    with open(out_data / "provenance.json", "w") as f:
        json.dump(prov, f, indent=2)


if __name__ == "__main__":
    main()
