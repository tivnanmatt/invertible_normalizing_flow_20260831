#!/usr/bin/env python
"""step14_lidc_gallery.py -- Six inverse problems on LIDC slices under the step-13 priors.

For the prior at one resolution (--res 32/64/128/256) and one held-out test slice per
system: denoising, box inpainting, random-pixel inpainting, 2x and 4x super-resolution,
sparse-view CT. Each is solved by latent-space MAP -- Adam on z with x = g_theta(z)
through the CUDA-graph sequential reverse (invlib.FastReverse), objective
J_z(z) = -log p(y | g(z)) - log N(z; 0, I), several independent random restarts -- which
is the step-11 estimator with the schedule the step-11 learning-rate sweep picked
(lr 0.1 -> 0.005, cosine). All systems' restarts are optimised together as ONE batch
(6 x 3 = 18 rows): the sequential reverse is latency-bound, so this costs the same per
step as a single system, and since Adam is elementwise and the objective a sum over
rows it is exactly the independent optimisations. The restarts are NOT posterior
samples: they are local optima of the MAP objective from different inits; their
spread is a (lower-bound) hint of multimodality only.

Every likelihood uses the exact forward operator: pixel masks, 2^L x 2^L average pooling,
and a matrix-free pixel-driven parallel-beam Radon transform (RadonOp, identical to
measlib.radon_matrix but without the dense M x D matrix and its SVD, which do not exist
at 256 px). The system parameters scale with the side N as stated in the config.

Outputs (outputs/step14_lidc_gallery/):
  data/     gallery_{res}.json (per row: objective/data/PSNR/|z| per restart, truth values,
            noise level, timings), gallery_{res}_solutions.pt, verify.json, summary.tex, provenance.json
  figures/  gallery_{res}.png (rows = systems: truth | measurement | restarts)

Usage:
  python step14_lidc_gallery.py --res 32 [--device cuda:0]
  python step14_lidc_gallery.py --verify            # RadonOp == measlib.radon_matrix at 32 px
  python step14_lidc_gallery.py --collect           # summary.tex from the json files
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
import torch.nn.functional as F
import yaml

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "step14_lidc_gallery.yml"

import invlib  # noqa: E402
import measlib  # noqa: E402
import step2_tarflow as s2  # noqa: E402
import step10_variational_posterior as s10  # noqa: E402
import step11_map_adam as s11  # noqa: E402
import step13_lidc_tarflow as s13  # noqa: E402


def load_config(path=DEFAULT_CONFIG):
    with open(path) as f:
        return yaml.safe_load(f)


def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


# --------------------------------------------------------------------------
# forward operators
# --------------------------------------------------------------------------

class RadonOp:
    """Pixel-driven parallel-beam Radon transform of N x N images, matrix-free:
    each pixel centre is projected onto the detector axis of every view and shared
    between the two nearest bins by linear interpolation (unit spacings, views
    equally spaced over [0, pi)) -- the construction of measlib.radon_matrix."""

    def __init__(self, N, n_angles, device):
        n_det = int(math.ceil(N * math.sqrt(2))) | 1
        u = torch.arange(N, dtype=torch.float64) - (N - 1) / 2
        yy, xx = torch.meshgrid(u, u, indexing="ij")
        pix = torch.arange(N * N)
        idx, w, src = [], [], []
        for a in range(n_angles):
            th = math.pi * a / n_angles
            t = (xx * math.cos(th) + yy * math.sin(th)).flatten() + (n_det - 1) / 2
            i0 = torch.floor(t).long()
            w1 = t - i0.double()
            for i, ww in ((i0, 1 - w1), (i0 + 1, w1)):
                ok = (i >= 0) & (i < n_det)
                idx.append((a * n_det + i.clamp(0, n_det - 1)))
                w.append(torch.where(ok, ww, torch.zeros_like(ww)))
                src.append(pix)
        self.N, self.n_angles, self.n_det = N, n_angles, n_det
        self.index = torch.cat(idx).to(device)
        self.weight = torch.cat(w).to(device)                 # float64; cast per call
        self.src = torch.cat(src).to(device)
        self.n_meas = n_angles * n_det

    def __call__(self, x):
        """(B, 1, N, N) -> (B, n_angles, n_det)."""
        v = x.flatten(1)[:, self.src] * self.weight.to(x.dtype)
        y = torch.zeros(x.size(0), self.n_meas, device=x.device, dtype=x.dtype)
        y.index_add_(1, self.index, v)
        return y.view(x.size(0), self.n_angles, self.n_det)


def apply_A(op, x):
    """The exact forward operator, y-shaped output."""
    if isinstance(op, measlib._IdentityBasis):
        return op.sing(x.device, x.dtype).unsqueeze(0) * x
    if isinstance(op, measlib.AveragePoolSR):
        return F.avg_pool2d(x, 2 ** op.levels)
    if isinstance(op, RadonOp):
        return op(x)
    raise TypeError(type(op))


def system_params(spec, N):
    """The system's constructor kwargs at side N (config values are for 64 px)."""
    kw = {k: v for k, v in spec.items() if k not in ("name", "label", "sigma", "box_frac", "views_per_px", "sigma_per_px", "mask_seed")}
    sigma = float(spec.get("sigma", 0.0))
    if spec["name"] == "inpaint_box":
        kw["box"] = int(round(spec["box_frac"] * N))
    if spec["name"] == "inpaint_random":
        kw["seed"] = spec.get("mask_seed", 0)
    if spec["name"] == "ct":
        kw["n_angles"] = int(round(spec["views_per_px"] * N))
        sigma = float(spec["sigma_per_px"] * N)
    return kw, sigma


def make_problem(spec, x, gen, y=None):
    """y = A x + sigma eps (noise only on measured coordinates) and the exact -log p(y|x).
    With `y` given (a stored measurement, e.g. step 15 re-using this gallery's problems)
    no noise is drawn and the generator is untouched."""
    N = x.shape[-1]
    kw, sigma = system_params(spec, N)
    if spec["name"] == "ct":
        op = RadonOp(N, kw["n_angles"], x.device)
    else:
        op = measlib.build(spec["name"], tuple(x.shape[1:]), sigma=sigma, **kw)
    Ax = apply_A(op, x)
    eps = None if y is not None else torch.randn(Ax.shape, generator=gen).to(x.device)
    if isinstance(op, measlib._IdentityBasis):
        mask = op.sing(x.device, x.dtype).unsqueeze(0)
        M = int(mask.sum())
        y = mask * (x + sigma * eps) if eps is not None else y.to(x.device)
    else:
        M = int(Ax[0].numel())
        y = Ax + sigma * eps if eps is not None else y.to(x.device)
    const = 0.5 * M * math.log(2 * math.pi * sigma ** 2)

    def nll(xx):
        return ((apply_A(op, xx) - y) ** 2).flatten(1).sum(-1) / (2 * sigma ** 2) + const

    desc = dict(spec, **kw, sigma=sigma, M=M)
    return dict(op=op, name=spec["name"], sigma=sigma, y=y, M=M, nll=nll, noise_level=const + 0.5 * M, params=desc)


def describe(spec, N):
    kw, sigma = system_params(spec, N)
    if spec["name"] == "inpaint_box":
        return f"{spec['label']} {kw['box']}$\\times${kw['box']}\n$\\sigma={sigma}$"
    if spec["name"] == "ct":
        return f"{spec['label']}, {kw['n_angles']} views\n$\\sigma={sigma:g}$ (sinogram)"
    return f"{spec['label']}\n$\\sigma={sigma}$"


def psnr01(x, x_true):
    mse = ((s10._to01(x) - s10._to01(x_true)) ** 2).flatten(1).mean(-1)
    return 10 * torch.log10(1 / mse.clamp_min(1e-12))


def show_measurement(ax, op, y, M):
    if isinstance(op, measlib._IdentityBasis):
        y01 = s10._to01(y)[0, 0].cpu().numpy()
        rgb = np.stack([y01] * 3, -1)
        hole = (op.sing(y.device, y.dtype)[0] == 0).cpu().numpy()
        if hole.any():
            rgb[hole] = 0.75 * rgb[hole] + 0.25 * np.array([1.0, 0.0, 0.0])
            rgb[hole, 0] = np.maximum(rgb[hole, 0], 0.35)
        ax.imshow(rgb, interpolation="nearest")
        ttl = f"measurement ({M} px)"
    elif isinstance(op, measlib.AveragePoolSR):
        ax.imshow(s10._to01(y)[0, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ttl = f"measurement {y.shape[-1]}$\\times${y.shape[-1]}"
    else:
        ax.imshow(y[0].cpu().numpy(), cmap="gray", aspect="auto", interpolation="nearest")
        ttl = f"sinogram {op.n_angles}$\\times${op.n_det}"
    ax.set_title(ttl, fontsize=7)
    ax.axis("off")


# --------------------------------------------------------------------------
# the gallery
# --------------------------------------------------------------------------

def run(cfg, res, device, out_data, out_figs):
    s13.set_tf32(False)
    cfg13 = s13.load_config(resolve(cfg["priors"][res]))
    prior, ckpt = s13.load_prior(cfg13, device, cfg["prior_checkpoint"])
    for p in prior.parameters():
        p.requires_grad_(False)
    test = s2.TarFlowInput(s13.build_bundle(cfg13).test)
    T, D = s10.latent_shape(prior)
    R, K = int(cfg["restarts"]), int(cfg["iters"])
    systems = cfg["systems"][:cfg.get("n_rows", len(cfg["systems"]))]
    gen = torch.Generator().manual_seed(cfg["seed"])
    indices = torch.randperm(len(test), generator=gen)[:len(systems)].tolist()
    tag = f"[step14:{res}]"
    S = len(systems)
    print(f"{tag} prior {ckpt} (T={T}, D={D}); test slices {indices} of {len(test)}; "
          f"{S} systems x {R} restarts = one batch of {S * R}", flush=True)
    # the sequential reverse is latency-bound (a token step costs the same at batch 3 and
    # batch 18), so all systems' restarts are optimised together as one batch: Adam is
    # elementwise and the objective is a sum over rows, so this is exactly the R
    # independent optimisations per system, S times, for the price of one
    torch.cuda.synchronize()
    t0 = time.time()
    reverse = cfg.get("reverse", "fast")
    if reverse == "fast2":        # in-place cache (invlib.FastReverse2): O(B T) traffic per token step, not O(B T^2)
        fr = invlib.FastReverse2(prior, S * R, device, chunk=int(cfg.get("chunk", 128)), mode="graph")
    else:
        fr = invlib.FastReverse(prior, S * R, device, mode="compile")
    torch.cuda.synchronize()
    build_s = time.time() - t0
    print(f"{tag} graph build {build_s:.1f} s ({reverse})", flush=True)

    probs, x_trues, z0 = [], [], []
    for r, (spec, idx) in enumerate(zip(systems, indices)):
        x_true = test[idx][0].unsqueeze(0).to(device)
        probs.append(make_problem(spec, x_true, gen))
        x_trues.append(x_true)
        torch.manual_seed(cfg["seed"] + 100 * r)
        z0.append(torch.randn(R, T, D, device=device))
    z = torch.cat(z0).requires_grad_(True)
    sl = lambda s: slice(s * R, (s + 1) * R)                                    # rows of system s

    def data_terms(x):
        return torch.cat([p["nll"](x[sl(s)]) for s, p in enumerate(probs)])

    opt = torch.optim.Adam([z], lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=K, eta_min=cfg["lr_min"])
    J_tr, ts = np.zeros((K, S * R)), []
    torch.cuda.synchronize()
    t_all = time.time()
    for k in range(K):
        t1 = time.time()
        x, _ = fr(z)
        J = data_terms(x) + s11.neg_log_normal(z)
        opt.zero_grad(set_to_none=True)
        J.sum().backward()
        opt.step()
        sched.step()
        torch.cuda.synchronize()
        ts.append(time.time() - t1)
        J_tr[k] = J.detach().cpu().numpy()
        if k % 100 == 0 or k == K - 1:
            print(f"{tag} step {k:4d}: best-restart J per system "
                  + " ".join(f"{p['name']} {float(J[sl(s)].min()):9.1f}" for s, p in enumerate(probs))
                  + f"  lr {sched.get_last_lr()[0]:.4f}  {ts[-1]:.2f} s/step", flush=True)
    total = time.time() - t_all
    with torch.no_grad():
        x, _ = fr(z)
        data = data_terms(x)
        J = data + s11.neg_log_normal(z)

    rows, store = [], {}
    for s, (spec, idx, prob, x_true) in enumerate(zip(systems, indices, probs, x_trues)):
        name = spec["name"]
        xs, zs, Js, ds = x[sl(s)], z.detach()[sl(s)], J[sl(s)], data[sl(s)]
        with torch.no_grad():
            data_true = prob["nll"](x_true)
            zt, _, _ = prior(x_true)
            J_true = float(data_true[0] + s11.neg_log_normal(zt)[0])
            ps = psnr01(xs, x_true.expand_as(xs))
        row = dict(model=name, params=prob["params"], index=int(idx), M=prob["M"], sigma=prob["sigma"],
                   noise_level=float(prob["noise_level"]), data_true=float(data_true[0]), J_true=J_true,
                   J=[float(v) for v in Js], data=[float(v) for v in ds], psnr=[float(v) for v in ps],
                   z_norm=[float(v) for v in zs.flatten(1).norm(dim=-1)],
                   max_abs=[float(v) for v in xs.flatten(1).abs().max(-1).values],
                   pairwise_rms=float(((xs[:, None] - xs[None]) ** 2).mean((2, 3, 4)).sqrt().sum() / (R * (R - 1))),
                   seconds=total, step_seconds=float(np.mean(ts)), J_min_over_steps=[float(v) for v in J_tr[:, sl(s)].min(0)])
        rows.append(row)
        store[name] = dict(x_true=x_true.cpu(), y=prob["y"].cpu(), x=xs.cpu(), z=zs.cpu(), J_trace=J_tr[:, sl(s)], op=prob["op"])
        print(f"{tag} {name:15s} #{idx}: {total:.0f} s ({row['step_seconds']:.2f} s/step); J {['%.0f' % v for v in row['J']]} "
              f"(truth {J_true:.0f}); data {['%.0f' % v for v in row['data']]} (truth {row['data_true']:.0f}, noise level "
              f"{row['noise_level']:.0f}); PSNR {['%.1f' % v for v in row['psnr']]}; |z| {['%.1f' % v for v in row['z_norm']]}; "
              f"pairwise RMS {row['pairwise_rms']:.3f}", flush=True)
    res_json = dict(res=res, prior=str(ckpt), latent=dict(T=T, D=D), restarts=R, iters=K, lr=cfg["lr"], lr_min=cfg["lr_min"],
                    seed=cfg["seed"], indices=indices, batch=S * R, reverse=reverse, graph_build_seconds=build_s, rows=rows,
                    sqrt_D=math.sqrt(T * D))
    json.dump(res_json, open(out_data / f"gallery_{res}.json", "w"), indent=1)
    torch.save({k: {kk: vv for kk, vv in v.items() if kk != "op"} for k, v in store.items()},
               out_data / f"gallery_{res}_solutions.pt")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nrow, ncol = len(systems), 2 + R
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.9 * ncol + 1.2, 1.95 * nrow))
    for r, (spec, row) in enumerate(zip(systems, rows)):
        st = store[spec["name"]]
        s10._img(axes[r, 0], st["x_true"][0, 0], f"truth, test #{row['index']}")
        show_measurement(axes[r, 1], st["op"], st["y"], row["M"])
        for j in range(R):
            s10._img(axes[r, 2 + j], st["x"][j, 0], f"restart {j + 1}: {row['psnr'][j]:.1f} dB")
        axes[r, 0].text(-0.12, 0.5, describe(spec, res), transform=axes[r, 0].transAxes, rotation=90, va="center",
                        ha="center", fontsize=7)
    fig.suptitle(f"LIDC {res}$\\times${res}: latent MAP restarts (Adam, {K} steps, lr {cfg['lr']}$\\to${cfg['lr_min']} cosine, "
                 f"independent random inits) under the step-13 prior, one test slice per system", fontsize=8)
    fig.tight_layout(rect=(0.02, 0, 1, 0.97))
    fig.savefig(out_figs / f"gallery_{res}.png", dpi=150)
    plt.close(fig)
    print(f"{tag} gallery done", flush=True)
    return res_json


def verify(out_data):
    """RadonOp reproduces measlib.radon_matrix (dense) at 32 px, for random images."""
    N, n_angles = 32, 8
    A = measlib.radon_matrix(N, n_angles)
    op = RadonOp(N, n_angles, "cpu")
    x = torch.randn(4, 1, N, N, dtype=torch.float64)
    y_dense = (x.flatten(1) @ A.T).view(4, n_angles, -1)
    y_op = op(x)
    err = float((y_dense - y_op).abs().max())
    # gradient check of the likelihood through RadonOp
    xg = x[:1].clone().requires_grad_(True)
    (op(xg) ** 2).sum().backward()
    g_dense = 2 * (A.T @ (A @ x[0].flatten())).view(1, 1, N, N)
    gerr = float((xg.grad - g_dense).abs().max())
    res = dict(N=N, n_angles=n_angles, n_det=op.n_det, max_abs_err=err, grad_max_abs_err=gerr,
               ok=bool(err < 1e-10 and gerr < 1e-9))
    json.dump(res, open(out_data / "verify.json", "w"), indent=1)
    print("[step14] verify:", res, flush=True)
    return res


def _tex(s):
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")


def collect(cfg, out_data):
    lines = []
    for res in sorted(int(k) for k in cfg["priors"]):
        p = out_data / f"gallery_{res}.json"
        if not p.exists():
            continue
        g = json.load(open(p))
        parts = []
        for row in g["rows"]:
            # everything is the lowest-objective restart (the one the method would pick); the best
            # PSNR of the three is added only when another restart beats it (needs the truth)
            best = int(np.argmin(row["J"]))
            ps, pmax = row["psnr"][best], max(row["psnr"])
            psnr = f"PSNR {ps:.1f} dB" + (f" (best restart {pmax:.1f})" if pmax - ps >= 0.05 else "")
            parts.append(f"{_tex(row['model'])}: $J_z$ {min(row['J']):.0f} (truth {row['J_true']:.0f}), data {row['data'][best]:.0f} "
                         f"(noise level {row['noise_level']:.0f}), {psnr}, $\\|z\\|$ {row['z_norm'][best]:.0f} "
                         f"(typical {g['sqrt_D']:.0f}), restart spread {row['pairwise_rms']:.3f}")
        # one batch for all systems (`batch` recorded) or, in the archived first runs, one after the other
        secs = g["rows"][0]["seconds"] if "batch" in g else sum(r["seconds"] for r in g["rows"])
        lines.append(f"{res}$\\times${res} ($T={g['latent']['T']}$, $D={g['latent']['D']}$; {g['restarts']} restarts $\\times$ {g['iters']} steps"
                     + (f", all systems in one batch of {g['batch']}" if "batch" in g else "")
                     + (", in-place cache" if g.get("reverse") == "fast2" else "")
                     + f", {g['rows'][0]['step_seconds']:.2f} s per step, {secs / 60:.0f} min): " + "; ".join(parts) + ".")
    (out_data / "summary.tex").write_text("\n".join(lines) + "\n")
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--res", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--collect", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    out_root = REPO_ROOT / cfg["output_root"]
    out_data, out_figs = out_root / "data", out_root / "figures"
    out_data.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)
    if args.verify:
        verify(out_data)
    if args.res:
        run(cfg, args.res, torch.device(args.device), out_data, out_figs)
    collect(cfg, out_data)
    prov = dict(step="step14_lidc_gallery", command=" ".join(["python"] + (argv or sys.argv[1:])),
                finished=datetime.now(timezone.utc).isoformat(), config=cfg)
    with open(out_data / "provenance.json", "w") as f:
        json.dump(prov, f, indent=2)


if __name__ == "__main__":
    main()
