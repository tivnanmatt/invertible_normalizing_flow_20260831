"""step11_gallery.py -- Gallery: one row per measurement model in measlib, a DIFFERENT
ChestMNIST-64 test image per row; columns = truth | measurement | three latent-MAP
restarts (Adam in the latent space of the step-9 prior, independent random inits,
CUDA-graph reverse; the schedule chosen by step11_lr_sweep.py). The step-14 gallery
repeats this on the LIDC priors of every resolution.

Every system is y = A x + sigma eps with the exact forward operator A (not the SVD
truncation): inpainting/denoising in pixels, average pooling for SR, the Radon matrix
for CT (y is a 16 x 91 sinogram).

Outputs: outputs/step11_map_adam/data/gallery{.json,_solutions.pt}, figures/gallery.png
Usage (container): python step11_gallery.py --device cuda:0 --iters 300
"""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import invlib, measlib
import step2_tarflow as s2
import step9_chestmnist_tarflow as s9
import step10_variational_posterior as s10
import step11_map_adam as s11

OUT_DATA = REPO / "outputs" / "step11_map_adam" / "data"
OUT_FIGS = REPO / "outputs" / "step11_map_adam" / "figures"

# (name, kwargs incl. sigma, display label); sigma is in the units of y (image scale [-1, 1];
# for CT the sinogram, whose entries are ray sums of up to ~64 pixels)
MODELS = [
    ("denoise", dict(sigma=0.25), "denoising\n$\\sigma=0.25$"),
    ("inpaint_box", dict(box=24, sigma=0.05), "inpainting, 24$\\times$24 box\n$\\sigma=0.05$"),
    ("inpaint_random", dict(drop=0.7, sigma=0.05, seed=0), "inpainting, 70% pixels dropped\n$\\sigma=0.05$"),
    ("sr_2x", dict(sigma=0.05), "super-resolution 2$\\times$\n$\\sigma=0.05$"),
    ("sr_4x", dict(sigma=0.05), "super-resolution 4$\\times$\n$\\sigma=0.05$"),
    ("ct", dict(n_angles=16, sigma=1.0), "sparse-view CT, 16 views\n$\\sigma=1.0$ (sinogram)"),
]


def apply_A(op, x):
    """The exact forward operator, y-shaped output."""
    if isinstance(op, measlib._IdentityBasis):
        return op.sing(x.device, x.dtype).unsqueeze(0) * x
    if isinstance(op, measlib.AveragePoolSR):
        return F.avg_pool2d(x, 2 ** op.levels)
    if isinstance(op, measlib.DenseSystem):
        return x.flatten(1) @ op.A.to(device=x.device, dtype=x.dtype).T
    raise TypeError(type(op))


def make_problem(name, kw, x, gen):
    """y = A x + sigma eps (noise only on the measured coordinates), plus the likelihood."""
    kw = dict(kw); sigma = float(kw.pop("sigma"))
    op = measlib.build(name, tuple(x.shape[1:]), sigma=sigma, **kw)
    Ax = apply_A(op, x)
    eps = torch.randn(Ax.shape, generator=gen).to(x.device)
    if isinstance(op, measlib._IdentityBasis):
        mask = op.sing(x.device, x.dtype).unsqueeze(0)
        y = mask * (x + sigma * eps); M = int(mask.sum())
    else:
        y = Ax + sigma * eps; M = int(y.numel())
    const = 0.5 * M * math.log(2 * math.pi * sigma ** 2)

    def nll(xx):
        return ((apply_A(op, xx) - y) ** 2).flatten(1).sum(-1) / (2 * sigma ** 2) + const
    return dict(op=op, name=name, sigma=sigma, y=y, M=M, nll=nll, noise_level=const + 0.5 * M)


def psnr01(x, x_true):
    mse = ((s10._to01(x) - s10._to01(x_true)) ** 2).flatten(1).mean(-1)
    return 10 * torch.log10(1 / mse.clamp_min(1e-12))


def show_measurement(ax, prob, x_shape):
    op, y = prob["op"], prob["y"]
    if isinstance(op, measlib._IdentityBasis):
        y01 = s10._to01(y)[0, 0].cpu().numpy()
        rgb = np.stack([y01] * 3, -1)
        hole = (op.sing(y.device, y.dtype)[0] == 0).cpu().numpy()
        if hole.any():
            rgb[hole] = 0.75 * rgb[hole] + 0.25 * np.array([1.0, 0.0, 0.0]); rgb[hole, 0] = np.maximum(rgb[hole, 0], 0.35)
        ax.imshow(rgb, interpolation="nearest"); ttl = f"measurement ({prob['M']} px)"
    elif isinstance(op, measlib.AveragePoolSR):
        ax.imshow(s10._to01(y)[0, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ttl = f"measurement {y.shape[-1]}$\\times${y.shape[-1]}"
    else:
        sino = y.reshape(op.n_angles, op.n_det).cpu().numpy()
        ax.imshow(sino, cmap="gray", aspect="auto", interpolation="nearest")
        ttl = f"sinogram {op.n_angles}$\\times${op.n_det}"
    ax.set_title(ttl, fontsize=7); ax.axis("off")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--lr-min", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=20260913)
    args = ap.parse_args()
    OUT_DATA.mkdir(parents=True, exist_ok=True); OUT_FIGS.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    cfg = s11.load_config()
    cfg10 = s10.load_config(s11.resolve(cfg["step10_config"]))
    cfg9 = s9.load_config(s11.resolve(cfg10["prior_config"]))
    prior, _ = s9.load_prior(cfg9, device, cfg10["prior_checkpoint"])
    for p in prior.parameters():
        p.requires_grad_(False)
    test = s2.TarFlowInput(s9.build_bundle(cfg9).test)
    T, D = s10.latent_shape(prior)
    R = args.restarts
    gen = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(test), generator=gen)[:len(MODELS)].tolist()   # a different test image per row
    print(f"[gallery] test images {indices} of {len(test)}", flush=True)

    torch.cuda.synchronize(); t0 = time.time()
    fr = invlib.FastReverse(prior, R, device, mode="compile")
    torch.cuda.synchronize(); print(f"[gallery] graph build {time.time() - t0:.1f} s", flush=True)

    rows, store = [], {}
    for r, ((name, kw, label), idx) in enumerate(zip(MODELS, indices)):
        x_true, lab = test[idx]
        x_true = x_true.unsqueeze(0).to(device)
        prob = make_problem(name, kw, x_true, gen)
        torch.manual_seed(args.seed + 100 * r)
        z = torch.randn(R, T, D, device=device).requires_grad_(True)       # independent prior draws as inits
        opt = torch.optim.Adam([z], lr=args.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.iters, eta_min=args.lr_min)
        J_tr = np.zeros((args.iters, R)); ts = []
        torch.cuda.synchronize(); t_all = time.time()
        for k in range(args.iters):
            t0 = time.time()
            x, _ = fr(z)
            J = prob["nll"](x) + s11.neg_log_normal(z)
            opt.zero_grad(set_to_none=True)
            J.sum().backward()
            opt.step(); sched.step()
            torch.cuda.synchronize(); ts.append(time.time() - t0); J_tr[k] = J.detach().cpu().numpy()
            if k % 50 == 0 or k == args.iters - 1:
                print(f"[gallery] {name:15s} step {k:3d}: J " + " ".join(f"{float(j):9.1f}" for j in J) + f"  lr {sched.get_last_lr()[0]:.4f}  {ts[-1]:.2f} s", flush=True)
        total = time.time() - t_all
        with torch.no_grad():
            x, _ = fr(z)
            data = prob["nll"](x); J = data + s11.neg_log_normal(z)
            data_true = prob["nll"](x_true)
            zt, _, _ = prior(x_true)
            J_true = float(data_true[0] + s11.neg_log_normal(zt)[0])
            ps = psnr01(x, x_true.expand_as(x))
        row = dict(model=name, kwargs=kw, label=label, index=int(idx), chest_label=int(np.asarray(lab).reshape(-1)[0]),
                   M=prob["M"], sigma=prob["sigma"], noise_level=float(prob["noise_level"]), data_true=float(data_true[0]), J_true=J_true,
                   J=[float(v) for v in J], data=[float(v) for v in data], psnr=[float(v) for v in ps],
                   z_norm=[float(v) for v in z.detach().flatten(1).norm(dim=-1)], max_abs=[float(v) for v in x.flatten(1).abs().max(-1).values],
                   pairwise_rms=float(((x[:, None] - x[None]) ** 2).mean((2, 3, 4)).sqrt().sum() / (R * (R - 1))),
                   seconds=total, step_seconds=float(np.mean(ts)), J_min_over_steps=[float(v) for v in J_tr.min(0)])
        rows.append(row)
        store[name] = dict(x_true=x_true.cpu(), y=prob["y"].cpu(), x=x.cpu(), z=z.detach().cpu(), J_trace=J_tr)
        ops = globals().setdefault("_OPS", {}); ops[name] = prob["op"]
        print(f"[gallery] {name:15s} #{idx}: {total:.0f} s; J {['%.0f' % v for v in row['J']]} (truth {J_true:.0f}); "
              f"data {['%.0f' % v for v in row['data']]} (truth {row['data_true']:.0f}, noise level {row['noise_level']:.0f}); "
              f"PSNR {['%.1f' % v for v in row['psnr']]}; |z| {['%.1f' % v for v in row['z_norm']]}; pairwise RMS {row['pairwise_rms']:.3f}", flush=True)
        json.dump(dict(args=vars(args), indices=indices, rows=rows), open(OUT_DATA / "gallery.json", "w"), indent=1)
        torch.save(store, OUT_DATA / "gallery_solutions.pt")

    # ---- the gallery
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nrow, ncol = len(MODELS), 2 + R
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.9 * ncol + 1.2, 1.95 * nrow))
    for r, ((name, kw, label), row) in enumerate(zip(MODELS, rows)):
        st = store[name]
        s10._img(axes[r, 0], st["x_true"][0, 0], f"truth, test #{row['index']}")
        show_measurement(axes[r, 1], dict(op=_OPS[name], y=st["y"], M=row["M"]), st["x_true"].shape)
        for j in range(R):
            s10._img(axes[r, 2 + j], st["x"][j, 0], f"restart {j + 1}: {row['psnr'][j]:.1f} dB")
        axes[r, 0].text(-0.12, 0.5, label, transform=axes[r, 0].transAxes, rotation=90, va="center", ha="center", fontsize=7)
    fig.suptitle(f"latent MAP restarts (Adam, {args.iters} steps, lr {args.lr}$\\to${args.lr_min} cosine, independent random inits) "
                 f"under the step-9 ChestMNIST-64 prior, one test image per system", fontsize=8)
    fig.tight_layout(rect=(0.02, 0, 1, 0.97)); fig.savefig(OUT_FIGS / "gallery.png", dpi=150); plt.close(fig)
    print("gallery done", flush=True)


if __name__ == "__main__":
    main()
