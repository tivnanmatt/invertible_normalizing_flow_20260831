"""step11_lr_sweep.py -- Learning-rate sweep for the latent MAP of step 11 (Adam on z,
one param group per learning rate, all runs in one batch from the same random init,
through the CUDA-graph reverse invlib.FastReverse), N iterations each, on the step-10/11
measurement. Reports per learning rate: J_z per step, non-finite events, monotonicity,
final reconstruction quality; a figure with the curves and the reconstructions.
Written as a scratch script after step 11's 1000-step runs at lr 0.005 had not converged;
its answer (lr 0.1 best; no non-finite event up to lr 1; failure at lr >= 0.5 is latent
drift, not overflow) sets the step-14 schedule.

Outputs: outputs/step11_map_adam/data/lr_sweep{.json,_trace.npz,_solutions.pt},
         outputs/step11_map_adam/figures/lr_sweep.png
Usage (container): python step11_lr_sweep.py --device cuda:0 --iters 100
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import invlib
import step9_chestmnist_tarflow as s9
import step10_variational_posterior as s10
import step11_map_adam as s11

OUT_DATA = REPO / "outputs" / "step11_map_adam" / "data"
OUT_FIGS = REPO / "outputs" / "step11_map_adam" / "figures"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--lrs", type=str, default="5e-3,1e-2,2e-2,5e-2,1e-1,2e-1,5e-1,1.0")
    ap.add_argument("--seed", type=int, default=20260911)
    ap.add_argument("--tag", type=str, default="lr_sweep")
    args = ap.parse_args()
    lrs = [float(v) for v in args.lrs.split(",")]
    R = len(lrs)
    OUT_DATA.mkdir(parents=True, exist_ok=True); OUT_FIGS.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    cfg = s11.load_config()
    cfg10 = s10.load_config(s11.resolve(cfg["step10_config"]))
    cfg9 = s9.load_config(s11.resolve(cfg10["prior_config"]))
    prior, _ = s9.load_prior(cfg9, device, cfg10["prior_checkpoint"])
    for p in prior.parameters():
        p.requires_grad_(False)
    prob = s10.build_problem(cfg10, cfg9, device)
    T, D = s10.latent_shape(prior)

    torch.cuda.synchronize(); t0 = time.time()
    fr = invlib.FastReverse(prior, R, device, mode="compile")
    torch.cuda.synchronize(); build = time.time() - t0

    torch.manual_seed(args.seed + 1)
    z0 = torch.randn(1, T, D, device=device)                       # the same init as the 100-step test
    vs = [z0.clone().requires_grad_(True) for _ in lrs]
    opt = torch.optim.Adam([dict(params=[v], lr=lr) for v, lr in zip(vs, lrs)], betas=(0.9, 0.999))
    K = args.iters
    J_tr, data_tr, gn_tr, zn_tr, xmax_tr = (np.full((K, R), np.nan) for _ in range(5))
    n_nonfinite = np.zeros(R, dtype=int)
    ts = []
    torch.cuda.synchronize(); t_all = time.time()
    for k in range(K):
        t0 = time.time()
        z = torch.cat(vs, 0)
        x, _ = fr(z)
        data = s10.neg_log_lik(x, prob)
        J = data + s11.neg_log_normal(z)
        opt.zero_grad(set_to_none=True)
        J.sum().backward()
        for r, v in enumerate(vs):
            if not torch.isfinite(v.grad).all():
                n_nonfinite[r] += 1
                v.grad.zero_()
        gn = torch.stack([v.grad.flatten().norm() for v in vs])
        opt.step()
        torch.cuda.synchronize(); ts.append(time.time() - t0)
        J_tr[k], data_tr[k], gn_tr[k] = J.detach().cpu().numpy(), data.detach().cpu().numpy(), gn.cpu().numpy()
        zn_tr[k] = z.detach().flatten(1).norm(dim=-1).cpu().numpy()
        xmax_tr[k] = x.detach().flatten(1).abs().max(-1).values.cpu().numpy()
        if k % 10 == 0 or k == K - 1:
            print(f"[lr] step {k}: J " + "  ".join(f"{lr:g}:{float(j):.0f}" for lr, j in zip(lrs, J)) + f"  {ts[-1]:.2f}s", flush=True)
    torch.cuda.synchronize(); total = time.time() - t_all
    with torch.no_grad():
        z = torch.cat(vs, 0)
        x, _ = fr(z)
        a = s11.assess(prior, x, prob, Z=z)
        a_true = s11.assess(prior, prob["x"], prob)
    rows = []
    for r, lr in enumerate(lrs):
        Jr = J_tr[:, r]
        finite = np.isfinite(Jr)
        rows.append(dict(lr=lr, J_final=float(a["J_z"][r]), J_min=float(np.nanmin(Jr)), step_of_min=int(np.nanargmin(Jr)),
                         J_at_10=float(Jr[min(9, K - 1)]), J_at_50=float(Jr[min(49, K - 1)]),
                         n_increases=int(np.sum(np.diff(Jr[finite]) > 0)), n_nonfinite_grad=int(n_nonfinite[r]),
                         n_nonfinite_J=int((~finite).sum()), data_final=float(a["data"][r]), z_norm=float(a["z_norm"][r]),
                         max_abs_pixel=float(a["max_abs"][r]), psnr=float(a["psnr"][r]), psnr_hole=float(a["psnr_hole"][r]),
                         bias_hole=float(a["bias_hole"][r]), grad_norm_final=float(gn_tr[-1, r])))
    res = dict(iters=K, lrs=lrs, build_seconds=build, total_seconds=total, step_seconds=float(np.mean(ts)),
               truth=dict(J_z=float(a_true["J_z"][0]), data=float(a_true["data"][0])),
               noise_level=float(0.5 * prob["n_obs"] * np.log(2 * np.pi * prob["sigma"] ** 2) + 0.5 * prob["n_obs"]), table=rows)
    json.dump(res, open(OUT_DATA / f"{args.tag}.json", "w"), indent=1)
    np.savez(OUT_DATA / f"{args.tag}_trace.npz", J=J_tr, data=data_tr, grad_norm=gn_tr, z_norm=zn_tr, xmax=xmax_tr, lrs=np.array(lrs))
    torch.save(dict(lrs=lrs, z=z.cpu(), x=x.cpu()), OUT_DATA / f"{args.tag}_solutions.pt")
    print(f"[lr] {K} iterations x {R} learning rates in {total:.1f} s ({res['step_seconds']:.3f} s/step at batch {R})", flush=True)
    for row in rows:
        print(f"[lr] lr {row['lr']:<6g} J_final {row['J_final']:>10.1f}  min {row['J_min']:>10.1f} @ {row['step_of_min']:>3d}  "
              f"increases {row['n_increases']:>3d}  nonfinite {row['n_nonfinite_J']}/{row['n_nonfinite_grad']}  data {row['data_final']:>9.1f}  "
              f"|z| {row['z_norm']:5.1f}  max|x| {row['max_abs_pixel']:.2f}  PSNR {row['psnr']:.1f} (hole {row['psnr_hole']:.1f})", flush=True)

    # ---- figure
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x_true = prob["x"][0, 0].cpu()
    fig = plt.figure(figsize=(2.3 * (R + 2), 7.6))
    gs = fig.add_gridspec(2, R + 2, height_ratios=[1.35, 1])
    ax = fig.add_subplot(gs[0, :(R + 2) // 2])
    for r, lr in enumerate(lrs):
        ax.plot(np.arange(K), J_tr[:, r], lw=1.3, label=f"lr {lr:g}")
    ax.axhline(res["truth"]["J_z"], color="g", ls="--", label="J_z at truth")
    ax.set_yscale("symlog", linthresh=1000); ax.set_xlabel("Adam step"); ax.set_ylabel("J_z (nats)"); ax.legend(fontsize=8, ncol=2)
    ax.set_title(f"latent MAP objective, {K} steps, same random init, batch {R}: {total:.0f} s")
    ax2 = fig.add_subplot(gs[0, (R + 2) // 2:])
    for r, lr in enumerate(lrs):
        ax2.plot(np.arange(K), data_tr[:, r], lw=1.3, label=f"lr {lr:g}")
    ax2.axhline(res["noise_level"], color="k", ls="-.", lw=0.8, label="noise level")
    ax2.set_yscale("symlog", linthresh=1000); ax2.set_xlabel("Adam step"); ax2.set_ylabel("-log p(y|x) (nats)"); ax2.legend(fontsize=8, ncol=2)
    ax2.set_title("data term")
    s11._img(fig.add_subplot(gs[1, 0]), x_true, "truth")
    axm = fig.add_subplot(gs[1, 1]); axm.imshow(s10._measurement_rgb(prob)); axm.set_title("measurement"); axm.axis("off")
    for r, lr in enumerate(lrs):
        s11._img(fig.add_subplot(gs[1, r + 2]), x[r, 0].cpu(),
                 f"lr {lr:g}: J {rows[r]['J_final']:.0f}\nPSNR {rows[r]['psnr']:.1f} (hole {rows[r]['psnr_hole']:.1f})")
    fig.tight_layout(); fig.savefig(OUT_FIGS / f"{args.tag}.png", dpi=120); plt.close(fig)
    print("lr sweep done", flush=True)


if __name__ == "__main__":
    main()
