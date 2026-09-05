"""Why does the 64-px walk accept nothing? One leapfrog trajectory (L steps) from the
latent MAP of each of denoise / sr_4x / ct at several step sizes, with the same momentum,
printing Delta H per row; FastReverse2 and (optionally) FastReverse v1 on the same
trajectory. Integration error scales as eps^2 (Delta H falls 4x per halving of eps);
a wrong gradient does not (Delta H ~ eps * L)."""
import json, math, sys, time
import numpy as np
import torch
sys.path.insert(0, "/dev_ws/invertible_normalizing_flow_20260831")
import invlib, step10_variational_posterior as s10, step13_lidc_tarflow as s13, step14_lidc_gallery as s14
import step16_latent_langevin as s16

device = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
res = int(sys.argv[2]) if len(sys.argv) > 2 else 64
which = sys.argv[3] if len(sys.argv) > 3 else "fast2"          # fast2 | fast
L = int(sys.argv[4]) if len(sys.argv) > 4 else 8
eps_list = [float(v) for v in (sys.argv[5].split(",") if len(sys.argv) > 5 else ["2e-3", "1e-3", "5e-4", "2.5e-4"])]

cfg = s16.load_config("/dev_ws/invertible_normalizing_flow_20260831/configs/step16_latent_langevin_walk.yml")
s13.set_tf32(False)
cfg14 = s14.load_config(s16.resolve(cfg["step14_config"]))
out14 = s16.REPO_ROOT / cfg14["output_root"] / "data"
gal = json.load(open(out14 / f"gallery_{res}.json"))
sol = torch.load(out14 / f"gallery_{res}_solutions.pt", map_location="cpu", weights_only=False)
cfg13 = s13.load_config(s16.resolve(cfg14["priors"][res]))
prior, ckpt = s13.load_prior(cfg13, device, cfg14["prior_checkpoint"])
for p in prior.parameters():
    p.requires_grad_(False)
systems, keep_idx = s16.pick_systems(cfg, cfg14)
gal_rows = [gal["rows"][i] for i in keep_idx]
gen = torch.Generator().manual_seed(cfg14["seed"])
probs, z_map = [], []
for spec, row in zip(systems, gal_rows):
    st = sol[spec["name"]]
    probs.append(s14.make_problem(spec, st["x_true"].to(device), gen, y=st["y"]))
    z_map.append(st["z"][int(np.argmin(row["J"]))].to(device))
B = len(systems)
z0 = torch.stack(z_map)
print(f"{res} px, {which}, L={L}; |z_map| {[round(float(v), 1) for v in z0.flatten(1).norm(dim=-1)]}", flush=True)

t0 = time.time()
if which == "fast2":
    fr = invlib.FastReverse2(prior, B, device, chunk=128, mode="graph")
else:
    fr = invlib.FastReverse(prior, B, device, mode="compile")
torch.cuda.synchronize()
print(f"build {time.time() - t0:.0f} s", flush=True)

def energy(z):
    z = z.detach().requires_grad_(True)
    x, _ = fr(z)
    d = torch.cat([p["nll"](x[s:s + 1]) for s, p in enumerate(probs)])
    U = d + 0.5 * (z ** 2).flatten(1).sum(-1)
    (g,) = torch.autograd.grad(U.sum(), z)
    return U.detach(), g, d.detach()

U0, g0, d0 = energy(z0)
print(f"U0 {[round(float(v)) for v in U0]}  data {[round(float(v)) for v in d0]}  |g0| {[round(float(v), 1) for v in g0.flatten(1).norm(dim=-1)]}  max|g0| {[round(float(v), 2) for v in g0.flatten(1).abs().max(-1).values]}", flush=True)
# a second call at the same point: FR2 reuses its caches, so this checks reproducibility
U0b, g0b, _ = energy(z0)
print(f"repeat: |dU| {float((U0b - U0).abs().max()):.2e}  |dg|/|g| {float((g0b - g0).norm() / g0.norm()):.2e}", flush=True)
gen_p = torch.Generator(device=device).manual_seed(1)
p0 = torch.randn(z0.shape, generator=gen_p, device=device)
for eps in eps_list:
    t1 = time.time()
    e3 = torch.full((B, 1, 1), eps, device=device)
    z, p, g = z0, p0.clone(), g0
    H0 = U0 + 0.5 * (p ** 2).flatten(1).sum(-1)
    p = p - 0.5 * e3 * g
    Us = []
    for l in range(L):
        z = z + e3 * p
        U, g, d = energy(z)
        Us.append([round(float(v), 1) for v in U - U0])
        p = p - (e3 if l < L - 1 else 0.5 * e3) * g
    H1 = U + 0.5 * (p ** 2).flatten(1).sum(-1)
    dH = H1 - H0
    print(f"eps {eps:.1e}: dH {[round(float(v), 2) for v in dH]}  (U-U0 along the path: {Us[0]} ... {Us[-1]}; "
          f"|z| {[round(float(v), 2) for v in z.flatten(1).norm(dim=-1)]})  {(time.time() - t1) / L:.2f} s/grad", flush=True)
