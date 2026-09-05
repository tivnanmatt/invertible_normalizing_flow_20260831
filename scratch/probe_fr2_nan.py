"""Does a NaN pass leave FastReverse2 corrupt? Gradient at z0 clean, after a
non-finite trajectory, after zeroing the persistent buffers, after a rebuild."""
import json, sys, time
import numpy as np
import torch
sys.path.insert(0, "/dev_ws/invertible_normalizing_flow_20260831")
import invlib, step13_lidc_tarflow as s13, step14_lidc_gallery as s14
import step16_latent_langevin as s16

device, res = sys.argv[1], int(sys.argv[2])
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
gen = torch.Generator().manual_seed(cfg14["seed"])
probs, z_map = [], []
for spec, row in zip(systems, [gal["rows"][i] for i in keep_idx]):
    st = sol[spec["name"]]
    probs.append(s14.make_problem(spec, st["x_true"].to(device), gen, y=st["y"]))
    z_map.append(st["z"][int(np.argmin(row["J"]))].to(device))
B = len(systems)
z0 = torch.stack(z_map)
build = lambda: invlib.FastReverse2(prior, B, device, chunk=128, mode="graph")
t0 = time.time(); fr = build(); torch.cuda.synchronize(); print(f"build {time.time() - t0:.1f} s", flush=True)

def energy(fr, z, backward=True):
    z = z.detach().requires_grad_(True)
    x, _ = fr(z)
    d = torch.cat([p["nll"](x[s:s + 1]) for s, p in enumerate(probs)])
    U = d + 0.5 * (z ** 2).flatten(1).sum(-1)
    if not backward:
        return U.detach(), None
    (g,) = torch.autograd.grad(U.sum(), z)
    return U.detach(), g

def report(tag, fr):
    U, g = energy(fr, z0)
    print(f"{tag}: U {[round(float(v), 3) for v in U]}  |g-g0|/|g0| {['%.2e' % float(v) for v in (g - g0).flatten(1).norm(dim=-1) / g0.flatten(1).norm(dim=-1)]}  finite {bool(torch.isfinite(g).all())}", flush=True)

U0, g0 = energy(fr, z0)
report("clean repeat", fr)
# a huge but finite pass, then a non-finite pass
gen_p = torch.Generator(device=device).manual_seed(1)
big = z0 + 3.0 * torch.randn(z0.shape, generator=gen_p, device=device)
U, g = energy(fr, big); print(f"big pass: U {['%.3g' % float(v) for v in U]} finite g {bool(torch.isfinite(g).all())}", flush=True)
report("after big finite pass", fr)
huge = z0 * 1e4
U, g = energy(fr, huge); print(f"huge pass: U {['%.3g' % float(v) for v in U]} finite g {bool(torch.isfinite(g).all())}", flush=True)
report("after non-finite pass", fr)
nanz = z0.clone(); nanz[:, -1, 0] = float("nan")
U, g = energy(fr, nanz); print(f"nan pass: U {['%.3g' % float(v) for v in U]} finite g {bool(torch.isfinite(g).all())}", flush=True)
report("after nan pass", fr)
# forward-only non-finite pass (no backward), then a clean pass
fr2 = build()
report("fresh build", fr2)
U, _ = energy(fr2, nanz, backward=False); print(f"nan forward only: U {['%.3g' % float(v) for v in U]}", flush=True)
report("after nan forward-only pass", fr2)
# zero the persistent buffers of the corrupt one
sh = fr.shared
for t in (sh.s_tok, sh.s_za, sh.s_zb, sh.s_gza, sh.s_gzb, sh.s_gtok, sh.gK, sh.gV, sh.rec.P, sh.rec.GS, sh.rec.Q, sh.rec.GO):
    t.zero_()
for gb in fr.blocks:
    gb.K.zero_(); gb.V.zero_()
report("after zeroing buffers", fr)
fr3 = build()
report("rebuilt", fr3)
