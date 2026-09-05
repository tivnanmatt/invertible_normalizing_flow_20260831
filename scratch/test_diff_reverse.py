"""Check invlib.differentiable_reverse against the official reverse and the
official forward log-density, on a random (perturbed) model, bounded and not."""
import sys, math, time, torch
sys.path.insert(0, "/dev_ws/invertible_normalizing_flow_20260831")
sys.path.insert(0, "/dev_ws/ml-tarflow")
import transformer_flow as tf
import invlib

torch.manual_seed(0)
dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else "cpu")


def make(bound, patch=4, ch=128, blocks=4, layers=2, img=32, dtype=torch.float32):
    m = tf.Model(1, img, patch, ch, blocks, layers)
    for blk in m.blocks:  # non-trivial coupling: random proj_out
        torch.nn.init.normal_(blk.proj_out.weight, std=0.02)
        torch.nn.init.normal_(blk.proj_out.bias, std=0.1)
    if bound:
        invlib.bound_log_scale(m, bound)
    return m.to(dev, dtype)


def grads(m, z, checkpoint):
    m.zero_grad(set_to_none=True)
    z = z.clone().requires_grad_(True)
    x, lq = invlib.differentiable_reverse(m, z, checkpoint=checkpoint)
    ((x ** 2).sum() + lq.sum()).backward()
    return [z.grad.clone()] + [p.grad.clone() for p in m.parameters() if p.grad is not None]


for bound in [0, 8.0]:
    m = make(bound)
    B = 5
    z = torch.randn(B, m.num_patches, m.patch_size ** 2, device=dev)
    with torch.no_grad():
        x_off = m.reverse(z.clone())
    x, log_q = invlib.differentiable_reverse(m, z, checkpoint=False)
    print(f"bound={bound}: max|x - official| = {(x - x_off).abs().max():.2e}")
    zf, _, ld = m(x)
    n = x[0].numel()
    log_p = -0.5 * (zf ** 2 + math.log(2 * math.pi)).flatten(1).sum(-1) + ld * n
    print(f"   max|z_fwd - z| = {(zf - z).abs().max():.2e};  max|log_q - log_p_fwd| = {(log_q - log_p).abs().max():.2e}  (|log_p| ~ {log_p.abs().mean():.1f})")
    g1, g2 = grads(m, z, False), grads(m, z, True)
    diffs = [((a - b).abs().max()).item() for a, b in zip(g1, g2)]
    finite = all(torch.isfinite(a).all() for a in g1)
    print(f"   grad(checkpoint) vs grad(plain): max diff {max(diffs):.2e} over {len(diffs)} tensors; finite={finite}")
    flags = [a.attention.sample for blk in m.blocks for a in blk.attn_blocks]
    print(f"   sample mode left on: {any(flags)}")

# finite-difference check of d(pixel)/dz and d(log q)/dz on a tiny model (fp32,
# the official LayerNorm casts to float32 so fp64 is not available)
m = make(8.0, patch=8, ch=64, blocks=2, layers=1, img=16)
z = torch.randn(2, m.num_patches, m.patch_size ** 2, device=dev)
worst = 0
for name, sel in [("x[0,0,5,9]", lambda x, lq: x[0, 0, 5, 9]), ("log_q[1]/100", lambda x, lq: lq[1] / 100)]:
    m.zero_grad(set_to_none=True)
    zz = z.clone().requires_grad_(True)
    x, lq = invlib.differentiable_reverse(m, zz, checkpoint=True)
    sel(x, lq).backward()
    g = zz.grad
    eps = 1e-2
    for (b, t, d) in [(0, 0, 0), (0, 1, 5), (1, 2, 60), (1, 3, 1)]:
        zp = z.clone(); zp[b, t, d] += eps
        zm = z.clone(); zm[b, t, d] -= eps
        with torch.no_grad():
            fd = (sel(*invlib.differentiable_reverse(m, zp, checkpoint=False))
                  - sel(*invlib.differentiable_reverse(m, zm, checkpoint=False))) / (2 * eps)
        worst = max(worst, abs(float(g[b, t, d] - fd)))
        print(f"   d {name} / d z[{b},{t},{d}]: autograd {g[b,t,d]:.5f} vs finite diff {fd:.5f}")
print(f"   worst |autograd - fd| = {worst:.2e}")

# timing / memory for the step-9 sized model on GPU
if dev.type == "cuda":
    m = make(8.0, patch=8, ch=768, blocks=8, layers=8, img=64)
    for B in [8, 16, 32]:
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t = time.time()
        z = torch.randn(B, m.num_patches, m.patch_size ** 2, device=dev)
        x, lq = invlib.differentiable_reverse(m, z, checkpoint=True)
        ((x ** 2).sum() + lq.sum()).backward()
        torch.cuda.synchronize()
        print(f"768-8-8 p8 B={B}: fwd+bwd {time.time()-t:.1f}s, peak {torch.cuda.max_memory_allocated()/2**30:.1f} GB")
        m.zero_grad(set_to_none=True)
