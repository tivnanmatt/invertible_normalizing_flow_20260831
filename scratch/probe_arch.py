"""Probe: peak memory and time per training step (TF32, AdamW) for candidate TarFlow
architectures on the resolution ladder, with and without per-block activation
checkpointing. Run inside recon-dev on GPU 0 when nothing else is running."""
import sys, time, torch
sys.path.insert(0, "/dev_ws/invertible_normalizing_flow_20260831")
import step13_lidc_tarflow as s13

dev = torch.device("cuda:0")
s13.set_tf32(True)

CANDS = [
    # name, img, patch, channels, blocks, layers, batch, ckpt
    ("64 p4 512/8/6 b64", 64, 4, 512, 8, 6, 64, False),
    ("64 p4 512/8/6 b64 ckpt", 64, 4, 512, 8, 6, 64, True),
    ("128 p8 768/8/8 b32 ckpt", 128, 8, 768, 8, 8, 32, True),
    ("128 p8 512/8/6 b32 ckpt", 128, 8, 512, 8, 6, 32, True),
    ("128 p4 512/8/6 b16 ckpt", 128, 4, 512, 8, 6, 16, True),
    ("256 p16 768/8/8 b32 ckpt", 256, 16, 768, 8, 8, 32, True),
    ("256 p8 512/8/6 b16 ckpt", 256, 8, 512, 8, 6, 16, True),
    ("256 p8 768/8/8 b8 ckpt", 256, 8, 768, 8, 8, 8, True),
]

for name, img, p, ch, nb, nl, b, ck in CANDS:
    cfg = dict(tarflow_repo="/dev_ws/ml-tarflow", channel_size=1, img_size=img, patch_size=p,
               model=dict(channels=ch, blocks=nb, layers_per_block=nl), scale_bound=8.0)
    try:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        model = s13.build_model(cfg, dev)
        n = sum(q.numel() for q in model.parameters()) / 1e6
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        x = torch.rand(b, 1, img, img, device=dev) * 2 - 1
        ts = []
        for it in range(4):
            torch.cuda.synchronize(); t0 = time.time()
            opt.zero_grad(set_to_none=True)
            if ck:
                z, ld = s13.forward_checkpointed(model, x)
            else:
                z, _, ld = model(x)
            loss = model.get_loss(z, ld)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            torch.cuda.synchronize(); ts.append(time.time() - t0)
        dt = min(ts[1:])
        T = (img // p) ** 2
        print(f"{name:28s} {n:6.1f}M T={T:5d} D={p*p:4d}  {dt:.3f} s/step  {1000*dt/b:6.2f} ms/img  "
              f"48k imgs: {48000*dt/b/60:5.1f} min  peak {torch.cuda.max_memory_allocated()/2**30:5.1f} GB", flush=True)
    except torch.cuda.OutOfMemoryError:
        print(f"{name:28s} OOM", flush=True)
    del model, opt, x
    torch.cuda.empty_cache()
