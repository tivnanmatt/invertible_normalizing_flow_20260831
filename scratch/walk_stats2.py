import numpy as np
for res, kb in ((32, 24), (64, 24), (128, 20)):
    d = np.load(f"outputs/step16_latent_langevin/data/langevin_{res}_walk_frames.npz")
    F = d["frames"][kb:]
    print(f"--- {res} px: sampling frames {F.shape[0]}")
    for s, n in enumerate(["denoise", "sr_4x", "ct"]):
        X = F[:, s].reshape(F.shape[0], -1).astype(np.float64)
        Xc = X - X.mean(0); var = (Xc ** 2).mean(); std = np.sqrt(var)
        lag1 = (Xc[1:] * Xc[:-1]).mean() / var
        rmsd = np.sqrt(((X[1:] - X[:-1]) ** 2).mean(1))
        distinct = 1 + int((rmsd > 1e-6).sum())
        acc = d["trace_accept"][kb:, s]
        # mean |z| distance between consecutive samples not available; pixel-domain
        print(f"{n:8s} distinct {distinct:2d}/{F.shape[0]}  accepted {int(acc.sum())}  lag1 rho {lag1:.2f}  rms consec diff {rmsd.mean():.4f} vs sqrt2*std {np.sqrt(2)*std:.4f}  pixel std {std:.4f}  U {d['trace_U'][kb:, s].min():.0f}..{d['trace_U'][kb:, s].max():.0f}  |z| {d['trace_z_norm'][kb:, s].min():.1f}..{d['trace_z_norm'][kb:, s].max():.1f}  psnr {d['trace_psnr'][kb:, s].min():.1f}..{d['trace_psnr'][kb:, s].max():.1f}  median acc prob {np.median(d['trace_accept_prob'][kb:, s]):.2f}")
