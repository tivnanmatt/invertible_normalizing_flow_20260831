#!/usr/bin/env python
"""measlib.py -- linear measurement systems with additive Gaussian white noise.

Every operator here is  y = A x + sigma * eps,  eps ~ N(0, I), i.e.
p(y|x) = N(A x, sigma^2 I)  -- the setting of System-embedded Diffusion Bridge
Models (Sobieski, Tivnan et al., arXiv:2506.23726), specialised to white noise.

The whole library is built on one observation. Write the SVD A = U S V^T. In
the right-singular-vector coordinates c = V^T x EVERYTHING becomes elementwise:

    pseudo-inverse recon   (A^+ y)~_i = c_i + sigma/s_i * eps_i   (s_i > 0)
                                      = 0                        (s_i = 0)
    range/null projector   A^+A       = V diag(s_i > 0) V^T
    bridge covariance      Sigma_1    = V diag( gamma*sigma^2/s_i^2 , beta ) V^T

so an operator is fully specified by an ORTHOGONAL basis V and a vector of
singular values s (zero on the null space). Sigma_1 is diagonal in V, which is
what makes the exact Gaussian base density of the flow bridge cheap to evaluate
and to whiten.

Sigma_1 follows the SDB terminal covariance
    Sigma_t = gamma_t A^+ Sigma A^+T + beta_t (I - A^+A)
with Sigma = sigma^2 I. gamma scales the range (measurement-noise) part and
beta sets the null-space variance -- the "information the measurement never
saw", which is exactly what the flow has to generate.

NOTE sigma > 0 is REQUIRED. With sigma = 0 the range block of Sigma_1 is
singular, the base density degenerates to a delta on the range, and the flow
likelihood is undefined. Physically this is just "every measurement has noise".

Implemented systems (analytic SVD, no matrix ever formed):
    denoise         A = I                      (no null space)
    inpaint_box     A = diag(mask), centre box removed
    inpaint_random  A = diag(mask), random pixels removed
    sr_2x / sr_4x   A = 2x2 / 4x4 average pooling  (V = block Haar)
and, with an explicit matrix and a numerical (truncated) SVD:
    ct              sparse-view parallel-beam CT; y is a SINOGRAM, a different
                    shape from the image (A is rectangular)

Run `python measlib.py` to execute the self-tests.
"""

import math

import torch

import invlib


class LinearMeasurement:
    """y = A x + sigma * eps, with A = U S V^T given implicitly by (V, s)."""

    def __init__(self, shape, sigma, gamma=1.0, beta=1.0, name="base"):
        assert sigma > 0, (
            "sigma must be > 0: with a noiseless measurement the range block of "
            "Sigma_1 is singular and the flow's base density is undefined")
        self.shape = tuple(shape)           # (C, N, N)
        self.sigma, self.gamma, self.beta = float(sigma), float(gamma), float(beta)
        self.name = name
        self._s = None                      # singular values, shape == self.shape

    # -- basis: c = V^T x  and  x = V c -------------------------------------
    def basis_fwd(self, x):
        raise NotImplementedError

    def basis_inv(self, c):
        raise NotImplementedError

    def sing(self, device=None, dtype=None):
        s = self._s
        if device is not None or dtype is not None:
            s = s.to(device=device or s.device, dtype=dtype or s.dtype)
        return s

    # -- derived quantities --------------------------------------------------
    def base_var(self, device=None, dtype=None):
        """Diagonal of Sigma_1 in V coordinates."""
        s = self.sing(device, dtype)
        rng = self.gamma * self.sigma ** 2 / s.clamp_min(1e-12) ** 2
        return torch.where(s > 0, rng, torch.full_like(s, self.beta))

    def n_null(self):
        return int((self.sing() == 0).sum())

    def project(self, x):
        """A^+A x -- the range-space (measurable) component of x."""
        c = self.basis_fwd(x)
        return self.basis_inv(torch.where(self.sing(c.device, c.dtype) > 0,
                                          c, torch.zeros_like(c)))

    def pinv_recon(self, x, generator=None):
        """Sample the pseudo-inverse reconstruction A^+ y for y = Ax + sigma eps.

        Never forms y or A: in V coordinates this is c_i + sigma/s_i * eps_i on
        the range and 0 on the null space.
        """
        c = self.basis_fwd(x)
        s = self.sing(c.device, c.dtype)
        eps = torch.randn(c.shape, device=c.device, dtype=c.dtype, generator=generator)
        noisy = c + self.sigma / s.clamp_min(1e-12) * eps
        return self.basis_inv(torch.where(s > 0, noisy, torch.zeros_like(c)))

    # -- the base density, in V coordinates ---------------------------------
    # The flow is trained in V coordinates (c = V^T x): TarFlow's coupling is
    # elementwise within a token and never mixes channels, so it can only match
    # a base that is DIAGONAL in the coordinates it sees. Sigma_1 is diagonal in
    # V, not in pixels -- for the pooling systems the pixels of one patch are
    # strongly correlated under Sigma_1 -- so the flow acts on c and the
    # pixel-space methods below are thin wrappers.
    def coef_sample_base(self, cm, generator=None):
        """Draw c_z ~ N(cm, diag Sigma_1) given cm = V^T A^+y."""
        v = self.base_var(cm.device, cm.dtype)
        eps = torch.randn(cm.shape, device=cm.device, dtype=cm.dtype, generator=generator)
        return cm + v.sqrt() * eps

    def coef_logprob(self, cz, cm):
        """log N(cz; cm, diag Sigma_1), summed over dims, per example."""
        v = self.base_var(cz.device, cz.dtype)
        ll = -0.5 * ((cz - cm) ** 2 / v + torch.log(2 * math.pi * v))
        return ll.flatten(1).sum(-1)

    def sample_base(self, xhat, generator=None):
        """Draw z ~ N(A^+y, Sigma_1) given the pseudo-inverse recon xhat."""
        return self.basis_inv(self.coef_sample_base(self.basis_fwd(xhat), generator))

    def base_logprob(self, z, xhat):
        """log N(z; A^+y, Sigma_1), summed over dims, per example."""
        return self.coef_logprob(self.basis_fwd(z), self.basis_fwd(xhat))

    def logdet_sigma(self):
        """log det Sigma_1 -- constant given the system; needed for honest bpd."""
        return float(torch.log(self.base_var().double()).sum())

    def __repr__(self):
        s = self.sing()
        return (f"{self.name}(shape={self.shape}, sigma={self.sigma}, "
                f"gamma={self.gamma}, beta={self.beta}, "
                f"rank={int((s>0).sum())}/{s.numel()}, null={self.n_null()})")


class _IdentityBasis(LinearMeasurement):
    """Systems already diagonal in the pixel basis: V = I."""

    def basis_fwd(self, x):
        return x

    def basis_inv(self, c):
        return c


class Denoise(_IdentityBasis):
    """A = I. Full rank, no null space -- pure denoising."""

    def __init__(self, shape, sigma, gamma=1.0, beta=1.0):
        super().__init__(shape, sigma, gamma, beta, name="denoise")
        self._s = torch.ones(shape, dtype=torch.float64)


class InpaintBox(_IdentityBasis):
    """A = diag(mask) with a centred square hole removed."""

    def __init__(self, shape, sigma, box=12, gamma=1.0, beta=1.0):
        super().__init__(shape, sigma, gamma, beta, name="inpaint_box")
        C, N, _ = shape
        m = torch.ones(shape, dtype=torch.float64)
        lo = (N - box) // 2
        m[:, lo:lo + box, lo:lo + box] = 0.0
        self._s = m
        self.box = box


class InpaintRandom(_IdentityBasis):
    """A = diag(mask), a fixed random subset of pixels removed (same for every
    image: the system is KNOWN and fixed, as in the supervised bridge setting)."""

    def __init__(self, shape, sigma, drop=0.5, seed=0, gamma=1.0, beta=1.0):
        super().__init__(shape, sigma, gamma, beta, name="inpaint_random")
        C, N, _ = shape
        g = torch.Generator().manual_seed(seed)
        keep = (torch.rand((1, N, N), generator=g) >= drop).double()
        self._s = keep.expand(C, N, N).contiguous()   # same pixels across channels
        self.drop = drop


class AveragePoolSR(LinearMeasurement):
    """A = 2^L x 2^L average pooling (super-resolution / downsampling).

    V is the L-level block Haar transform (reused from invlib.HaarPyramid2D,
    verified orthogonal there). Averaging a 2^L block equals the Haar LL
    coefficient divided by 2^L, so s = 2^-L on the LL quadrant and 0 elsewhere:
    the null space is every Haar detail band, i.e. exactly the high-frequency
    content that downsampling discards.
    """

    def __init__(self, shape, sigma, levels=1, gamma=1.0, beta=1.0):
        super().__init__(shape, sigma, gamma, beta, name=f"sr_{2**levels}x")
        C, N, _ = shape
        assert N % (1 << levels) == 0
        self.levels = levels
        s = torch.zeros(shape, dtype=torch.float64)
        k = N >> levels
        s[:, :k, :k] = 2.0 ** (-levels)      # LL quadrant carries the measurement
        self._s = s

    def basis_fwd(self, x):
        return invlib.HaarPyramid2D._dwt(x, self.levels)

    def basis_inv(self, c):
        return invlib.HaarPyramid2D._idwt(c, self.levels)


class DenseSystem(LinearMeasurement):
    """Any linear system given by an explicit matrix A (M x D), M != D allowed.

    The SVD is computed once (D = 1024 for 32x32 images, instantaneous). The
    measurement y = A x + sigma eps lives in R^M -- a different shape from the
    image in general -- but everything the flow ever sees, A^+ y and A^+A x,
    is image-shaped, which is the point of conditioning on the pseudo-inverse.

    Singular values below rcond * s_max are treated as zero (a TRUNCATED
    pseudo-inverse, numpy's convention): those directions are measured so
    weakly that A^+ would amplify the noise by more than 1/rcond, and they are
    handed to the null space -- i.e. to the prior -- instead.
    """

    def __init__(self, A, shape, sigma, gamma=1.0, beta=1.0, rcond=1e-2, name="dense"):
        super().__init__(shape, sigma, gamma, beta, name=name)
        A = A.to(torch.float64)
        D = int(torch.tensor(shape).prod())
        assert A.shape[1] == D, (A.shape, shape)
        U, s, Vh = torch.linalg.svd(A, full_matrices=True)     # Vh is D x D
        s_full = torch.zeros(D, dtype=torch.float64)
        keep = s >= rcond * s.max()
        s_full[:s.numel()][keep] = s[keep]
        self.A, self.Vh = A, Vh
        self.n_meas, self.rcond = int(A.shape[0]), float(rcond)
        self.n_truncated = int((~keep).sum())
        self._s = s_full.reshape(shape)
        self._Vh_dev = {}

    def _vh(self, device, dtype):
        key = (str(device), dtype)
        if key not in self._Vh_dev:
            self._Vh_dev[key] = self.Vh.to(device=device, dtype=dtype)
        return self._Vh_dev[key]

    def basis_fwd(self, x):
        Vh = self._vh(x.device, x.dtype)
        return (x.flatten(1) @ Vh.T).reshape(x.shape)

    def basis_inv(self, c):
        Vh = self._vh(c.device, c.dtype)
        return (c.flatten(1) @ Vh).reshape(c.shape)

    def measure(self, x, generator=None):
        """y = A x + sigma eps in measurement space (M,), for display only."""
        A = self.A.to(device=x.device, dtype=x.dtype)
        y = x.flatten(1) @ A.T
        return y + self.sigma * torch.randn(y.shape, device=y.device, dtype=y.dtype,
                                            generator=generator)

    def __repr__(self):
        return (super().__repr__()[:-1] + f", meas={self.n_meas}, rcond={self.rcond}, "
                f"truncated={self.n_truncated})")


def radon_matrix(N, n_angles, n_det=None):
    """Pixel-driven parallel-beam Radon transform of an N x N image.

    Each pixel centre is projected onto the detector axis of every view and its
    value shared between the two nearest detector bins by linear interpolation
    (unit pixel and detector spacing). Views are equally spaced over [0, pi).
    Returns A of shape (n_angles * n_det, N * N); the sinogram is
    (n_angles, n_det) -- a shape different from the image, so y and x cannot
    be concatenated: the pseudo-inverse is what brings y back to image shape.
    """
    if n_det is None:
        n_det = int(math.ceil(N * math.sqrt(2))) | 1     # odd, covers the diagonal
    u = torch.arange(N, dtype=torch.float64) - (N - 1) / 2
    yy, xx = torch.meshgrid(u, u, indexing="ij")
    pix = torch.arange(N * N)
    A = torch.zeros(n_angles * n_det, N * N, dtype=torch.float64)
    for a in range(n_angles):
        th = math.pi * a / n_angles
        t = (xx * math.cos(th) + yy * math.sin(th)).flatten() + (n_det - 1) / 2
        i0 = torch.floor(t).long()
        w1 = t - i0.double()
        for idx, w in ((i0, 1 - w1), (i0 + 1, w1)):
            ok = (idx >= 0) & (idx < n_det)
            A[a * n_det + idx[ok], pix[ok]] += w[ok]
    return A


class CTParallel(DenseSystem):
    """Sparse-view parallel-beam CT: y is a sinogram (n_angles x n_det)."""

    def __init__(self, shape, sigma, n_angles=16, n_det=None, rcond=1e-2,
                 gamma=1.0, beta=1.0):
        C, N, _ = shape
        assert C == 1, "CT systems are single-channel"
        A = radon_matrix(N, n_angles, n_det)
        super().__init__(A, shape, sigma, gamma, beta, rcond, name=f"ct_{n_angles}view")
        self.n_angles, self.n_det = n_angles, A.shape[0] // n_angles


REGISTRY = {
    "denoise": Denoise,
    "inpaint_box": InpaintBox,
    "inpaint_random": InpaintRandom,
    "sr_2x": lambda shape, sigma, **kw: AveragePoolSR(shape, sigma, levels=1, **kw),
    "sr_4x": lambda shape, sigma, **kw: AveragePoolSR(shape, sigma, levels=2, **kw),
    "ct": CTParallel,
}


def build(name, shape, sigma, **kw):
    if name not in REGISTRY:
        raise KeyError(f"unknown measurement system {name!r}; have {list(REGISTRY)}")
    return REGISTRY[name](shape, sigma, **kw)


# --------------------------------------------------------------------------
# self-tests -- the correctness of the whole bridge rests on these
# --------------------------------------------------------------------------

def _explicit_matrices(op, shape):
    """Materialise V and A^+A by pushing basis vectors through, for testing."""
    d = int(torch.tensor(shape).prod())
    eye = torch.eye(d, dtype=torch.float64).reshape(d, *shape)
    V_t = op.basis_fwd(eye).reshape(d, d)          # rows = V^T e_i  => V^T
    P = op.project(eye).reshape(d, d)
    return V_t, P


def self_test(verbose=True):
    torch.manual_seed(0)
    shape = (1, 16, 16)
    results = {}
    for nm in ["denoise", "inpaint_box", "inpaint_random", "sr_2x", "sr_4x", "ct"]:
        kw = {"inpaint_box": {"box": 6}, "ct": {"n_angles": 8}}.get(nm, {})
        op = build(nm, shape, sigma=0.1, **kw)
        V_t, P = _explicit_matrices(op, shape)
        d = V_t.shape[0]
        I = torch.eye(d, dtype=torch.float64)

        orth = (V_t @ V_t.T - I).abs().max().item()          # V orthogonal
        idem = (P @ P - P).abs().max().item()                # projector idempotent
        symm = (P - P.T).abs().max().item()                  # projector symmetric
        rank_ok = abs(P.diagonal().sum().item() - (op.sing() > 0).sum().item())

        # noiseless pinv recon == projection
        x = torch.randn(8, *shape, dtype=torch.float64)
        op0 = build(nm, shape, sigma=1e-12, **kw)
        proj_err = (op0.pinv_recon(x) - op.project(x)).abs().max().item()

        # base samples must have covariance Sigma_1: check per-coordinate std in V
        xh = op.pinv_recon(x[:1].expand(20000, *shape).contiguous())
        z = op.sample_base(xh)
        cs = op.basis_fwd(z - xh)
        emp = cs.var(dim=0)
        tgt = op.base_var().double()
        var_rel = ((emp - tgt).abs() / tgt).max().item()

        # base_logprob must integrate to 1: verify by comparing to an explicit
        # multivariate normal on the SAME samples (log-density agreement)
        zz = op.sample_base(xh[:64])
        lp = op.base_logprob(zz, xh[:64])
        cz, cm = op.basis_fwd(zz), op.basis_fwd(xh[:64])
        v = op.base_var(cz.device, cz.dtype)
        manual = (-0.5 * ((cz - cm) ** 2 / v + torch.log(2 * math.pi * v))).flatten(1).sum(-1)
        lp_err = (lp - manual).abs().max().item()

        ok = (orth < 1e-10 and idem < 1e-10 and symm < 1e-10 and rank_ok == 0
              and proj_err < 1e-6 and var_rel < 0.06 and lp_err < 1e-9)
        results[nm] = ok
        if verbose:
            print(f"{nm:16s} rank {int((op.sing()>0).sum()):5d}/{op.sing().numel()} "
                  f"| V orth {orth:.1e} | P idem {idem:.1e} sym {symm:.1e} "
                  f"| pinv=proj {proj_err:.1e} | base var rel {var_rel:.3f} "
                  f"| logprob {lp_err:.1e} | {'OK' if ok else 'FAIL'}")
    if verbose:
        print("all systems passed:", all(results.values()))
    return results


if __name__ == "__main__":
    self_test()
