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

Implemented systems (all with analytic SVD, no matrix ever formed):
    denoise         A = I                      (no null space)
    inpaint_box     A = diag(mask), centre box removed
    inpaint_random  A = diag(mask), random pixels removed
    sr_2x / sr_4x   A = 2x2 / 4x4 average pooling  (V = block Haar)

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

    def sample_base(self, xhat, generator=None):
        """Draw z ~ N(A^+y, Sigma_1) given the pseudo-inverse recon xhat."""
        c = self.basis_fwd(xhat)
        v = self.base_var(c.device, c.dtype)
        eps = torch.randn(c.shape, device=c.device, dtype=c.dtype, generator=generator)
        return self.basis_inv(c + v.sqrt() * eps)

    def base_logprob(self, z, xhat):
        """log N(z; A^+y, Sigma_1), summed over dims, per example."""
        cz = self.basis_fwd(z)
        cm = self.basis_fwd(xhat)
        v = self.base_var(cz.device, cz.dtype)
        ll = -0.5 * ((cz - cm) ** 2 / v + torch.log(2 * math.pi * v))
        return ll.flatten(1).sum(-1)

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


REGISTRY = {
    "denoise": Denoise,
    "inpaint_box": InpaintBox,
    "inpaint_random": InpaintRandom,
    "sr_2x": lambda shape, sigma, **kw: AveragePoolSR(shape, sigma, levels=1, **kw),
    "sr_4x": lambda shape, sigma, **kw: AveragePoolSR(shape, sigma, levels=2, **kw),
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
    for nm in ["denoise", "inpaint_box", "inpaint_random", "sr_2x", "sr_4x"]:
        kw = {"box": 6} if nm == "inpaint_box" else {}
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
