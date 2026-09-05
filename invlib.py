"""invlib.py -- Library of invertible torch modules for normalizing flows.

Two module families with two contracts:

1. SeqTransform: drop-in replacements for TarFlow's Permutation slot.
     forward(x, dim=1, inverse=False) -> Tensor
   All members are ORTHOGONAL (|det| = 1, logdet = 0), which is exactly the
   condition under which the TarFlow metablock's likelihood accounting stays
   valid and the N(0,I) prior stays invariant. The autoregression then runs
   over transform coefficients (Haar, DCT, ...) instead of raw patch order.

2. InvertibleModule: general invertible maps with explicit Jacobian.
     forward(x) -> (y, logdet)   logdet: per-sample (B,), natural log
     inverse(y) -> x
   Members include invertible convolutions, elementwise monotone
   nonlinearities, invertible linear parameterizations, and coupling.

References:
  Glow invertible 1x1 conv / PLU: Kingma & Dhariwal, arXiv:1807.03039
  Periodic (Fourier) + emerging invertible dxd convs: Hoogeboom, van den Berg,
      Welling, ICML 2019, arXiv:1901.11137
  Invertible Convolutional Flow: Karami et al., NeurIPS 2019
  Neural Spline Flows (rational-quadratic): Durkan et al., arXiv:1906.04032
  Sum-of-Squares Polynomial Flow: Jaini et al., arXiv:1905.02325
  Woodbury transformations: Lu & Huang, NeurIPS 2020
  Householder/Cayley orthogonal parameterizations: see JMLR 22(57) flows review
  SpectralFloorLinear: eigenvalue-floored low-rank map (M. Tivnan, this work)
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

# ==========================================================================
# Orthogonal matrix builders (float64, cast on use)
# ==========================================================================


def haar_matrix(n: int) -> torch.Tensor:
    """Orthonormal Haar transform matrix, n must be a power of 2."""
    assert n & (n - 1) == 0 and n > 0, f"Haar needs power-of-2 size, got {n}"
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < n:
        top = torch.kron(h, torch.tensor([[1.0, 1.0]], dtype=torch.float64))
        bot = torch.kron(torch.eye(h.shape[0], dtype=torch.float64),
                         torch.tensor([[1.0, -1.0]], dtype=torch.float64))
        h = torch.cat([top, bot], dim=0) / math.sqrt(2.0)
    return h


def hadamard_matrix(n: int) -> torch.Tensor:
    """Orthonormal Walsh-Hadamard matrix, n must be a power of 2."""
    assert n & (n - 1) == 0 and n > 0, f"Hadamard needs power-of-2 size, got {n}"
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0) / math.sqrt(2.0)
    return h


def dct_matrix(n: int) -> torch.Tensor:
    """Orthonormal DCT-II matrix (any n)."""
    k = torch.arange(n, dtype=torch.float64).unsqueeze(1)
    m = torch.arange(n, dtype=torch.float64).unsqueeze(0)
    c = torch.cos(math.pi * k * (2 * m + 1) / (2 * n)) * math.sqrt(2.0 / n)
    c[0] /= math.sqrt(2.0)
    return c


def hartley_matrix(n: int) -> torch.Tensor:
    """Orthonormal discrete Hartley transform (real 'Fourier'; involution)."""
    k = torch.arange(n, dtype=torch.float64).unsqueeze(1)
    m = torch.arange(n, dtype=torch.float64).unsqueeze(0)
    a = 2 * math.pi * k * m / n
    return (torch.cos(a) + torch.sin(a)) / math.sqrt(n)


def random_orthogonal_matrix(n: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(n, n, generator=g, dtype=torch.float64)
    q, r = torch.linalg.qr(a)
    return q * torch.sign(torch.diagonal(r)).unsqueeze(0)  # deterministic sign


# ==========================================================================
# Family 1: SeqTransform -- TarFlow permutation-slot compatible (orthogonal)
# ==========================================================================


class SeqTransform(nn.Module):
    """Orthogonal transform along one tensor dimension.

    TarFlow-compatible signature: forward(x, dim=1, inverse=False) -> Tensor.
    Convention: y = Q x along `dim` (coefficients are rows of Q).
    """

    def __init__(self, seq_length: int):
        super().__init__()
        self.seq_length = seq_length

    def matrix(self) -> torch.Tensor:
        """The (T,T) orthogonal matrix (float32). Override or set buffer Q."""
        return self.Q

    def forward(self, x, dim: int = 1, inverse: bool = False):
        q = self.matrix().to(dtype=x.dtype, device=x.device)
        xm = x.movedim(dim, -1)
        ym = xm @ (q if inverse else q.t())
        return ym.movedim(-1, dim)


class SeqIdentity(SeqTransform):
    def forward(self, x, dim: int = 1, inverse: bool = False):
        return x

    def matrix(self):
        return torch.eye(self.seq_length)


class SeqFlip(SeqTransform):
    def forward(self, x, dim: int = 1, inverse: bool = False):
        return x.flip(dims=[dim])

    def matrix(self):
        return torch.eye(self.seq_length).flip(0)


class SeqRandomPermutation(SeqTransform):
    def __init__(self, seq_length: int, seed: int = 0):
        super().__init__(seq_length)
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(seq_length, generator=g)
        self.register_buffer("perm", perm)
        self.register_buffer("inv_perm", torch.argsort(perm))

    def forward(self, x, dim: int = 1, inverse: bool = False):
        idx = self.inv_perm if inverse else self.perm
        return x.index_select(dim, idx.to(x.device))

    def matrix(self):
        return torch.eye(self.seq_length)[self.perm]


class _FixedMatrixSeq(SeqTransform):
    def __init__(self, seq_length: int, q64: torch.Tensor):
        super().__init__(seq_length)
        self.register_buffer("Q", q64.to(torch.float32))


class SeqHaar(_FixedMatrixSeq):
    def __init__(self, seq_length: int):
        super().__init__(seq_length, haar_matrix(seq_length))


class SeqHadamard(_FixedMatrixSeq):
    def __init__(self, seq_length: int):
        super().__init__(seq_length, hadamard_matrix(seq_length))


class SeqDCT(_FixedMatrixSeq):
    def __init__(self, seq_length: int):
        super().__init__(seq_length, dct_matrix(seq_length))


class SeqHartley(_FixedMatrixSeq):
    def __init__(self, seq_length: int):
        super().__init__(seq_length, hartley_matrix(seq_length))


class SeqRandomOrthogonal(_FixedMatrixSeq):
    def __init__(self, seq_length: int, seed: int = 0):
        super().__init__(seq_length, random_orthogonal_matrix(seq_length, seed))


class SeqHouseholder(SeqTransform):
    """Learnable orthogonal map: product of k Householder reflections."""

    def __init__(self, seq_length: int, k: int = 8, seed: int = 0):
        super().__init__(seq_length)
        g = torch.Generator().manual_seed(seed)
        self.v = nn.Parameter(torch.randn(k, seq_length, generator=g))

    def matrix(self):
        q = torch.eye(self.seq_length, device=self.v.device, dtype=self.v.dtype)
        for i in range(self.v.shape[0]):
            v = self.v[i] / self.v[i].norm().clamp_min(1e-8)
            q = q - 2.0 * torch.outer(v, v @ q)
        return q


class SeqCayley(SeqTransform):
    """Learnable orthogonal map via the Cayley transform of a skew matrix."""

    def __init__(self, seq_length: int, seed: int = 0, init_scale: float = 0.1):
        super().__init__(seq_length)
        g = torch.Generator().manual_seed(seed)
        self.w = nn.Parameter(init_scale * torch.randn(seq_length, seq_length, generator=g)
                              / math.sqrt(seq_length))

    def matrix(self):
        a = self.w - self.w.t()
        eye = torch.eye(self.seq_length, device=a.device, dtype=a.dtype)
        return torch.linalg.solve(eye + a, eye - a)


class SeqCompose(SeqTransform):
    """Compose transforms t1 then t2 then ... (forward order)."""

    def __init__(self, transforms):
        super().__init__(transforms[0].seq_length)
        self.transforms = nn.ModuleList(transforms)

    def forward(self, x, dim: int = 1, inverse: bool = False):
        ts = reversed(self.transforms) if inverse else self.transforms
        for t in ts:
            x = t(x, dim=dim, inverse=inverse)
        return x

    def matrix(self):
        q = torch.eye(self.seq_length)
        for t in self.transforms:
            q = t.matrix().to(q.dtype) @ q
        return q


# --------------------------------------------------------------------------
# Feature-domain transforms: replace the MEANING of the token sequence.
#
# Unlike the SeqTransforms above (which mix token vectors across the sequence
# position only), these act on the whole image: tokens -> image (fold) ->
# separable 2D orthogonal transform -> coarse-to-fine coefficient ordering ->
# re-chunk into tokens. Token 0 becomes the coarse approximation (containing
# the DC coefficient) and later tokens are progressively finer detail, so a
# causal AR model conditions fine structure on coarse structure.
#
# They keep TarFlow's permutation-slot signature and are orthogonal on the
# flattened (T*C) space (fold/unfold are permutations; H X H^T with orthogonal
# H is orthogonal; reordering is a permutation), so |det| = 1 and the
# metablock likelihood accounting is untouched.
# --------------------------------------------------------------------------


class FeatureDomain2D(SeqTransform):
    """Base: tokens (B,T,C) -> image -> 2D transform (per channel) ->
    coefficient ordering -> tokens. Single-channel-per-pixel images with
    C = channels * patch^2; matches Model.patchify/unpatchify conventions."""

    def __init__(self, img_size: int, patch_size: int, channels: int = 1):
        T = (img_size // patch_size) ** 2
        super().__init__(T)
        self.N, self.p, self.ch = img_size, patch_size, channels
        h64 = self._matrix_1d(img_size)                    # (N,N) float64
        self.register_buffer("H", h64.to(torch.float32))
        order = self._coeff_order(img_size)                # (N*N,) long
        self.register_buffer("order", order)
        self.register_buffer("inv_order", torch.argsort(order))

    def _matrix_1d(self, n):
        raise NotImplementedError

    def _coeff_order(self, n):
        raise NotImplementedError

    def forward(self, x, dim: int = 1, inverse: bool = False):
        if dim == 0:
            # MetaBlock transforms its learnable pos_embed table (T, width)
            # through this slot. In the coefficient domain each position t is
            # a fixed coefficient group, so the table is returned unchanged:
            # positions get their own learned embeddings directly (identity is
            # a valid, fully general choice for a free parameter).
            return x
        B, T, C = x.shape
        N, p, ch = self.N, self.p, self.ch
        assert C == ch * p * p, f"expected token dim {ch * p * p}, got {C}"
        h = self.H.to(dtype=x.dtype, device=x.device)
        if not inverse:
            img = F.fold(x.transpose(1, 2), (N, N), p, stride=p)   # (B,ch,N,N)
            coef = h @ img @ h.t()                                  # analysis
            flat = coef.reshape(B, ch, N * N)[:, :, self.order]     # coarse->fine
            return flat.reshape(B, ch, T, p * p).permute(0, 2, 1, 3).reshape(B, T, C)
        flat = x.reshape(B, T, ch, p * p).permute(0, 2, 1, 3).reshape(B, ch, N * N)
        coef = torch.zeros_like(flat)
        coef[:, :, self.order] = flat
        img = h.t() @ coef.reshape(B, ch, N, N) @ h                 # synthesis
        return F.unfold(img, p, stride=p).transpose(1, 2)


class Haar2DFeatures(FeatureDomain2D):
    """Full 2D multiresolution Haar analysis; ordering by max(scale_i, scale_j)
    so token 0 is the coarsest (LL) band including the DC coefficient."""

    def _matrix_1d(self, n):
        return haar_matrix(n)

    def _coeff_order(self, n):
        s = [0] + [int(math.floor(math.log2(r))) + 1 for r in range(1, n)]
        keys = [(max(s[i], s[j]), i, j) for i in range(n) for j in range(n)]
        return torch.tensor(sorted(range(n * n), key=lambda k: keys[k]), dtype=torch.long)


class DCT2DFeatures(FeatureDomain2D):
    """Full 2D orthonormal DCT-II; zigzag-style low-to-high frequency ordering
    (JPEG-like), token 0 = lowest spatial frequencies including DC."""

    def _matrix_1d(self, n):
        return dct_matrix(n)

    def _coeff_order(self, n):
        keys = [(i + j, i, j) for i in range(n) for j in range(n)]
        return torch.tensor(sorted(range(n * n), key=lambda k: keys[k]), dtype=torch.long)


class HaarPyramid2D(SeqTransform):
    """Depth-L Mallat 2D Haar pyramid in the STANDARD subband layout, applied
    to the folded image and handed straight back to patchify.

    Unlike Haar2DFeatures (which applies the full-depth tensor-product Haar
    and then reorders all N*N coefficients coarse-to-fine), this recurses only
    on the LL band L times and keeps the classical quadrant layout. With
    L = log2(patch_size) + 1 chosen so that N >> L == patch_size, the LL_L
    band exactly fills the top-left patch, so after patchify **token 0 is a
    patch_size x patch_size thumbnail of the whole image** -- i.e. an
    orthonormal rescaling of average-pooling the image down to one patch --
    and later tokens carry progressively finer detail bands. No coefficient
    reordering is needed: the subband layout already delivers coarse-to-fine
    at patch granularity.

    Each level is the orthonormal Haar step
    LL,HL,LH,HH = (a+b+c+d)/2, (a-b+c-d)/2, (a+b-c-d)/2, (a-b-c+d)/2,
    so the map is orthogonal (Q Q^T = I, |det| = 1, logdet 0) and the N(0,I)
    prior and TarFlow likelihood accounting are untouched.
    """

    def __init__(self, img_size: int, patch_size: int, channels: int = 1,
                 levels: int | None = None):
        T = (img_size // patch_size) ** 2
        super().__init__(T)
        self.N, self.p, self.ch = img_size, patch_size, channels
        if levels is None:                      # take LL down to exactly one patch
            levels = int(round(math.log2(img_size // patch_size)))
        assert img_size % (1 << levels) == 0, "img_size must be divisible by 2**levels"
        self.levels = levels

    @staticmethod
    def _dwt(x, levels):
        x = x.clone()
        N = x.shape[-1]
        for l in range(levels):
            n = N >> l
            s = x[..., :n, :n]
            a, b = s[..., 0::2, 0::2], s[..., 0::2, 1::2]
            c, d = s[..., 1::2, 0::2], s[..., 1::2, 1::2]
            x[..., :n, :n] = torch.cat(
                [torch.cat([(a + b + c + d) / 2, (a - b + c - d) / 2], -1),
                 torch.cat([(a + b - c - d) / 2, (a - b - c + d) / 2], -1)], -2)
        return x

    @staticmethod
    def _idwt(y, levels):
        y = y.clone()
        N = y.shape[-1]
        for l in reversed(range(levels)):
            n = N >> l
            h = n // 2
            q = y[..., :n, :n]
            LL, HL = q[..., :h, :h], q[..., :h, h:]
            LH, HH = q[..., h:, :h], q[..., h:, h:]
            s = torch.zeros_like(q)
            s[..., 0::2, 0::2] = (LL + HL + LH + HH) / 2
            s[..., 0::2, 1::2] = (LL - HL + LH - HH) / 2
            s[..., 1::2, 0::2] = (LL + HL - LH - HH) / 2
            s[..., 1::2, 1::2] = (LL - HL - LH + HH) / 2
            y[..., :n, :n] = s
        return y

    def forward(self, x, dim: int = 1, inverse: bool = False):
        if dim == 0:
            # MetaBlock routes its learnable pos_embed table (T, width) through
            # this slot; in the coefficient domain each position is a fixed
            # band, so the free parameter table passes through unchanged.
            return x
        B, T, C = x.shape
        N, p, ch = self.N, self.p, self.ch
        assert C == ch * p * p, f"expected token dim {ch * p * p}, got {C}"
        img = F.fold(x.transpose(1, 2), (N, N), p, stride=p)
        img = self._idwt(img, self.levels) if inverse else self._dwt(img, self.levels)
        return F.unfold(img, p, stride=p).transpose(1, 2)

    def matrix(self):
        raise NotImplementedError(
            "HaarPyramid2D acts on the folded image, not as a (T,T) token "
            "matrix; orthogonality is verified directly in step6.")


FEATURE_REGISTRY = {
    "haar2d": Haar2DFeatures,
    "dct2d": DCT2DFeatures,
    "haar_pyramid": HaarPyramid2D,
}


SEQ_REGISTRY = {
    "identity": SeqIdentity,
    "flip": SeqFlip,
    "random_perm": SeqRandomPermutation,
    "haar": SeqHaar,
    "hadamard": SeqHadamard,
    "dct": SeqDCT,
    "hartley": SeqHartley,
    "rand_ortho": SeqRandomOrthogonal,
    "householder": SeqHouseholder,
    "cayley": SeqCayley,
}


# --------------------------------------------------------------------------
# Bounded log-scale for TarFlow's affine coupling
# --------------------------------------------------------------------------

def bound_log_scale(model, bound):
    """Soft-bound the log-scale head of every TarFlow MetaBlock in `model`.

    The official coupling is z = (x - xb) * exp(-xa) with xa the raw output of
    a linear head applied to the un-normalised residual stream, so xa grows
    linearly with the block's input magnitude and exp() turns a moderately
    unusual token into an overflow that the next block amplifies further. This
    replaces xa by  bound * tanh(xa / bound)  in BOTH directions, so a single
    block can scale by at most exp(+-bound); everything else -- parameters,
    state_dict keys, permutations, KV-cached sampling -- is untouched, and
    for |xa| << bound the map is the official one to first order.

    The official repository is not modified: the blocks are re-classed in place
    to a subclass that only overrides `forward` and `reverse_step`.
    """
    base_cls = type(model.blocks[0])
    if any(blk.class_embed is not None for blk in model.blocks):
        raise NotImplementedError("bound_log_scale only supports unconditional models")
    b = float(bound)

    class BoundedMetaBlock(base_cls):
        log_scale_bound = b

        def _squash(self, xa):
            return self.log_scale_bound * torch.tanh(xa / self.log_scale_bound)

        def forward(self, x, y=None):
            x = self.permutation(x)
            pos_embed = self.permutation(self.pos_embed, dim=0)
            x_in = x
            x = self.proj_in(x) + pos_embed
            if self.class_embed is not None:
                x = x + (self.class_embed[y] if y is not None
                         else self.class_embed.mean(dim=0))
            for block in self.attn_blocks:
                x = block(x, self.attn_mask)
            x = self.proj_out(x)
            x = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
            if self.nvp:
                xa, xb = x.chunk(2, dim=-1)
                xa = self._squash(xa)
            else:
                xb, xa = x, torch.zeros_like(x)
            scale = (-xa.float()).exp().type(xa.dtype)
            return self.permutation((x_in - xb) * scale, inverse=True), -xa.mean(dim=[1, 2])

        def reverse_step(self, x, pos_embed, i, y=None, attn_temp=1.0, which_cache="cond"):
            xa, xb = base_cls.reverse_step(self, x, pos_embed, i, y, attn_temp, which_cache)
            return (self._squash(xa) if self.nvp else xa), xb

    for blk in model.blocks:
        blk.__class__ = BoundedMetaBlock
    return model


def condition_on_image(model, cond_channels, bound=None):
    """Make every TarFlow MetaBlock in `model` conditional on an IMAGE.

    The conditioning image has the same spatial size as the data (any number
    of channels): it is patchified exactly like the image, embedded by its own
    linear projection and positional table -- an extra input channel of the
    coupling network -- and prepended to the token sequence as a PREFIX. Image
    token i attends to every conditioning token and, causally, to image tokens
    <= i; the output at the LAST conditioning token supplies the affine
    parameters of image token 0 (the official model uses zeros there), and the
    output at image token i those of token i+1, as in the official model. The
    coupling itself, the permutations and the KV-cached sampling loop are the
    official ones, so the map stays exactly invertible given the conditioning
    image and its log-determinant is unchanged. Sampling first runs the prefix
    through the cache (bidirectional over the prefix), then the official
    one-token-at-a-time loop.

    The model's `y` argument now carries the patchified conditioning image
    (B, T, cond_channels * patch^2), so `Model.forward(x, y)` and
    `Model.reverse(z, y)` are used unchanged. `bound` optionally applies the
    same soft log-scale bound as bound_log_scale. The official repository is
    not modified: blocks are re-classed in place and gain two parameters
    (`cond_proj`, `cond_pos_embed`) and one buffer (`cond_attn_mask`).
    """
    base_cls = type(model.blocks[0])
    if any(blk.class_embed is not None for blk in model.blocks):
        raise NotImplementedError("condition_on_image: class embeddings are not combined")
    b = float(bound) if bound else None

    class CondMetaBlock(base_cls):
        log_scale_bound = b

        def _squash(self, xa):
            if self.log_scale_bound is None:
                return xa
            return self.log_scale_bound * torch.tanh(xa / self.log_scale_bound)

        def _split(self, h):
            if self.nvp:
                xa, xb = h.chunk(2, dim=-1)
                return self._squash(xa), xb
            return torch.zeros_like(h), h

        def forward(self, x, y=None):
            if y is None:
                raise ValueError("conditional block needs the conditioning tokens as y")
            x = self.permutation(x)
            c = self.permutation(y)
            pos_embed = self.permutation(self.pos_embed, dim=0)
            cond_pos = self.permutation(self.cond_pos_embed, dim=0)
            x_in, T = x, x.size(1)
            h = torch.cat([self.cond_proj(c) + cond_pos, self.proj_in(x) + pos_embed], dim=1)
            for block in self.attn_blocks:
                h = block(h, self.cond_attn_mask)
            h = self.proj_out(h)[:, T - 1:2 * T - 1]     # prefix end -> token 0, token i -> i+1
            xa, xb = self._split(h)
            scale = (-xa.float()).exp().type(xa.dtype)
            return self.permutation((x_in - xb) * scale, inverse=True), -xa.mean(dim=[1, 2])

        def reverse_step(self, x, pos_embed, i, y=None, attn_temp=1.0, which_cache="cond"):
            xa, xb = base_cls.reverse_step(self, x, pos_embed, i, None, attn_temp, which_cache)
            return (self._squash(xa) if self.nvp else xa), xb

        def reverse(self, x, y=None, guidance=0, guide_what="ab", attn_temp=1.0,
                    annealed_guidance=False):
            if y is None or guidance:
                raise ValueError("conditional reverse needs y and supports no guidance")
            x = self.permutation(x)
            c = self.permutation(y)
            pos_embed = self.permutation(self.pos_embed, dim=0)
            cond_pos = self.permutation(self.cond_pos_embed, dim=0)
            self.set_sample_mode(True)
            # prefix: all conditioning tokens at once, no mask, filling the KV cache
            h = self.cond_proj(c) + cond_pos
            for block in self.attn_blocks:
                h = block(h, attn_temp=attn_temp, which_cache="cond")
            za, zb = self._split(self.proj_out(h[:, -1:]))
            x[:, 0] = x[:, 0] * za[:, 0].float().exp().type(za.dtype) + zb[:, 0]
            for i in range(x.size(1) - 1):
                za, zb = self.reverse_step(x, pos_embed, i, None, attn_temp, "cond")
                x[:, i + 1] = x[:, i + 1] * za[:, 0].float().exp().type(za.dtype) + zb[:, 0]
            self.set_sample_mode(False)
            return self.permutation(x, inverse=True)

    cond_dim = int(cond_channels) * model.patch_size ** 2
    for blk in model.blocks:
        T, C = blk.pos_embed.shape
        dev = blk.pos_embed.device
        blk.__class__ = CondMetaBlock
        blk.cond_proj = nn.Linear(cond_dim, C).to(dev)
        blk.cond_pos_embed = nn.Parameter(torch.randn(T, C, device=dev) * 1e-2)
        mask = torch.zeros(2 * T, 2 * T, device=dev)
        mask[:T, :T] = 1.0                                   # prefix: bidirectional
        mask[T:, :T] = 1.0                                   # image tokens see the prefix
        mask[T:, T:] = torch.tril(torch.ones(T, T, device=dev))   # and causally each other
        blk.register_buffer("cond_attn_mask", mask)
    return model


# --------------------------------------------------------------------------
# Differentiable sampling direction for TarFlow
# --------------------------------------------------------------------------

def _block_reverse_differentiable(blk, z, attn_temp=1.0):
    """One MetaBlock, sampling direction, with autograd intact.

    The official MetaBlock.reverse writes each recovered token into the input
    tensor in place (`x[:, i+1] = ...`); autograd rejects that because the
    tensor was already consumed by `proj_in`. This loop keeps the recovered
    tokens in a list instead and calls the official (or re-classed)
    `reverse_step` for the coupling parameters, one token at a time with the
    official KV cache. Returns the recovered tokens and, per example, the sum
    of the log-scales `za` it applied -- so that
        log q(x) = log N(z) - sum_blocks sum(za)
    which is the official forward log-density evaluated at x.
    """
    x = blk.permutation(z)
    pos_embed = blk.permutation(blk.pos_embed, dim=0)
    blk.set_sample_mode(True)
    try:
        toks = [x[:, 0]]                          # token 0: xa = xb = 0 in the official model
        sum_za = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        for i in range(x.size(1) - 1):
            # reverse_step slices x[:, i:i+1] and pos_embed[i:i+1]; feed it the
            # current token alone and the shifted table so the official code
            # (and any re-classed subclass) runs unchanged
            za, zb = blk.reverse_step(toks[i].unsqueeze(1), pos_embed[i:], 0, None, attn_temp, "cond")
            scale = za[:, 0].float().exp().type(za.dtype)
            toks.append(x[:, i + 1] * scale + zb[:, 0])
            sum_za = sum_za + za[:, 0].flatten(1).sum(-1)
    finally:
        blk.set_sample_mode(False)                # always leave the block in parallel mode
    return blk.permutation(torch.stack(toks, dim=1), inverse=True), sum_za


def differentiable_reverse(model, z, checkpoint=True, attn_temp=1.0):
    """x = f^{-1}(z) through every block of a TarFlow `model`, differentiable
    w.r.t. z and the parameters.

    Returns (x, log_q) with x the image (B, C, H, W) and log_q = log N(z) -
    sum_blocks sum(za) the model's log-density at x in nats per image -- the
    same quantity the official forward pass gives, obtained for free from the
    sampling pass (verified by step 10's self-test). Memory: each block's
    KV cache is O(T^2 C) per layer; with `checkpoint` a block's activations
    are recomputed in the backward pass so only one block is live at a time
    (the re-entrant checkpoint: the non-re-entrant one stops the recompute
    early with an exception, which leaves the blocks in sample mode with
    their caches full -- both a leak and a correctness hazard for the next
    parallel forward pass). The re-entrant checkpoint needs an input that
    requires grad and works with `.backward()`, not `torch.autograd.grad`.
    Only the unconditional model (`y = None`) and unit `var` are supported.
    """
    if not bool(torch.all(model.var == 1)):
        raise NotImplementedError("differentiable_reverse assumes the nvp unit base variance")
    log_q = -0.5 * (z ** 2 + math.log(2 * math.pi)).flatten(1).sum(-1)
    use_ckpt = checkpoint and torch.is_grad_enabled()
    x = z
    if use_ckpt and not x.requires_grad:
        x = x.detach().requires_grad_(True)
    for blk in reversed(model.blocks):
        if use_ckpt:
            x, sum_za = torch.utils.checkpoint.checkpoint(
                _block_reverse_differentiable, blk, x, attn_temp, use_reentrant=True)
        else:
            x, sum_za = _block_reverse_differentiable(blk, x, attn_temp)
        log_q = log_q - sum_za
    return model.unpatchify(x), log_q


# --------------------------------------------------------------------------
# Fast differentiable sampling direction: CUDA-graph-replayed token steps
# --------------------------------------------------------------------------
#
# The sampling direction of a MetaBlock is autoregressive: token i+1 needs the
# transformer evaluated on tokens 0..i, so one block is 63 sequential
# evaluations and one image 8 x 63 = 504 of them, each on a single token.
# Each evaluation is ~150 tiny kernels whose cost is launch latency, not
# arithmetic (batch 1 and batch 32 take the same time), and eager autograd
# pays the same latency again in the backward. Here one token step is a pure
# function of static shape -- the KV cache is a fixed (B, L, T, C) tensor with
# the new slot written functionally -- so it can be captured once per block as
# two CUDA graphs (forward; forward-recompute + backward) and replayed 504
# times per pass with ~zero launch overhead. `torch.compile` on the pure
# function additionally fuses the elementwise work before capture. The
# official module is only read (parameters, permutation); it is never put in
# sample mode. Saved state per token step is the token and the new cache slot
# (B, L, C), so no activation checkpointing is needed at batch 32.

N_LAYER_PARAMS = 12


def _block_params(blk):
    """Flat tuple of a MetaBlock's parameter tensors (references, no copies)."""
    ps = [blk.proj_in.weight, blk.proj_in.bias, blk.proj_out.weight, blk.proj_out.bias]
    for ab in blk.attn_blocks:
        a, m = ab.attention, ab.mlp
        ps += [a.norm.weight, a.norm.bias, a.qkv.weight, a.qkv.bias, a.proj.weight, a.proj.bias,
               m.norm.weight, m.norm.bias, m.main[0].weight, m.main[0].bias, m.main[2].weight, m.main[2].bias]
    return tuple(ps)


def _token_step(tok, K, V, pos, bias, onehot, scale, n_heads, bound, params):
    """One token of a MetaBlock's sampling direction as a pure, static-shape function.

    tok (B, Cin): recovered token i; K, V (B, L, T, C): KV caches with slots
    0..i-1 filled and the rest zero; pos (1, C): positional embedding of slot i;
    bias (1, T): additive attention bias, 0 for slots <= i and -1e30 beyond;
    onehot (1, T): indicator of slot i. Returns za, zb (B, Cin) -- the affine
    parameters for token i+1, as `MetaBlock.reverse_step` -- and the caches
    with slot i written. Mirrors `Attention.forward_spda` / `MLP` / `MetaBlock.
    reverse_step` of the official code exactly (LayerNorm eps 1e-5, exact GELU,
    softmax scale 1/sqrt(head_dim)), all in float32; `bound` (a float or None)
    applies the `bound_log_scale` squash za -> bound * tanh(za / bound) of the
    re-classed blocks."""
    B, L, T, C = K.shape
    H = n_heads
    D = C // H
    w_in, b_in, w_out, b_out = params[:4]
    x = F.linear(tok, w_in, b_in) + pos
    Kn, Vn = [], []
    for l in range(L):
        (ln1w, ln1b, wqkv, bqkv, wp, bp, ln2w, ln2b, w1, b1, w2, b2) = \
            params[4 + N_LAYER_PARAMS * l: 4 + N_LAYER_PARAMS * (l + 1)]
        h = F.layer_norm(x, (C,), ln1w, ln1b, 1e-5)
        q, k, v = F.linear(h, wqkv, bqkv).split(C, dim=-1)
        Kl = K[:, l] + onehot[:, :, None] * k[:, None, :]            # slot i written, functionally
        Vl = V[:, l] + onehot[:, :, None] * v[:, None, :]
        qh = q.view(B, H, 1, D)
        Kh = Kl.view(B, T, H, D).transpose(1, 2)                     # (B, H, T, D)
        Vh = Vl.view(B, T, H, D).transpose(1, 2)
        s = torch.matmul(qh, Kh.transpose(-1, -2)) * scale + bias[:, None, None, :]
        o = torch.matmul(s.softmax(-1), Vh).reshape(B, C)
        x = x + F.linear(o, wp, bp)
        h = F.layer_norm(x, (C,), ln2w, ln2b, 1e-5)
        x = x + F.linear(F.gelu(F.linear(h, w1, b1)), w2, b2)
        Kn.append(Kl)
        Vn.append(Vl)
    za, zb = F.linear(x, w_out, b_out).chunk(2, dim=-1)
    if bound is not None:
        za = bound * torch.tanh(za / bound)
    return za, zb, torch.stack(Kn, 1), torch.stack(Vn, 1)


class _GraphedBlock:
    """The two CUDA graphs (forward; recompute + backward) of one MetaBlock's
    token step at a fixed batch size, with their static buffers."""

    def __init__(self, blk, batch, device, compile=False, warmup=3):
        self.blk = blk
        T, C = blk.pos_embed.shape
        L, Cin = len(blk.attn_blocks), blk.proj_in.in_features
        a0 = blk.attn_blocks[0].attention
        self.H, self.scale = a0.num_heads, a0.sqrt_scale ** 2
        self.bound = float(blk.log_scale_bound) if hasattr(blk, "log_scale_bound") else None
        self.B, self.T, self.L, self.C, self.Cin = batch, T, L, C, Cin
        self.params = _block_params(blk)
        with torch.no_grad():
            self.pos_table = blk.permutation(blk.pos_embed.detach(), dim=0).contiguous()
        self.bias_table = torch.triu(torch.full((T, T), -1e30, device=device), 1)   # row i: 0 up to slot i
        self.onehot_table = torch.eye(T, device=device)
        z = lambda *s: torch.zeros(*s, device=device)
        self.s_tok, self.s_K, self.s_V = z(batch, Cin), z(batch, L, T, C), z(batch, L, T, C)
        self.s_i = torch.zeros(1, dtype=torch.long, device=device)
        self.s_gza, self.s_gzb = z(batch, Cin), z(batch, Cin)
        self.s_gKn, self.s_gVn = z(batch, L, T, C), z(batch, L, T, C)
        self.fn = torch.compile(_token_step, dynamic=False, fullgraph=True) if compile else _token_step
        leaves = [t.detach().requires_grad_(True) for t in (self.s_tok, self.s_K, self.s_V)]   # share storage

        def run(tok, K, V):
            i = self.s_i
            return self.fn(tok, K, V, self.pos_table.index_select(0, i), self.bias_table.index_select(0, i),
                           self.onehot_table.index_select(0, i), self.scale, self.H, self.bound, self.params)

        def run_bwd():
            with torch.enable_grad():
                outs = run(*leaves)
                return torch.autograd.grad(outs, leaves, grad_outputs=(self.s_gza, self.s_gzb, self.s_gKn, self.s_gVn))

        # warm-up on a side stream (compiles, autotunes, initialises workspaces), then capture
        s = torch.cuda.Stream(device)
        s.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(s):
            for _ in range(warmup):
                with torch.no_grad():
                    run(self.s_tok, self.s_K, self.s_V)
                run_bwd()
        torch.cuda.current_stream(device).wait_stream(s)
        torch.cuda.synchronize(device)
        self.g_fwd = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(self.g_fwd):
            self.s_za, self.s_zb, self.s_Kn, self.s_Vn = run(self.s_tok, self.s_K, self.s_V)
        self.g_bwd = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g_bwd):
            self.s_gtok, self.s_gK, self.s_gV = run_bwd()
        torch.cuda.synchronize(device)


class _GraphedStep(torch.autograd.Function):
    """Replays the block's forward graph; the backward replays the
    recompute+backward graph with the cache rebuilt from the saved slots."""

    @staticmethod
    def forward(ctx, tok, K, V, i, gb, hist):
        gb.s_tok.copy_(tok); gb.s_K.copy_(K); gb.s_V.copy_(V); gb.s_i.fill_(i)
        gb.g_fwd.replay()
        hist.append((gb.s_Kn[:, :, i].clone(), gb.s_Vn[:, :, i].clone()))         # the new slot, (B, L, C)
        ctx.gb, ctx.hist, ctx.i = gb, hist, i
        ctx.save_for_backward(tok)
        return gb.s_za.clone(), gb.s_zb.clone(), gb.s_Kn.clone(), gb.s_Vn.clone()

    @staticmethod
    def backward(ctx, g_za, g_zb, g_Kn, g_Vn):
        gb, hist, i = ctx.gb, ctx.hist, ctx.i
        (tok,) = ctx.saved_tensors
        gb.s_tok.copy_(tok); gb.s_i.fill_(i)
        gb.s_K.zero_(); gb.s_V.zero_()
        if i > 0:
            gb.s_K[:, :, :i] = torch.stack([h[0] for h in hist[:i]], 2)
            gb.s_V[:, :, :i] = torch.stack([h[1] for h in hist[:i]], 2)
        gb.s_gza.copy_(g_za); gb.s_gzb.copy_(g_zb); gb.s_gKn.copy_(g_Kn); gb.s_gVn.copy_(g_Vn)
        gb.g_bwd.replay()
        return gb.s_gtok.clone(), gb.s_gK.clone(), gb.s_gV.clone(), None, None, None


def _block_reverse_graphed(gb, z):
    """Sampling direction of one MetaBlock with graphed token steps; returns the
    recovered tokens and the per-example sum of the applied log-scales za."""
    x = gb.blk.permutation(z)
    B, T, _ = x.shape
    toks = [x[:, 0]]
    K = torch.zeros(B, gb.L, gb.T, gb.C, device=x.device)
    V = torch.zeros_like(K)
    sum_za = torch.zeros(B, device=x.device)
    hist = []
    for i in range(T - 1):
        za, zb, K, V = _GraphedStep.apply(toks[i], K, V, i, gb, hist)
        toks.append(x[:, i + 1] * za.exp() + zb)
        sum_za = sum_za + za.sum(-1)
    return gb.blk.permutation(torch.stack(toks, 1), inverse=True), sum_za


def _block_reverse_eager(blk, z):
    """The same recursion with `_token_step` in eager mode (reference / CPU)."""
    x = blk.permutation(z)
    B, T, _ = x.shape
    Tp, C = blk.pos_embed.shape
    L = len(blk.attn_blocks)
    a0 = blk.attn_blocks[0].attention
    bound = float(blk.log_scale_bound) if hasattr(blk, "log_scale_bound") else None
    params = _block_params(blk)
    pos_table = blk.permutation(blk.pos_embed, dim=0)
    bias_table = torch.triu(torch.full((T, T), -1e30, device=x.device), 1)
    onehot_table = torch.eye(T, device=x.device)
    toks = [x[:, 0]]
    K = torch.zeros(B, L, T, C, device=x.device)
    V = torch.zeros_like(K)
    sum_za = torch.zeros(B, device=x.device)
    for i in range(T - 1):
        za, zb, K, V = _token_step(toks[i], K, V, pos_table[i:i + 1], bias_table[i:i + 1], onehot_table[i:i + 1],
                                   a0.sqrt_scale ** 2, a0.num_heads, bound, params)
        toks.append(x[:, i + 1] * za.exp() + zb)
        sum_za = sum_za + za.sum(-1)
    return blk.permutation(torch.stack(toks, 1), inverse=True), sum_za


class FastReverse:
    """x = g(z) = f^{-1}(z) for a TarFlow `model`, differentiable w.r.t. z, with
    every token step replayed from CUDA graphs. Built once per (model, batch
    size); `mode` is 'graph' (eager kernels inside the graphs), 'compile'
    (torch.compile'd token step inside the graphs) or 'eager' (no graphs).
    Returns (x, log_q) exactly as `differentiable_reverse`. The model must be
    frozen (its parameters are captured by reference) and have unit `var`."""

    def __init__(self, model, batch, device, mode="compile"):
        if not bool(torch.all(model.var == 1)):
            raise NotImplementedError("FastReverse assumes the nvp unit base variance")
        self.model, self.batch, self.mode = model, batch, mode
        self.device = torch.device(device)
        self.blocks = [_GraphedBlock(blk, batch, self.device, compile=(mode == "compile"))
                       for blk in model.blocks] if mode != "eager" else None

    def __call__(self, z):
        if z.size(0) != self.batch:
            raise ValueError(f"FastReverse was built for batch {self.batch}, got {z.size(0)}")
        log_q = -0.5 * (z ** 2 + math.log(2 * math.pi)).flatten(1).sum(-1)
        x = z
        for j in reversed(range(len(self.model.blocks))):
            if self.blocks is None:
                x, sum_za = _block_reverse_eager(self.model.blocks[j], x)
            else:
                x, sum_za = _block_reverse_graphed(self.blocks[j], x)
            log_q = log_q - sum_za
        return self.model.unpatchify(x), log_q


# --------------------------------------------------------------------------
# FastReverse2: the same sampling direction with an in-place KV cache
# --------------------------------------------------------------------------
#
# FastReverse passes the whole (B, L, T, C) cache through every token step
# functionally -- copied into the static buffers, rewritten with the new slot,
# cloned out, and in the backward rebuilt from the saved slots and cloned
# again -- about twenty full passes over the cache per token step, so one
# pass through the model costs O(B T^2) memory traffic: measured 2.4 s at
# (B, T) = (3, 256), 7.7 s at (18, 256), and of the order of a minute at
# (18, 1024). Here the cache is a static buffer written in place -- slot i is
# written once, at step i, and never changed, so the cache at the end of a
# block pass restricted to the slots below i IS the cache the step saw -- and
# the backward treats it as a constant: the gradient with respect to the
# cached keys and values (a rank-one update per token step, the only thing
# that touches the whole cache in the backward) is deferred, the attention
# probabilities and score gradients of a chunk of steps are kept as rows and
# applied as one matrix product per chunk. The current token's own key and
# value enter the attention as an explicit (n+1)-th entry, so their gradient
# goes through autograd like the rest of the token network. A token step then
# reads the cache six times (twice in the forward, four times in the
# recompute-and-backward) and writes one slot, and the graphs are captured
# per chunk so that a step only reads the slots below its chunk's end.
# Results agree with FastReverse to floating-point order (see
# `check_fast_reverse2`).


def _decode_attn(q, k, v, Kc, Vc, bias, scale):
    """Attention of one query against a cache and itself. q, k, v (B, H, 1, D):
    the current token's query, key, value; Kc, Vc (B, H, n, D): cache slots
    0..n-1 of which those >= i are masked by bias (1, 1, 1, n). Returns the
    output (B, H, 1, D) and the probabilities (B, H, 1, n + 1), the last entry
    being the token's own."""
    s = torch.matmul(q, Kc.transpose(-1, -2)) * scale + bias
    s_self = (q * k).sum(-1, keepdim=True) * scale
    p = torch.cat([s, s_self], -1).softmax(-1)
    o = torch.matmul(p[..., :-1], Vc) + p[..., -1:] * v
    return o, p


class _DecodeAttn(torch.autograd.Function):
    """`_decode_attn` with the cache as a constant; its backward hands the
    rows needed for the cache gradient to a `_ChunkRecorder`."""

    @staticmethod
    def forward(ctx, q, k, v, Kc, Vc, bias, scale, rec, l):
        o, p = _decode_attn(q, k, v, Kc, Vc, bias, scale)
        ctx.save_for_backward(q, k, v, Kc, Vc, p)
        ctx.scale, ctx.rec, ctx.l = scale, rec, l
        return o

    @staticmethod
    def backward(ctx, g_o):
        q, k, v, Kc, Vc, p = ctx.saved_tensors
        scale = ctx.scale
        g_p = torch.cat([torch.matmul(g_o, Vc.transpose(-1, -2)), (g_o * v).sum(-1, keepdim=True)], -1)
        g_s = p * (g_p - (p * g_p).sum(-1, keepdim=True))
        g_q = (torch.matmul(g_s[..., :-1], Kc) + g_s[..., -1:] * k) * scale
        g_k = g_s[..., -1:] * q * scale
        g_v = p[..., -1:] * g_o
        if ctx.rec is not None:
            ctx.rec.record(ctx.l, p[..., :-1], g_s[..., :-1] * scale, q, g_o)
        return g_q, g_k, g_v, None, None, None, None, None, None


class _ChunkRecorder:
    """Rows of one chunk of token steps, per layer: attention probabilities P
    and scaled score gradients GS over the cache slots (L, B, H, chunk, T), and
    the query Q and output gradient GO (L, B, H, chunk, D). The gradient of
    the cache is  gK[:, :, :n] += GS^T Q,  gV[:, :, :n] += P^T GO  once per
    chunk, and the part of slot i's gradient owed by the steps of its own
    chunk above it is the column i of these rows."""

    def __init__(self, L, B, H, chunk, T, D, device):
        z = lambda *s: torch.zeros(*s, device=device)
        self.P, self.GS = z(L, B, H, chunk, T), z(L, B, H, chunk, T)
        self.Q, self.GO = z(L, B, H, chunk, D), z(L, B, H, chunk, D)
        self.iloc = torch.zeros(1, dtype=torch.long, device=device)       # row of the current step

    def record(self, l, p, gs, q, g_o):
        n = p.shape[-1]
        self.P[l, :, :, :, :n].index_copy_(2, self.iloc, p)
        self.GS[l, :, :, :, :n].index_copy_(2, self.iloc, gs)
        self.Q[l].index_copy_(2, self.iloc, q)
        self.GO[l].index_copy_(2, self.iloc, g_o)

    def slot_grad(self, i):
        """(gK-part, gV-part) of slot i from this chunk's rows, (L, B, H, 1, D)."""
        gs_col = self.GS.index_select(4, i).transpose(-1, -2)                # (L, B, H, 1, chunk)
        p_col = self.P.index_select(4, i).transpose(-1, -2)
        return torch.matmul(gs_col, self.Q), torch.matmul(p_col, self.GO)


class _Shared2:
    """Buffers shared by all blocks of a FastReverse2 (blocks run one after
    the other): the token step's inputs and outputs, the cache gradient of
    the block being back-propagated and the chunk recorder."""

    def __init__(self, L, B, H, chunk, T, D, C, Cin, device):
        z = lambda *s: torch.zeros(*s, device=device)
        self.s_tok, self.s_za, self.s_zb, self.s_gza, self.s_gzb, self.s_gtok = (z(B, Cin) for _ in range(6))
        self.s_i = torch.zeros(1, dtype=torch.long, device=device)
        self.gK, self.gV = z(L, B, H, T, D), z(L, B, H, T, D)
        self.rec = _ChunkRecorder(L, B, H, chunk, T, D, device)
        self.pool = torch.cuda.graph_pool_handle() if device.type == "cuda" else None


def _token_step2(tok, i, n, gb, rec, write):
    """One token of a MetaBlock's sampling direction against gb's in-place
    cache, reading slots < n. tok (B, Cin): recovered token i; i (1,) long.
    `write` writes the token's key and value into slot i (forward pass);
    `rec` records the deferred cache gradients (backward pass). Returns za,
    zb (B, Cin) and the per-layer keys and values (B, H, 1, D). Same
    arithmetic as `_token_step`."""
    B, H, D, C, L = gb.B, gb.H, gb.D, gb.C, gb.L
    w_in, b_in, w_out, b_out = gb.params[:4]
    x = F.linear(tok, w_in, b_in) + gb.pos_table.index_select(0, i)
    bias = gb.bias_table.index_select(0, i)[:, :n][:, None, None, :]
    ks, vs = [], []
    for l in range(L):
        (ln1w, ln1b, wqkv, bqkv, wp, bp, ln2w, ln2b, w1, b1, w2, b2) = \
            gb.params[4 + N_LAYER_PARAMS * l: 4 + N_LAYER_PARAMS * (l + 1)]
        h = F.layer_norm(x, (C,), ln1w, ln1b, 1e-5)
        q, k, v = F.linear(h, wqkv, bqkv).split(C, dim=-1)
        qh, kh, vh = q.view(B, H, 1, D), k.view(B, H, 1, D), v.view(B, H, 1, D)
        o = _DecodeAttn.apply(qh, kh, vh, gb.K[l, :, :, :n], gb.V[l, :, :, :n], bias, gb.scale, rec, l)
        x = x + F.linear(o.reshape(B, C), wp, bp)
        h = F.layer_norm(x, (C,), ln2w, ln2b, 1e-5)
        x = x + F.linear(F.gelu(F.linear(h, w1, b1)), w2, b2)
        if write:
            gb.K[l].index_copy_(2, i, kh)
            gb.V[l].index_copy_(2, i, vh)
        ks.append(kh)
        vs.append(vh)
    za, zb = F.linear(x, w_out, b_out).chunk(2, dim=-1)
    if gb.bound is not None:
        za = gb.bound * torch.tanh(za / gb.bound)
    return za, zb, ks, vs


class _GraphedBlock2:
    """One MetaBlock's in-place cache (L, B, H, T, D) and, per chunk of token
    steps, its two CUDA graphs (forward step; recompute + backward step)."""

    def __init__(self, blk, batch, device, shared, chunk, graph=True):
        self.blk, self.shared = blk, shared
        T, C = blk.pos_embed.shape
        L, Cin = len(blk.attn_blocks), blk.proj_in.in_features
        a0 = blk.attn_blocks[0].attention
        self.H, self.scale = a0.num_heads, a0.sqrt_scale ** 2
        self.D = C // self.H
        self.bound = float(blk.log_scale_bound) if hasattr(blk, "log_scale_bound") else None
        self.B, self.T, self.L, self.C, self.Cin = batch, T, L, C, Cin
        self.params = _block_params(blk)
        with torch.no_grad():
            self.pos_table = blk.permutation(blk.pos_embed.detach(), dim=0).contiguous()
        self.bias_table = torch.triu(torch.full((T, T), -1e30, device=device), 0)   # row i: masked from slot i on
        self.K = torch.zeros(L, batch, self.H, T, self.D, device=device)
        self.V = torch.zeros_like(self.K)
        self.chunks = [(c0, min(c0 + chunk, T)) for c0 in range(0, T - 1, chunk)]   # token steps are 0..T-2
        self.pass_id = 0
        self.graphs = None
        if graph:
            self.graphs = []
            s = torch.cuda.Stream(device)
            s.wait_stream(torch.cuda.current_stream(device))
            for c0, c1 in self.chunks:
                with torch.cuda.stream(s):                                  # warm-up
                    self._fwd(c1)
                    self._bwd(c1)
                torch.cuda.current_stream(device).wait_stream(s)
                torch.cuda.synchronize(device)
                g_fwd, g_bwd = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
                with torch.cuda.graph(g_fwd, pool=shared.pool):
                    self._fwd(c1)
                with torch.cuda.graph(g_bwd, pool=shared.pool):
                    self._bwd(c1)
                self.graphs.append((g_fwd, g_bwd))
            torch.cuda.synchronize(device)

    def _fwd(self, n):
        sh = self.shared
        with torch.no_grad():
            za, zb, _, _ = _token_step2(sh.s_tok, sh.s_i, n, self, None, True)
            sh.s_za.copy_(za)
            sh.s_zb.copy_(zb)

    def _bwd(self, n):
        sh, rec = self.shared, self.shared.rec
        gk, gv = rec.slot_grad(sh.s_i)                                        # this chunk's steps above i
        gk = gk + sh.gK.index_select(3, sh.s_i)                               # plus the flushed chunks above
        gv = gv + sh.gV.index_select(3, sh.s_i)
        tok = sh.s_tok.detach().requires_grad_(True)
        with torch.enable_grad():
            za, zb, ks, vs = _token_step2(tok, sh.s_i, n, self, rec, False)
            (g_tok,) = torch.autograd.grad((za, zb, *ks, *vs), (tok,),
                                           grad_outputs=(sh.s_gza, sh.s_gzb, *gk.unbind(0), *gv.unbind(0)))
        sh.s_gtok.copy_(g_tok)

    def fwd_step(self, c, n):
        if self.graphs is None:
            self._fwd(n)
        else:
            self.graphs[c][0].replay()

    def bwd_step(self, c, n):
        if self.graphs is None:
            self._bwd(n)
        else:
            self.graphs[c][1].replay()

    def flush(self, n):
        """Apply the recorder's rows to the cache gradient of slots < n."""
        sh, rec = self.shared, self.shared.rec
        B, H, D = self.B, self.H, self.D
        for l in range(self.L):
            GS = rec.GS[l, :, :, :, :n].reshape(B * H, -1, n)
            P = rec.P[l, :, :, :, :n].reshape(B * H, -1, n)
            sh.gK[l, :, :, :n].view(B * H, n, D).baddbmm_(GS.transpose(1, 2), rec.Q[l].view(B * H, -1, D))
            sh.gV[l, :, :, :n].view(B * H, n, D).baddbmm_(P.transpose(1, 2), rec.GO[l].view(B * H, -1, D))


class _BlockReverse2(torch.autograd.Function):
    """Sampling direction of one MetaBlock on the permuted tokens x (B, T,
    Cin) -> (recovered tokens, per-example sum of za), with the block's cache
    filled in the forward and consumed by the backward."""

    @staticmethod
    def forward(ctx, x, gb):
        sh = gb.shared
        B, T, _ = x.shape
        gb.K.zero_()
        gb.V.zero_()
        toks, zas, zbs = [x[:, 0]], [], []
        for c, (c0, c1) in enumerate(gb.chunks):
            for i in range(c0, min(c1, T - 1)):
                sh.s_tok.copy_(toks[i])
                sh.s_i.fill_(i)
                gb.fwd_step(c, c1)
                za, zb = sh.s_za.clone(), sh.s_zb.clone()
                toks.append(x[:, i + 1] * za.exp() + zb)
                zas.append(za)
                zbs.append(zb)
        toks, za, zb = torch.stack(toks, 1), torch.stack(zas, 1), torch.stack(zbs, 1)
        ctx.save_for_backward(x, toks, za, zb)
        ctx.gb, ctx.pass_id = gb, gb.pass_id
        return toks, za.sum((1, 2))

    @staticmethod
    def backward(ctx, g_toks, g_sumza):
        gb = ctx.gb
        if gb.pass_id != ctx.pass_id:
            raise RuntimeError("FastReverse2: the block's cache was overwritten by a later forward pass; "
                               "call backward before the next forward")
        sh, rec = gb.shared, gb.shared.rec
        x, toks, za, zb = ctx.saved_tensors
        B, T, _ = x.shape
        if g_sumza is None:                          # the objective did not use log_q
            g_sumza = torch.zeros(B, device=x.device)
        ea = za.exp()
        g_x = torch.zeros_like(x)
        g_tok = g_toks.clone()                       # d/d tok_i, accumulated as the steps above i are processed
        sh.gK.zero_()
        sh.gV.zero_()
        for c in reversed(range(len(gb.chunks))):
            c0, c1 = gb.chunks[c]
            rec.P[..., :c1].zero_()
            rec.GS[..., :c1].zero_()
            for i in reversed(range(c0, min(c1, T - 1))):
                g_next = g_tok[:, i + 1]                                     # tok_{i+1} = x_{i+1} e^{za_i} + zb_i
                g_x[:, i + 1] = g_next * ea[:, i]
                sh.s_gza.copy_(g_next * x[:, i + 1] * ea[:, i] + g_sumza[:, None])
                sh.s_gzb.copy_(g_next)
                sh.s_tok.copy_(toks[:, i])
                sh.s_i.fill_(i)
                rec.iloc.fill_(i - c0)
                gb.bwd_step(c, c1)
                g_tok[:, i] += sh.s_gtok
            gb.flush(c1)
        g_x[:, 0] = g_tok[:, 0]
        return g_x, None


class FastReverse2:
    """x = g(z) for a TarFlow `model`, differentiable w.r.t. z, with the
    in-place cache above; the interface of FastReverse. `chunk` is the number
    of token steps per captured graph (and per deferred cache-gradient
    update); `mode` is 'graph' (CUDA graphs) or 'eager' (no graphs; also the
    CPU reference). A pass must be back-propagated before the next forward
    pass (the caches are reused)."""

    def __init__(self, model, batch, device, chunk=128, mode="graph"):
        if not bool(torch.all(model.var == 1)):
            raise NotImplementedError("FastReverse2 assumes the nvp unit base variance")
        self.model, self.batch, self.mode, self.chunk = model, batch, mode, chunk
        self.device = torch.device(device)
        blk = model.blocks[0]
        T, C = blk.pos_embed.shape
        L, Cin, H = len(blk.attn_blocks), blk.proj_in.in_features, blk.attn_blocks[0].attention.num_heads
        self.shared = _Shared2(L, batch, H, min(chunk, T), T, C // H, C, Cin, self.device)
        self.blocks = [_GraphedBlock2(b, batch, self.device, self.shared, chunk, graph=(mode == "graph"))
                       for b in model.blocks]
        self.pass_id = 0

    def __call__(self, z):
        if z.size(0) != self.batch:
            raise ValueError(f"FastReverse2 was built for batch {self.batch}, got {z.size(0)}")
        self.pass_id += 1
        log_q = -0.5 * (z ** 2 + math.log(2 * math.pi)).flatten(1).sum(-1)
        x = z
        for j in reversed(range(len(self.model.blocks))):
            gb = self.blocks[j]
            gb.pass_id = self.pass_id
            toks, sum_za = _BlockReverse2.apply(gb.blk.permutation(x), gb)
            x = gb.blk.permutation(toks, inverse=True)
            log_q = log_q - sum_za
        return self.model.unpatchify(x), log_q

    def reset(self):
        """Zero the persistent buffers. A backward pass through a non-finite
        objective leaves NaNs behind in them (0 x NaN in the deferred cache
        gradient) and every later gradient is NaN while the forward stays
        correct; a forward pass that is not back-propagated is harmless.
        Zeroing restores the gradient (verified at 32 px). Callers with an
        objective that can overflow should not back-propagate rows whose
        value is not finite and call this if the gradient comes back
        non-finite anyway."""
        sh = self.shared
        for t in (sh.s_tok, sh.s_za, sh.s_zb, sh.s_gza, sh.s_gzb, sh.s_gtok, sh.gK, sh.gV,
                  sh.rec.P, sh.rec.GS, sh.rec.Q, sh.rec.GO):
            t.zero_()
        for gb in self.blocks:
            gb.K.zero_()
            gb.V.zero_()


def check_fast_reverse2(model, batch, device, chunk, mode="graph", ref_mode="eager", seed=0):
    """Compare FastReverse2 with FastReverse on random z: max |dx|, |dlog_q|
    and |dgrad| (of a random linear functional of x and log_q w.r.t. z),
    relative to the reference's scale."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    blk = model.blocks[0]
    T, Cin = blk.pos_embed.shape[0], blk.proj_in.in_features
    z = torch.randn(batch, T, Cin, generator=g).to(device)                 # tokens, as FastReverse takes them
    w = None
    out = {}
    # built one at a time: the eager reference keeps every token step's graph (O(T^2) memory),
    # so at T = 256 use ref_mode="compile" (the mode the galleries ran with)
    for name, build in (("ref", lambda: FastReverse(model, batch, device, mode=ref_mode)),
                        ("new", lambda: FastReverse2(model, batch, device, chunk=chunk, mode=mode))):
        fr = build()
        zz = z.clone().requires_grad_(True)
        x, log_q = fr(zz)
        if w is None:
            w = torch.randn(x.shape, generator=g).to(device)
        ((x * w).sum() + log_q.sum()).backward()
        out[name] = (x.detach(), log_q.detach(), zz.grad.detach())
        del fr, x, log_q, zz
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()
    rel = lambda a, b: float((a - b).abs().max() / b.abs().max())
    return dict(dx=rel(out["new"][0], out["ref"][0]), dlogq=rel(out["new"][1], out["ref"][1]),
                dgrad=rel(out["new"][2], out["ref"][2]))


# ==========================================================================
# Family 2: InvertibleModule -- general invertible maps with logdet
# ==========================================================================


class InvertibleModule(nn.Module):
    """Contract: forward(x) -> (y, logdet[B]); inverse(y) -> x."""

    def forward(self, x):
        raise NotImplementedError

    def inverse(self, y):
        raise NotImplementedError


def _per_sample(t, batch):
    """Broadcast a scalar logdet to per-sample shape (B,)."""
    if t.dim() == 0:
        return t.expand(batch)
    return t


class InvertibleDiagonal(InvertibleModule):
    """y = s * x + b elementwise over feature dims (ActNorm without data init)."""

    def __init__(self, shape):
        super().__init__()
        self.log_s = nn.Parameter(0.05 * torch.randn(shape))
        self.b = nn.Parameter(torch.zeros(shape))

    def forward(self, x):
        y = x * self.log_s.exp() + self.b
        return y, _per_sample(self.log_s.sum(), x.shape[0])

    def inverse(self, y):
        return (y - self.b) * (-self.log_s).exp()


class ActNorm2d(InvertibleModule):
    """Per-channel affine for images (B,C,H,W); Glow-style actnorm."""

    def __init__(self, channels):
        super().__init__()
        self.log_s = nn.Parameter(0.05 * torch.randn(channels))
        self.b = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        s = self.log_s.view(1, -1, 1, 1)
        y = x * s.exp() + self.b.view(1, -1, 1, 1)
        ld = self.log_s.sum() * x.shape[2] * x.shape[3]
        return y, _per_sample(ld, x.shape[0])

    def inverse(self, y):
        s = self.log_s.view(1, -1, 1, 1)
        return (y - self.b.view(1, -1, 1, 1)) * (-s).exp()


class PLULinear(InvertibleModule):
    """Dense invertible linear via PLU parameterization (Glow-style).
    y = P L U x with L unit-lower, U upper with parameterized nonzero diag."""

    def __init__(self, dim, seed: int = 0):
        super().__init__()
        self.dim = dim
        w = random_orthogonal_matrix(dim, seed).to(torch.float32)
        p, l, u = torch.linalg.lu(w)
        self.register_buffer("P", p)
        self.l_raw = nn.Parameter(l)
        s = torch.diagonal(u)
        self.register_buffer("sign_s", torch.sign(s))
        self.log_s = nn.Parameter(s.abs().log())
        self.u_raw = nn.Parameter(torch.triu(u, 1))
        self.register_buffer("l_mask", torch.tril(torch.ones(dim, dim), -1))
        self.register_buffer("eye", torch.eye(dim))

    def _lu(self):
        l = self.l_raw * self.l_mask + self.eye
        u = torch.triu(self.u_raw, 1) + torch.diag(self.sign_s * self.log_s.exp())
        return l, u

    def forward(self, x):
        l, u = self._lu()
        y = x @ (self.P @ l @ u).t()
        return y, _per_sample(self.log_s.sum(), x.shape[0])

    def inverse(self, y):
        l, u = self._lu()
        z = y @ torch.linalg.inv(self.P).t()
        z = torch.linalg.solve_triangular(l, z.t(), upper=False).t()
        return torch.linalg.solve_triangular(u, z.t(), upper=True).t()


class Invertible1x1Conv2d(InvertibleModule):
    """Glow's invertible 1x1 convolution over channels of (B,C,H,W)."""

    def __init__(self, channels, seed: int = 0):
        super().__init__()
        self.lin = PLULinear(channels, seed)

    def forward(self, x):
        B, C, H, W = x.shape
        y, ld = self.lin(x.permute(0, 2, 3, 1).reshape(-1, C))
        y = y.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return y, _per_sample(self.lin.log_s.sum() * H * W, B)

    def inverse(self, y):
        B, C, H, W = y.shape
        x = self.lin.inverse(y.permute(0, 2, 3, 1).reshape(-1, C))
        return x.reshape(B, H, W, C).permute(0, 3, 1, 2)


class CircularConv2d(InvertibleModule):
    """Invertible periodic (circular) depthwise convolution, decoupled in the
    frequency domain (Hoogeboom et al. 2019, 'periodic convolutions').
    Per channel: y = ifft2(fft2(x) * H), H = fft2(kernel); invertible iff
    all |H| > 0. logdet = sum log|H|. Initialized near identity."""

    def __init__(self, channels, kernel_size=3, img_size=16, init_noise=0.01, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        k = init_noise * torch.randn(channels, kernel_size, kernel_size, generator=g)
        k[:, 0, 0] += 1.0  # delta at origin -> identity map
        self.kernel = nn.Parameter(k)
        self.img_size = img_size

    def _H(self, H, W, device, dtype):
        pad = torch.zeros(self.kernel.shape[0], H, W, device=device, dtype=dtype)
        kh, kw = self.kernel.shape[1:]
        pad[:, :kh, :kw] = self.kernel.to(dtype)
        return torch.fft.fft2(pad)

    def forward(self, x):
        B, C, H, W = x.shape
        Hf = self._H(H, W, x.device, x.dtype)
        y = torch.fft.ifft2(torch.fft.fft2(x) * Hf.unsqueeze(0)).real
        ld = Hf.abs().clamp_min(1e-12).log().sum()
        return y, _per_sample(ld, B)

    def inverse(self, y):
        B, C, H, W = y.shape
        Hf = self._H(H, W, y.device, y.dtype)
        return torch.fft.ifft2(torch.fft.fft2(y) / Hf.unsqueeze(0)).real


class CircularConv1d(InvertibleModule):
    """Invertible circular convolution along the last dim of (B, T):
    the 'invertible Fourier filter'. Same construction as CircularConv2d."""

    def __init__(self, length, kernel_size=5, init_noise=0.01, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        k = init_noise * torch.randn(kernel_size, generator=g)
        k[0] += 1.0
        self.kernel = nn.Parameter(k)
        self.length = length

    def _H(self, T, device):
        pad = torch.zeros(T, device=device)
        pad[: self.kernel.shape[0]] = self.kernel
        return torch.fft.fft(pad)

    def forward(self, x):
        Hf = self._H(x.shape[-1], x.device)
        y = torch.fft.ifft(torch.fft.fft(x) * Hf).real
        ld = Hf.abs().clamp_min(1e-12).log().sum()
        return y, _per_sample(ld, x.shape[0])

    def inverse(self, y):
        Hf = self._H(y.shape[-1], y.device)
        return torch.fft.ifft(torch.fft.fft(y) / Hf).real


class MonotonicCubic(InvertibleModule):
    """Elementwise y = x + a x^3 with a >= 0 (strictly increasing).
    Closed-form inverse via Cardano (single real root since disc > 0).
    logdet = sum log(1 + 3 a x^2)."""

    def __init__(self, init_a=0.1):
        super().__init__()
        self.raw_a = nn.Parameter(torch.tensor(float(np.log(np.expm1(init_a)))))

    @property
    def a(self):
        return F.softplus(self.raw_a)

    def forward(self, x):
        a = self.a
        y = x + a * x**3
        ld = (1 + 3 * a * x**2).log().flatten(1).sum(-1)
        return y, ld

    def inverse(self, y):
        a = self.a.double()
        yd = y.double()
        p = 1.0 / a
        q = -yd / a
        disc = (q / 2) ** 2 + (p / 3) ** 3  # > 0 always
        r = disc.sqrt()
        x = torch.sign(-q / 2 + r) * (-q / 2 + r).abs().pow(1 / 3) \
            + torch.sign(-q / 2 - r) * (-q / 2 - r).abs().pow(1 / 3)
        return x.to(y.dtype)


class MonotonicPolynomial(InvertibleModule):
    """Elementwise odd polynomial y = c x + sum_j a_j x^(2j+3), c>0, a_j>=0
    (derivative strictly positive -> invertible; SOS-flow-style).
    Inverse via bisection + Newton refinement. logdet = sum log f'(x)."""

    def __init__(self, degrees=(3, 5), init=0.05):
        super().__init__()
        self.degrees = tuple(degrees)
        self.raw_c = nn.Parameter(torch.tensor(float(np.log(np.expm1(1.0)))))
        self.raw_a = nn.Parameter(torch.full((len(self.degrees),),
                                             float(np.log(np.expm1(init)))))

    def _coefs(self):
        return F.softplus(self.raw_c), F.softplus(self.raw_a)

    def _f(self, x):
        c, a = self._coefs()
        y = c * x
        for j, d in enumerate(self.degrees):
            y = y + a[j] * x**d
        return y

    def _fp(self, x):
        c, a = self._coefs()
        d1 = torch.ones_like(x) * c
        for j, d in enumerate(self.degrees):
            d1 = d1 + a[j] * d * x ** (d - 1)
        return d1

    def forward(self, x):
        return self._f(x), self._fp(x).log().flatten(1).sum(-1)

    @torch.no_grad()
    def inverse(self, y, iters=60):
        lo = torch.full_like(y, -1.0)
        hi = torch.ones_like(y)
        while (self._f(lo) > y).any():
            lo = torch.where(self._f(lo) > y, lo * 2, lo)
        while (self._f(hi) < y).any():
            hi = torch.where(self._f(hi) < y, hi * 2, hi)
        for _ in range(iters):  # bisection (robust)
            mid = 0.5 * (lo + hi)
            below = self._f(mid) < y
            lo = torch.where(below, mid, lo)
            hi = torch.where(below, hi, mid)
        x = 0.5 * (lo + hi)
        for _ in range(3):  # Newton polish
            x = x - (self._f(x) - y) / self._fp(x).clamp_min(1e-12)
        return x


class InvertibleLeakyReLU(InvertibleModule):
    def __init__(self, slope=0.5):
        super().__init__()
        self.slope = slope

    def forward(self, x):
        y = torch.where(x >= 0, x, self.slope * x)
        ld = (x < 0).flatten(1).sum(-1) * math.log(self.slope)
        return y, ld.to(x.dtype)

    def inverse(self, y):
        return torch.where(y >= 0, y, y / self.slope)


class Logit(InvertibleModule):
    """Data-space transform (0,1) -> R: y = logit(alpha + (1-2 alpha) x)."""

    def __init__(self, alpha=0.05):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        s = self.alpha + (1 - 2 * self.alpha) * x
        y = s.log() - (1 - s).log()
        ld = (math.log(1 - 2 * self.alpha) - s.log() - (1 - s).log()).flatten(1).sum(-1)
        return y, ld

    def inverse(self, y):
        s = torch.sigmoid(y)
        return (s - self.alpha) / (1 - 2 * self.alpha)


class RationalQuadraticSpline(InvertibleModule):
    """Elementwise monotonic rational-quadratic spline (Durkan et al. 2019),
    K bins on [-B, B], identity tails; unconditional learnable knots shared
    across all elements. Analytic inverse (quadratic root)."""

    def __init__(self, num_bins=8, bound=4.0, min_size=1e-3):
        super().__init__()
        self.K, self.B, self.eps = num_bins, bound, min_size
        self.w_raw = nn.Parameter(torch.zeros(num_bins))
        self.h_raw = nn.Parameter(torch.zeros(num_bins))
        self.d_raw = nn.Parameter(torch.zeros(num_bins - 1))

    def _knots(self):
        w = F.softmax(self.w_raw, -1) * (1 - self.K * self.eps) + self.eps
        h = F.softmax(self.h_raw, -1) * (1 - self.K * self.eps) + self.eps
        xk = F.pad(torch.cumsum(w, -1), (1, 0)) * 2 * self.B - self.B
        yk = F.pad(torch.cumsum(h, -1), (1, 0)) * 2 * self.B - self.B
        d = F.pad(F.softplus(self.d_raw) + self.eps, (1, 1), value=1.0)
        return xk, yk, d

    def _search(self, v, knots):
        return (torch.searchsorted(knots[1:-1].contiguous(), v.contiguous().detach())
                ).clamp(0, self.K - 1)

    def forward(self, x):
        xk, yk, d = self._knots()
        inside = (x > -self.B) & (x < self.B)
        xc = x.clamp(-self.B + 1e-6, self.B - 1e-6)
        i = self._search(xc, xk)
        x0, x1 = xk[i], xk[i + 1]
        y0, y1 = yk[i], yk[i + 1]
        d0, d1 = d[i], d[i + 1]
        w = x1 - x0
        s = (y1 - y0) / w
        t = (xc - x0) / w
        num = (y1 - y0) * (s * t**2 + d0 * t * (1 - t))
        den = s + (d1 + d0 - 2 * s) * t * (1 - t)
        y = torch.where(inside, y0 + num / den, x)
        dnum = s**2 * (d1 * t**2 + 2 * s * t * (1 - t) + d0 * (1 - t) ** 2)
        deriv = torch.where(inside, dnum / den**2, torch.ones_like(x))
        return y, deriv.clamp_min(1e-12).log().flatten(1).sum(-1)

    def inverse(self, y):
        xk, yk, d = self._knots()
        inside = (y > -self.B) & (y < self.B)
        yc = y.clamp(-self.B + 1e-6, self.B - 1e-6)
        i = self._search(yc, yk)
        x0, x1 = xk[i], xk[i + 1]
        y0, y1 = yk[i], yk[i + 1]
        d0, d1 = d[i], d[i + 1]
        w = x1 - x0
        s = (y1 - y0) / w
        r = (yc - y0)
        a = (y1 - y0) * (s - d0) + r * (d1 + d0 - 2 * s)
        b = (y1 - y0) * d0 - r * (d1 + d0 - 2 * s)
        c = -s * r
        t = 2 * c / (-b - (b**2 - 4 * a * c).clamp_min(0).sqrt())
        return torch.where(inside, x0 + t * w, y)


class SpectralFloorLinear(InvertibleModule):
    """Eigenvalue-floored low-rank symmetric PSD map (M. Tivnan, this work).

    A = U diag(lam) U^T + lam0 (I - U U^T), U in R^{D x k} orthonormal,
    lam_i = lam0 + softplus(theta_i) >= lam0 > 0.

    A full-rank invertible stand-in for a rank-k eigendecomposition: instead of
    truncating unretained eigenvalues to zero (singular), they are all set to
    the floor lam0, which is <= every retained eigenvalue by construction.
    Analytic inverse A^-1 = U diag(1/lam) U^T + (1/lam0)(I - U U^T);
    logdet = sum log lam_i + (D - k) log lam0."""

    def __init__(self, dim, rank, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.dim, self.rank = dim, rank
        self.w = nn.Parameter(torch.randn(dim, rank, generator=g) / math.sqrt(dim))
        self.raw_lam0 = nn.Parameter(torch.tensor(float(np.log(np.expm1(1.0)))))
        self.theta = nn.Parameter(0.1 * torch.randn(rank, generator=g))

    def _factors(self):
        u, _ = torch.linalg.qr(self.w)  # orthonormal columns
        lam0 = F.softplus(self.raw_lam0) + 1e-6
        lam = lam0 + F.softplus(self.theta)
        return u, lam0, lam

    def forward(self, x):
        u, lam0, lam = self._factors()
        proj = x @ u  # (B,k)
        y = (proj * lam) @ u.t() + lam0 * (x - proj @ u.t())
        ld = lam.log().sum() + (self.dim - self.rank) * lam0.log()
        return y, _per_sample(ld, x.shape[0])

    def inverse(self, y):
        u, lam0, lam = self._factors()
        proj = y @ u
        return (proj / lam) @ u.t() + (y - proj @ u.t()) / lam0


class WoodburyLinear(InvertibleModule):
    """y = (diag(d) + U V^T) x with d > 0; inverse via the Woodbury identity,
    logdet via the matrix determinant lemma (Lu & Huang, NeurIPS 2020)."""

    def __init__(self, dim, rank, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.raw_d = nn.Parameter(torch.zeros(dim))
        self.u = nn.Parameter(0.1 * torch.randn(dim, rank, generator=g) / math.sqrt(dim))
        self.v = nn.Parameter(0.1 * torch.randn(dim, rank, generator=g) / math.sqrt(dim))

    def _parts(self):
        d = F.softplus(self.raw_d) + 1e-4
        k = self.u.shape[1]
        cap = torch.eye(k, device=d.device) + self.v.t() @ (self.u / d.unsqueeze(1))
        return d, cap

    def forward(self, x):
        d, cap = self._parts()
        y = x * d + (x @ self.v) @ self.u.t()  # row form of (D + U V^T) x
        sign, logabs = torch.linalg.slogdet(cap)
        ld = d.log().sum() + logabs  # sign must be +1 (checked in verification)
        return y, _per_sample(ld, x.shape[0])

    def inverse(self, y):
        d, cap = self._parts()
        z = y / d
        w = torch.linalg.solve(cap, (z @ self.v).t()).t()  # cap^-1 V^T D^-1 y
        return z - (w @ self.u.t()) / d


class AffineCoupling(InvertibleModule):
    """Classic affine coupling on flat features (RealNVP-style)."""

    def __init__(self, dim, hidden=64, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.d1 = dim // 2
        self.net = nn.Sequential(
            nn.Linear(self.d1, hidden), nn.GELU(),
            nn.Linear(hidden, 2 * (dim - self.d1)),
        )
        self.net[-1].weight.data.zero_()
        self.net[-1].bias.data.zero_()

    def forward(self, x):
        x1, x2 = x[:, : self.d1], x[:, self.d1:]
        s, t = self.net(x1).chunk(2, -1)
        s = 0.5 * torch.tanh(s)  # bounded scale for stability
        y2 = x2 * s.exp() + t
        return torch.cat([x1, y2], -1), s.sum(-1)

    def inverse(self, y):
        y1, y2 = y[:, : self.d1], y[:, self.d1:]
        s, t = self.net(y1).chunk(2, -1)
        s = 0.5 * torch.tanh(s)
        return torch.cat([y1, (y2 - t) * (-s).exp()], -1)


class SeqOrthogonalAsInvertible(InvertibleModule):
    """Adapter: verify any SeqTransform under the InvertibleModule contract
    (applied along the last dim of (B,T); logdet identically zero)."""

    def __init__(self, seq_transform):
        super().__init__()
        self.t = seq_transform

    def forward(self, x):
        return self.t(x, dim=-1), torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

    def inverse(self, y):
        return self.t(y, dim=-1, inverse=True)


# ==========================================================================
# Registry for the step-3 verification harness
# ==========================================================================
# Each entry: factory(dim_flat, img_shape, seq_len) -> (module, sample_input_fn)
# kind: 'flat' (B,D) | 'image' (B,C,H,W) | 'seq' orthogonal along last dim


def build_registry(D=64, img=(2, 8, 8), T=64, spline_bound=4.0):
    C, H, W = img

    def flat(b, scale=1.0):
        return scale * torch.randn(b, D)

    def img_in(b):
        return torch.randn(b, C, H, W)

    def unit(b):
        return 0.05 + 0.9 * torch.rand(b, D)

    reg = {}
    # --- orthogonal sequence transforms (verified through the adapter) ---
    for name, cls in SEQ_REGISTRY.items():
        def make(cls=cls):
            m = cls(T)
            return SeqOrthogonalAsInvertible(m), (lambda b: torch.randn(b, T)), m
        reg[f"seq_{name}"] = dict(make=make, kind="seq")
    # --- linear invertible ---
    reg["invertible_diagonal"] = dict(make=lambda: (InvertibleDiagonal(D), flat, None), kind="flat")
    reg["actnorm2d"] = dict(make=lambda: (ActNorm2d(C), img_in, None), kind="image")
    reg["plu_linear"] = dict(make=lambda: (PLULinear(D), flat, None), kind="flat")
    reg["invertible_1x1_conv"] = dict(make=lambda: (Invertible1x1Conv2d(C), img_in, None), kind="image")
    reg["circular_conv2d"] = dict(make=lambda: (CircularConv2d(C, 3, H), img_in, None), kind="image")
    reg["circular_conv1d_fourier_filter"] = dict(
        make=lambda: (CircularConv1d(D), flat, None), kind="flat")
    reg["spectral_floor_linear"] = dict(
        make=lambda: (SpectralFloorLinear(D, rank=8), flat, None), kind="flat")
    reg["woodbury_linear"] = dict(make=lambda: (WoodburyLinear(D, rank=8), flat, None), kind="flat")
    # --- elementwise nonlinear ---
    reg["monotonic_cubic"] = dict(make=lambda: (MonotonicCubic(0.2), flat, None), kind="flat")
    reg["monotonic_polynomial"] = dict(
        make=lambda: (MonotonicPolynomial((3, 5), 0.1), flat, None), kind="flat")
    reg["invertible_leaky_relu"] = dict(
        make=lambda: (InvertibleLeakyReLU(0.5), flat, None), kind="flat")
    reg["logit"] = dict(make=lambda: (Logit(0.05), unit, None), kind="flat")
    reg["rq_spline"] = dict(
        make=lambda: (RationalQuadraticSpline(8, spline_bound), lambda b: 2.0 * torch.randn(b, D), None),
        kind="flat")
    # --- coupling ---
    reg["affine_coupling"] = dict(make=lambda: (AffineCoupling(D), flat, None), kind="flat")
    return reg
