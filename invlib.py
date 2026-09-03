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
