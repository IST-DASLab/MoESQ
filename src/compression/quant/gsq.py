import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseQuant

# GSQ's 2-bit grid, verbatim from the paper (arXiv:2604.18556, Sec 3.2, "The 2-bit
# case"): G_2 = {-2, -1, 0, 1}. It is deliberately skewed toward negative values;
# the paper compensates by letting the shared scale s take NEGATIVE values, "thereby
# removing any inherent bias toward either side". So `scale` here is unconstrained --
# do not clamp it positive.
GSQ_GRIDS = {
    2: (-2.0, -1.0, 0.0, 1.0),
    #   ternary (1.58-bit) written as a 3-point grid; the paper parameterizes it as
    #   mask x sign instead, which is cheaper, but the grid form is equivalent here.
    1: (-1.0, 0.0, 1.0),
}


class GSQQuant(BaseQuant):
    """GSQ: Gumbel-Softmax scalar quantization (arXiv:2604.18556).

    Unlike NVFP4Quant, the grid assignment is **learned, not derived from the
    weight**: each coordinate owns |G| trainable logits and the assignment is a
    Gumbel-Softmax sample over the grid. Per-group scales are trained jointly.
    Consequently `fake_quantize` IGNORES the incoming effective_weight -- the
    reconstruction is s * q and nothing else, exactly as in the paper. There is no
    continuous master weight in GSQ, so build this with learn_weights=False.

    Because the sampler needs the annealed temperature and logit scale that
    ExpertCompressor owns, this sets needs_anneal=True and takes them as forward
    arguments; NVFP4Quant is untouched and keeps its 1-argument signature.

    Memory: |G| logits per weight. At 2 bits that is 4x the weight count, which is
    why the paper notes logits cost "2-5x that of the weights being quantized" --
    keep logits_dtype at bfloat16 unless a run demonstrably needs fp32.
    """

    needs_anneal = True

    def __init__(self, weight_shape, groupsize, device, dtype, wbits=2):
        super().__init__()
        if wbits not in GSQ_GRIDS:
            raise ValueError(
                f"GSQQuant has no grid for wbits={wbits}. Available: {sorted(GSQ_GRIDS)}."
            )
        rows, cols = weight_shape
        if cols % groupsize != 0:
            raise ValueError(
                f"GSQQuant requires weight width divisible by groupsize={groupsize}, got {cols}."
            )
        self.weight_shape = (rows, cols)
        self.groupsize = groupsize
        self.n_groups = cols // groupsize
        self.wbits = wbits
        self.device = device
        self.dtype = dtype
        grid = torch.tensor(GSQ_GRIDS[wbits], device=device, dtype=torch.float32)
        self.register_buffer("grid", grid)
        self.register_buffer("idx", torch.arange(cols, device=device) // groupsize)

    def init_from_weight(self, weight, std, strength, logits_dtype, init_scales=None):
        """Warm-start scale + logits from the GPTQ solution.

        The paper initializes logits so the induced distribution is peaked on the
        grid point GPTQ chose (Eq 3), then injects isotropic Gaussian noise (Eq 4)
        to avoid getting stuck. We reuse this repo's existing convention for that,
        `std * (randn + onehot * strength)`, so std/strength mean the same thing
        here as they do for the sparsity mask logits.
        """
        W = weight.detach().to(device=self.device, dtype=torch.float32)
        rows, cols = self.weight_shape
        if tuple(W.shape) != self.weight_shape:
            raise ValueError(f"GSQQuant expected weight {self.weight_shape}, got {tuple(W.shape)}.")

        if init_scales is not None and tuple(init_scales.shape) == (rows, self.n_groups):
            s = init_scales.detach().to(device=self.device, dtype=torch.float32).clone()
        else:
            # grid spans [-2, +1] at 2 bits, so |w|max / 2 puts the widest weight on
            # the outermost negative grid point rather than clipping it.
            blocks = W.reshape(rows, self.n_groups, self.groupsize)
            s = blocks.abs().amax(dim=-1) / float(max(abs(g) for g in GSQ_GRIDS[self.wbits]))
        s = torch.where(s.abs() < 1e-8, torch.full_like(s, 1e-8), s)
        self.scale = nn.Parameter(s)

        # nearest grid point under the initialized scale == the GPTQ assignment
        q = W / s[:, self.idx]
        nearest = (q.unsqueeze(-1) - self.grid).abs().argmin(dim=-1)
        onehot = F.one_hot(nearest, num_classes=self.grid.numel()).float()
        logits = std * (torch.randn_like(onehot) + onehot * strength)
        self.logits = nn.Parameter(logits.to(logits_dtype))
        return self

    def _soft_assign(self, temperature, scale):
        eps = 1e-8
        logits = self.logits.float() * float(scale)
        u = torch.rand_like(logits)
        noise = -torch.log(-torch.log(u + eps) + eps)
        return F.softmax((logits + noise) / float(temperature), dim=-1)

    def fake_quantize(self, effective_weight, temperature=1.0, scale=1.0):
        # effective_weight is deliberately unused: in GSQ the reconstruction is
        # s * q with q from the learned logits. See the class docstring.
        p = self._soft_assign(temperature, scale)
        q_soft = (p * self.grid).sum(dim=-1)
        return q_soft * self.scale[:, self.idx]

    def quantize_hard(self, effective_weight):
        idx = torch.argmax(self.logits.float(), dim=-1)
        q = self.grid[idx]
        s_per_col = self.scale.detach()[:, self.idx]
        # scales are returned per group; there is no second-level global scale in
        # GSQ (that is an NVFP4 artifact), so global_scale is None.
        return q * s_per_col, self.scale.detach(), None

    def hard_indices(self):
        """Grid indices for serialization: uint2 codes in [0, |G|)."""
        return torch.argmax(self.logits.float(), dim=-1).to(torch.uint8)


# ---------------------------------------------------------------------------
# Humming-compatible uint2 packing.
#
# Layout derived from Humming's own kernel (humming/include/humming/kernel/
# pack_weight.cuh, common_pack_weight<2>), not guessed:
#
#   for i in 0..31:  index = i*2;  word = index/32;  offset = index%32
#                    out[word] |= (code_i & 0x3) << offset
#
# Each thread handles 32 consecutive codes and writes 2 consecutive int32 words
# (in_offset = tid*32, out_offset = tid*2), so the mapping is order-preserving:
# codes 0..15 -> word 0, codes 16..31 -> word 1, low bits first. Equivalently,
# chunk the flat code array into 16s and pack each chunk little-endian.
#
# Codes are UNSIGNED grid indices, not signed values: humming's dtype is
# uint2 = IntegerType(is_signed=False, num_bits=2), and the published checkpoint
# (ISTA-DASLab/Kimi-K2.5-2Bit-GSQ) declares codebook {-2,-1,0,+1} x scale, whose
# order matches GSQ_GRIDS[2]. So code == index into the grid.
CODES_PER_INT32 = 16


def pack_uint2_to_int32(codes):
    """[rows, cols] uint8 grid indices -> [rows, cols/16] int32, Humming layout."""
    if codes.dtype not in (torch.uint8, torch.int32, torch.int64):
        raise ValueError(f"pack_uint2_to_int32 expects integer codes, got {codes.dtype}.")
    rows, cols = codes.shape
    if cols % CODES_PER_INT32 != 0:
        raise ValueError(
            f"pack_uint2_to_int32 needs cols divisible by {CODES_PER_INT32}, got {cols}."
        )
    if int(codes.max()) > 3 or int(codes.min()) < 0:
        raise ValueError("uint2 codes must lie in [0, 3].")
    c = codes.reshape(rows, cols // CODES_PER_INT32, CODES_PER_INT32).to(torch.int64)
    shifts = (torch.arange(CODES_PER_INT32, device=codes.device, dtype=torch.int64) * 2)
    packed = (c << shifts).sum(dim=-1)
    # wrap into signed int32 without changing the bit pattern
    packed = torch.where(packed >= 2 ** 31, packed - 2 ** 32, packed)
    return packed.to(torch.int32)


def unpack_int32_to_uint2(packed, cols):
    """Inverse of pack_uint2_to_int32, mirroring common_unpack_weight<2>."""
    rows = packed.shape[0]
    p = packed.to(torch.int64) & 0xFFFFFFFF
    shifts = (torch.arange(CODES_PER_INT32, device=packed.device, dtype=torch.int64) * 2)
    codes = (p.unsqueeze(-1) >> shifts) & 0x3
    return codes.reshape(rows, -1)[:, :cols].to(torch.uint8)


def dequantize_gsq_packed(packed, scale, wbits=2):
    """Inverse of the save path: int32 codes + per-group scale -> fp32 s * grid[code].

    Every load_from_disc that reads a GSQ shard goes through this, so the fused-expert
    wrappers cannot drift from base.py's dequantization.
    """
    rows, n_words = packed.shape
    cols = n_words * CODES_PER_INT32
    s = scale.to(device=packed.device, dtype=torch.float32)
    if s.shape[0] != rows or cols % s.shape[1] != 0:
        raise ValueError(
            f"dequantize_gsq_packed: scale {tuple(s.shape)} does not tile codes [{rows}, {cols}]."
        )
    groupsize = cols // s.shape[1]
    codes = unpack_int32_to_uint2(packed, cols)
    grid = torch.tensor(GSQ_GRIDS[wbits], dtype=torch.float32, device=packed.device)
    col_group = torch.arange(cols, device=packed.device) // groupsize
    return s[:, col_group] * grid[codes.long()]


def codes_from_dequantized(weight, scale, groupsize, grid):
    """Recover grid indices from a dequantized GSQ weight (w == scale * grid[idx]).

    Used by the save path: the module holds s*q after get_hard_weights(), and the
    per-group scale comes alongside it, so the codes are recoverable exactly rather
    than re-derived by a heuristic.
    """
    rows, cols = weight.shape
    idx = torch.arange(cols, device=weight.device) // groupsize
    s = scale.to(weight.device, torch.float32)[:, idx]
    q = weight.to(torch.float32) / torch.where(s.abs() < 1e-12, torch.full_like(s, 1e-12), s)
    g = grid.to(weight.device, torch.float32)
    return (q.unsqueeze(-1) - g).abs().argmin(dim=-1).to(torch.uint8)
