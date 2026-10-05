import torch
import torch.nn as nn

from .expert_compressor import split_hard_output_full
from .quant.nvfp4 import (
    NVFP4Quant, fp4_project, fp8_project,
    FP8_E4M3_MIN_POSITIVE, FP8_E4M3_MAX,
)


class ScaleFinetuneCompressor(nn.Module):
    """Post-refinement scale-only fine-tuning for a paired-N:M + NVFP4 expert weight.

    Built FROM a refined ExpertCompressor, after the mask and weight values are done
    training (and after best-epoch restore). Everything discrete is frozen: the
    hard support mask and the FP4 grid value ("code") of every surviving weight. The
    only trainable quantity is a per-group log-multiplier on the scale,

        s = s_0 * exp(theta),    theta init 0  ->  s = s_0 reproduces the refined
                                                    hard weights bit for bit.

    The multiplicative form makes Lion's sign-based, fixed-magnitude steps scale-free:
    one step moves every group by the same RELATIVE amount whatever its |w|max.

    Format fidelity: NVFP4 stores one FP8-E4M3 scale per group times one fp32 global
    scale per tensor, so a continuous s is not what gets served. The forward projects
    s onto that grid (global = FP8_MAX / max s, then FP8 rounding) with a straight-
    through estimator, exactly as NVFP4Quant does for the dynamic scale -- what the
    loss sees is what `get_hard_weights` exports. This is the GSQ "scale-only
    fine-tuning" stage (arXiv:2604.18556 App. H) transplanted to FP4 and run
    layer-wise inside the same reconstruction objective as refinement.

    `forward(temperature, scale)` keeps ExpertCompressor's signature so the trainer's
    step/validation code is reused unchanged; both arguments are ignored.
    """

    def __init__(self, codes, base_scales, hard_mask, groupsize, device, dtype):
        super().__init__()
        rows, cols = codes.shape
        if cols % groupsize != 0 or base_scales.shape != (rows, cols // groupsize):
            raise ValueError(
                f"ScaleFinetuneCompressor: codes {tuple(codes.shape)} and scales "
                f"{tuple(base_scales.shape)} disagree with groupsize {groupsize}."
            )
        if not torch.isfinite(base_scales).all() or torch.any(base_scales <= 0):
            raise ValueError("ScaleFinetuneCompressor requires finite positive base scales.")
        self.weight_shape = (rows, cols)
        self.groupsize = groupsize
        self.device = device
        self.dtype = dtype
        # FP4 grid values are exactly representable in bf16, so codes cost 2 B/weight.
        self.register_buffer("codes", codes.detach().to(device=device, dtype=torch.bfloat16))
        self.register_buffer("base_scales", base_scales.detach().to(device=device, dtype=torch.float32))
        self.register_buffer("mask", hard_mask.detach().to(device=device, dtype=torch.bool))
        self.register_buffer("idx", torch.arange(cols, device=device) // groupsize)
        self.register_buffer("scale_min", torch.tensor(FP8_E4M3_MIN_POSITIVE, device=device, dtype=torch.float32))
        self.register_buffer("scale_max", torch.tensor(FP8_E4M3_MAX, device=device, dtype=torch.float32))
        self.log_scale = nn.Parameter(torch.zeros_like(self.base_scales))

    # ------------------------------------------------------------------ build
    @staticmethod
    def supports(compressor):
        return isinstance(getattr(compressor, "quant", None), NVFP4Quant)

    @classmethod
    def from_compressor(cls, compressor):
        """Freeze a refined NVFP4 ExpertCompressor into codes + scales.

        Recovers the FP4 codes from the hard output rather than re-deriving them from
        the master weight, so theta=0 ships exactly what the refined compressor would
        have shipped. bf16 rounding of the hard weight is ~2^-8 relative, far inside
        the FP4 grid's 0.25 decision half-widths, so the snap-back is exact.
        """
        if not cls.supports(compressor):
            raise TypeError("ScaleFinetuneCompressor only wraps NVFP4 ExpertCompressors.")
        with torch.no_grad():
            weight, hard_scales, global_scale, mask = split_hard_output_full(
                compressor.get_hard_weights())
            if hard_scales is None or global_scale is None or mask is None:
                raise ValueError("NVFP4 hard output must carry scales, a global scale and a mask.")
            groupsize = compressor.quant.groupsize
            base_scales = hard_scales.to(torch.float32) / global_scale.to(torch.float32)
            idx = torch.arange(weight.shape[1], device=weight.device) // groupsize
            codes = fp4_project(weight.to(torch.float32) / base_scales[:, idx])
            rebuilt = (codes * base_scales[:, idx]).to(weight.dtype)
            if not torch.equal(rebuilt, weight):
                raise RuntimeError(
                    "ScaleFinetuneCompressor: could not recover FP4 codes exactly from the "
                    "refined hard weight; refusing to fine-tune from a different starting point."
                )
        return cls(codes, base_scales, mask, groupsize, compressor.device, compressor.dtype)

    # ---------------------------------------------------------------- forward
    def _scales(self):
        return self.base_scales * torch.exp(self.log_scale)

    def _project(self, scales):
        """Snap per-group scales onto NVFP4's FP8 x global grid.

        Returns (scale_per_group with STE gradient, hard FP8-valued scales, global).
        Mirrors NVFP4Quant._project_hard_scales / _compute_global_scale so the export
        path and the training forward agree to the bit.
        """
        max_scale = scales.detach().amax().to(torch.float32)
        if max_scale <= 0:
            global_scale = torch.ones(1, device=scales.device, dtype=torch.float32)
        else:
            global_scale = (self.scale_max / max_scale).reshape(1)
        scaled = torch.clamp(scales * global_scale, min=self.scale_min, max=self.scale_max)
        hard = fp8_project(scaled)
        ste = scaled + (hard - scaled).detach()
        return ste / global_scale, hard, global_scale

    def forward(self, temperature=None, scale=None):
        s_eff, _, _ = self._project(self._scales())
        return (self.codes.float() * s_eff[:, self.idx]).to(self.dtype)

    @torch.no_grad()
    def get_hard_weights(self):
        _, hard_scales, global_scale = self._project(self._scales())
        if not torch.isfinite(hard_scales).all() or torch.any(hard_scales <= 0):
            raise ValueError("ScaleFinetuneCompressor hard scales must be finite and positive.")
        weight = self.codes.float() * (hard_scales / global_scale)[:, self.idx]
        return weight.to(self.dtype), hard_scales, global_scale, self.mask
