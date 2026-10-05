import torch

from .base import BaseActivationQuant
from .quant.nvfp4 import (
    FP4_E2M1_MAX,
    FP8_E4M3_MAX,
    FP8_E4M3_MIN_POSITIVE,
    NVFP4_GROUPSIZE,
    fp4_ste,
    fp8_project,
)


ACT_AMAX_EPS = 1e-8


class NVFP4ActivationQuant(BaseActivationQuant):
    """NVFP4 dynamic-local activation fake-quant.

    Group-wise FP4 with FP8-projected per-group scales and a global tensor-wise
    scale. Used when quantization.fake_quantize_activations is enabled.
    """

    def __init__(self, groupsize=NVFP4_GROUPSIZE):
        super().__init__()
        self.groupsize = groupsize

    def fake_quantize(self, x):
        return fake_quantize_activation_nvfp4(x, groupsize=self.groupsize)


def fake_quantize_activation_nvfp4(x, groupsize=NVFP4_GROUPSIZE):
    if x.shape[-1] % groupsize != 0:
        raise ValueError(
            f"fake_quantize_activation_nvfp4: last dim ({x.shape[-1]}) "
            f"must be divisible by groupsize ({groupsize})."
        )

    x32 = x.float()
    amax_t = x32.detach().abs().amax().clamp(min=ACT_AMAX_EPS)
    g = (FP8_E4M3_MAX * FP4_E2M1_MAX) / amax_t

    grouped = x32.reshape(*x32.shape[:-1], -1, groupsize)
    amax_g = grouped.detach().abs().amax(dim=-1, keepdim=True).clamp(min=ACT_AMAX_EPS)
    s_g_raw = (amax_g * g / FP4_E2M1_MAX).clamp(
        min=FP8_E4M3_MIN_POSITIVE, max=FP8_E4M3_MAX
    )
    s_g = fp8_project(s_g_raw)

    scale_per_elem = (s_g / g).expand_as(grouped).reshape_as(x32)
    x_q = fp4_ste(x32 / scale_per_elem) * scale_per_elem
    return x_q.to(x.dtype)
