"""On-disk sparse storage for paired-4:8 NVFP4 MoE expert weights.

Copy of vllm/model_executor/layers/quantization/utils/paired48_sparse.py from the
vLLM patch (integrations/vllm/moe-sq-v0.30.0.patch), which decodes this format at
load time; keep the two byte-compatible. save_model.py uses it to write
``assembled/`` in sparse storage directly.

A paired-4:8 NVFP4 weight keeps 2 of every 4 consecutive FP4 *pairs* along K
(4 of every 8 elements). Packed NVFP4 stores two adjacent K elements per byte,
so a pair IS one byte of the dense ``weight_packed`` tensor, and the pattern is
"2 nonzero bytes out of every 4". The ``pair-bitmask`` layout stores exactly
that, with no kernel-specific tiling or padding:

    weight_sparse_packed  uint8 [..., K/4]   the 2 kept bytes of every 4, in K order
    weight_sparse_mask    uint8 [..., K/16]  4 bits per 4-byte chunk (exactly 2
                                             set, bit i = byte i kept); the low
                                             nibble is the lower-K chunk

Both are proportional to K, so the generic FusedMoE loader TP/EP-slices them
exactly like ``weight_packed`` / ``weight_scale``. Expert bytes drop from
M*K*(1/2 + 1/32) to M*K*(1/4 + 1/16 + 1/32), i.e. 0.65x. At load vLLM rebuilds
the dense packed weight one layer at a time and hands it to the kernel's own
``paired_nvfp4_compress``, so the format is independent of the kernel's
internal A/E/SFA layouts.

A chunk with fewer than 2 nonzero bytes (a kept pair that quantized to zero)
marks its lowest-index zero bytes as kept, so every chunk has exactly 2 bits
set and the encoding is deterministic and lossless.

This module depends only on torch so offline tools can import it without
initializing vLLM.
"""

from typing import Any

import torch

CONFIG_KEY = "paired48_sparse"
LAYOUT = "pair-bitmask"
VERSION = 1
PACKED_SUFFIX = "weight_sparse_packed"
MASK_SUFFIX = "weight_sparse_mask"


def _check_k(packed_k: int) -> None:
    # K/2 packed bytes; K % 16 == 0 so the mask is whole bytes (2 chunks/byte).
    if packed_k % 8 != 0:
        raise ValueError(
            f"paired48 sparse storage needs K % 16 == 0 (packed last dim % 8 == 0); "
            f"got packed last dim {packed_k}"
        )


# Both directions are elementwise over the four byte planes of each 4-byte chunk
# (no gathers, int64 indices or size-4 reductions): exactly two of k0..k3 are
# set, so the first kept byte is the lowest set index and the second the highest.


def compress_pair_bitmask(
    w_packed: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense packed NVFP4 ``[..., K/2]`` -> (``[..., K/4]``, ``[..., K/16]``).

    Raises if any 4-byte chunk has more than 2 nonzero bytes (not paired-4:8).
    """
    assert w_packed.dtype == torch.uint8, w_packed.dtype
    *lead, kb = w_packed.shape
    _check_k(kb)
    b = w_packed.reshape(*lead, kb // 4, 4).unbind(-1)
    k = [bi != 0 for bi in b]
    nnz = sum(ki.to(torch.uint8) for ki in k)
    if bool((nnz > 2).any()):
        bad = int((nnz > 2).sum())
        raise ValueError(
            f"{bad} chunk(s) have more than 2 nonzero bytes out of 4: the weight "
            "is not paired-4:8 sparse"
        )
    # Top up chunks with <2 nonzero bytes using their lowest-index zero bytes.
    need = 2 - nnz
    seen = torch.zeros_like(nnz)
    for i in range(4):
        free = ~k[i]
        seen += free
        k[i] = k[i] | (free & (seen <= need))

    v0 = torch.where(k[0], b[0], torch.where(k[1], b[1], b[2]))
    v1 = torch.where(k[3], b[3], torch.where(k[2], b[2], b[1]))
    values = torch.stack((v0, v1), dim=-1).reshape(*lead, kb // 2)
    nib = sum(ki.to(torch.uint8) << i for i, ki in enumerate(k))
    mask = nib[..., 0::2] | (nib[..., 1::2] << 4)
    return values, mask.contiguous()


def _decompress_into(
    out: torch.Tensor, values: torch.Tensor, mask: torch.Tensor
) -> int:
    *lead, mb = mask.shape
    nib = torch.stack((mask & 0xF, mask >> 4), dim=-1).reshape(*lead, mb * 2)
    k = [((nib >> i) & 1).bool() for i in range(4)]
    pv = values.reshape(*lead, mb * 2, 2)
    v0, v1 = pv[..., 0], pv[..., 1]
    planes = (
        torch.where(k[0], v0, 0),
        torch.where(k[1], torch.where(k[0], v1, v0), 0),
        torch.where(k[2], torch.where(k[0] | k[1], v1, v0), 0),
        torch.where(k[3], v1, 0),
    )
    torch.stack(planes, dim=-1, out=out.view(*lead, mb * 2, 4))
    popcount = sum(ki.to(torch.uint8) for ki in k)
    return int((popcount != 2).sum())


def decompress_pair_bitmask(
    values: torch.Tensor,
    mask: torch.Tensor,
    strict: bool = True,
    experts_per_slice: int = 8,
) -> tuple[torch.Tensor, int]:
    """(``[..., K/4]``, ``[..., K/16]``) -> dense packed NVFP4 ``[..., K/2]``.

    Expert-stacked ``[E, rows, *]`` inputs are decoded ``experts_per_slice``
    experts at a time to bound the temporaries (a few x the slice's size).
    Returns ``(dense, n_invalid_nibbles)``; with ``strict`` a nonzero count
    raises instead.
    """
    assert values.dtype == torch.uint8 and mask.dtype == torch.uint8
    *lead, mb = mask.shape
    if list(values.shape) != [*lead, mb * 4]:
        raise ValueError(
            f"sparse values shape {tuple(values.shape)} does not match mask shape "
            f"{tuple(mask.shape)} (expected last dim {mb * 4})"
        )
    out = torch.empty(*lead, mb * 8, dtype=torch.uint8, device=mask.device)
    n_invalid = 0
    if len(lead) < 2:  # a single [rows, K] weight: no expert dim to slice
        n_invalid = _decompress_into(out, values, mask)
    else:
        for s in range(0, lead[0], experts_per_slice):
            e = min(s + experts_per_slice, lead[0])
            n_invalid += _decompress_into(out[s:e], values[s:e], mask[s:e])
    if strict and n_invalid:
        raise ValueError(
            f"{n_invalid} invalid mask nibble(s) (not exactly 2 bits set): "
            "corrupt paired48 sparse weight"
        )
    return out, n_invalid


def is_enabled(quant_config: dict[str, Any] | None) -> bool:
    """Whether a compressed-tensors ``quantization_config`` stores experts sparse.

    Validates the marker; raises on an unknown layout or version rather than
    loading tensors under the wrong interpretation.
    """
    marker = (quant_config or {}).get(CONFIG_KEY)
    if not marker:
        return False
    if (
        not isinstance(marker, dict)
        or marker.get("layout") != LAYOUT
        or marker.get("version") != VERSION
    ):
        raise ValueError(
            f"unsupported {CONFIG_KEY} marker {marker}; this vLLM understands "
            f"layout={LAYOUT!r} version={VERSION}"
        )
    return True


def make_marker() -> dict[str, Any]:
    return {"layout": LAYOUT, "version": VERSION}
