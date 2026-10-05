"""The compressed-tensors surface MoESQ uses, stable across library versions.

compressed-tensors 0.15 rewrote its compressor API: the per-weight instance methods
`compress_weight` / `decompress_weight` became per-module classmethods
`compress(state_dict, scheme)` / `decompress(state_dict, scheme)`, the sparse
compressors (`BitmaskCompressor`, `Sparse24BitMaskCompressor`) were removed, and the
NVFP4 compressor moved out of `compressors.quantized_compressors.fp4_quantized`.
0.14 is also the last release that caps `transformers<5`.

This module keeps the old call signatures so the model wrappers do not care which
side of that split is installed:

* On 0.14 the library classes are re-exported unchanged.
* On >= 0.15 thin adapters call the new classmethods. Each call forces the eager
  torch implementation (the library otherwise dispatches CUDA inputs to Triton
  kernels), and decompression infers the group size from the scale shape exactly as
  0.14's `dequantize(args=None)` did.
* The sparse bitmask formats are small and fixed, so they are reimplemented here
  (ported from compressed-tensors 0.14, Apache-2.0) on top of the library's
  `pack_bitmasks` / `unpack_bitmasks`.

The outputs are byte-identical to 0.14 for every path MoESQ exercises (NVFP4 group
16/32, INT4 pack-quantized group 32, sparse bitmask, 2:4 bitmask).
"""
from types import SimpleNamespace

import torch

from compressed_tensors.config import (
    BitmaskConfig,
    Sparse24BitMaskConfig,
    SparsityStructure,
)
from compressed_tensors.quantization import (
    FP8_E4M3_DATA,
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStrategy,
    QuantizationType,
)
from compressed_tensors.utils import pack_bitmasks, unpack_bitmasks

__all__ = [
    "BitmaskCompressor",
    "BitmaskConfig",
    "FP8_E4M3_DATA",
    "NVFP4PackedCompressor",
    "PackedQuantizationCompressor",
    "QuantizationArgs",
    "QuantizationStrategy",
    "QuantizationType",
    "Sparse24BitMaskCompressor",
    "Sparse24BitMaskConfig",
    "pack_bitmasks",
]

try:  # compressed-tensors <= 0.14: the old API is still there, use it as is.
    from compressed_tensors import (  # noqa: F401
        BitmaskCompressor,
        PackedQuantizationCompressor,
        Sparse24BitMaskCompressor,
    )
    from compressed_tensors.compressors.quantized_compressors.fp4_quantized import (  # noqa: F401
        NVFP4PackedCompressor,
    )

    LEGACY_API = True
except ImportError:
    LEGACY_API = False


if not LEGACY_API:
    from compressed_tensors.compressors.nvfp4.base import (
        NVFP4PackedCompressor as _NVFP4,
    )
    from compressed_tensors.compressors.pack_quantized.base import (
        PackedQuantizationCompressor as _Packed,
    )
    from compressed_tensors.utils import impl_backend as _impl_backend
    from compressed_tensors.utils import patch_attr

    def _to_plain(x):
        if isinstance(x, SimpleNamespace):
            x = vars(x)
        if isinstance(x, dict):
            return {k: _to_plain(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_to_plain(v) for v in x]
        return x

    def _as_args(args):
        """QuantizationArgs from what the wrappers pass: a QuantizationArgs, a dict,
        or the SimpleNamespace that BaseModelWrapper.dict_to_ns builds from a config."""
        if isinstance(args, QuantizationArgs):
            return args
        fields = _to_plain(args)
        return QuantizationArgs.model_validate(
            {k: v for k, v in fields.items() if k in QuantizationArgs.model_fields}
        )

    def _scheme(args):
        return QuantizationScheme(targets=["Linear"], weights=_as_args(args))

    def _scheme_from_shapes(args, n_cols, scale):
        """0.14's decompress_weight called dequantize(args=None), which infers
        GROUP with group_size = columns / scale columns from the shapes. >= 0.15
        requires args; keep the shape-derived group size so a scale whose width
        disagrees with the latched config decodes exactly as it did before."""
        args = _as_args(args)
        if scale.ndim == 2 and scale.shape[1] > 1:
            args = args.model_copy(update={"group_size": n_cols // scale.shape[1]})
        return QuantizationScheme(targets=["Linear"], weights=args)

    def _check_bits(args):
        # >= 0.15 packs widths that do not divide 32 (3, 5, 6, 7) densely across
        # int32 words; 0.14 padded each word. Refuse rather than silently write or
        # read the other layout. MoESQ only uses 4-bit (1, 2, 8 are unchanged too).
        if 32 % args.num_bits != 0:
            raise NotImplementedError(
                f"{args.num_bits}-bit pack-quantized changed layout in "
                "compressed-tensors 0.15; only widths dividing 32 are supported")
        return args

    def _eager():
        # Registered Triton backends run on CUDA inputs; 0.14 had only the eager
        # torch path, so pin that one for bit-for-bit reproducibility.
        return patch_attr(_impl_backend, "ENFORCE_EAGER", True)

    class NVFP4PackedCompressor:
        def compress_weight(self, weight, scale, global_scale, quantization_args,
                            device=None, zero_point=None, g_idx=None):
            if g_idx is not None:
                raise ValueError("g_idx is not supported by compressed-tensors >= 0.15")
            sd = {"weight": weight, "weight_scale": scale,
                  "weight_global_scale": global_scale}
            if zero_point is not None:
                sd["weight_zero_point"] = zero_point
            with _eager():
                out = _NVFP4.compress(sd, _scheme(quantization_args))
            packed = out["weight_packed"]
            if device is not None:
                packed = packed.to(device)
            return {"weight_packed": packed, "weight_scale": out["weight_scale"]}

        def decompress_weight(self, compressed_data, quantization_args=None):
            packed = compressed_data["weight_packed"]
            scale = compressed_data["weight_scale"]
            sd = {"weight_packed": packed, "weight_scale": scale,
                  "weight_global_scale": compressed_data["weight_global_scale"]}
            if quantization_args is None:
                quantization_args = QuantizationArgs(
                    num_bits=4, type="float", strategy="tensor_group", group_size=16,
                    symmetric=True)
            scheme = _scheme_from_shapes(quantization_args, packed.shape[1] * 2, scale)
            with _eager():
                return _NVFP4.decompress(sd, scheme)["weight"]

    class PackedQuantizationCompressor:
        def compress_weight(self, weight, scale, quantization_args, zero_point=None,
                            g_idx=None, device=None, global_scale=None):
            if global_scale is not None:
                raise ValueError(
                    "global_scale is not supported for the PackQuantizationCompressor")
            if g_idx is not None:
                raise ValueError("g_idx is not supported by compressed-tensors >= 0.15")
            if not torch.is_floating_point(weight):
                # 0.14 packed already-integer weights without re-quantizing them.
                raise ValueError("PackedQuantizationCompressor expects a float weight")
            sd = {"weight": weight, "weight_scale": scale}
            if zero_point is not None:
                sd["weight_zero_point"] = zero_point
            scheme = _scheme(quantization_args)
            _check_bits(scheme.weights)
            with _eager():
                out = _Packed.compress(sd, scheme)
            res = {"weight_shape": out["weight_shape"],
                   "weight_packed": out["weight_packed"]}
            if "weight_zero_point" in out:
                res["weight_zero_point"] = out["weight_zero_point"]
            if device is not None:
                res = {k: v.to(device) for k, v in res.items()}
            return res

        def decompress_weight(self, compressed_data, quantization_args=None):
            if compressed_data.get("weight_g_idx") is not None:
                raise ValueError("g_idx is not supported by compressed-tensors >= 0.15")
            shape = torch.Size(compressed_data["weight_shape"])
            scale = compressed_data["weight_scale"]
            sd = {k: compressed_data[k]
                  for k in ("weight_packed", "weight_scale", "weight_zero_point")
                  if compressed_data.get(k) is not None}
            sd["weight_shape"] = shape
            scheme = _scheme_from_shapes(quantization_args, shape[-1], scale)
            _check_bits(scheme.weights)
            with _eager():
                return _Packed.decompress(sd, scheme)["weight"]

    def _merge(prefix, name):
        return f"{prefix}.{name}"

    def _masked_values(tensor, bytemasks):
        if tensor.dtype == FP8_E4M3_DATA.dtype:  # index the raw bytes
            return tensor.view(torch.int8)[bytemasks].view(FP8_E4M3_DATA.dtype)
        return tensor[bytemasks]

    class BitmaskCompressor:
        """Unstructured `sparse-bitmask`: flat non-zero values + packed bitmask +
        per-row offsets (ported from compressed-tensors 0.14)."""

        def __init__(self, config=None):
            self.config = config if config is not None else BitmaskConfig()

        def compress_weight(self, name, value):
            shape = value.shape
            tensor = value.cpu()
            bytemasks = tensor != 0
            row_counts = bytemasks.sum(dim=-1)
            row_offsets = torch.cumsum(row_counts, 0) - row_counts
            return {
                _merge(name, "shape"): torch.tensor(list(shape), device="cpu"),
                _merge(name, "compressed"): _masked_values(tensor, bytemasks).to("cpu"),
                _merge(name, "bitmask"): pack_bitmasks(bytemasks).to("cpu"),
                _merge(name, "row_offsets"): row_offsets.to("cpu"),
            }

        def decompress_weight(self, weight_data):
            shape = list(weight_data["shape"])
            values = weight_data["compressed"]
            mask = unpack_bitmasks(weight_data["bitmask"], shape)
            dense = torch.zeros(shape, dtype=values.dtype)
            dense[mask] = values
            return dense

    def _get_24_bytemasks(tensor):
        if tensor.dtype == FP8_E4M3_DATA.dtype:
            tensor = tensor.view(torch.int8)
        if tensor.numel() % 4 != 0:
            raise ValueError("Tensor size must be a multiple of 4 for TWO_FOUR sparsity")
        reshaped = tensor.view(-1, 4)
        topk = reshaped.abs().topk(2, dim=1).indices
        mask = torch.zeros_like(reshaped, dtype=torch.bool)
        mask.scatter_(1, topk, True)
        return mask.view(tensor.shape)

    class Sparse24BitMaskCompressor:
        """`sparse-24-bitmask`: [rows, cols/2] kept values + packed bitmask
        (ported from compressed-tensors 0.14)."""

        def __init__(self, config=None):
            self.config = config if config is not None else Sparse24BitMaskConfig()

        def compress_weight(self, name, value):
            if SparsityStructure(self.config.sparsity_structure) != SparsityStructure.TWO_FOUR:
                raise ValueError("Only 2:4 sparsity is supported")
            if value.ndim != 2:
                raise ValueError("Only 2D tensors are supported")
            if name.endswith(".weight"):
                name = name[: -len(".weight")]
            rows, cols = value.shape
            if value.is_meta:
                device = "meta"
                compressed = torch.empty((rows, cols // 2), dtype=value.dtype, device="meta")
                bitmask = torch.empty((rows, (cols + 7) // 8), dtype=torch.uint8,
                                      device="meta")
            else:
                device = "cpu"
                tensor = value.cpu()
                bytemasks = _get_24_bytemasks(tensor)
                compressed = _masked_values(tensor, bytemasks).reshape(rows, cols // 2)
                bitmask = pack_bitmasks(bytemasks)
            return {
                _merge(name, "shape"): torch.tensor([rows, cols], device=device).reshape(-1, 1),
                _merge(name, "compressed"): compressed.to(device),
                _merge(name, "bitmask"): bitmask.to(device),
            }

        def decompress_weight(self, weight_data):
            shape = weight_data["shape"]
            if isinstance(shape, list):
                shape = torch.tensor(shape)
            shape = shape.flatten().tolist()
            values = weight_data["compressed"]
            mask = unpack_bitmasks(weight_data["bitmask"], shape)
            dense = torch.zeros(shape, dtype=values.dtype).to(values.device)
            dense[mask] = values.flatten()
            return dense
