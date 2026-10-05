import torch

from .expert_compressor import ExpertCompressor
from .sparsity import Paired48, Paired24, NoSparsity
from .quant import NVFP4Quant, GSQQuant
from .quant.nvfp4 import NVFP4_GROUPSIZE, dense_scale_groupsize


def build_compressor(config, device, dtype,
                     init_compressed_weight, init_scales,
                     init_dense_weight=None, init_support_mask=None):
    """Build an ExpertCompressor instance from config + GPTQ init tensors.

    Pipeline is selected by compression.quant_type:
      "nvfp4"  -> sparse + quantized (paired-4:8 | 2:4) + NVFP4 fake-quant. Which of the
                  support / values are trainable is set by refine.learn_masks and
                  compression.learn_weight_values (the joint-adaptation ablation).
      "gsq"    -> GSQ scalar quantization (arXiv:2604.18556): learned grid assignment
                  + learned per-group scale, dense (no sparsity), no master weight
      None     -> sparsity-only baseline (no weight quant, no learnable dense weight)

    Adding a new sparse+quantized target (e.g. paired-4:8 + INT8) means adding
    a quant component and one elif in `_build_sparse_quantized` below.
    """
    groupsize = config.compression.groupsize
    std = config.refine.std
    strength = config.refine.strength
    prunen = config.compression.prunen
    prunem = config.compression.prunem
    learn_weight_values = bool(config.compression.learn_weight_values)
    learn_masks = bool(config.refine.learn_masks)
    logits_dtype = torch.float32 if config.refine.logits_dtype == "float32" else dtype
    quant_type = config.compression.quant_type

    if quant_type is None:
        W_dense = init_dense_weight if init_dense_weight is not None else init_compressed_weight
        mask_prior = init_support_mask if init_support_mask is not None else init_scales
        return _build_sparsity_only(
            device, dtype, W_dense, mask_prior, prunen, prunem, std, strength, logits_dtype,
        )

    if quant_type == "gsq":
        return _build_gsq(
            config, device, dtype,
            init_compressed_weight, init_dense_weight, init_scales,
            groupsize, std, strength, logits_dtype,
        )

    if quant_type == "nvfp4":
        if not (learn_masks or learn_weight_values):
            raise ValueError(
                "compression.learn_weight_values=False with refine.learn_masks=False leaves "
                "nothing to train; use refine.enabled=false for the initialization-only arm."
            )
        return _build_sparse_quantized(
            config, device, dtype,
            init_compressed_weight, init_dense_weight, init_support_mask, init_scales,
            prunen, prunem, groupsize, std, strength, logits_dtype,
            learn_masks, learn_weight_values,
        )

    raise ValueError(
        f"Unsupported compression.quant_type={quant_type!r}. Supported: 'nvfp4', 'gsq' or null."
    )


def _build_sparse_quantized(config, device, dtype,
                            init_compressed_weight, init_dense_weight, init_support_mask, init_scales,
                            prunen, prunem, groupsize, std, strength, logits_dtype,
                            learn_masks=True, learn_weights=True):
    if init_dense_weight is None or init_support_mask is None:
        raise ValueError("Sparse+quantized fake-quantized learning requires both init_dense_weight and init_support_mask.")
    weight_shape = tuple(init_compressed_weight.shape)
    _check_init_consistency(device, init_compressed_weight, init_dense_weight, init_support_mask)

    sparsity = _select_sparsity(device, dtype, prunen, prunem)

    dense_groupsize = dense_scale_groupsize(prunen, prunem, groupsize)
    _check_nvfp4_init_shapes(weight_shape, init_scales, groupsize, dense_groupsize)
    quant = NVFP4Quant(
        weight_shape, dense_groupsize, device, dtype,
        scale_grad_p=config.refine.scale_grad_p,
        coupling_stats=getattr(config.refine, "_coupling_stats", None),
    )

    # Frozen support: take the initializer's mask directly instead of freezing the logits.
    # init_mask_logits adds N(0, std) noise, so an argmax over frozen logits is the init
    # support only with high probability, not by construction -- and a frozen-logit forward
    # would still SAMPLE, handing the values a support that moves every step.
    mask_logits_init = (sparsity.init_mask_logits(init_support_mask, std, strength)
                        if learn_masks else None)
    # Frozen values are the DENSE GPTQ solution (error-compensated and FP4-projected at
    # every position, pruned ones included), so alternative supports are scored fairly.
    # fp32 buffer, matching the learned arm's fp32 master, so the arms differ only in
    # what is trainable.
    dense = init_dense_weight.to(device=device)
    return ExpertCompressor(
        weight_shape=weight_shape,
        sparsity=sparsity,
        quant=quant,
        weight_buffer=None if learn_weights else dense,
        weight_master_init=dense if learn_weights else None,
        learn_weights=learn_weights,
        mask_logits_init=mask_logits_init,
        device=device,
        dtype=dtype,
        logits_dtype=logits_dtype,
        fixed_mask=None if learn_masks else init_support_mask,
        weight_buffer_dtype=torch.float32,
        init_support=init_support_mask,
        keep_weight_ref=bool(getattr(config.refine, "weight_drift_diagnostics", False)),
    )


def _build_gsq(config, device, dtype,
               init_compressed_weight, init_dense_weight, init_scales,
               groupsize, std, strength, logits_dtype):
    """GSQ: quantization-only, no sparsity, no continuous master weight.

    The warm-start weight is the DENSE GPTQ output when available. GSQ's paper
    initializes its logits from the GPTQ solution, and init_dense_weight is that
    solution before masking; init_compressed_weight is the same thing already
    masked, which for a dense (0:0) run is identical.
    """
    W = init_dense_weight if init_dense_weight is not None else init_compressed_weight
    if W is None:
        raise ValueError("GSQ requires a GPTQ warm-start weight (init_dense_weight).")
    W = W.to(device=device)
    weight_shape = tuple(W.shape)

    sparsity = NoSparsity(device, dtype)
    quant = GSQQuant(weight_shape, groupsize, device, dtype, wbits=config.init.wbits)
    quant.init_from_weight(W, std, strength, logits_dtype, init_scales=None)

    mask_logits_init = sparsity.init_mask_logits(W, std, strength)
    return ExpertCompressor(
        weight_shape=weight_shape,
        sparsity=sparsity,
        quant=quant,
        # GSQ reconstructs as scale * grid_code, so the buffer is never read by the
        # forward; it is kept only because ExpertCompressor requires a weight source.
        weight_buffer=W,
        weight_master_init=None,
        learn_weights=False,
        mask_logits_init=mask_logits_init,
        device=device,
        dtype=dtype,
        logits_dtype=logits_dtype,
    )


def _build_sparsity_only(device, dtype, W_dense, mask_prior, prunen, prunem, std, strength, logits_dtype):
    if mask_prior is None:
        raise ValueError("Sparsity-only baseline requires a mask prior (init_support_mask or binary init_scales).")
    sparsity = _select_sparsity(device, dtype, prunen, prunem)
    mask_logits_init = sparsity.init_mask_logits(mask_prior, std, strength)
    return ExpertCompressor(
        weight_shape=tuple(W_dense.shape),
        sparsity=sparsity,
        quant=None,
        weight_buffer=W_dense.to(device=device),
        weight_master_init=None,
        learn_weights=False,
        mask_logits_init=mask_logits_init,
        device=device,
        dtype=dtype,
        logits_dtype=logits_dtype,
    )


def _select_sparsity(device, dtype, prunen, prunem):
    if (prunen, prunem) == (4, 8):
        return Paired48(device, dtype)
    if (prunen, prunem) == (2, 4):
        return Paired24(device, dtype)
    raise ValueError(
        f"Unsupported sparsity pattern compression.prunen:prunem={prunen}:{prunem}. "
        "Supported: 4:8 or 2:4."
    )


def _check_nvfp4_init_shapes(weight_shape, init_scales, groupsize, dense_groupsize):
    if groupsize != NVFP4_GROUPSIZE:
        raise ValueError(f"NVFP4 expects compression.groupsize={NVFP4_GROUPSIZE}, got {groupsize}.")
    if weight_shape[1] % dense_groupsize != 0:
        raise ValueError(
            f"NVFP4 requires weight width divisible by {dense_groupsize}, got {weight_shape[1]}."
        )
    expected_scale_shape = (weight_shape[0], weight_shape[1] // dense_groupsize)
    if tuple(init_scales.shape) != expected_scale_shape:
        raise ValueError(
            f"NVFP4 expects init_scales with shape {expected_scale_shape}, got {tuple(init_scales.shape)}."
        )
    if not torch.isfinite(init_scales).all() or torch.any(init_scales <= 0):
        raise ValueError("NVFP4 requires finite positive init_scales.")


def _check_init_consistency(device, init_compressed_weight, init_dense_weight, init_support_mask):
    support = init_support_mask.to(device=device, dtype=torch.bool)
    dense = init_dense_weight.to(device=device)
    q_expected = dense.masked_fill(~support, 0)
    if not torch.allclose(init_compressed_weight.to(device=device, dtype=dense.dtype), q_expected):
        raise ValueError("init_compressed_weight must equal init_dense_weight with masked positions zeroed.")
