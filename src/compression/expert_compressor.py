import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertCompressor(nn.Module):
    """Trainable compression module for a single MoE expert linear weight.

    Holds learnable mask logits (sparsity-pattern selection per block) and
    optionally a learnable dense weight (`weight_master`). Delegates the
    quantization scheme to a pluggable `quant` component (or skips it for
    the sparsity-only baseline). Forward samples a soft mask via
    Gumbel-Softmax, multiplies the weight source by the mask, and passes
    the result through the quant component.

    The support (A) and the values (V) are independently freezable, which is
    what the joint-adaptation ablation varies:

      - learn_masks=True,  learn_weights=True  : joint adaptation (main path).
      - learn_masks=True,  learn_weights=False : support-only adaptation. The
        frozen values are the DENSE GPTQ solution, so every candidate position
        carries a meaningful value and alternative supports are comparable.
      - learn_masks=False, learn_weights=True  : fixed-mask QAT. No logits are
        allocated at all; the support is the initializer's, bit for bit, and the
        forward is deterministic (no Gumbel sample to make it wander).
      - learn_masks=False, learn_weights=False : nothing to train; the config
        validator points this at refine.enabled=false instead.

    Adding a new compression target (e.g. INT8) is a new `quant` component
    that exposes `fake_quantize(effective_weight)` and `quantize_hard(...)`.
    """

    def __init__(self, *, weight_shape, sparsity, quant, weight_buffer,
                 weight_master_init, learn_weights, mask_logits_init,
                 device, dtype, logits_dtype=None, fixed_mask=None,
                 weight_buffer_dtype=None, init_support=None, keep_weight_ref=False):
        super().__init__()
        self.weight_shape = tuple(weight_shape)
        self.device = device
        self.dtype = dtype
        self.logits_dtype = logits_dtype if logits_dtype is not None else dtype
        self.sparsity = sparsity
        self.quant = quant
        self.learn_weights = bool(learn_weights)
        self.learn_masks = mask_logits_init is not None

        if self.learn_masks:
            self.mask_logits = nn.Parameter(mask_logits_init.to(self.logits_dtype).detach())
            self.register_buffer("fixed_mask", None)
        else:
            if fixed_mask is None:
                raise ValueError("learn_masks=False requires fixed_mask (the initializer's support).")
            self.mask_logits = None
            self.register_buffer(
                "fixed_mask",
                fixed_mask.to(device=device, dtype=torch.bool).detach(),
            )

        if self.learn_weights:
            if weight_master_init is None:
                raise ValueError("learn_weights=True requires weight_master_init.")
            self.weight_master = nn.Parameter(weight_master_init.float().detach())
            self.register_buffer("W", None)
        else:
            if weight_buffer is None:
                raise ValueError("learn_weights=False requires weight_buffer.")
            buffer_dtype = weight_buffer_dtype if weight_buffer_dtype is not None else self.dtype
            self.register_buffer("W", weight_buffer.to(buffer_dtype).detach())

        # Diagnostics for the support/value ablation: how far the shipped support and the
        # shipped block scales moved from the initializer. Only meaningful where the
        # variable is trainable, so the buffers are allocated only there (the support one
        # costs 1 byte per weight, the scale one 4 bytes per group).
        self.register_buffer(
            "init_support",
            init_support.to(device=device, dtype=torch.bool).detach()
            if (self.learn_masks and init_support is not None) else None,
        )
        # Opt-in fp32 copy of the initializer's dense weight, for the drift diagnostic:
        # dL/dW is the upstream gradient multiplied ELEMENTWISE by the soft mask, but Lion
        # updates by sign(), so a weight the mask has pruned (m ~ 0, gradient ~ 0) takes the
        # same full step as one it has kept. Whether that random walk accumulates is what
        # separates "wasted compute" from "flipped-in weights arrive degraded". Doubles the
        # weight memory, so it is only allocated when asked for.
        self.register_buffer(
            "init_weight_ref",
            self._weight_source().detach().clone().float() if keep_weight_ref else None,
        )
        self.register_buffer("init_scale_ref", self.effective_scales())

    def _weight_source(self):
        return self.weight_master.float() if self.learn_weights else self.W

    def _sample_soft_mask(self, temperature, scale):
        eps = 1e-8
        u = torch.rand_like(self.mask_logits)
        noise = -torch.log(-torch.log(u + eps) + eps)
        return F.softmax(
            (self.mask_logits.float() * float(scale) + noise) / float(temperature),
            dim=-1,
        )

    def hard_mask(self):
        if self.learn_masks:
            return self.sparsity.hard_mask(self.mask_logits, self.weight_shape)
        return self.fixed_mask

    def effective_scales(self):
        """Dense-domain per-group scale the export would ship, or None.

        `hard_scales / global_scale` is the quantity that multiplies the FP4 codes,
        so it is the one comparable against the initializer's scales.
        """
        if self.quant is None or not hasattr(self.quant, "effective_scales"):
            return None
        with torch.no_grad():
            effective_weight = self.hard_mask().to(torch.float32) * self._weight_source()
            return self.quant.effective_scales(effective_weight)

    def forward(self, temperature, scale=1.0):
        if self.learn_masks:
            soft_mask = self._sample_soft_mask(temperature, scale)
            mask_in_weight_shape = self.sparsity.apply_soft_mask(soft_mask, self.weight_shape)
        else:
            mask_in_weight_shape = self.fixed_mask.to(torch.float32)
        effective_weight = mask_in_weight_shape * self._weight_source()
        if self.quant is None:
            return effective_weight.to(self.dtype)
        if getattr(self.quant, "needs_anneal", False):
            # GSQ samples its own grid assignment, so it needs the same annealed
            # temperature / logit scale the mask sampler uses. NVFP4Quant keeps its
            # 1-argument signature.
            return self.quant.fake_quantize(effective_weight, temperature, scale).to(self.dtype)
        return self.quant.fake_quantize(effective_weight).to(self.dtype)

    def get_hard_weights(self):
        hard_mask = self.hard_mask()
        if self.quant is None:
            output = hard_mask.to(self.dtype) * self._weight_source()
            return output.to(self.dtype), hard_mask

        effective_weight = hard_mask.to(torch.float32) * self._weight_source()
        output, scales, global_scale = self.quant.quantize_hard(effective_weight)
        return output.to(self.dtype), scales, global_scale, hard_mask

    # ---- sequential-arm phase switches -------------------------------------------------
    # The joint arm trains twice the variables of either one-sided arm, so joint > one-sided
    # confounds "both variables" with "simultaneously". The sequential arms hold the
    # trainable set fixed and vary only the coupling, which is what the method actually
    # claims. Both switches are one-way and idempotent.

    def freeze_support(self):
        """Discretize the current support and stop training it.

        Phase 2 must see the HARD mask -- leaving the Gumbel sampler on would keep the
        support moving under the values, which is the thing the sequential arm exists to
        rule out. `init_support` is left in place so mask_flip_frac still reads.
        """
        if not self.learn_masks:
            return
        with torch.no_grad():
            hard = self.hard_mask().to(device=self.device, dtype=torch.bool).detach()
        self.learn_masks = False
        self.mask_logits = None
        self.fixed_mask = hard

    def freeze_values(self):
        """Stop training the dense values, holding them at their current point.

        Kept in fp32 rather than cast to the storage dtype: a bf16 round-trip at the phase
        boundary would be a precision artifact that only the sequential arms pay, and it
        would show up as a loss the comparison would wrongly attribute to sequencing.
        """
        if not self.learn_weights:
            return
        with torch.no_grad():
            values = self.weight_master.detach().clone().float()
        self.learn_weights = False
        self.weight_master = None
        self.W = values


# Helpers for unpacking get_hard_weights() return tuples. Co-located with the
# producer so the tuple-shape contract and its consumers stay in lock-step.

def split_hard_output_full(hard_output):
    """Return (weight, scales, global_scale, mask), padding with None."""
    if not isinstance(hard_output, tuple):
        return hard_output, None, None, None
    if len(hard_output) == 2:
        weight, aux = hard_output
        if isinstance(aux, torch.Tensor) and aux.dtype == torch.bool:
            return weight, None, None, aux
        return weight, aux, None, None
    if len(hard_output) == 3:
        weight, aux, mask = hard_output
        return weight, aux, None, mask
    if len(hard_output) == 4:
        return hard_output
    raise ValueError(f"Unexpected hard-weight tuple length: {len(hard_output)}")


def split_hard_output(hard_output):
    weight, scales, _, mask = split_hard_output_full(hard_output)
    return weight, scales, mask


def hard_tensor(compressor):
    weight, _, _, _ = split_hard_output_full(compressor.get_hard_weights())
    return weight


def hard_pair(compressor):
    weight, scales, global_scale, mask = split_hard_output_full(compressor.get_hard_weights())
    if global_scale is not None:
        return weight, scales, global_scale, mask
    if mask is not None:
        return weight, scales, mask
    return weight, scales
