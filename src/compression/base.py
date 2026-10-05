import torch.nn as nn


class BaseSparsity(nn.Module):
    """Pluggable sparsity component for ExpertCompressor.

    Concrete subclasses (e.g. Paired48, Paired24) must implement the
    differentiable / discrete mask interface used by ExpertCompressor.
    """

    block_size = None
    num_patterns = None

    def init_mask_logits(self, init_support_mask, std, strength):
        raise NotImplementedError

    def apply_soft_mask(self, soft_mask, weight_shape):
        raise NotImplementedError

    def hard_mask(self, mask_logits, weight_shape):
        raise NotImplementedError

    def _extract_init_mask(self, init_support_mask):
        raise NotImplementedError


class BaseQuant(nn.Module):
    """Pluggable weight-quantization component for ExpertCompressor.

    Concrete subclasses (e.g. NVFP4Quant) implement fake_quantize in the
    training forward and quantize_hard to materialize the deployable
    integer/FP4 tensor plus its scales.
    """

    def fake_quantize(self, effective_weight):
        raise NotImplementedError

    def quantize_hard(self, effective_weight):
        raise NotImplementedError


class BaseActivationQuant(nn.Module):
    """Pluggable activation-quantization component.

    Wraps the dynamic-local activation fake-quant path used when
    quantization.fake_quantize_activations is enabled.
    """

    def fake_quantize(self, x):
        raise NotImplementedError
