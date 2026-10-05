import torch

from ..base import BaseSparsity


class NoSparsity(BaseSparsity):
    """Dense placeholder so ExpertCompressor can run a quantization-only pipeline.

    GSQ is weight-quantization only -- there is no mask to learn -- but
    ExpertCompressor's forward is built around a sparsity component. Rather than
    branch that forward, this exposes the same interface with a single all-ones
    pattern per row: apply_soft_mask returns ones, hard_mask returns all-True.

    The logits it allocates are [rows, 1], and a softmax over a length-1 axis is
    identically 1.0, so the mask contributes no gradient and cannot drift. Cost is
    one logit per row, not per weight.
    """

    num_patterns = 1

    def __init__(self, device, dtype):
        super().__init__()
        self.device = device
        self.dtype = dtype

    def num_blocks(self, weight_shape):
        rows, _ = weight_shape
        return rows

    def init_mask_logits(self, init_support_mask, std, strength):
        rows = init_support_mask.shape[0]
        return torch.zeros(rows, 1, device=self.device, dtype=torch.float32)

    def apply_soft_mask(self, soft_mask, weight_shape):
        rows, cols = weight_shape
        # soft_mask is [rows, 1] and always exactly 1.0 (softmax over one entry).
        return soft_mask.expand(rows, cols)

    def hard_mask(self, mask_logits, weight_shape):
        return torch.ones(weight_shape, device=self.device, dtype=torch.bool)

    def _extract_init_mask(self, init_support_mask):
        rows = init_support_mask.shape[0]
        return torch.ones(rows, 1, device=self.device, dtype=torch.bool)
