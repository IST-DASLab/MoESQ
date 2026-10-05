import torch

from ..base import BaseSparsity


PAIRED_2_4_PATTERNS = (
    (1, 1, 0, 0),
    (1, 0, 1, 0),
    (1, 0, 0, 1),
    (0, 1, 1, 0),
    (0, 1, 0, 1),
    (0, 0, 1, 1),
)


class Paired24(BaseSparsity):
    block_size = 4
    num_patterns = 6

    def __init__(self, device, dtype):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.register_buffer(
            "possible_masks",
            torch.tensor(PAIRED_2_4_PATTERNS, dtype=dtype, device=device),
        )

    def num_blocks(self, weight_shape):
        rows, cols = weight_shape
        if cols % self.block_size != 0:
            raise ValueError(
                f"Paired24 expects weight width divisible by {self.block_size}, got {cols}."
            )
        return rows * cols // self.block_size

    def init_mask_logits(self, init_support_mask, std, strength):
        init_support_mask = init_support_mask.to(device=self.device, dtype=torch.bool)
        init_masks = self._extract_init_mask(init_support_mask)
        logits = torch.matmul(init_masks.float(), self.possible_masks.T.float())
        logits = logits - logits.mean(dim=-1, keepdim=True)
        return std * (torch.randn_like(logits) + logits * strength)

    def apply_soft_mask(self, soft_mask, weight_shape):
        return torch.matmul(soft_mask, self.possible_masks.float()).reshape(weight_shape)

    def hard_mask(self, mask_logits, weight_shape):
        idx = torch.argmax(mask_logits, dim=-1)
        return self.possible_masks[idx].reshape(weight_shape).bool()

    def _extract_init_mask(self, init_support_mask):
        if init_support_mask.shape[1] % self.block_size != 0:
            raise ValueError(
                f"Expected support mask columns divisible by {self.block_size}, got {init_support_mask.shape[1]}"
            )

        block_mask = init_support_mask.reshape(-1, self.block_size)
        valid_nnz = block_mask.sum(dim=-1) == 2
        matches = (block_mask.unsqueeze(1) == self.possible_masks.unsqueeze(0).bool()).all(dim=-1)
        valid_pattern = matches.sum(dim=-1) == 1
        valid = valid_nnz & valid_pattern
        if not valid.all():
            bad = (~valid).nonzero(as_tuple=False).flatten()[:8].tolist()
            raise ValueError(f"Support mask is not valid 2:4 sparsity; bad block ids: {bad}")

        matched_idx = matches.to(torch.int64).argmax(dim=-1)
        return self.possible_masks[matched_idx]
