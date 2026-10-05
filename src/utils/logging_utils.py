import logging
import os
from datetime import datetime

import torch
import torch.distributed as dist
import wandb


class QuantizationLogger:
    def __init__(self, log_dir='logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.logger = logging.getLogger('quantization')
        self.logger.setLevel(logging.INFO)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fh = logging.FileHandler(
            os.path.join(log_dir, f'quantization_{timestamp}.log')
        )
        fh.setLevel(logging.INFO)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def log_config(self, config):
        self.logger.info("Configuration:")
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")

    def log_metrics(self, metrics, step=None):
        step_str = f" at step {step}" if step is not None else ""
        self.logger.info(f"Metrics{step_str}:")
        for key, value in metrics.items():
            self.logger.info(f"  {key}: {value}")


class MoeExpertTokenCounter:
    """Wraps `model._batched_expert_forward` to count tokens routed to each expert.

    Used during compression training to surface zero-token experts (dead experts
    that received no calibration tokens). Records counts only when armed via
    `arm_next_dispatch()`, so non-training forwards are not double-counted.
    """

    def __init__(self, model, world_size, global_rank, wandb_enabled):
        self.model = model
        self.world_size = world_size
        self.global_rank = global_rank
        self.use_dist = world_size > 1
        self.wandb_enabled = wandb_enabled

        self.counts = {}
        self._arm_next = False
        self._original_forward = None

    def install(self):
        if not getattr(self.model, "is_moe", False):
            return
        if not hasattr(self.model, "_batched_expert_forward"):
            return
        if self._original_forward is not None:
            return

        original = self.model._batched_expert_forward

        def counted(xin, eids, *args, **kwargs):
            self._record(eids)
            return original(xin, eids, *args, **kwargs)

        self._original_forward = original
        self.model._batched_expert_forward = counted

    def restore(self):
        if self._original_forward is None:
            return
        self.model._batched_expert_forward = self._original_forward
        self._original_forward = None

    def arm_next_dispatch(self):
        if getattr(self.model, "is_moe", False):
            self._arm_next = True

    def disarm(self):
        self._arm_next = False

    def reset(self):
        self.counts.clear()

    def _record(self, eids):
        if not self._arm_next:
            return
        self._arm_next = False
        if eids is None or eids.numel() == 0:
            return
        unique_eids, cnts = torch.unique(
            eids.detach().to(torch.long).cpu(), sorted=True, return_counts=True
        )
        for expert_id, count in zip(unique_eids.tolist(), cnts.tolist()):
            self.counts[int(expert_id)] = (
                self.counts.get(int(expert_id), 0) + int(count)
            )

    @staticmethod
    def _expert_id_from_tensor_name(tensor_name):
        marker = ".experts."
        if marker not in tensor_name:
            return None
        expert_id = tensor_name.split(marker, 1)[1].split(".", 1)[0]
        return int(expert_id) if expert_id.isdigit() else None

    def report_zero_token_experts(self, layer_name, compressors, logger):
        if not getattr(self.model, "is_moe", False):
            return
        if self._original_forward is None and not self.counts:
            return

        local_experts = set()
        for tensor_name in compressors:
            expert_id = self._expert_id_from_tensor_name(tensor_name)
            if expert_id is not None:
                local_experts.add(expert_id)

        if self.use_dist:
            gathered = [None for _ in range(self.world_size)]
            dist.all_gather_object(gathered, (local_experts, self.counts))
            if self.global_rank != 0:
                return
            expert_ids = set()
            token_counts = {}
            for rank_experts, rank_counts in gathered:
                expert_ids.update(rank_experts)
                for expert_id, count in rank_counts.items():
                    token_counts[expert_id] = token_counts.get(expert_id, 0) + count
        else:
            expert_ids = local_experts
            token_counts = self.counts

        zero_token_experts = [e for e in sorted(expert_ids) if token_counts.get(e, 0) == 0]
        if logger is not None:
            if zero_token_experts:
                logger.warning(
                    f"Layer {layer_name} - MoESQ zero-token experts: {zero_token_experts}"
                )
            else:
                logger.info(f"Layer {layer_name} - MoESQ zero-token experts: none")

        if self.wandb_enabled and self.global_rank == 0:
            per_expert_counts = [token_counts.get(e, 0) for e in expert_ids]
            wandb.log({
                f"{layer_name}/zero_token_expert_count": len(zero_token_experts),
                f"{layer_name}/zero_token_expert_fraction": (
                    len(zero_token_experts) / len(expert_ids) if expert_ids else 0.0
                ),
                f"{layer_name}/min_expert_tokens": min(per_expert_counts) if per_expert_counts else 0,
            })
            wandb.summary[f"{layer_name}/zero_token_experts"] = (
                ",".join(map(str, zero_token_experts)) if zero_token_experts else "none"
            )
