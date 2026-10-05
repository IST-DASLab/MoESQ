import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import math
import time
from lion_pytorch import Lion
import torch.nn.functional as F
import torch.distributed as dist
from src.compression import build_compressor, hard_pair, hard_tensor
from src.compression.scale_finetune import ScaleFinetuneCompressor
from src.utils.logging_utils import MoeExpertTokenCounter
from src.utils.progress_reporter import report_refine_epoch, report_refine_step

class MultiOptimizer:
    """Presents several optimizers as one.

    The mask logits are a 6-way softmax whose gradient dies once the anneal saturates it
    (scale/temp reaches 10000; p_max > 0.99 on 98.8% of blocks). Lion's sign() keeps taking
    full-size steps on stale momentum through that dead zone; Adam's m/sqrt(v) decays to
    zero with the signal. Running Adam on the logits and Lion on the weights needs two
    optimizers, and every call site here reads `.param_groups` / `.step()` / `.zero_grad()`,
    so this exposes exactly that. Group dicts are shared by reference, so the LR scheduler
    mutating them still works.
    """

    def __init__(self, optimizers):
        self.optimizers = list(optimizers)

    @property
    def param_groups(self):
        return [g for o in self.optimizers for g in o.param_groups]

    def step(self):
        for o in self.optimizers:
            o.step()

    def zero_grad(self, set_to_none=True):
        for o in self.optimizers:
            o.zero_grad(set_to_none=set_to_none)


class CompressionTrainer:
    def __init__(self, model, config, dtype, self_attn=False):
        self.model = model
        self.config = config
        self.compressors = {}
        self.optimizer = None
        self.scheduler = None
        self.device = model.device
        self.dtype = dtype
        self.loss_fn = nn.MSELoss(reduction='mean')
        self.optimizer_params = []
        self.min_loss = float('inf')
        self.routing_cache = None
        self.batch_size = self.config.data.batch_size
        self.global_rank = getattr(self.model, 'rank', 0)
        self.world_size = getattr(self.model, 'world_size', 1)
        self.use_dist = self.world_size > 1
        if hasattr(self.model, 'save_dir'):
            self.model.save_dir = config.training.checkpoint_dir
        self.train_attn = self_attn
        self.token_counter = MoeExpertTokenCounter(
            self.model, self.world_size, self.global_rank,
            wandb_enabled=self.config.wandb.enabled,
        )

    def _ensure_coupling_stats(self):
        if not hasattr(self, '_coupling_stats'):
            self._coupling_stats = dict(groups=0.0, dropped=0.0, kept=0.0,
                                        ratio=0.0, ste_resid=0.0, r_mag=0.0)
        return self._coupling_stats

    def setup_layer_training(self, tensor_name, init_compressed_weight, init_scales,
                             init_dense_weight=None, init_support_mask=None):
        if self.config.refine.coupling_diagnostics:
            self.config.refine._coupling_stats = self._ensure_coupling_stats()
        compressor = build_compressor(
            self.config, self.device, self.dtype,
            init_compressed_weight=init_compressed_weight,
            init_scales=init_scales,
            init_dense_weight=init_dense_weight,
            init_support_mask=init_support_mask,
        )

        if self.use_dist and not self.model.is_moe:
            for p in compressor.parameters():
                dist.broadcast(p.data, src=0)

        self.compressors[tensor_name] = compressor
        self._append_param_groups(compressor)

    def _append_param_groups(self, compressor):
        named_params = list(compressor.named_parameters())
        # GSQ splits into grid logits and group scales, which the paper trains at
        # different rates (Tables 7/8: scale is 2-20x slower than the assignments).
        # Both fall back to masks_lr when unset, so NVFP4 runs are unchanged.
        refine = self.config.refine
        logits_lr = refine.logits_lr if refine.logits_lr is not None else refine.masks_lr
        scales_lr = refine.group_scales_lr if refine.group_scales_lr is not None else refine.masks_lr

        weight_params = [p for n, p in named_params if n == 'weight_master']
        gsq_logit_params = [p for n, p in named_params if n == 'quant.logits']
        gsq_scale_params = [p for n, p in named_params if n == 'quant.scale']
        mask_params = [
            p for n, p in named_params
            if n not in ('weight_master', 'quant.logits', 'quant.scale')
        ]
        mask_wd = (refine.mask_weight_decay if refine.mask_weight_decay is not None
                   else refine.weight_decay)
        if mask_params:
            self.optimizer_params.append({
                'name': 'mask_logits',
                'params': mask_params,
                'lr': refine.masks_lr,
                'weight_decay': mask_wd,
                'lr_decay_tag': True,
            })
        if gsq_logit_params:
            self.optimizer_params.append({
                'name': 'gsq_logits',
                'params': gsq_logit_params,
                'lr': logits_lr,
                'weight_decay': refine.weight_decay,
                'lr_decay_tag': True,
            })
        if gsq_scale_params:
            self.optimizer_params.append({
                'name': 'gsq_group_scales',
                'params': gsq_scale_params,
                'lr': scales_lr,
                'weight_decay': refine.weight_decay,
                'lr_decay_tag': True,
            })
        if weight_params:
            self.optimizer_params.append({
                'name': 'weight_master',
                'params': weight_params,
                'lr': self.config.refine.weights_lr,
                'weight_decay': self.config.refine.weight_decay,
                'lr_decay_tag': True,
            })

    def _make_optimizer(self):
        """Lion over everything (default), or Adam on the mask logits and Lion on the rest."""
        betas = tuple(self.config.refine.lion_betas)
        which = (self.config.refine.mask_optimizer or 'lion').lower()
        if which == 'lion':
            return Lion(self.optimizer_params, betas=betas)
        mask_groups = [g for g in self.optimizer_params if g.get('name') == 'mask_logits']
        other_groups = [g for g in self.optimizer_params if g.get('name') != 'mask_logits']
        if not mask_groups:
            return Lion(other_groups, betas=betas)
        cls = torch.optim.AdamW if which == 'adamw' else torch.optim.Adam
        opts = [cls(mask_groups, betas=tuple(self.config.refine.mask_adam_betas))]
        if other_groups:
            opts.append(Lion(other_groups, betas=betas))
        return MultiOptimizer(opts)

    def _rebuild_optimizer_params(self):
        self.optimizer_params = []
        for compressor in self.compressors.values():
            self._append_param_groups(compressor)

    def _switch_sequential_phase(self, layer_name, remaining_steps, logging):
        """Hand the layer from phase 1 to phase 2 of a sequential arm.

        Freezes whichever variable phase 1 trained, then rebuilds the optimizer and the
        scheduler over the remaining steps -- Lion's momentum is per-parameter and the
        phase-2 set is disjoint from phase 1's, so there is no state worth carrying.
        """
        mode = self.config.refine.sequential
        for compressor in self.compressors.values():
            if mode == 'support_first':
                compressor.freeze_support()
            else:
                compressor.freeze_values()
        self._rebuild_optimizer_params()
        self.optimizer = self._make_optimizer()
        self.scheduler = CustomLRScheduler(
            self.optimizer, max(1, remaining_steps),
            self.config.refine.warmup_steps,
            lr_decay_type=self.config.refine.lr_decay_type,
            min_lr=self.config.refine.scheduler_min_lr,
        )
        if self.global_rank == 0 and logging is not None:
            frozen = 'support' if mode == 'support_first' else 'values'
            trains = 'values' if mode == 'support_first' else 'support'
            logging.info(
                f'Layer {layer_name} - sequential switch: froze {frozen}, '
                f'phase 2 trains {trains} for {remaining_steps} steps.')

    def train_layer(self, layer_name, train_all, val_all, logging,
                    layer_idx=None, num_layers=None):
        if logging is not None:
            logging = logging.logger
        num_epochs = self.config.refine.num_epochs
        num_samples = train_all['input'].shape[0]
        batch_size = self.config.data.batch_size // self.world_size

        self.optimizer = self._make_optimizer()
        self._logit_grad_stats = dict(blocks=0.0, coords=0.0, top2_frac=0.0,
                                      negligible=0.0, zerosum_resid=0.0)
        if self.config.refine.coupling_diagnostics:
            for c in self.compressors.values():
                if getattr(c.quant, '_coupling_stats', None) is None:
                    c.quant._coupling_stats = self._coupling_stats
        self.token_counter.reset()
        self.token_counter.install()

        num_training_steps = (num_samples + batch_size - 1) // batch_size * num_epochs
        steps_per_epoch = (num_samples + batch_size - 1) // batch_size
        self.scheduler = CustomLRScheduler(
            self.optimizer, num_training_steps,
            self.config.refine.warmup_steps,
            lr_decay_type=self.config.refine.lr_decay_type,
            min_lr=self.config.refine.scheduler_min_lr,
        )

        initial_temperature, final_temperature = self.config.refine.temperature
        initial_scale, final_scale = self.config.refine.scale

        step = 0
        phase_start = time.time()
        step_report_interval = max(1, steps_per_epoch // self.config.logging.step_report_divisor)

        micro = max(1, self.config.refine.device_microbatch_size)
        layer_train_losses = []
        # Optional best-epoch selection (refine.best_epoch_snapshot, deprecated): track
        # the hard val loss -- invariant to the temperature anneal, unlike the soft train
        # loss -- and restore the best state before export. The GPTQ init counts as
        # epoch 0. The metric is all-reduced across ranks, so every rank picks the same
        # epoch.
        snapshot_on = self.config.refine.best_epoch_snapshot
        best_val_hard, best_epoch, best_state = float('inf'), None, None
        best_metrics = (float('nan'), float('nan'))  # (recon_unw, block) at the best epoch
        last_val_hard = float('nan')
        # The temperature / logit-scale anneal must span whichever phase is LEARNING the
        # mask. Annealing across the whole run in a sequential arm would harden a
        # half-annealed support at the switch (support_first) or start the support phase
        # already cold (values_first) -- a handicap the comparison would misread as a cost
        # of sequencing. Joint runs are unchanged: lo=0, hi=num_epochs-1.
        seq_mode = self.config.refine.sequential
        p1 = self.config.refine.sequential_phase1_epochs
        if seq_mode == 'support_first':
            anneal_lo, anneal_hi = 0, p1 - 1
        elif seq_mode == 'values_first':
            anneal_lo, anneal_hi = p1, num_epochs - 1
        else:
            anneal_lo, anneal_hi = 0, num_epochs - 1

        for epoch in range(num_epochs):
            if seq_mode is not None and epoch == p1:
                self._switch_sequential_phase(
                    layer_name, (num_epochs - p1) * steps_per_epoch, logging)
            span = anneal_hi - anneal_lo
            t = min(1.0, max(0.0, (epoch - anneal_lo) / span)) if span > 0 else 0.0
            temperature = initial_temperature + (final_temperature - initial_temperature) * t
            scale = initial_scale + (final_scale - initial_scale) * t

            if epoch == 0:
                (_, init_val_hard, init_recon, init_block, init_sink,
                 init_blk_sink, init_blk_ord) = self._validate_epoch(
                    val_all, batch_size, temperature, scale, micro)
                last_val_hard = init_val_hard
                if self.global_rank == 0:
                    # the GPTQ-init point of every per-layer curve; without it the
                    # refinement gain per layer cannot be read from the log
                    extra = (f', Recon(unw) = {init_recon:.2e}, Block = {init_block:.2e}'
                             if init_recon == init_recon else '')
                    logging.info(f'Layer {layer_name} - Epoch 0 (init): '
                                 f'Val Hard Loss = {init_val_hard:.2e}{extra}')
                    if init_sink == init_sink:
                        # dense-model property, logged once per layer: where the sink lives
                        logging.info(f'Layer {layer_name} - dense block-output concentration: '
                                     f'top-8 tokens per micro-batch carry {100 * init_sink:.1f}% of '
                                     f'sum ||y||^2 (sink layer if >> 10%)')
                        # GPTQ-init block error split by the same top-8 tokens. Compare
                        # block_err_ord ACROSS layers to tell a badly-fit layer from one
                        # whose metric is merely sink-weighted.
                        logging.info(f'Layer {layer_name} - init block error split: '
                                     f'ordinary {init_blk_ord:.3e}  sink {init_blk_sink:.3e}  '
                                     f'(sink/ordinary = {init_blk_sink/init_blk_ord:.1f}x)')
                        if self.config.wandb.enabled:
                            wandb.log({f"{layer_name}/val_sink_share_top8": init_sink,
                                       f"{layer_name}/init_block_err_ordinary": init_blk_ord,
                                       f"{layer_name}/init_block_err_sink": init_blk_sink})
                if snapshot_on:
                    best_val_hard, best_epoch, best_state = init_val_hard, 0, self._snapshot_params()
                    best_metrics = (init_recon, init_block)

            epoch_losses = []
            epoch_start = time.time()

            for indices in self.get_random_batch_indices(num_samples, batch_size):
                temperature = initial_temperature + (final_temperature - initial_temperature) * step / (num_training_steps - 1)
                scale = initial_scale + (final_scale - initial_scale) * step / (num_training_steps - 1)
                loss = self.train_step(
                    train_all['input'][indices],
                    temperature,
                    scale,
                    micro
                )
                epoch_losses.append(loss)

                if self.global_rank == 0:
                    report_refine_step(step, num_training_steps, loss,
                                       interval=step_report_interval)

                if self.config.wandb.enabled and self.global_rank == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    wandb.log({
                        "train/step_loss": loss,
                        "train/learning_rate": current_lr,
                        "train/temperature": temperature,
                        "train/scale": scale,
                        "train/global_step": step,
                    })

                step += 1

            (avg_val_soft_loss, avg_val_hard_loss, avg_val_recon, avg_val_block, _,
             avg_val_blk_sink, avg_val_blk_ord) = \
                self._validate_epoch(val_all, batch_size, temperature, scale, micro)
            has_recon = avg_val_recon == avg_val_recon  # not NaN (MoE layers only)
            avg_train_loss = sum(epoch_losses) / len(epoch_losses)
            last_val_hard = avg_val_hard_loss
            if snapshot_on and avg_val_hard_loss < best_val_hard:
                best_val_hard, best_epoch, best_state = avg_val_hard_loss, epoch + 1, self._snapshot_params()
                best_metrics = (avg_val_recon, avg_val_block)
            epoch_time = time.time() - epoch_start
            phase_elapsed = time.time() - phase_start

            layer_train_losses.append(avg_train_loss)
            best_train_loss = min(layer_train_losses)
            # train loss rising for 3 consecutive epochs signals an oscillating layer
            rising = (len(layer_train_losses) >= 3
                      and layer_train_losses[-1] > layer_train_losses[-2] > layer_train_losses[-3])

            if self.global_rank == 0:
                if not math.isfinite(avg_train_loss):
                    logging.error(
                        f'[DIVERGENCE] Layer {layer_name} epoch {epoch+1}: '
                        f'train loss is {avg_train_loss} (non-finite). This layer is unrecoverable.'
                    )
                elif rising:
                    logging.warning(
                        f'[DIVERGENCE] Layer {layer_name} epoch {epoch+1}: train loss rising '
                        f'3 epochs in a row ({layer_train_losses[-3]:.3e} -> '
                        f'{layer_train_losses[-2]:.3e} -> {layer_train_losses[-1]:.3e}); '
                        f'best was {best_train_loss:.3e}. Suspect LR too high for this arm.'
                    )

            if self.global_rank == 0:
                extra = ''
                if has_recon:
                    extra = (f', Recon(unw) = {avg_val_recon:.2e}, '
                             f'Block = {avg_val_block:.2e}')
                logging.info(
                    f'Layer {layer_name} - Epoch {epoch+1}: '
                    f'Train Loss = {avg_train_loss:.2e}, '
                    f'Val Soft Loss = {avg_val_soft_loss:.2e}, '
                    f'Val Hard Loss = {avg_val_hard_loss:.2e}'
                    f'{extra}'
                )
                report_refine_epoch(
                    layer_name, epoch, num_epochs, phase_elapsed,
                    avg_train_loss=avg_train_loss,
                    avg_val_loss=avg_val_hard_loss,
                    epoch_time=epoch_time,
                    temperature=temperature,
                    scale=scale,
                )

            if self.config.wandb.enabled and self.global_rank == 0:
                wandb.log({
                    f"{layer_name}/train_loss": avg_train_loss,
                    f"{layer_name}/val_soft_loss": avg_val_soft_loss,
                    f"{layer_name}/val_hard_loss": avg_val_hard_loss,
                    f"{layer_name}/temperature": temperature,
                    f"{layer_name}/scale": scale,
                    f"{layer_name}/epoch": epoch + 1,
                    f"{layer_name}/epoch_time_sec": epoch_time,
                    f"{layer_name}/train_loss_over_best": (
                        avg_train_loss / best_train_loss
                        if math.isfinite(avg_train_loss) and best_train_loss > 0 else float('inf')
                    ),
                    **({
                        f"{layer_name}/val_recon_unweighted": avg_val_recon,
                        f"{layer_name}/val_block_error": avg_val_block,
                    } if has_recon else {}),
                })

            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()

        self.report_layer_convergence(layer_name, layer_train_losses, logging)

        if self.use_dist:
            dist.barrier()

        self.token_counter.report_zero_token_experts(layer_name, self.compressors, logging)
        self.token_counter.restore()

        if snapshot_on:
            self._restore_best_epoch(layer_name, best_state, best_epoch, best_val_hard,
                                     last_val_hard, num_epochs, logging, best_metrics)
        self._report_logit_gradients(layer_name, logging)
        if self.config.refine.coupling_diagnostics:
            self._report_coupling(layer_name, logging)
        if self.config.refine.weight_drift_diagnostics:
            self._log_weight_drift(layer_name, logging)
        self._log_adaptation_diagnostics(layer_name, logging)
        if self.config.refine.error_decomposition:
            self._log_error_decomposition(layer_name, val_all, batch_size, micro, logging)

        if self.config.refine.scale_ft_epochs > 0:
            self._scale_finetune_layer(layer_name, train_all, val_all, batch_size, micro, logging)

        for tensor_name, compressor in self.compressors.items():
            if self.train_attn:
                self.model.update_compressed_weights(tensor_name, hard_pair(compressor))
            else:
                if "gate_proj" in tensor_name:
                    base = tensor_name[: -len(".gate_proj")]
                    pairs = {
                        "gate_proj": hard_pair(compressor),
                        "up_proj": hard_pair(self.compressors[f"{base}.up_proj"]),
                        "down_proj": hard_pair(self.compressors[f"{base}.down_proj"])
                    }
                    if self.model.is_moe or self.global_rank == 0:
                        self.model.save_to_disc(base, pairs)

        if self.use_dist:
            dist.barrier()
        self.compressors.clear()

        del self.optimizer, self.optimizer_params, self.compressors
        torch.cuda.empty_cache()

    def report_layer_convergence(self, layer_name, layer_train_losses, logging):
        if self.global_rank != 0 or not layer_train_losses:
            return False

        warn_ratio = self.config.refine.divergence_warn_ratio
        first, final = layer_train_losses[0], layer_train_losses[-1]
        best = min(layer_train_losses)
        finite = all(math.isfinite(v) for v in layer_train_losses)
        ratio = final / best if finite and best > 0 else float('inf')
        diverged = warn_ratio > 0 and (not finite or ratio > warn_ratio)

        if diverged and not finite:
            logging.error(
                f'[DIVERGENCE] Layer {layer_name} did NOT converge: train loss went '
                f'non-finite (NaN/Inf) during refinement. Epoch losses: '
                f'{[f"{v:.3e}" for v in layer_train_losses]}. This layer is unrecoverable.'
            )
        elif diverged:
            logging.warning(
                f'[DIVERGENCE] Layer {layer_name} did NOT converge: train loss '
                f'{first:.3e} (epoch 1) -> {final:.3e} (epoch {len(layer_train_losses)}), '
                f'best {best:.3e} at epoch {layer_train_losses.index(best)+1}; '
                f'final/best = {ratio:.2f}x > {warn_ratio:.2f}x threshold. '
                f'This layer is compressed with worse weights than it reached mid-training.'
            )
        else:
            logging.info(
                f'Layer {layer_name} converged: train loss {first:.3e} -> {final:.3e} '
                f'(final/best = {ratio:.2f}x)'
            )

        if self.config.wandb.enabled and wandb.run is not None:
            wandb.log({
                f"{layer_name}/train_loss_final_over_best": ratio,
                f"{layer_name}/diverged": int(diverged),
            })
            if diverged:
                summary = wandb.run.summary
                summary["refine/diverged_layer_count"] = (
                    summary.get("refine/diverged_layer_count") or 0) + 1
                seen = summary.get("refine/diverged_layers") or ""
                summary["refine/diverged_layers"] = f"{seen},{layer_name}".lstrip(",")

        return diverged

    def _validate_epoch(self, val_all, batch_size, temperature, scale, microbatch_size):
        """Run validation_step over val_all; return mean (soft, hard, recon_unw, block,
        sink_share, block_err_sink, block_err_ord).

        All but soft/hard are NaN outside MoE layers, mirroring validation_step.
        sink_share = share of the dense block-output energy on the top-8 tokens per
        (rank, micro-batch): ~1% on an ordinary layer, ~90-100% where the attention-sink
        massive activation is produced. A property of the dense model, not of the epoch.
        block_err_{sink,ord} split the block ERROR by those same top-8 tokens, on the same
        per-element normalisation as block_error, so the three are directly comparable.
        The split is what separates "this layer is badly fit" from "this layer's metric is
        sink-weighted": compare block_err_ord across layers, not block_error.
        """
        soft, hard, recon, block, sink, bsink, bord = [], [], [], [], [], [], []
        num_batches = (val_all['input'].shape[0] + batch_size - 1) // batch_size
        with torch.no_grad():
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, val_all['input'].shape[0])
                s, h, r, b, k, bs, bo = self.validation_step(
                    val_all['input'][start_idx:end_idx], temperature, scale, microbatch_size)
                soft.append(s)
                hard.append(h)
                if r == r:  # not NaN (MoE layers only)
                    recon.append(r)
                    block.append(b)
                    sink.append(k)
                    bsink.append(bs)
                    bord.append(bo)
        mean = lambda xs: sum(xs) / len(xs) if xs else float('nan')
        return (mean(soft), mean(hard), mean(recon), mean(block), mean(sink),
                mean(bsink), mean(bord))

    def _snapshot_params(self):
        # Only trainable tensors: buffers (pattern tables, group indices) never change.
        return {
            (tensor_name, pname): p.detach().clone()
            for tensor_name, compressor in self.compressors.items()
            for pname, p in compressor.named_parameters()
        }

    def _load_params(self, state):
        with torch.no_grad():
            for tensor_name, compressor in self.compressors.items():
                for pname, p in compressor.named_parameters():
                    p.copy_(state[(tensor_name, pname)])

    def _restore_best_epoch(self, layer_name, best_state, best_epoch, best_val_hard,
                            last_val_hard, num_epochs, logging, best_metrics=(float('nan'),) * 2):
        if best_state is None:
            return
        shipped_last = best_epoch == num_epochs
        ratio = (last_val_hard / best_val_hard
                 if best_val_hard > 0 and math.isfinite(last_val_hard) else float('inf'))
        if not shipped_last:
            self._load_params(best_state)
        if self.global_rank == 0:
            which = 'GPTQ init' if best_epoch == 0 else f'epoch {best_epoch}'
            if shipped_last:
                logging.info(f'Layer {layer_name} best epoch is the last one ({num_epochs}); shipping as is.')
            else:
                logging.warning(
                    f'Layer {layer_name} shipping {which} (val hard {best_val_hard:.3e}) instead of '
                    f'epoch {num_epochs} ({last_val_hard:.3e}); last/best = {ratio:.2f}x.'
                )
            if self.config.wandb.enabled:
                # the shipped layer's metrics: wandb's summary keeps only the LAST epoch
                # of the per-epoch keys, which is not what ships once a restore happens
                wandb.log({
                    f"{layer_name}/best_epoch": best_epoch,
                    f"{layer_name}/val_hard_last_over_best": ratio,
                    f"{layer_name}/shipped_best_epoch": int(not shipped_last),
                    f"{layer_name}/best_val_hard_loss": best_val_hard,
                    **({
                        f"{layer_name}/best_val_recon_unweighted": best_metrics[0],
                        f"{layer_name}/best_val_block_error": best_metrics[1],
                    } if best_metrics[0] == best_metrics[0] else {}),
                })
        best_state.clear()

    def _log_adaptation_diagnostics(self, layer_name, logging):
        """How far the shipped support and block scales moved from the initializer.

        The support/value ablation cannot be read without these: a mask-learning arm whose
        flip fraction is ~0 has an optimizer that never moved, which is a different finding
        from support adaptation not helping. The scale pair is the coupling the joint
        hypothesis rests on -- the exported scale sits on the FP8 grid, so `scale_moved_frac`
        counts groups whose SHIPPED value actually changed, not groups that drifted inside a
        grid cell. Measured on the refinement output, before any post-hoc scale stage.
        """
        totals = torch.zeros(5, device=self.device, dtype=torch.float64)
        with torch.no_grad():
            for compressor in self.compressors.values():
                init_support = getattr(compressor, 'init_support', None)
                block_size = getattr(compressor.sparsity, 'block_size', None)
                if init_support is not None and block_size:
                    changed = (compressor.hard_mask() != init_support).reshape(-1, block_size).any(dim=-1)
                    totals[0] += changed.sum()
                    totals[1] += changed.numel()
                ref = getattr(compressor, 'init_scale_ref', None)
                cur = compressor.effective_scales() if ref is not None else None
                if cur is not None:
                    totals[2] += (cur != ref).sum()
                    totals[3] += ref.numel()
                    totals[4] += (cur / ref - 1.0).abs().sum()
        if self.use_dist:
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        if self.global_rank != 0:
            return
        blocks, groups = totals[1].item(), totals[3].item()
        if not blocks and not groups:
            return
        metrics = {}
        if blocks:
            metrics[f"{layer_name}/mask_flip_frac"] = totals[0].item() / blocks
        if groups:
            metrics[f"{layer_name}/scale_moved_frac"] = totals[2].item() / groups
            metrics[f"{layer_name}/scale_rel_change_mean"] = totals[4].item() / groups
        logging.info(
            f'Layer {layer_name} - adaptation: ' + '  '.join(
                f'{k.split("/")[-1]} = {v:.4f}' for k, v in metrics.items()))
        if self.config.wandb.enabled:
            wandb.log(metrics)

    def _log_weight_drift(self, layer_name, logging):
        """Where did the weight optimizer actually spend its steps?

        dL/dW is the upstream gradient multiplied elementwise by the soft mask, so a pruned
        position gets ~no gradient -- but Lion's sign() gives it a full step anyway. If that
        random walk accumulates, the ~50% of positions the mask discards are not merely wasted
        compute: the 18% of blocks that FLIP pull their weights out of that degraded pool.
        Reported as RMS displacement from the initializer, split by what the support did.
        """
        tot = torch.zeros(8, device=self.device, dtype=torch.float64)
        with torch.no_grad():
            for c in self.compressors.values():
                ref = getattr(c, 'init_weight_ref', None)
                init_sup = getattr(c, 'init_support', None)
                if ref is None or init_sup is None or not c.learn_weights:
                    continue
                d = (c.weight_master.detach().float() - ref) ** 2
                final = c.hard_mask()
                kept = init_sup & final          # kept throughout
                dropped = init_sup & ~final      # flipped OUT
                added = ~init_sup & final        # flipped IN -- trained while pruned
                dead = ~init_sup & ~final        # pruned throughout
                for i, sel in enumerate((kept, added, dropped, dead)):
                    tot[2 * i] += d[sel].sum()
                    tot[2 * i + 1] += sel.sum()
        if self.use_dist:
            dist.all_reduce(tot, op=dist.ReduceOp.SUM)
        if self.global_rank != 0 or logging is None or tot[1].item() == 0:
            return
        names = ('kept', 'flipped_in', 'flipped_out', 'never_kept')
        rms, frac = {}, {}
        total_n = sum(tot[2 * i + 1].item() for i in range(4))
        for i, nm in enumerate(names):
            n = tot[2 * i + 1].item()
            rms[nm] = (tot[2 * i].item() / n) ** 0.5 if n else float('nan')
            frac[nm] = n / total_n if total_n else 0.0
        logging.info(
            f'Layer {layer_name} - weight drift from init (RMS): ' + '  '.join(
                f'{nm} {rms[nm]:.3e} ({frac[nm]*100:.1f}%)' for nm in names))
        if rms['kept'] > 0:
            logging.info(
                f'Layer {layer_name} - drift ratio vs kept: ' + '  '.join(
                    f'{nm} {rms[nm]/rms["kept"]:.2f}x' for nm in names[1:]))
        if self.config.wandb.enabled:
            wandb.log({f"{layer_name}/weight_drift_{nm}": rms[nm] for nm in names})

    def _original_weight(self, tensor_name):
        """The layer's pre-compression weight, or None if the model cannot resolve it.

        GPTQ does not write back into the module (`GPTQ.fasterquant` clones the weight to
        score its own loss), and the refinement forward takes compressed weights as an
        argument rather than installing them, so the original is live right up to export.
        """
        try:
            module = self.model._get_layer_by_name(tensor_name)
        except Exception:
            return None
        weight = getattr(module, 'weight', None)
        return None if weight is None else weight.data

    def _decomposition_weights(self, variant):
        """Weight dict for one point of the error ladder, or None if unavailable."""
        weights = {}
        for tensor_name, compressor in self.compressors.items():
            original = self._original_weight(tensor_name)
            if original is None or compressor.quant is None:
                return None
            W0 = original.to(torch.float32)
            if tuple(W0.shape) != tuple(compressor.weight_shape):
                return None
            if variant == 'quant_only':
                w = compressor.quant.quantize_hard(W0)[0]
            else:
                mask = compressor.hard_mask()
                masked = mask.to(torch.float32) * W0
                if variant == 'sparse_only':
                    w = masked
                else:
                    quantized = compressor.quant.quantize_hard(masked)[0]
                    # quant_on_support keeps W0 on the pruned positions, so its residual is
                    # EXACTLY the rounding the sparse model actually pays; sparse_quant zeroes
                    # them. In weight space the two residuals then add to sparse_quant's
                    # exactly, which is what makes the interaction term well posed.
                    w = quantized if variant == 'sparse_quant' else torch.where(mask, quantized, W0)
            w = w.to(self.dtype)
            if self.model.is_moe:
                prefix, leaf = tensor_name.rsplit(".", 1)
                weights.setdefault(prefix, {})[leaf] = w
            else:
                weights[tensor_name] = w
        return weights

    def _recon_metrics_for(self, weights, val_all, batch_size, microbatch_size):
        """(recon_unweighted, block_error, block_error_ordinary) for a weight dict."""
        recon_se = recon_n = block_se = block_n = sink_se = sink_n = 0.0
        num_batches = (val_all['input'].shape[0] + batch_size - 1) // batch_size
        accumulation_steps = max(1, batch_size // max(1, microbatch_size))
        with torch.no_grad():
            for batch_idx in range(num_batches):
                batch = val_all['input'][batch_idx * batch_size:(batch_idx + 1) * batch_size]
                for i in range(accumulation_steps):
                    micro = batch[i * microbatch_size:(i + 1) * microbatch_size]
                    if micro.shape[0] == 0:
                        continue
                    m = self.model.moe_val_recon_metrics(
                        micro.to(self.device), weights,
                        fake_act_quant=self.config.compression.fake_quantize_activations)
                    rse, rn, bse, bn = m[:4]
                    recon_se += rse.item(); recon_n += rn.item()
                    block_se += bse.item(); block_n += bn.item()
                    # Split by the same top-8 sink tokens the epoch curve uses. Without it a
                    # sink layer's decomposition cannot be read: layers 2-3 carry ~50% of their
                    # block energy on those tokens, so "rounding dominates sparsity" there could
                    # be a statement about the sink rather than about the layer.
                    sink_se += m[6].item(); sink_n += m[7].item()
        if self.use_dist:
            r = torch.tensor([recon_se, recon_n, block_se, block_n, sink_se, sink_n],
                             device=self.device, dtype=torch.float64)
            dist.all_reduce(r, op=dist.ReduceOp.SUM, group=dist.group.WORLD)
            recon_se, recon_n, block_se, block_n, sink_se, sink_n = r.tolist()
        # ordinary = everything the sink split did not claim, on the same normalisation
        ord_se, ord_n = block_se - sink_se, block_n - sink_n
        return (recon_se / recon_n if recon_n else float('nan'),
                block_se / block_n if block_n else float('nan'),
                ord_se / ord_n if ord_n else float('nan'))

    def _log_error_decomposition(self, layer_name, val_all, batch_size, microbatch_size, logging):
        """How the shipped error accumulates: sparsity, quantization, and their interaction.

        Three extra points, all on the ORIGINAL weight W0 (which is still live in the module)
        and all through the same recon metric as the epoch curve, so the five numbers form
        one ladder per layer:

            E_S   = err(M . W0)                  sparsity alone, no quantizer
            E_Q   = err(Q(W0))                   quantization alone, no mask (dense NVFP4)
            E_R   = err(where(M, Q(M.W0), W0))   rounding ON THE SUPPORT, nothing pruned
            E_SQ  = err(Q(M . W0))               both constraints, no GPTQ, no refinement
            E_init  (epoch 0, logged)            both + GPTQ initialization
            E_ship  (epoch curve)                both + GPTQ + refinement

        The interaction is E_SQ - E_S - E_R, and E_R rather than E_Q is what makes it a
        decomposition rather than an accounting error. Write the weight residuals:
        d_S = -(1-M).W0 removes the pruned values, d_R = M.(Q(M.W0) - W0) is the rounding
        paid on the surviving support, and d_SQ = d_S + d_R EXACTLY. E_Q cannot play that
        role: it charges rounding on the pruned half of the weights too, which the sparse
        model never keeps, so it over-counts the quantization share by roughly 2x and
        would make every layer look spuriously sub-additive. E_Q is still reported, as the
        standalone "what quantization costs if you do not prune" baseline it actually is.

        The residual is a genuine interaction (the cross term 2<X.d_S, X.d_R>, plus the
        expert MLP's nonlinearity in the weights) and not an artifact of how the pieces
        were defined. M is the SHIPPED support, so E_S is arm-dependent on purpose -- a
        support learned against the quantized objective may well be a WORSE pure-sparsity
        mask while giving a better E_SQ, and that dissociation is the cleanest evidence
        that the mask is being chosen for the quantizer rather than for sparsity alone.

        Pure measurement: no parameter is touched and the RNG state is restored, so the
        weights this run ships are identical to a run with the flag off.
        """
        if not (self.model.is_moe and hasattr(self.model, "moe_val_recon_metrics")
                and self.model._is_moe_layer(self.model.current_layer_idx)):
            return
        # self.device is a string like "cuda:0", so go through torch.device to test it.
        on_cuda = torch.device(self.device).type == 'cuda'
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state(self.device) if on_cuda else None
        try:
            results = {}
            for variant in ('sparse_only', 'quant_only', 'quant_on_support', 'sparse_quant'):
                weights = self._decomposition_weights(variant)
                if weights is None:
                    if self.global_rank == 0:
                        logging.info(f'Layer {layer_name} - error decomposition skipped '
                                     f'(no resolvable dense weight or no weight quantizer).')
                    return
                results[variant] = self._recon_metrics_for(
                    weights, val_all, batch_size, microbatch_size)
                weights.clear()
                torch.cuda.empty_cache()
        finally:
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng, self.device)

        if self.global_rank != 0:
            return
        e_s, e_q = results['sparse_only'][1], results['quant_only'][1]
        e_r, e_sq = results['quant_on_support'][1], results['sparse_quant'][1]
        interaction = e_sq - e_s - e_r
        o_s, o_r, o_sq = (results['sparse_only'][2], results['quant_on_support'][2],
                          results['sparse_quant'][2])
        share = f' ({100 * interaction / e_sq:.1f}% of both)' if e_sq else ''
        logging.info(
            f'Layer {layer_name} - error decomposition (block): sparsity {e_s:.3e}  '
            f'rounding-on-support {e_r:.3e}  both {e_sq:.3e}  interaction {interaction:+.3e}'
            f'{share}  [dense-quant baseline {e_q:.3e}]')
        if o_sq == o_sq:
            logging.info(
                f'Layer {layer_name} - error decomposition (ordinary tokens): '
                f'sparsity {o_s:.3e}  rounding-on-support {o_r:.3e}  both {o_sq:.3e}  '
                f'interaction {o_sq - o_s - o_r:+.3e}')
        if self.config.wandb.enabled:
            metrics = {}
            for variant, (recon, block, block_ord) in results.items():
                metrics[f"{layer_name}/decomp_{variant}_recon"] = recon
                metrics[f"{layer_name}/decomp_{variant}_block"] = block
                metrics[f"{layer_name}/decomp_{variant}_block_ord"] = block_ord
            metrics[f"{layer_name}/decomp_interaction_block_ord"] = o_sq - o_s - o_r
            metrics[f"{layer_name}/decomp_interaction_block"] = interaction
            if e_sq:
                metrics[f"{layer_name}/decomp_interaction_share"] = interaction / e_sq
            wandb.log(metrics)

    def _scale_finetune_layer(self, layer_name, train_all, val_all, batch_size, microbatch_size, logging):
        """Post-refinement scale-only fine-tuning; see ScaleFinetuneCompressor.

        Replaces self.compressors with frozen-code wrappers and trains only their
        per-group log-scales under the same loss, keeping the best of {before, every
        (lr, epoch) pair} by hard val loss -- so the stage can never ship a worse layer
        than it started from. The export loop after this reads the wrappers through
        hard_pair unchanged.

        refine.scale_ft_lr may be a LIST, in which case the stage is re-run once per
        rate from theta = 0 with the SAME batch order, and the winner ships. Sweeping
        here rather than across runs costs k x scale_ft_epochs instead of k full
        refinements, and every rate starts from bit-identical refined weights, so the
        comparison has none of the run-to-run spread that makes layer 2 unreadable.

        Why the rate matters more than it looks: NVFP4 stores the group scale as
        FP8-E4M3, whose neighbouring values are 6.7-12.5 % apart, and the forward
        projects onto that grid. A group's EXPORTED scale therefore does not move until
        |theta| crosses a rounding boundary at 3.3-6.25 %. Lion moves theta by exactly
        +-lr per step, so one epoch of S steps decaying to min_lr reaches at most
        S * lr * (1 + min_lr) / 2 even if every step agrees in sign: at lr 1e-3 over
        8192/64 = 128 steps that is 0.070, i.e. 0.78 of ONE grid step -- most groups
        cannot change what they ship, which is why that rate moved the hard val loss by
        a near-constant -1.3 % everywhere. Reaching the +-1-2 grid steps where the
        MSE-optimal FP4 clip actually sits wants lr in the 3e-3 to 1e-2 range.
        """
        cfg = self.config.refine
        if not all(ScaleFinetuneCompressor.supports(c) for c in self.compressors.values()):
            if self.global_rank == 0:
                logging.warning(f'Layer {layer_name}: scale fine-tuning skipped (not an NVFP4 layer).')
            return

        lrs = cfg.scale_ft_lr if isinstance(cfg.scale_ft_lr, (list, tuple)) else [cfg.scale_ft_lr]

        frozen = {name: ScaleFinetuneCompressor.from_compressor(c) for name, c in self.compressors.items()}
        self.compressors.clear()
        self.compressors.update(frozen)
        self.optimizer = None
        torch.cuda.empty_cache()

        num_samples = train_all['input'].shape[0]
        steps_per_epoch = (num_samples + batch_size - 1) // batch_size
        params = [p for c in self.compressors.values() for p in c.parameters()]

        # temperature/scale are ignored by the frozen forward; pass neutral values.
        before = self._validate_epoch(val_all, batch_size, 1.0, 1.0, microbatch_size)
        init_state = self._snapshot_params()          # theta = 0, the pre-FT scales
        # state stays None until some (lr, epoch) beats theta = 0; init_state is the
        # fallback, so the pre-FT scales cost one snapshot, not two.
        best = {'hard': before[1], 'metrics': before, 'lr': None, 'epoch': 0, 'state': None}
        # Same batch order for every rate, so lr is the only difference between passes.
        orders = [list(self.get_random_batch_indices(num_samples, batch_size))
                  for _ in range(cfg.scale_ft_epochs)]
        per_lr = {}
        t0 = time.time()

        for lr_idx, lr in enumerate(lrs):
            if lr_idx > 0:
                self._load_params(init_state)
            # No weight decay: decaying theta pulls the scale back to amax/6, which is the
            # starting point, not a prior worth enforcing on a one-epoch stage.
            self.optimizer = Lion(
                [{'name': 'log_scale', 'params': params, 'lr': lr,
                  'weight_decay': 0.0, 'lr_decay_tag': True}],
                betas=tuple(cfg.lion_betas),
            )
            self.scheduler = CustomLRScheduler(
                self.optimizer, steps_per_epoch * cfg.scale_ft_epochs, 0,
                lr_decay_type=cfg.scale_ft_lr_decay_type, min_lr=cfg.scheduler_min_lr,
            )
            lr_best = {'hard': before[1], 'metrics': before, 'epoch': 0}
            for epoch in range(cfg.scale_ft_epochs):
                losses = []
                for indices in orders[epoch]:
                    losses.append(self.train_step(train_all['input'][indices], 1.0, 1.0, microbatch_size))
                cur = self._validate_epoch(val_all, batch_size, 1.0, 1.0, microbatch_size)
                if cur[1] < lr_best['hard']:
                    lr_best.update(hard=cur[1], metrics=cur, epoch=epoch + 1)
                if cur[1] < best['hard']:
                    best.update(hard=cur[1], metrics=cur, lr=lr, epoch=epoch + 1,
                                state=self._snapshot_params())
                if self.global_rank == 0:
                    logging.info(
                        f'Layer {layer_name} - ScaleFT lr={lr:g} epoch {epoch+1}/{cfg.scale_ft_epochs}: '
                        f'Train Loss = {sum(losses)/len(losses):.2e}, Val Hard Loss = {cur[1]:.2e} '
                        f'(before {before[1]:.2e}), Recon(unw) = {cur[2]:.2e}, Block = {cur[3]:.2e} '
                        f'[ord {cur[6]:.2e} sink {cur[5]:.2e}]'
                    )
            per_lr[lr] = lr_best

        # None means no (lr, epoch) beat theta = 0; the live params are the last rate's
        # last epoch either way, so a reload is always required.
        self._load_params(best['state'] if best['state'] is not None else init_state)
        init_state.clear()
        if best['state'] is not None:
            best['state'].clear()

        after = best['metrics']
        if self.global_rank == 0:
            rel = lambda m: 1.0 - m[1] / before[1] if before[1] > 0 else float('nan')
            gain = rel(after)
            shipped = ('pre-FT scales' if best['epoch'] == 0
                       else f"lr={best['lr']:g} epoch {best['epoch']}")
            logging.info(
                f'Layer {layer_name} ScaleFT done in {time.time()-t0:.0f}s over '
                f'{len(lrs)} lr(s): shipping {shipped}; '
                f'val hard {before[1]:.3e} -> {after[1]:.3e} ({100*gain:+.2f}%), '
                f'block {before[3]:.3e} -> {after[3]:.3e}'
            )
            if len(lrs) > 1:
                logging.info(
                    f'Layer {layer_name} ScaleFT lr sweep (hard val): ' + ' | '.join(
                        f"{lr:g}: {100*rel(per_lr[lr]['metrics']):+.2f}% @ep{per_lr[lr]['epoch']}"
                        for lr in lrs)
                )
                # The same ranking on ORDINARY tokens only. If a rate wins above but not
                # here, it bought sink-token fit, which need not transfer downstream.
                ord0 = before[6]
                relord = lambda m: 1.0 - m[6] / ord0 if ord0 > 0 else float('nan')
                logging.info(
                    f'Layer {layer_name} ScaleFT lr sweep (block err, ordinary tokens, '
                    f'before {ord0:.3e}): ' + ' | '.join(
                        f"{lr:g}: {100*relord(per_lr[lr]['metrics']):+.2f}%" for lr in lrs)
                )
                logging.info(
                    f'Layer {layer_name} block error split before ScaleFT: '
                    f'ordinary {before[6]:.3e}  sink {before[5]:.3e}  '
                    f'(sink/ordinary = {before[5]/before[6]:.1f}x)'
                )
            if self.config.wandb.enabled:
                metrics = {
                    f"{layer_name}/scale_ft_val_hard_before": before[1],
                    f"{layer_name}/scale_ft_val_hard_after": after[1],
                    f"{layer_name}/scale_ft_val_hard_gain": gain,
                    f"{layer_name}/scale_ft_block_before": before[3],
                    f"{layer_name}/scale_ft_block_after": after[3],
                    f"{layer_name}/scale_ft_recon_before": before[2],
                    f"{layer_name}/scale_ft_recon_after": after[2],
                    f"{layer_name}/scale_ft_best_epoch": best['epoch'],
                    f"{layer_name}/scale_ft_best_lr": best['lr'] if best['lr'] is not None else 0.0,
                }
                metrics[f"{layer_name}/scale_ft_block_ord_before"] = before[6]
                metrics[f"{layer_name}/scale_ft_block_ord_after"] = after[6]
                metrics[f"{layer_name}/scale_ft_block_sink_before"] = before[5]
                metrics[f"{layer_name}/scale_ft_block_sink_after"] = after[5]
                for lr in lrs:
                    metrics[f"{layer_name}/scale_ft_lr_{lr:g}/val_hard_gain"] = rel(per_lr[lr]['metrics'])
                    metrics[f"{layer_name}/scale_ft_lr_{lr:g}/val_hard"] = per_lr[lr]['hard']
                    metrics[f"{layer_name}/scale_ft_lr_{lr:g}/block"] = per_lr[lr]['metrics'][3]
                    metrics[f"{layer_name}/scale_ft_lr_{lr:g}/block_ord"] = per_lr[lr]['metrics'][6]
                    metrics[f"{layer_name}/scale_ft_lr_{lr:g}/block_sink"] = per_lr[lr]['metrics'][5]
                    metrics[f"{layer_name}/scale_ft_lr_{lr:g}/best_epoch"] = per_lr[lr]['epoch']
                wandb.log(metrics)

    def train_step(self, batch, temperature, scale, microbatch_size):
        self.optimizer.zero_grad(set_to_none=True)

        batch_size, seq_len, hidden_dim = batch.shape
        accumulation_steps = max(1, batch_size // microbatch_size)

        total_loss = 0.0

        for i in range(accumulation_steps):
            micro_batch = batch[i*microbatch_size:(i+1)*microbatch_size]

            compressed_weights = {}
            for tensor_name, compressor in self.compressors.items():
                # MoE always needs the per-expert grouped dict (_batched_expert_forward
                # keys on "...experts.{id}"); grouping is independent of world size.
                if self.model.is_moe:
                    prefix, leaf = tensor_name.rsplit(".", 1)
                    if prefix not in compressed_weights:
                        compressed_weights[prefix] = {}
                    compressed_weights[prefix][leaf] = compressor.forward(temperature, scale)
                else:
                    compressed_weights[tensor_name] = compressor.forward(temperature, scale)

            self.token_counter.arm_next_dispatch()
            try:
                soft_loss = self.model.calculate_mse(
                    micro_batch.to(self.device), compressed_weights, self.train_attn,
                    accumulation_steps=accumulation_steps,
                    fake_act_quant=self.config.compression.fake_quantize_activations,
                )
            finally:
                self.token_counter.disarm()
            total_loss += soft_loss / accumulation_steps

        self.scheduler.step()
        if not self.model.is_moe and self.use_dist:
            self.average_grads()
        if self.config.refine.logit_grad_diagnostics:
            self._probe_logit_gradients()
        self.optimizer.step()

        compressed_weights.clear()

        if self.use_dist:
            pg = dist.group.WORLD
            t = torch.tensor(total_loss, device=self.device, dtype=torch.float32)
            if self.model.is_moe:
                dist.all_reduce(t, op=dist.ReduceOp.SUM, group=pg)
            else:
                dist.all_reduce(t, op=dist.ReduceOp.AVG, group=pg)
            total_loss = t.item()

        return total_loss

    def _probe_logit_gradients(self):
        """Is Lion's sign() spending full steps on coordinates carrying no gradient?

        The mask logits are a 6-way softmax per block (paired-4:8), and the softmax
        gradient is analytically zero-sum, so the descent information lives in how the
        mass is DISTRIBUTED across the six. Lion updates by sign(), which gives a
        coordinate at 1e-9 the same +-lr step as one at 1e-2. If the mass is concentrated
        on one or two patterns, the other four random-walk at full step size and
        `masks_lr` is an exploration temperature rather than a descent rate -- which
        would change how the masks_lr sweep is read, not just how fast it converges.

        Accumulates over blocks so the numbers are population statistics, not one block.
        """
        acc = self._logit_grad_stats
        for compressor in self.compressors.values():
            g = getattr(compressor, 'mask_logits', None)
            if g is None or g.grad is None:
                continue
            gr = g.grad.detach().float().reshape(-1, g.shape[-1])
            a = gr.abs()
            l1 = a.sum(-1)
            live = l1 > 0
            if not bool(live.any()):
                continue
            a, l1 = a[live], l1[live]
            top2 = a.topk(2, dim=-1).values.sum(-1)
            mx = a.amax(-1, keepdim=True)
            acc['blocks'] += a.shape[0]
            acc['top2_frac'] += (top2 / l1).sum().item()
            # coordinates that Lion will step at full size but that carry <1% of the
            # block's largest gradient -- the ones the sign() cannot distinguish from noise
            acc['negligible'] += (a < 0.01 * mx).float().sum(-1).sum().item()
            acc['coords'] += a.numel()
            acc['zerosum_resid'] += (gr[live].sum(-1).abs() / l1).sum().item()

    def _report_coupling(self, layer_name, logging):
        """How much gradient does the detached scale path actually throw away?

        C = <g, q-r> is the scale-coupling term from the derivation, and e = q-r is the
        straight-through residual, so this reads both the dropped coupling and the STE
        bias. `dropped/kept` is the one to look at: the fraction by which the gradient on
        the amax element is wrong.
        """
        acc = self._coupling_stats
        if not acc['groups'] or self.global_rank != 0 or logging is None:
            return
        g = acc['groups']
        logging.info(
            f'Layer {layer_name} - scale coupling over {int(g):,} group-samples: '
            f'dropped/kept = {acc["ratio"]/g:.4f}  '
            f'mean|C|/6 = {acc["dropped"]/g:.3e}  mean|g|_amax = {acc["kept"]/g:.3e}  '
            f'STE residual |q-r|/|r| = {acc["ste_resid"]/max(acc["r_mag"], 1e-30):.4f}')
        if self.config.wandb.enabled:
            wandb.log({
                f"{layer_name}/coupling_dropped_over_kept": acc['ratio'] / g,
                f"{layer_name}/ste_residual_rel": acc['ste_resid'] / max(acc['r_mag'], 1e-30),
            })
        for k in acc:
            acc[k] = 0.0

    def _report_logit_gradients(self, layer_name, logging):
        acc = self._logit_grad_stats
        if not acc['blocks'] or self.global_rank != 0 or logging is None:
            return
        b, c = acc['blocks'], acc['coords']
        logging.info(
            f'Layer {layer_name} - logit-grad probe over {int(b):,} block-samples: '
            f'top2_mass_frac = {acc["top2_frac"]/b:.4f}  '
            f'negligible_coords_per_block = {acc["negligible"]/b:.2f} of {c/b:.0f}  '
            f'zero_sum_residual = {acc["zerosum_resid"]/b:.2e}')
        if self.config.wandb.enabled:
            wandb.log({
                f"{layer_name}/logit_grad_top2_mass_frac": acc['top2_frac']/b,
                f"{layer_name}/logit_grad_negligible_coords": acc['negligible']/b,
            })
        for k in acc: acc[k] = 0.0

    def _build_weights(self, mode, temperature, scale):
        weights = {}
        for tensor_name, compressor in self.compressors.items():
            if mode == 'soft':
                w = compressor.forward(temperature, scale)
            else:
                w = hard_tensor(compressor)
            if self.model.is_moe:
                prefix, leaf = tensor_name.rsplit(".", 1)
                if prefix not in weights:
                    weights[prefix] = {}
                weights[prefix][leaf] = w
            else:
                weights[tensor_name] = w
        return weights

    def validation_step(self, batch, temperature, scale, microbatch_size):
        batch_size, seq_len, hidden_dim = batch.shape
        microbatch_size = max(1, microbatch_size)
        accumulation_steps = max(1, batch_size // microbatch_size)

        total_soft_loss = 0.0
        total_hard_loss = 0.0

        soft_weights = self._build_weights('soft', temperature, scale)
        for i in range(accumulation_steps):
            micro_batch = batch[i*microbatch_size:(i+1)*microbatch_size].to(self.device)
            total_soft_loss += self.model.calculate_mse(micro_batch, soft_weights, self.train_attn, validation=True, fake_act_quant=self.config.compression.fake_quantize_activations) / accumulation_steps
        soft_weights.clear()

        hard_weights = self._build_weights('hard', temperature, scale)
        # exponent-invariant recon metrics (MoE layers only): unweighted per-expert
        # error and true block-output error, comparable across gate_weight_exponent.
        moe_layer = (self.model.is_moe
                     and hasattr(self.model, "moe_val_recon_metrics")
                     and self.model._is_moe_layer(self.model.current_layer_idx))
        recon_se = recon_n = block_se = block_n = sink8_se = sink_tot_se = 0.0
        blk_sink_se = blk_sink_n = 0.0
        for i in range(accumulation_steps):
            micro_batch = batch[i*microbatch_size:(i+1)*microbatch_size].to(self.device)
            total_hard_loss += self.model.calculate_mse(micro_batch, hard_weights, self.train_attn, validation=True, fake_act_quant=self.config.compression.fake_quantize_activations) / accumulation_steps
            if moe_layer:
                rse, rn, bse, bn, s8, stot, bs8, bs8n = self.model.moe_val_recon_metrics(
                    micro_batch, hard_weights,
                    fake_act_quant=self.config.compression.fake_quantize_activations)
                recon_se += rse.item(); recon_n += rn.item()
                block_se += bse.item(); block_n += bn.item()
                sink8_se += s8.item(); sink_tot_se += stot.item()
                blk_sink_se += bs8.item(); blk_sink_n += bs8n.item()
        hard_weights.clear()

        if self.use_dist:
            pg = dist.group.WORLD
            t_soft = torch.tensor(total_soft_loss, device=self.device, dtype=torch.float32)
            t_hard = torch.tensor(total_hard_loss, device=self.device, dtype=torch.float32)
            if self.model.is_moe:
                dist.all_reduce(t_soft, op=dist.ReduceOp.SUM, group=pg)
                dist.all_reduce(t_hard, op=dist.ReduceOp.SUM, group=pg)
            else:
                dist.all_reduce(t_soft, op=dist.ReduceOp.AVG, group=pg)
                dist.all_reduce(t_hard, op=dist.ReduceOp.AVG, group=pg)
            total_soft_loss = t_soft.item()
            total_hard_loss = t_hard.item()

        recon_unweighted = block_error = sink_share = float('nan')
        block_err_sink = block_err_ord = float('nan')
        if moe_layer:
            if self.use_dist:
                r = torch.tensor([recon_se, recon_n, block_se, block_n, sink8_se, sink_tot_se,
                                  blk_sink_se, blk_sink_n],
                                 device=self.device, dtype=torch.float64)
                dist.all_reduce(r, op=dist.ReduceOp.SUM, group=dist.group.WORLD)
                (recon_se, recon_n, block_se, block_n, sink8_se, sink_tot_se,
                 blk_sink_se, blk_sink_n) = r.tolist()
            recon_unweighted = recon_se / recon_n if recon_n > 0 else float('nan')
            block_error = block_se / block_n if block_n > 0 else float('nan')
            # share of the dense block-output energy on the top-8 tokens per (rank, micro-batch)
            sink_share = sink8_se / sink_tot_se if sink_tot_se > 0 else float('nan')
            # per-element block error ON the sink tokens vs on everything else. Same
            # normalisation as block_error, so the three are directly comparable.
            ord_se, ord_n = block_se - blk_sink_se, block_n - blk_sink_n
            block_err_sink = blk_sink_se / blk_sink_n if blk_sink_n > 0 else float('nan')
            block_err_ord = ord_se / ord_n if ord_n > 0 else float('nan')

        return (total_soft_loss, total_hard_loss, recon_unweighted, block_error, sink_share,
                block_err_sink, block_err_ord)

    def get_random_batch_indices(self, num_samples, batch_size):
        perm = torch.randperm(num_samples)
        if self.use_dist:
            perm = perm.to(self.device)
            dist.broadcast(perm, src=0)
            perm = perm.cpu()
        for i in range(0, num_samples, batch_size):
            yield perm[i:i+batch_size]

    def average_grads(self):
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad.div_(self.world_size)

class CustomLRScheduler:
    def __init__(self, optimizer, total_steps, warmup_steps, lr_decay_type='linear', min_lr=0.0):
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.lr_decay_type = lr_decay_type
        self.min_lr = min_lr
        self.current_step = 0

        self.initial_lrs = [group['lr'] for group in self.optimizer.param_groups]

    def step(self):
        for i, group in enumerate(self.optimizer.param_groups):
            tag = group.get('lr_decay_tag')
            init_lr = self.initial_lrs[i]

            if tag:
                group['lr'] = self._compute_lr(init_lr)

        self.current_step += 1

    def _compute_lr(self, base_lr):
        step = self.current_step
        if step < self.warmup_steps:
            return base_lr * (self.min_lr + (1 - self.min_lr) * step / self.warmup_steps)

        decay_step = step - self.warmup_steps
        decay_total = self.total_steps - 1 - self.warmup_steps
        if decay_total == 0:
            progress = 1
        else:
            progress = decay_step / decay_total

        if self.lr_decay_type == 'linear':
            return base_lr * (self.min_lr + (1 - self.min_lr) * (1 - progress))
        elif self.lr_decay_type == 'cosine':
            return base_lr * (self.min_lr + 0.5 * (1 - self.min_lr) * (1 + math.cos(math.pi * progress)))
        elif self.lr_decay_type == 'constant':
            return base_lr
        else:
            raise ValueError(f"Unknown lr_decay_type: {self.lr_decay_type}")
