import os, gc, sys, traceback
import argparse
import yaml
from datetime import timedelta
import torch
import wandb
try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(**_kwargs): pass
from transformers import AutoTokenizer
from src.config import load_config, validate as validate_config
from src.data.dataset import create_dataloader
from src.trainer import CompressionTrainer
from src.utils.logging_utils import QuantizationLogger
from src.utils import progress_reporter
from src.utils.progress_reporter import (
    report_layer, report_gptq_done, report_throughput,
)
from src.utils.utils import (
    generate_run_id, run_dir, load_progress, save_progress, find_latest_run,
    get_model_wrapper,
    create_act_cache, cleanup_act_cache_mmap,
    local_rank_from_env, local_world_size_from_env, resolve_device,
    to_plain_data, normalize_raw_config_dict, deep_merge_dicts,
    is_known_config_key, apply_config_dict,
    build_run_name, print_effective_config, config_to_plain_dict,
)
import time
import torch.distributed as dist


GLOBAL_RANK = 0


def parse_args():
    parser = argparse.ArgumentParser(description='Model Quantization Training')
    parser.add_argument('--config', type=str, default='configs/p48_nvfp4_refine_actq.yaml',
                      help='Path to configuration file')
    parser.add_argument('--resume', type=str, nargs='?', const='latest', default=None,
                      help='Resume training. Pass a run_id to resume a specific run, '
                           'or omit the value to resume the latest run.')
    parser.add_argument('--max-layers', type=int, default=None,
                      help='Stop after quantizing this many layers (for smoke tests).')
    return parser.parse_args()


def train_all_layers(model, train_loader, val_loader, gpt_loader, logger, config,
                     resume_from_layer=-1, run_id=None, max_layers=None):
    """Train all layers, optionally resuming after a previously completed layer."""
    global GLOBAL_RANK
    device = model.device

    dtype = model.dtype
    model.save_dir = config.training.checkpoint_dir
    model.configure_quantization_from_config(config)
    current_layer = model.get_current_layer()
    _cfg = model.model.config
    if hasattr(_cfg, "text_config"):
        _cfg = _cfg.text_config
    first_k_dense_replace = getattr(_cfg, "first_k_dense_replace", 0)
    hidden_size = getattr(model, "activation_hidden_size", _cfg.hidden_size)

    first_layer_was_trained = (resume_from_layer >= 0
                               and 1 > first_k_dense_replace
                               and 1 > config.refine.start_layer)
    if first_layer_was_trained:
        model.load_from_disc(current_layer)
    elif config.refine.start_layer == 0 or first_k_dense_replace > 0:
        if GLOBAL_RANK == 0:
            logger.logger.info(f"Loading weights for layer: {current_layer}")
        model.move_layer_to_gpu(current_layer)
    else:
        model.load_from_disc(current_layer)
    if GLOBAL_RANK == 0:
        logger.logger.info("Loading embedding weights")
    model.move_embed_to(device)

    mmap_dir = config.training.act_cache_dir or None
    mmap_threshold = int(config.training.act_cache_mmap_threshold_gb * 1024 ** 3)
    ## train_all/val_all exist only to feed trainer.train_layer, which runs only when
    ## refinement is on. A GPTQ-only baseline still captured them and then propagated
    ## them through all 60 layers, paying for activations nothing ever reads. Allocate
    ## them empty in that case so the cache objects stay valid for cleanup.
    refine_enabled = config.refine.enabled
    train_n = len(train_loader.dataset) if refine_enabled else 0
    val_n = len(val_loader.dataset) if refine_enabled else 0
    train_all = create_act_cache(train_n, model.model.seqlen, hidden_size, dtype, mmap_dir, mmap_threshold)
    val_all = create_act_cache(val_n, model.model.seqlen, hidden_size, dtype, mmap_dir, mmap_threshold)
    gpt_all = create_act_cache(len(gpt_loader.dataset), model.model.seqlen, hidden_size, dtype, mmap_dir, mmap_threshold)

    try:
        if GLOBAL_RANK == 0:
            logger.logger.info(f"Capturing GPTQ inputs ({len(gpt_loader.dataset)} samples)")
        model.get_inputs(gpt_all, gpt_loader)
        if refine_enabled:
            if GLOBAL_RANK == 0:
                logger.logger.info(f"Capturing train inputs ({len(train_loader.dataset)} samples)")
            model.get_inputs(train_all, train_loader)
            if GLOBAL_RANK == 0:
                logger.logger.info(f"Capturing val inputs ({len(val_loader.dataset)} samples)")
            model.get_inputs(val_all, val_loader)
        elif GLOBAL_RANK == 0:
            logger.logger.info("Skipping train/val capture (refine.enabled=false; nothing reads them)")

        if GLOBAL_RANK == 0:
            logger.logger.info("Offloading embedding to meta")
        model.move_embed_to('meta')

        num_layers = model.num_layers if hasattr(model, 'num_layers') else 0
        wandb_run_id = wandb.run.id if (config.wandb.enabled and GLOBAL_RANK == 0 and wandb.run) else None
        num_experts = getattr(model, 'num_experts', None)
        if resume_from_layer < 0 and config.training.eval_baseline:
            if GLOBAL_RANK == 0:
                logger.logger.info("Measuring baseline (dense) perplexity before quantization")
            baseline_ppl = model.ppl_evaluation(-1)
            if GLOBAL_RANK == 0:
                logger.logger.info(f"eval/baseline_ppl: {baseline_ppl:.4f}")
            if config.wandb.enabled and GLOBAL_RANK == 0:
                wandb.log({"eval/baseline_ppl": baseline_ppl}, step=0)
            if config.refine.start_layer == 0 or first_k_dense_replace > 0:
                model.move_layer_to_gpu(current_layer)
            else:
                model.load_from_disc(current_layer)
            model.move_embed_to('meta')

        run_start = time.time()
        layer_times = []
        gptq_times = []
        train_times = []
        refine_enabled = config.refine.enabled
        count = 0
        while current_layer is not None:
            count += 1
            layer_idx = count - 1
            is_already_done = layer_idx <= resume_from_layer

            needs_training = (count > first_k_dense_replace
                              and count > config.refine.start_layer)

            if needs_training and not is_already_done:
                if GLOBAL_RANK == 0:
                    init_label = config.init.method.upper()
                    logger.logger.info(f"Starting {'quantization' if refine_enabled else init_label + '-only'} for layer: {current_layer}")

                if dist.is_initialized():
                    dist.barrier()
                layer_start = time.time()
                elapsed_total = layer_start - run_start

                if GLOBAL_RANK == 0:
                    report_layer(layer_idx, num_layers, config.init.method.upper(), elapsed_total)

                if config.refine.self_attn and not model.is_moe:
                    trainer = CompressionTrainer(model, config, dtype, self_attn=True)
                    model.get_layer_initialization(trainer, gpt_all, config, logger, is_attn=True)
                    torch.cuda.synchronize()
                    gc.collect()
                    torch.cuda.empty_cache()
                    if refine_enabled:
                        trainer.train_layer(current_layer, train_all, val_all, logger,
                                            layer_idx=layer_idx, num_layers=num_layers)
                    del trainer
                    if GLOBAL_RANK == 0:
                        model.save_prefixes_to_disc(model._layer_prefixes(current_layer)['non_mlp'], exclude=["self_attn"])
                        model.save_attention_to_disc(f"{model.layer_prefix}.{count-1}.self_attn")
                else:
                    if GLOBAL_RANK == 0:
                        model.save_prefixes_to_disc(model._layer_prefixes(current_layer)['non_mlp'])

                trainer = CompressionTrainer(model, config, dtype)
                gptq_start = time.time()
                model.get_layer_initialization(trainer, gpt_all, config, logger)
                torch.cuda.synchronize()
                gptq_time = time.time() - gptq_start
                gptq_times.append(gptq_time)

                if GLOBAL_RANK == 0:
                    gptq_avg_loss_tmp = getattr(trainer, 'gptq_avg_loss', None)
                    report_gptq_done(gptq_time, avg_loss=gptq_avg_loss_tmp)

                if refine_enabled:
                    if model.is_moe:
                        model.get_mlp_input_all(train_all)
                        model.get_mlp_input_all(val_all)

                    torch.cuda.synchronize()
                    gc.collect()
                    torch.cuda.empty_cache()

                    if GLOBAL_RANK == 0:
                        report_layer(layer_idx, num_layers, "Refine",
                                     time.time() - run_start)

                    train_start = time.time()
                    trainer.train_layer(current_layer, train_all, val_all, logger,
                                        layer_idx=layer_idx, num_layers=num_layers)
                    torch.cuda.synchronize()
                    train_time = time.time() - train_start
                    train_times.append(train_time)
                else:
                    train_time = 0.0
                    if model.is_moe:
                        model.save_moe_experts_to_disc()
                    elif GLOBAL_RANK == 0:
                        mlp_pfx = f"{model.layer_prefix}.{count-1}.mlp"
                        model.save_mlp_to_disc(mlp_pfx)

                model.temp_weights.clear()

                gptq_avg_loss = getattr(trainer, 'gptq_avg_loss', None)
                del trainer
                torch.cuda.empty_cache()

                if dist.is_initialized():
                    dist.barrier()
                layer_time = time.time() - layer_start
                layer_times.append(layer_time)
                avg_layer_time = sum(layer_times) / len(layer_times)

                if GLOBAL_RANK == 0:
                    report_throughput(
                        layer_idx, num_layers, layer_time, gptq_time, train_time,
                        num_experts=num_experts if model.is_moe else None,
                        avg_layer_time=avg_layer_time,
                    )

                if config.wandb.enabled and GLOBAL_RANK == 0:
                    mem_alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
                    mem_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
                    layer_logs = {
                        "layer/index": layer_idx,
                        "layer/progress": layer_idx / max(num_layers, 1),
                        "timing/gptq_init_sec": gptq_time,
                        "timing/training_sec": train_time,
                        "timing/layer_total_sec": layer_time,
                        "timing/gptq_phase_sec_avg": sum(gptq_times) / len(gptq_times),
                        "timing/layer_sec_avg": avg_layer_time,
                        "throughput/layers_per_hour": 3600.0 / avg_layer_time if avg_layer_time > 0 else 0,
                        "gpu/max_memory_allocated_gb": mem_alloc,
                        "gpu/max_memory_reserved_gb": mem_reserved,
                    }
                    if train_times:
                        layer_logs["timing/gumbel_phase_sec_avg"] = sum(train_times) / len(train_times)
                    if model.is_moe and num_experts:
                        layer_logs["timing/sec_per_expert"] = layer_time / num_experts
                        layer_logs["throughput/experts_per_hour"] = (num_experts * 3600.0 / layer_time) if layer_time > 0 else 0
                    if gptq_avg_loss is not None:
                        layer_logs["gptq/avg_loss"] = gptq_avg_loss
                    wandb.log(layer_logs)
                    torch.cuda.reset_peak_memory_stats()

                # ppl_eval_every_n_layers > num_layers (or <= 0) means "never". Without this,
                # (count-1) % N is 0 % N == 0 at the FIRST layer for every N, so the eval
                # would always fire once no matter how large N was set.
                _ppl_n = config.training.ppl_eval_every_n_layers
                _ppl_off = _ppl_n <= 0 or (num_layers and _ppl_n > num_layers)
                if _ppl_off and count == 1 and GLOBAL_RANK == 0:
                    logger.logger.info(
                        f"PPL eval disabled (ppl_eval_every_n_layers={_ppl_n} > num_layers={num_layers})"
                    )
                if not _ppl_off and (count - 1) % _ppl_n == 0:
                    dataset_ppl = model.ppl_evaluation(count - 1)
                    if GLOBAL_RANK == 0:
                        logger.logger.info(f"eval/ppl (layer {layer_idx}): {dataset_ppl:.4f}")
                    if config.wandb.enabled and GLOBAL_RANK == 0:
                        wandb.log({"eval/ppl": dataset_ppl, "layer/index": layer_idx})

                if GLOBAL_RANK == 0:
                    logger.logger.info(f"Loading quantized weights for layer: {current_layer}")
                model.load_from_disc(current_layer)

                if GLOBAL_RANK == 0:
                    logger.logger.info("Propagating activations through layer")
                ## Without refinement only gpt_all needs advancing (done below); train/val
                ## are never read, so propagating them is pure cost on every layer.
                if refine_enabled:
                    if model.is_moe:
                        model.get_mlp_output_all(train_all)
                        model.get_mlp_output_all(val_all)
                    else:
                        model.get_layer_activations(train_all)
                        model.get_layer_activations(val_all)

                if GLOBAL_RANK == 0:
                    save_progress(config.training.checkpoint_dir, layer_idx,
                                  run_id=run_id, wandb_run_id=wandb_run_id)

            elif needs_training and is_already_done:
                if GLOBAL_RANK == 0:
                    logger.logger.info(
                        f"Skipping layer {current_layer} (already completed, replaying activations)")
                if refine_enabled:
                    if model.is_moe:
                        model.get_mlp_input_all(train_all)
                        model.get_mlp_input_all(val_all)
                        model.get_mlp_output_all(train_all)
                        model.get_mlp_output_all(val_all)
                    else:
                        model.get_layer_activations(train_all)
                        model.get_layer_activations(val_all)
            else:
                if refine_enabled:
                    model.get_layer_activations(train_all)
                    model.get_layer_activations(val_all)
            model.get_layer_activations(gpt_all)

            model.offload_to_meta(current_layer)

            if max_layers is not None and count >= max_layers:
                if GLOBAL_RANK == 0:
                    logger.logger.info(
                        f"Reached --max-layers={max_layers}; stopping early (smoke test)")
                break

            current_layer = model.move_to_next_layer()

            if current_layer:
                if GLOBAL_RANK == 0:
                    logger.logger.info(f"Moving to next layer: {current_layer}")
                next_layer_idx = count
                next_count = count + 1
                next_was_trained = (next_layer_idx <= resume_from_layer
                                    and next_count > first_k_dense_replace
                                    and next_count > config.refine.start_layer)
                if next_was_trained:
                    model.load_from_disc(current_layer)
                elif count >= config.refine.start_layer:
                    model.move_layer_to_gpu(current_layer)
                else:
                    model.load_from_disc(current_layer)
            else:
                if GLOBAL_RANK == 0:
                    logger.logger.info("Finished training all layers")
    finally:
        cleanup_act_cache_mmap(train_all, val_all, gpt_all)

def main():
    load_dotenv()
    global GLOBAL_RANK
    args = parse_args()
    config = load_config(args.config)
    config_dict = normalize_raw_config_dict(yaml.safe_load(open(args.config, 'r')) or {})

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    device = resolve_device()
    if device.startswith("cuda"):
        torch.cuda.set_device(torch.device(device))

    if not dist.is_initialized():
        if world_size > 1:
            dist.init_process_group(backend="nccl", timeout=timedelta(hours=config.distributed.timeout_hours))
            GLOBAL_RANK = dist.get_rank()
        else:
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29501")
            os.environ.setdefault("RANK", "0")
            os.environ.setdefault("WORLD_SIZE", "1")
            dist.init_process_group(backend="gloo", rank=0, world_size=1)

    resume_from_layer = -1
    saved_wandb_id = None

    if GLOBAL_RANK == 0:
        if args.resume is not None:
            if args.resume == 'latest':
                run_id = find_latest_run(config.training.checkpoint_dir)
            else:
                run_id = args.resume
            if run_id is not None:
                rd = run_dir(config.training.checkpoint_dir, run_id)
                progress = load_progress(rd)
                if progress is not None:
                    resume_from_layer = progress["last_completed_layer"]
                    saved_wandb_id = progress.get("wandb_run_id")
                    print(f"Resuming run '{run_id}': skipping layers 0..{resume_from_layer}")
                else:
                    print(f"Run '{run_id}' has no progress file; starting fresh")
                    run_id = generate_run_id()
            else:
                print("--resume specified but no runs found; starting fresh")
                run_id = generate_run_id()
        else:
            run_id = generate_run_id()
    else:
        run_id = None

    if world_size > 1:
        id_list = [run_id] if GLOBAL_RANK == 0 else [None]
        dist.broadcast_object_list(id_list, src=0)
        run_id = id_list[0]
        layer_list = [resume_from_layer] if GLOBAL_RANK == 0 else [None]
        dist.broadcast_object_list(layer_list, src=0)
        resume_from_layer = layer_list[0]

    merged_config_dict = config_dict
    config_source = "local config"
    if config.wandb.enabled and GLOBAL_RANK == 0:
        wandb_kwargs = dict(
            project=config.wandb.project,
            config=config_dict,
        )
        if config.wandb.entity:
            wandb_kwargs["entity"] = config.wandb.entity
        if saved_wandb_id is not None:
            wandb_kwargs["id"] = saved_wandb_id
            wandb_kwargs["resume"] = "must"
        wandb.init(**wandb_kwargs)
        raw_wandb_config = to_plain_data(wandb.config)
        wandb_overrides = {
            k: v for k, v in raw_wandb_config.items()
            if is_known_config_key(config, k)
        }
        dropped = sorted(set(raw_wandb_config) - set(wandb_overrides))
        print(f"[wandb] raw wandb.config keys: {sorted(raw_wandb_config)}", flush=True)
        print(f"[wandb] applied override keys: {sorted(wandb_overrides)}", flush=True)
        if dropped:
            print(f"[wandb] dropped (unknown to local config): {dropped}", flush=True)
        merged_config_dict = normalize_raw_config_dict(
            deep_merge_dicts(config_dict, wandb_overrides)
        )
        config_source = "wandb overrides + local config"

    if world_size > 1:
        cfg_list = [merged_config_dict] if GLOBAL_RANK == 0 else [None]
        dist.broadcast_object_list(cfg_list, src=0)
        merged_config_dict = normalize_raw_config_dict(cfg_list[0])

    apply_config_dict(config, merged_config_dict)
    validate_config(config)
    progress_reporter.init(config)

    rd = run_dir(config.training.checkpoint_dir, run_id)
    config.training.checkpoint_dir = rd
    if GLOBAL_RANK == 0:
        os.makedirs(rd, exist_ok=True)
        print(f"Run ID: {run_id}")
        print(f"Checkpoints: {rd}")
        print_effective_config(config, config_source)

    if config.wandb.enabled and GLOBAL_RANK == 0:
        wandb.run.name = build_run_name(config)
        wandb.config.update(config_to_plain_dict(config), allow_val_change=True)
        wandb.config.update({
            "checkpoint_run_id": run_id,
            "world_size": world_size,
            "num_nodes": max(1, world_size // local_world_size_from_env()),
            "gpus_per_node": local_world_size_from_env(),
            "init_method": config.init.method,
            "init_wbits": config.init.wbits,
            "compression_quant_type": config.compression.quant_type,
            "compression_prunen": config.compression.prunen,
            "compression_prunem": config.compression.prunem,
            "compression_groupsize": config.compression.groupsize,
            "compression_learn_weight_values": config.compression.learn_weight_values,
            "compression_fake_quantize_activations": config.compression.fake_quantize_activations,
            "refine_enabled": config.refine.enabled,
            "refine_learn_masks": config.refine.learn_masks,
            "refine_logits_dtype": config.refine.logits_dtype,
            "refine_weights_lr": config.refine.weights_lr,
        }, allow_val_change=True)
        _code_exclude = {"transformers/", ".venv/", "venv-moe-sq/", "wandb/", "__pycache__/", ".git/"}
        wandb.run.log_code(
            ".",
            include_fn=lambda path: any(path.endswith(ext) for ext in [".py", ".yaml", ".sh", ".toml"])
                and not any(dir in path for dir in _code_exclude),
        )

    tokenizer = AutoTokenizer.from_pretrained(config.model.name, use_fast=True, trust_remote_code=True)
    per_rank_batch = max(1, config.data.batch_size // world_size)
    model = get_model_wrapper(config.model.name, tokenizer, per_rank_batch, config.data.max_length, device, config.model.dtype, world_size, dummy=config.model.dummy)
    current_device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    print(
        f"[RANK {GLOBAL_RANK}] wrapper={type(model).__name__} "
        f"device={model.device} current_cuda_device={current_device} "
        f"local_rank={os.environ.get('LOCAL_RANK', '<unset>')} "
        f"world_size={world_size} cuda_visible={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}",
        flush=True,
    )
    model.meta_init_std = config.training.meta_init_std
    model.calib_report_divisor = config.logging.calib_report_divisor
    model.batch_report_divisor = config.logging.batch_report_divisor

    logger = QuantizationLogger(config.training.log_dir) if GLOBAL_RANK == 0 else None

    train_loader, val_loader, gpt_loader = create_dataloader(
        config.data.dataset_name,
        tokenizer,
        batch_size=per_rank_batch,
        train_samples=config.data.num_samples,
        val_samples=config.data.val_samples,
        gpt_samples=config.init.nsamples,
        num_workers=config.data.num_workers,
        max_length=config.data.max_length,
        seed=config.data.seed,
        shuffle_seed=config.data.shuffle_seed,
        shuffle_buffer_size=config.data.shuffle_buffer_size,
        open_thoughts_max_samples=config.data.open_thoughts_max_samples,
        mixed_source_weights=config.data.mixed_source_weights,
    )

    try:
        tick = time.time()
        train_all_layers(model, train_loader, val_loader, gpt_loader, logger, config,
                         resume_from_layer=resume_from_layer, run_id=run_id,
                         max_layers=args.max_layers)
        total_time = time.time() - tick
        if GLOBAL_RANK == 0:
            print("Total time:", total_time)
            if config.wandb.enabled:
                wandb.log({"timing/total_wall_clock_sec": total_time})

    except KeyboardInterrupt:
        if GLOBAL_RANK == 0:
            logger.logger.info("Training interrupted.")
        if dist.is_initialized():
            dist.destroy_process_group()

    except Exception:
        # destroy_process_group() in the finally blocks while peers are still alive,
        # so the traceback never reaches stderr unless it is printed here first.
        print(f"[RANK {GLOBAL_RANK}] uncaught exception in train_all_layers:", flush=True)
        traceback.print_exc()
        sys.stderr.flush()
        raise

    finally:
        if GLOBAL_RANK == 0:
            logger.logger.info("Training finished.")
            if config.wandb.enabled:
                wandb.finish()
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    main()
