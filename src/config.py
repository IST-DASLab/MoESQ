from dataclasses import dataclass, field, fields
from typing import Optional
import os
import yaml


@dataclass
class ModelConfig:
    name: str = ""
    device: str = "cuda"
    dtype: str = "bfloat16"
    dummy: bool = False


@dataclass
class DataConfig:
    dataset_name: str = "open_thoughts"
    num_samples: int = 4096
    val_samples: int = 128
    batch_size: int = 64
    max_length: int = 4096
    num_workers: int = 8
    seed: int = 0
    shuffle_seed: int = 1234
    shuffle_buffer_size: int = 100_000
    open_thoughts_max_samples: int = 10_000
    mixed_source_weights: list = field(default_factory=lambda: [0.1, 0.45, 0.45])


@dataclass
class CompressionConfig:
    """What we compress to: sparsity pattern + (optional) weight quant scheme."""
    prunen: int = 4
    prunem: int = 8
    quant_type: Optional[str] = "nvfp4"  # "nvfp4" | "gsq" | None (sparsity-only)
    groupsize: int = 16
    learn_weight_values: bool = True
    fake_quantize_activations: bool = True


@dataclass
class InitConfig:
    """How we initialize the compressed weights (e.g. GPTQ second-order init)."""
    method: str = "gptq"  # gptq | obr | jsq | rtn | random
    wbits: int = 4
    nsamples: int = 512
    sym: bool = True
    trits: bool = False
    percdamp: float = 0.01
    blocksize: int = 128
    static_groups: bool = False
    # Cap on tokens per fp32 block when accumulating the GPTQ Hessian. 0 = no chunking,
    # which is correct for hook-driven models (one small micro-batch per add_batch call).
    # Fused-expert models pass every token routed to an expert in a single call, so the
    # copy scales with routing skew; set this to bound it. Does not change H.
    hessian_chunk_tokens: int = 0
    obr_alpha: float = 0.5
    obr_mask_metric: str = "wanda"       # wanda | magnitude | sparsegpt
    obr_partition: str = "column"        # column | error
    obr_solver: str = "cg"               # cg | cholesky
    obr_cg_max_iters: int = 1000
    obr_cg_tol: float = 1e-6
    obr_scale_on_masked: bool = True
    jsq_lambda_norm: float = 1.0         # 0 = pure pair-summed Wanda (the -SAR ablation)
    jsq_edit_r: float = 5e-5             # 0 = no editing; paper's "-Search" point is 5e-5
    jsq_edit_mode: str = "range"         # range (paper Eq. 5) | quantile (released code)
    jsq_token_cap: int = 0               # 0 = exact range term over all routed tokens
    jsq_range_dtype: str = "float32"     # float32 | bfloat16 (halves accumulators at Kimi scale)


@dataclass
class RefineConfig:
    """Gumbel-Softmax + fake-quantization refinement loop."""
    enabled: bool = True
    start_layer: int = 0
    self_attn: bool = False
    # Trainability of the SUPPORT, the partner of compression.learn_weight_values (the
    # values). The two flags span the joint-adaptation ablation: learn/learn is the method,
    # fixed/learn is fixed-mask QAT, learn/fixed is support-only adaptation, and fixed/fixed
    # is the initialization-only arm (refine.enabled=false). With this false no mask logits
    # are allocated and the forward uses the initializer's support deterministically, so the
    # temperature/scale anneal below has nothing to act on.
    learn_masks: bool = True
    # Sequential arms: None (joint, the default), 'support_first' (A then V) or
    # 'values_first' (V then A). Phase 1 runs epochs [0, sequential_phase1_epochs),
    # phase 2 the rest of num_epochs, so num_epochs=10/phase1=5 is budget-matched
    # against the joint arm and num_epochs=20/phase1=10 gives each phase the joint
    # arm's full budget.
    # 'lion' (default, the GSQ recipe) or 'adam'/'adamw' for the MASK LOGITS ONLY --
    # the weights stay on Lion either way, so existing arms stay comparable.
    mask_optimizer: str = "lion"
    mask_adam_betas: list = None
    # Lion applies weight decay DECOUPLED; torch.optim.Adam couples it into the gradient.
    # The inherited weight_decay=1 is harmless under Lion (a 5e-5 shrink per step) but under
    # Adam it adds wd*theta ~ 4e-2 to a logit gradient of ~1e-2 -- swamping the real signal.
    # None inherits refine.weight_decay; set 0 to isolate the optimizer change.
    mask_weight_decay: float = None
    logit_grad_diagnostics: bool = False
    weight_drift_diagnostics: bool = False
    # None keeps the detached scale (default). A float p replaces the amax derivative with
    # a p-norm surrogate in the BACKWARD ONLY, re-enabling the support/value coupling term
    # C * ds/dw with a derivative spread over the top few elements instead of exactly one.
    # The forward scale stays the exact amax, so the export is unchanged.
    scale_grad_p: float = None
    coupling_diagnostics: bool = False
    sequential: str = None
    sequential_phase1_epochs: int = None
    num_epochs: int = 10
    device_microbatch_size: int = 16
    warmup_steps: int = 0
    lr_decay_type: str = "cosine"
    scheduler_min_lr: float = 0.1
    masks_lr: float = 0.0002
    weights_lr: float = 0.0001
    # GSQ (arXiv:2604.18556) trains TWO parameter families with different rates, and
    # neither is a "mask" -- GSQ has no mask. Tables 7/8 give logits lr 1e-4 (Llama)
    # / 2e-4 (Kimi) against group-scale lr 5e-5 / 1e-5, i.e. the scale trains 2-20x
    # SLOWER than the assignments. Left at None these fall back to masks_lr, so
    # existing NVFP4 runs are bit-identical.
    logits_lr: Optional[float] = None
    group_scales_lr: Optional[float] = None
    weight_decay: float = 1.0
    lion_betas: list = field(default_factory=lambda: [0.9, 0.95])
    temperature: list = field(default_factory=lambda: [2.0, 0.05])
    scale: list = field(default_factory=lambda: [100, 500])
    strength: float = 6.0
    std: float = 0.01
    logits_dtype: str = "bfloat16"
    gate_weight_exponent: float = 0.0
    # Flag a layer as diverged when its final-epoch train loss exceeds the best
    # epoch by this factor. Warn only (never aborts). <= 0 disables the check.
    divergence_warn_ratio: float = 1.5
    # Deprecated; not used by any shipped config.
    best_epoch_snapshot: bool = False
    # Post-refinement scale-only fine-tuning (GSQ App. H, transplanted to NVFP4): freeze
    # the mask and FP4 codes, learn a per-group log-multiplier on the scale for this
    # many extra epochs under the same loss. 0 disables. The LR is in log-scale units
    # (Lion step = relative change per step), so 1e-3 means <=0.1% per step.
    scale_ft_epochs: int = 0
    # Scalar, or a LIST of rates to sweep. A list re-runs the stage once per rate from
    # the same frozen codes (theta reset to 0), keeping the globally best by hard val
    # loss -- so k rates cost k x scale_ft_epochs instead of k full refinements, and
    # the comparison carries no cross-run noise. See _scale_finetune_layer.
    scale_ft_lr: float = 1.0e-3
    scale_ft_lr_decay_type: str = "linear"
    # Per-layer error decomposition: split the shipped reconstruction error into the part
    # sparsity costs, the part quantization costs, and what GPTQ and refinement each buy
    # back. Measurement only -- it runs after the layer is trained, touches no parameter,
    # and restores the RNG state, so a run with it on ships the same weights as one without.
    error_decomposition: bool = False
    # Sink-aware reconstruction loss.
    # Each (token, expert) row of the refinement loss is weighted by
    # min(1, (k * median ||f_e(x)||)^2 / ||f_e(x)||^2), computed from the DENSE expert
    # output of that row. Ordinary rows keep weight 1; the massive-activation (attention-
    # sink) rows, whose dense output is ~10^2-10^3x the median, are capped at the weight of
    # a k-times-median row so they stop owning the objective at the layers that produce
    # the sink (Qwen3-30B: layers 1-3). 0 disables (bit-identical to the old loss).
    token_weight_clip_k: float = 0.0


@dataclass
class TrainingConfig:
    """Layer-by-layer training pipeline + checkpointing knobs."""
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    eval_baseline: bool = True
    ppl_eval_every_n_layers: int = 6
    meta_init_std: float = 0.02
    act_cache_dir: str = "act_cache"
    act_cache_mmap_threshold_gb: float = 2.0


@dataclass
class EvalConfig:
    default_tasks: str = "gsm8k,arc_challenge,arc_easy,winogrande,piqa"
    ppl_seed: int = 1234
    ppl_max_samples: int = 1000
    test_size: float = 0.2
    split_seed: int = 42


@dataclass
class DistributedConfig:
    timeout_hours: float = 2.0


@dataclass
class WandBConfig:
    enabled: bool = True
    project: str = ""
    entity: str = ""


@dataclass
class LoggingConfig:
    gumbel_step_log_interval: int = 50
    step_report_divisor: int = 5
    calib_report_divisor: int = 10
    ppl_report_divisor: int = 10
    batch_report_divisor: int = 5


@dataclass
class MoESQConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    init: InitConfig = field(default_factory=InitConfig)
    refine: RefineConfig = field(default_factory=RefineConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


_SECTION_TO_DATACLASS = {
    "model": ModelConfig,
    "data": DataConfig,
    "compression": CompressionConfig,
    "init": InitConfig,
    "refine": RefineConfig,
    "training": TrainingConfig,
    "eval": EvalConfig,
    "distributed": DistributedConfig,
    "wandb": WandBConfig,
    "logging": LoggingConfig,
}


def _build_dataclass(cls, raw):
    if raw is None:
        return cls()
    if not isinstance(raw, dict):
        raise TypeError(f"Expected dict for {cls.__name__}, got {type(raw).__name__}")
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(
            f"Unknown key(s) in '{cls.__name__}' section: {sorted(unknown)}. "
            f"Allowed keys: {sorted(known)}"
        )
    return cls(**raw)


def load_config(path):
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise TypeError(f"Expected top-level YAML mapping, got {type(raw).__name__}")

    # Bare `wandb: true/false` shorthand -> WandBConfig
    if "wandb" in raw and not isinstance(raw["wandb"], dict):
        raw["wandb"] = {"enabled": bool(raw["wandb"])}

    unknown_top = set(raw) - set(_SECTION_TO_DATACLASS)
    if unknown_top:
        legacy = unknown_top & {"quantization", "gptq"}
        if legacy:
            raise ValueError(
                f"Legacy config section(s) found: {sorted(legacy)}. See the "
                "Configuration section of README.md for the current layout "
                "(compression / init / refine / training). Update the YAML before loading."
            )
        raise ValueError(
            f"Unknown top-level config section(s): {sorted(unknown_top)}. "
            f"Allowed sections: {sorted(_SECTION_TO_DATACLASS)}"
        )

    kwargs = {}
    for section_name, cls in _SECTION_TO_DATACLASS.items():
        section = raw.get(section_name, None)
        kwargs[section_name] = _build_dataclass(cls, section)

    cfg = MoESQConfig(**kwargs)

    if not cfg.wandb.project:
        cfg.wandb.project = os.environ.get("WANDB_PROJECT", "moe-sq")
    if not cfg.wandb.entity:
        cfg.wandb.entity = os.environ.get("WANDB_ENTITY", "")

    validate(cfg)

    return cfg


def _default_mask_betas(cfg):
    if cfg.refine.mask_adam_betas is None:
        cfg.refine.mask_adam_betas = [0.9, 0.999]
    return cfg


def validate(cfg):
    quant_type = cfg.compression.quant_type
    if quant_type not in (None, "nvfp4", "gsq"):
        raise ValueError(
            f"Unsupported compression.quant_type={quant_type!r}. Expected 'nvfp4', 'gsq' or null."
        )

    if quant_type == "gsq":
        # GSQ (arXiv:2604.18556) is weight-quantization only: dense, symmetric,
        # group-wise, no zero point. The grid is fixed per bit-width.
        # NOTE: `errors` is local to each quant_type branch -- the nvfp4 branch below
        # rebinds it -- so this branch owns and raises its own list.
        errors = []
        from src.compression.quant.gsq import GSQ_GRIDS
        if cfg.init.wbits not in GSQ_GRIDS:
            errors.append(
                f"compression.quant_type='gsq' supports init.wbits in {sorted(GSQ_GRIDS)}, "
                f"got {cfg.init.wbits!r}."
            )
        if (cfg.compression.prunen, cfg.compression.prunem) != (0, 0):
            errors.append(
                "compression.quant_type='gsq' is dense-only: expected prunen:prunem=0:0, got "
                f"{cfg.compression.prunen}:{cfg.compression.prunem}."
            )
        if not cfg.refine.enabled:
            errors.append(
                "compression.quant_type='gsq' requires refine.enabled=true -- the grid "
                "assignment IS the Gumbel-Softmax optimization. Without it this is just "
                "GPTQ at init.wbits bits (use quant_type=null for that)."
            )
        if cfg.compression.learn_weight_values:
            errors.append(
                "compression.quant_type='gsq' requires learn_weight_values=false: GSQ has no "
                "continuous master weight, it learns grid assignments and scales."
            )
        if not cfg.refine.learn_masks:
            errors.append(
                "compression.quant_type='gsq' has no mask to freeze; refine.learn_masks "
                "must be true (NoSparsity's logits are a length-1 softmax)."
            )
        if cfg.compression.fake_quantize_activations:
            errors.append(
                "compression.quant_type='gsq' is weight-only; set "
                "compression.fake_quantize_activations=false."
            )
        if errors:
            raise ValueError("Invalid GSQ compression config: " + "; ".join(errors))

    if quant_type == "nvfp4":
        errors = []
        if cfg.compression.groupsize != 16:
            errors.append(f"compression.groupsize must be 16, got {cfg.compression.groupsize!r}")
        if cfg.init.wbits != 4:
            errors.append(f"init.wbits must be 4, got {cfg.init.wbits!r}")
        # (0, 0) == dense NVFP4, no sparsity. dense_scale_groupsize(0, 0, 16) is 16,
        # which is one FP8 scale per 16 STORED elements -- the same thing the kernel
        # sees for sparse gs=32 at 50% sparsity, so the formats are comparable.
        if (cfg.compression.prunen, cfg.compression.prunem) not in ((0, 0), (2, 4), (4, 8)):
            errors.append(
                "compression.prunen:prunem must be 0:0 (dense), 2:4 or paired 4:8, "
                f"got {cfg.compression.prunen}:{cfg.compression.prunem}"
            )
        # Refinement learns a sparsity pattern; with no sparsity there is nothing to
        # learn and builders._select_sparsity would raise mid-run instead.
        if cfg.compression.prunen == 0 and cfg.refine.enabled:
            errors.append(
                "compression.prunen:prunem=0:0 (dense) is incompatible with "
                "refine.enabled=true: there is no sparsity pattern to learn."
            )
        # learn_weight_values is only consulted by the refinement compressor;
        # for refine-disabled baselines (GPTQ-only / sparseGPTQ-only) the flag
        # is moot, so don't require it. With refinement on, values and support may be
        # frozen independently (the ablation arms) but not both -- that is the
        # initialization-only arm and wants refine.enabled=false, not a no-op training loop.
        if (cfg.refine.enabled and not cfg.compression.learn_weight_values
                and not cfg.refine.learn_masks):
            errors.append(
                "compression.learn_weight_values=false with refine.learn_masks=false leaves "
                "nothing to train; use refine.enabled=false for the initialization-only arm."
            )
        if errors:
            raise ValueError("Invalid NVFP4 compression config: " + "; ".join(errors))

    if cfg.refine.scale_grad_p is not None:
        if cfg.refine.scale_grad_p <= 1:
            raise ValueError(
                "refine.scale_grad_p is the exponent of a p-norm surrogate for amax and "
                f"must be > 1 (higher = closer to amax), got {cfg.refine.scale_grad_p!r}.")
        if quant_type != "nvfp4":
            raise ValueError(
                "refine.scale_grad_p only applies to compression.quant_type='nvfp4'.")

    mo = (cfg.refine.mask_optimizer or "lion").lower()
    if mo not in ("lion", "adam", "adamw"):
        raise ValueError(
            f"refine.mask_optimizer must be 'lion', 'adam' or 'adamw', got "
            f"{cfg.refine.mask_optimizer!r}.")
    if cfg.refine.mask_adam_betas is None:
        cfg.refine.mask_adam_betas = [0.9, 0.999]
    if len(cfg.refine.mask_adam_betas) != 2:
        raise ValueError(
            f"refine.mask_adam_betas must be a 2-list, got {cfg.refine.mask_adam_betas!r}.")
    if mo != "lion" and not cfg.refine.learn_masks:
        raise ValueError(
            "refine.mask_optimizer only affects the mask logits, but refine.learn_masks "
            "is false so there are none to optimize.")

    seq = cfg.refine.sequential
    if seq is not None:
        if seq not in ("support_first", "values_first"):
            raise ValueError(
                "refine.sequential must be null, 'support_first' or 'values_first', "
                f"got {seq!r}.")
        if not (cfg.refine.learn_masks and cfg.compression.learn_weight_values):
            raise ValueError(
                "refine.sequential trains BOTH variables in sequence, so it needs "
                "refine.learn_masks=true and compression.learn_weight_values=true; "
                "a one-sided arm is arm 2 or 3, not a sequential arm.")
        p1 = cfg.refine.sequential_phase1_epochs
        if p1 is None or not (1 <= p1 < cfg.refine.num_epochs):
            raise ValueError(
                "refine.sequential requires sequential_phase1_epochs in "
                f"[1, num_epochs-1] = [1, {cfg.refine.num_epochs - 1}], got {p1!r}.")
    elif cfg.refine.sequential_phase1_epochs is not None:
        raise ValueError(
            "refine.sequential_phase1_epochs is set but refine.sequential is null.")

    if not cfg.refine.learn_masks and quant_type != "nvfp4":
        raise ValueError(
            "refine.learn_masks=false is only wired for compression.quant_type='nvfp4' "
            f"(got {quant_type!r}); the sparsity-only path has no other variable to train."
        )

    if cfg.compression.fake_quantize_activations and quant_type != "nvfp4":
        raise ValueError(
            "compression.fake_quantize_activations=True (the default) requires "
            "compression.quant_type='nvfp4'; set it to false for other quant types."
        )

    if cfg.init.method not in ("gptq", "obr", "jsq", "rtn", "random"):
        raise ValueError(
            f"Unsupported init.method={cfg.init.method!r}. Supported: 'gptq', 'obr', 'jsq', 'rtn', 'random'."
        )

    if cfg.init.method == "jsq":
        if cfg.init.jsq_edit_mode not in ("range", "quantile"):
            raise ValueError(
                f"Unsupported init.jsq_edit_mode={cfg.init.jsq_edit_mode!r}. "
                "Supported: 'range', 'quantile'."
            )
        if not 0.0 <= cfg.init.jsq_edit_r < 0.5:
            raise ValueError(f"init.jsq_edit_r must be in [0, 0.5), got {cfg.init.jsq_edit_r}.")
        if cfg.init.jsq_lambda_norm < 0:
            raise ValueError(f"init.jsq_lambda_norm must be >= 0, got {cfg.init.jsq_lambda_norm}.")
        if cfg.init.jsq_token_cap < 0:
            raise ValueError(f"init.jsq_token_cap must be >= 0 (0 = exact), got {cfg.init.jsq_token_cap}.")
        if cfg.init.jsq_range_dtype not in ("float32", "bfloat16"):
            raise ValueError(
                f"Unsupported init.jsq_range_dtype={cfg.init.jsq_range_dtype!r}. "
                "Supported: 'float32', 'bfloat16'."
            )
        if cfg.refine.enabled:
            raise ValueError(
                "init.method='jsq' is a one-shot baseline: it does not populate the "
                "init_dense_weight/init_support_mask side channels the refiner needs. "
                "Set refine.enabled: false."
            )

    if cfg.init.method == "obr":
        if cfg.init.obr_mask_metric not in ("wanda", "magnitude", "sparsegpt"):
            raise ValueError(
                f"Unsupported init.obr_mask_metric={cfg.init.obr_mask_metric!r}. "
                "Supported: 'wanda', 'magnitude', 'sparsegpt'."
            )
        if cfg.init.obr_partition not in ("column", "error"):
            raise ValueError(
                f"Unsupported init.obr_partition={cfg.init.obr_partition!r}. "
                "Supported: 'column', 'error'."
            )
        if cfg.init.obr_solver not in ("cg", "cholesky"):
            raise ValueError(
                f"Unsupported init.obr_solver={cfg.init.obr_solver!r}. Supported: 'cg', 'cholesky'."
            )
        if not 0.0 <= cfg.init.obr_alpha <= 1.0:
            raise ValueError(f"init.obr_alpha must be in [0, 1], got {cfg.init.obr_alpha}.")

    if cfg.init.hessian_chunk_tokens < 0:
        raise ValueError(
            f"init.hessian_chunk_tokens must be >= 0 (0 disables chunking), "
            f"got {cfg.init.hessian_chunk_tokens!r}."
        )

    if cfg.refine.token_weight_clip_k < 0:
        raise ValueError(
            f"refine.token_weight_clip_k must be >= 0 (0 disables), got {cfg.refine.token_weight_clip_k!r}."
        )
    if cfg.refine.scale_ft_epochs < 0:
        raise ValueError(
            f"refine.scale_ft_epochs must be >= 0 (0 disables), got {cfg.refine.scale_ft_epochs!r}."
        )
    if cfg.refine.scale_ft_epochs > 0:
        errors = []
        if quant_type != "nvfp4":
            errors.append("scale fine-tuning is an NVFP4 stage (FP8 group scales); "
                          f"compression.quant_type is {quant_type!r}")
        if not cfg.refine.enabled:
            errors.append("it runs after refinement, so refine.enabled must be true")
        lrs = cfg.refine.scale_ft_lr
        lrs = lrs if isinstance(lrs, (list, tuple)) else [lrs]
        if not lrs:
            errors.append("refine.scale_ft_lr must not be an empty list")
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0 for v in lrs):
            errors.append(f"every refine.scale_ft_lr must be a number > 0, got {cfg.refine.scale_ft_lr!r}")
        if len(lrs) != len(set(lrs)):
            errors.append(f"refine.scale_ft_lr has duplicates: {cfg.refine.scale_ft_lr!r}")
        if cfg.refine.scale_ft_lr_decay_type not in ("linear", "cosine", "constant"):
            errors.append(f"refine.scale_ft_lr_decay_type must be linear|cosine|constant, "
                          f"got {cfg.refine.scale_ft_lr_decay_type!r}")
        if errors:
            raise ValueError("Invalid refine.scale_ft_* config: " + "; ".join(errors))

    # None means "fall back to masks_lr" and is the default; an explicit 0 or negative
    # would train nothing and produce a run that looks healthy in the logs.
    for _lr_name in ("logits_lr", "group_scales_lr"):
        _lr = getattr(cfg.refine, _lr_name)
        if _lr is not None and _lr <= 0:
            raise ValueError(
                f"refine.{_lr_name} must be > 0 when set (omit it to fall back to "
                f"refine.masks_lr), got {_lr!r}."
            )

