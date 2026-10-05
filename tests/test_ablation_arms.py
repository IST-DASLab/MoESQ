"""The four support/value arms of the joint-adaptation ablation.

Arms (refine.learn_masks x compression.learn_weight_values):
  init-only (refine off) / fixed-mask QAT / support-only / joint.

What must hold: each arm trains exactly its own variable, the fixed-mask arm's support
is the initializer's bit for bit and its forward is deterministic, and all three refined
arms start from the same hard weights, so any divergence is adaptation and not setup.

Run:  .venv/bin/python -m pytest tests/test_ablation_arms.py -q
"""
import copy

import torch

from src.compression.builders import build_compressor
from src.compression.expert_compressor import split_hard_output_full
from src.compression.quant.nvfp4 import dense_scale_groupsize, fp4_project
from src.compression.sparsity import Paired48
from src.config import MoESQConfig, validate

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16
ROWS, COLS = 64, 256


def _config(learn_masks=True, learn_weight_values=True):
    cfg = MoESQConfig()
    cfg.compression.prunen, cfg.compression.prunem = 4, 8
    cfg.compression.quant_type = "nvfp4"
    cfg.compression.groupsize = 16
    cfg.compression.learn_weight_values = learn_weight_values
    cfg.refine.enabled = True
    cfg.refine.learn_masks = learn_masks
    cfg.refine.std, cfg.refine.strength = 0.01, 4.0
    cfg.refine.logits_dtype = "bfloat16"
    return cfg


def _init_tensors(seed=0):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    patterns = Paired48(DEVICE, DTYPE).possible_masks.bool()
    idx = torch.randint(0, 6, (ROWS, COLS // 8), generator=g, device=DEVICE)
    support = patterns[idx].reshape(ROWS, COLS)
    dense = torch.randn(ROWS, COLS, generator=g, device=DEVICE) * 0.02
    gs = dense_scale_groupsize(4, 8, 16)
    scales = dense.reshape(ROWS, -1, gs).abs().amax(-1) / 6.0
    return dense, dense.masked_fill(~support, 0), support, scales.clamp(min=1e-12)


def _build(cfg, seed=0):
    dense, compressed, support, scales = _init_tensors(seed)
    return build_compressor(
        cfg, DEVICE, DTYPE,
        init_compressed_weight=compressed, init_scales=scales,
        init_dense_weight=dense, init_support_mask=support,
    ), support


def _trainable(compressor):
    return sorted(n for n, _ in compressor.named_parameters())


def test_each_arm_exposes_only_its_own_variable():
    joint, _ = _build(_config(True, True))
    qat, _ = _build(_config(False, True))
    mask_only, _ = _build(_config(True, False))
    assert _trainable(joint) == ["mask_logits", "weight_master"]
    assert _trainable(qat) == ["weight_master"]
    assert _trainable(mask_only) == ["mask_logits"]
    # the frozen-value arm keeps the DENSE solution, not the masked one, so every
    # candidate position carries a meaningful value
    assert mask_only.W.dtype == torch.float32
    assert (mask_only.W != 0).all()


def test_fixed_mask_arm_ships_the_initializer_support_and_is_deterministic():
    qat, support = _build(_config(False, True))
    assert torch.equal(qat.hard_mask(), support)
    _, _, _, mask = split_hard_output_full(qat.get_hard_weights())
    assert torch.equal(mask, support)
    a = qat(temperature=2.0, scale=100.0)
    b = qat(temperature=0.05, scale=500.0)
    assert torch.equal(a, b)          # no Gumbel sample, so the anneal cannot move it
    assert qat.mask_logits is None    # and no logits are allocated at all


def test_gradients_reach_only_the_trainable_variable():
    for learn_masks, learn_values in ((True, True), (False, True), (True, False)):
        comp, _ = _build(_config(learn_masks, learn_values))
        comp(temperature=1.0, scale=100.0).float().pow(2).sum().backward()
        for name, p in comp.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), (name, learn_masks, learn_values)
            assert p.grad.abs().sum() > 0, (name, learn_masks, learn_values)


def test_all_arms_start_from_the_same_hard_weights():
    ref = None
    for learn_masks, learn_values in ((True, True), (False, True), (True, False)):
        comp, _ = _build(_config(learn_masks, learn_values))
        weight, scales, gscale, _ = split_hard_output_full(comp.get_hard_weights())
        if ref is None:
            ref = (weight, scales, gscale)
        else:
            assert torch.equal(weight, ref[0])
            assert torch.equal(scales, ref[1])
            assert torch.equal(gscale, ref[2])


def test_support_only_arm_moves_the_scale_through_the_mask():
    # the coupling the joint hypothesis rests on: with V frozen, changing the support
    # still changes the exported block scale, so the mask gradient sees the quantizer
    comp, _ = _build(_config(True, False))
    before = comp.effective_scales().clone()
    with torch.no_grad():
        comp.mask_logits.copy_(torch.randn_like(comp.mask_logits))
    after = comp.effective_scales()
    assert not torch.equal(before, after)
    assert torch.equal(comp.init_scale_ref, before)


def test_validator_rejects_the_degenerate_and_unwired_combinations():
    cfg = _config(False, False)
    try:
        validate(cfg)
    except ValueError as e:
        assert "nothing to train" in str(e)
    else:
        raise AssertionError("both-frozen + refine.enabled must be rejected")

    cfg = _config(False, True)
    cfg.compression.quant_type = None
    cfg.compression.learn_weight_values = False
    try:
        validate(cfg)
    except ValueError as e:
        assert "learn_masks" in str(e)
    else:
        raise AssertionError("learn_masks=false outside nvfp4 must be rejected")


def test_diagnostic_buffers_track_the_initializer():
    joint, support = _build(_config(True, True))
    assert torch.equal(joint.init_support, support)
    # GPTQ's support survives the init-logit noise: the argmax gap is strength*2 = 8 std
    assert torch.equal(joint.hard_mask(), support)
    qat, _ = _build(_config(False, True))
    assert qat.init_support is None   # 0 by construction, nothing to store




def test_decomposition_residuals_add_exactly_in_weight_space():
    """d_S + d_R = d_SQ, which is what makes the interaction term a decomposition.

    This is the property E_Q cannot provide: quantizing the dense weight charges rounding
    on the pruned half too, so it over-counts the quantization share by ~2x.
    """
    comp, support = _build(_config(True, True))
    W0 = torch.randn(ROWS, COLS, device=DEVICE) * 0.02
    mask = comp.hard_mask()
    with torch.no_grad():
        quantized = comp.quant.quantize_hard(mask.to(torch.float32) * W0)[0]
        w_sq = quantized                                   # pruned positions are zero
        w_r = torch.where(mask, quantized, W0)             # pruned positions keep W0
        w_s = mask.to(torch.float32) * W0
    d_s, d_r, d_sq = w_s - W0, w_r - W0, w_sq - W0
    assert torch.allclose(d_s + d_r, d_sq, atol=1e-6), (d_s + d_r - d_sq).abs().max()
    # and the two pieces live on disjoint supports, as the derivation claims
    assert (d_s[mask] == 0).all()
    assert (d_r[~mask] == 0).all()


def test_freeze_support_hardens_and_stops_training():
    """support_first phase 2 must see a FIXED HARD mask, not a live sampler."""
    c, _ = _build(_config())
    assert c.learn_masks and c.learn_weights
    before = c.hard_mask().clone()
    c.freeze_support()
    assert not c.learn_masks, "learn_masks must be off after freeze_support"
    assert c.mask_logits is None, "mask logits must leave parameters()"
    assert torch.equal(c.fixed_mask, before), "frozen support must equal the pre-switch hard mask"
    names = {n for n, _ in c.named_parameters()}
    assert "mask_logits" not in names, f"mask_logits still trainable: {names}"
    assert "weight_master" in names, "values must remain trainable in phase 2"
    # the sampler is gone: two forwards at a sampling temperature must agree
    a = c.forward(2.0, 100.0)
    b = c.forward(2.0, 100.0)
    assert torch.equal(a, b), "support still stochastic after freeze_support"
    # and the mask genuinely does not move
    assert torch.equal(c.hard_mask(), before)
    c.freeze_support()  # idempotent


def test_freeze_values_holds_fp32_and_stops_training():
    """values_first phase 2 must hold the learned values exactly, in fp32."""
    c, _ = _build(_config())
    with torch.no_grad():
        c.weight_master.add_(0.123)      # pretend phase 1 moved them
        expected = c.weight_master.detach().clone().float()
    c.freeze_values()
    assert not c.learn_weights
    assert c.weight_master is None
    assert c.W.dtype is torch.float32, f"values must stay fp32, got {c.W.dtype}"
    assert torch.equal(c.W, expected), "frozen values must equal the pre-switch values"
    names = {n for n, _ in c.named_parameters()}
    assert "weight_master" not in names, f"weight_master still trainable: {names}"
    assert "mask_logits" in names, "support must remain trainable in phase 2"
    c.freeze_values()  # idempotent


def test_sequential_config_validation():
    from src import config as cfgmod

    def _raises(c, frag):
        try:
            cfgmod.validate(c)
        except ValueError as e:
            assert frag in str(e), f'wrong error for {frag!r}: {e}'
            return
        raise AssertionError(f'expected ValueError containing {frag!r}')

    base = _config()
    base.refine.sequential = "support_first"
    base.refine.sequential_phase1_epochs = None
    _raises(base, "sequential_phase1_epochs")
    base.refine.sequential_phase1_epochs = base.refine.num_epochs
    _raises(base, "sequential_phase1_epochs")
    base.refine.sequential_phase1_epochs = 5
    cfgmod.validate(base)                      # valid
    base.refine.learn_masks = False
    _raises(base, "BOTH variables")
    base.refine.learn_masks = True
    base.refine.sequential = "sideways"
    _raises(base, "must be null")


def test_multi_optimizer_is_scheduler_compatible():
    """Adam-on-logits needs two optimizers; the LR scheduler must still drive both."""
    import torch.optim as optim
    from lion_pytorch import Lion
    from src.trainer import MultiOptimizer, CustomLRScheduler

    a = torch.nn.Parameter(torch.randn(4, 6))
    b = torch.nn.Parameter(torch.randn(4, 8))
    gm = {"name": "mask_logits", "params": [a], "lr": 1e-3, "lr_decay_tag": True}
    gw = {"name": "weight_master", "params": [b], "lr": 5e-5, "lr_decay_tag": True}
    mo = MultiOptimizer([optim.Adam([gm], betas=(0.9, 0.999)), Lion([gw], betas=(0.9, 0.95))])

    groups = mo.param_groups
    assert len(groups) == 2, f"expected 2 groups, got {len(groups)}"
    assert [g["name"] for g in groups] == ["mask_logits", "weight_master"], "group order unstable"
    # the property must hand back the SAME dicts, or scheduler mutation is lost
    assert mo.param_groups[0] is groups[0], "param_groups returns copies; scheduler writes would be dropped"

    sched = CustomLRScheduler(mo, total_steps=10, warmup_steps=0, lr_decay_type="linear", min_lr=0.0)
    assert sched.initial_lrs == [1e-3, 5e-5]
    for _ in range(5):
        sched.step()
    assert mo.param_groups[0]["lr"] < 1e-3, "mask lr did not decay"
    assert mo.param_groups[1]["lr"] < 5e-5, "weight lr did not decay"

    # both parameters actually move, and each under its own rule
    a0, b0 = a.detach().clone(), b.detach().clone()
    mo.zero_grad()
    (a.sum() * 1e-3 + b.sum() * 1e-3).backward()
    mo.step()
    assert not torch.equal(a.detach(), a0), "Adam did not step the logits"
    assert not torch.equal(b.detach(), b0), "Lion did not step the weights"


def test_scale_grad_surrogate_leaves_the_forward_bit_exact():
    """p-norm surrogate must change ONLY the backward -- the exported values must not move."""
    from src.compression.quant.nvfp4 import NVFP4Quant, dense_scale_groupsize
    gs = dense_scale_groupsize(4, 8, 16)
    w = torch.randn(ROWS, COLS, device=DEVICE) * 0.02
    base = NVFP4Quant((ROWS, COLS), gs, DEVICE, DTYPE)
    surr = NVFP4Quant((ROWS, COLS), gs, DEVICE, DTYPE, scale_grad_p=8.0)
    a = base.fake_quantize(w.clone())
    b = surr.fake_quantize(w.clone())
    assert torch.equal(a, b), "surrogate changed the forward; export would differ"

    # and the backward genuinely differs: detached gives no scale path at all
    w1 = w.clone().requires_grad_(True)
    w2 = w.clone().requires_grad_(True)
    up = torch.randn_like(w)
    (base.fake_quantize(w1) * up).sum().backward()
    (surr.fake_quantize(w2) * up).sum().backward()
    assert not torch.allclose(w1.grad, w2.grad), "surrogate did not change the gradient"

    # the surrogate must spread over MORE than one element per group
    diff = (w2.grad - w1.grad).abs().reshape(ROWS, -1, gs)
    touched = (diff > diff.amax(dim=-1, keepdim=True) * 1e-3).float().sum(-1).mean().item()
    assert touched > 1.5, f"surrogate still one-hot: {touched:.2f} elements per group"


def test_coupling_probe_measures_the_dropped_term():
    """C = <g, q-r> must be captured, and be zero when there is no rounding residual."""
    from src.compression.quant.nvfp4 import NVFP4Quant, dense_scale_groupsize
    gs = dense_scale_groupsize(4, 8, 16)
    acc = dict(groups=0.0, dropped=0.0, kept=0.0, ratio=0.0, ste_resid=0.0, r_mag=0.0)
    q = NVFP4Quant((ROWS, COLS), gs, DEVICE, DTYPE, coupling_stats=acc)
    w = (torch.randn(ROWS, COLS, device=DEVICE) * 0.02).requires_grad_(True)
    out = q.fake_quantize(w)
    (out * torch.randn_like(out)).sum().backward()
    assert acc['groups'] > 0, "probe captured nothing"
    assert acc['ste_resid'] > 0, "STE residual should be nonzero on random weights"
    assert acc['ratio'] > 0, "dropped/kept should be nonzero"

    # must be a no-op under no_grad -- fake_quantize runs there during validation
    acc2 = dict(groups=0.0, dropped=0.0, kept=0.0, ratio=0.0, ste_resid=0.0, r_mag=0.0)
    q2 = NVFP4Quant((ROWS, COLS), gs, DEVICE, DTYPE, coupling_stats=acc2)
    with torch.no_grad():
        q2.fake_quantize(torch.randn(ROWS, COLS, device=DEVICE) * 0.02)
    assert acc2['groups'] == 0.0, "probe should not fire under no_grad"


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
