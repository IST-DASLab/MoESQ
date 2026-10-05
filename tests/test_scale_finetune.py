"""ScaleFinetuneCompressor: theta=0 must reproduce the refined hard weights exactly,
and any theta must export onto NVFP4's FP8 x global scale grid.

Run:  .venv/bin/python -m pytest tests/test_scale_finetune.py -q
  or  .venv/bin/python tests/test_scale_finetune.py
"""
import torch

from src.compression.expert_compressor import ExpertCompressor, hard_tensor, split_hard_output_full
from src.compression.quant.nvfp4 import NVFP4Quant, fp8_project, FP4_E2M1_VALUES, dense_scale_groupsize
from src.compression.scale_finetune import ScaleFinetuneCompressor
from src.compression.sparsity import Paired48

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16


def _paired48_support(rows, cols, g):
    # one of the six paired-4:8 patterns per block of 8, chosen at random
    patterns = torch.tensor(Paired48(DEVICE, DTYPE).possible_masks.tolist(), device=DEVICE).bool()
    idx = torch.randint(0, 6, (rows, cols // 8), generator=g, device=DEVICE)
    return patterns[idx].reshape(rows, cols)


def _refined_compressor(rows=64, cols=256, seed=0):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    W = torch.randn(rows, cols, generator=g, device=DEVICE) * 0.02
    support = _paired48_support(rows, cols, g)
    sparsity = Paired48(DEVICE, DTYPE)
    gs = dense_scale_groupsize(4, 8, 16)
    quant = NVFP4Quant((rows, cols), gs, DEVICE, DTYPE)
    comp = ExpertCompressor(
        weight_shape=(rows, cols), sparsity=sparsity, quant=quant, weight_buffer=None,
        weight_master_init=W, learn_weights=True,
        mask_logits_init=sparsity.init_mask_logits(support, 0.01, 4.0),
        device=DEVICE, dtype=DTYPE, logits_dtype=DTYPE,
    )
    # perturb the master so the weights are not a trivial GPTQ init
    with torch.no_grad():
        comp.weight_master.add_(torch.randn_like(comp.weight_master) * 0.002)
    return comp


def test_theta_zero_reproduces_refined_hard_weights():
    comp = _refined_compressor()
    ref_w, ref_scales, ref_global, ref_mask = split_hard_output_full(comp.get_hard_weights())
    ft = ScaleFinetuneCompressor.from_compressor(comp)

    w, scales, gscale, mask = ft.get_hard_weights()
    assert torch.equal(w, ref_w), "hard weight changed at theta=0"
    assert torch.equal(scales, ref_scales), "FP8 scales changed at theta=0"
    assert torch.allclose(gscale, ref_global), "global scale changed at theta=0"
    assert torch.equal(mask, ref_mask)
    # the training forward at theta=0 is the same tensor too (STE is identity there)
    assert torch.equal(ft(2.0, 100.0), ref_w)
    # codes live on the FP4 grid, and pruned positions carry code 0
    grid = torch.tensor(FP4_E2M1_VALUES, device=DEVICE)
    assert torch.isin(ft.codes.float(), grid).all()
    assert (ft.codes.float()[~mask] == 0).all()


def test_perturbed_theta_exports_onto_the_nvfp4_grid():
    comp = _refined_compressor(seed=1)
    ft = ScaleFinetuneCompressor.from_compressor(comp)
    with torch.no_grad():
        ft.log_scale.add_(torch.randn_like(ft.log_scale) * 0.05)

    w, scales, gscale, mask = ft.get_hard_weights()
    # scales are FP8-representable, positive, and reach the FP8 max via the global
    assert torch.equal(scales, fp8_project(scales))
    assert (scales > 0).all()
    assert torch.isclose(scales.amax(), torch.tensor(448.0, device=DEVICE), rtol=0.13)
    # weight / (scale/global) is on the FP4 grid, i.e. the export round-trips through
    # compressed-tensors' NVFP4 packer without re-rounding
    idx = torch.arange(w.shape[1], device=DEVICE) // ft.groupsize
    q = w.float() / (scales / gscale)[:, idx]
    grid = torch.tensor(FP4_E2M1_VALUES, device=DEVICE)
    assert (q.unsqueeze(-1) - grid).abs().amin(-1).max() < 0.05
    # sparsity is untouched
    assert torch.equal(w != 0, (w != 0) & mask)
    # theta changed the shipped weight (the stage is not a no-op)
    assert not torch.equal(w, hard_tensor(comp))


def test_forward_is_differentiable_in_theta_only():
    comp = _refined_compressor(seed=2)
    ft = ScaleFinetuneCompressor.from_compressor(comp)
    names = [n for n, _ in ft.named_parameters()]
    assert names == ["log_scale"], names
    x = torch.randn(8, ft.weight_shape[1], device=DEVICE, dtype=DTYPE)
    y = torch.nn.functional.linear(x, ft(1.0, 1.0))
    y.float().pow(2).mean().backward()
    assert ft.log_scale.grad is not None
    assert torch.isfinite(ft.log_scale.grad).all()
    assert ft.log_scale.grad.abs().sum() > 0


def test_refuses_non_nvfp4():
    class Dummy:
        quant = None
    assert not ScaleFinetuneCompressor.supports(Dummy())


def test_config_accepts_scale_ft_lr_list():
    """refine.scale_ft_lr may be a scalar or a list of distinct positive rates."""
    from src.config import MoESQConfig, validate

    def cfg(lr):
        c = MoESQConfig()
        c.compression.quant_type = "nvfp4"
        c.compression.learn_weight_values = True
        c.refine.enabled = True
        c.refine.scale_ft_epochs = 1
        c.refine.scale_ft_lr = lr
        return c

    validate(cfg(1.0e-3))
    validate(cfg([1e-3, 3e-3, 1e-2, 3e-2]))
    for bad in ([], [1e-3, 1e-3], [1e-3, 0.0], [1e-3, -1.0], [1e-3, "x"], 0.0):
        try:
            validate(cfg(bad))
        except ValueError:
            continue
        raise AssertionError(f"validate() accepted refine.scale_ft_lr={bad!r}")


def test_theta_reset_returns_the_exact_starting_scales():
    """Each sweep pass restarts from theta=0; reloading it must be bit-exact.

    _scale_finetune_layer restores init_state between rates, so a rate can only be
    compared against the others if that restore reproduces the pre-FT export exactly.
    """
    comp = _refined_compressor(seed=5)
    ft = ScaleFinetuneCompressor.from_compressor(comp)
    w0, s0, g0, m0 = ft.get_hard_weights()
    init = ft.log_scale.detach().clone()

    with torch.no_grad():
        ft.log_scale.add_(torch.randn_like(ft.log_scale) * 0.3)
    assert not torch.equal(ft.get_hard_weights()[1], s0), "perturbation did not move the scales"

    with torch.no_grad():
        ft.log_scale.copy_(init)
    w1, s1, g1, m1 = ft.get_hard_weights()
    assert torch.equal(w1, w0) and torch.equal(s1, s0)
    assert torch.equal(g1, g0) and torch.equal(m1, m0)


if __name__ == "__main__":
    for fn in [test_theta_zero_reproduces_refined_hard_weights,
               test_perturbed_theta_exports_onto_the_nvfp4_grid,
               test_forward_is_differentiable_in_theta_only,
               test_refuses_non_nvfp4,
               test_config_accepts_scale_ft_lr_list,
               test_theta_reset_returns_the_exact_starting_scales]:
        fn()
        print(f"ok  {fn.__name__}")
