"""The block-error sink/ordinary split must be exact and anchored to the DENSE energy.

The split exists to separate "this layer is badly fit" from "this layer's metric is
sink-weighted". Two properties make it trustworthy:
  1. sink + ordinary == total, so nothing is double counted or dropped;
  2. the sink token set is chosen by the DENSE block output, so it does not move when
     the compression changes -- otherwise the split would track the error it measures.

Run:  .venv/bin/python tests/test_block_error_split.py
"""
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _split(y_err, y_dense, k=8):
    """Mirror of the arithmetic in qwen3_moe.moe_val_recon_metrics."""
    n2 = y_dense.pow(2).sum(dim=-1)
    kk = min(k, n2.numel())
    top = n2.topk(kk)
    e2 = y_err.pow(2).sum(dim=-1)
    block_se = y_err.pow(2).sum()
    sink_se = e2[top.indices].sum()
    sink_n = float(kk * y_err.shape[1])
    block_n = float(y_err.numel())
    return (block_se.item(), block_n, sink_se.item(), sink_n, top.indices)


def test_split_is_exact_and_sums_to_the_total():
    g = torch.Generator(device=DEVICE).manual_seed(0)
    y_dense = torch.randn(64, 32, generator=g, device=DEVICE)
    y_err = torch.randn(64, 32, generator=g, device=DEVICE) * 0.01
    block_se, block_n, sink_se, sink_n, _ = _split(y_err, y_dense)
    ord_se, ord_n = block_se - sink_se, block_n - sink_n
    assert abs((sink_se + ord_se) - block_se) < 1e-4 * block_se
    assert sink_n + ord_n == block_n
    assert ord_se > 0 and sink_se > 0
    # per-element normalisation, comparable to block_error itself
    assert 0 < sink_se / sink_n and 0 < ord_se / ord_n


def test_sink_set_follows_the_dense_output_not_the_error():
    """A massive-activation token must be selected even when its error is tiny.

    If the split keyed on the error instead, a layer that fits the sink perfectly would
    report an empty sink bucket and its ordinary error would silently absorb the sinks.
    """
    g = torch.Generator(device=DEVICE).manual_seed(1)
    y_dense = torch.randn(64, 32, generator=g, device=DEVICE) * 0.01
    y_dense[3] *= 1000.0          # the sink token: 10^3x the others, as in Qwen3 L1-3
    y_err = torch.randn(64, 32, generator=g, device=DEVICE)
    y_err[3] *= 1e-6              # ...fitted almost perfectly
    _, _, sink_se, _, idx = _split(y_err, y_dense)
    assert 3 in idx.tolist(), "sink token was not selected despite dominating the dense output"
    assert sink_se < y_err.pow(2).sum().item() * 0.5, "a well-fitted sink should carry little error"


def test_a_sink_weighted_layer_is_distinguishable_from_a_badly_fit_one():
    """The whole point: two layers with the SAME total block error, one sink-located."""
    g = torch.Generator(device=DEVICE).manual_seed(2)
    y_dense = torch.randn(64, 32, generator=g, device=DEVICE) * 0.01
    y_dense[:8] *= 1000.0

    err_sink = torch.zeros(64, 32, device=DEVICE)
    err_sink[:8] = 1.0                                   # all error on the sink tokens
    err_spread = torch.full((64, 32), (8.0 / 64) ** 0.5, device=DEVICE)   # same total, spread
    assert abs(err_sink.pow(2).sum() - err_spread.pow(2).sum()) < 1e-3

    _, _, s_sink, sn, _ = _split(err_sink, y_dense)
    tot_a = err_sink.pow(2).sum().item()
    _, _, s_spread, _, _ = _split(err_spread, y_dense)
    tot_b = err_spread.pow(2).sum().item()

    ord_a = (tot_a - s_sink) / (err_sink.numel() - sn)
    ord_b = (tot_b - s_spread) / (err_spread.numel() - sn)
    assert ord_a < ord_b / 100, (
        f"ordinary-token error failed to separate the two layers: {ord_a:.3e} vs {ord_b:.3e}")


if __name__ == "__main__":
    for fn in [test_split_is_exact_and_sums_to_the_total,
               test_sink_set_follows_the_dense_output_not_the_error,
               test_a_sink_weighted_layer_is_distinguishable_from_a_badly_fit_one]:
        fn()
        print(f"ok  {fn.__name__}")
