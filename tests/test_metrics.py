"""Metrics are the thing every conclusion rests on, so they get hand-checked
cases rather than only self-consistency checks."""
from __future__ import annotations

import numpy as np
import pytest

from bench.metrics import evaluate, fill_holes


def test_perfect_prediction_scores_zero():
    gt = np.array([[1.0, 2.0, 3.0, 4.0]], np.float32)
    m = evaluate(gt.copy(), np.ones_like(gt, bool), gt)
    assert m.epe_valid == 0.0
    assert m.rmse_valid == 0.0
    assert m.density == 1.0
    assert all(v == 0.0 for v in m.bad_all.values())
    assert m.epe_filled == 0.0


def test_bad_threshold_is_strictly_greater():
    # Error of exactly 2.0 must NOT count as "bad 2.0".
    gt = np.array([[10.0, 10.0]], np.float32)
    pred = np.array([[12.0, 12.5]], np.float32)
    m = evaluate(pred, np.ones_like(pred, bool), gt)
    assert m.bad_all["2"] == pytest.approx(50.0)
    assert m.bad_all["1"] == pytest.approx(100.0)


def test_unanswered_pixels_count_as_bad_but_not_in_epe():
    gt = np.array([[5.0, 5.0, 5.0, 5.0]], np.float32)
    pred = np.array([[5.0, 5.0, np.nan, np.nan]], np.float32)
    valid = np.isfinite(pred)
    m = evaluate(pred, valid, gt)

    assert m.density == pytest.approx(0.5)
    # The two answered pixels are exact.
    assert m.epe_valid == pytest.approx(0.0)
    assert all(v == 0.0 for v in m.bad_valid.values())
    # But half the region has no answer, so bad_all is 50%.
    assert m.bad_all["2"] == pytest.approx(50.0)


def test_sparse_method_cannot_win_on_bad_all():
    """A method that answers one easy pixel must not outrank a dense one."""
    gt = np.full((1, 100), 7.0, np.float32)
    dense = gt + 0.4  # everywhere, slightly off
    sparse = np.full((1, 100), np.nan, np.float32)
    sparse[0, 0] = 7.0  # one perfect answer

    m_dense = evaluate(dense, np.ones_like(dense, bool), gt)
    m_sparse = evaluate(sparse, np.isfinite(sparse), gt)

    assert m_sparse.epe_valid < m_dense.epe_valid  # flattered by valid-only
    assert m_dense.bad_all["2"] < m_sparse.bad_all["2"]  # but not by bad_all


def test_eval_mask_restricts_region():
    gt = np.array([[1.0, 1.0]], np.float32)
    pred = np.array([[1.0, 50.0]], np.float32)
    mask = np.array([[True, False]])
    m = evaluate(pred, np.ones_like(pred, bool), gt, eval_mask=mask)
    assert m.n_eval_px == 1
    assert m.epe_valid == pytest.approx(0.0)


def test_nan_ground_truth_is_excluded():
    gt = np.array([[1.0, np.nan, 3.0]], np.float32)
    pred = np.array([[1.0, 999.0, 3.0]], np.float32)
    m = evaluate(pred, np.ones_like(pred, bool), gt)
    assert m.n_eval_px == 2
    assert m.epe_valid == pytest.approx(0.0)


def test_bias_detects_systematic_offset():
    gt = np.full((4, 4), 10.0, np.float32)
    m = evaluate(gt - 3.0, np.ones_like(gt, bool), gt)
    assert m.bias_valid == pytest.approx(-3.0)


def test_fill_holes_takes_nearer_bracketing_value():
    d = np.array([[np.nan, 2.0, np.nan, np.nan, 5.0]], np.float32)
    out = fill_holes(d, np.isfinite(d))
    # Leading hole gets the only neighbour; interior holes take min(2, 5).
    assert out.tolist() == [[2.0, 2.0, 2.0, 2.0, 5.0]]
    assert np.isfinite(out).all()


def test_fill_holes_handles_all_invalid_row():
    d = np.full((2, 3), np.nan, np.float32)
    out = fill_holes(d, np.isfinite(d))
    assert np.isfinite(out).all()


def test_fill_holes_is_identity_when_dense():
    d = np.arange(12, dtype=np.float32).reshape(3, 4)
    out = fill_holes(d, np.ones_like(d, bool))
    assert np.array_equal(out, d)


def test_empty_region_is_safe():
    gt = np.full((2, 2), np.nan, np.float32)
    m = evaluate(np.zeros((2, 2), np.float32), np.ones((2, 2), bool), gt)
    assert m.n_eval_px == 0
    assert np.isnan(m.epe_valid)


def test_flat_keys_are_unique_and_named():
    gt = np.ones((3, 3), np.float32)
    m = evaluate(gt.copy(), np.ones_like(gt, bool), gt)
    flat = m.flat()
    for key in ("bad2_all", "bad0p5_all", "bad2_valid", "epe_valid",
                "epe_filled", "density"):
        assert key in flat, key
