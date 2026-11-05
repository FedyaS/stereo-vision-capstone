"""Correctness of the from-scratch stages.

The important tests here compare a fast vectorised implementation against a
slow, obviously-correct reference. That is the only way to trust a
cost-volume: an off-by-one in the shift direction still produces a
plausible-looking disparity map.
"""
from __future__ import annotations

import numpy as np
import pytest

from bench.algorithms.scratch import (
    BIG,
    COST_SCALE,
    ScratchMatcher,
    census_transform,
    cost_volume,
    geometric_validity,
    left_right_check,
    normalize_cost,
    potts_scanline_dp,
    sgm_aggregate,
    winner_takes_all,
)


# --------------------------------------------------------------------------
# the cost volume against a brute-force reference
# --------------------------------------------------------------------------


def _naive_sad_cost(left, right, ndisp, block):
    """The textbook triple loop: for every pixel, every disparity, sum |diff|.

    This is the same computation as the nested-loop block matcher in
    ``notebooks/practice/StereoAlgo1.ipynb``, kept as the reference the vectorised
    version must reproduce.
    """
    lf = left.astype(np.float64)
    rf = right.astype(np.float64)
    h, w = lf.shape[:2]
    r = block // 2
    out = np.full((ndisp, h, w), np.nan)
    for d in range(ndisp):
        for y in range(r, h - r):
            for x in range(r, w - r):
                if x - d < 0:
                    continue
                total = 0.0
                for dy in range(-r, r + 1):
                    for dx in range(-r, r + 1):
                        xr = x + dx - d
                        if 0 <= xr < w:
                            total += np.abs(
                                lf[y + dy, x + dx] - rf[y + dy, xr]
                            ).sum()
                out[d, y, x] = total / (block * block)
    return out


def test_vectorised_sad_matches_naive_reference():
    rng = np.random.default_rng(0)
    left = rng.integers(0, 255, (14, 20, 3), dtype=np.uint8)
    right = rng.integers(0, 255, (14, 20, 3), dtype=np.uint8)
    block, ndisp = 3, 5

    fast = cost_volume(left, right, ndisp, metric="sad", block=block)
    slow = _naive_sad_cost(left, right, ndisp, block)

    # Compare only where both are computing the same thing: rows at least `r`
    # from the top and bottom (the reference is undefined past that, the fast
    # path reflects), and columns at least `d + r` from the left, so the whole
    # window has a real right-image counterpart with no boundary fill in it.
    r = block // 2
    h, w = left.shape[:2]
    for d in range(ndisp):
        lo, hi = max(r, d + r), w - r
        assert hi > lo
        region = slow[d, r : h - r, lo:hi]
        got = fast[d, r : h - r, lo:hi]
        assert not np.isnan(region).any()
        assert np.allclose(got, region, rtol=1e-5, atol=1e-4), f"mismatch at d={d}"


def test_cost_volume_recovers_a_known_shift():
    """A pure translation must put the global minimum at the true disparity."""
    rng = np.random.default_rng(3)
    base = rng.integers(0, 255, (40, 60), dtype=np.uint8)
    shift = 7
    left = base
    right = np.zeros_like(base)
    right[:, : 60 - shift] = base[:, shift:]

    for metric, block in (("sad", 5), ("ssd", 5), ("zncc", 7), ("census", 5)):
        cost = cost_volume(left, right, 16, metric=metric, block=block)
        disp, valid = winner_takes_all(cost, subpixel=False)
        interior = np.zeros_like(valid)
        interior[8:-8, 20:-8] = True
        sel = valid & interior
        assert sel.sum() > 50
        # Allow a 1 px tolerance: a box window on random texture occasionally
        # ties with its neighbour.
        assert np.abs(disp[sel] - shift).max() <= 1, metric


def test_invalid_strip_is_marked_big_not_zero():
    rng = np.random.default_rng(1)
    left = rng.integers(0, 255, (10, 20, 3), dtype=np.uint8)
    right = rng.integers(0, 255, (10, 20, 3), dtype=np.uint8)
    cost = cost_volume(left, right, 8, metric="sad", block=3)
    for d in range(1, 8):
        assert np.all(cost[d, :, :d] == BIG)


# --------------------------------------------------------------------------
# census transform
# --------------------------------------------------------------------------


def test_census_is_invariant_to_monotonic_intensity_change():
    rng = np.random.default_rng(5)
    img = rng.integers(20, 150, (25, 30), dtype=np.uint8)
    # Gain and offset preserve the ordering of neighbours, so the codes match.
    # The range is kept well clear of 255: saturation would clip distinct
    # values together and destroy the ordering the invariance relies on.
    brighter = (img.astype(np.int32) * 1.5 + 5).astype(np.uint8)
    assert brighter.max() < 255
    assert np.array_equal(census_transform(img), census_transform(brighter))


def test_census_differs_when_structure_differs():
    rng = np.random.default_rng(6)
    a = rng.integers(0, 255, (20, 20), dtype=np.uint8)
    b = rng.integers(0, 255, (20, 20), dtype=np.uint8)
    assert not np.array_equal(census_transform(a), census_transform(b))


def test_census_window_cannot_exceed_64_bits():
    with pytest.raises(ValueError):
        census_transform(np.zeros((8, 8), np.uint8), win_h=9, win_w=9)


# --------------------------------------------------------------------------
# dynamic programming against an exact reference
# --------------------------------------------------------------------------


def _naive_potts_dp(cost_row, penalty):
    """O(W * D^2) Viterbi on one row -- the definition, written out."""
    dnum, w = cost_row.shape
    f = cost_row[:, 0].astype(np.float64).copy()
    back = np.zeros((w, dnum), np.int32)
    for x in range(1, w):
        new = np.empty(dnum)
        for d in range(dnum):
            best, arg = np.inf, -1
            for k in range(dnum):
                v = f[k] + (0.0 if k == d else penalty)
                if v < best:
                    best, arg = v, k
            new[d] = cost_row[d, x] + best
            back[x, d] = arg
        f = new
    d = int(np.argmin(f))
    path = np.zeros(w, np.int32)
    path[w - 1] = d
    for x in range(w - 1, 0, -1):
        d = back[x, d]
        path[x - 1] = d
    return path


def test_potts_dp_matches_exact_viterbi():
    rng = np.random.default_rng(11)
    dnum, h, w = 6, 3, 15
    cost = rng.random((dnum, h, w)).astype(np.float32) * 10.0
    penalty = 3.0

    fast = potts_scanline_dp(cost, jump_penalty=penalty)
    for y in range(h):
        ref = _naive_potts_dp(cost[:, y, :], penalty)
        fast_row = fast[y].astype(np.int32)
        # Compare the achieved objective rather than the path: ties are real,
        # and two different optimal paths are both correct.
        def objective(path):
            total = cost[path, y, np.arange(w)].sum()
            return total + penalty * np.count_nonzero(np.diff(path))

        assert objective(fast_row) == pytest.approx(objective(ref), rel=1e-5)


def test_potts_dp_with_huge_penalty_is_constant_per_row():
    rng = np.random.default_rng(12)
    cost = rng.random((5, 4, 12)).astype(np.float32)
    out = potts_scanline_dp(cost, jump_penalty=1e6)
    for y in range(out.shape[0]):
        assert len(np.unique(out[y])) == 1


# --------------------------------------------------------------------------
# aggregation and validation helpers
# --------------------------------------------------------------------------


def test_sgm_preserves_shape_and_is_finite():
    rng = np.random.default_rng(13)
    cost = normalize_cost(rng.random((8, 10, 12)).astype(np.float32) * 50)
    for paths in (4, 8):
        agg = sgm_aggregate(cost, p1=10, p2=120, paths=paths)
        assert agg.shape == cost.shape
        assert np.isfinite(agg).all()


def test_sgm_rejects_bad_path_count():
    cost = np.zeros((3, 4, 5), np.float32)
    with pytest.raises(ValueError):
        sgm_aggregate(cost, paths=5)


def test_sgm_overrules_a_single_outlier_pixel():
    """One pixel voting for a far-away disparity must be pulled back.

    This is the whole purpose of the P2 term: an isolated pixel preferring a
    disparity nine steps from its neighbours' has to pay the large jump
    penalty twice, which outweighs its local cost advantage.
    """
    dnum, h, w = 10, 9, 21
    cost = np.full((dnum, h, w), 10.0, np.float32)
    cost[3] = 0.0  # everywhere prefers d = 3
    # One pixel instead prefers d = 9, as strongly as its neighbours prefer 3.
    cost[3, 4, 10] = 10.0
    cost[9, 4, 10] = 0.0

    raw, _ = winner_takes_all(cost, subpixel=False)
    assert raw[4, 10] == 9, "the un-aggregated cost should fall for the outlier"

    agg = sgm_aggregate(normalize_cost(cost), p1=10, p2=120, paths=8)
    smoothed, _ = winner_takes_all(agg, subpixel=False)
    assert smoothed[4, 10] == 3
    # And the rest of the image is untouched.
    assert (smoothed[:, 12:] == 3).all()


def test_normalize_cost_maps_to_expected_range_and_keeps_big():
    rng = np.random.default_rng(14)
    cost = rng.random((6, 8, 9)).astype(np.float32) * 700
    cost[2, :, :2] = BIG
    out = normalize_cost(cost)
    finite = out[out < BIG * 0.5]
    assert finite.min() >= 0.0
    assert finite.max() <= float(COST_SCALE) + 1e-4
    assert np.all(out[2, :, :2] == BIG)


def test_geometric_validity_rejects_out_of_image_matches():
    disp = np.array([[0.0, 5.0, 1.0, 2.0]], np.float32)
    # x=1 with d=5 would match at x=-4, which does not exist.
    assert geometric_validity(disp).tolist() == [[True, False, True, True]]


def test_left_right_check_flags_disagreement():
    # Left says 2 everywhere; right map agrees at x=5 but not at x=9.
    disp_l = np.full((1, 12), 2.0, np.float32)
    disp_r = np.full((1, 12), 2.0, np.float32)
    disp_r[0, 7] = 9.0  # x=9 looks up right pixel 7
    ok = left_right_check(disp_l, disp_r, tolerance=1.0)
    assert ok[0, 5]
    assert not ok[0, 9]


def test_subpixel_refinement_lands_between_integers():
    # A parabola with its vertex at d = 2.25, over an image wide enough that
    # geometric validity (x >= d) is satisfied.
    dnum, w = 6, 10
    cost = np.empty((dnum, 1, w), np.float32)
    for d in range(dnum):
        cost[d, 0, :] = (d - 2.25) ** 2
    disp, valid = winner_takes_all(cost, subpixel=True)
    assert valid[0, 5]
    assert disp[0, 5] == pytest.approx(2.25, abs=0.02)


def test_subpixel_can_be_disabled():
    dnum, w = 6, 10
    cost = np.empty((dnum, 1, w), np.float32)
    for d in range(dnum):
        cost[d, 0, :] = (d - 2.25) ** 2
    disp, _ = winner_takes_all(cost, subpixel=False)
    assert disp[0, 5] == 2.0


def test_uniqueness_ratio_rejects_ambiguous_pixels():
    dnum = 8
    # Two equally good minima -> ambiguous, must be rejected.
    cost = np.full((dnum, 1, 2), 10.0, np.float32)
    cost[1, 0, 0] = 1.0
    cost[6, 0, 0] = 1.0
    # A single clear minimum -> accepted.
    cost[1, 0, 1] = 1.0
    _, valid = winner_takes_all(cost, subpixel=False, uniqueness_ratio=0.25)
    assert not valid[0, 0]
    assert valid[0, 1]


def test_cost_volume_memory_guard_raises():
    from bench.algorithms.scratch import VolumeTooLarge

    left = np.zeros((500, 500, 3), np.uint8)
    with pytest.raises(VolumeTooLarge):
        cost_volume(left, left, 256, max_volume_gb=0.001)


def test_lr_check_with_dp_is_rejected_not_ignored():
    from bench.types import StereoPair

    left = np.zeros((12, 16, 3), np.uint8)
    pair = StereoPair(name="t", left=left, right=left, ndisp=4)
    with pytest.raises(ValueError, match="lr_check"):
        ScratchMatcher(aggregation="dp", lr_check=True).compute(pair)


def test_unknown_options_raise():
    left = np.zeros((8, 8, 3), np.uint8)
    with pytest.raises(ValueError):
        cost_volume(left, left, 4, metric="nope")
    with pytest.raises(ValueError):
        ScratchMatcher(aggregation="nope").compute(
            __import__("bench.types", fromlist=["StereoPair"]).StereoPair(
                name="t", left=left, right=left, ndisp=4
            )
        )
