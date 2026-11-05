"""Contract tests: every registered algorithm must honour the same interface.

Parametrised over the registry, so a newly added algorithm is tested the
moment it is registered.
"""
from __future__ import annotations

import numpy as np
import pytest

from bench.algorithms import all_specs, select
from bench.algorithms.base import from_opencv_fixed_point, round_up_16
from bench.metrics import evaluate

SPECS = all_specs()
AVAILABLE = [s for s in SPECS if s.availability()[0]]


def test_registry_is_populated():
    assert len(SPECS) >= 15
    families = {s.family for s in SPECS}
    assert {"opencv", "scratch", "torch", "learned"} <= families


def test_names_are_unique():
    names = [s.name for s in SPECS]
    assert len(names) == len(set(names))


def test_availability_never_raises():
    for s in SPECS:
        ok, why = s.availability()
        assert isinstance(ok, bool)
        assert isinstance(why, str)
        if not ok:
            assert why, f"{s.name} unavailable without a reason"


def test_every_spec_has_a_description():
    for s in SPECS:
        assert s.description.strip()


@pytest.mark.parametrize("spec", AVAILABLE, ids=lambda s: s.name)
def test_algorithm_honours_the_output_contract(spec, tiny_pair):
    """Shape, dtype, nan-for-invalid, and no negative disparities."""
    algo = spec.build()
    result = algo.compute(tiny_pair)
    disp, valid = result.normalized()

    h, w = tiny_pair.shape
    assert disp.shape == (h, w), f"{spec.name} returned {disp.shape}"
    assert disp.dtype == np.float32
    assert valid.shape == (h, w)
    assert valid.dtype == np.bool_

    # Invalid pixels must be nan, not 0 or -1 masquerading as a real estimate.
    assert np.all(np.isnan(disp[~valid]))
    if valid.any():
        answered = disp[valid]
        assert np.isfinite(answered).all()
        assert (answered >= 0).all(), f"{spec.name} produced negative disparity"


@pytest.mark.parametrize("spec", AVAILABLE, ids=lambda s: s.name)
def test_algorithm_is_deterministic(spec, tiny_pair):
    algo = spec.build()
    a, _ = algo.compute(tiny_pair).normalized()
    b, _ = algo.compute(tiny_pair).normalized()
    assert np.array_equal(np.nan_to_num(a, nan=-1), np.nan_to_num(b, nan=-1)), (
        f"{spec.name} is not reproducible"
    )


@pytest.mark.parametrize("spec", AVAILABLE, ids=lambda s: s.name)
def test_algorithm_beats_a_constant_guess(spec, sanity_pair):
    """A sanity floor: richly textured synthetic data is not a hard scene.

    Anything that cannot beat "predict the mean disparity everywhere" has a
    bug, not a limitation. Deliberately a weak bar -- it catches inverted
    disparity signs, off-by-one shifts and broken validity masks, which all
    still produce a plausible-looking map.
    """
    algo = spec.build()
    disp, valid = algo.compute(sanity_pair).normalized()
    got = evaluate(disp, valid, sanity_pair.gt, eval_mask=sanity_pair.occ_mask)

    const = np.full(sanity_pair.shape, float(np.nanmean(sanity_pair.gt)), np.float32)
    baseline = evaluate(
        const, np.ones(sanity_pair.shape, bool), sanity_pair.gt,
        eval_mask=sanity_pair.occ_mask,
    )

    if got.density < 0.02:
        pytest.skip(f"{spec.name} answered almost nothing on this pair")
    assert got.bad_all["2"] < baseline.bad_all["2"], (
        f"{spec.name}: bad2_all {got.bad_all['2']:.1f}% is no better than the "
        f"constant-guess baseline {baseline.bad_all['2']:.1f}%"
    )


def test_numpy_and_torch_backends_agree(tiny_pair):
    """The same maths in two libraries must give the same answer.

    This is the cross-check that catches a transcription error in either
    backend -- an indexing slip in one would not reproduce in the other.
    """
    np_disp, np_valid = select(["scratch_sad_wta"])[0].build().compute(
        tiny_pair
    ).normalized()
    t_disp, t_valid = select(["torch_sad_wta"])[0].build().compute(
        tiny_pair
    ).normalized()

    both = np_valid & t_valid
    assert both.mean() > 0.5
    # Both accumulate in float32 with a different order of operations, so the
    # sub-pixel fraction can differ slightly; the integer winner must not.
    diff = np.abs(np_disp[both] - t_disp[both])
    assert diff.mean() < 0.02, f"mean disagreement {diff.mean():.4f}"
    assert np.quantile(diff, 0.99) < 1.0, "backends pick different winners"


# --------------------------------------------------------------------------
# selection and adapter helpers
# --------------------------------------------------------------------------


def test_select_by_family_tag_glob_and_name():
    assert {s.name for s in select(["opencv"])} == {
        s.name for s in SPECS if s.family == "opencv"
    }
    assert all("sgbm" in s.name for s in select(["opencv_sgbm*"]))
    assert {s.name for s in select(["opencv_bm"])} == {"opencv_bm"}
    assert select(["census"]), "tag selection returned nothing"
    assert len(select(None)) == len(SPECS)
    assert len(select(["all"])) == len(SPECS)


def test_select_rejects_unknown_selector():
    with pytest.raises(KeyError):
        select(["definitely-not-an-algorithm"])


def test_select_deduplicates_overlapping_selectors():
    both = select(["opencv", "opencv_bm"])
    assert len(both) == len({s.name for s in both})


def test_round_up_16():
    assert round_up_16(1) == 16
    assert round_up_16(16) == 16
    assert round_up_16(17) == 32
    assert round_up_16(64) == 64
    assert round_up_16(70) == 80


def test_opencv_fixed_point_conversion_marks_no_match_invalid():
    # OpenCV stores disparity * 16 and uses (min_disp - 1) * 16 for no match.
    raw = np.array([[-16, 0, 160, 32]], np.int16)
    res = from_opencv_fixed_point(raw, min_disp=0)
    disp, valid = res.normalized()
    assert not valid[0, 0]
    assert np.isnan(disp[0, 0])
    assert disp[0, 2] == pytest.approx(10.0)
    assert disp[0, 3] == pytest.approx(2.0)


# --------------------------------------------------------------------------
# the torch backend must stay a backend, not a second algorithm
# --------------------------------------------------------------------------

#: (scratch name, torch name) pairs that must compute the same function.
MIRRORED = [
    ("scratch_sad_wta", "torch_sad_wta"),
    ("scratch_zncc_wta", "torch_zncc_wta"),
    ("scratch_census_sgm4", "torch_census_sgm4"),
]


@pytest.mark.parametrize("scratch_name,torch_name", MIRRORED)
def test_torch_defaults_mirror_scratch(scratch_name, torch_name):
    """Shared parameters must be identical between the two backends.

    The whole point of the torch entries is to isolate the backend's effect,
    which only holds if both run the same configuration. Tuning them
    separately -- or sweeping them over different grids, which is exactly
    what happened the first time -- turns the comparison into one between two
    different algorithms without anything flagging it.
    """
    a = select([scratch_name])[0].build()
    b = select([torch_name])[0].build()
    shared = ["metric", "block", "census_win", "aggregation", "p1", "p2", "subpixel"]
    mismatched = {
        k: (getattr(a, k), getattr(b, k))
        for k in shared
        if hasattr(a, k) and hasattr(b, k) and getattr(a, k) != getattr(b, k)
    }
    assert not mismatched, (
        f"{scratch_name} and {torch_name} disagree on {mismatched} -- they are "
        "supposed to be the same algorithm on two backends"
    )
