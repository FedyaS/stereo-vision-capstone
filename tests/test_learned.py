"""The learned models' preprocessing, pinned by measurement.

The input normalisation an ONNX export expects is not discoverable from the
file, and getting it wrong does not crash -- it quietly triples the error. So
the choice is pinned here against ground truth rather than left as a comment.
"""
from __future__ import annotations

import pytest

from bench.algorithms.learned import MODEL_DIR, MODEL_SOURCES, OnnxStereoMatcher
from bench.metrics import evaluate

pytest.importorskip("onnxruntime")

R4_FILE = MODEL_SOURCES["raft_stereo_384x1280_r4"][0]


def _matcher(**kw):
    if not (MODEL_DIR / R4_FILE).exists():
        pytest.skip("weights absent (run scripts/get_models.py)")
    return OnnxStereoMatcher(filename=R4_FILE, **kw)


def _bad2(matcher, pair):
    disp, valid = matcher.compute(pair).normalized()
    return evaluate(disp, valid, pair.gt, eval_mask=pair.occ_mask).bad_all["2"]


def test_raw_normalisation_is_the_right_choice(teddy):
    """`raw` must beat the alternatives, or the default is wrong."""
    raw = _bad2(_matcher(normalize="raw"), teddy)
    scaled = _bad2(_matcher(normalize="raft"), teddy)
    unit = _bad2(_matcher(normalize="unit"), teddy)
    assert raw < scaled
    assert raw < unit


def test_default_matches_the_validated_choice():
    assert _matcher().normalize == "raw"


def test_raft_r4_is_accurate_on_teddy(teddy):
    """Guards against a silent regression in resizing or sign handling."""
    assert _bad2(_matcher(), teddy) < 10.0


def test_disparity_is_rescaled_for_the_model_input_size(teddy):
    """Disparity is a horizontal length, so it must scale with the width."""
    result = _matcher().compute(teddy)
    assert result.extra["model_input"] == "1280x384"
    expected = teddy.shape[1] / 1280
    assert result.extra["resize_factor"] == pytest.approx(expected, rel=1e-3)


def test_session_is_reused_across_calls(tiny_pair):
    m = _matcher()
    m.compute(tiny_pair)
    first = m._session
    m.compute(tiny_pair)
    assert m._session is first


def test_unknown_normalisation_raises(tiny_pair):
    with pytest.raises(ValueError):
        _matcher(normalize="bogus").compute(tiny_pair)


def test_missing_weights_report_unavailable_not_crash():
    from bench.algorithms import select

    spec = select(["learned_raft_stereo_r4"])[0]
    ok, why = spec.availability()
    assert isinstance(ok, bool)
    if not ok:
        assert "get_models" in why or "onnxruntime" in why
