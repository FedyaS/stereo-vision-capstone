"""Sweep machinery, and the split that keeps it honest."""
from __future__ import annotations

import numpy as np
import pytest

from bench.algorithms import select
from bench.algorithms.base import AlgorithmSpec
from bench.splits import (
    HELDOUT_SCENES,
    SCENE_GROUPS,
    SPLITS,
    TUNE_SCENES,
    split_scenes,
    split_spec,
)
from bench.sweep import (
    GRIDS,
    GRID_SETS,
    SweepCell,
    best_config,
    build_with,
    current_defaults,
    iter_configs,
    run_config,
    run_sweeps,
    sweep_algorithm,
)
from bench.sweep_report import summarise, write_all


# --------------------------------------------------------------------------
# the split
# --------------------------------------------------------------------------


def test_tune_and_heldout_do_not_overlap():
    assert not set(TUNE_SCENES) & set(HELDOUT_SCENES)


def test_split_covers_every_grouped_scene():
    grouped = {s for g in SCENE_GROUPS for s in g}
    assert grouped == set(TUNE_SCENES) | set(HELDOUT_SCENES)


def test_scene_variants_never_straddle_the_split():
    """Motorcycle and MotorcycleE are one scene; they must stay together."""
    tune = set(TUNE_SCENES)
    for group in SCENE_GROUPS:
        in_tune = [s in tune for s in group]
        assert all(in_tune) or not any(in_tune), (
            f"group {group} is split across tune and held-out -- the same "
            "geometry would appear on both sides"
        )


def test_both_sides_have_a_photometrically_hard_scene():
    """Otherwise the held-out set cannot test the cost-function finding."""
    hard = {"ArtL", "MotorcycleE", "PianoL", "PlaytableP"}
    assert hard & set(TUNE_SCENES)
    assert hard & set(HELDOUT_SCENES)


def test_split_scenes_filters_to_what_is_downloaded():
    assert split_scenes("tune", available=["Teddy", "Nonexistent"]) == ["Teddy"]
    assert split_scenes("heldout", available=[]) == []


def test_split_scenes_rejects_unknown_split():
    with pytest.raises(KeyError):
        split_scenes("nope", available=[])


def test_all_split_is_the_union():
    assert set(SPLITS["all"]) == set(TUNE_SCENES) | set(HELDOUT_SCENES)


def test_split_spec_formats_a_dataset_string(middlebury_scenes):
    if not middlebury_scenes:
        pytest.skip("Middlebury not downloaded")
    spec = split_spec("tune")
    assert spec.startswith("middlebury:")
    assert "Teddy" in spec


# --------------------------------------------------------------------------
# grid expansion
# --------------------------------------------------------------------------


def test_iter_configs_is_the_cartesian_product():
    cfgs = iter_configs({"a": [1, 2], "b": ["x", "y", "z"]})
    assert len(cfgs) == 6
    assert {"a": 1, "b": "x"} in cfgs
    assert len({tuple(sorted(c.items())) for c in cfgs}) == 6


def test_iter_configs_empty_grid_yields_one_default_config():
    assert iter_configs({}) == [{}]


def test_iter_configs_drops_p2_below_p1():
    cfgs = iter_configs({"p1": [10.0, 20.0], "p2": [10.0, 120.0]})
    assert all(c["p1"] < c["p2"] for c in cfgs)
    assert {"p1": 10.0, "p2": 10.0} not in cfgs


def test_iter_configs_drops_p2_mult_below_p1_mult():
    cfgs = iter_configs({"p1_mult": [8, 32], "p2_mult": [16, 32]})
    assert all(c["p1_mult"] < c["p2_mult"] for c in cfgs)


@pytest.mark.parametrize("grid_set", sorted(GRID_SETS))
def test_every_grid_names_real_parameters(grid_set):
    """A typo in a grid key would produce a flat, meaningless sweep."""
    for name, grid in GRID_SETS[grid_set].items():
        spec = select([name])[0]
        if not spec.availability()[0]:
            continue
        algo = spec.build()
        for key in grid:
            assert hasattr(algo, key), f"{name}: no parameter {key!r}"


# --------------------------------------------------------------------------
# truncation diagnostic
# --------------------------------------------------------------------------


def test_boundary_hits_flags_min_max_and_ignores_interior():
    """A winner on the edge of its range means the search was truncated."""
    from bench.sweep_report import boundary_hits

    grids = {"x": {"low": [1, 2, 3], "high": [1, 2, 3], "mid": [1, 2, 3]}}
    cells = {
        "x": [
            SweepCell(algorithm="x", params={"low": 1, "high": 3, "mid": 2},
                      bad2_all=5.0, wall_s=1.0),
            SweepCell(algorithm="x", params={"low": 3, "high": 1, "mid": 1},
                      bad2_all=9.0, wall_s=1.0),
        ]
    }
    hits = {(h["parameter"], h["edge"]) for h in boundary_hits(cells, grids=grids)}
    assert ("low", "min") in hits
    assert ("high", "max") in hits
    assert not [h for h in hits if h[0] == "mid"]


def test_boundary_hits_ignores_categorical_parameters():
    from bench.sweep_report import boundary_hits

    grids = {"x": {"mode": ["a", "b", "c"]}}
    cells = {"x": [SweepCell(algorithm="x", params={"mode": "a"},
                             bad2_all=5.0, wall_s=1.0)]}
    assert boundary_hits(cells, grids=grids) == []


def test_boundary_hits_reports_the_known_truncations_of_round_two():
    """The four winner-takes-all windows really did pin at the top.

    Regression guard for the diagnostic that drove the extension sweep: if
    this stops firing, `boundary_hits` has stopped seeing a real truncation.
    """
    from pathlib import Path

    from bench.sweep import REFINED_GRIDS, load_sweep
    from bench.sweep_report import boundary_hits

    csv = Path(__file__).resolve().parents[1] / "results" / "sweeps" / "20261105-031423" / "sweep.csv"
    if not csv.exists():
        pytest.skip("round-two sweep results not present")

    hits = boundary_hits(load_sweep(csv), grids=REFINED_GRIDS)
    pinned = {h["algorithm"] for h in hits if h["parameter"] == "block" and h["edge"] == "max"}
    assert {"scratch_census_wta", "scratch_sad_wta", "scratch_ssd_wta"} <= pinned


# --------------------------------------------------------------------------
# build_with
# --------------------------------------------------------------------------


def test_build_with_overrides_parameters():
    spec = select(["scratch_sad_wta"])[0]
    algo = build_with(spec, {"block": 15})
    assert algo.block == 15


def test_build_with_rejects_unknown_parameter():
    spec = select(["scratch_sad_wta"])[0]
    with pytest.raises(AttributeError, match="no parameter"):
        build_with(spec, {"not_a_real_param": 3})


def test_build_with_returns_independent_instances():
    spec = select(["scratch_sad_wta"])[0]
    a = build_with(spec, {"block": 5})
    b = build_with(spec, {"block": 11})
    assert a.block == 5 and b.block == 11


def test_overridden_parameter_actually_changes_the_result(tiny_pair):
    """Guards against a parameter that is stored but never read."""
    spec = select(["scratch_sad_wta"])[0]
    small = build_with(spec, {"block": 5}).compute(tiny_pair).normalized()[0]
    large = build_with(spec, {"block": 21}).compute(tiny_pair).normalized()[0]
    assert not np.allclose(
        np.nan_to_num(small, nan=-1), np.nan_to_num(large, nan=-1)
    )


def test_census_window_is_plumbed_through(tiny_pair):
    spec = select(["scratch_census_wta"])[0]
    a = build_with(spec, {"census_win": (5, 5)}).compute(tiny_pair).normalized()[0]
    b = build_with(spec, {"census_win": (7, 9)}).compute(tiny_pair).normalized()[0]
    assert not np.allclose(np.nan_to_num(a, nan=-1), np.nan_to_num(b, nan=-1))


def test_sgbm_penalty_multipliers_are_plumbed_through(tiny_pair):
    spec = select(["opencv_sgbm_3way"])[0]
    a = build_with(spec, {"p1_mult": 1, "p2_mult": 4}).compute(tiny_pair)
    b = build_with(spec, {"p1_mult": 32, "p2_mult": 512}).compute(tiny_pair)
    assert not np.allclose(
        np.nan_to_num(a.normalized()[0], nan=-1),
        np.nan_to_num(b.normalized()[0], nan=-1),
    )


# --------------------------------------------------------------------------
# selection rule
# --------------------------------------------------------------------------


def _cell(bad2, wall, **params):
    return SweepCell(
        algorithm="x", params=params or {"k": bad2}, bad2_all=bad2, wall_s=wall
    )


def test_best_config_prefers_accuracy():
    cells = [_cell(20.0, 0.1), _cell(10.0, 5.0)]
    assert best_config(cells, tolerance_pp=0.5).bad2_all == 10.0


def test_best_config_breaks_near_ties_by_speed():
    cells = [_cell(10.0, 5.0), _cell(10.3, 0.2), _cell(12.0, 0.01)]
    best = best_config(cells, tolerance_pp=0.5)
    assert best.bad2_all == 10.3  # within tolerance of 10.0, and 25x faster
    assert best.wall_s == 0.2


def test_best_config_ignores_timing_noise():
    """Near-identical wall times must not decide the winner.

    A block-size sweep barely changes the cost of a box filter, so the run
    order alone can make the less accurate setting look marginally faster.
    """
    cells = [_cell(21.21, 0.481), _cell(21.29, 0.478)]
    assert best_config(cells, tolerance_pp=0.5).bad2_all == 21.21


def test_best_config_still_prefers_a_real_speedup():
    cells = [_cell(21.21, 1.00), _cell(21.29, 0.20)]
    assert best_config(cells, tolerance_pp=0.5).bad2_all == 21.29


def test_best_config_rejects_a_trivial_speedup():
    """0.4 points is not worth 1.11x, which is inside measurement noise.

    This is the case the accuracy-anchored rule exists for. The previous
    rule -- near-ties, then everything within 5% of the fastest, then the
    most accurate of those -- excluded the better configuration from the
    tier entirely and returned the worse one.
    """
    cells = [_cell(10.0, 1.00), _cell(10.4, 0.90)]
    assert best_config(cells, tolerance_pp=0.5).bad2_all == 10.0


def test_best_config_cannot_repair_an_implausible_timing():
    """Documents a real limit: a wrong measurement still wins.

    These are `torch_zncc_wta`'s actual round-two rows. The 17 px window
    measured 1.39x faster than the 13 px one, which a larger box filter
    cannot be, and the rule takes the trade because 1.39x clears the bar.
    Guarding this is the job of `test_torch_defaults_mirror_scratch`, which
    copies the NumPy side's parameters over, not of the selection rule.
    """
    cells = [_cell(21.21, 1.488), _cell(21.29, 1.649), _cell(21.67, 1.069)]
    assert best_config(cells, tolerance_pp=0.5).bad2_all == 21.67


def test_best_config_tolerance_is_respected():
    cells = [_cell(10.0, 5.0), _cell(10.8, 0.1)]
    assert best_config(cells, tolerance_pp=0.5).bad2_all == 10.0
    assert best_config(cells, tolerance_pp=1.0).bad2_all == 10.8


def test_best_config_ignores_failed_and_nan_cells():
    good = _cell(15.0, 1.0)
    bad = SweepCell(algorithm="x", params={}, status="error", message="boom")
    nan = _cell(float("nan"), 0.001)
    assert best_config([bad, nan, good]) is good


def test_best_config_returns_none_when_everything_failed():
    bad = SweepCell(algorithm="x", params={}, status="error")
    assert best_config([bad]) is None


# --------------------------------------------------------------------------
# running
# --------------------------------------------------------------------------


class _Exploding:
    block = 1

    def compute(self, pair):
        raise RuntimeError("boom")


def test_run_config_records_failure_without_raising(tiny_pair):
    spec = AlgorithmSpec(
        name="x", family="test", description="d", build=_Exploding
    )
    cell = run_config(spec, {"block": 1}, [tiny_pair])
    assert cell.status == "error"
    assert "boom" in cell.message


def test_run_config_marks_oversized_volumes_skipped(tiny_pair):
    spec = select(["scratch_sad_wta"])[0]
    cell = run_config(spec, {"block": 5}, [tiny_pair], max_volume_gb=1e-9)
    assert cell.status == "skipped"


def test_run_config_averages_over_pairs(tiny_pair, sanity_pair):
    spec = select(["opencv_sgbm_3way"])[0]
    cell = run_config(spec, {"block_size": 5}, [tiny_pair, sanity_pair])
    assert cell.n_scenes == 2
    assert cell.status == "ok"
    assert 0 <= cell.bad2_all <= 100
    assert cell.wall_s > 0


def test_sweep_algorithm_runs_every_config(tiny_pair):
    spec = select(["scratch_sad_wta"])[0]
    cells = sweep_algorithm(
        spec, [tiny_pair], {"block": [5, 9]}, verbose=False
    )
    assert len(cells) == 2
    assert {c.params["block"] for c in cells} == {5, 9}
    assert all(c.status == "ok" for c in cells)


def test_end_to_end_sweep_writes_outputs(tiny_pair, tmp_path, monkeypatch):
    monkeypatch.setitem(GRIDS, "scratch_sad_wta", {"block": [5, 9]})
    monkeypatch.setitem(GRIDS, "opencv_sgbm_3way", {"block_size": [3, 5]})

    all_cells, out_dir = run_sweeps(
        ["scratch_sad_wta", "opencv_sgbm_3way"], [tiny_pair], tmp_path / "s",
        verbose=False,
    )
    assert set(all_cells) == {"scratch_sad_wta", "opencv_sgbm_3way"}
    assert (out_dir / "sweep.csv").exists()
    assert (out_dir / "sweep.json").exists()

    summary = write_all(all_cells, out_dir, tune_scenes=[tiny_pair.name])
    text = summary.read_text(encoding="utf-8")
    assert "Parameter sweep" in text
    assert "scratch_sad_wta" in text
    assert "held-out" in text
    assert (out_dir / "figures" / "scratch_sad_wta.png").exists()


def test_summarise_reports_grid_spread():
    cells = [_cell(10.0, 1.0, block=5), _cell(30.0, 1.0, block=9)]
    rows = summarise({"scratch_sad_wta": cells})
    assert rows[0]["grid_spread"] == pytest.approx(20.0)
    assert rows[0]["best_bad2"] == 10.0


# --------------------------------------------------------------------------
# grid ranges that were chosen by measurement
# --------------------------------------------------------------------------


def test_texture_threshold_is_inert_at_tutorial_values(teddy):
    """Documents why the grid searches 0..2000 rather than the usual 0..20.

    StereoBM compares `textureThreshold` against a SAD-based texture score,
    not an intensity, so the values most tutorials suggest sit in a dead
    range and a sweep over them silently measures nothing.
    """
    spec = select(["opencv_bm"])[0]

    def density(t):
        disp, valid = build_with(spec, {"texture_threshold": t}).compute(
            teddy
        ).normalized()
        return float(valid.mean())

    assert density(0) == pytest.approx(density(20), abs=1e-6)
    assert density(2000) < density(0) - 0.005


def test_prefilter_type_changes_the_result(teddy):
    spec = select(["opencv_bm"])[0]
    a = build_with(spec, {"prefilter_type": "xsobel"}).compute(teddy)
    b = build_with(spec, {"prefilter_type": "normalized"}).compute(teddy)
    assert not np.allclose(
        np.nan_to_num(a.normalized()[0], nan=-1),
        np.nan_to_num(b.normalized()[0], nan=-1),
    )


def test_unknown_prefilter_type_is_rejected(tiny_pair):
    spec = select(["opencv_bm"])[0]
    with pytest.raises(ValueError, match="prefilter_type"):
        build_with(spec, {"prefilter_type": "bogus"}).compute(tiny_pair)


# --------------------------------------------------------------------------
# round-tripping and the recommended diff
# --------------------------------------------------------------------------


def test_load_sweep_round_trips_through_csv(tiny_pair, tmp_path, monkeypatch):
    from bench.sweep import load_sweep

    monkeypatch.setitem(GRIDS, "scratch_census_wta",
                        {"block": [3, 5], "census_win": [(5, 5), (7, 9)]})
    all_cells, out_dir = run_sweeps(
        ["scratch_census_wta"], [tiny_pair], tmp_path / "s", verbose=False
    )
    back = load_sweep(out_dir / "sweep.csv")

    assert set(back) == set(all_cells)
    assert len(back["scratch_census_wta"]) == 4
    # The tuple-valued parameter must survive the CSV round trip as a tuple.
    wins = {c.params["census_win"] for c in back["scratch_census_wta"]}
    assert wins == {(5, 5), (7, 9)}
    for a, b in zip(all_cells["scratch_census_wta"], back["scratch_census_wta"]):
        assert a.params == b.params
        assert a.bad2_all == pytest.approx(b.bad2_all)


def _shipped_block(name: str = "scratch_sad_wta") -> int:
    """The live default, read rather than hardcoded.

    These tests are about the diffing logic, not about which block size the
    sweep happens to have chosen, so they must not pin a real default --
    retuning the algorithm would otherwise break them for no reason.
    """
    return current_defaults(select([name])[0], ["block"])["block"]


def test_recommended_changes_lists_only_real_edits():
    from bench.sweep_report import recommended_changes

    shipped = _shipped_block()
    other = shipped + 2
    cells = [_cell(30.0, 1.0, block=shipped), _cell(10.0, 1.0, block=other)]

    changes = recommended_changes({"scratch_sad_wta": cells})
    assert len(changes) == 1
    assert changes[0]["parameter"] == "block"
    assert changes[0]["old"] == shipped
    assert changes[0]["new"] == other
    assert changes[0]["gain_pp"] == pytest.approx(20.0)


def test_recommended_changes_empty_when_default_wins():
    from bench.sweep_report import recommended_changes

    shipped = _shipped_block()
    cells = [_cell(10.0, 1.0, block=shipped), _cell(30.0, 1.0, block=shipped + 2)]
    assert recommended_changes({"scratch_sad_wta": cells}) == []


#: torch entries copy their parameters from the matching scratch entry rather
#: than from their own sweep rows, so their defaults are not expected to appear
#: in any grid. `test_torch_defaults_mirror_scratch` is what guards them.
MIRRORED_FROM_SCRATCH = {
    "torch_sad_wta",
    "torch_zncc_wta",
    "torch_census_sgm4",
}


def _searched_values(name, key):
    """Every value of `name.key` that any grid set actually swept."""
    seen = []
    for grid_set in GRID_SETS.values():
        for v in (grid_set.get(name) or {}).get(key, []):
            if v not in seen:
                seen.append(v)
    return seen


def test_every_shipped_default_was_actually_searched():
    """Each default must come from a sweep, or be a declared mirror.

    The grid sets are a historical record -- `default` is round one,
    `refined` round two, `wta-extended` the pass that found where the
    window turns over -- so a default can legitimately sit outside any
    single one of them. What must hold is that it appears in at least one
    grid that was run, otherwise "these defaults came from a sweep" is not
    true of that parameter.
    """
    missing = []
    for name in sorted({n for gs in GRID_SETS.values() for n in gs}):
        if name in MIRRORED_FROM_SCRATCH:
            continue
        spec = select([name])[0]
        if not spec.availability()[0]:
            continue
        keys = {k for gs in GRID_SETS.values() for k in (gs.get(name) or {})}
        for key, value in current_defaults(spec, keys).items():
            searched = _searched_values(name, key)
            if searched and value not in searched:
                missing.append(f"{name}.{key}={value!r} not in {searched!r}")
    assert not missing, "defaults never swept:\n  " + "\n  ".join(missing)


@pytest.mark.parametrize("grid_set", sorted(GRID_SETS))
def test_grids_are_internally_consistent(grid_set):
    """A grid must name real parameters and offer a choice worth making."""
    for name, grid in GRID_SETS[grid_set].items():
        spec = select([name])[0]
        if not spec.availability()[0]:
            continue
        algo = spec.build()
        for key, values in grid.items():
            assert hasattr(algo, key), f"{grid_set}/{name}: no parameter {key!r}"
            assert len(values) == len(
                [v for i, v in enumerate(values) if v not in values[:i]]
            ), f"{grid_set}/{name}.{key} repeats a value"


# --------------------------------------------------------------------------
# before/after reporting survives applying the tuned values
# --------------------------------------------------------------------------


def test_sweep_records_the_defaults_it_ran_against(tiny_pair, tmp_path, monkeypatch):
    """Without this, applying the results makes the summary claim no change."""
    import json

    from bench.sweep import load_baseline

    monkeypatch.setitem(GRIDS, "scratch_sad_wta", {"block": [5, 9]})
    _, out_dir = run_sweeps(
        ["scratch_sad_wta"], [tiny_pair], tmp_path / "s", verbose=False
    )
    meta = json.loads((out_dir / "sweep.json").read_text())
    assert "baseline_defaults" in meta

    shipped = current_defaults(select(["scratch_sad_wta"])[0], ["block"])
    assert load_baseline(out_dir / "sweep.json")["scratch_sad_wta"] == shipped


def test_summarise_uses_the_supplied_baseline_not_live_code():
    from bench.sweep_report import recommended_changes

    cells = [_cell(30.0, 1.0, block=3), _cell(10.0, 1.0, block=7)]
    baseline = {"scratch_sad_wta": {"block": 3}}

    rows = summarise({"scratch_sad_wta": cells}, baseline=baseline)
    assert rows[0]["default_params"] == {"block": 3}
    assert rows[0]["default_bad2"] == pytest.approx(30.0)
    assert rows[0]["best_bad2"] == pytest.approx(10.0)

    change = recommended_changes({"scratch_sad_wta": cells}, baseline=baseline)[0]
    assert (change["old"], change["new"]) == (3, 7)
    assert change["gain_pp"] == pytest.approx(20.0)


def test_baseline_round_trips_tuple_parameters(tmp_path):
    import json

    from bench.sweep import load_baseline

    path = tmp_path / "sweep.json"
    path.write_text(json.dumps(
        {"baseline_defaults": {"scratch_census_wta": {"census_win": "(7, 9)"}}}
    ))
    assert load_baseline(path)["scratch_census_wta"]["census_win"] == (7, 9)


def test_baseline_restores_tuples_written_as_json_arrays(tmp_path):
    """json.dumps turns a tuple into an array, so it must come back a tuple.

    Without this the recorded default never compares equal to the matching
    sweep cell's params, the "pre-sweep" column goes blank and the gain is
    reported as nan -- which is exactly what `scratch_census_wta` did.
    """
    import json

    from bench.sweep import load_baseline

    path = tmp_path / "sweep.json"
    path.write_text(json.dumps(
        {"baseline_defaults": {"scratch_census_wta": {"census_win": (7, 7)}}}
    ))
    assert load_baseline(path)["scratch_census_wta"]["census_win"] == (7, 7)


def test_baseline_tuple_survives_a_real_sweep(tiny_pair, tmp_path, monkeypatch):
    """End to end: a tuple-valued default must produce a real gain number."""
    from bench.sweep import load_baseline
    from bench.sweep_report import summarise

    monkeypatch.setitem(
        GRIDS, "scratch_census_wta",
        {"census_win": [(5, 5), (7, 7), (7, 9)]},
    )
    all_cells, out_dir = run_sweeps(
        ["scratch_census_wta"], [tiny_pair], tmp_path / "s", verbose=False
    )
    baseline = load_baseline(out_dir / "sweep.json")
    row = summarise(all_cells, baseline=baseline)[0]
    assert not np.isnan(row["default_bad2"]), "shipped default was not matched"
