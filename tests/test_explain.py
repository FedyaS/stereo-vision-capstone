"""The report generator.

Two things are worth pinning here and they are quite different.

The *derivations* are claims about the data -- "the cost function moved the
error more than the aggregation did" -- and a claim assembled from the wrong
rows is worse than no claim at all, because it reads as confidently as a right
one. Those are tested against synthetic runs whose answer is known by
construction.

The *rendering* is tested for the failures that are silent in a browser: a
self-contained page that turns out to reference a file, an id used twice, a
number formatted so the page's own JavaScript cannot find it.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bench.explain import (
    COSTS,
    METRIC_COLUMNS,
    Findings,
    Recipe,
    decompose,
    derive,
    load_run,
    write_report,
)
from bench.explain_html import CHART_METRICS, TABLE_COLUMNS, render
from bench.report import PRIMARY


# --------------------------------------------------------------------------
# a synthetic run, built so every derived claim has a known right answer
# --------------------------------------------------------------------------

SCENES = ["middlebury/Motorcycle", "middlebury/MotorcycleE", "middlebury/Teddy"]

#: (algorithm, family, bad2 on the base scenes, extra bad2 on MotorcycleE).
#: Census is deliberately both the most accurate and the only cost that does
#: not degrade under the exposure change, so the photometric finding and the
#: cost finding have unambiguous answers.
ALGOS = [
    ("scratch_census_sgm8", "scratch", 10.0, 0.5, 4.0, 4.1),
    ("scratch_census_wta", "scratch", 16.0, 0.5, 1.0, 1.0),
    ("scratch_sad_wta", "scratch", 34.0, 30.0, 1.0, 1.0),
    ("scratch_ssd_wta", "scratch", 36.0, 28.0, 1.1, 1.1),
    ("torch_census_sgm8", "torch", 10.1, 0.5, 4.2, 48.0),
    ("opencv_sgbm", "opencv", 22.0, 6.0, 0.1, 0.15),
    ("opencv_bm", "opencv", 33.0, 6.5, 0.02, 0.04),
]


def _make_run(tmp_path: Path, algos=ALGOS, scenes=SCENES) -> Path:
    rows = []
    for name, family, base, penalty, wall, cpu in algos:
        for scene in scenes:
            bad2 = base + (penalty if scene.endswith("E") else 0.0)
            rows.append({
                "algorithm": name, "family": family, "scene": scene,
                "status": "ok", "width": 741, "height": 497, "ndisp": 70,
                "wall_s": wall, "wall_s_min": wall, "wall_s_std": 0.0,
                "cpu_s": cpu, "parallelism": cpu / wall,
                "peak_rss_mb": 500.0, "rss_delta_mb": 100.0,
                "peak_threads": 8, "gpu_peak_mb": 0.0, "repeats": 1,
                "mdps": 10.0, "mpix_per_s": 1.0,
                "bad0p5_all": bad2 * 2, "bad1_all": bad2 * 1.4,
                "bad2_all": bad2, "bad4_all": bad2 * 0.7,
                "epe_filled": bad2 / 10, "rmse_filled": bad2 / 5,
                "bad0p5_valid": bad2 * 2, "bad1_valid": bad2 * 1.4,
                "bad2_valid": bad2, "bad4_valid": bad2 * 0.7,
                "epe_valid": bad2 / 10, "rmse_valid": bad2 / 5,
                "bias_valid": 0.1, "a90": 1.0, "a95": 2.0, "a99": 5.0,
                "density": 1.0, "n_eval_px": 300000, "message": "", "extra": "",
            })
    run_dir = tmp_path / "20260101-000000"
    run_dir.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(run_dir / "results.csv", index=False)
    (run_dir / "run.json").write_text(json.dumps({
        "created_utc": "2026-01-01T00:00:00+00:00",
        "elapsed_s": 1.0,
        "config": {"repeats": 1, "warmup": 1, "threads": None,
                   "eval_region": "nonocc", "save_maps": False,
                   "save_figures": False, "max_volume_gb": 4.0,
                   "out_dir": "results"},
        "machine": {"cpu_brand": "Test CPU", "cpu_count_physical": 8,
                    "ram_total_gb": 16.0, "opencv": "4.13.0"},
        "n_cells": len(rows), "n_ok": len(rows),
    }))
    (run_dir / "figures").mkdir()
    return run_dir


@pytest.fixture
def findings(tmp_path) -> Findings:
    return derive(load_run(_make_run(tmp_path)))


# --------------------------------------------------------------------------
# decomposing names
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name,family,cost,agg,extra", [
    ("scratch_sad_wta", "scratch", "sad", "wta", ""),
    ("scratch_census_sgm8_lrcheck", "scratch", "census", "sgm8", "lrcheck"),
    ("torch_zncc_wta", "torch", "zncc", "wta", ""),
    ("opencv_sgbm_wls", "opencv", None, "sgbm", "wls"),
    ("opencv_bm", "opencv", None, "bm", ""),
])
def test_decompose(name, family, cost, agg, extra):
    r = decompose(name, family)
    assert (r.cost, r.agg, r.extra) == (cost, agg, extra)


def test_tags_win_over_the_name_for_the_cost():
    """The registry knows things the name does not spell out."""
    r = decompose("scratch_mystery_sgm8", "scratch", tags=("census", "sgm"))
    assert r.cost == "census"
    assert r.photometric_invariant


@pytest.mark.parametrize("name,family", [
    ("opencv_quasi_dense", "opencv"),
    ("learned_raft_stereo_r4", "learned"),
])
def test_a_name_that_is_not_a_recipe_produces_no_recipe(name, family):
    """Splitting on underscores would yield 'quasi + dense' -- confident
    nonsense. Better to say nothing than to invent a pipeline."""
    r = decompose(name, family)
    assert not r.understood
    assert r.label() == ""


def test_a_name_that_is_a_recipe_produces_one():
    r = decompose("scratch_census_sgm8_lrcheck", "scratch")
    assert r.understood
    assert r.label() == "census + semi-global, 8 paths + left-right consistency check"


def test_every_cost_is_classified_as_invariant_or_not():
    for cost in COSTS:
        rec = Recipe(name="x", family="scratch", cost=cost)
        assert isinstance(rec.photometric_invariant, bool)
    assert Recipe("x", "scratch", cost="census").photometric_invariant
    assert Recipe("x", "scratch", cost="zncc").photometric_invariant
    assert not Recipe("x", "scratch", cost="sad").photometric_invariant
    assert not Recipe("x", "scratch", cost="ssd").photometric_invariant


# --------------------------------------------------------------------------
# the controlled comparisons
# --------------------------------------------------------------------------


def test_contrasts_only_vary_one_component(findings):
    """A contrast whose members differ in more than one place is not a
    controlled comparison, and any conclusion drawn from it is unfounded."""
    for c in findings.contrasts:
        recs = [r["recipe"] for r in c.rows]
        fixed = {
            "cost": lambda r: (r.family, r.agg, r.extra),
            "aggregation": lambda r: (r.family, r.cost, r.extra),
            "backend": lambda r: (r.cost, r.agg, r.extra),
        }[c.axis]
        assert len({fixed(r) for r in recs}) == 1, c.axis
        varied = {"cost": "cost", "aggregation": "agg", "backend": "family"}[c.axis]
        assert len({getattr(r, varied) for r in recs}) == len(recs)


def test_the_cost_contrast_finds_the_planted_answer(findings):
    cost = [c for c in findings.contrasts if c.axis == "cost"]
    assert cost, "the scratch *_wta family differs only in its cost function"
    c = cost[0]
    assert c.best["varied"] == "census"
    assert c.worst["varied"] in ("sad", "ssd")
    # census 16.17 vs ssd ~45.3 averaged over the three scenes
    assert c.spread > 20


def test_the_backend_contrast_pairs_the_two_implementations(findings):
    back = [c for c in findings.contrasts if c.axis == "backend"]
    assert back
    names = {r["algorithm"] for r in back[0].rows}
    assert names == {"scratch_census_sgm8", "torch_census_sgm8"}
    assert back[0].spread < 1.0, "the same maths should give the same answer"


def test_contrasts_are_ordered_by_how_much_they_explain(findings):
    for axis in ("cost", "aggregation", "backend"):
        cs = [c.spread for c in findings.contrasts if c.axis == axis]
        assert cs == sorted(cs, reverse=True)


def test_a_single_member_is_not_a_contrast(tmp_path):
    """One algorithm with no comparable sibling must not become a comparison
    of one, which would report a spread of zero and read as a finding."""
    f = derive(load_run(_make_run(
        tmp_path, algos=[("scratch_sad_wta", "scratch", 30.0, 10.0, 1.0, 1.0)]
    )))
    assert f.contrasts == []


# --------------------------------------------------------------------------
# the photometric experiment
# --------------------------------------------------------------------------


def test_photometric_separates_invariant_from_intensity_costs(findings):
    by_name = {p["algorithm"]: p for p in findings.photometric}
    # Planted: census gains 0.5, SAD gains 30 between Motorcycle and -E.
    assert by_name["scratch_census_sgm8"]["mean_delta"] == pytest.approx(0.5)
    assert by_name["scratch_sad_wta"]["mean_delta"] == pytest.approx(30.0)
    assert by_name["scratch_census_sgm8"]["invariant"]
    assert not by_name["scratch_sad_wta"]["invariant"]


def test_photometric_is_sorted_best_first(findings):
    deltas = [p["mean_delta"] for p in findings.photometric]
    assert deltas == sorted(deltas)


def test_photometric_needs_a_variant_pair(tmp_path):
    """Without two captures of one scene there is no experiment to report."""
    f = derive(load_run(_make_run(tmp_path, scenes=["middlebury/Teddy"])))
    assert f.photometric == []


# --------------------------------------------------------------------------
# the prose
# --------------------------------------------------------------------------


def test_the_abstract_names_the_actual_winner(findings):
    top = findings.lb.iloc[0]["algorithm"]
    assert top == "scratch_census_sgm8"
    assert top in findings.abstract


def test_notes_are_present_and_classified(findings):
    assert findings.notes
    assert {n.kind for n in findings.notes} <= {"headline", "caution", "detail"}
    assert any(n.kind == "headline" for n in findings.notes)


def test_the_headline_names_the_axis_that_actually_moved_more(findings):
    """The claim is comparative, so it must follow the measurement rather than
    a sentence written once and left there."""
    cost = max(c.spread for c in findings.contrasts if c.axis == "cost")
    agg = [c.spread for c in findings.contrasts if c.axis == "aggregation"]
    if not agg:
        pytest.skip("no aggregation contrast in this fixture")
    note = next(n for n in findings.notes if "matters more" in n.title)
    expected = "cost function" if cost > max(agg) else "aggregation"
    assert expected in note.title


def test_a_broken_method_is_flagged_rather_than_dropped(tmp_path):
    f = derive(load_run(_make_run(tmp_path, algos=ALGOS + [
        ("learned_broken", "learned", 95.0, 0.0, 1.0, 1.0),
    ])))
    assert "learned_broken" in set(f.lb["algorithm"]), "still in the table"
    assert any("did not converge" in n.title for n in f.notes)


def test_sparse_methods_get_the_density_caveat(tmp_path):
    """A method answering half the image scores well on 'answered' metrics;
    the report has to say so, because the two orderings genuinely differ."""
    run_dir = _make_run(tmp_path)
    df = pd.read_csv(run_dir / "results.csv")
    sparse = df["algorithm"] == "opencv_bm"
    df.loc[sparse, "density"] = 0.55
    df.loc[sparse, "bad2_valid"] = 8.0        # excellent where it answered
    df.loc[sparse, "bad2_all"] = 48.0         # poor once blanks count
    df.to_csv(run_dir / "results.csv", index=False)

    f = derive(load_run(run_dir))
    note = next(n for n in f.notes if "Sparse" in n.title)
    assert "opencv_bm" in note.body
    assert "55%" in note.body


def test_scene_stats_are_sorted_hardest_first(findings):
    means = [s["mean"] for s in findings.scene_stats]
    assert means == sorted(means, reverse=True)
    assert findings.scene_stats[0]["short"] == "MotorcycleE"


def test_scene_stats_label_the_split(findings):
    by_short = {s["short"]: s for s in findings.scene_stats}
    assert by_short["Motorcycle"]["split"] == "tune"
    assert by_short["Teddy"]["split"] == "tune"


# --------------------------------------------------------------------------
# degenerate runs must not raise
# --------------------------------------------------------------------------


def test_a_run_with_no_successful_cells(tmp_path):
    run_dir = tmp_path / "empty"
    run_dir.mkdir()
    pd.DataFrame([{
        "algorithm": "a", "family": "opencv", "scene": "s", "status": "error",
        "message": "boom", "width": 1, "height": 1, "ndisp": 1,
    }]).to_csv(run_dir / "results.csv", index=False)
    f = derive(load_run(run_dir))
    assert f.notes == [] and f.contrasts == []
    html = render(f)
    assert "<html" in html


def test_a_run_without_ground_truth(tmp_path):
    """Timing-only runs have no accuracy columns at all."""
    run_dir = tmp_path / "notruth"
    run_dir.mkdir()
    pd.DataFrame([{
        "algorithm": "opencv_bm", "family": "opencv", "scene": "custom/rig",
        "status": "ok", "width": 640, "height": 480, "ndisp": 64,
        "wall_s": 0.1, "cpu_s": 0.2, "parallelism": 2.0, "peak_rss_mb": 10.0,
        "rss_delta_mb": 1.0, "mdps": 1.0, "mpix_per_s": 1.0, "message": "",
    }]).to_csv(run_dir / "results.csv", index=False)
    f = derive(load_run(run_dir))
    assert "<html" in render(f)


def test_missing_results_csv_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not look like a benchmark run"):
        load_run(tmp_path)


# --------------------------------------------------------------------------
# the rendered page
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def page(tmp_path_factory) -> str:
    run = _make_run(tmp_path_factory.mktemp("render"))
    return render(derive(load_run(run)))


def test_the_page_is_self_contained(page):
    """One file, no network. A CDN link would make the report useless offline
    and make a printed copy silently lose its styling."""
    for pattern in (r'src="https?://', r'href="https?://[^"]*\.css',
                    r'@import\s+url\(', r'<link[^>]+stylesheet'):
        assert not re.search(pattern, page), pattern
    # Images, if any, are data URIs rather than paths into the run folder.
    for src in re.findall(r'<img[^>]+src="([^"]+)"', page):
        assert src.startswith("data:"), src


def test_every_element_id_is_unique(page):
    """A duplicate id makes getElementById return the wrong node, which fails
    at runtime in the browser and nowhere else."""
    ids = re.findall(r'\sid="([^"]+)"', page)
    dupes = {i for i in ids if ids.count(i) > 1}
    assert not dupes, f"duplicate ids: {dupes}"


def test_the_table_of_contents_resolves(page):
    ids = set(re.findall(r'\sid="([^"]+)"', page))
    for href in re.findall(r'<a href="#([^"]+)"', page):
        assert href in ids, href


def test_every_table_column_has_a_glossary_entry():
    """A column the report shows but never explains is the whole failure this
    document exists to avoid."""
    for key in TABLE_COLUMNS:
        assert key in METRIC_COLUMNS, key
        assert METRIC_COLUMNS[key]["long"], key
    for key in CHART_METRICS:
        assert key in METRIC_COLUMNS, key


def test_the_primary_metric_is_offered_in_the_charts():
    assert PRIMARY in CHART_METRICS
    assert METRIC_COLUMNS[PRIMARY]["better"] == "low"


def test_the_payload_the_javascript_reads_is_valid_json(page):
    blob = re.search(r"window\.__REPORT__ = (\{.*?\});</script>", page, re.S)
    assert blob, "the charts are drawn from this and nothing else"
    data = json.loads(blob.group(1).replace("\\u003c", "<"))
    assert data["algorithms"] and data["scenes"] and data["matrix"]
    assert data["primary"] in data["metrics"]
    # Every metric the selector offers must actually have a matrix behind it.
    for key in data["chartMetrics"]:
        assert key in data["metrics"], key


def test_the_payload_has_no_nan(page):
    """NaN is not JSON. json.dumps emits it anyway and JSON.parse rejects it,
    which would leave the page with no charts at all."""
    blob = re.search(r"window\.__REPORT__ = (\{.*?\});</script>", page, re.S).group(1)
    assert "NaN" not in blob and "Infinity" not in blob


def test_the_page_states_its_limits(page):
    """A report that only reports the good news is the thing to avoid."""
    for phrase in ("does not show", "rectified", "under-estimate"):
        assert phrase in page, phrase


def test_write_report_lands_in_the_run_folder(tmp_path):
    run = _make_run(tmp_path)
    out = write_report(run)
    assert out == run / "report.html"
    assert out.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_write_report_honours_an_explicit_path(tmp_path):
    run = _make_run(tmp_path)
    target = tmp_path / "elsewhere" / "r.html"
    assert write_report(run, target) == target
    assert target.exists()
