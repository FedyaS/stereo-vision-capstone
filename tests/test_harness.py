"""Profiling, runner and reporting."""
from __future__ import annotations

import time

import numpy as np
import pytest

from bench.algorithms import select
from bench.algorithms.base import AlgorithmSpec
from bench.profiling import ResourceUsage, machine_info, measure, set_thread_limit
from bench.report import leaderboard, pareto_front, to_dataframe, write_all
from bench.runner import CellResult, RunConfig, run_benchmark, run_cell
from bench.types import DisparityResult, StereoPair


# --------------------------------------------------------------------------
# profiling
# --------------------------------------------------------------------------


def test_measure_reports_plausible_wall_time():
    def work():
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 0.05:
            pass

    _, usage = measure(work, repeats=2, warmup=0)
    assert 0.04 < usage.wall_s < 1.0
    assert usage.repeats == 2
    assert usage.wall_s_min <= usage.wall_s
    assert usage.peak_rss_mb > 0
    assert usage.peak_threads >= 1


def test_measure_returns_the_callables_result():
    result, _ = measure(lambda: 42, repeats=1, warmup=1)
    assert result == 42


def test_measure_computes_throughput_when_given_sizes():
    _, usage = measure(lambda: None, repeats=1, n_pixels=1000, ndisp=10)
    assert usage.mpix_per_s > 0
    assert usage.mdps > 0
    assert np.isfinite(usage.parallelism)


def test_measure_without_sizes_leaves_throughput_nan():
    _, usage = measure(lambda: None, repeats=1)
    assert np.isnan(usage.mdps)
    assert np.isnan(usage.mpix_per_s)


def test_warmup_runs_are_not_timed():
    calls = []
    measure(lambda: calls.append(1), repeats=3, warmup=2)
    assert len(calls) == 5  # 2 warmup + 3 timed


def test_set_thread_limit_is_applied_and_reversible():
    import cv2

    original = cv2.getNumThreads()
    try:
        set_thread_limit(1)
        assert cv2.getNumThreads() == 1
        set_thread_limit(None)  # no-op, must not reset to default
        assert cv2.getNumThreads() == 1
    finally:
        cv2.setNumThreads(original)


def test_machine_info_has_the_essentials():
    info = machine_info()
    for key in ("platform", "python", "cpu_count_logical", "ram_total_gb"):
        assert key in info
    assert "opencv" in info


# --------------------------------------------------------------------------
# runner error handling
# --------------------------------------------------------------------------


class _Exploding:
    def compute(self, pair):
        raise RuntimeError("boom")


class _WrongShape:
    def compute(self, pair):
        return DisparityResult(np.zeros((3, 3), np.float32))


def _spec(name, builder, check=None):
    return AlgorithmSpec(
        name=name, family="test", description="test fixture",
        build=builder, check=check,
    )


def test_failing_algorithm_is_recorded_not_raised(tiny_pair):
    cell, disp = run_cell(_spec("x", _Exploding), tiny_pair, RunConfig(warmup=0))
    assert cell.status == "error"
    assert "boom" in cell.message
    assert "traceback" in cell.extra
    assert disp is None


def test_wrong_output_shape_is_an_error(tiny_pair):
    cell, _ = run_cell(_spec("x", _WrongShape), tiny_pair, RunConfig(warmup=0))
    assert cell.status == "error"
    assert "expected" in cell.message


def test_unavailable_algorithm_is_skipped_with_a_reason(tiny_pair):
    spec = _spec("x", _Exploding, check=lambda: (False, "no weights here"))
    cell, disp = run_cell(spec, tiny_pair, RunConfig(warmup=0))
    assert cell.status == "skipped"
    assert cell.message == "no weights here"
    assert disp is None


def test_oversized_cost_volume_is_skipped_not_fatal(tiny_pair):
    spec = select(["scratch_sad_wta"])[0]
    cell, _ = run_cell(
        spec, tiny_pair, RunConfig(warmup=0, max_volume_gb=1e-9)
    )
    assert cell.status == "skipped"
    assert "GB" in cell.message


def test_thread_limit_reaches_onnx_sessions(tiny_pair):
    """`--threads N` must reach ONNX Runtime, which ignores the env vars."""
    spec = next(
        (s for s in [*select(["learned"])] if s.availability()[0]), None
    )
    if spec is None:
        pytest.skip("no learned model available")

    captured = {}
    real_build = spec.build

    def spy():
        algo = real_build()
        captured["algo"] = algo
        return algo

    run_cell(
        _spec(spec.name, spy), tiny_pair, RunConfig(warmup=0, threads=2)
    )
    assert captured["algo"].intra_op_threads == 2


def test_pair_without_ground_truth_still_gets_timings(tiny_pair):
    bare = StereoPair(
        name="no-gt", left=tiny_pair.left, right=tiny_pair.right,
        ndisp=tiny_pair.ndisp,
    )
    cell, disp = run_cell(
        select(["opencv_sgbm_3way"])[0], bare, RunConfig(warmup=0)
    )
    assert cell.status == "ok"
    assert cell.accuracy is None
    assert cell.usage.wall_s > 0
    assert disp is not None


# --------------------------------------------------------------------------
# end to end
# --------------------------------------------------------------------------


def test_end_to_end_run_writes_everything(tiny_pair, tmp_path):
    specs = select(["opencv_sgbm_3way", "opencv_bm", "scratch_zncc_wta"])
    config = RunConfig(
        repeats=1, warmup=0, out_dir=tmp_path, save_maps=True, save_figures=True
    )
    results, out_dir = run_benchmark([tiny_pair], specs, config, verbose=False)

    assert len(results) == 3
    assert all(c.status == "ok" for c in results)
    assert (out_dir / "results.csv").exists()
    assert (out_dir / "run.json").exists()
    assert list((out_dir / "maps").glob("*.pfm"))

    summary = write_all(results, out_dir)
    assert summary.exists()
    text = summary.read_text(encoding="utf-8")
    assert "Leaderboard" in text
    assert "opencv_sgbm_3way" in text
    assert "Pareto" in text

    figures = out_dir / "figures"
    assert (figures / "accuracy_vs_time.png").exists()
    assert (figures / "accuracy_by_scene.png").exists()
    assert list(figures.glob("qualitative_*.png"))


def test_leaderboard_is_sorted_by_accuracy(tiny_pair, tmp_path):
    specs = select(["opencv_sgbm_3way", "opencv_bm", "scratch_census_wta"])
    results, _ = run_benchmark(
        [tiny_pair], specs,
        RunConfig(repeats=1, warmup=0, out_dir=tmp_path, save_figures=False),
        verbose=False,
    )
    lb = leaderboard(to_dataframe(results))
    assert list(lb["bad2_all"]) == sorted(lb["bad2_all"])


def test_pareto_front_excludes_dominated_rows():
    import pandas as pd

    df = pd.DataFrame({
        "algorithm": ["fast_bad", "slow_good", "dominated"],
        "wall_s": [0.1, 1.0, 0.5],
        "bad2_all": [20.0, 5.0, 25.0],
    })
    front = pareto_front(df)
    assert front == ["fast_bad", "slow_good"]
    assert "dominated" not in front


def test_report_handles_an_all_failed_run(tmp_path):
    results = [
        CellResult(algorithm="a", family="test", scene="s", status="error",
                   message="nope")
    ]
    (tmp_path).mkdir(exist_ok=True)
    summary = write_all(results, tmp_path)
    text = summary.read_text(encoding="utf-8")
    assert "Skipped and failed" in text
