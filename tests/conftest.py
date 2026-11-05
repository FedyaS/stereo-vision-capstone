import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench.datasets import list_middlebury_scenes, make_synthetic_pair  # noqa: E402


@pytest.fixture(autouse=True)
def _results_dir_is_a_tmp_dir(tmp_path, monkeypatch):
    """Keep test output out of the real ``results/``.

    ``app.server.jobs`` writes a batch run to ``bench.paths.RESULTS_DIR``, so
    ``test_benchmark_job_runs_to_completion`` used to leave a junk one-cell
    run folder in the repository's own ``results/`` on every test run -- which
    then showed up in the app's Runs tab and as ``bench explain``'s "latest
    run". Redirect it per test.

    Reads of the real ``results/`` (the app listing stored runs) are
    deliberately left alone: those tests skip when the folder is empty, and
    pointing them at a tmp dir would make them skip always.
    """
    try:
        from app.server import jobs
    except ImportError:
        return  # app extras not installed; test_app.py skips itself too
    monkeypatch.setattr(jobs, "RESULTS_ROOT", tmp_path / "results")


@pytest.fixture(scope="session")
def tiny_pair():
    """A small synthetic pair -- fast enough to run every algorithm on."""
    return make_synthetic_pair(seed=7, size=(96, 128), ndisp=24)


@pytest.fixture(scope="session")
def sanity_pair():
    """A pair for "is this method working at all?" checks.

    Bigger and with a wider disparity range than ``tiny_pair`` for two
    reasons. A narrow range makes "guess the mean disparity everywhere" an
    unreasonably strong baseline -- on a 16 px range, a constant guess is
    within 4 px of most of the image. And the fixed-input ONNX models resample
    the pair to 640 or 1280 px wide, so judging them on a 128 px-wide input
    measures the resampling, not the network.
    """
    return make_synthetic_pair(seed=3, size=(256, 320), ndisp=48)


@pytest.fixture(scope="session")
def middlebury_scenes():
    return list_middlebury_scenes()


@pytest.fixture(scope="session")
def teddy(middlebury_scenes):
    if "Teddy" not in middlebury_scenes:
        pytest.skip("Middlebury not downloaded (run scripts/get_dataset.py)")
    from bench.datasets import load_middlebury_scene

    return load_middlebury_scene("Teddy")
