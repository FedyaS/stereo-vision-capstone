import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench.datasets import list_middlebury_scenes, make_synthetic_pair  # noqa: E402


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
