"""Where everything lives on disk -- the single source of truth.

Six modules used to each recompute the repository root with
``Path(__file__).resolve().parents[N]``, which meant the directory layout was
encoded in six places and the ``N`` changed whenever a file moved. Import from
here instead::

    from bench.paths import RESULTS_DIR

The constants are plain module attributes on purpose, so a test can redirect
one with ``monkeypatch.setattr`` and keep its output out of the real
directories. Read them at call time rather than binding them at import time
(``RESULTS_DIR / stamp`` inside the function, not ``OUT = RESULTS_DIR / x``
at module level) or a monkeypatch will not be seen.

See ``docs/repository.md`` for what belongs in each of these and which are
committed.
"""
from __future__ import annotations

from pathlib import Path

#: The repository checkout. ``bench/paths.py`` -> ``bench/`` -> the root.
REPO_ROOT = Path(__file__).resolve().parents[1]

# -- inputs: fetched or supplied, never committed -------------------------
DATA_DIR = REPO_ROOT / "data"
MIDDLEBURY_DIR = DATA_DIR / "middlebury" / "MiddEval3"
CUSTOM_DATA_DIR = DATA_DIR / "custom"
DOWNLOADS_DIR = DATA_DIR / "downloads"
#: Training sets written by ``bench synth generate``. Not committed: they are
#: reproducible from the ``rules.json`` each one carries.
SYNTH_DATA_DIR = DATA_DIR / "synth"
MODELS_DIR = REPO_ROOT / "models"

# -- outputs: generated, ignored except results/keep/ ---------------------
RESULTS_DIR = REPO_ROOT / "results"
SWEEPS_DIR = RESULTS_DIR / "sweeps"
#: Un-ignored, so anything copied here is committed. See scripts/keep_result.py.
KEPT_RESULTS_DIR = RESULTS_DIR / "keep"

# -- committed: code, docs and the frontend bundle ------------------------
WEB_DIST_DIR = REPO_ROOT / "app" / "web" / "dist"
DOCS_DIR = REPO_ROOT / "docs"
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
#: The full-resolution pair beside the original hand-written notebook; the
#: worked example for the "any folder with im0/im1" dataset.
PRACTICE_DIR = NOTEBOOKS_DIR / "practice"


def describe(path: Path) -> str:
    """``path`` relative to the repository root when it is inside it.

    Used in output a human reads, where ``results/20261105-035941`` is clearer
    than an absolute path -- but a path outside the checkout (a tmp dir under
    test, say) must still print rather than raise.
    """
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


__all__ = [
    "REPO_ROOT",
    "DATA_DIR",
    "MIDDLEBURY_DIR",
    "CUSTOM_DATA_DIR",
    "DOWNLOADS_DIR",
    "SYNTH_DATA_DIR",
    "MODELS_DIR",
    "RESULTS_DIR",
    "SWEEPS_DIR",
    "KEPT_RESULTS_DIR",
    "WEB_DIST_DIR",
    "DOCS_DIR",
    "NOTEBOOKS_DIR",
    "PRACTICE_DIR",
    "describe",
]
