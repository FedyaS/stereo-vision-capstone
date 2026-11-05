"""Run one algorithm on one scene, keep the result, and re-score it cheaply.

The playground's loop is "tweak a parameter, look again", so matching is the
expensive part and scoring is not. Results are therefore cached as raw float
arrays: changing the evaluation region, the bad-pixel threshold or the
hole-filling then costs a few milliseconds instead of re-running the matcher.
That separation is also the honest one -- it makes it obvious in the UI which
numbers come from the algorithm and which come from how you chose to score it.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from bench.algorithms import select
from bench.metrics import BAD_THRESHOLDS, evaluate
from bench.profiling import ResourceUsage, measure
from bench.types import StereoPair

from . import catalog
from .params import coerce_params

_LIMIT = 48
_results: "OrderedDict[str, StoredResult]" = OrderedDict()
_lock = threading.Lock()


@dataclass
class StoredResult:
    """A completed single-pair run, kept so it can be re-scored and re-drawn."""

    id: str
    scene_id: str
    algorithm: str
    params: dict[str, Any]
    disparity: np.ndarray
    valid: np.ndarray
    usage: ResourceUsage
    extra: dict[str, Any]
    created: float = field(default_factory=time.time)

    @property
    def pair(self) -> StereoPair:
        return catalog.load(self.scene_id, downscale=self.params.get("_downscale", 1.0))


def _store(result: StoredResult) -> None:
    with _lock:
        _results[result.id] = result
        while len(_results) > _LIMIT:
            _results.popitem(last=False)


def get(result_id: str) -> StoredResult:
    with _lock:
        if result_id not in _results:
            raise KeyError(
                "that result has expired -- the server keeps the last "
                f"{_LIMIT} runs. Run it again."
            )
        _results.move_to_end(result_id)
        return _results[result_id]


def cache_key(scene_id: str, algorithm: str, params: dict, downscale: float) -> str:
    blob = json.dumps(
        {"s": scene_id, "a": algorithm, "p": params, "d": downscale},
        sort_keys=True, default=str,
    )
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def find_cached(key: str) -> StoredResult | None:
    with _lock:
        for r in _results.values():
            if r.extra.get("_key") == key:
                return r
    return None


def run(
    scene_id: str,
    algorithm: str,
    params: dict[str, Any] | None = None,
    *,
    downscale: float = 1.0,
    repeats: int = 1,
    reuse: bool = True,
) -> StoredResult:
    """Match one pair and store the result."""
    specs = select([algorithm])
    if not specs:
        raise KeyError(f"unknown algorithm {algorithm!r}")
    spec = specs[0]
    ok, why = spec.availability()
    if not ok:
        raise RuntimeError(f"{algorithm} is unavailable: {why}")

    clean = coerce_params(spec, params or {})
    key = cache_key(scene_id, algorithm, clean, downscale)
    if reuse:
        hit = find_cached(key)
        if hit is not None:
            return hit

    pair = catalog.load(scene_id, downscale=downscale)
    algo = spec.build()
    for name, value in clean.items():
        setattr(algo, name, value)

    h, w = pair.shape
    result, usage = measure(
        lambda: algo.compute(pair),
        repeats=max(1, repeats),
        warmup=1,
        n_pixels=h * w,
        ndisp=pair.ndisp,
        track_gpu=spec.family in ("torch", "learned"),
    )
    disp, valid = result.normalized()

    stored = StoredResult(
        id=uuid.uuid4().hex[:12],
        scene_id=scene_id,
        algorithm=algorithm,
        params={**clean, "_downscale": downscale},
        disparity=disp,
        valid=valid,
        usage=usage,
        extra={**result.extra, "_key": key, "family": spec.family},
    )
    _store(stored)
    return stored


def score(
    stored: StoredResult,
    *,
    eval_region: str = "nonocc",
    thresholds: tuple[float, ...] = BAD_THRESHOLDS,
) -> dict[str, Any] | None:
    """Score a stored result. Cheap enough to redo on every UI change."""
    pair = catalog.load(stored.scene_id, downscale=stored.params.get("_downscale", 1.0))
    if not pair.has_gt:
        return None
    mask = None if eval_region == "all" else pair.occ_mask
    m = evaluate(
        stored.disparity, stored.valid, pair.gt,
        eval_mask=mask, thresholds=tuple(thresholds),
    )
    flat = m.flat()
    # Key on "%g" of the threshold ("0.5", "1", "2", "4"), which is what
    # JavaScript's String(n) produces for the same numbers. Using str() here
    # would emit "2.0" and the UI would look up "2" and find nothing -- a
    # mismatch that shows up as a blank metric rather than an error.
    return {
        "bad": {_key(t): m.bad_all.get(_fmt(t), float("nan")) for t in thresholds},
        "bad_valid": {
            _key(t): m.bad_valid.get(_fmt(t), float("nan")) for t in thresholds
        },
        "epe_valid": m.epe_valid,
        "epe_filled": m.epe_filled,
        "rmse_valid": m.rmse_valid,
        "rmse_filled": m.rmse_filled,
        "bias_valid": m.bias_valid,
        "a90": m.a90,
        "a95": m.a95,
        "a99": m.a99,
        "density": m.density,
        "n_eval_px": m.n_eval_px,
        "eval_region": eval_region,
        "flat": {k: _json_safe(v) for k, v in flat.items()},
    }


def _fmt(t: float) -> str:
    """The key `bench.metrics` uses internally (``0p5``, ``2``)."""
    return f"{t:g}".replace(".", "p")


def _key(t: float) -> str:
    """The key the browser will look up; matches JavaScript ``String(n)``."""
    return f"{t:g}"


def _json_safe(v: Any) -> Any:
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if not np.isfinite(f) else f
    if isinstance(v, (np.integer, int)):
        return int(v)
    return v


def summarise(stored: StoredResult, scoring: dict | None) -> dict[str, Any]:
    """The JSON payload the UI gets back from a run."""
    pair = catalog.load(stored.scene_id, downscale=stored.params.get("_downscale", 1.0))
    u = stored.usage
    disp = stored.disparity
    finite = np.isfinite(disp)
    return {
        "result_id": stored.id,
        "scene_id": stored.scene_id,
        "algorithm": stored.algorithm,
        "family": stored.extra.get("family"),
        "params": {k: v for k, v in stored.params.items() if not k.startswith("_")},
        "downscale": stored.params.get("_downscale", 1.0),
        "width": pair.shape[1],
        "height": pair.shape[0],
        "ndisp": pair.ndisp,
        "has_gt": pair.has_gt,
        "timing": {
            "wall_s": _json_safe(u.wall_s),
            "wall_s_min": _json_safe(u.wall_s_min),
            "cpu_s": _json_safe(u.cpu_s),
            "parallelism": _json_safe(u.parallelism),
            "peak_rss_mb": _json_safe(u.peak_rss_mb),
            "rss_delta_mb": _json_safe(u.rss_delta_mb),
            "mdps": _json_safe(u.mdps),
            "mpix_per_s": _json_safe(u.mpix_per_s),
            "repeats": u.repeats,
        },
        "disparity_range": [
            _json_safe(np.nanmin(disp)) if finite.any() else None,
            _json_safe(np.nanmax(disp)) if finite.any() else None,
        ],
        "metrics": scoring,
        "extra": {
            k: _json_safe(v)
            for k, v in stored.extra.items()
            if not k.startswith("_") and k != "traceback"
        },
    }
