"""Describe an algorithm's tunable parameters for the UI.

Every adapter is a dataclass, so its fields *are* its parameter list. The
useful ranges are already written down too -- the sweep grids in
``bench.sweep`` say which values are worth trying -- so the UI controls are
derived from those rather than invented here. A parameter the sweep knows
about becomes a slider or a dropdown over exactly the values that were
searched; anything else falls back to a free input with a sane bound.
"""
from __future__ import annotations

import dataclasses
from typing import Any

from bench.algorithms import AlgorithmSpec
from bench.sweep import GRIDS, REFINED_GRIDS

# Fields that exist on an adapter but are not meaningful to expose: internal
# handles, opaque library constants chosen by the registry entry, and
# collections that have no sensible single control.
_HIDDEN = {"mode", "providers", "base", "max_volume_gb", "filename", "device",
           "dtype", "outlier_factor", "intra_op_threads"}

# Fallback bounds for numeric fields the sweep never touched, so a slider
# still has somewhere sensible to go.
_FALLBACK_RANGE: dict[str, tuple[float, float]] = {
    "block": (1, 51),
    "block_size": (3, 51),
    "p1": (1, 200),
    "p2": (10, 600),
    "p1_mult": (1, 64),
    "p2_mult": (4, 512),
    "jump_penalty": (1, 400),
    "uniqueness_ratio": (0, 50),
    "texture_threshold": (0, 4000),
    "speckle_window_size": (0, 400),
    "speckle_range": (0, 16),
    "disp12_max_diff": (-1, 32),
    "prefilter_cap": (1, 63),
    "lambda_": (10, 32000),
    "sigma_color": (0.1, 5.0),
    "correlation_threshold": (0.0, 1.0),
    "corr_win_size": (3, 21),
    "disparity_gradient": (0, 8),
}

# Parameters that must stay odd for a centred window to exist.
_ODD_ONLY = {"block", "block_size", "corr_win_size"}

# Sets the adapters accept but no sweep grid enumerates, because the sweep
# tunes an algorithm rather than turning it into a different one. In the UI
# they are the most interesting controls there are -- swapping the matching
# cost or the aggregation on a live scene is the fastest way to see what each
# stage actually contributes.
_EXPLICIT_CHOICES: dict[str, list] = {
    "metric": ["sad", "ssd", "zncc", "census"],
    "aggregation": ["none", "sgm4", "sgm8", "dp"],
    "prefilter_type": ["xsobel", "normalized"],
    "normalize": ["raw", "raft", "unit"],
    "census_win": [[5, 5], [7, 7], [9, 7], [7, 9], [5, 9]],
}

#: One-line notes shown beside a control, so the UI teaches rather than just
#: exposes. Keyed by parameter name.
HINTS: dict[str, str] = {
    "metric": "How two patches are compared. census keeps only the ordering "
              "of neighbouring pixels, so it survives exposure differences "
              "that destroy sad/ssd.",
    "aggregation": "What enforces smoothness. none = winner-takes-all per "
                   "pixel; dp = exact per-scanline optimum (watch for "
                   "streaking); sgm4/sgm8 = semi-global over 4 or 8 paths.",
    "block": "Window side length. With no aggregation, bigger windows "
             "suppress ambiguous matches but blur depth edges.",
    "block_size": "Matching window side length. Must be odd.",
    "p1": "Penalty for a one-step disparity change. Small = surfaces may "
          "stair-step.",
    "p2": "Penalty for a larger jump. Large = depth edges get smoothed away.",
    "p1_mult": "P1 as a multiple of channels x blockSize^2 (OpenCV suggests 8).",
    "p2_mult": "P2 as a multiple of channels x blockSize^2 (OpenCV suggests 32).",
    "jump_penalty": "Potts penalty for changing disparity along a scanline.",
    "uniqueness_ratio": "How much better the best match must be than the "
                        "runner-up. Raising it rejects more pixels, which "
                        "counts as error under bad2_all.",
    "texture_threshold": "Minimum texture to attempt a match. Compared "
                         "against a SAD score, so it does nothing below ~200.",
    "lr_check": "Discard pixels where the left and right disparity maps "
                "disagree. This is what detects occlusions.",
    "subpixel": "Fit a parabola through the winning cost and its neighbours. "
                "Usually the cheapest accuracy available.",
    "census_win": "Census neighbourhood (height x width). At most 65 pixels, "
                  "so the code fits in a uint64.",
    "lambda_": "WLS smoothness strength. High values drag disparity across "
               "depth edges.",
    "sigma_color": "How strongly the WLS filter respects image edges.",
    "correlation_threshold": "How good a match must be to keep growing the "
                             "dense region.",
    "normalize": "Input scaling. These ONNX exports fold RAFT-Stereo's own "
                 "normalisation into the graph, so they want raw 0-255.",
}


def _grid_values(algorithm: str, field: str) -> list[Any]:
    """Union of the values any sweep grid tried for this parameter."""
    seen: list[Any] = []
    for grids in (GRIDS, REFINED_GRIDS):
        for value in (grids.get(algorithm) or {}).get(field, []):
            if value not in seen:
                seen.append(value)
    return seen


def _choices_for(algorithm: str, field: str, current: Any) -> list[Any]:
    """Every value the UI should offer for a discrete parameter."""
    out: list[Any] = []
    for value in _EXPLICIT_CHOICES.get(field, []) + _grid_values(algorithm, field):
        value = list(value) if isinstance(value, tuple) else value
        if value not in out:
            out.append(value)
    current = list(current) if isinstance(current, tuple) else current
    if current not in out:
        out.insert(0, current)
    return out


def _kind(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, tuple):
        return "choice"
    return "text"


def describe_params(spec: AlgorithmSpec) -> list[dict[str, Any]]:
    """Parameter descriptors for one algorithm, in display order."""
    try:
        algo = spec.build()
    except Exception:
        return []
    if not dataclasses.is_dataclass(algo):
        return []

    out: list[dict[str, Any]] = []
    for field in dataclasses.fields(algo):
        name = field.name
        if name.startswith("_") or name in _HIDDEN:
            continue
        value = getattr(algo, name)
        if isinstance(value, (dict, list)):
            continue
        if isinstance(value, tuple) and not all(
            isinstance(v, (int, float)) for v in value
        ):
            continue

        swept = _grid_values(spec.name, name)
        kind = _kind(value)
        item: dict[str, Any] = {
            "name": name,
            "kind": kind,
            "value": list(value) if isinstance(value, tuple) else value,
            "tuned": bool(swept),
        }

        if name in _EXPLICIT_CHOICES or kind == "choice" or kind == "text":
            item["kind"] = "choice"
            item["choices"] = _choices_for(spec.name, name, value)
        elif kind in ("int", "float"):
            numeric = [v for v in swept if isinstance(v, (int, float))]
            lo, hi = _FALLBACK_RANGE.get(name, (0.0, max(1.0, float(value) * 4)))
            if numeric:
                # Let the UI reach a little past what the sweep searched --
                # several of those searches hit their boundary, so the
                # interesting values are often just outside.
                lo = min(lo, min(numeric))
                hi = max(hi, max(numeric))
            item["min"] = int(lo) if kind == "int" else float(lo)
            item["max"] = int(hi) if kind == "int" else float(hi)
            item["step"] = (2 if name in _ODD_ONLY else 1) if kind == "int" else 0.05
            item["odd_only"] = name in _ODD_ONLY
            if numeric:
                item["swept"] = sorted(numeric)
        elif kind == "text":
            # A string field with no grid: offer the current value only.
            item["kind"] = "choice"
            item["choices"] = [value]

        if name in HINTS:
            item["hint"] = HINTS[name]
        out.append(item)

    for item in out:
        if item["kind"] == "choice" and not item.get("choices"):
            item["choices"] = [item["value"]]
    return out


def coerce_params(
    spec: AlgorithmSpec, raw: dict[str, Any]
) -> dict[str, Any]:
    """Validate and convert UI values into real constructor arguments.

    The UI sends JSON, so tuples arrive as lists and ints as floats. Anything
    the adapter does not define is rejected rather than ignored -- silently
    dropping a parameter would make the UI lie about what it just ran.
    """
    try:
        algo = spec.build()
    except Exception as exc:  # pragma: no cover - availability checked upstream
        raise ValueError(f"cannot build {spec.name}: {exc}") from exc

    fields = {f.name: f for f in dataclasses.fields(algo)}
    out: dict[str, Any] = {}
    for key, value in (raw or {}).items():
        if key not in fields or key.startswith("_") or key in _HIDDEN:
            raise ValueError(f"{spec.name} has no tunable parameter {key!r}")
        current = getattr(algo, key)
        if isinstance(current, bool):
            out[key] = bool(value)
        elif isinstance(current, int):
            n = int(round(float(value)))
            if key in _ODD_ONLY and n % 2 == 0:
                n += 1  # a centred window needs an odd side
            out[key] = n
        elif isinstance(current, float):
            out[key] = float(value)
        elif isinstance(current, tuple):
            out[key] = tuple(int(v) for v in value)
        else:
            out[key] = value
    return out
