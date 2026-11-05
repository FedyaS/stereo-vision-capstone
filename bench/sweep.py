"""Parameter sweeps over the tune split.

Every algorithm here ships with one fixed configuration, and the honest
question about any benchmark result is whether a method lost because it is
worse or because its defaults were worse. This module answers that by running
a grid per algorithm and recording the whole response surface, not just the
winner.

Two rules keep the result meaningful.

**The sweep only ever sees the tune scenes.** :mod:`bench.splits` holds a
scene-group split; the held-out scenes are never loaded here. Tuning and
reporting on the same scenes measures the search, not the parameters.

**Accuracy first, then speed.** :func:`best_config` takes the lowest
``bad2_all``, then among configurations within ``tolerance_pp`` percentage
points of it takes the fastest. Without the tie-break a grid will happily pick
a configuration that costs 4x the runtime for a hundredth of a point, which
makes a bad default.

Run it with ``python -m bench sweep``; results land in
``results/sweeps/<timestamp>/``.
"""
from __future__ import annotations

import itertools
import json
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .algorithms import AlgorithmSpec, select
from .algorithms.scratch import VolumeTooLarge
from .metrics import evaluate
from .types import StereoPair

# --------------------------------------------------------------------------
# the grids
# --------------------------------------------------------------------------
# Sizing note: cost is (configs x tune scenes x per-pair runtime). The
# expensive aggregating matchers get deliberately coarser grids than the
# OpenCV ones, which run in milliseconds. Values bracket the current default
# on both sides, so a sweep can move a parameter either way.

_SGBM_GRID: dict[str, list] = {
    "block_size": [3, 5, 7, 9],
    "p1_mult": [4, 8, 16],
    "p2_mult": [16, 32, 64],
    "uniqueness_ratio": [5, 10, 15],
}
_WLS_LAMBDA = [1000.0, 4000.0, 8000.0, 16000.0]
_WLS_SIGMA = [0.8, 1.2, 1.5, 2.0]
# The two WLS entries cannot share a block-size list: StereoSGBM accepts a
# 3x3 block, StereoBM asserts on anything below 5.
_WLS_GRID_SGBM: dict[str, list] = {
    "lambda_": _WLS_LAMBDA, "sigma_color": _WLS_SIGMA, "block_size": [3, 5, 7],
}
_WLS_GRID_BM: dict[str, list] = {
    "lambda_": _WLS_LAMBDA, "sigma_color": _WLS_SIGMA, "block_size": [5, 9, 15],
}
_CENSUS_WINDOWS = [(5, 5), (7, 7), (7, 9), (9, 7)]

GRIDS: dict[str, dict[str, list]] = {
    # --- opencv -----------------------------------------------------------
    "opencv_bm": {
        "block_size": [7, 9, 11, 15, 21],
        "uniqueness_ratio": [5, 10, 15],
        # Measured, not guessed: this knob does nothing below ~200 on
        # Middlebury, so the conventional 0/10/20 grid searches three
        # identical configurations. See test_texture_threshold_is_inert_at_
        # tutorial_values.
        "texture_threshold": [0, 10, 500, 1000, 2000],
        "prefilter_type": ["xsobel", "normalized"],
    },
    "opencv_sgbm": dict(_SGBM_GRID),
    "opencv_sgbm_hh": dict(_SGBM_GRID),
    "opencv_sgbm_hh4": dict(_SGBM_GRID),
    "opencv_sgbm_3way": dict(_SGBM_GRID),
    "opencv_sgbm_wls": dict(_WLS_GRID_SGBM),
    "opencv_bm_wls": dict(_WLS_GRID_BM),
    "opencv_quasi_dense": {
        "corr_win_size": [3, 5, 7],
        "correlation_threshold": [0.4, 0.6, 0.8],
    },
    # --- scratch ----------------------------------------------------------
    "scratch_sad_wta": {"block": [5, 7, 9, 11, 13, 15]},
    "scratch_ssd_wta": {"block": [5, 7, 9, 11, 13, 15]},
    "scratch_zncc_wta": {"block": [5, 7, 9, 11, 13, 15]},
    "scratch_census_wta": {
        "block": [3, 5, 7, 9, 11],
        "census_win": _CENSUS_WINDOWS,
    },
    "scratch_sad_wta_lrcheck": {
        "block": [7, 9, 11],
        "uniqueness_ratio": [0.05, 0.1, 0.2],
    },
    "scratch_census_dp": {
        "block": [1, 3, 5],
        "jump_penalty": [30.0, 45.0, 60.0, 90.0, 120.0],
    },
    "scratch_census_sgm4": {
        "block": [1, 3, 5],
        "p1": [5.0, 10.0, 15.0, 20.0],
        "p2": [60.0, 120.0, 240.0],
    },
    # The 8-path variants cost ~2x the 4-path one per run, so they get a
    # narrower grid: P1/P2 enter the same recursion either way, so the 4-path
    # sweep above is where the penalty pair is actually searched and these
    # only need to confirm the choice survives the extra paths.
    "scratch_census_sgm8": {
        "block": [1, 3],
        "p1": [10.0, 20.0],
        "p2": [120.0, 240.0],
    },
    "scratch_census_sgm8_lrcheck": {
        "block": [3],
        "p1": [10.0, 20.0],
        "p2": [60.0, 120.0, 240.0],
    },
    "scratch_sad_sgm8": {
        "block": [3, 5],
        "p1": [10.0],
        "p2": [120.0, 240.0],
    },
    # --- torch ------------------------------------------------------------
    # These compute the same thing as their `scratch_` counterparts to within
    # float32 rounding (`test_numpy_and_torch_backends_agree` enforces it), so
    # re-searching the same grid would spend an hour re-deriving a known
    # answer. The grids here are confirmatory: wide enough to catch a backend
    # that does *not* track NumPy, narrow enough to be cheap.
    "torch_sad_wta": {"block": [5, 9, 13, 15]},
    "torch_zncc_wta": {"block": [5, 9, 13]},
    "torch_census_sgm4": {"block": [1, 3, 5]},
    # `learned_*` has nothing to tune: the ONNX exports have frozen weights,
    # a frozen input size and a frozen iteration count.
}

# --------------------------------------------------------------------------
# round two: refined grids
# --------------------------------------------------------------------------
# Built from the first sweep's own diagnostics (`boundary_hits`), which found
# 33 parameters whose winning value sat on the edge of its range -- meaning
# the grid ran out before the response did. Each entry below either *extends*
# past such an edge or *bisects* around an interior optimum, centred on the
# round-one winner, so this is a coarse-to-fine second pass rather than a
# bigger version of the same search.
#
# Run with `python -m bench sweep --grid refined`.
REFINED_GRIDS: dict[str, dict[str, list]] = {
    # uniqueness_ratio and p1_mult both bottomed out at the low end, so both
    # extend downward; p2_mult topped out, so it extends upward.
    "opencv_bm": {
        "block_size": [11, 13, 15, 17, 19],
        "uniqueness_ratio": [0, 2, 5, 8],
        "texture_threshold": [0, 10],
        "prefilter_type": ["xsobel", "normalized"],
    },
    "opencv_sgbm": {
        "block_size": [5, 7, 9],
        "uniqueness_ratio": [0, 2, 5, 8],
        "p1_mult": [1, 2, 4, 8],
        "p2_mult": [32, 64, 128, 256],
    },
    "opencv_sgbm_hh": {
        "block_size": [5, 7, 9],
        "uniqueness_ratio": [0, 2, 5, 8],
        "p1_mult": [1, 2, 4, 8],
        "p2_mult": [32, 64, 128, 256],
    },
    "opencv_sgbm_hh4": {
        "block_size": [5, 7, 9],
        "uniqueness_ratio": [0, 2, 5, 8],
        "p1_mult": [1, 2, 4, 8],
        "p2_mult": [32, 64, 128, 256],
    },
    "opencv_sgbm_3way": {
        "block_size": [5, 7, 9],
        "uniqueness_ratio": [0, 2, 5, 8],
        "p1_mult": [1, 2, 4, 8],
        "p2_mult": [32, 64, 128, 256],
    },
    # lambda hit the bottom of 1000..16000; the WLS prior was already too
    # strong at the ximgproc default, so this reaches an order lower.
    "opencv_sgbm_wls": {
        "lambda_": [100.0, 250.0, 500.0, 1000.0, 2000.0],
        "sigma_color": [1.2, 1.5, 1.8],
        "block_size": [3, 5, 7],
    },
    "opencv_bm_wls": {
        "lambda_": [1000.0, 2000.0, 4000.0, 8000.0],
        "sigma_color": [1.2, 1.5, 1.8],
        "block_size": [5, 7, 9],
    },
    "opencv_quasi_dense": {
        "corr_win_size": [3, 5, 7],
        "correlation_threshold": [0.2, 0.3, 0.4, 0.5],
    },
    # Every winner-takes-all block size pinned at the top of its range.
    "scratch_sad_wta": {"block": [15, 17, 19, 21, 25]},
    "scratch_ssd_wta": {"block": [15, 17, 19, 21, 25]},
    "scratch_zncc_wta": {"block": [13, 15, 17, 19]},
    "scratch_census_wta": {
        "block": [11, 13, 15, 17],
        # 9x9 would need 80 bits; the transform caps at 64 neighbours.
        "census_win": [(7, 7), (9, 7), (7, 9), (5, 9)],
    },
    "scratch_sad_wta_lrcheck": {
        "block": [11, 13, 15, 17],
        "uniqueness_ratio": [0.0, 0.02, 0.05],
    },
    # The aggregating methods pinned at the *top* of their block range too,
    # which was only [1, 3, 5] -- small by construction, so worth extending.
    "scratch_census_dp": {
        "block": [5, 7, 9],
        "jump_penalty": [60.0, 90.0, 120.0, 150.0],
    },
    "scratch_census_sgm4": {
        "block": [5, 7, 9],
        "p1": [20.0, 30.0, 45.0],
        "p2": [90.0, 120.0, 180.0],
    },
    "scratch_census_sgm8": {
        "block": [3, 5, 7],
        "p1": [20.0, 30.0],
        "p2": [60.0, 90.0, 120.0],
    },
    "scratch_census_sgm8_lrcheck": {
        "block": [3, 5],
        "p1": [20.0, 30.0],
        "p2": [90.0, 120.0],
    },
    "scratch_sad_sgm8": {
        "block": [5, 7],
        "p1": [10.0, 20.0],
        "p2": [60.0, 90.0, 120.0],
    },
    # Confirmatory only -- these must end up wherever `scratch_` does.
    "torch_sad_wta": {"block": [15, 19, 21]},
    "torch_zncc_wta": {"block": [13, 15, 17]},
    "torch_census_sgm4": {"block": [5, 7]},
}

#: Selectable grid sets, by the name `--grid` takes.
GRID_SETS: dict[str, dict[str, dict[str, list]]] = {
    "default": GRIDS,
    "refined": REFINED_GRIDS,
}


#: Grids whose p1/p2 pairs only make sense with p1 < p2.
_ORDERED_P1_P2 = True


@dataclass
class SweepCell:
    """One configuration of one algorithm, averaged over the tune scenes."""

    algorithm: str
    params: dict[str, Any]
    bad2_all: float = float("nan")
    bad1_all: float = float("nan")
    epe_valid: float = float("nan")
    density: float = float("nan")
    wall_s: float = float("nan")
    n_scenes: int = 0
    status: str = "ok"
    message: str = ""

    def row(self) -> dict[str, Any]:
        out: dict[str, Any] = {"algorithm": self.algorithm}
        for k, v in self.params.items():
            out[f"p_{k}"] = str(v) if isinstance(v, tuple) else v
        out.update({
            "bad2_all": self.bad2_all,
            "bad1_all": self.bad1_all,
            "epe_valid": self.epe_valid,
            "density": self.density,
            "wall_s": self.wall_s,
            "n_scenes": self.n_scenes,
            "status": self.status,
            "message": self.message,
        })
        return out


def build_with(spec: AlgorithmSpec, params: dict[str, Any]):
    """Build an algorithm, then override the named attributes.

    Every adapter is a dataclass, so overriding is a ``setattr`` -- but an
    unknown name would silently do nothing and the sweep would report a grid
    of identical results, so it raises instead.
    """
    algo = spec.build()
    for key, value in params.items():
        if not hasattr(algo, key):
            raise AttributeError(
                f"{spec.name} ({type(algo).__name__}) has no parameter {key!r}; "
                f"available: {sorted(getattr(algo, '__dataclass_fields__', {}))}"
            )
        setattr(algo, key, value)
    return algo


def iter_configs(grid: dict[str, list]) -> list[dict[str, Any]]:
    """Cartesian product of a grid, dropping nonsensical combinations.

    The only rule applied is ``p1 < p2``: in the SGM recursion P1 penalises a
    one-step disparity change and P2 a larger jump, so P2 below P1 inverts the
    model's meaning and is never worth a run.
    """
    if not grid:
        return [{}]
    keys = list(grid)
    configs = []
    for values in itertools.product(*(grid[k] for k in keys)):
        cfg = dict(zip(keys, values))
        if _ORDERED_P1_P2 and "p1" in cfg and "p2" in cfg and cfg["p1"] >= cfg["p2"]:
            continue
        if (
            _ORDERED_P1_P2
            and "p1_mult" in cfg
            and "p2_mult" in cfg
            and cfg["p1_mult"] >= cfg["p2_mult"]
        ):
            continue
        configs.append(cfg)
    return configs


def run_config(
    spec: AlgorithmSpec,
    params: dict[str, Any],
    pairs: Sequence[StereoPair],
    *,
    eval_region: str = "nonocc",
    max_volume_gb: float = 4.0,
) -> SweepCell:
    """Run one configuration over every tune pair and average the results."""
    cell = SweepCell(algorithm=spec.name, params=dict(params))
    bad2, bad1, epe, dens, walls = [], [], [], [], []

    for pair in pairs:
        try:
            algo = build_with(spec, params)
            if hasattr(algo, "max_volume_gb"):
                algo.max_volume_gb = max_volume_gb
            t0 = time.perf_counter()
            result = algo.compute(pair)
            walls.append(time.perf_counter() - t0)

            disp, valid = result.normalized()
            mask = None if eval_region == "all" else pair.occ_mask
            m = evaluate(disp, valid, pair.gt, eval_mask=mask)
            bad2.append(m.bad_all.get("2", np.nan))
            bad1.append(m.bad_all.get("1", np.nan))
            epe.append(m.epe_valid)
            dens.append(m.density)
        except (VolumeTooLarge, MemoryError) as exc:
            cell.status, cell.message = "skipped", str(exc)[:200]
            return cell
        except Exception as exc:
            cell.status = "error"
            cell.message = f"{type(exc).__name__}: {exc}"[:200]
            cell.params["_traceback"] = traceback.format_exc(limit=3)
            return cell

    cell.n_scenes = len(walls)
    if cell.n_scenes:
        cell.bad2_all = float(np.nanmean(bad2))
        cell.bad1_all = float(np.nanmean(bad1))
        cell.epe_valid = float(np.nanmean(epe))
        cell.density = float(np.nanmean(dens))
        cell.wall_s = float(np.mean(walls))
    return cell


def sweep_algorithm(
    spec: AlgorithmSpec,
    pairs: Sequence[StereoPair],
    grid: dict[str, list] | None = None,
    *,
    grids: dict[str, dict[str, list]] | None = None,
    verbose: bool = True,
    **kw: Any,
) -> list[SweepCell]:
    """Run the whole grid for one algorithm."""
    if grid is None:
        grid = (grids if grids is not None else GRIDS).get(spec.name, {})
    configs = iter_configs(grid)
    cells: list[SweepCell] = []
    for i, cfg in enumerate(configs, 1):
        cell = run_config(spec, cfg, pairs, **kw)
        cells.append(cell)
        if verbose:
            desc = ", ".join(f"{k}={v}" for k, v in cfg.items()) or "(defaults)"
            if cell.status == "ok":
                print(
                    f"  [{i:3d}/{len(configs)}] {desc:52s} "
                    f"bad2={cell.bad2_all:6.2f}%  {cell.wall_s:6.2f}s",
                    flush=True,
                )
            else:
                print(
                    f"  [{i:3d}/{len(configs)}] {desc:52s} "
                    f"{cell.status.upper()}: {cell.message[:50]}",
                    flush=True,
                )
    return cells


def best_config(
    cells: Sequence[SweepCell],
    tolerance_pp: float = 0.5,
    speed_margin: float = 1.05,
) -> SweepCell | None:
    """Lowest ``bad2_all``, trading accuracy for speed only when speed differs.

    Three steps:

    1. Take every configuration within ``tolerance_pp`` percentage points of
       the best ``bad2_all``. Half a point is well inside the scene-to-scene
       spread, so it is worth trading for a real speedup.
    2. Of those, keep the ones within ``speed_margin`` of the fastest.
    3. Among *those*, take the most accurate.

    Step 2 is what stops the rule from chasing timing noise. Sorting the
    near-ties by raw wall time picks whichever run happened to land a few
    milliseconds quicker, which for a block-size sweep -- where the window
    size barely changes the cost of a box filter -- is pure jitter and
    silently gives up real accuracy for it.
    """
    ok = [c for c in cells if c.status == "ok" and np.isfinite(c.bad2_all)]
    if not ok:
        return None
    floor = min(c.bad2_all for c in ok)
    near = [c for c in ok if c.bad2_all <= floor + tolerance_pp]

    walls = [c.wall_s for c in near if np.isfinite(c.wall_s) and c.wall_s > 0]
    if walls:
        fastest = min(walls)
        tier = [
            c
            for c in near
            if not np.isfinite(c.wall_s) or c.wall_s <= fastest * speed_margin
        ]
        near = tier or near
    return min(near, key=lambda c: c.bad2_all)


def current_defaults(spec: AlgorithmSpec, keys: Iterable[str]) -> dict[str, Any]:
    """The algorithm's shipped values for the swept parameters."""
    algo = spec.build()
    return {k: getattr(algo, k) for k in keys if hasattr(algo, k)}


def run_sweeps(
    selectors: Sequence[str] | None,
    pairs: Sequence[StereoPair],
    out_dir: Path,
    *,
    grids: dict[str, dict[str, list]] | None = None,
    verbose: bool = True,
    **kw: Any,
) -> tuple[dict[str, list[SweepCell]], Path]:
    """Sweep every selected algorithm that has a grid. Writes raw results."""
    grids = GRIDS if grids is None else grids
    specs = [s for s in select(selectors) if s.availability()[0]]
    specs = [s for s in specs if grids.get(s.name)]
    out_dir.mkdir(parents=True, exist_ok=True)

    all_cells: dict[str, list[SweepCell]] = {}
    t_start = time.perf_counter()
    for n, spec in enumerate(specs, 1):
        grid = grids[spec.name]
        n_cfg = len(iter_configs(grid))
        if verbose:
            print(
                f"\n[{n}/{len(specs)}] {spec.name}: {n_cfg} configs "
                f"x {len(pairs)} scenes",
                flush=True,
            )
        all_cells[spec.name] = sweep_algorithm(
            spec, pairs, grid, verbose=verbose, **kw
        )

    elapsed = time.perf_counter() - t_start
    _write_raw(all_cells, pairs, out_dir, elapsed, grids=grids)
    if verbose:
        print(f"\nSwept {len(specs)} algorithms in {elapsed:.0f}s -> {out_dir}")
    return all_cells, out_dir


def load_sweep(csv_path: Path | str) -> dict[str, list[SweepCell]]:
    """Rebuild sweep cells from a written ``sweep.csv``.

    Lets the tables and figures be regenerated without re-running an hour of
    matching -- and makes the raw CSV, not the summary, the artifact of record.
    """
    import ast

    import pandas as pd

    df = pd.read_csv(csv_path)
    out: dict[str, list[SweepCell]] = {}
    for _, r in df.iterrows():
        params: dict[str, Any] = {}
        for col in df.columns:
            if not col.startswith("p_") or pd.isna(r[col]):
                continue
            key, value = col[2:], r[col]
            if isinstance(value, str) and value.startswith("("):
                try:
                    value = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    pass
            params[key] = value
        cell = SweepCell(
            algorithm=str(r["algorithm"]),
            params=params,
            bad2_all=float(r.get("bad2_all", np.nan)),
            bad1_all=float(r.get("bad1_all", np.nan)),
            epe_valid=float(r.get("epe_valid", np.nan)),
            density=float(r.get("density", np.nan)),
            wall_s=float(r.get("wall_s", np.nan)),
            n_scenes=int(r.get("n_scenes", 0) or 0),
            status=str(r.get("status", "ok")),
            message="" if pd.isna(r.get("message")) else str(r["message"]),
        )
        out.setdefault(cell.algorithm, []).append(cell)
    return out


def _write_raw(
    all_cells: dict[str, list[SweepCell]],
    pairs: Sequence[StereoPair],
    out_dir: Path,
    elapsed: float,
    grids: dict[str, dict[str, list]] | None = None,
) -> None:
    import pandas as pd

    from .profiling import machine_info

    rows = [c.row() for cells in all_cells.values() for c in cells]
    pd.DataFrame(rows).to_csv(out_dir / "sweep.csv", index=False)

    # Capture the defaults *as they were when the sweep ran*. Once the tuned
    # values are applied to the adapters, re-deriving the "default" column
    # from live code would compare the winner against itself and the summary
    # would claim the sweep changed nothing.
    g = GRIDS if grids is None else grids
    baseline: dict[str, dict[str, Any]] = {}
    for name in all_cells:
        try:
            spec = select([name])[0]
            baseline[name] = current_defaults(spec, (g.get(name) or {}).keys())
        except Exception:
            baseline[name] = {}

    (out_dir / "sweep.json").write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_s": round(elapsed, 1),
                "tune_scenes": [p.name for p in pairs],
                "n_configs": {k: len(v) for k, v in all_cells.items()},
                "grid_set": next(
                    (k for k, v in GRID_SETS.items() if v is g), "custom"
                ),
                "baseline_defaults": baseline,
                "machine": machine_info(),
            },
            indent=2,
            default=str,
        )
    )


def load_baseline(sweep_json: Path | str) -> dict[str, dict[str, Any]]:
    """The defaults recorded when a sweep ran, for before/after reporting."""
    import ast

    data = json.loads(Path(sweep_json).read_text())
    out: dict[str, dict[str, Any]] = {}
    for name, params in (data.get("baseline_defaults") or {}).items():
        clean: dict[str, Any] = {}
        for k, v in params.items():
            if isinstance(v, str) and v.startswith("("):
                try:
                    v = ast.literal_eval(v)
                except (ValueError, SyntaxError):
                    pass
            clean[k] = v
        out[name] = clean
    return out
