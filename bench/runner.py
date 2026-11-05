"""Orchestration: run every selected algorithm over every selected pair.

One cell of the benchmark is one ``(pair, algorithm)`` combination. A cell can
fail -- an algorithm may not fit the scene's cost volume in memory, or an
optional dependency may be absent -- and a failure is recorded as a row with a
``status`` other than ``ok`` rather than aborting the run, so a long sweep
survives one bad matcher.
"""
from __future__ import annotations

import json
import platform
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .algorithms import AlgorithmSpec
from .algorithms.scratch import VolumeTooLarge
from .metrics import AccuracyMetrics, colorize, evaluate
from .profiling import ResourceUsage, machine_info, measure, set_thread_limit
from .types import StereoPair


@dataclass
class CellResult:
    """One algorithm on one pair."""

    algorithm: str
    family: str
    scene: str
    status: str = "ok"  # ok | skipped | error
    message: str = ""
    width: int = 0
    height: int = 0
    ndisp: int = 0
    usage: ResourceUsage = field(default_factory=ResourceUsage)
    accuracy: AccuracyMetrics | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def row(self) -> dict[str, Any]:
        """Flatten to one dataframe row."""
        out: dict[str, Any] = {
            "algorithm": self.algorithm,
            "family": self.family,
            "scene": self.scene,
            "status": self.status,
            "width": self.width,
            "height": self.height,
            "ndisp": self.ndisp,
        }
        out.update(self.usage.flat())
        if self.accuracy is not None:
            out.update(self.accuracy.flat())
        out["message"] = self.message
        if self.extra:
            out["extra"] = json.dumps(self.extra, default=str)
        return out


@dataclass
class RunConfig:
    """Everything that affects the numbers, recorded alongside them."""

    repeats: int = 1
    warmup: int = 1
    threads: int | None = None
    eval_region: str = "nonocc"  # nonocc | all
    save_maps: bool = False
    save_figures: bool = True
    max_volume_gb: float = 4.0
    out_dir: Path = Path("results")

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["out_dir"] = str(self.out_dir)
        return d


def _eval_mask(pair: StereoPair, region: str) -> np.ndarray | None:
    if region == "all" or pair.occ_mask is None:
        return None
    return pair.occ_mask


def run_cell(
    spec: AlgorithmSpec, pair: StereoPair, config: RunConfig
) -> tuple[CellResult, np.ndarray | None]:
    """Run one algorithm on one pair. Returns ``(result, disparity_or_None)``."""
    h, w = pair.shape
    cell = CellResult(
        algorithm=spec.name,
        family=spec.family,
        scene=pair.name,
        width=w,
        height=h,
        ndisp=pair.ndisp,
    )

    ok, why = spec.availability()
    if not ok:
        cell.status, cell.message = "skipped", why
        return cell, None

    try:
        algo = spec.build()
        # Let the memory guard follow the run config where the matcher has one.
        if hasattr(algo, "max_volume_gb"):
            algo.max_volume_gb = config.max_volume_gb
        # ONNX Runtime does not read the environment variables or the OpenCV
        # thread count that `set_thread_limit` touches -- its pool size is a
        # per-session option -- so `--threads N` has to be handed over here or
        # the learned rows would quietly keep running on every core.
        if config.threads and hasattr(algo, "intra_op_threads"):
            algo.intra_op_threads = int(config.threads)

        result, usage = measure(
            lambda: algo.compute(pair),
            repeats=config.repeats,
            warmup=config.warmup,
            n_pixels=h * w,
            ndisp=pair.ndisp,
            track_gpu=spec.family in ("torch", "learned"),
        )
        cell.usage = usage
        cell.extra = dict(result.extra)

        disp, valid = result.normalized()
        if disp.shape != (h, w):
            raise ValueError(
                f"{spec.name} returned {disp.shape}, expected {(h, w)}"
            )
        if pair.has_gt:
            cell.accuracy = evaluate(
                disp, valid, pair.gt, eval_mask=_eval_mask(pair, config.eval_region)
            )
        return cell, disp

    except VolumeTooLarge as exc:
        cell.status, cell.message = "skipped", str(exc)
    except MemoryError as exc:
        cell.status, cell.message = "skipped", f"out of memory: {exc}"
    except Exception as exc:
        cell.status = "error"
        cell.message = f"{type(exc).__name__}: {exc}"
        cell.extra["traceback"] = traceback.format_exc(limit=6)
    return cell, None


def run_benchmark(
    pairs: Sequence[StereoPair],
    specs: Sequence[AlgorithmSpec],
    config: RunConfig | None = None,
    *,
    verbose: bool = True,
) -> tuple[list[CellResult], Path]:
    """Run the full grid and write results to a timestamped folder.

    Returns the cell results and the output directory.
    """
    config = config or RunConfig()
    set_thread_limit(config.threads)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_dir = Path(config.out_dir) / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "maps").mkdir(exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)

    results: list[CellResult] = []
    total = len(pairs) * len(specs)
    i = 0
    t_start = time.perf_counter()

    for pair in pairs:
        # Save the inputs and ground truth once per scene for the figure grid.
        if config.save_figures:
            _save_scene_reference(pair, out_dir / "figures")

        for spec in specs:
            i += 1
            if verbose:
                print(
                    f"[{i:3d}/{total}] {pair.name:28s} {spec.name:30s}",
                    end=" ",
                    flush=True,
                )
            cell, disp = run_cell(spec, pair, config)
            results.append(cell)

            if verbose:
                print(_cell_summary(cell), flush=True)

            if disp is not None and config.save_maps:
                from .datasets.middlebury import write_pfm

                tag = f"{_slug(pair.name)}__{spec.name}"
                write_pfm(out_dir / "maps" / f"{tag}.pfm", np.nan_to_num(disp, nan=0.0))
            if disp is not None and config.save_figures:
                _save_disparity_png(
                    disp, pair, out_dir / "figures" / f"{_slug(pair.name)}__{spec.name}.png"
                )

    elapsed = time.perf_counter() - t_start
    _write_raw(results, config, out_dir, elapsed)
    if verbose:
        print(f"\nRan {total} cells in {elapsed:.1f}s -> {out_dir}")
    return results, out_dir


# --------------------------------------------------------------------------
# output helpers
# --------------------------------------------------------------------------


def _cell_summary(cell: CellResult) -> str:
    if cell.status != "ok":
        return f"{cell.status.upper()}: {cell.message[:70]}"
    bits = [f"{cell.usage.wall_s:7.3f}s"]
    if cell.accuracy is not None:
        bits.append(f"bad2={cell.accuracy.bad_all.get('2', float('nan')):5.2f}%")
        bits.append(f"epe={cell.accuracy.epe_valid:5.2f}")
        bits.append(f"dens={cell.accuracy.density * 100:5.1f}%")
    bits.append(f"{cell.usage.peak_rss_mb:6.0f}MB")
    return "  ".join(bits)


def _slug(name: str) -> str:
    return name.replace("/", "-").replace("\\", "-")


def _save_scene_reference(pair: StereoPair, folder: Path) -> None:
    import cv2

    base = _slug(pair.name)
    left_path = folder / f"{base}__00_left.png"
    if not left_path.exists():
        cv2.imwrite(str(left_path), cv2.cvtColor(pair.left, cv2.COLOR_RGB2BGR))
    if pair.has_gt:
        gt_path = folder / f"{base}__00_gt.png"
        if not gt_path.exists():
            rgb = colorize(pair.gt)
            cv2.imwrite(str(gt_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def _save_disparity_png(disp: np.ndarray, pair: StereoPair, path: Path) -> None:
    import cv2

    # Use the ground truth's range so every method is on one colour scale.
    vmin = vmax = None
    if pair.has_gt:
        finite = np.isfinite(pair.gt)
        if finite.any():
            vmin = float(np.nanpercentile(pair.gt[finite], 1))
            vmax = float(np.nanpercentile(pair.gt[finite], 99))
    rgb = colorize(disp, vmin=vmin, vmax=vmax)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def _write_raw(
    results: list[CellResult], config: RunConfig, out_dir: Path, elapsed: float
) -> None:
    import pandas as pd

    rows = [c.row() for c in results]
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "results.csv", index=False)

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": round(elapsed, 2),
        "config": config.as_dict(),
        "machine": machine_info(),
        "n_cells": len(results),
        "n_ok": int(sum(c.status == "ok" for c in results)),
    }
    (out_dir / "run.json").write_text(json.dumps(meta, indent=2, default=str))
