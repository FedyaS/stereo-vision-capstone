"""Background batch runs, so the UI can trigger a real benchmark sweep.

A single-pair run in the playground is synchronous -- it takes milliseconds to
a few seconds. A batch across scenes and algorithms is minutes, so it goes to
a worker thread with pollable progress and a cancel flag, and writes its
output through the normal :mod:`bench.runner` path. That means a run triggered
from the browser lands in ``results/<timestamp>/`` exactly like a run
triggered from the CLI, with the same ``summary.md`` and figures.
"""
from __future__ import annotations

import threading
import time
import traceback
import uuid
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from typing import Any

from bench.algorithms import select
from bench.runner import CellResult, RunConfig, _save_disparity_png, _save_scene_reference, _slug, run_cell
from bench.types import StereoPair

from bench.paths import RESULTS_DIR as RESULTS_ROOT
from bench.paths import describe as _describe

from . import catalog


@dataclass
class Job:
    id: str
    kind: str
    total: int
    done: int = 0
    status: str = "running"  # running | done | error | cancelled
    message: str = ""
    current: str = ""
    out_dir: str | None = None
    started: float = field(default_factory=time.time)
    finished: float | None = None
    log: list[str] = field(default_factory=list)

    def payload(self) -> dict[str, Any]:
        d = asdict(self)
        d["elapsed_s"] = round((self.finished or time.time()) - self.started, 1)
        d["log"] = self.log[-120:]
        return d


_jobs: "OrderedDict[str, Job]" = OrderedDict()
_cancel: dict[str, threading.Event] = {}
_lock = threading.Lock()

#: How many finished jobs to remember. The server is long-lived and every
#: batch run left a Job (with up to 120 log lines) in here forever, so a day
#: of poking at the UI grew the dict without bound. Only ``listing()``'s most
#: recent 20 are ever shown; keep twice that so a cancel or a poll arriving
#: late still finds its job. Running jobs are never evicted.
_JOB_LIMIT = 40


def _evict_locked() -> None:
    """Drop the oldest *finished* jobs. Caller holds ``_lock``."""
    while len(_jobs) > _JOB_LIMIT:
        for job_id, job in _jobs.items():
            if job.status != "running":
                del _jobs[job_id]
                break
        else:  # every remaining job is still running -- nothing to evict
            return


def get(job_id: str) -> Job:
    with _lock:
        if job_id not in _jobs:
            raise KeyError(f"no job {job_id!r}")
        return _jobs[job_id]


def listing() -> list[dict[str, Any]]:
    with _lock:
        jobs = sorted(_jobs.values(), key=lambda j: j.started, reverse=True)
    return [j.payload() for j in jobs[:20]]


def cancel(job_id: str) -> None:
    with _lock:
        if job_id in _cancel:
            _cancel[job_id].set()


def start_benchmark(
    scene_ids: list[str],
    algorithms: list[str],
    *,
    repeats: int = 1,
    warmup: int = 1,
    downscale: float = 1.0,
    eval_region: str = "nonocc",
    save_figures: bool = True,
) -> Job:
    """Kick off a benchmark grid in the background."""
    specs = [s for s in select(algorithms) if s.availability()[0]]
    if not specs:
        raise ValueError("none of the selected algorithms is available")
    if not scene_ids:
        raise ValueError("pick at least one scene")

    job = Job(
        id=uuid.uuid4().hex[:12],
        kind="benchmark",
        total=len(scene_ids) * len(specs),
    )
    stop = threading.Event()
    with _lock:
        _jobs[job.id] = job
        _cancel[job.id] = stop
        _evict_locked()

    def worker() -> None:
        from datetime import datetime, timezone

        from bench.report import write_all
        from bench.runner import _write_raw

        try:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            out_dir = RESULTS_ROOT / stamp
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "maps").mkdir(exist_ok=True)
            (out_dir / "figures").mkdir(exist_ok=True)
            # A repo-relative path for the UI to show -- but fall back to the
            # absolute one, because a test redirects RESULTS_ROOT to a tmp dir
            # outside the checkout and relative_to() would raise there.
            job.out_dir = _describe(out_dir)

            config = RunConfig(
                repeats=repeats, warmup=warmup, eval_region=eval_region,
                save_maps=False, save_figures=save_figures, out_dir=RESULTS_ROOT,
            )
            results: list[CellResult] = []
            t0 = time.perf_counter()

            for scene_id in scene_ids:
                if stop.is_set():
                    break
                pair: StereoPair = catalog.load(scene_id, downscale=downscale)
                if save_figures:
                    _save_scene_reference(pair, out_dir / "figures")
                for spec in specs:
                    if stop.is_set():
                        break
                    job.current = f"{scene_id} / {spec.name}"
                    cell, disp = run_cell(spec, pair, config)
                    results.append(cell)
                    job.done += 1
                    job.log.append(
                        f"{scene_id} {spec.name}: {cell.status} "
                        f"{cell.usage.wall_s:.3f}s"
                        + (
                            f" bad2={cell.accuracy.bad_all.get('2', float('nan')):.2f}%"
                            if cell.accuracy
                            else ""
                        )
                    )
                    if disp is not None and save_figures:
                        _save_disparity_png(
                            disp, pair,
                            out_dir / "figures"
                            / f"{_slug(pair.name)}__{spec.name}.png",
                        )

            _write_raw(results, config, out_dir, time.perf_counter() - t0)
            try:
                write_all(results, out_dir)
            except Exception as exc:  # a bad figure must not lose the data
                job.log.append(f"report generation failed: {exc}")

            job.status = "cancelled" if stop.is_set() else "done"
            job.message = f"{len(results)} cells"
        except Exception as exc:
            job.status = "error"
            job.message = f"{type(exc).__name__}: {exc}"
            job.log.append(traceback.format_exc(limit=6))
        finally:
            job.finished = time.time()
            with _lock:
                _cancel.pop(job.id, None)

    threading.Thread(target=worker, daemon=True, name=f"job-{job.id}").start()
    return job
