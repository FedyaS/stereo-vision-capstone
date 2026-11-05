"""Timing and compute-resource measurement.

:func:`measure` runs a callable a few times and reports wall time, CPU time,
peak resident memory and (when a CUDA device is used) peak GPU memory.

Two derived numbers are worth more than raw seconds when comparing matchers:

``parallelism``
    ``cpu_seconds / wall_seconds`` -- roughly how many cores the method kept
    busy. OpenCV's SGBM multithreads internally, a hand-written NumPy loop
    does not, so a NumPy method that looks 8x slower in wall time may be
    using 8x less compute.
``mdps``
    Mega *disparity hypotheses* per second, ``H * W * ndisp / wall / 1e6``.
    Stereo cost scales with the search range, so this normalises away both
    image size and how wide a disparity range the scene needed.
"""
from __future__ import annotations

import gc
import os
import platform
import statistics
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, TypeVar

import psutil

T = TypeVar("T")


@dataclass
class ResourceUsage:
    """Timing and resource cost of one algorithm run."""

    wall_s: float = float("nan")
    wall_s_min: float = float("nan")
    wall_s_std: float = 0.0
    cpu_s: float = float("nan")
    parallelism: float = float("nan")
    peak_rss_mb: float = float("nan")
    rss_delta_mb: float = float("nan")
    peak_threads: int = 0
    gpu_peak_mb: float = float("nan")
    repeats: int = 1
    mdps: float = float("nan")
    mpix_per_s: float = float("nan")

    def flat(self) -> dict[str, Any]:
        return dict(asdict(self))


class _Sampler(threading.Thread):
    """Polls process RSS and thread count while the measured code runs."""

    def __init__(self, proc: psutil.Process, interval: float = 0.005) -> None:
        super().__init__(daemon=True)
        self.proc, self.interval = proc, interval
        self.peak_rss = 0
        self.peak_threads = 0
        self._stop_evt = threading.Event()

    def run(self) -> None:
        while not self._stop_evt.is_set():
            try:
                self.peak_rss = max(self.peak_rss, self.proc.memory_info().rss)
                self.peak_threads = max(self.peak_threads, self.proc.num_threads())
            except psutil.Error:  # pragma: no cover - process went away
                return
            self._stop_evt.wait(self.interval)

    def stop(self) -> None:
        self._stop_evt.set()
        self.join(timeout=1.0)


def measure(
    fn: Callable[[], T],
    *,
    repeats: int = 1,
    warmup: int = 0,
    n_pixels: int | None = None,
    ndisp: int | None = None,
    track_gpu: bool = False,
) -> tuple[T, ResourceUsage]:
    """Run ``fn`` and measure it. Returns ``(last_result, usage)``.

    ``repeats`` runs are timed and the *median* wall time reported; the fastest
    run is reported separately as ``wall_s_min`` since it is the least polluted
    by background load. ``warmup`` runs are discarded first -- use at least one
    for anything with lazy initialisation (ONNX sessions, OpenCV thread pools).
    """
    proc = psutil.Process(os.getpid())

    for _ in range(max(0, warmup)):
        fn()

    torch_cuda = None
    if track_gpu:
        try:
            import torch

            if torch.cuda.is_available():
                torch_cuda = torch
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
        except Exception:
            torch_cuda = None

    gc.collect()
    rss_before = proc.memory_info().rss
    sampler = _Sampler(proc)
    sampler.start()

    walls: list[float] = []
    cpu_before = proc.cpu_times()
    result: Any = None
    try:
        for _ in range(max(1, repeats)):
            t0 = time.perf_counter()
            result = fn()
            if torch_cuda is not None:
                torch_cuda.cuda.synchronize()
            walls.append(time.perf_counter() - t0)
    finally:
        cpu_after = proc.cpu_times()
        sampler.stop()

    cpu_total = (cpu_after.user - cpu_before.user) + (
        cpu_after.system - cpu_before.system
    )
    n = len(walls)
    u = ResourceUsage(
        wall_s=float(statistics.median(walls)),
        wall_s_min=float(min(walls)),
        wall_s_std=float(statistics.stdev(walls)) if n > 1 else 0.0,
        cpu_s=cpu_total / n,
        repeats=n,
        peak_rss_mb=sampler.peak_rss / 1e6,
        rss_delta_mb=max(0.0, (sampler.peak_rss - rss_before) / 1e6),
        peak_threads=sampler.peak_threads,
    )
    u.parallelism = u.cpu_s / u.wall_s if u.wall_s > 0 else float("nan")
    if torch_cuda is not None:
        u.gpu_peak_mb = torch_cuda.cuda.max_memory_allocated() / 1e6
    if n_pixels:
        u.mpix_per_s = n_pixels / u.wall_s / 1e6 if u.wall_s > 0 else float("nan")
        if ndisp:
            u.mdps = n_pixels * ndisp / u.wall_s / 1e6 if u.wall_s > 0 else float("nan")
    return result, u


def set_thread_limit(n: int | None) -> None:
    """Pin the thread count of every backend we can reach.

    Call before any measurement when you want numbers that compare compute
    cost rather than how well each library happens to multithread. ``None``
    leaves every backend at its default.
    """
    if n is None:
        return
    n = max(1, int(n))
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[var] = str(n)
    try:
        import cv2

        cv2.setNumThreads(n)
    except Exception:
        pass
    try:
        import torch

        torch.set_num_threads(n)
    except Exception:
        pass


def machine_info() -> dict[str, Any]:
    """Describe the machine, so a results file is interpretable later."""
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "ram_total_gb": round(psutil.virtual_memory().total / 1e9, 2),
    }
    try:
        import cpuinfo

        info["cpu_brand"] = cpuinfo.get_cpu_info().get("brand_raw")
    except Exception:
        pass
    try:
        import cv2

        info["opencv"] = cv2.__version__
        info["opencv_threads"] = cv2.getNumThreads()
    except Exception:
        pass
    try:
        import numpy

        info["numpy"] = numpy.__version__
    except Exception:
        pass
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        import onnxruntime

        info["onnxruntime"] = onnxruntime.__version__
        info["onnxruntime_providers"] = onnxruntime.get_available_providers()
    except Exception:
        pass
    return info
