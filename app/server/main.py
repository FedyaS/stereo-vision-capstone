"""FastAPI backend for the stereo playground.

Start it with ``python -m app.server`` (see ``app/server/__main__.py``), or
``uvicorn app.server.main:app --reload`` while developing.

Scene ids contain a slash (``middlebury/Teddy``), so they travel as query
parameters rather than path segments.
"""
from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from bench.algorithms import all_specs
from bench.metrics import BAD_THRESHOLDS
from bench.profiling import machine_info

from . import catalog, compute, images, jobs
from .params import describe_params

from bench.paths import RESULTS_DIR as RESULTS_ROOT
from bench.paths import WEB_DIST_DIR as WEB_DIST

app = FastAPI(title="Stereo Vision Playground", version="1.0.0")

# The Vite dev server runs on another port; in production the API serves the
# built bundle itself and this is a no-op.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

PNG = {"Content-Type": "image/png", "Cache-Control": "public, max-age=300"}


def _clean(obj: Any) -> Any:
    """JSON cannot carry NaN or Infinity; send null instead."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if not math.isfinite(f) else f
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def ok(payload: Any) -> JSONResponse:
    return JSONResponse(_clean(payload))


# --------------------------------------------------------------------------
# catalogue
# --------------------------------------------------------------------------


@app.get("/api/health")
def health() -> JSONResponse:
    return ok({"status": "ok", "machine": machine_info()})


@app.get("/api/algorithms")
def algorithms() -> JSONResponse:
    out = []
    for spec in all_specs():
        available, reason = spec.availability()
        out.append({
            "name": spec.name,
            "family": spec.family,
            "description": spec.description,
            "tags": list(spec.tags),
            "available": available,
            "reason": reason,
            "params": describe_params(spec) if available else [],
        })
    return ok(out)


@app.get("/api/scenes")
def scenes() -> JSONResponse:
    out = []
    for scene_id in catalog.scene_ids():
        try:
            out.append(catalog.describe(scene_id))
        except Exception as exc:
            out.append({
                "id": scene_id,
                "kind": scene_id.split("/")[0],
                "name": scene_id.split("/")[-1],
                "error": str(exc),
            })
    return ok(out)


@app.get("/api/scene/image")
def scene_image(
    id: str = Query(...),
    view: str = Query("left"),
    downscale: float = Query(1.0, ge=1.0, le=16.0),
) -> Response:
    try:
        pair = catalog.load(id, downscale=downscale)
    except Exception as exc:
        raise HTTPException(404, str(exc)) from exc

    if view == "left":
        return Response(images.image_png(pair.left), headers=PNG)
    if view == "right":
        return Response(images.image_png(pair.right), headers=PNG)
    if view == "gt":
        if not pair.has_gt:
            raise HTTPException(404, "this scene has no ground truth")
        vmin, vmax = images.scale_from(pair.gt, pair.gt)
        return Response(images.disparity_png(pair.gt, vmin, vmax), headers=PNG)
    if view == "nonocc":
        if pair.occ_mask is None:
            raise HTTPException(404, "this scene has no occlusion mask")
        return Response(images.valid_png(pair.occ_mask, pair.left), headers=PNG)
    raise HTTPException(400, f"unknown view {view!r}")


# --------------------------------------------------------------------------
# single-pair runs
# --------------------------------------------------------------------------


class ComputeRequest(BaseModel):
    scene_id: str
    algorithm: str
    params: dict[str, Any] = Field(default_factory=dict)
    downscale: float = Field(1.0, ge=1.0, le=16.0)
    repeats: int = Field(1, ge=1, le=15)
    eval_region: str = "nonocc"
    thresholds: list[float] = Field(default_factory=lambda: list(BAD_THRESHOLDS))
    reuse: bool = True


@app.post("/api/compute")
def compute_one(req: ComputeRequest) -> JSONResponse:
    try:
        stored = compute.run(
            req.scene_id, req.algorithm, req.params,
            downscale=req.downscale, repeats=req.repeats, reuse=req.reuse,
        )
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, f"{type(exc).__name__}: {exc}") from exc

    scoring = compute.score(
        stored, eval_region=req.eval_region, thresholds=tuple(req.thresholds)
    )
    return ok(compute.summarise(stored, scoring))


class RescoreRequest(BaseModel):
    result_id: str
    eval_region: str = "nonocc"
    thresholds: list[float] = Field(default_factory=lambda: list(BAD_THRESHOLDS))


@app.post("/api/rescore")
def rescore(req: RescoreRequest) -> JSONResponse:
    """Re-evaluate a finished run without re-matching."""
    try:
        stored = compute.get(req.result_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    scoring = compute.score(
        stored, eval_region=req.eval_region, thresholds=tuple(req.thresholds)
    )
    return ok(compute.summarise(stored, scoring))


@app.get("/api/result/image")
def result_image(
    result_id: str = Query(...),
    view: str = Query("disparity"),
    threshold: float = Query(2.0, ge=0.05, le=64.0),
    max_error: float = Query(4.0, ge=0.25, le=64.0),
) -> Response:
    try:
        stored = compute.get(result_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc

    pair = catalog.load(
        stored.scene_id, downscale=stored.params.get("_downscale", 1.0)
    )
    vmin, vmax = images.scale_from(pair.gt, stored.disparity)

    if view == "disparity":
        return Response(
            images.disparity_png(stored.disparity, vmin, vmax), headers=PNG
        )
    if view == "valid":
        return Response(images.valid_png(stored.valid, pair.left), headers=PNG)
    if not pair.has_gt:
        raise HTTPException(404, "this scene has no ground truth")
    if view == "error":
        return Response(
            images.error_png(stored.disparity, pair.gt, max_error), headers=PNG
        )
    if view == "bad":
        return Response(
            images.bad_png(stored.disparity, pair.gt, pair.left, threshold),
            headers=PNG,
        )
    raise HTTPException(400, f"unknown view {view!r}")


# --------------------------------------------------------------------------
# datasets
# --------------------------------------------------------------------------


@app.post("/api/datasets")
async def add_dataset(
    name: str = Form(...),
    ndisp: int = Form(128),
    left: UploadFile = File(...),
    right: UploadFile = File(...),
    ground_truth: UploadFile | None = File(None),
) -> JSONResponse:
    gt_bytes = await ground_truth.read() if ground_truth is not None else None
    try:
        scene_id = catalog.create_custom(
            name,
            await left.read(),
            await right.read(),
            ndisp=int(ndisp),
            gt_bytes=gt_bytes,
            gt_filename=ground_truth.filename if ground_truth else None,
        )
    except FileExistsError as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return ok(catalog.describe(scene_id))


@app.delete("/api/datasets/{name}")
def remove_dataset(name: str) -> JSONResponse:
    try:
        catalog.delete_custom(name)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return ok({"deleted": name})


class SyntheticRequest(BaseModel):
    seed: int = Field(0, ge=0, le=99999)


@app.post("/api/datasets/synthetic")
def synthetic(req: SyntheticRequest) -> JSONResponse:
    """A freshly generated pair with exact ground truth -- no download."""
    scene_id = catalog.remember_seed(req.seed)
    try:
        return ok(catalog.describe(scene_id))
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


# --------------------------------------------------------------------------
# stored benchmark runs
# --------------------------------------------------------------------------


def _run_dirs() -> list[Path]:
    """Stored runs, newest first.

    ``results/keep/`` holds committed copies of runs, nested one level deeper,
    so it has no ``results.csv`` of its own and is skipped by the check below
    -- deliberately: a kept run is an archive, not a run this server owns.
    """
    if not RESULTS_ROOT.is_dir():
        return []
    return sorted(
        (d for d in RESULTS_ROOT.iterdir()
         if d.is_dir() and (d / "results.csv").exists()),
        key=lambda d: d.name,
        reverse=True,
    )


def _run_dir(run_id: str) -> Path:
    """Resolve a run id to its folder, or 404.

    ``Path(run_id).name`` strips any directory part. Starlette decodes path
    parameters *after* routing, so ``{run_id}`` can arrive holding separators
    (``..%2F..%2Fdata``) even though the route pattern does not match a literal
    slash -- without this, ``RESULTS_ROOT / run_id`` would read outside
    ``results/``. The report route already did this; the others did not.
    """
    d = RESULTS_ROOT / Path(run_id).name
    if not (d.is_dir() and (d / "results.csv").exists()):
        raise HTTPException(404, f"no run {run_id!r}")
    return d


@app.get("/api/runs")
def runs() -> JSONResponse:
    out = []
    for d in _run_dirs():
        meta: dict[str, Any] = {}
        if (d / "run.json").exists():
            try:
                meta = json.loads((d / "run.json").read_text())
            except Exception:
                meta = {}
        with open(d / "results.csv", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        out.append({
            "id": d.name,
            "n_cells": len(rows),
            "n_ok": sum(r.get("status") == "ok" for r in rows),
            "algorithms": sorted({r["algorithm"] for r in rows}),
            "scenes": sorted({r["scene"] for r in rows}),
            "created_utc": meta.get("created_utc"),
            "elapsed_s": meta.get("elapsed_s"),
            "config": meta.get("config"),
            "has_summary": (d / "summary.md").exists(),
        })
    return ok(out)


@app.get("/api/runs/{run_id}")
def run_detail(run_id: str) -> JSONResponse:
    d = _run_dir(run_id)
    with open(d / "results.csv", newline="", encoding="utf-8") as f:
        rows = []
        for raw in csv.DictReader(f):
            row: dict[str, Any] = {}
            for k, v in raw.items():
                if v in ("", None):
                    row[k] = None
                    continue
                try:
                    f_ = float(v)
                    row[k] = None if not math.isfinite(f_) else f_
                except (TypeError, ValueError):
                    row[k] = v
            rows.append(row)
    summary = (d / "summary.md").read_text(encoding="utf-8") if (
        d / "summary.md"
    ).exists() else None
    return ok({"id": run_id, "rows": rows, "summary": summary})


@app.get("/api/runs/{run_id}/figure")
def run_figure(run_id: str, name: str = Query(...)) -> Response:
    safe = Path(name).name  # strip any directory component
    path = _run_dir(run_id) / "figures" / safe
    if not path.is_file() or path.suffix.lower() != ".png":
        raise HTTPException(404, f"no figure {safe!r} in run {run_id!r}")
    return FileResponse(path, media_type="image/png")


@app.get("/api/runs/{run_id}/report")
def run_report(
    run_id: str,
    format: str = Query("html", pattern="^(html|pdf)$"),
    refresh: bool = Query(False),
) -> Response:
    """The narrative report for one run, generated on first request.

    Generation takes a couple of seconds (most of it resampling the gallery
    thumbnails), so the file is written into the run folder and reused.
    ``refresh=1`` rebuilds it from the run's data -- but with the code this
    process imported, so an edit to ``bench/explain.py`` needs ``--reload``
    or a restart to take effect.
    """
    from bench.explain import write_report

    d = _run_dir(run_id)
    wanted = d / ("report.pdf" if format == "pdf" else "report.html")
    if refresh or not wanted.exists():
        try:
            write_report(d, pdf=(format == "pdf"))
        except RuntimeError as exc:  # no browser available to print with
            raise HTTPException(503, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(500, f"could not build the report: {exc}") from exc

    if format == "pdf":
        return FileResponse(wanted, media_type="application/pdf",
                            filename=f"stereo-report-{d.name}.pdf")
    # Served inline rather than as JSON so the browser can just open the URL.
    return Response(wanted.read_text(encoding="utf-8"), media_type="text/html")


@app.get("/api/runs/{run_id}/figures")
def run_figures(run_id: str) -> JSONResponse:
    folder = _run_dir(run_id) / "figures"
    if not folder.is_dir():
        return ok([])
    return ok(sorted(p.name for p in folder.glob("*.png")))


# --------------------------------------------------------------------------
# background jobs
# --------------------------------------------------------------------------


class BenchmarkRequest(BaseModel):
    scene_ids: list[str]
    algorithms: list[str]
    repeats: int = Field(1, ge=1, le=10)
    warmup: int = Field(1, ge=0, le=3)
    downscale: float = Field(1.0, ge=1.0, le=16.0)
    eval_region: str = "nonocc"
    save_figures: bool = True


@app.post("/api/jobs/benchmark")
def start_benchmark(req: BenchmarkRequest) -> JSONResponse:
    try:
        job = jobs.start_benchmark(
            req.scene_ids, req.algorithms,
            repeats=req.repeats, warmup=req.warmup, downscale=req.downscale,
            eval_region=req.eval_region, save_figures=req.save_figures,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return ok(job.payload())


@app.get("/api/jobs")
def list_jobs() -> JSONResponse:
    return ok(jobs.listing())


@app.get("/api/jobs/{job_id}")
def job_detail(job_id: str) -> JSONResponse:
    try:
        return ok(jobs.get(job_id).payload())
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.post("/api/jobs/{job_id}/cancel")
def job_cancel(job_id: str) -> JSONResponse:
    jobs.cancel(job_id)
    try:
        return ok(jobs.get(job_id).payload())
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


# --------------------------------------------------------------------------
# the built frontend
# --------------------------------------------------------------------------

if WEB_DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(WEB_DIST), html=True), name="web")
else:

    @app.get("/")
    def no_bundle() -> JSONResponse:
        return ok({
            "message": "API is running, but the web bundle is not built.",
            "build": "cd app/web && npm install && npm run build",
            "dev": "cd app/web && npm run dev  (then open http://localhost:5173)",
            "api_docs": "/docs",
        })
