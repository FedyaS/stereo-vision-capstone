"""The playground backend.

These run against the FastAPI app in-process, so they cover the contract the
browser actually depends on -- including the metric key format, which is the
kind of mismatch that produces a blank number in the UI rather than an error
anywhere.
"""
from __future__ import annotations

import io
import warnings

import numpy as np
import pytest

pytest.importorskip("fastapi")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from fastapi.testclient import TestClient

from app.server import catalog
from app.server.main import app
from app.server.params import coerce_params, describe_params
from bench.algorithms import select


@pytest.fixture(scope="module")
def client():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with TestClient(app) as c:
            yield c


# --------------------------------------------------------------------------
# catalogue
# --------------------------------------------------------------------------


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert "cpu_count_logical" in body["machine"]


def test_algorithms_listing_matches_the_registry(client):
    body = client.get("/api/algorithms").json()
    assert len(body) == len(list(select(None)))
    names = {a["name"] for a in body}
    assert "scratch_census_sgm4" in names
    for a in body:
        assert isinstance(a["available"], bool)
        if a["available"]:
            assert isinstance(a["params"], list)


def test_scene_listing_is_never_empty(client):
    body = client.get("/api/scenes").json()
    # Synthetic scenes need no download, so there is always something.
    assert any(s["kind"] == "synthetic" for s in body)
    for s in body:
        if "error" not in s:
            assert s["width"] > 0 and s["height"] > 0 and s["ndisp"] > 0


def test_scene_images_render(client):
    r = client.get("/api/scene/image", params={"id": "synthetic/seed0", "view": "left"})
    assert r.status_code == 200
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_unknown_scene_is_404(client):
    r = client.get("/api/scene/image", params={"id": "nope/nope", "view": "left"})
    assert r.status_code == 404


def test_unknown_view_is_400(client):
    r = client.get("/api/scene/image", params={"id": "synthetic/seed0", "view": "zzz"})
    assert r.status_code == 400


# --------------------------------------------------------------------------
# compute and scoring
# --------------------------------------------------------------------------


def _compute(client, **kw):
    body = {
        "scene_id": "synthetic/seed0",
        "algorithm": "opencv_sgbm_3way",
        "params": {},
        **kw,
    }
    r = client.post("/api/compute", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def test_compute_returns_metrics_and_timing(client):
    d = _compute(client)
    assert d["metrics"] is not None
    assert d["timing"]["wall_s"] > 0
    assert 0 <= d["metrics"]["density"] <= 1


def test_metric_keys_match_javascript_string_formatting(client):
    """`String(2)` is "2", not "2.0" -- the UI looks up exactly these keys.

    Getting this wrong renders a blank metric instead of raising, so it is
    pinned here rather than left to a visual check.
    """
    d = _compute(client, thresholds=[0.5, 1, 2, 4])
    assert set(d["metrics"]["bad"]) == {"0.5", "1", "2", "4"}
    assert set(d["metrics"]["bad_valid"]) == {"0.5", "1", "2", "4"}


def test_bad_thresholds_are_monotonic(client):
    bad = _compute(client, thresholds=[0.5, 1, 2, 4])["metrics"]["bad"]
    values = [bad[k] for k in ("0.5", "1", "2", "4")]
    assert values == sorted(values, reverse=True)


def test_rescore_reuses_the_stored_map(client):
    first = _compute(client)
    r = client.post("/api/rescore", json={
        "result_id": first["result_id"], "eval_region": "all",
    })
    assert r.status_code == 200
    again = r.json()
    assert again["result_id"] == first["result_id"]
    assert again["metrics"]["eval_region"] == "all"
    # Same disparity map, so the timing is the original run's, not a new one.
    assert again["timing"]["wall_s"] == first["timing"]["wall_s"]


def test_rescore_of_unknown_result_is_404(client):
    r = client.post("/api/rescore", json={"result_id": "nope", "eval_region": "all"})
    assert r.status_code == 404


def test_parameters_actually_change_the_result(client):
    a = _compute(client, algorithm="scratch_sad_wta", params={"block": 5})
    b = _compute(client, algorithm="scratch_sad_wta", params={"block": 21})
    assert a["result_id"] != b["result_id"]
    assert a["metrics"]["bad"]["2"] != b["metrics"]["bad"]["2"]


def test_unknown_parameter_is_rejected(client):
    r = client.post("/api/compute", json={
        "scene_id": "synthetic/seed0", "algorithm": "opencv_bm",
        "params": {"not_a_param": 3},
    })
    assert r.status_code == 400
    assert "not_a_param" in r.text


def test_unknown_algorithm_is_404(client):
    r = client.post("/api/compute", json={
        "scene_id": "synthetic/seed0", "algorithm": "nope", "params": {},
    })
    assert r.status_code == 404


def test_result_views_render(client):
    d = _compute(client)
    for view in ("disparity", "error", "bad", "valid"):
        r = client.get("/api/result/image", params={
            "result_id": d["result_id"], "view": view,
        })
        assert r.status_code == 200, view
        assert r.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_downscale_shrinks_the_pair(client):
    full = _compute(client, downscale=1)
    half = _compute(client, downscale=2)
    assert half["width"] < full["width"]
    assert half["ndisp"] <= full["ndisp"]


# --------------------------------------------------------------------------
# parameter introspection
# --------------------------------------------------------------------------


def test_describe_params_offers_the_interesting_switches():
    spec = select(["scratch_sad_wta"])[0]
    by_name = {p["name"]: p for p in describe_params(spec)}
    assert set(by_name["metric"]["choices"]) >= {"sad", "ssd", "zncc", "census"}
    assert set(by_name["aggregation"]["choices"]) >= {"none", "sgm4", "sgm8", "dp"}
    assert by_name["block"]["kind"] == "int"
    assert by_name["block"]["odd_only"] is True


def test_describe_params_marks_swept_parameters():
    spec = select(["opencv_sgbm_3way"])[0]
    swept = {p["name"] for p in describe_params(spec) if p["tuned"]}
    assert "block_size" in swept


def test_coerce_forces_odd_windows():
    spec = select(["scratch_sad_wta"])[0]
    assert coerce_params(spec, {"block": 10})["block"] % 2 == 1


def test_coerce_rejects_unknown_names():
    spec = select(["scratch_sad_wta"])[0]
    with pytest.raises(ValueError):
        coerce_params(spec, {"bogus": 1})


# --------------------------------------------------------------------------
# datasets
# --------------------------------------------------------------------------


def _png_bytes(w: int, h: int, seed: int = 0) -> bytes:
    import cv2

    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    return cv2.imencode(".png", img)[1].tobytes()


def test_dataset_upload_round_trip(client):
    name = "pytest-upload"
    catalog.delete_custom(name) if name in catalog.list_custom() else None
    files = {
        "left": ("im0.png", io.BytesIO(_png_bytes(64, 48, 1)), "image/png"),
        "right": ("im1.png", io.BytesIO(_png_bytes(64, 48, 2)), "image/png"),
    }
    r = client.post("/api/datasets", data={"name": name, "ndisp": 32}, files=files)
    assert r.status_code == 200, r.text
    try:
        body = r.json()
        assert body["id"] == f"custom/{name}"
        assert (body["width"], body["height"]) == (64, 48)
        assert body["has_gt"] is False
        assert f"custom/{name}" in [s["id"] for s in client.get("/api/scenes").json()]
    finally:
        assert client.delete(f"/api/datasets/{name}").status_code == 200
    assert client.delete(f"/api/datasets/{name}").status_code == 404


def test_upload_rejects_mismatched_sizes(client):
    files = {
        "left": ("im0.png", io.BytesIO(_png_bytes(64, 48)), "image/png"),
        "right": ("im1.png", io.BytesIO(_png_bytes(32, 24)), "image/png"),
    }
    r = client.post("/api/datasets", data={"name": "pytest-bad"}, files=files)
    assert r.status_code == 400
    assert "same size" in r.text


@pytest.mark.parametrize("name", ["../escape", "a/b", "", "x" * 80])
def test_upload_rejects_unsafe_names(client, name):
    files = {
        "left": ("im0.png", io.BytesIO(_png_bytes(16, 16)), "image/png"),
        "right": ("im1.png", io.BytesIO(_png_bytes(16, 16)), "image/png"),
    }
    r = client.post("/api/datasets", data={"name": name}, files=files)
    assert r.status_code in (400, 404, 422)


def test_synthetic_sampling(client):
    r = client.post("/api/datasets/synthetic", json={"seed": 123})
    assert r.status_code == 200
    assert r.json()["id"] == "synthetic/seed123"
    assert r.json()["has_gt"] is True


# --------------------------------------------------------------------------
# runs and jobs
# --------------------------------------------------------------------------


def test_runs_listing(client):
    r = client.get("/api/runs")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_unknown_run_is_404(client):
    assert client.get("/api/runs/not-a-run").status_code == 404


def test_run_report_is_generated_and_cached(client):
    runs = client.get("/api/runs").json()
    if not runs:
        pytest.skip("no stored runs")
    run_id = runs[0]["id"]
    r = client.get(f"/api/runs/{run_id}/report")
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("text/html")
    assert r.text.startswith("<!doctype html>")
    # Written into the run folder, so the second request is a file read.
    from app.server.main import RESULTS_ROOT

    assert (RESULTS_ROOT / run_id / "report.html").exists()
    again = client.get(f"/api/runs/{run_id}/report")
    assert again.status_code == 200 and again.text == r.text


def test_run_report_of_unknown_run_is_404(client):
    assert client.get("/api/runs/not-a-run/report").status_code == 404


def test_run_report_rejects_an_unknown_format(client):
    runs = client.get("/api/runs").json()
    if not runs:
        pytest.skip("no stored runs")
    r = client.get(f"/api/runs/{runs[0]['id']}/report", params={"format": "docx"})
    assert r.status_code == 422


def test_run_report_path_traversal_is_blocked(client):
    assert client.get("/api/runs/..%2F..%2Fetc/report").status_code in (404, 400)


def test_run_figure_path_traversal_is_blocked(client):
    runs = client.get("/api/runs").json()
    if not runs:
        pytest.skip("no stored runs")
    r = client.get(
        f"/api/runs/{runs[0]['id']}/figure",
        params={"name": "../../../../etc/passwd"},
    )
    assert r.status_code == 404


def test_benchmark_job_runs_to_completion(client):
    import time

    r = client.post("/api/jobs/benchmark", json={
        "scene_ids": ["synthetic/seed0"],
        "algorithms": ["opencv_sgbm_3way"],
        "downscale": 2,
        "save_figures": False,
    })
    assert r.status_code == 200, r.text
    job_id = r.json()["id"]

    for _ in range(80):
        body = client.get(f"/api/jobs/{job_id}").json()
        if body["status"] != "running":
            break
        time.sleep(0.25)
    assert body["status"] == "done", body
    assert body["done"] == body["total"] == 1
    assert body["out_dir"]


def test_benchmark_job_rejects_an_empty_selection(client):
    r = client.post("/api/jobs/benchmark", json={
        "scene_ids": [], "algorithms": ["opencv_bm"],
    })
    assert r.status_code == 400


def test_generated_seed_joins_the_catalogue(client):
    """A sample you generate must be selectable everywhere, not just by id."""
    seed = 4242
    scene_id = f"synthetic/seed{seed}"
    assert scene_id not in [s["id"] for s in client.get("/api/scenes").json()]

    assert client.post("/api/datasets/synthetic", json={"seed": seed}).status_code == 200
    assert scene_id in [s["id"] for s in client.get("/api/scenes").json()]

    # And it is usable: the playground, Compare and batch runs all go through
    # this same path.
    r = client.post("/api/compute", json={
        "scene_id": scene_id, "algorithm": "opencv_sgbm_3way", "params": {},
    })
    assert r.status_code == 200
    assert r.json()["metrics"] is not None


def test_catalog_lists_and_loads_split_generated_datasets(tmp_path, monkeypatch):
    """A split dataset must be addressable as a scene, like a flat one.

    The catalogue originally assumed `synth/<set>/<sample>`. Splits add a
    path segment, and without this the generated scenes silently vanish from
    every picker -- the dataset is on disk and simply cannot be selected.
    """
    from app.server import catalog
    from bench.synth import generate_dataset, preset

    # Redirect the path constant rather than the repo root, per bench.paths:
    # the module reads it at call time, so the monkeypatch is seen, and the
    # test cannot pick up a real dataset sitting in data/synth.
    root = tmp_path / "data" / "synth"
    monkeypatch.setattr(catalog, "SYNTH_DATA_DIR", root)
    rules = preset("void").replace(
        width=160, height=112, ndisp=32, supersample=1, shadows=False,
        min_objects=2, max_objects=3,
    )
    generate_dataset(
        rules, 4, root / "withsplits", workers=1,
        splits={"train": 0.5, "val": 0.5},
    )
    generate_dataset(rules, 2, root / "flat", workers=1)

    ids = catalog.scene_ids()
    assert "synth/withsplits/train/00000" in ids
    assert "synth/withsplits/val/00002" in ids
    assert "synth/flat/00000" in ids

    catalog._cache.clear()
    pair = catalog.load("synth/withsplits/train/00000")
    assert pair.has_gt and pair.occ_mask is not None

    for bad in ("synth/flat/../../escape", "synth/flat/a/b/c"):
        with pytest.raises((ValueError, FileNotFoundError)):
            catalog.load(bad)
