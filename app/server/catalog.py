"""Scene discovery and custom datasets.

Three sources, all addressed by one id scheme so the UI never needs to care
which is which:

``middlebury/Teddy``   a downloaded benchmark scene
``synthetic/seed3``    generated on demand, exact ground truth, no download
``custom/<name>``      a folder the user added through the UI

Loaded pairs are cached, because every parameter tweak in the playground
re-runs matching on the same images and re-decoding a 741x497 PNG pair each
time is pure latency.
"""
from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bench.datasets import list_middlebury_scenes, load_middlebury_scene
from bench.datasets.local import load_image_pair
from bench.datasets.synthetic import make_synthetic_pair
from bench.paths import CUSTOM_DATA_DIR as CUSTOM_ROOT
from bench.paths import REPO_ROOT
from bench.types import StereoPair

_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.-]{0,63}$")

_cache: dict[str, StereoPair] = {}
_CACHE_LIMIT = 12

#: Seeds beyond the always-listed ones that the UI has asked for this session.
#: Without this, generating a sample would produce a scene that works if you
#: address it directly but never appears in a picker, so it could not be used
#: in the playground, a comparison or a batch run.
_extra_seeds: set[int] = set()
_DEFAULT_SEEDS = (0, 1, 2, 3)


def remember_seed(seed: int) -> str:
    """Add a generated synthetic seed to the catalogue and return its id."""
    if seed not in _DEFAULT_SEEDS:
        _extra_seeds.add(int(seed))
    return f"synthetic/seed{int(seed)}"


def safe_name(name: str) -> str:
    """Reject anything that could escape the custom-dataset folder."""
    name = (name or "").strip()
    if not _SAFE_NAME.match(name) or ".." in name:
        raise ValueError(
            "dataset name must be 1-64 characters of letters, digits, "
            "space, dot, dash or underscore"
        )
    return name


def list_custom() -> list[str]:
    if not CUSTOM_ROOT.is_dir():
        return []
    return sorted(
        d.name
        for d in CUSTOM_ROOT.iterdir()
        if d.is_dir() and any((d / f"im0{e}").exists() for e in (".png", ".jpg"))
    )


def scene_ids() -> list[str]:
    ids = [f"middlebury/{s}" for s in list_middlebury_scenes()]
    ids += [f"custom/{s}" for s in list_custom()]
    seeds = list(_DEFAULT_SEEDS) + sorted(_extra_seeds)
    ids += [f"synthetic/seed{i}" for i in seeds]
    return ids


def load(scene_id: str, *, downscale: float = 1.0) -> StereoPair:
    """Load (and cache) one scene by id."""
    key = f"{scene_id}@{downscale:g}"
    if key in _cache:
        return _cache[key]

    kind, _, rest = scene_id.partition("/")
    if kind == "middlebury":
        pair = load_middlebury_scene(rest)
    elif kind == "synthetic":
        seed = int(rest.replace("seed", "") or 0)
        pair = make_synthetic_pair(seed=seed, size=(320, 448), ndisp=64)
        pair.name = scene_id
    elif kind == "custom":
        folder = CUSTOM_ROOT / safe_name(rest)
        if not folder.is_dir():
            raise FileNotFoundError(f"no custom dataset named {rest!r}")
        pair = load_image_pair(folder, name=scene_id)
    else:
        raise ValueError(f"unknown scene id {scene_id!r}")

    if downscale and downscale != 1.0:
        pair = _downscale(pair, downscale)

    if len(_cache) >= _CACHE_LIMIT:
        _cache.pop(next(iter(_cache)))
    _cache[key] = pair
    return pair


def _downscale(pair: StereoPair, factor: float) -> StereoPair:
    """Shrink a pair, scaling the disparity range and ground truth with it."""
    import cv2

    s = 1.0 / float(factor)
    size = (max(1, int(round(pair.left.shape[1] * s))),
            max(1, int(round(pair.left.shape[0] * s))))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    gt = None
    if pair.gt is not None:
        gt = cv2.resize(pair.gt, size, interpolation=cv2.INTER_NEAREST) * s
    mask = None
    if pair.occ_mask is not None:
        mask = cv2.resize(
            pair.occ_mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST
        ) > 0
    return StereoPair(
        name=pair.name,
        left=cv2.resize(pair.left, size, interpolation=interp),
        right=cv2.resize(pair.right, size, interpolation=interp),
        ndisp=max(16, int(round(pair.ndisp * s))),
        gt=gt,
        occ_mask=mask,
        focal_length_px=pair.focal_length_px,
        baseline_mm=pair.baseline_mm,
        doffs=pair.doffs,
        meta={**pair.meta, "downscale": factor},
    )


def describe(scene_id: str) -> dict[str, Any]:
    """Light metadata for the scene list, without decoding the full pair."""
    pair = load(scene_id)
    h, w = pair.shape
    return {
        "id": scene_id,
        "kind": scene_id.split("/")[0],
        "name": scene_id.split("/")[-1],
        "width": w,
        "height": h,
        "ndisp": pair.ndisp,
        "has_gt": pair.has_gt,
        "megapixels": round(pair.megapixels, 3),
        "gt_coverage": (
            round(float(np.isfinite(pair.gt).mean()), 4) if pair.has_gt else None
        ),
    }


def create_custom(
    name: str,
    left_bytes: bytes,
    right_bytes: bytes,
    *,
    ndisp: int = 128,
    gt_bytes: bytes | None = None,
    gt_filename: str | None = None,
) -> str:
    """Write an uploaded pair into ``data/custom/<name>/`` and return its id."""
    import cv2

    name = safe_name(name)
    folder = CUSTOM_ROOT / name
    if folder.exists():
        raise FileExistsError(f"a dataset named {name!r} already exists")

    def decode(raw: bytes, which: str) -> np.ndarray:
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"could not decode the {which} image")
        return img

    left, right = decode(left_bytes, "left"), decode(right_bytes, "right")
    if left.shape != right.shape:
        raise ValueError(
            f"the two images must have the same size; got {left.shape[1]}x"
            f"{left.shape[0]} and {right.shape[1]}x{right.shape[0]}"
        )

    folder.mkdir(parents=True)
    try:
        cv2.imwrite(str(folder / "im0.png"), left)
        cv2.imwrite(str(folder / "im1.png"), right)
        (folder / "calib.txt").write_text(
            f"width={left.shape[1]}\nheight={left.shape[0]}\nndisp={int(ndisp)}\n"
        )
        if gt_bytes:
            ext = Path(gt_filename or "disp0GT.pfm").suffix.lower()
            if ext == ".pfm":
                (folder / "disp0GT.pfm").write_bytes(gt_bytes)
            else:
                # A PNG/other depth image: read it and store as PFM so the
                # loader treats it as float disparity, with 0 meaning unknown.
                raw = cv2.imdecode(
                    np.frombuffer(gt_bytes, np.uint8), cv2.IMREAD_UNCHANGED
                )
                if raw is None:
                    raise ValueError("could not decode the ground-truth image")
                if raw.ndim == 3:
                    raw = raw[..., 0]
                from bench.datasets.middlebury import write_pfm

                disp = raw.astype(np.float32)
                disp[disp <= 0] = np.nan
                write_pfm(folder / "disp0GT.pfm", disp)
    except Exception:
        shutil.rmtree(folder, ignore_errors=True)
        raise

    _cache.clear()
    return f"custom/{name}"


def delete_custom(name: str) -> None:
    folder = CUSTOM_ROOT / safe_name(name)
    if not folder.is_dir():
        raise FileNotFoundError(f"no custom dataset named {name!r}")
    shutil.rmtree(folder)
    _cache.clear()
