"""Scene discovery and custom datasets.

Three sources, all addressed by one id scheme so the UI never needs to care
which is which:

``middlebury/Teddy``   a downloaded benchmark scene
``synthetic/seed3``    generated on demand, exact ground truth, no download
``custom/<name>``      a folder the user added through the UI
``custom/<name>/<rel>`` one pair inside a dataset folder the user added

The same id scheme addresses whole *collections* -- a folder of many pairs
rather than one pair -- so ``custom/my-set`` or ``synth/room/test`` names a
dataset that :func:`expand` turns into the scene ids it contains. That is
what lets the UI offer "run on this dataset" beside "run on this pair".

Loaded pairs are cached, because every parameter tweak in the playground
re-runs matching on the same images and re-decoding a 741x497 PNG pair each
time is pure latency.
"""
from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from bench.datasets import list_middlebury_scenes, load_middlebury_scene
from bench.datasets.local import (
    find_pair_dirs,
    is_pair_dir,
    load_image_pair,
    pair_has_gt,
)
from bench.datasets.synthetic import make_synthetic_pair
from bench.paths import CUSTOM_DATA_DIR as CUSTOM_ROOT
from bench.paths import MIDDLEBURY_DIR
from bench.paths import REPO_ROOT
from bench.paths import SYNTH_DATA_DIR
from bench.paths import describe as _describe_path
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
    """Names of single-pair datasets directly under ``data/custom``.

    A folder holding *many* pairs is not one of these -- it is a collection,
    listed by :func:`list_collections` -- so the Datasets tab can keep
    offering "delete this pair" for uploads without offering it for a
    hundred-pair set the user copied in.
    """
    if not CUSTOM_ROOT.is_dir():
        return []
    return sorted(
        d.name for d in CUSTOM_ROOT.iterdir() if d.is_dir() and is_pair_dir(d)
    )


#: How many samples of each dataset folder to expose as individually
#: selectable scenes. A generated set can hold thousands of pairs, and a
#: picker listing all of them is unusable; selecting the *collection* and
#: running a batch over it is what the Runs tab is for.
SYNTH_SAMPLES_LISTED = 12

def _collection_roots() -> dict[str, Path]:
    """Where each collection kind lives on disk.

    A function rather than a module-level dict because the path constants are
    read at call time on purpose (see :mod:`bench.paths`): a test redirects
    ``SYNTH_DATA_DIR`` with ``monkeypatch.setattr`` and a dict built at import
    time would never see it.
    """
    return {"custom": CUSTOM_ROOT, "synth": SYNTH_DATA_DIR}


def list_synth_datasets() -> list[str]:
    """Names of generated datasets under ``data/synth``."""
    root = SYNTH_DATA_DIR
    if not root.is_dir():
        return []
    return sorted(
        d.name for d in root.iterdir()
        if d.is_dir() and (d / "manifest.json").is_file()
    )


def _resolve_within(root: Path, rest: str) -> Path:
    """Turn the part of an id after the kind into a path inside ``root``.

    ``safe_name`` guards the first segment and the containment check guards
    the rest, so a crafted id such as ``custom/x/../../../etc`` cannot
    address anything outside the dataset root.
    """
    parts = [
        part for part in PurePosixPath(rest).parts if part not in ("", ".", "/")
    ]
    if not parts or any(part == ".." for part in parts):
        raise ValueError(f"bad dataset path {rest!r}")
    safe_name(parts[0])
    folder = root.joinpath(*parts)
    if not folder.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"bad dataset path {rest!r}")
    return folder


def _collection_row(
    kind: str, root: Path, folder: Path, dirs: list[Path]
) -> dict[str, Any]:
    """One row of the collection listing, built without decoding any image.

    Counting pairs and reading one ``calib.txt`` is a few hundred stat calls;
    decoding a pair is tens of milliseconds. A dataset list has to stay
    instant even when a set holds thousands of samples, so nothing here opens
    an image.
    """
    rel = folder.relative_to(root).as_posix()
    groups: dict[str, int] = {}
    for d in dirs:
        parent = d.parent
        key = parent.relative_to(folder).as_posix() if parent != folder else ""
        groups[key] = groups.get(key, 0) + 1
    ndisp = None
    if dirs:
        calib = dirs[0] / "calib.txt"
        if calib.is_file():
            from bench.datasets.middlebury import parse_calib

            try:
                ndisp = int(parse_calib(calib).get("ndisp") or 0) or None
            except Exception:
                ndisp = None
    return {
        "id": f"{kind}/{rel}" if rel else kind,
        "kind": kind,
        "name": rel or kind,
        "count": len(dirs),
        # Only meaningful when the samples sit in subfolders; a flat set
        # reports {} rather than {"": n}, which would read in the UI as a
        # split named after the empty string.
        "splits": {k: v for k, v in sorted(groups.items()) if k},
        "has_gt": bool(dirs) and pair_has_gt(dirs[0]),
        "ndisp": ndisp,
        "path": _describe_path(folder),
    }


def list_collections() -> list[dict[str, Any]]:
    """Every dataset that holds more than one pair, plus its subsets.

    A collection is any folder under ``data/custom`` or ``data/synth`` with
    stereo pairs below it, at whatever depth -- which is the whole point: a
    dataset copied in by hand is addressable without having to be reshaped
    into the layout the uploader happens to write.

    For each one this reports the whole set *and* each group of samples
    beneath it, so a generated set with ``train``/``val``/``test`` offers all
    four as separate things to run on. Reporting numbers on the subset you
    also tuned on is the easiest way to fool yourself, so the split has to be
    selectable, not just the whole.

    Built-in sources are listed the same way, so "all Middlebury scenes" is
    one checkbox rather than fifteen.
    """
    out: list[dict[str, Any]] = []

    scenes = list_middlebury_scenes()
    if scenes:
        out.append({
            "id": "middlebury",
            "kind": "middlebury",
            "name": "all Middlebury scenes",
            "count": len(scenes),
            "splits": {},
            "has_gt": True,
            "ndisp": None,
            "path": _describe_path(MIDDLEBURY_DIR / "trainingQ"),
        })

    seeds = list(_DEFAULT_SEEDS) + sorted(_extra_seeds)
    out.append({
        "id": "synthetic",
        "kind": "synthetic",
        "name": "warp-based synthetic pairs",
        "count": len(seeds),
        "splits": {},
        "has_gt": True,
        "ndisp": 64,
        "path": "generated in memory",
    })

    for kind, root in _collection_roots().items():
        if not root.is_dir():
            continue
        by_id: dict[str, dict[str, Any]] = {}
        for top in sorted(d for d in root.iterdir() if d.is_dir()):
            if top.name.startswith(".") or is_pair_dir(top):
                continue  # a single uploaded pair, not a collection
            dirs = find_pair_dirs(top)
            if not dirs:
                continue
            row = _collection_row(kind, root, top, dirs)
            by_id[row["id"]] = row
            # Each group of samples as its own selectable subset. For a flat
            # set that group is the set itself, which dedupes on the id.
            for group in sorted({d.parent for d in dirs}):
                if group == top:
                    continue
                members = [d for d in dirs if d.parent == group]
                sub = _collection_row(kind, root, group, members)
                by_id.setdefault(sub["id"], sub)
        out.extend(by_id[k] for k in sorted(by_id))

    return out


def expand(dataset_id: str, *, limit: int | None = None) -> list[str]:
    """The scene ids a dataset id stands for, in a stable order.

    A plain scene id expands to itself, so callers never need to know whether
    what the user picked was one pair or a thousand.
    """
    kind, _, rest = dataset_id.partition("/")
    if kind == "middlebury":
        names = [rest] if rest else list_middlebury_scenes()
        ids = [f"middlebury/{n}" for n in names]
    elif kind == "synthetic":
        if rest:
            ids = [dataset_id]
        else:
            seeds = list(_DEFAULT_SEEDS) + sorted(_extra_seeds)
            ids = [f"synthetic/seed{i}" for i in seeds]
    elif kind in _collection_roots():
        root = _collection_roots()[kind]
        folder = _resolve_within(root, rest)
        if not folder.is_dir():
            raise FileNotFoundError(f"no dataset at {dataset_id!r}")
        dirs = find_pair_dirs(folder, limit=limit)
        if not dirs:
            raise FileNotFoundError(
                f"{dataset_id!r} holds no stereo pair. A pair is a folder "
                f"with im0/im1 (or left/right) images in it."
            )
        ids = [f"{kind}/{d.relative_to(root).as_posix()}" for d in dirs]
    else:
        raise ValueError(f"unknown dataset id {dataset_id!r}")
    return ids[:limit] if limit else ids


def resolve_selection(
    scenes: list[str] | None = None,
    datasets: list[str] | None = None,
    *,
    limit: int | None = None,
) -> list[str]:
    """Flatten a mixed selection of pairs and whole datasets into scene ids.

    Order is the order picked and duplicates are dropped: selecting both a
    dataset and one of its pairs must not benchmark that pair twice.
    """
    out: list[str] = []
    seen: set[str] = set()
    for sid in list(scenes or []):
        if sid not in seen:
            seen.add(sid)
            out.append(sid)
    for did in list(datasets or []):
        for sid in expand(did, limit=limit):
            if sid not in seen:
                seen.add(sid)
                out.append(sid)
    return out


def scene_ids() -> list[str]:
    """Individually selectable scenes.

    Dataset folders contribute only their first :data:`SYNTH_SAMPLES_LISTED`
    samples. Listing every sample of a 20,000-pair set would have this
    endpoint decode 20,000 images to describe them, and produce a picker
    nobody can scroll; the whole set stays selectable as a collection.
    """
    ids = [f"middlebury/{s}" for s in list_middlebury_scenes()]
    ids += [f"custom/{s}" for s in list_custom()]
    for kind, root in _collection_roots().items():
        if not root.is_dir():
            continue
        for top in sorted(d for d in root.iterdir() if d.is_dir()):
            if top.name.startswith(".") or is_pair_dir(top):
                continue
            for d in find_pair_dirs(top, limit=SYNTH_SAMPLES_LISTED):
                ids.append(f"{kind}/{d.relative_to(root).as_posix()}")
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
        # ``custom/<name>`` for an uploaded pair, ``custom/<name>/<rel>`` for
        # one sample of a dataset folder the user copied in.
        folder = _resolve_within(CUSTOM_ROOT, rest)
        if not folder.is_dir():
            raise FileNotFoundError(f"no custom dataset named {rest!r}")
        pair = load_image_pair(folder, name=scene_id)
    elif kind == "synth":
        # synth/<dataset>/<sample> or synth/<dataset>/<split>/<sample>,
        # written by the synthesizer in the Middlebury layout, so the
        # ordinary folder loader reads it.
        folder = _resolve_within(SYNTH_DATA_DIR, rest)
        if not folder.is_dir():
            raise FileNotFoundError(f"no generated sample {rest!r}")
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
    kind, _, rest = scene_id.partition("/")
    return {
        "id": scene_id,
        "kind": kind,
        # Everything after the kind, so the twelve listed samples of two
        # different generated datasets do not all show up as "00000".
        "name": rest or scene_id,
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
