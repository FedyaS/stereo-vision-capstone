"""Middlebury MiddEval3 loader.

The quarter-resolution training split ships 15 scenes with dense float
ground truth (``disp0GT.pfm``), a non-occluded mask (``mask0nocc.png``) and a
calibration file that states the disparity range the scene needs (``ndisp``).
Using the scene's own ``ndisp`` matters: Jadeplant needs 160 disparities while
Teddy needs 64, and giving every method the same oversized range would make
the easy scenes look needlessly slow.

Fetch the data with ``python scripts/get_dataset.py``.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from ..paths import MIDDLEBURY_DIR as DEFAULT_MIDDLEBURY_ROOT
from ..types import StereoPair

# Mask values in mask0nocc.png.
_NONOCC = 255


def read_pfm(path: str | Path) -> np.ndarray:
    """Read a PFM file (as used for Middlebury ground truth).

    Returns a float32 array in normal top-down row order, with ``inf``
    preserved -- Middlebury writes ``inf`` for "no ground truth here".
    """
    with open(path, "rb") as f:
        header = f.readline().rstrip()
        if header not in (b"Pf", b"PF"):
            raise ValueError(f"{path}: not a PFM file (header {header!r})")
        channels = 3 if header == b"PF" else 1

        # Skip comment lines, then read "width height".
        line = f.readline()
        while line.startswith(b"#"):
            line = f.readline()
        m = re.match(rb"^\s*(\d+)\s+(\d+)\s*$", line)
        if not m:
            raise ValueError(f"{path}: bad PFM dimensions line {line!r}")
        width, height = int(m.group(1)), int(m.group(2))

        scale = float(f.readline().rstrip())
        dtype = "<f4" if scale < 0 else ">f4"
        data = np.frombuffer(f.read(width * height * channels * 4), dtype=dtype)

    data = data.reshape(height, width, channels) if channels == 3 else data.reshape(
        height, width
    )
    # PFM stores rows bottom-to-top.
    return np.ascontiguousarray(data[::-1]).astype(np.float32)


def write_pfm(path: str | Path, image: np.ndarray, scale: float = 1.0) -> None:
    """Write a single-channel float array as a little-endian PFM."""
    img = np.asarray(image, dtype=np.float32)
    if img.ndim != 2:
        raise ValueError("write_pfm expects a 2-D array")
    with open(path, "wb") as f:
        f.write(b"Pf\n")
        f.write(f"{img.shape[1]} {img.shape[0]}\n".encode())
        f.write(f"{-abs(scale)}\n".encode())
        f.write(np.ascontiguousarray(img[::-1]).astype("<f4").tobytes())


def parse_calib(path: str | Path) -> dict[str, float | str]:
    """Parse MiddEval3 ``calib.txt`` into a dict of floats where possible."""
    out: dict[str, float | str] = {}
    for line in Path(path).read_text().splitlines():
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        try:
            out[key.strip()] = float(value)
        except ValueError:
            out[key.strip()] = value
    return out


def list_middlebury_scenes(
    root: str | Path = DEFAULT_MIDDLEBURY_ROOT, split: str = "trainingQ"
) -> list[str]:
    """Names of scenes in ``split`` that have both images and ground truth."""
    base = Path(root) / split
    if not base.is_dir():
        return []
    return sorted(
        d.name
        for d in base.iterdir()
        if d.is_dir() and (d / "im0.png").exists() and (d / "disp0GT.pfm").exists()
    )


def load_middlebury_scene(
    scene: str,
    root: str | Path = DEFAULT_MIDDLEBURY_ROOT,
    split: str = "trainingQ",
    *,
    ndisp: int | None = None,
) -> StereoPair:
    """Load one MiddEval3 scene as a :class:`StereoPair`."""
    import cv2

    d = Path(root) / split / scene
    if not d.is_dir():
        raise FileNotFoundError(
            f"Middlebury scene not found: {d}\n"
            "Run `python scripts/get_dataset.py` to download the dataset."
        )

    left = cv2.cvtColor(cv2.imread(str(d / "im0.png"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    right = cv2.cvtColor(cv2.imread(str(d / "im1.png"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    gt = None
    gt_path = d / "disp0GT.pfm"
    if gt_path.exists():
        gt = read_pfm(gt_path)
        gt = np.where(np.isfinite(gt), gt, np.nan).astype(np.float32)

    occ_mask = None
    mask_path = d / "mask0nocc.png"
    if mask_path.exists():
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        occ_mask = mask == _NONOCC

    calib: dict[str, float | str] = {}
    calib_path = d / "calib.txt"
    if calib_path.exists():
        calib = parse_calib(calib_path)

    scene_ndisp = int(ndisp or calib.get("ndisp", 128) or 128)
    focal = None
    cam0 = calib.get("cam0")
    if isinstance(cam0, str):
        nums = re.findall(r"[-+]?\d*\.?\d+", cam0)
        if nums:
            focal = float(nums[0])

    return StereoPair(
        name=f"middlebury/{scene}",
        left=left,
        right=right,
        ndisp=scene_ndisp,
        gt=gt,
        occ_mask=occ_mask,
        focal_length_px=focal,
        baseline_mm=float(calib["baseline"]) if "baseline" in calib else None,
        doffs=float(calib.get("doffs", 0.0) or 0.0),
        meta={"split": split, "calib": calib, "source": str(d)},
    )
