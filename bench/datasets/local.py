"""Loader for a plain folder of rectified images on disk.

Expects ``im0``/``im1`` (Middlebury naming) or ``left``/``right``, in any
common image format. Ground truth is picked up if ``disp0GT.pfm`` or
``disp0.pfm`` sits alongside. Without ground truth you still get timing and
resource numbers plus the output maps -- useful for the full-resolution pair
already in ``practice/``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..types import StereoPair
from .middlebury import parse_calib, read_pfm

_LEFT_NAMES = ("im0", "left", "imL", "view1")
_RIGHT_NAMES = ("im1", "right", "imR", "view5")
_EXTS = (".png", ".jpg", ".jpeg", ".ppm", ".pgm", ".tif", ".tiff", ".bmp")


def _find(folder: Path, stems: tuple[str, ...]) -> Path:
    for stem in stems:
        for ext in _EXTS:
            p = folder / f"{stem}{ext}"
            if p.exists():
                return p
    raise FileNotFoundError(
        f"{folder}: no image matching {stems} with extension in {_EXTS}"
    )


def load_image_pair(
    folder: str | Path,
    *,
    ndisp: int | None = None,
    downscale: float = 1.0,
    name: str | None = None,
) -> StereoPair:
    """Load a stereo pair from ``folder``.

    ``downscale`` divides both image dimensions, and scales the disparity
    range and ground truth with them -- handy for the 2964x2000 pair in
    ``practice/``, where a from-scratch matcher at full size would take a very
    long time.
    """
    import cv2

    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"{folder} is not a directory")

    left = cv2.cvtColor(
        cv2.imread(str(_find(folder, _LEFT_NAMES)), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
    )
    right = cv2.cvtColor(
        cv2.imread(str(_find(folder, _RIGHT_NAMES)), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
    )

    calib: dict = {}
    if (folder / "calib.txt").exists():
        calib = parse_calib(folder / "calib.txt")

    gt = None
    for gt_name in ("disp0GT.pfm", "disp0.pfm"):
        if (folder / gt_name).exists():
            raw = read_pfm(folder / gt_name)
            gt = np.where(np.isfinite(raw), raw, np.nan).astype(np.float32)
            break

    occ_mask = None
    if (folder / "mask0nocc.png").exists():
        mask = cv2.imread(str(folder / "mask0nocc.png"), cv2.IMREAD_GRAYSCALE)
        occ_mask = mask == 255

    n = int(ndisp or calib.get("ndisp", 128) or 128)

    if downscale and downscale != 1.0:
        s = 1.0 / float(downscale)
        interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
        new_size = (
            max(1, int(round(left.shape[1] * s))),
            max(1, int(round(left.shape[0] * s))),
        )
        left = cv2.resize(left, new_size, interpolation=interp)
        right = cv2.resize(right, new_size, interpolation=interp)
        if gt is not None:
            # Nearest keeps nan holes from bleeding; disparity scales with width.
            gt = cv2.resize(gt, new_size, interpolation=cv2.INTER_NEAREST) * s
        if occ_mask is not None:
            occ_mask = (
                cv2.resize(
                    occ_mask.astype(np.uint8), new_size, interpolation=cv2.INTER_NEAREST
                )
                > 0
            )
        n = max(16, int(round(n * s)))

    return StereoPair(
        name=name or f"local/{folder.name}",
        left=left,
        right=right,
        ndisp=n,
        gt=gt,
        occ_mask=occ_mask,
        doffs=float(calib.get("doffs", 0.0) or 0.0),
        meta={"source": str(folder), "downscale": downscale, "calib": calib},
    )
