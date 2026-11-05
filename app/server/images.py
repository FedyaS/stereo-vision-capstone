"""Render arrays as PNG bytes for the browser.

All disparity views are drawn on a **shared colour scale** taken from the
scene's ground truth, so two algorithms can be compared by eye without one of
them being rescaled into looking better. Where there is no ground truth the
scale comes from the prediction itself and is reported back, so the UI can
say what the colours mean.
"""
from __future__ import annotations

import io

import numpy as np

# Pixels the algorithm declined to estimate.
_NO_ESTIMATE = (24, 24, 28)
# Pixels with no ground truth, when drawing an error view.
_NO_GT = (70, 70, 76)
# "Wrong by more than the threshold", in the bad-pixel view.
_BAD = (227, 73, 72)
_GOOD = (248, 248, 246)


def _png(rgb: np.ndarray) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb.astype(np.uint8))).save(
        buf, format="PNG", optimize=False, compress_level=3
    )
    return buf.getvalue()


def scale_from(gt: np.ndarray | None, fallback: np.ndarray) -> tuple[float, float]:
    """The (vmin, vmax) every view of one scene should share."""
    source = gt if gt is not None and np.isfinite(gt).any() else fallback
    finite = np.isfinite(source)
    if not finite.any():
        return 0.0, 1.0
    lo = float(np.nanpercentile(source[finite], 1))
    hi = float(np.nanpercentile(source[finite], 99))
    return (lo, hi + 1.0) if hi <= lo else (lo, hi)


def image_png(img: np.ndarray) -> bytes:
    """A plain RGB or grayscale image."""
    a = np.asarray(img)
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    return _png(a)


def disparity_png(
    disp: np.ndarray, vmin: float, vmax: float, cmap: str = "turbo"
) -> bytes:
    """Colourised disparity; unestimated pixels stay dark, not zero-valued."""
    import matplotlib.cm as cm

    d = np.asarray(disp, dtype=np.float32)
    finite = np.isfinite(d)
    norm = np.clip((d - vmin) / max(vmax - vmin, 1e-6), 0, 1)
    rgb = (getattr(cm, cmap)(np.nan_to_num(norm))[..., :3] * 255).astype(np.uint8)
    rgb[~finite] = _NO_ESTIMATE
    return _png(rgb)


def error_png(
    disp: np.ndarray, gt: np.ndarray, max_error: float = 4.0
) -> bytes:
    """Absolute disparity error, clipped at ``max_error``.

    Grey means "no ground truth here", dark means "the method did not answer"
    -- the two are kept visually distinct because confusing them is how a
    sparse method gets mistaken for an accurate one.
    """
    import matplotlib.cm as cm

    d = np.asarray(disp, dtype=np.float32)
    g = np.asarray(gt, dtype=np.float32)
    err = np.abs(d - g)
    norm = np.clip(err / max(max_error, 1e-6), 0, 1)
    rgb = (cm.inferno(np.nan_to_num(norm))[..., :3] * 255).astype(np.uint8)
    rgb[~np.isfinite(g)] = _NO_GT
    rgb[~np.isfinite(d)] = _NO_ESTIMATE
    return _png(rgb)


def bad_png(
    disp: np.ndarray,
    gt: np.ndarray,
    left: np.ndarray,
    threshold: float = 2.0,
) -> bytes:
    """The Middlebury-style bad-pixel view: red where the error exceeds N px.

    Drawn over a faded copy of the left image so the errors can be read
    against the scene that caused them.
    """
    d = np.asarray(disp, dtype=np.float32)
    g = np.asarray(gt, dtype=np.float32)

    gray = np.asarray(left)
    if gray.ndim == 3:
        gray = gray.mean(axis=2)
    base = np.stack([gray] * 3, axis=-1).astype(np.float32)
    rgb = (base * 0.35 + np.array(_GOOD, np.float32) * 0.65).astype(np.uint8)

    known = np.isfinite(g)
    answered = np.isfinite(d)
    bad = known & answered & (np.abs(d - g) > threshold)
    rgb[known & ~answered] = _NO_ESTIMATE
    rgb[~known] = _NO_GT
    rgb[bad] = _BAD
    return _png(rgb)


def valid_png(valid: np.ndarray, left: np.ndarray) -> bytes:
    """Where the method produced an estimate at all."""
    gray = np.asarray(left)
    if gray.ndim == 3:
        gray = gray.mean(axis=2)
    base = np.stack([gray] * 3, axis=-1).astype(np.float32)
    rgb = (base * 0.35 + np.array(_GOOD, np.float32) * 0.65).astype(np.uint8)
    rgb[~np.asarray(valid, dtype=bool)] = _NO_ESTIMATE
    return _png(rgb)
