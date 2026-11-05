"""Accuracy metrics for disparity maps.

The headline numbers follow the Middlebury stereo convention: ``bad N`` is the
percentage of evaluated pixels whose absolute disparity error exceeds ``N``
pixels, and the default evaluation region is the non-occluded area.

One wrinkle matters for cross-method comparison. Some matchers (StereoBM,
QuasiDenseStereo) return sparse maps with large holes; others return a value
everywhere. Scoring only where a method chose to answer rewards methods that
answer rarely. So every error metric is reported twice:

``*_valid``
    computed only over pixels where the method produced an estimate.
``bad*_all`` / ``*_filled``
    computed over the whole evaluation region, with missing estimates either
    counted as errors (``bad*_all``) or hole-filled first (``*_filled``).

Read ``density`` alongside the ``*_valid`` columns, and treat ``bad2_all`` as
the single most comparable accuracy number.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

# Error thresholds (in pixels) for the "bad" metric.
BAD_THRESHOLDS: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)


@dataclass
class AccuracyMetrics:
    """Accuracy of one disparity map against ground truth."""

    # --- comparability-first metrics (whole evaluation region) -------------
    bad_all: dict[str, float] = field(default_factory=dict)
    epe_filled: float = float("nan")
    rmse_filled: float = float("nan")

    # --- metrics over the pixels the method actually answered --------------
    bad_valid: dict[str, float] = field(default_factory=dict)
    epe_valid: float = float("nan")
    rmse_valid: float = float("nan")
    bias_valid: float = float("nan")
    a90: float = float("nan")
    a95: float = float("nan")
    a99: float = float("nan")

    # --- coverage ----------------------------------------------------------
    density: float = float("nan")
    n_eval_px: int = 0

    def flat(self) -> dict[str, float]:
        """Flatten into a single-level dict suitable for a dataframe row.

        The two ``bad`` dicts become ``bad<threshold>_all`` and
        ``bad<threshold>_valid`` columns, e.g. ``bad2_all``, ``bad0p5_valid``.
        """
        out: dict[str, float] = {}
        for field_name, value in asdict(self).items():
            if field_name == "bad_all":
                out.update({f"bad{t}_all": v for t, v in value.items()})
            elif field_name == "bad_valid":
                out.update({f"bad{t}_valid": v for t, v in value.items()})
            else:
                out[field_name] = value
        return out


def fill_holes(disparity: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Fill invalid pixels by nearest valid neighbour along each scanline.

    This is the standard cheap densification used when a sparse map has to be
    scored against a dense benchmark: scan left and right along the row and
    take the smaller (further-away) of the two bracketing disparities, which
    is the usual choice because holes mostly sit in occluded background.
    """
    d = np.asarray(disparity, dtype=np.float32).copy()
    v = np.asarray(valid, dtype=bool)
    if v.all():
        return d
    if not v.any():
        return np.zeros_like(d)

    h, w = d.shape
    idx = np.where(v, np.arange(w)[None, :], -1)
    # Nearest valid index to the left (running max of valid indices).
    left_idx = np.maximum.accumulate(idx, axis=1)
    # Nearest valid index to the right (reverse running min over valid-only).
    idx_r = np.where(v, np.arange(w)[None, :], w)
    right_idx = np.minimum.accumulate(idx_r[:, ::-1], axis=1)[:, ::-1]

    rows = np.arange(h)[:, None]
    has_l, has_r = left_idx >= 0, right_idx < w
    dl = np.where(has_l, d[rows, np.clip(left_idx, 0, w - 1)], np.inf)
    dr = np.where(has_r, d[rows, np.clip(right_idx, 0, w - 1)], np.inf)
    filled = np.minimum(dl, dr)

    # Rows with no valid pixel at all: fall back to the global median.
    filled = np.where(np.isfinite(filled), filled, float(np.nanmedian(d[v])))
    return np.where(v, d, filled).astype(np.float32)


def evaluate(
    disparity: np.ndarray,
    valid: np.ndarray,
    gt: np.ndarray,
    *,
    eval_mask: np.ndarray | None = None,
    thresholds: tuple[float, ...] = BAD_THRESHOLDS,
) -> AccuracyMetrics:
    """Score ``disparity`` against ``gt``.

    Parameters
    ----------
    disparity, valid:
        Prediction and its validity mask, as returned by
        :meth:`bench.types.DisparityResult.normalized`.
    gt:
        Ground-truth disparity, ``nan``/``inf`` where unknown.
    eval_mask:
        Restricts the evaluation region, e.g. the non-occluded mask. Combined
        with the "ground truth is known" mask.
    """
    disparity = np.asarray(disparity, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    gt = np.asarray(gt, dtype=np.float32)

    region = np.isfinite(gt)
    if eval_mask is not None:
        region &= np.asarray(eval_mask, dtype=bool)

    m = AccuracyMetrics()
    m.n_eval_px = int(region.sum())
    if m.n_eval_px == 0:
        return m

    answered = region & valid
    m.density = float(answered.sum()) / float(m.n_eval_px)

    # --- over answered pixels ---------------------------------------------
    if answered.any():
        err = np.abs(disparity[answered] - gt[answered])
        signed = disparity[answered] - gt[answered]
        m.epe_valid = float(err.mean())
        m.rmse_valid = float(np.sqrt((err.astype(np.float64) ** 2).mean()))
        m.bias_valid = float(signed.mean())
        m.a90, m.a95, m.a99 = (float(x) for x in np.percentile(err, [90, 95, 99]))
        m.bad_valid = {
            _fmt(t): float((err > t).mean() * 100.0) for t in thresholds
        }

    # --- over the whole region, unanswered counted as bad ------------------
    err_all = np.where(
        answered, np.abs(disparity - gt), np.inf
    )[region]
    m.bad_all = {_fmt(t): float((err_all > t).mean() * 100.0) for t in thresholds}

    # --- over the whole region, holes filled first -------------------------
    filled = fill_holes(disparity, valid)
    err_f = np.abs(filled[region] - gt[region]).astype(np.float64)
    m.epe_filled = float(err_f.mean())
    m.rmse_filled = float(np.sqrt((err_f**2).mean()))
    return m


def _fmt(t: float) -> str:
    return f"{t:g}".replace(".", "p")


def colorize(
    disparity: np.ndarray,
    vmin: float | None = None,
    vmax: float | None = None,
) -> np.ndarray:
    """Render a disparity map to an RGB uint8 image for figures."""
    import matplotlib.cm as cm

    d = np.asarray(disparity, dtype=np.float32)
    finite = np.isfinite(d)
    if not finite.any():
        return np.zeros((*d.shape, 3), np.uint8)
    lo = float(np.nanpercentile(d[finite], 1)) if vmin is None else vmin
    hi = float(np.nanpercentile(d[finite], 99)) if vmax is None else vmax
    if hi <= lo:
        hi = lo + 1.0
    norm = np.clip((d - lo) / (hi - lo), 0, 1)
    rgb = (cm.turbo(np.nan_to_num(norm))[..., :3] * 255).astype(np.uint8)
    rgb[~finite] = 0  # holes in black
    return rgb
