"""Sample a few pairs from a rule set and lay them out for inspection.

The point of a preview is to answer "is this the data I meant?" before
spending an hour generating ten thousand of them. That question is not
answered by the left image alone: a rule set can produce a handsome picture
whose ground truth covers a third of the frame, or whose disparities never
exceed 6 px. So the contact sheet shows the left image, the right image, the
ground truth and the occlusion mask side by side, and the caller gets the
measured statistics alongside.
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from ..types import StereoPair
from .pipeline import generate_pair
from .rules import SceneRules

#: Views a contact sheet can show, in the order they are drawn.
PREVIEW_VIEWS = ("left", "right", "disparity", "nonocc")


def preview_pairs(
    rules: SceneRules,
    samples: int = 3,
    *,
    seed_offset: int = 0,
    progress: Callable[[int, int], None] | None = None,
) -> list[StereoPair]:
    """Generate a handful of example pairs from ``rules``.

    Uses the same seeds the real dataset would (``rules.seed + n``), so what
    you preview is literally the first ``samples`` pairs you will get.
    """
    out: list[StereoPair] = []
    for n in range(max(1, int(samples))):
        out.append(generate_pair(rules, int(rules.seed) + seed_offset + n))
        if progress:
            progress(n + 1, samples)
    return out


def colorize_disparity(
    disp: np.ndarray, vmin: float | None = None, vmax: float | None = None
) -> np.ndarray:
    """Disparity as an RGB image, with unknown pixels left dark.

    Unknown stays a flat near-black rather than being mapped to the low end of
    the colour map, so "no ground truth here" cannot be mistaken for "far
    away" -- a distinction that matters a great deal when you are eyeballing
    whether a rule set produces usable supervision.
    """
    # `matplotlib.colormaps[...]`, not the `cm.get_cmap` spelling: the latter
    # is deprecated and is removed in matplotlib 3.11.
    from matplotlib import colormaps

    d = np.asarray(disp, dtype=np.float32)
    finite = np.isfinite(d)
    if not finite.any():
        return np.zeros((*d.shape, 3), np.uint8)
    lo = float(np.nanpercentile(d[finite], 1)) if vmin is None else vmin
    hi = float(np.nanpercentile(d[finite], 99)) if vmax is None else vmax
    if hi <= lo:
        hi = lo + 1.0
    norm = np.clip((np.nan_to_num(d, nan=lo) - lo) / (hi - lo), 0, 1)
    rgb = (colormaps["turbo"](norm)[..., :3] * 255).astype(np.uint8)
    rgb[~finite] = (24, 24, 28)
    return rgb


def colorize_mask(mask: np.ndarray, under: np.ndarray) -> np.ndarray:
    """The occlusion mask over a dimmed copy of the image it belongs to."""
    base = (np.asarray(under, dtype=np.float32) * 0.45).astype(np.uint8)
    out = base.copy()
    m = np.asarray(mask, dtype=bool)
    # Visible-to-both stays in natural colour; occluded is tinted red, which
    # reads instantly as "no algorithm can get this right".
    out[m] = np.asarray(under, dtype=np.uint8)[m]
    red = np.zeros_like(out)
    red[..., 0] = 200
    out[~m] = (0.45 * out[~m] + 0.55 * red[~m]).astype(np.uint8)
    return out


def pair_views(pair: StereoPair) -> dict[str, np.ndarray]:
    """The four display images for one pair, as uint8 RGB."""
    disp = (
        colorize_disparity(pair.gt)
        if pair.has_gt
        else np.zeros((*pair.shape, 3), np.uint8)
    )
    mask = (
        colorize_mask(pair.occ_mask, pair.left)
        if pair.occ_mask is not None
        else np.zeros((*pair.shape, 3), np.uint8)
    )
    return {"left": pair.left, "right": pair.right, "disparity": disp, "nonocc": mask}


def contact_sheet(
    pairs: Sequence[StereoPair],
    *,
    views: Sequence[str] = PREVIEW_VIEWS,
    scale: float = 1.0,
    gap: int = 6,
    label: bool = True,
) -> np.ndarray:
    """Tile ``pairs`` into one image: one row per pair, one column per view."""
    import cv2

    if not pairs:
        raise ValueError("nothing to tile")

    tiles: list[list[np.ndarray]] = []
    for pair in pairs:
        available = pair_views(pair)
        row = []
        for v in views:
            img = available.get(v)
            if img is None:
                raise ValueError(f"unknown preview view {v!r}")
            if scale != 1.0:
                img = cv2.resize(
                    img, None, fx=scale, fy=scale,
                    interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
                )
            row.append(np.ascontiguousarray(img))
        tiles.append(row)

    th, tw = tiles[0][0].shape[:2]
    header = 22 if label else 0
    rows, cols = len(tiles), len(views)
    sheet = np.full(
        (header + rows * th + (rows - 1) * gap, cols * tw + (cols - 1) * gap, 3),
        18, np.uint8,
    )

    for c, v in enumerate(views):
        x = c * (tw + gap)
        if label:
            cv2.putText(
                sheet, v, (x + 4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (210, 210, 210), 1, cv2.LINE_AA,
            )
        for r in range(rows):
            y = header + r * (th + gap)
            sheet[y:y + th, x:x + tw] = tiles[r][c]
    return sheet


def preview_report(pairs: Sequence[StereoPair]) -> dict[str, Any]:
    """Statistics across the previewed pairs, plus a plain-language warning list.

    The warnings are the real value here. They catch the rule sets that render
    beautifully and are useless as supervision -- which is a mistake that is
    very easy to make and very hard to notice by looking at the pictures.
    """
    stats = [p.meta.get("stats", {}) for p in pairs]

    def mean(key: str) -> float | None:
        vals = [s[key] for s in stats if isinstance(s.get(key), (int, float))]
        return round(float(np.mean(vals)), 4) if vals else None

    summary = {k: mean(k) for k in (
        "gt_coverage", "nonocc_fraction", "occluded_fraction",
        "out_of_range_fraction", "disparity_min", "disparity_max",
        "disparity_mean", "sky_fraction", "edge_fraction", "baseline_m",
    )}

    warnings: list[str] = []
    ndisp = pairs[0].ndisp if pairs else 0
    dmax, cov = summary.get("disparity_max"), summary.get("gt_coverage")
    oor = summary.get("out_of_range_fraction") or 0.0

    if dmax is not None and ndisp and dmax < 0.35 * ndisp:
        warnings.append(
            f"Disparities only reach {dmax:.0f} px of the {ndisp} px search "
            f"range, so most of the range is never exercised. Raise "
            f"'disparity fill', or lower ndisp."
        )
    if oor > 0.02:
        warnings.append(
            f"{oor * 100:.1f}% of pixels exceed the {ndisp} px search range "
            f"and were dropped from the ground truth. Lower 'disparity fill' "
            f"or raise ndisp."
        )
    if cov is not None and cov < 0.55:
        warnings.append(
            f"Only {cov * 100:.0f}% of pixels have ground truth. That is "
            f"normal outdoors (sky), but wasteful indoors."
        )
    nonocc = summary.get("nonocc_fraction")
    if nonocc is not None and nonocc < 0.45:
        warnings.append(
            f"Only {nonocc * 100:.0f}% of pixels are visible to both cameras. "
            f"A large baseline relative to the scene makes most of the frame "
            f"unmatchable."
        )
    width = pairs[0].meta.get("rules", {}).get("width", 0) if pairs else 0
    if width and ndisp > 0.4 * width:
        warnings.append(
            f"The {ndisp} px search range is a large fraction of the "
            f"{width} px image width. The leftmost {ndisp} columns can never "
            f"be matched -- that is {ndisp / width * 100:.0f}% of every frame "
            f"with no supervision. Use a wider image or a smaller ndisp."
        )
    if pairs and pairs[0].meta.get("rules", {}).get("vertical_jitter_px", 0) > 0:
        warnings.append(
            "Rectification error is non-zero: the images violate the epipolar "
            "assumption that the ground truth is scored under. Useful for "
            "robustness training, misleading for benchmarking."
        )
    return {"summary": summary, "warnings": warnings, "per_sample": stats}
