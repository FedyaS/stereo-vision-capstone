"""Stereo matching written from scratch on NumPy/SciPy.

These exist so the pipeline is open to inspection rather than hidden behind a
library call: every stage -- matching cost, aggregation, winner selection,
sub-pixel fit, consistency check -- is a separate function you can swap or
plot. They are the ones to modify when the question is "what does changing
the cost function do?" rather than "what is the fastest matcher?".

Pipeline
--------
1. **Matching cost** (:func:`cost_volume`): a ``(D, H, W)`` volume holding the
   dissimilarity of left pixel ``(y, x)`` against right pixel ``(y, x - d)``.
   Four cost functions: ``sad``, ``ssd``, ``zncc``, ``census``.
2. **Aggregation**: none (winner-takes-all on the box-filtered cost),
   :func:`sgm_aggregate` (semi-global, 4 or 8 paths), or
   :func:`potts_scanline_dp` (exact per-scanline optimum).
3. **Selection** (:func:`winner_takes_all`): arg-min with an optional
   parabolic sub-pixel fit and a uniqueness ratio test.
4. **Validation** (:func:`left_right_check`): recompute disparity for the
   right view from the same volume and discard disagreements.

A note on cost: the volume is ``D * H * W`` float32. At Middlebury quarter
resolution that is ~200 MB for the widest scene, which is why
``max_volume_gb`` exists -- it raises :class:`VolumeTooLarge` rather than
letting the machine swap.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..types import DisparityResult, StereoPair
from .base import register, to_gray

_FAMILY = "scratch"

# Cost assigned where no match can exist (x - d falls outside the image).
# Large but finite, so the aggregation recursions stay numerically sane.
BIG = np.float32(1e4)

# Aggregation penalties are only meaningful relative to the cost function's
# own scale, and SAD-over-colour (0..765), ZNCC (0..2) and census Hamming
# (0..63) are three very different scales. Rather than tune P1/P2 per metric,
# the cost volume is rescaled to 0..COST_SCALE before aggregation, which lets
# the literature's classic P1=10 / P2=120 apply to all of them.
COST_SCALE = np.float32(64.0)


class VolumeTooLarge(RuntimeError):
    """Raised when the requested cost volume would not fit in memory."""


def _check_numpy() -> tuple[bool, str]:
    try:
        import cv2  # noqa: F401
        import numpy  # noqa: F401
    except Exception as exc:
        return False, f"numpy/opencv not importable: {exc}"
    return True, ""


# --------------------------------------------------------------------------
# 1. matching cost
# --------------------------------------------------------------------------


def _box(img: np.ndarray, size: int) -> np.ndarray:
    """Unnormalised box filter -- a window sum. Separable, so O(1) per pixel."""
    import cv2

    return cv2.boxFilter(
        img, ddepth=-1, ksize=(size, size), normalize=False, borderType=cv2.BORDER_REFLECT
    )


def census_transform(gray: np.ndarray, win_h: int = 7, win_w: int = 9) -> np.ndarray:
    """Census transform: each pixel becomes a bit-string of neighbour comparisons.

    Bit ``i`` is 1 when the ``i``-th neighbour is brighter than the centre.
    Because only the *ordering* of intensities is kept, the result is invariant
    to any monotonic change in brightness or gain -- which is why census costs
    hold up where SAD collapses on a pair with mismatched exposure (try it on
    the Middlebury ``*E`` and ``*L`` scenes, which are deliberately
    mis-exposed).

    ``win_h * win_w - 1`` must be at most 64 so the code fits in a ``uint64``.
    """
    if win_h * win_w - 1 > 64:
        raise ValueError("census window has more than 64 neighbours")
    g = np.asarray(gray, dtype=np.int16)
    h, w = g.shape
    padded = np.pad(g, ((win_h // 2,), (win_w // 2,)), mode="symmetric")

    code = np.zeros((h, w), dtype=np.uint64)
    bit = 0
    for dy in range(win_h):
        for dx in range(win_w):
            if dy == win_h // 2 and dx == win_w // 2:
                continue
            neigh = padded[dy : dy + h, dx : dx + w]
            code |= (neigh > g).astype(np.uint64) << np.uint64(bit)
            bit += 1
    return code


def _popcount(x: np.ndarray) -> np.ndarray:
    """Bits set per element, as float32."""
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x).astype(np.float32)
    # Fallback: byte-wise lookup table.
    lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.float32)
    view = x.view(np.uint8).reshape(*x.shape, 8)
    return lut[view].sum(axis=-1, dtype=np.float32)


def _shift_right(img: np.ndarray, d: int) -> np.ndarray:
    """Shift ``img`` right by ``d`` columns, so ``out[:, x] == img[:, x - d]``.

    The vacated columns ``0..d-1`` are filled by replicating column 0 rather
    than with zeros. They are outside the image, so any choice is arbitrary,
    but zeros are the one actively harmful choice: a window straddling the
    boundary would then compare real pixels against black and report a huge
    dissimilarity, which leaks into the ``r`` columns just inside the valid
    region (``cost[d, :, :d] = BIG`` only covers ``x < d``, not the window
    radius beyond it). Edge replication keeps those costs meaningful.
    """
    if d == 0:
        return img
    out = np.empty_like(img)
    out[:, d:] = img[:, :-d]
    out[:, :d] = img[:, :1]
    return out


def cost_volume(
    left: np.ndarray,
    right: np.ndarray,
    ndisp: int,
    *,
    metric: str = "sad",
    block: int = 9,
    census_win: tuple[int, int] = (7, 9),
    max_volume_gb: float = 4.0,
) -> np.ndarray:
    """Build the ``(D, H, W)`` float32 matching-cost volume.

    ``cost[d, y, x]`` is the dissimilarity between left pixel ``(y, x)`` and
    right pixel ``(y, x - d)``, aggregated over a ``block x block`` window.
    Where ``x - d < 0`` there is no candidate at all, and the cost is set to
    :data:`BIG` rather than to zero -- otherwise the left border would look
    like a perfect match at every large disparity.
    """
    h, w = left.shape[:2]
    nbytes = ndisp * h * w * 4
    if nbytes / 1e9 > max_volume_gb:
        raise VolumeTooLarge(
            f"cost volume would be {nbytes / 1e9:.1f} GB "
            f"(D={ndisp}, {w}x{h}); limit is {max_volume_gb} GB"
        )

    cost = np.empty((ndisp, h, w), dtype=np.float32)
    n_win = float(block * block)

    if metric in ("sad", "ssd"):
        lf = np.asarray(left, dtype=np.float32)
        rf = np.asarray(right, dtype=np.float32)
        for d in range(ndisp):
            diff = lf - _shift_right(rf, d)
            per_px = np.abs(diff) if metric == "sad" else diff * diff
            if per_px.ndim == 3:
                per_px = per_px.sum(axis=2)
            cost[d] = _box(per_px, block) / n_win

    elif metric == "zncc":
        lf = np.asarray(to_gray(left), dtype=np.float32)
        rf = np.asarray(to_gray(right), dtype=np.float32)
        # Left-image window statistics do not depend on d.
        mu_l = _box(lf, block) / n_win
        var_l = np.maximum(_box(lf * lf, block) / n_win - mu_l * mu_l, 0.0)
        for d in range(ndisp):
            rs = _shift_right(rf, d)
            mu_r = _box(rs, block) / n_win
            var_r = np.maximum(_box(rs * rs, block) / n_win - mu_r * mu_r, 0.0)
            cross = _box(lf * rs, block) / n_win - mu_l * mu_r
            denom = np.sqrt(var_l * var_r) + 1e-6
            # 1 - ZNCC maps perfect correlation to 0 and anti-correlation to 2.
            cost[d] = 1.0 - np.clip(cross / denom, -1.0, 1.0)

    elif metric == "census":
        cl = census_transform(to_gray(left), *census_win)
        cr = census_transform(to_gray(right), *census_win)
        for d in range(ndisp):
            ham = _popcount(cl ^ _shift_right(cr, d))
            cost[d] = _box(ham, block) / n_win

    else:
        raise ValueError(f"unknown cost metric {metric!r}")

    # Invalidate the strip where no right-image candidate exists.
    for d in range(1, ndisp):
        cost[d, :, :d] = BIG
    return cost


def normalize_cost(cost: np.ndarray, scale: float = float(COST_SCALE)) -> np.ndarray:
    """Robustly rescale a cost volume to ``0..scale``, preserving :data:`BIG`.

    Uses the 1st and 99th percentiles of the valid costs so a handful of
    extreme values cannot squash the useful range. Run this before
    :func:`sgm_aggregate` or :func:`potts_scanline_dp` so their penalties mean
    the same thing whatever cost function produced the volume.
    """
    valid = cost < BIG * 0.5
    sample = cost[valid]
    if sample.size == 0:
        return cost
    # Percentiles on a 200 MB volume are slow and unnecessary; subsample.
    stride = max(1, sample.size // 200_000)
    lo, hi = np.percentile(sample[::stride], [1.0, 99.0])
    if hi <= lo:
        hi = lo + 1.0
    out = np.clip((cost - lo) / (hi - lo), 0.0, 1.0) * np.float32(scale)
    return np.where(valid, out, BIG).astype(np.float32)


def geometric_validity(disp: np.ndarray) -> np.ndarray:
    """True where the matched right-image point ``x - d`` is inside the image.

    This is the validity test to use *after* aggregation. Testing "is the
    winning cost small?" works on a raw cost volume but not on an aggregated
    one, where costs are sums along several paths and :data:`BIG` has been
    smeared across neighbours -- which silently drops most of the image.
    """
    w = disp.shape[1]
    xx = np.arange(w, dtype=np.float32)[None, :]
    return np.isfinite(disp) & (disp <= xx) & (disp >= 0)


# --------------------------------------------------------------------------
# 2. aggregation
# --------------------------------------------------------------------------


def _aggregate_direction(
    cost: np.ndarray, p1: float, p2: float, dy: int, dx: int
) -> np.ndarray:
    """One SGM path: the Hirschmuller recursion along direction ``(dy, dx)``.

    ``L(p, d) = C(p, d) + min( L(p-r, d),
                               L(p-r, d-1) + P1,
                               L(p-r, d+1) + P1,
                               min_k L(p-r, k) + P2 ) - min_k L(p-r, k)``

    The trailing subtraction is the standard trick that keeps the running
    sum bounded; it adds the same constant to every disparity at a pixel, so
    the arg-min is unchanged.
    """
    dnum, h, w = cost.shape
    out = np.empty_like(cost)

    if dx != 0:
        # March over columns; each step works on a (D, H) slab.
        order = range(w) if dx > 0 else range(w - 1, -1, -1)
        prev = np.zeros((dnum, h), dtype=np.float32)
        for x in order:
            if dy != 0:
                shifted = np.zeros_like(prev)
                if dy > 0:
                    shifted[:, dy:] = prev[:, :-dy]
                else:
                    shifted[:, :dy] = prev[:, -dy:]
                prev = shifted
            out[:, :, x] = prev = _step(cost[:, :, x], prev, p1, p2)
    else:
        order = range(h) if dy > 0 else range(h - 1, -1, -1)
        prev = np.zeros((dnum, w), dtype=np.float32)
        for y in order:
            out[:, y, :] = prev = _step(cost[:, y, :], prev, p1, p2)
    return out


def _step(
    c: np.ndarray, prev: np.ndarray, p1: float, p2: float
) -> np.ndarray:
    """One disparity-smoothness message pass. ``c``, ``prev`` are ``(D, N)``."""
    m = prev.min(axis=0)
    best = prev.copy()
    # prev[d - 1] and prev[d + 1], padded with the row minimum so the
    # d = 0 and d = D-1 ends do not get an artificial advantage.
    np.minimum(best[1:], prev[:-1] + p1, out=best[1:])
    np.minimum(best[:-1], prev[1:] + p1, out=best[:-1])
    np.minimum(best, m + p2, out=best)
    return (c + best - m).astype(np.float32)


def sgm_aggregate(
    cost: np.ndarray, p1: float = 10.0, p2: float = 120.0, paths: int = 4
) -> np.ndarray:
    """Sum the SGM recursion over ``paths`` directions (4 or 8)."""
    dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    if paths == 8:
        dirs += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    elif paths != 4:
        raise ValueError("paths must be 4 or 8")

    agg = np.zeros_like(cost)
    for dy, dx in dirs:
        agg += _aggregate_direction(cost, p1, p2, dy, dx)
    return agg


def potts_scanline_dp(cost: np.ndarray, jump_penalty: float = 20.0) -> np.ndarray:
    """Exact per-scanline optimum under a Potts (constant) smoothness prior.

    Minimises ``sum_x C(x, d_x) + P * [d_x != d_{x-1}]`` along each row
    independently, by Viterbi. With a Potts prior the forward message is
    ``min(F[d], min_k F[k] + P)``, so each step is O(D) and the whole thing
    is exact -- unlike SGM, which only approximates the 2-D optimum.

    The characteristic failure is horizontal streaking: rows are solved
    independently, so there is nothing tying one row's answer to the next.
    That contrast with :func:`sgm_aggregate` on the same cost volume is the
    clearest illustration of what the extra aggregation paths actually buy.
    """
    dnum, h, w = cost.shape
    # jumped[x, d, y]: at column x, disparity d was reached by a jump rather
    # than by continuing from d. Needed for backtracking.
    jumped = np.zeros((w, dnum, h), dtype=bool)
    argmins = np.zeros((w, h), dtype=np.int32)

    f = cost[:, :, 0].copy()  # (D, H) -- all rows marched in parallel
    for x in range(1, w):
        m = f.min(axis=0)
        am = f.argmin(axis=0)
        jump = m + jump_penalty
        jumped[x] = f > jump
        argmins[x] = am
        f = cost[:, :, x] + np.minimum(f, jump)

    disp = np.zeros((h, w), dtype=np.int32)
    rows = np.arange(h)
    d_cur = f.argmin(axis=0)
    disp[:, w - 1] = d_cur
    for x in range(w - 1, 0, -1):
        d_cur = np.where(jumped[x][d_cur, rows], argmins[x], d_cur)
        disp[:, x - 1] = d_cur
    return disp.astype(np.float32)


# --------------------------------------------------------------------------
# 3. winner selection and 4. validation
# --------------------------------------------------------------------------


def winner_takes_all(
    cost: np.ndarray,
    *,
    subpixel: bool = True,
    uniqueness_ratio: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Arg-min over disparity, with optional sub-pixel fit and ratio test.

    The sub-pixel step fits a parabola through the winning cost and its two
    neighbours and takes the vertex, which recovers a fraction of a pixel of
    accuracy for free -- usually the single cheapest accuracy win available,
    worth a couple of points of ``bad 0.5``.

    ``uniqueness_ratio`` rejects a pixel when the best cost is not better than
    the runner-up (excluding its immediate neighbours) by that relative
    margin, which is how a method declines to guess in textureless regions.
    """
    dnum = cost.shape[0]
    best_d = cost.argmin(axis=0).astype(np.int32)
    h, w = best_d.shape
    yy, xx = np.indices((h, w))
    best_c = cost[best_d, yy, xx]

    # Geometric, not cost-magnitude based: see `geometric_validity`.
    valid = geometric_validity(best_d.astype(np.float32))

    disp = best_d.astype(np.float32)
    if subpixel and dnum >= 3:
        dm = np.clip(best_d - 1, 0, dnum - 1)
        dp = np.clip(best_d + 1, 0, dnum - 1)
        c_m = cost[dm, yy, xx]
        c_p = cost[dp, yy, xx]
        denom = c_m - 2.0 * best_c + c_p
        safe = np.abs(denom) > 1e-6
        delta = np.divide(
            c_m - c_p, 2.0 * denom, out=np.zeros_like(denom), where=safe
        )
        # A vertex further than half a pixel away means the parabola is a bad
        # model there (usually a tie or a flat cost); fall back to integer.
        delta = np.where(np.abs(delta) <= 0.5, delta, 0.0)
        interior = (best_d > 0) & (best_d < dnum - 1)
        disp = disp + np.where(interior, delta, 0.0).astype(np.float32)

    if uniqueness_ratio > 0 and dnum > 3:
        masked = cost.copy()
        for off in (-1, 0, 1):
            dd = np.clip(best_d + off, 0, dnum - 1)
            masked[dd, yy, xx] = np.inf
        second = masked.min(axis=0)
        # Reject when the runner-up is nearly as good as the winner.
        valid &= second > best_c * (1.0 + uniqueness_ratio)

    return disp.astype(np.float32), valid


def right_disparity(cost: np.ndarray) -> np.ndarray:
    """Disparity map for the *right* view, reusing the left cost volume.

    Right pixel ``x`` at disparity ``d`` corresponds to left pixel ``x + d``,
    so ``cost_R[d, y, x] = cost_L[d, y, x + d]``. Reusing the volume this way
    makes the consistency check nearly free -- no second matching pass.
    """
    dnum, h, w = cost.shape
    cr = np.full_like(cost, BIG)
    for d in range(dnum):
        if d == 0:
            cr[d] = cost[d]
        else:
            cr[d, :, : w - d] = cost[d, :, d:]
    return cr.argmin(axis=0).astype(np.float32)


def left_right_check(
    disp_left: np.ndarray, disp_right: np.ndarray, tolerance: float = 1.0
) -> np.ndarray:
    """True where the left and right disparity maps agree within ``tolerance``.

    This is what detects occlusions: a left pixel with no counterpart in the
    right view will match something arbitrary, and the reverse lookup will
    disagree.
    """
    h, w = disp_left.shape
    xx = np.arange(w)[None, :]
    x_right = np.clip(np.rint(xx - disp_left).astype(np.int32), 0, w - 1)
    rows = np.arange(h)[:, None]
    return np.abs(disp_left - disp_right[rows, x_right]) <= tolerance


# --------------------------------------------------------------------------
# the matchers
# --------------------------------------------------------------------------


@dataclass
class ScratchMatcher:
    """Configurable from-scratch matcher: cost -> aggregation -> WTA -> check."""

    metric: str = "sad"
    block: int = 9
    #: Census neighbourhood as ``(height, width)``; at most 65 pixels so the
    #: code fits in a uint64. Only used when ``metric == "census"``.
    census_win: tuple[int, int] = (7, 9)
    aggregation: str = "none"  # none | sgm4 | sgm8 | dp
    # P1/P2 and jump_penalty are absolute, on the 0..COST_SCALE range that
    # `normalize_cost` puts every cost volume onto before aggregation.
    p1: float = 10.0
    p2: float = 120.0
    jump_penalty: float = 30.0
    subpixel: bool = True
    uniqueness_ratio: float = 0.0
    lr_check: bool = False
    max_volume_gb: float = 4.0

    def compute(self, pair: StereoPair) -> DisparityResult:
        cost = cost_volume(
            pair.left,
            pair.right,
            pair.ndisp,
            metric=self.metric,
            block=self.block,
            census_win=self.census_win,
            max_volume_gb=self.max_volume_gb,
        )

        if self.aggregation == "none":
            agg = cost
        elif self.aggregation in ("sgm4", "sgm8"):
            agg = sgm_aggregate(
                normalize_cost(cost),
                p1=self.p1,
                p2=self.p2,
                paths=8 if self.aggregation == "sgm8" else 4,
            )
        elif self.aggregation == "dp":
            # The DP produces a disparity map directly rather than an
            # aggregated volume, so there is nothing for `right_disparity` to
            # re-index. Fail loudly instead of silently ignoring the option.
            if self.lr_check:
                raise ValueError(
                    "lr_check is not supported with aggregation='dp': the "
                    "scanline DP returns a disparity map, not a cost volume "
                    "to re-index for the right view"
                )
            disp = potts_scanline_dp(
                normalize_cost(cost), jump_penalty=self.jump_penalty
            )
            valid = geometric_validity(disp)
            return DisparityResult(
                np.where(valid, disp, np.nan).astype(np.float32), valid
            )
        else:
            raise ValueError(f"unknown aggregation {self.aggregation!r}")

        disp, valid = winner_takes_all(
            agg, subpixel=self.subpixel, uniqueness_ratio=self.uniqueness_ratio
        )

        extra: dict = {}
        if self.lr_check:
            dr = right_disparity(agg)
            consistent = left_right_check(disp, dr)
            extra["lr_reject_frac"] = float((valid & ~consistent).mean())
            valid &= consistent

        return DisparityResult(
            np.where(valid, disp, np.nan).astype(np.float32), valid, extra
        )


def _entry(name: str, blurb: str, tags: tuple[str, ...], **kw):
    @register(
        name, family=_FAMILY, description=blurb, tags=tags, check=_check_numpy
    )
    def _build(_kw=kw) -> ScratchMatcher:
        return ScratchMatcher(**_kw)

    return _build


# Block sizes, penalties and windows below come from the sweep in
# `bench/sweep.py`, run on the tune split only (results/sweeps/). The single
# clearest pattern: every winner-takes-all variant wanted a *larger* window
# than the conventional 9x9 -- without an aggregation step the only thing
# suppressing ambiguous matches is window support, so 13-15 px wins by 3-4
# points. The aggregating variants want the opposite, a small window, because
# SGM supplies that support globally and a large window only blurs edges.

# --- winner-takes-all over four different matching costs -------------------
_entry(
    "scratch_sad_wta",
    "SAD cost volume, box aggregation, winner-takes-all + sub-pixel",
    ("local", "wta", "sad"),
    metric="sad",
    block=15,
)
_entry(
    "scratch_ssd_wta",
    "SSD cost volume, box aggregation, winner-takes-all",
    ("local", "wta", "ssd"),
    metric="ssd",
    block=15,
)
_entry(
    "scratch_zncc_wta",
    "Zero-mean normalised cross-correlation, winner-takes-all",
    ("local", "wta", "zncc", "photometric-invariant"),
    metric="zncc",
    block=13,
)
_entry(
    "scratch_census_wta",
    "Census transform + Hamming cost, winner-takes-all",
    ("local", "wta", "census", "photometric-invariant"),
    metric="census",
    block=11,
    census_win=(7, 7),
)

# --- the same cost with increasing amounts of global reasoning -------------
_entry(
    "scratch_sad_wta_lrcheck",
    "SAD + winner-takes-all + uniqueness test + left-right consistency",
    ("local", "wta", "sad", "validated"),
    metric="sad",
    block=11,
    uniqueness_ratio=0.05,
    lr_check=True,
)
_entry(
    "scratch_census_dp",
    "Census cost, exact per-scanline Viterbi under a Potts prior",
    ("global", "dp", "census"),
    metric="census",
    block=5,
    aggregation="dp",
    jump_penalty=90.0,
)
_entry(
    "scratch_census_sgm4",
    "Census cost, semi-global aggregation over 4 paths",
    ("global", "sgm", "census"),
    metric="census",
    block=5,
    aggregation="sgm4",
    p1=20.0,
)
_entry(
    "scratch_census_sgm8",
    "Census cost, semi-global aggregation over 8 paths",
    ("global", "sgm", "census", "accurate"),
    metric="census",
    block=3,
    aggregation="sgm8",
    p1=20.0,
)
_entry(
    "scratch_census_sgm8_lrcheck",
    "Census + 8-path SGM + left-right consistency (trades density for precision)",
    ("global", "sgm", "census", "accurate", "validated"),
    metric="census",
    block=3,
    aggregation="sgm8",
    p1=20.0,
    lr_check=True,
)
_entry(
    "scratch_sad_sgm8",
    "SAD cost, semi-global aggregation over 8 paths",
    ("global", "sgm", "sad"),
    metric="sad",
    block=5,
    aggregation="sgm8",
)
