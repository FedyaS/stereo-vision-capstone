"""OpenCV's stereo matchers, plus the ximgproc post-filters.

Covered here:

================================  =========================================
``opencv_bm``                     Classic block matching (SAD + prefilter)
``opencv_sgbm``                   Semi-global block matching, 5 paths
``opencv_sgbm_hh``                SGBM, full 8-path, whole-image DP
``opencv_sgbm_3way``              SGBM, 3 paths -- the speed-tuned variant
``opencv_sgbm_hh4``              SGBM, 4-path HH
``opencv_sgbm_wls``               SGBM + left-right consistency + WLS filter
``opencv_bm_wls``                 BM + WLS filter
``opencv_quasi_dense``            Sparse seed growing (``cv2.stereo``)
================================  =========================================

The SGBM "mode" variants differ only in how many aggregation paths the solver
walks, which is exactly the accuracy/time trade-off this harness is meant to
expose, so they are registered as separate algorithms rather than as one entry
with a flag.

``P1``/``P2`` follow the OpenCV documentation's recommendation
(``8 * channels * blockSize^2`` and ``32 * channels * blockSize^2``), scaled
with the block size so the smoothness penalty stays proportional to the
matching cost.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..types import DisparityResult, StereoPair
from .base import from_opencv_fixed_point, register, round_up_16, to_gray

_FAMILY = "opencv"


def _check_cv2() -> tuple[bool, str]:
    try:
        import cv2  # noqa: F401
    except Exception as exc:
        return False, f"opencv not importable: {exc}"
    return True, ""


def _check_ximgproc() -> tuple[bool, str]:
    ok, why = _check_cv2()
    if not ok:
        return ok, why
    import cv2

    if not hasattr(cv2, "ximgproc"):
        return False, "needs opencv-contrib-python (cv2.ximgproc missing)"
    return True, ""


def _check_cv_stereo() -> tuple[bool, str]:
    ok, why = _check_cv2()
    if not ok:
        return ok, why
    import cv2

    if not hasattr(cv2, "stereo") or not hasattr(cv2.stereo, "QuasiDenseStereo_create"):
        return False, "needs opencv-contrib-python (cv2.stereo missing)"
    return True, ""


# --------------------------------------------------------------------------
# block matching
# --------------------------------------------------------------------------


@dataclass
class OpenCVBlockMatcher:
    """StereoBM: SAD over a window, after an x-Sobel prefilter.

    Grayscale only, and it refuses to estimate in low-texture regions, so
    expect a sparse map -- which is the point of also reporting ``density``.
    """

    # Defaults below were chosen by the sweep in `bench/sweep.py` on the tune
    # split (see results/sweeps/). Dropping uniqueness_ratio from 10 to 5 and
    # the texture threshold to 0 is worth 1.8 points of bad2_all: StereoBM is
    # already sparse, and tightening its acceptance tests further costs more
    # in blanks (each counted as an error) than it saves in bad matches.
    block_size: int = 15
    uniqueness_ratio: int = 5
    # Note this is compared against a SAD-based texture score, not an
    # intensity: measured on Middlebury it does nothing at all below ~200 and
    # only starts rejecting pixels past ~1000. Sweeping it over 0..20, as the
    # usual tutorial values suggest, searches a dead range.
    texture_threshold: int = 0
    #: "xsobel" or "normalized" -- which prefilter runs before matching.
    prefilter_type: str = "xsobel"
    speckle_window_size: int = 100
    speckle_range: int = 2
    disp12_max_diff: int = 1
    prefilter_cap: int = 31

    def compute(self, pair: StereoPair) -> DisparityResult:
        import cv2

        prefilters = {
            "xsobel": cv2.StereoBM_PREFILTER_XSOBEL,
            "normalized": cv2.StereoBM_PREFILTER_NORMALIZED_RESPONSE,
        }
        if self.prefilter_type not in prefilters:
            raise ValueError(
                f"unknown prefilter_type {self.prefilter_type!r}; "
                f"expected one of {sorted(prefilters)}"
            )

        n = round_up_16(pair.ndisp)
        bm = cv2.StereoBM_create(numDisparities=n, blockSize=self.block_size)
        bm.setPreFilterType(prefilters[self.prefilter_type])
        bm.setPreFilterCap(self.prefilter_cap)
        bm.setTextureThreshold(self.texture_threshold)
        bm.setUniquenessRatio(self.uniqueness_ratio)
        bm.setSpeckleWindowSize(self.speckle_window_size)
        bm.setSpeckleRange(self.speckle_range)
        bm.setDisp12MaxDiff(self.disp12_max_diff)
        raw = bm.compute(to_gray(pair.left), to_gray(pair.right))
        return from_opencv_fixed_point(raw, min_disp=0)


@dataclass
class OpenCVSGBM:
    """StereoSGBM: Hirschmuller semi-global matching on a Birchfield-Tomasi cost."""

    mode: int = 0
    # Tuned on the tune split (see results/sweeps/). The sweep moved the
    # OpenCV-documented P1/P2 recommendation from 8/32 to 4/64 and halved
    # uniqueness_ratio, worth ~1.2 points of bad2_all across all four modes:
    # a wider P2/P1 ratio lets the solver hold a surface through texture
    # while still paying heavily for a genuine depth jump.
    block_size: int = 5
    # P1/P2 as multiples of `channels * blockSize^2`, which is how the OpenCV
    # documentation states its recommendation (8 and 32). Keeping them as
    # multipliers rather than absolute values means the smoothness penalty
    # stays proportional to the matching cost when the block size changes.
    p1_mult: int = 4
    p2_mult: int = 64
    uniqueness_ratio: int = 5
    speckle_window_size: int = 100
    speckle_range: int = 2
    disp12_max_diff: int = 1
    prefilter_cap: int = 31

    def _matcher(self, pair: StereoPair, channels: int):
        import cv2

        bs = self.block_size
        return cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=round_up_16(pair.ndisp),
            blockSize=bs,
            P1=self.p1_mult * channels * bs * bs,
            P2=self.p2_mult * channels * bs * bs,
            disp12MaxDiff=self.disp12_max_diff,
            uniquenessRatio=self.uniqueness_ratio,
            speckleWindowSize=self.speckle_window_size,
            speckleRange=self.speckle_range,
            preFilterCap=self.prefilter_cap,
            mode=self.mode,
        )

    def compute(self, pair: StereoPair) -> DisparityResult:
        sgbm = self._matcher(pair, channels=3)
        raw = sgbm.compute(
            np.ascontiguousarray(pair.left), np.ascontiguousarray(pair.right)
        )
        return from_opencv_fixed_point(raw, min_disp=0)


@dataclass
class OpenCVWLS:
    """A base matcher plus the ximgproc WLS edge-aware disparity filter.

    Runs the matcher twice (left-to-right and right-to-left), uses the
    disagreement between the two as a confidence map, then solves a weighted
    least-squares problem that smooths the disparity while respecting image
    edges. This fills holes, so density goes to ~100% -- compare its
    ``bad2_all`` against the unfiltered matcher's to see what filling buys.
    """

    base: str = "sgbm"
    # Tuned per base matcher (see the registry entries below): the sweep
    # pushed lambda far down from the ximgproc default of 8000, because a
    # strong smoothness prior drags the already-dense SGBM output across
    # depth edges faster than it fills holes.
    lambda_: float = 1000.0
    sigma_color: float = 1.5
    block_size: int = 5

    def compute(self, pair: StereoPair) -> DisparityResult:
        import cv2

        n = round_up_16(pair.ndisp)
        if self.base == "bm":
            # StereoBM needs an odd block of at least 5; StereoSGBM is happy
            # with 3. Raising here keeps a shared parameter grid from dying on
            # a bare OpenCV assertion that names neither the value nor the
            # limit.
            if self.block_size < 5 or self.block_size % 2 == 0:
                raise ValueError(
                    f"StereoBM needs an odd block_size >= 5, got "
                    f"{self.block_size}"
                )
            left_in, right_in = to_gray(pair.left), to_gray(pair.right)
            left_matcher = cv2.StereoBM_create(numDisparities=n, blockSize=self.block_size)
            left_matcher.setPreFilterType(cv2.StereoBM_PREFILTER_XSOBEL)
            left_matcher.setTextureThreshold(10)
            left_matcher.setUniquenessRatio(10)
        else:
            left_in = np.ascontiguousarray(pair.left)
            right_in = np.ascontiguousarray(pair.right)
            bs = self.block_size
            left_matcher = cv2.StereoSGBM_create(
                minDisparity=0,
                numDisparities=n,
                blockSize=bs,
                P1=8 * 3 * bs * bs,
                P2=32 * 3 * bs * bs,
                disp12MaxDiff=1,
                uniquenessRatio=10,
                speckleWindowSize=100,
                speckleRange=2,
                mode=cv2.StereoSGBM_MODE_SGBM_3WAY,
            )

        right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
        disp_l = left_matcher.compute(left_in, right_in)
        disp_r = right_matcher.compute(right_in, left_in)

        wls = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
        wls.setLambda(self.lambda_)
        wls.setSigmaColor(self.sigma_color)
        filtered = wls.filter(
            disp_l, np.ascontiguousarray(pair.left), disparity_map_right=disp_r
        )

        res = from_opencv_fixed_point(filtered, min_disp=0)
        conf = wls.getConfidenceMap()
        if conf is not None:
            res.extra["mean_confidence"] = float(np.asarray(conf).mean())
        return res


@dataclass
class OpenCVQuasiDense:
    """``cv2.stereo.QuasiDenseStereo``: sparse feature seeds, then growing.

    Matches a sparse set of good features, then propagates matches into
    neighbouring pixels while a correlation test holds. Very different
    resource profile from the cost-volume methods -- no ``ndisp``-sized
    volume is ever built -- at the cost of a non-dense result.
    """

    corr_win_size: int = 5
    # 0.6 -> 0.4 is worth 1.7 points: a laxer growing test trades a little
    # precision for noticeably more coverage, and blanks count as errors.
    correlation_threshold: float = 0.4
    # `neighborhoodSize` is deliberately left at OpenCV's default of 5: the
    # growing step indexes a neighbourhood against a fixed 15px border, and
    # raising it throws an untyped C++ exception out of `process`.
    disparity_gradient: int = 1
    outlier_factor: float = 1.5

    def compute(self, pair: StereoPair) -> DisparityResult:
        import cv2

        h, w = pair.shape
        # The binding takes only (size, paramFilepath) -- note size is (w, h),
        # and passing (h, w) throws -- so parameters go through the `Param`
        # attribute after construction. `PropagationParameters()`
        # default-constructs to all zeros, which makes goodFeaturesToTrack
        # assert, so the object already on the instance (holding OpenCV's real
        # defaults) is edited in place and assigned back.
        qds = cv2.stereo.QuasiDenseStereo_create((w, h))
        params = qds.Param
        params.corrWinSizeX = self.corr_win_size
        params.corrWinSizeY = self.corr_win_size
        params.correlationThreshold = self.correlation_threshold
        params.disparityGradient = self.disparity_gradient
        qds.Param = params

        qds.process(to_gray(pair.left), to_gray(pair.right))
        disp = np.asarray(qds.getDisparity(), dtype=np.float32)

        # Unmatched pixels come back as nan, and the growing step occasionally
        # emits wild values (hundreds of pixels on a 64-disparity scene), so
        # anything beyond the scene's range is dropped rather than scored.
        valid = (
            np.isfinite(disp) & (disp > 0) & (disp < pair.ndisp * self.outlier_factor)
        )
        return DisparityResult(
            np.where(valid, disp, np.nan).astype(np.float32),
            valid,
            {
                "sparse_seeds": len(qds.getSparseMatches()),
                "dense_matches": len(qds.getDenseMatches()),
            },
        )


# --------------------------------------------------------------------------
# registry entries
# --------------------------------------------------------------------------


@register(
    "opencv_bm",
    family=_FAMILY,
    description="StereoBM: SAD block matching with x-Sobel prefilter",
    tags=("classical", "local", "fast"),
    check=_check_cv2,
)
def _bm() -> OpenCVBlockMatcher:
    return OpenCVBlockMatcher()


def _sgbm_entry(name: str, mode_attr: str, blurb: str, *tags: str, **overrides):
    """Register one SGBM mode.

    ``overrides`` carries the per-mode results of the parameter sweep: the
    five-path default mode prefers a larger window and a tighter P2 than the
    others, so the tuned values cannot all live on the dataclass.
    """

    @register(
        name,
        family=_FAMILY,
        description=blurb,
        tags=("classical", "global", *tags),
        check=_check_cv2,
    )
    def _build(_ov=overrides) -> OpenCVSGBM:
        import cv2

        return OpenCVSGBM(mode=getattr(cv2, mode_attr), **_ov)

    return _build


_sgbm_entry(
    "opencv_sgbm",
    "StereoSGBM_MODE_SGBM",
    "StereoSGBM, 5-path aggregation (OpenCV default)",
    block_size=7,
    p2_mult=32,
)
_sgbm_entry(
    "opencv_sgbm_hh",
    "StereoSGBM_MODE_HH",
    "StereoSGBM, full 8-path whole-image dynamic programming",
    "accurate",
)
_sgbm_entry(
    "opencv_sgbm_3way",
    "StereoSGBM_MODE_SGBM_3WAY",
    "StereoSGBM, 3-path aggregation (fastest SGBM mode)",
    "fast",
)
_sgbm_entry(
    "opencv_sgbm_hh4",
    "StereoSGBM_MODE_HH4",
    "StereoSGBM, 4-path HH variant",
)


@register(
    "opencv_sgbm_wls",
    family=_FAMILY,
    description="SGBM(3way) + right matcher + ximgproc WLS edge-aware filter",
    tags=("classical", "global", "filtered", "accurate"),
    check=_check_ximgproc,
)
def _sgbm_wls() -> OpenCVWLS:
    return OpenCVWLS(base="sgbm", lambda_=1000.0, sigma_color=1.5, block_size=5)


@register(
    "opencv_bm_wls",
    family=_FAMILY,
    description="StereoBM + right matcher + ximgproc WLS edge-aware filter",
    tags=("classical", "local", "filtered"),
    check=_check_ximgproc,
)
def _bm_wls() -> OpenCVWLS:
    return OpenCVWLS(base="bm", lambda_=4000.0, sigma_color=1.5, block_size=5)


@register(
    "opencv_quasi_dense",
    family=_FAMILY,
    description="QuasiDenseStereo: sparse seed matching with region growing",
    tags=("classical", "sparse"),
    check=_check_cv_stereo,
)
def _qds() -> OpenCVQuasiDense:
    return OpenCVQuasiDense()
