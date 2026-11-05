"""Core data types shared by datasets, algorithms and metrics."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Sentinel used for "no disparity estimated here".
INVALID = np.float32(np.nan)


@dataclass
class StereoPair:
    """A rectified stereo pair, optionally with ground truth.

    Attributes
    ----------
    name:
        Identifier, e.g. ``"middlebury/Teddy"``.
    left, right:
        ``(H, W, 3)`` uint8 RGB images, already rectified.
    ndisp:
        Disparity search range to use, in pixels (``0 <= d < ndisp``).
    gt:
        ``(H, W)`` float32 ground-truth disparity in left-image pixels, with
        ``nan`` where unknown. ``None`` when the dataset has no ground truth.
    occ_mask:
        ``(H, W)`` bool, True where the pixel is visible in *both* views.
        Used for the Middlebury "nonocc" evaluation region.
    focal_length_px, baseline_mm, doffs:
        Calibration, when known. Only needed to convert disparity to depth.
    """

    name: str
    left: np.ndarray
    right: np.ndarray
    ndisp: int
    gt: np.ndarray | None = None
    occ_mask: np.ndarray | None = None
    focal_length_px: float | None = None
    baseline_mm: float | None = None
    doffs: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return self.left.shape[0], self.left.shape[1]

    @property
    def megapixels(self) -> float:
        h, w = self.shape
        return h * w / 1e6

    @property
    def has_gt(self) -> bool:
        return self.gt is not None

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        h, w = self.shape
        return (
            f"StereoPair({self.name!r}, {w}x{h}, ndisp={self.ndisp}, "
            f"gt={'yes' if self.has_gt else 'no'})"
        )


@dataclass
class DisparityResult:
    """What an algorithm returns.

    ``disparity`` is float32 in left-image pixel units. Pixels the algorithm
    declined to estimate must be ``nan`` (or masked out via ``valid``);
    :func:`bench.metrics.evaluate` treats those as "no estimate" rather than
    as a zero-disparity estimate, which otherwise flatters sparse methods.
    """

    disparity: np.ndarray
    valid: np.ndarray | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def normalized(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(disparity_float32, valid_bool)`` with nan-invalids applied."""
        d = np.asarray(self.disparity, dtype=np.float32)
        v = np.isfinite(d)
        if self.valid is not None:
            v &= np.asarray(self.valid, dtype=bool)
        d = np.where(v, d, np.nan).astype(np.float32)
        return d, v
