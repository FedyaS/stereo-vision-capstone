"""Algorithm interface and registry.

Every matcher is wrapped so the harness can treat them identically:

* input is always ``(left_rgb_uint8, right_rgb_uint8, ndisp)``
* output is always float32 disparity in **left-image pixels**, with ``nan``
  where the method declined to estimate

That second point is where most stereo comparisons go wrong. OpenCV returns
fixed-point ``int16`` scaled by 16 with ``-16`` for "no match"; a naive
``/16`` turns every unmatched pixel into a confident ``-1`` and then into an
apparently-correct ``0`` after clipping. The adapters below convert properly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Protocol, runtime_checkable

import numpy as np

from ..types import DisparityResult, StereoPair


@runtime_checkable
class StereoAlgorithm(Protocol):
    """What the runner needs from a matcher."""

    def compute(self, pair: StereoPair) -> DisparityResult: ...


@dataclass(frozen=True)
class AlgorithmSpec:
    """Registry entry: how to build a matcher, and whether we can."""

    name: str
    family: str
    description: str
    build: Callable[[], StereoAlgorithm]
    tags: tuple[str, ...] = ()
    check: Callable[[], tuple[bool, str]] | None = None

    def availability(self) -> tuple[bool, str]:
        """``(is_available, reason)``. Never raises."""
        if self.check is None:
            return True, ""
        try:
            return self.check()
        except Exception as exc:  # pragma: no cover - defensive
            return False, f"{type(exc).__name__}: {exc}"


REGISTRY: dict[str, AlgorithmSpec] = {}


def register(
    name: str,
    *,
    family: str,
    description: str,
    tags: Iterable[str] = (),
    check: Callable[[], tuple[bool, str]] | None = None,
) -> Callable[[Callable[[], StereoAlgorithm]], Callable[[], StereoAlgorithm]]:
    """Decorator registering a zero-argument builder under ``name``."""

    def deco(build: Callable[[], StereoAlgorithm]) -> Callable[[], StereoAlgorithm]:
        if name in REGISTRY:
            raise ValueError(f"duplicate algorithm name: {name}")
        REGISTRY[name] = AlgorithmSpec(
            name=name,
            family=family,
            description=description,
            build=build,
            tags=tuple(tags),
            check=check,
        )
        return build

    return deco


def load_all() -> None:
    """Import every algorithm module so the registry is populated."""
    from . import learned, opencv_classical, scratch, torch_impl  # noqa: F401


def all_specs(include_unavailable: bool = True) -> list[AlgorithmSpec]:
    load_all()
    specs = sorted(REGISTRY.values(), key=lambda s: (s.family, s.name))
    if include_unavailable:
        return specs
    return [s for s in specs if s.availability()[0]]


def select(patterns: Iterable[str] | None) -> list[AlgorithmSpec]:
    """Resolve selectors into specs.

    A selector is an exact name, a family name (``opencv``, ``scratch``,
    ``torch``, ``learned``), a tag, or an ``fnmatch`` pattern such as
    ``opencv_sgbm*``. ``None`` or ``["all"]`` selects everything.
    """
    import fnmatch

    specs = all_specs()
    pats = list(patterns or [])
    if not pats or "all" in pats:
        return specs

    chosen: dict[str, AlgorithmSpec] = {}
    for pat in pats:
        hits = [
            s
            for s in specs
            if s.name == pat
            or s.family == pat
            or pat in s.tags
            or fnmatch.fnmatch(s.name, pat)
        ]
        if not hits:
            raise KeyError(
                f"no algorithm matches {pat!r}. Known names: "
                + ", ".join(s.name for s in specs)
            )
        chosen.update({s.name: s for s in hits})
    return [s for s in specs if s.name in chosen]


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------


def to_gray(img: np.ndarray) -> np.ndarray:
    """RGB uint8 -> grayscale uint8."""
    import cv2

    if img.ndim == 2:
        return np.ascontiguousarray(img)
    return cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2GRAY)


def from_opencv_fixed_point(disp16: np.ndarray, min_disp: int = 0) -> DisparityResult:
    """Convert an OpenCV ``int16`` disparity map to float with a validity mask.

    OpenCV marks "no match" with ``(min_disp - 1) * 16`` and otherwise stores
    ``disparity * 16``. Anything at or below the minimum is treated as invalid
    rather than silently clamped.
    """
    d = np.asarray(disp16).astype(np.float32) / 16.0
    valid = d > (min_disp - 0.5)
    return DisparityResult(np.where(valid, d, np.nan).astype(np.float32), valid)


def round_up_16(n: int) -> int:
    """OpenCV requires ``numDisparities`` to be a positive multiple of 16."""
    return max(16, int(np.ceil(n / 16.0) * 16))
