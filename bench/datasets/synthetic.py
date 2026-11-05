"""Procedurally generated stereo pairs with exact ground truth.

Always available -- no download -- so the harness and its tests can run on a
clean checkout. Useful as a sanity floor too: a matcher that cannot score well
on richly textured synthetic data has a bug, not a hard scene.

Construction. A texture ``T`` is used *as* the right image, and the left image
is produced by sampling ``T`` at ``x - d(y, x)``, because a left pixel ``x``
with disparity ``d`` matches right pixel ``x - d``. That makes ``d`` exact
ground truth in the left frame by construction, with no interpolation of the
disparity field itself. Occlusion is not physically modelled (the warp is
many-to-one at depth edges), so the evaluation mask drops a border band of
``ndisp`` pixels and a ring around each depth discontinuity.
"""
from __future__ import annotations

import numpy as np

from ..types import StereoPair


def _band_limited_noise(
    shape: tuple[int, int], rng: np.random.Generator, sigma: float = 1.2
) -> np.ndarray:
    """Smoothed uniform noise -- textured enough to match, not pure aliasing."""
    import cv2

    noise = rng.random(shape, dtype=np.float32)
    k = max(3, int(sigma * 6) | 1)
    return cv2.GaussianBlur(noise, (k, k), sigma)


def _disparity_field(
    h: int, w: int, rng: np.random.Generator, dmin: float, dmax: float
) -> np.ndarray:
    """A few slanted planes plus ellipsoidal blobs: smooth regions, hard edges."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # Background: a gently slanted plane, which punishes fronto-parallel bias.
    d = dmin + (dmax - dmin) * 0.35 * (0.3 + 0.7 * yy / h)
    d += (dmax - dmin) * 0.1 * (xx / w)

    for _ in range(int(rng.integers(3, 6))):
        cy, cx = rng.uniform(0.15, 0.85) * h, rng.uniform(0.15, 0.85) * w
        ry, rx = rng.uniform(0.08, 0.22) * h, rng.uniform(0.08, 0.22) * w
        inside = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0
        level = rng.uniform(dmin + 0.4 * (dmax - dmin), dmax)
        d = np.where(inside, level, d)

    return np.clip(d, dmin, dmax).astype(np.float32)


def make_synthetic_pair(
    *,
    seed: int = 0,
    size: tuple[int, int] = (320, 448),
    ndisp: int = 64,
    name: str | None = None,
) -> StereoPair:
    """Generate one synthetic stereo pair with exact ground-truth disparity."""
    import cv2

    h, w = size
    rng = np.random.default_rng(seed)
    dmin, dmax = 4.0, float(ndisp - 4)

    gt = _disparity_field(h, w, rng, dmin, dmax)

    # Right image: coloured band-limited texture with some structure on top.
    tex = np.stack(
        [_band_limited_noise((h, w), rng, sigma=1.0 + 0.4 * c) for c in range(3)],
        axis=-1,
    )
    # A few hard edges, so block matchers have something to lock onto.
    for _ in range(14):
        p0 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        color = [float(rng.random()) for _ in range(3)]
        cv2.line(tex, p0, p1, color, 2)
    right = np.clip(tex * 255.0, 0, 255).astype(np.uint8)

    # Left image: sample the right-image texture at x - d(y, x).
    xx = np.tile(np.arange(w, dtype=np.float32), (h, 1))
    yy = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))
    left = cv2.remap(
        right,
        xx - gt,
        yy,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    # Evaluation mask: drop the left border band (no valid match exists there)
    # and a thin ring around depth discontinuities, which the warp cannot model.
    mask = np.ones((h, w), bool)
    mask[:, : int(dmax) + 1] = False
    edge = (np.abs(cv2.Laplacian(gt, cv2.CV_32F)) > 1.0).astype(np.uint8)
    mask &= cv2.dilate(edge, np.ones((7, 7), np.uint8)) == 0

    return StereoPair(
        name=name or f"synthetic/seed{seed}",
        left=left,
        right=right,
        ndisp=ndisp,
        gt=gt,
        occ_mask=mask,
        meta={"source": "synthetic", "seed": seed, "dmin": dmin, "dmax": dmax},
    )
