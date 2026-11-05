"""Analytic ray-traceable primitives.

Analytic surfaces rather than triangle meshes, for one reason: the ground
truth is then *exact*. A mesh renderer gives you depth quantised by a z-buffer
and geometry quantised by tessellation, so a sphere's ground-truth disparity
is really a faceted approximation of one. Solving a quadratic gives the
distance to the true surface to float precision, and the disparity derived
from it is correct to the same precision. For a benchmark whose whole purpose
is measuring sub-pixel disparity error, that distinction matters.

Every ``intersect`` takes ray origins ``(N, 3)`` (or a single ``(3,)`` origin,
broadcast) and unit directions ``(N, 3)``, and returns ``(N,)`` distances with
``inf`` where the ray misses. The same code therefore serves primary rays,
shadow rays and reflection rays without a second implementation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .materials import Material

EPS = np.float32(1e-4)
INF = np.float32(np.inf)


def _as_rays(origin: np.ndarray, direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = np.asarray(direction, dtype=np.float32)
    o = np.asarray(origin, dtype=np.float32)
    if o.ndim == 1:
        o = np.broadcast_to(o, d.shape)
    return o, d


def yaw_matrix(yaw: float) -> np.ndarray:
    """Rotation about the +Y (up) axis, mapping local coordinates to world."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32
    )


@dataclass
class Shape:
    """Base class: a surface plus the material covering it."""

    material: Material = field(default_factory=Material)

    def intersect(self, origin: np.ndarray, direction: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def normal_at(self, points: np.ndarray) -> np.ndarray:
        raise NotImplementedError


@dataclass
class Plane(Shape):
    """An infinite plane through ``point`` with unit normal ``normal``.

    Used for floors, ceilings and the backdrop. Infinite rather than bounded
    because a bounded floor leaves a hole with no geometry behind it, and a
    hole means ground truth that is genuinely unknown rather than merely far.
    """

    point: tuple[float, float, float] = (0.0, 0.0, 0.0)
    normal: tuple[float, float, float] = (0.0, 1.0, 0.0)

    def intersect(self, origin: np.ndarray, direction: np.ndarray) -> np.ndarray:
        o, d = _as_rays(origin, direction)
        n = np.asarray(self.normal, dtype=np.float32)
        p0 = np.asarray(self.point, dtype=np.float32)
        denom = d @ n
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((p0 - o) @ n) / denom
        # Grazing rays (|denom| tiny) are numerically worthless: a huge t that
        # flickers between pixels. Treat them as misses.
        return np.where((np.abs(denom) > 1e-6) & (t > EPS), t, INF).astype(np.float32)

    def normal_at(self, points: np.ndarray) -> np.ndarray:
        n = np.asarray(self.normal, dtype=np.float32)
        return np.broadcast_to(n, np.asarray(points).shape).astype(np.float32)


@dataclass
class Sphere(Shape):
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius: float = 1.0

    def intersect(self, origin: np.ndarray, direction: np.ndarray) -> np.ndarray:
        o, d = _as_rays(origin, direction)
        oc = o - np.asarray(self.center, dtype=np.float32)
        # Directions are unit length, so the t^2 coefficient is 1.
        b = 2.0 * np.einsum("ij,ij->i", oc, d)
        c = np.einsum("ij,ij->i", oc, oc) - np.float32(self.radius) ** 2
        disc = b * b - 4.0 * c
        hit = disc >= 0.0
        sq = np.sqrt(np.where(hit, disc, 0.0))
        t0 = 0.5 * (-b - sq)
        t1 = 0.5 * (-b + sq)
        # Nearest positive root: t0 unless the origin is inside the sphere.
        t = np.where(t0 > EPS, t0, t1)
        return np.where(hit & (t > EPS), t, INF).astype(np.float32)

    def normal_at(self, points: np.ndarray) -> np.ndarray:
        n = np.asarray(points, dtype=np.float32) - np.asarray(
            self.center, dtype=np.float32
        )
        return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)


@dataclass
class Box(Shape):
    """A box of half-extents ``half`` at ``center``, rotated ``yaw`` about +Y.

    Yaw-only rotation is deliberate: boxes stand for walls, slabs, crates and
    shelves, which sit upright in every environment here, and keeping them
    upright means their vertical edges stay vertical in the image. Slanted
    *surfaces* -- the thing that actually punishes a fronto-parallel matcher --
    come from the box faces being oblique to the camera, which yaw gives you.
    """

    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    half: tuple[float, float, float] = (0.5, 0.5, 0.5)
    yaw: float = 0.0

    def __post_init__(self) -> None:
        self._R = yaw_matrix(self.yaw)          # local -> world
        self._Rt = self._R.T                    # world -> local

    def intersect(self, origin: np.ndarray, direction: np.ndarray) -> np.ndarray:
        o, d = _as_rays(origin, direction)
        ol = (o - np.asarray(self.center, dtype=np.float32)) @ self._R
        dl = d @ self._R
        h = np.asarray(self.half, dtype=np.float32)

        # A ray exactly parallel to a slab would divide by zero and produce a
        # nan that poisons the min/max reduction. Nudging the denominator to a
        # tiny *signed* value keeps the arithmetic finite and still correct:
        # inside the slab the two roots become -huge and +huge, so that axis
        # stops constraining; outside it both land on the same side, so
        # ``t_far < t_near`` and the ray misses, which is the right answer.
        dl_safe = np.where(np.abs(dl) < 1e-9, 1e-9, dl)
        inv = 1.0 / dl_safe
        t_a = (-h[None, :] - ol) * inv
        t_b = (h[None, :] - ol) * inv
        t_near = np.max(np.minimum(t_a, t_b), axis=1)
        t_far = np.min(np.maximum(t_a, t_b), axis=1)

        hit = (t_far >= t_near) & (t_far > EPS)
        t = np.where(t_near > EPS, t_near, t_far)
        return np.where(hit & (t > EPS), t, INF).astype(np.float32)

    def normal_at(self, points: np.ndarray) -> np.ndarray:
        pl = (np.asarray(points, dtype=np.float32) - np.asarray(
            self.center, dtype=np.float32
        )) @ self._R
        h = np.asarray(self.half, dtype=np.float32)
        # The face you are on is the axis where |local| is closest to its
        # half-extent, measured relatively so thin slabs behave.
        rel = np.abs(pl) / np.maximum(h[None, :], 1e-9)
        axis = np.argmax(rel, axis=1)
        nl = np.zeros_like(pl)
        rows = np.arange(len(pl))
        nl[rows, axis] = np.sign(pl[rows, axis])
        return (nl @ self._Rt).astype(np.float32)


@dataclass
class Cylinder(Shape):
    """A capped cylinder of ``radius`` and ``height``, axis along +Y.

    ``base`` is the centre of the bottom cap. Cylinders earn their place by
    being *curved in the matching direction*: every column of a cylinder has a
    different disparity, so a block matcher cannot win by guessing a plane,
    and the silhouette gives a hard occlusion boundary at a non-integer
    sub-pixel position.
    """

    base: tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius: float = 0.5
    height: float = 1.0

    def intersect(self, origin: np.ndarray, direction: np.ndarray) -> np.ndarray:
        o, d = _as_rays(origin, direction)
        b = np.asarray(self.base, dtype=np.float32)
        r = np.float32(self.radius)
        y0, y1 = b[1], b[1] + np.float32(self.height)

        oc = o - b
        # Side surface: the same quadratic as a sphere, but in the XZ plane.
        a = d[:, 0] ** 2 + d[:, 2] ** 2
        bq = 2.0 * (oc[:, 0] * d[:, 0] + oc[:, 2] * d[:, 2])
        cq = oc[:, 0] ** 2 + oc[:, 2] ** 2 - r * r
        disc = bq * bq - 4.0 * a * cq
        side_ok = (disc >= 0.0) & (a > 1e-9)
        sq = np.sqrt(np.where(side_ok, disc, 0.0))
        denom = np.where(a > 1e-9, 2.0 * a, 1.0)
        best = np.full(len(d), INF, dtype=np.float32)
        for t_cand in ((-bq - sq) / denom, (-bq + sq) / denom):
            y = o[:, 1] + t_cand * d[:, 1]
            ok = side_ok & (t_cand > EPS) & (y >= y0) & (y <= y1)
            best = np.where(ok & (t_cand < best), t_cand, best)

        # Caps: plane hits constrained to the disc.
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_dy = 1.0 / d[:, 1]
            for y_cap in (y0, y1):
                t_cand = (y_cap - o[:, 1]) * inv_dy
                px = o[:, 0] + t_cand * d[:, 0] - b[0]
                pz = o[:, 2] + t_cand * d[:, 2] - b[2]
                ok = (
                    (np.abs(d[:, 1]) > 1e-9)
                    & (t_cand > EPS)
                    & (px * px + pz * pz <= r * r)
                )
                best = np.where(ok & (t_cand < best), t_cand, best)

        return best.astype(np.float32)

    def normal_at(self, points: np.ndarray) -> np.ndarray:
        p = np.asarray(points, dtype=np.float32)
        b = np.asarray(self.base, dtype=np.float32)
        y0, y1 = b[1], b[1] + np.float32(self.height)
        n = np.zeros_like(p)
        n[:, 0] = p[:, 0] - b[0]
        n[:, 2] = p[:, 2] - b[2]
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
        # Within a hair of a cap plane, the cap normal wins over the side one.
        tol = np.float32(1e-3)
        on_bottom = np.abs(p[:, 1] - y0) < tol
        on_top = np.abs(p[:, 1] - y1) < tol
        n[on_bottom] = (0.0, -1.0, 0.0)
        n[on_top] = (0.0, 1.0, 0.0)
        return n
