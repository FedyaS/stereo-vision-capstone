"""A vectorised CPU ray tracer, and the rectified stereo rig built on it.

Why a ray tracer and not an off-the-shelf 3-D engine
----------------------------------------------------
The obvious candidates all cost more than they give here. ``bpy`` (Blender as
a Python module) is a ~1 GB wheel pinned to one exact CPython version, and can
only be imported once per process, which fights the multiprocess fan-out that
mass generation needs. Open3D, pyrender and anything else on an offscreen
OpenGL context have a long history of failing *silently* on headless Windows
-- returning a plausible-looking buffer of zeros -- which is the worst possible
failure mode for a ground-truth generator, because the dataset looks fine
until you train on it. And all of them hand back depth from a z-buffer, i.e.
quantised, interpolated across triangles, and clipped by the near/far planes.

Solving the ray-surface equation instead gives exact distances to analytic
surfaces, in pure NumPy, with no new dependency, no GPU, no display, and
identical output on every platform. For the scene complexity this needs --
tens of primitives, not millions of triangles -- it is also fast enough.

The stereo convention
---------------------
Two pinhole cameras share intrinsics and orientation; the right one is
displaced by ``baseline`` along the camera's own right axis. For a point at
depth ``Z`` along the optical axis::

    x_left - x_right = focal_px * baseline / Z  =  disparity

so ``disparity`` is positive and the match for left pixel ``x`` sits at
``x - disparity`` in the right image -- the convention the rest of the
benchmark already uses. Because both views are rendered independently from
real geometry, occlusion is *physical*: a surface hidden from the right camera
is genuinely absent from the right image, rather than smeared by a warp.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .materials import Light, Material
from .shapes import EPS, INF, Shape

_UP = np.array([0.0, 1.0, 0.0], dtype=np.float32)


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return (v / np.maximum(n, 1e-9)).astype(np.float32)


@dataclass
class Camera:
    """A pinhole camera, positioned by look-at.

    ``focal_px`` is in pixels at the *output* resolution, so it fixes the field
    of view together with the image width. The principal point sits at the
    image centre, which is what rectified stereo assumes.
    """

    position: tuple[float, float, float] = (0.0, 1.4, -5.0)
    target: tuple[float, float, float] = (0.0, 1.0, 0.0)
    focal_px: float = 500.0

    def basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Orthonormal ``(forward, right, up)`` in world space."""
        forward = _normalize(
            np.asarray(self.target, dtype=np.float32)
            - np.asarray(self.position, dtype=np.float32)
        )
        world_up = _UP
        if abs(float(forward @ world_up)) > 0.999:  # looking straight up/down
            world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        right = _normalize(np.cross(world_up, forward))
        up = _normalize(np.cross(forward, right))
        return forward, right, up


@dataclass
class Scene:
    """Everything the tracer needs: geometry, lights and the atmosphere."""

    shapes: list[Shape] = field(default_factory=list)
    lights: list[Light] = field(default_factory=list)
    camera: Camera = field(default_factory=Camera)
    ambient: float = 0.22
    sky_top: tuple[float, float, float] = (0.42, 0.56, 0.78)
    sky_bottom: tuple[float, float, float] = (0.78, 0.82, 0.86)
    fog_density: float = 0.0
    fog_color: tuple[float, float, float] = (0.72, 0.76, 0.82)
    exposure: float = 1.0
    #: Set by the environment sampler: roughly the nearest and farthest
    #: distance the camera should see, used to auto-fit the baseline.
    depth_hint: tuple[float, float] = (1.0, 30.0)


def ray_grid(
    camera: Camera, height: int, width: int, offset: tuple[float, float] = (0.5, 0.5)
) -> np.ndarray:
    """Unit ray directions for every pixel, as ``(height * width, 3)``.

    ``offset`` places the sample inside the pixel footprint; the default is the
    pixel centre.
    """
    forward, right, up = camera.basis()
    cx, cy = width / 2.0, height / 2.0
    xs = np.arange(width, dtype=np.float32) + np.float32(offset[0])
    ys = np.arange(height, dtype=np.float32) + np.float32(offset[1])
    px, py = np.meshgrid(xs, ys)
    f = np.float32(camera.focal_px)
    # Image y grows downward, hence the minus on the up axis.
    d = (
        forward[None, None, :] * f
        + right[None, None, :] * (px - cx)[:, :, None]
        - up[None, None, :] * (py - cy)[:, :, None]
    )
    return _normalize(d.reshape(-1, 3))


def closest_hit(
    shapes: list[Shape], origin: np.ndarray, direction: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest surface along each ray: ``(distance, shape_index)``.

    ``distance`` is ``inf`` and the index ``-1`` where the ray hits nothing.
    """
    n = len(direction)
    best_t = np.full(n, INF, dtype=np.float32)
    best_i = np.full(n, -1, dtype=np.int32)
    for i, shape in enumerate(shapes):
        t = shape.intersect(origin, direction)
        closer = t < best_t
        best_t = np.where(closer, t, best_t)
        best_i = np.where(closer, np.int32(i), best_i)
    return best_t, best_i


def any_hit(
    shapes: list[Shape],
    origin: np.ndarray,
    direction: np.ndarray,
    max_dist: np.ndarray,
) -> np.ndarray:
    """True where *something* blocks the ray before ``max_dist`` -- a shadow test.

    Cheaper than :func:`closest_hit` in two ways: it needs no index bookkeeping,
    and it stops as soon as every ray is known to be blocked, which in a closed
    room happens after very few shapes.
    """
    blocked = np.zeros(len(direction), dtype=bool)
    for shape in shapes:
        t = shape.intersect(origin, direction)
        blocked |= (t > EPS) & (t < max_dist)
        if blocked.all():
            break
    return blocked


def _sky(scene: Scene, direction: np.ndarray) -> np.ndarray:
    """Background colour: a vertical gradient, so "up" and "down" differ."""
    top = np.asarray(scene.sky_top, dtype=np.float32)
    bottom = np.asarray(scene.sky_bottom, dtype=np.float32)
    w = np.clip(0.5 + 0.5 * direction[:, 1], 0.0, 1.0)[:, None]
    return (bottom[None, :] * (1.0 - w) + top[None, :] * w).astype(np.float32)


def _surface_properties(
    shapes: list[Shape], points: np.ndarray, shape_idx: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Gather per-point ``(albedo, normal, specular, shininess, reflectivity)``.

    Grouped by shape because a material evaluates its texture over a batch of
    points; doing it per pixel would be orders of magnitude slower.
    """
    n = len(points)
    albedo = np.zeros((n, 3), dtype=np.float32)
    normal = np.zeros((n, 3), dtype=np.float32)
    spec = np.zeros(n, dtype=np.float32)
    shine = np.full(n, 32.0, dtype=np.float32)
    refl = np.zeros(n, dtype=np.float32)

    for i, shape in enumerate(shapes):
        m = shape_idx == i
        if not m.any():
            continue
        pts = points[m]
        mat: Material = shape.material
        albedo[m] = mat.albedo(pts)
        normal[m] = shape.normal_at(pts)
        spec[m] = mat.specular
        shine[m] = mat.shininess
        refl[m] = mat.reflectivity
    return albedo, normal, spec, shine, refl


def _direct_lighting(
    scene: Scene,
    points: np.ndarray,
    normal: np.ndarray,
    view_dir: np.ndarray,
    albedo: np.ndarray,
    spec: np.ndarray,
    shine: np.ndarray,
    shadows: bool,
) -> np.ndarray:
    """Lambert diffuse plus Blinn-Phong specular, summed over the lights.

    The specular term is the one part of the image that is deliberately *not*
    the same in both views: it depends on the direction to the camera, so a
    highlight sits on a slightly different part of the surface in each eye.
    Real stereo pairs have exactly this, and a matcher that assumes perfect
    photometric consistency should be penalised for it.
    """
    out = albedo * np.float32(scene.ambient)
    toward_camera = -view_dir

    for light in scene.lights:
        lpos = np.asarray(light.position, dtype=np.float32)
        lcol = np.asarray(light.color, dtype=np.float32) * np.float32(light.intensity)

        if light.directional:
            ldir = _normalize(-lpos)[None, :] if lpos.ndim == 1 else _normalize(-lpos)
            ldir = np.broadcast_to(ldir, points.shape).astype(np.float32)
            dist = np.full(len(points), np.float32(1e7))
            atten = np.ones(len(points), dtype=np.float32)
        else:
            delta = lpos[None, :] - points
            dist = np.linalg.norm(delta, axis=1)
            ldir = delta / np.maximum(dist[:, None], 1e-9)
            # Softened inverse-square: keeps a light close to a surface from
            # blowing out to pure white, which would destroy all texture there.
            atten = (
                1.0 / (1.0 + np.float32(light.falloff) * dist * dist)
            ).astype(np.float32)

        ndl = np.maximum(np.einsum("ij,ij->i", normal, ldir), 0.0)
        lit = ndl > 0.0

        visible = np.ones(len(points), dtype=np.float32)
        if shadows and light.casts_shadow and lit.any():
            idx = np.flatnonzero(lit)
            shadow_origin = points[idx] + normal[idx] * np.float32(2e-3)
            blocked = any_hit(
                scene.shapes, shadow_origin, ldir[idx], dist[idx] - np.float32(2e-3)
            )
            visible[idx] = (~blocked).astype(np.float32)

        weight = (ndl * atten * visible)[:, None]
        out += albedo * lcol[None, :] * weight

        half = _normalize(ldir + toward_camera)
        ndh = np.maximum(np.einsum("ij,ij->i", normal, half), 0.0)
        out += (
            lcol[None, :]
            * (spec * np.power(ndh, shine) * atten * visible)[:, None]
        )

    return out


def trace(
    scene: Scene,
    origin: np.ndarray,
    direction: np.ndarray,
    *,
    shadows: bool = True,
    bounces: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Shade a batch of rays.

    Returns ``(colour (N, 3) linear float, distance (N,) with inf for misses)``.
    Shading is computed only on the rays that hit something, so an open scene
    where most rays escape to the sky costs almost nothing.
    """
    t, idx = closest_hit(scene.shapes, origin, direction)
    color = _sky(scene, direction)
    hit = idx >= 0
    if not hit.any():
        return color, t

    h = np.flatnonzero(hit)
    o = np.asarray(origin, dtype=np.float32)
    o_h = o[h] if o.ndim == 2 else o[None, :]
    d_h = direction[h]
    points = o_h + d_h * t[h][:, None]

    albedo, normal, spec, shine, refl = _surface_properties(
        scene.shapes, points, idx[h]
    )
    # Flip normals that face away from the viewer, so the far side of a plane
    # or the inside of a room is lit rather than black.
    facing = np.einsum("ij,ij->i", normal, d_h) > 0.0
    normal[facing] *= -1.0

    shaded = _direct_lighting(
        scene, points, normal, d_h, albedo, spec, shine, shadows
    )

    if bounces > 0 and (refl > 1e-3).any():
        r = np.flatnonzero(refl > 1e-3)
        n_r, d_r = normal[r], d_h[r]
        mirror = d_r - 2.0 * np.einsum("ij,ij->i", d_r, n_r)[:, None] * n_r
        bounce_color, _ = trace(
            scene,
            points[r] + n_r * np.float32(2e-3),
            _normalize(mirror),
            shadows=shadows,
            bounces=bounces - 1,
        )
        k = refl[r][:, None]
        shaded[r] = shaded[r] * (1.0 - k) + bounce_color * k

    color[h] = shaded
    return color, t


def render_view(
    scene: Scene,
    camera: Camera,
    height: int,
    width: int,
    *,
    supersample: int = 1,
    shadows: bool = True,
    bounces: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render one view.

    Returns ``(rgb (H, W, 3) linear float, depth (H, W), depth_spread (H, W))``.

    ``depth`` is distance along the optical axis -- the ``Z`` that disparity
    divides into -- not distance along the ray. ``depth_spread`` is the
    relative range of depth seen by the sub-samples inside one output pixel; it
    is non-zero only at silhouettes, and is what lets the caller mark the
    pixels where a single "true" disparity does not exist because the pixel
    straddles two surfaces. Middlebury masks those out too.

    Colour is averaged over sub-samples (anti-aliasing, as a real sensor does),
    while depth is *point sampled* at one sub-position. Averaging depth across
    an occlusion boundary would invent a surface that is not in the scene.
    """
    ss = max(1, int(supersample))
    forward, _, _ = camera.basis()

    hi_cam = Camera(
        position=camera.position,
        target=camera.target,
        focal_px=camera.focal_px * ss,
    )
    dirs = ray_grid(hi_cam, height * ss, width * ss)
    origin = np.asarray(camera.position, dtype=np.float32)

    color, dist = trace(scene, origin, dirs, shadows=shadows, bounces=bounces)
    # Along-ray distance -> along-axis depth.
    depth = dist * (dirs @ forward)

    color = color.reshape(height * ss, width * ss, 3)
    depth = depth.reshape(height * ss, width * ss)

    if scene.fog_density > 0:
        with np.errstate(over="ignore"):
            fog = 1.0 - np.exp(-np.float32(scene.fog_density) * np.nan_to_num(
                depth, nan=0.0, posinf=1e6
            ))
        fc = np.asarray(scene.fog_color, dtype=np.float32)
        color = color * (1.0 - fog[..., None]) + fc[None, None, :] * fog[..., None]

    if ss == 1:
        return color, depth, np.zeros((height, width), dtype=np.float32)

    blocks = depth.reshape(height, ss, width, ss)
    rgb = color.reshape(height, ss, width, ss, 3).mean(axis=(1, 3))
    picked = blocks[:, ss // 2, :, ss // 2]
    finite = np.isfinite(blocks)
    with np.errstate(invalid="ignore"):
        lo = np.where(finite, blocks, np.inf).min(axis=(1, 3))
        hi = np.where(finite, blocks, -np.inf).max(axis=(1, 3))
        spread = np.where(
            np.isfinite(lo) & np.isfinite(hi),
            (hi - lo) / np.maximum(np.abs(picked), 1e-6),
            0.0,
        )
    # A pixel that is part sky and part surface also straddles a boundary.
    spread = np.where(finite.all(axis=(1, 3)), spread, np.inf)
    return rgb.astype(np.float32), picked.astype(np.float32), spread.astype(np.float32)


def fit_baseline(
    depth: np.ndarray, focal_px: float, target_max_disparity: float
) -> float:
    """Choose a baseline so the scene's disparities fill the search range.

    A fixed baseline makes the disparity range a lottery: the same camera rig
    gives 8 px of disparity in a corridor and 400 px on a tabletop, so half a
    generated dataset either wastes the search range or overflows it. Reading
    the near depth off the rendered depth map and solving
    ``B = d_max * Z_near / f`` pins the maximum disparity where the caller
    asked for it, whatever the scene turned out to look like.

    The 2nd percentile rather than the minimum: one stray pixel clipping the
    corner of a nearby object should not halve the baseline for the whole
    image.
    """
    finite = depth[np.isfinite(depth) & (depth > 1e-3)]
    if finite.size == 0:
        return 0.1
    z_near = float(np.percentile(finite, 2.0))
    return max(1e-4, float(target_max_disparity) * z_near / float(focal_px))


@dataclass
class StereoRender:
    """The raw output of one stereo render, before any sensor simulation."""

    left_rgb: np.ndarray      # (H, W, 3) linear float
    right_rgb: np.ndarray
    depth_left: np.ndarray    # (H, W) along-axis depth, inf where sky
    depth_right: np.ndarray
    disparity: np.ndarray     # (H, W) float32, nan where unknown
    nonocc: np.ndarray        # (H, W) bool: visible in both views
    edge_pixels: np.ndarray   # (H, W) bool: pixel straddles a depth boundary
    baseline: float
    focal_px: float


def render_stereo(
    scene: Scene,
    height: int,
    width: int,
    *,
    baseline: float | None = None,
    target_max_disparity: float = 60.0,
    supersample: int = 1,
    shadows: bool = True,
    bounces: int = 0,
    edge_spread_tol: float = 0.02,
) -> StereoRender:
    """Render a rectified pair with exact disparity and a true occlusion mask.

    With ``baseline=None`` the rig is sized to the scene: the left view is
    traced first, its near depth sets the baseline via :func:`fit_baseline`,
    and only then is the right view rendered. The left render is not repeated,
    so fitting the baseline is nearly free.
    """
    cam = scene.camera
    forward, right, _ = cam.basis()

    left_rgb, depth_l, spread_l = render_view(
        scene, cam, height, width,
        supersample=supersample, shadows=shadows, bounces=bounces,
    )

    if baseline is None:
        baseline = fit_baseline(depth_l, cam.focal_px, target_max_disparity)
    baseline = float(baseline)

    right_cam = Camera(
        position=tuple(
            float(v) for v in np.asarray(cam.position, dtype=np.float32)
            + right * np.float32(baseline)
        ),
        target=tuple(
            float(v) for v in np.asarray(cam.target, dtype=np.float32)
            + right * np.float32(baseline)
        ),
        focal_px=cam.focal_px,
    )
    right_rgb, depth_r, _ = render_view(
        scene, right_cam, height, width,
        supersample=supersample, shadows=shadows, bounces=bounces,
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        disparity = (np.float32(cam.focal_px) * np.float32(baseline) / depth_l)
    disparity = np.where(np.isfinite(depth_l), disparity, np.nan).astype(np.float32)

    nonocc = _occlusion_mask(disparity, depth_l, depth_r)
    edge_pixels = ~np.isfinite(spread_l) | (spread_l > edge_spread_tol)

    return StereoRender(
        left_rgb=left_rgb,
        right_rgb=right_rgb,
        depth_left=depth_l,
        depth_right=depth_r,
        disparity=disparity,
        nonocc=nonocc,
        edge_pixels=edge_pixels,
        baseline=baseline,
        focal_px=float(cam.focal_px),
    )


def _occlusion_mask(
    disparity: np.ndarray,
    depth_left: np.ndarray,
    depth_right: np.ndarray,
    *,
    rel_tol: float = 0.02,
) -> np.ndarray:
    """True where a left pixel's surface is also visible to the right camera.

    This is a geometric test, not a heuristic. Walk each left pixel to where
    its match must be, ``x - d``, and ask the right camera's depth map what it
    sees there. If the right camera sees the *same* depth, the point is visible
    in both views. If it sees something closer, that something is standing in
    the way and the point is occluded. Because both depth maps come from real
    ray casts, this reproduces exactly the regions a real stereo rig cannot
    match -- which is the region Middlebury's "nonocc" score is computed over.
    """
    h, w = disparity.shape
    xs = np.arange(w, dtype=np.float32)[None, :].repeat(h, axis=0)
    xr = xs - disparity

    inside = np.isfinite(xr) & (xr >= 0) & (xr <= w - 1)
    xi = np.clip(np.nan_to_num(xr, nan=0.0), 0, w - 1)
    x0 = np.floor(xi).astype(np.int64)
    x1 = np.minimum(x0 + 1, w - 1)
    rows = np.arange(h)[:, None]

    # Nearest of the two bracketing samples, in depth terms: a bilinear blend
    # across an occlusion edge in the *right* depth map would be as wrong here
    # as averaging depth inside a pixel.
    d0 = depth_right[rows, x0]
    d1 = depth_right[rows, x1]
    zl = depth_left
    with np.errstate(invalid="ignore"):
        pick_d1 = np.abs(d1 - zl) < np.abs(d0 - zl)
    z_right = np.where(pick_d1, d1, d0)

    with np.errstate(invalid="ignore"):
        agrees = np.abs(z_right - zl) <= rel_tol * np.maximum(zl, 1e-6)
    return (inside & np.isfinite(zl) & np.isfinite(z_right) & agrees)


#: Luminance weights used for metering.
_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def auto_exposure(
    linear: np.ndarray,
    *,
    mid_target: float = 0.34,
    highlight_target: float = 1.5,
    highlight_percentile: float = 99.5,
) -> float:
    """Gain that lands the scene in the usable part of the tone curve.

    Without this, exposure is whatever the light intensities happen to add up
    to, and no fixed rig can be right for every environment at once: the same
    lights that expose a 1.5 m tabletop leave a 40 m corridor at a tenth of
    the range, where the transfer curve has so little slope that surface
    texture survives as two or three grey levels. Metering the render is what
    a real camera does, and it makes texture contrast depend on the *rules*
    rather than on how far away the scene happened to be.

    Two meters, and the darker of the two wins. Metering the median alone
    blows out a scene with a bright window in it; metering the highlights
    alone leaves a mostly-dark scene underexposed. Taking the minimum exposes
    for the midtones *unless* that would clip the highlights, and clipping is
    the one failure that destroys texture outright instead of merely dimming
    it -- a saturated region has no gradient left to match on.
    """
    lum = np.asarray(linear, dtype=np.float32) @ _LUMA
    lum = lum[np.isfinite(lum)]
    if lum.size == 0:
        return 1.0
    mid = float(np.median(lum))
    high = float(np.percentile(lum, highlight_percentile))
    candidates = []
    if mid > 1e-5:
        candidates.append(mid_target / mid)
    if high > 1e-5:
        candidates.append(highlight_target / high)
    if not candidates:
        return 1.0
    return float(np.clip(min(candidates), 0.05, 2000.0))


def to_srgb8(
    linear: np.ndarray, exposure: float = 1.0, white: float = 2.0
) -> np.ndarray:
    """Tone-map linear radiance to 8-bit sRGB.

    The transfer function is not cosmetic. Matching costs are computed on the
    stored pixel values, so it decides how much surface texture survives --
    and training data whose texture is crushed into a few grey levels teaches
    a matcher that real surfaces are untextured.

    Extended Reinhard, which is linear near zero and rolls off only as it
    approaches ``white``, rather than plain ``c / (1 + c)``, which starts
    compressing immediately and halves the contrast of a mid-grey surface
    before it has reached a quarter of the output range.
    """
    c = np.clip(np.asarray(linear, dtype=np.float32) * np.float32(exposure), 0.0, None)
    w = np.float32(max(white, 1.01))
    c = np.clip(c * (1.0 + c / (w * w)) / (1.0 + c), 0.0, 1.0)
    srgb = np.where(
        c <= 0.0031308, c * 12.92, 1.055 * np.power(np.maximum(c, 1e-8), 1 / 2.4) - 0.055
    )
    return np.clip(srgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
