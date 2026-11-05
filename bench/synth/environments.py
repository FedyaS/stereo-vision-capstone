"""Scene samplers: turn a rule set and a seed into concrete 3-D geometry.

Each environment is a different *shape of depth distribution*, which is the
axis along which real stereo benchmarks actually differ. Subject matter is
cosmetic; what changes a matcher's error rate is whether depth is bimodal
(tabletop: objects, then a wall), near-uniform over a huge range (corridor),
or mostly smooth with sparse cliffs (terrain). Training on one and testing on
another is where stereo methods fall over, so the generator makes that axis
explicit and easy to sweep.

Everything is driven by a single :class:`numpy.random.Generator` seeded from
the rules, so a scene is reproducible from ``(rules, seed)`` alone.
"""
from __future__ import annotations

import math

import numpy as np

from .materials import Light, sample_material
from .render import Camera, Scene
from .rules import SceneRules
from .shapes import Box, Cylinder, Plane, Shape, Sphere


#: Target spacing of texture detail, in image pixels, at the surface's
#: expected distance. The low end is near the Nyquist limit of a supersampled
#: render; the high end is coarse blotches. Matchers want the low end --
#: a correlation window needs several features inside it -- but a dataset
#: made only of fine detail is unrealistically easy, so a range is sampled.
_DETAIL_PX = (3.5, 16.0)


def texture_scale_for(rules: SceneRules, ref_depth: float, px_per_cycle: float) -> float:
    """Texture frequency, in cycles per metre, that puts one cycle every
    ``px_per_cycle`` pixels on a surface ``ref_depth`` metres away.

    This is the piece that makes texture *matchable*. Defining scale in world
    units instead -- "four cycles per metre" -- seems natural and is a trap:
    the same number gives dense detail on a nearby tabletop and a flat wash on
    a corridor wall thirty metres off, so half the environments come out with
    no usable signal. Projecting through the camera makes the one knob the
    user sees (``texture_scale``) mean the same thing everywhere.
    """
    cycles_per_px = 1.0 / max(px_per_cycle, 0.75)
    return float(rules.focal_px * cycles_per_px / max(ref_depth, 0.05))


def _material(
    rules: SceneRules,
    rng: np.random.Generator,
    *,
    ref_depth: float,
    detail_px: tuple[float, float] = _DETAIL_PX,
):
    """Draw a material honouring the appearance rules.

    ``textureless_fraction`` is applied here rather than inside
    :func:`sample_material` so that every surface in every environment is
    subject to it, including floors and walls -- which is where real
    untextured regions overwhelmingly are.
    """
    flat = rng.random() < rules.textureless_fraction
    kinds = ("flat",) if flat else tuple(rules.texture_kinds)
    # texture_scale > 1 means finer detail, so it divides the pixel spacing.
    px = float(rng.uniform(*detail_px)) / max(rules.texture_scale, 1e-3)
    # Surface grain sits just above the Nyquist limit of the render: fine
    # enough to fill a correlation window, coarse enough not to alias into
    # a difference between the two views.
    micro_px = 2.0 * max(rules.supersample, 1) / max(rules.texture_scale, 1e-3)
    return sample_material(
        rng,
        kinds=kinds,
        richness=0.0 if flat else rules.texture_richness,
        scale=texture_scale_for(rules, ref_depth, px),
        micro_scale=texture_scale_for(rules, ref_depth, micro_px),
        specular=rules.specular_strength,
        reflective=(not flat) and rng.random() < rules.reflective_fraction,
        palette_warmth=rules.palette_warmth,
    )


def _n_objects(rules: SceneRules, rng: np.random.Generator) -> int:
    if rules.max_objects <= rules.min_objects:
        return int(rules.min_objects)
    return int(rng.integers(rules.min_objects, rules.max_objects + 1))


def _place_apart(
    rng: np.random.Generator,
    n: int,
    x_range: tuple[float, float],
    z_range: tuple[float, float],
    min_gap: float,
    tries: int = 24,
) -> list[tuple[float, float]]:
    """Rejection-sample ``n`` XZ positions that are not on top of each other.

    Interpenetrating objects are not a rendering problem -- the tracer handles
    them -- but they make the clutter read as one blob, and the point of
    clutter is to produce many *separate* occlusion boundaries.
    """
    spots: list[tuple[float, float]] = []
    for _ in range(n):
        for _ in range(tries):
            x = float(rng.uniform(*x_range))
            z = float(rng.uniform(*z_range))
            if all((x - a) ** 2 + (z - b) ** 2 >= min_gap**2 for a, b in spots):
                spots.append((x, z))
                break
        else:
            spots.append((x, z))  # give up on the gap rather than lose the object
    return spots


def _clutter(
    rules: SceneRules,
    rng: np.random.Generator,
    spots: list[tuple[float, float]],
    *,
    ground_y: float,
    size: tuple[float, float],
    tall_bias: float = 1.0,
) -> list[Shape]:
    """Boxes, cylinders and spheres standing on a ground plane."""
    out: list[Shape] = []
    lo, hi = size[0] * rules.object_scale, size[1] * rules.object_scale
    for x, z in spots:
        s = float(rng.uniform(lo, hi))
        kind = rng.random()
        mat = _material(rules, rng, ref_depth=max(z, 0.4))
        if kind < 0.45:
            h = s * float(rng.uniform(0.6, 2.2)) * tall_bias
            out.append(Box(
                material=mat,
                center=(x, ground_y + h / 2, z),
                half=(s * float(rng.uniform(0.4, 1.0)), h / 2,
                      s * float(rng.uniform(0.4, 1.0))),
                yaw=float(rng.uniform(0, math.pi)),
            ))
        elif kind < 0.75:
            h = s * float(rng.uniform(0.8, 2.4)) * tall_bias
            out.append(Cylinder(
                material=mat,
                base=(x, ground_y, z),
                radius=s * float(rng.uniform(0.25, 0.6)),
                height=h,
            ))
        else:
            r = s * float(rng.uniform(0.3, 0.6))
            out.append(Sphere(material=mat, center=(x, ground_y + r, z), radius=r))
    return out


def _lights(
    rules: SceneRules,
    rng: np.random.Generator,
    *,
    centre: tuple[float, float, float],
    extent: float,
    outdoor: bool = False,
) -> list[Light]:
    """Light the scene at a brightness appropriate to its physical size.

    The falloff constant is derived from ``extent`` so that a 1.5 m tabletop
    and a 40 m corridor both come out with usable contrast; a fixed constant
    would blow out one and black out the other.
    """
    lights: list[Light] = []
    n = int(rules.light_count)
    if n == 0:
        return lights

    warm = 0.08 * (rules.palette_warmth - 0.5)
    if outdoor:
        # A sun, with no distance falloff, plus optional fill lights.
        elev = float(rng.uniform(0.5, 1.4))
        azim = float(rng.uniform(0, 2 * math.pi))
        lights.append(Light(
            position=(
                math.cos(azim) * math.cos(elev),
                math.sin(elev),
                math.sin(azim) * math.cos(elev),
            ),
            color=(1.0 + warm, 0.98, 0.94 - warm),
            intensity=float(rng.uniform(0.9, 1.3)),
            directional=True,
            casts_shadow=rules.shadows,
        ))
        n -= 1

    falloff = 0.6 / max(extent, 0.2) ** 2
    cx, cy, cz = centre
    for i in range(max(0, n)):
        lights.append(Light(
            position=(
                cx + float(rng.uniform(-0.8, 0.8)) * extent,
                cy + float(rng.uniform(0.5, 1.4)) * extent * 0.4,
                cz + float(rng.uniform(-0.8, 0.8)) * extent,
            ),
            color=(1.0 + warm, 0.97, 0.93 - warm),
            intensity=float(rng.uniform(0.8, 1.4)) / (1 if i == 0 else 2),
            casts_shadow=rules.shadows and i == 0,
            falloff=falloff,
        ))
    return lights


def _jitter_camera(
    rng: np.random.Generator,
    rules: SceneRules,
    position: tuple[float, float, float],
    target: tuple[float, float, float],
    *,
    move: float,
    swing: float,
) -> Camera:
    """Perturb a nominal viewpoint by ``camera_variation``.

    The rig is rectified by construction -- the right camera is displaced
    along the left camera's own right axis -- so moving the camera freely here
    cannot break the epipolar geometry. It only changes what is in frame.
    """
    v = float(rules.camera_variation)
    p = np.asarray(position, dtype=np.float64) + rng.uniform(-move, move, 3) * v
    t = np.asarray(target, dtype=np.float64) + rng.uniform(-swing, swing, 3) * v
    return Camera(
        position=tuple(float(x) for x in p),
        target=tuple(float(x) for x in t),
        focal_px=rules.focal_px,
    )


# ----------------------------------------------------------------------
# the environments
# ----------------------------------------------------------------------


def _tabletop(rules: SceneRules, rng: np.random.Generator) -> Scene:
    table = Plane(point=(0, 0, 0), normal=(0, 1, 0),
                  material=_material(rules, rng, ref_depth=1.0))
    back = Plane(point=(0, 0, 2.2), normal=(0, 0, -1),
                 material=_material(rules, rng, ref_depth=2.4))
    shapes: list[Shape] = [table, back]
    spots = _place_apart(rng, _n_objects(rules, rng), (-0.75, 0.75), (0.25, 1.9), 0.17)
    shapes += _clutter(rules, rng, spots, ground_y=0.0, size=(0.10, 0.30))

    cam = _jitter_camera(
        rng, rules, (0.0, 0.42, -0.85), (0.0, 0.14, 0.75), move=0.18, swing=0.25
    )
    return Scene(
        shapes=shapes,
        lights=_lights(rules, rng, centre=(0, 0.6, 0.7), extent=1.6),
        camera=cam,
        ambient=rules.ambient,
        fog_density=rules.fog_density,
        depth_hint=(0.6, 3.2),
    )


def _room(rules: SceneRules, rng: np.random.Generator) -> Scene:
    w, d, h = 7.0, 11.0, 2.9
    shapes: list[Shape] = [
        Plane(point=(0, 0, 0), normal=(0, 1, 0), material=_material(rules, rng, ref_depth=4.0)),
        Plane(point=(0, h, 0), normal=(0, -1, 0), material=_material(rules, rng, ref_depth=5.0)),
        Plane(point=(0, 0, d), normal=(0, 0, -1), material=_material(rules, rng, ref_depth=d)),
        Plane(point=(-w / 2, 0, 0), normal=(1, 0, 0), material=_material(rules, rng, ref_depth=5.0)),
        Plane(point=(w / 2, 0, 0), normal=(-1, 0, 0), material=_material(rules, rng, ref_depth=5.0)),
    ]
    spots = _place_apart(rng, _n_objects(rules, rng), (-2.7, 2.7), (0.8, d - 0.8), 0.95)
    shapes += _clutter(rules, rng, spots, ground_y=0.0, size=(0.5, 1.4))

    cam = _jitter_camera(
        rng, rules, (0.0, 1.55, -1.2), (0.0, 1.1, 4.0), move=1.1, swing=1.6
    )
    return Scene(
        shapes=shapes,
        lights=_lights(rules, rng, centre=(0, h * 0.8, d * 0.4), extent=5.0),
        camera=cam,
        ambient=rules.ambient,
        fog_density=rules.fog_density,
        depth_hint=(1.0, 13.0),
    )


def _corridor(rules: SceneRules, rng: np.random.Generator) -> Scene:
    half_w, h, length = 1.7, 3.0, 38.0
    shapes: list[Shape] = [
        Plane(point=(0, 0, 0), normal=(0, 1, 0), material=_material(rules, rng, ref_depth=5.0)),
        Plane(point=(0, h, 0), normal=(0, -1, 0), material=_material(rules, rng, ref_depth=6.0)),
        Plane(point=(-half_w, 0, 0), normal=(1, 0, 0), material=_material(rules, rng, ref_depth=5.0)),
        Plane(point=(half_w, 0, 0), normal=(-1, 0, 0), material=_material(rules, rng, ref_depth=5.0)),
        Plane(point=(0, 0, length), normal=(0, 0, -1), material=_material(rules, rng, ref_depth=length)),
    ]
    # Pillars and crates hugging the walls, spread over the full length so the
    # depth histogram stays populated all the way to the far end.
    for _ in range(_n_objects(rules, rng)):
        side = -1.0 if rng.random() < 0.5 else 1.0
        z = float(rng.uniform(1.5, length - 2.0))
        x = side * float(rng.uniform(half_w * 0.55, half_w * 0.92))
        s = float(rng.uniform(0.18, 0.5)) * rules.object_scale
        if rng.random() < 0.5:
            tall = float(rng.uniform(0.8, 2.6))
            shapes.append(Box(
                material=_material(rules, rng, ref_depth=max(z, 1.0)),
                center=(x, tall / 2, z), half=(s, tall / 2, s),
                yaw=float(rng.uniform(0, math.pi)),
            ))
        else:
            shapes.append(Cylinder(
                material=_material(rules, rng, ref_depth=max(z, 1.0)),
                base=(x, 0.0, z), radius=s * 0.8,
                height=float(rng.uniform(1.0, h)),
            ))

    cam = _jitter_camera(
        rng, rules, (0.0, 1.6, 0.5), (0.0, 1.4, 12.0), move=0.55, swing=2.2
    )
    return Scene(
        shapes=shapes,
        lights=_lights(rules, rng, centre=(0, h * 0.85, 7.0), extent=7.0),
        camera=cam,
        ambient=rules.ambient,
        fog_density=rules.fog_density,
        depth_hint=(1.0, 40.0),
    )


def _warehouse(rules: SceneRules, rng: np.random.Generator) -> Scene:
    """Repeated shelving: the point is *ambiguity*, not realism.

    Near-identical units at a regular spacing along the viewing direction put
    many equally good matches on the same epipolar line, which is the failure
    mode that separates a matcher with a global/semi-global prior from a
    local one. The units share one material on purpose.
    """
    h, d = 5.0, 34.0
    floor_mat = _material(rules, rng, ref_depth=6.0)
    shelf_mat = _material(rules, rng, ref_depth=8.0)
    shapes: list[Shape] = [
        Plane(point=(0, 0, 0), normal=(0, 1, 0), material=floor_mat),
        Plane(point=(0, h, 0), normal=(0, -1, 0), material=_material(rules, rng, ref_depth=8.0)),
        Plane(point=(0, 0, d), normal=(0, 0, -1), material=_material(rules, rng, ref_depth=d)),
    ]
    pitch = float(rng.uniform(3.0, 4.5))
    aisle = float(rng.uniform(2.2, 3.2))
    n_units = int(min(14, max(4, (d - 3) // pitch)))
    for i in range(n_units):
        z = 3.0 + i * pitch
        for side in (-1.0, 1.0):
            shelf_h = h * float(rng.uniform(0.62, 0.78))
            shapes.append(Box(
                material=shelf_mat,
                center=(side * (aisle + 0.7), shelf_h / 2, z),
                half=(0.7, shelf_h / 2, pitch * 0.33),
                yaw=0.0,
            ))
    # A little non-repeating clutter in the aisle, so the scene is ambiguous
    # rather than literally periodic -- otherwise there is no correct answer.
    spots = _place_apart(rng, _n_objects(rules, rng) // 2, (-aisle, aisle), (2.0, d - 3), 1.3)
    shapes += _clutter(rules, rng, spots, ground_y=0.0, size=(0.4, 1.1))

    cam = _jitter_camera(
        rng, rules, (0.0, 1.7, 0.2), (0.0, 1.6, 10.0), move=0.7, swing=2.0
    )
    return Scene(
        shapes=shapes,
        lights=_lights(rules, rng, centre=(0, h * 0.9, 9.0), extent=9.0),
        camera=cam,
        ambient=rules.ambient,
        fog_density=rules.fog_density,
        depth_hint=(1.5, 36.0),
    )


def _street(rules: SceneRules, rng: np.random.Generator) -> Scene:
    """Open sky, so some pixels legitimately have no ground truth.

    Real outdoor stereo data always has a sky region where disparity is
    undefined, and a dataset without one teaches a model that every pixel is
    matchable. Here the sky is literally a ray that hits nothing, and the
    ground truth is ``nan`` there.
    """
    shapes: list[Shape] = [
        Plane(point=(0, 0, 0), normal=(0, 1, 0), material=_material(rules, rng, ref_depth=9.0)),
    ]
    for _ in range(_n_objects(rules, rng)):
        side = -1.0 if rng.random() < 0.5 else 1.0
        z = float(rng.uniform(4.0, 55.0))
        width = float(rng.uniform(2.5, 6.0)) * rules.object_scale
        height = float(rng.uniform(4.0, 16.0)) * rules.object_scale
        x = side * (float(rng.uniform(5.0, 9.0)) + width / 2)
        shapes.append(Box(
            material=_material(rules, rng, ref_depth=max(z, 2.0)),
            center=(x, height / 2, z),
            half=(width / 2, height / 2, float(rng.uniform(2.0, 6.0))),
            yaw=float(rng.uniform(-0.12, 0.12)),
        ))
    # Street furniture down the middle distance: the only near geometry, and
    # therefore what sets the disparity range.
    spots = _place_apart(rng, 5, (-3.5, 3.5), (3.0, 25.0), 2.5)
    shapes += _clutter(rules, rng, spots, ground_y=0.0, size=(0.3, 0.9), tall_bias=1.6)

    sky_hi = (0.30, 0.47, 0.78)
    sky_lo = (0.72, 0.80, 0.90)
    cam = _jitter_camera(
        rng, rules, (0.0, 1.7, -2.0), (0.0, 1.5, 14.0), move=1.4, swing=3.0
    )
    return Scene(
        shapes=shapes,
        lights=_lights(rules, rng, centre=(0, 12.0, 15.0), extent=20.0, outdoor=True),
        camera=cam,
        ambient=rules.ambient,
        sky_top=sky_hi,
        sky_bottom=sky_lo,
        fog_density=rules.fog_density,
        fog_color=(0.76, 0.81, 0.88),
        depth_hint=(2.0, 70.0),
    )


def _terrain(rules: SceneRules, rng: np.random.Generator) -> Scene:
    """Rolling ground under open sky, built from part-buried spheres.

    Half-sunk spheres give smoothly curved mounds whose silhouettes against
    the sky are exact analytic curves -- a depth field that is mostly smooth
    with occasional hard cliffs, which is what outdoor depth actually looks
    like, and which no plane-and-box scene reproduces.
    """
    ground = _material(rules, rng, ref_depth=8.0)
    shapes: list[Shape] = [Plane(point=(0, 0, 0), normal=(0, 1, 0), material=ground)]

    for _ in range(int(rng.integers(3, 7))):
        r = float(rng.uniform(4.0, 11.0)) * rules.object_scale
        shapes.append(Sphere(
            material=_material(rules, rng, ref_depth=18.0),
            center=(float(rng.uniform(-22, 22)), -r * float(rng.uniform(0.55, 0.8)),
                    float(rng.uniform(8, 48))),
            radius=r,
        ))
    spots = _place_apart(rng, _n_objects(rules, rng), (-14, 14), (2.5, 30.0), 1.6)
    shapes += _clutter(rules, rng, spots, ground_y=0.0, size=(0.3, 1.2))

    cam = _jitter_camera(
        rng, rules, (0.0, 1.8, -1.0), (0.0, 1.3, 12.0), move=1.2, swing=3.5
    )
    return Scene(
        shapes=shapes,
        lights=_lights(rules, rng, centre=(0, 14.0, 12.0), extent=18.0, outdoor=True),
        camera=cam,
        ambient=rules.ambient,
        sky_top=(0.38, 0.54, 0.80),
        sky_bottom=(0.80, 0.84, 0.86),
        fog_density=max(rules.fog_density, 0.004),
        fog_color=(0.80, 0.84, 0.88),
        depth_hint=(1.5, 60.0),
    )


def _void(rules: SceneRules, rng: np.random.Generator) -> Scene:
    """Floating primitives against a backdrop: every edge is an occlusion edge.

    No floor means nothing connects an object to its background, so there are
    no supporting cues and no continuous surface leading the eye in. It is the
    hardest case for anything that propagates disparity along surfaces, and
    the closest analogue to the warp-based generator this package replaces.
    """
    shapes: list[Shape] = [
        Plane(point=(0, 0, 13.0), normal=(0, 0, -1),
              material=_material(rules, rng, ref_depth=13.0)),
    ]
    for _ in range(_n_objects(rules, rng)):
        z = float(rng.uniform(3.0, 11.0))
        x = float(rng.uniform(-3.2, 3.2))
        y = float(rng.uniform(-2.2, 2.2))
        s = float(rng.uniform(0.35, 1.1)) * rules.object_scale
        mat = _material(rules, rng, ref_depth=max(z, 1.0))
        pick = rng.random()
        if pick < 0.4:
            shapes.append(Sphere(material=mat, center=(x, y, z), radius=s * 0.7))
        elif pick < 0.75:
            shapes.append(Box(
                material=mat, center=(x, y, z),
                half=(s * float(rng.uniform(0.4, 0.9)), s * float(rng.uniform(0.4, 0.9)),
                      s * float(rng.uniform(0.4, 0.9))),
                yaw=float(rng.uniform(0, math.pi)),
            ))
        else:
            shapes.append(Cylinder(
                material=mat, base=(x, y - s, z), radius=s * 0.5, height=s * 2.0
            ))

    cam = _jitter_camera(
        rng, rules, (0.0, 0.0, 0.0), (0.0, 0.0, 8.0), move=0.6, swing=1.2
    )
    return Scene(
        shapes=shapes,
        lights=_lights(rules, rng, centre=(0, 3.0, 4.0), extent=6.0),
        camera=cam,
        ambient=rules.ambient,
        fog_density=rules.fog_density,
        depth_hint=(2.5, 14.0),
    )


_BUILDERS = {
    "tabletop": _tabletop,
    "room": _room,
    "corridor": _corridor,
    "warehouse": _warehouse,
    "street": _street,
    "terrain": _terrain,
    "void": _void,
}


def build_scene(rules: SceneRules, seed: int) -> Scene:
    """Sample one concrete scene from ``rules`` at ``seed``.

    Deterministic: the same arguments always give the same geometry,
    materials, lighting and viewpoint.
    """
    try:
        builder = _BUILDERS[rules.environment]
    except KeyError:  # pragma: no cover - validate() rejects these first
        raise ValueError(f"unknown environment {rules.environment!r}") from None
    return builder(rules, np.random.default_rng(int(seed)))
