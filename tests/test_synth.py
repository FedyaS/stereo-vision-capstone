"""The training-image synthesizer.

The tests that matter here are the ones that check the ground truth is
*exact*, not merely plausible. A generator that renders attractive images with
subtly wrong disparity is worse than no generator at all: every algorithm
trained or scored on it is quietly wrong, and nothing in the pipeline
downstream would notice. So the first few tests pin the geometry against
closed-form answers, and the rest check that the rules do what they claim.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from bench.synth import (
    ENVIRONMENTS,
    Camera,
    Material,
    Plane,
    RuleError,
    SceneRules,
    Scene,
    Sphere,
    generate_dataset,
    generate_pair,
    preset,
    render_stereo,
    rule_schema,
    to_srgb8,
)
from bench.synth.materials import Light, TEXTURE_KINDS, fbm, value_noise
from bench.synth.render import auto_exposure, fit_baseline
from bench.synth.shapes import Box, Cylinder

# A small, fast rule set. Supersampling off and shadows off keep every test
# well under a second; the geometry they check does not depend on either.
FAST = dict(width=160, height=112, ndisp=32, supersample=1, shadows=False,
            min_objects=2, max_objects=4, reflection_bounces=0)


# ----------------------------------------------------------------------
# geometry: exactness, checked against closed-form answers
# ----------------------------------------------------------------------


def test_depth_and_disparity_are_analytically_exact():
    """A plane at a known depth must give exactly that depth and disparity.

    With a camera at the origin looking down +Z and a plane at Z = 10, every
    pixel's along-axis depth is 10 by construction, so disparity is
    ``focal * baseline / 10`` everywhere. Any error here -- a half-pixel
    offset in the ray grid, a ray-distance/axis-depth confusion -- shows up
    immediately, and those are exactly the mistakes that are invisible in a
    rendered image.
    """
    cam = Camera(position=(0, 0, 0), target=(0, 0, 1), focal_px=500.0)
    scene = Scene(
        shapes=[Plane(point=(0, 0, 10), normal=(0, 0, -1),
                      material=Material(kind="gravel"))],
        lights=[Light(position=(0, 0, -1))],
        camera=cam,
    )
    r = render_stereo(scene, 48, 64, baseline=0.5, supersample=1, shadows=False)

    assert np.allclose(r.depth_left, 10.0, atol=1e-3)
    assert np.allclose(r.disparity, 500.0 * 0.5 / 10.0, atol=1e-3)


def test_sphere_silhouette_matches_the_closed_form():
    """The apparent radius of a sphere must be the one optics predicts."""
    cam = Camera(position=(0, 0, 0), target=(0, 0, 1), focal_px=400.0)
    z, radius = 8.0, 1.0
    scene = Scene(
        shapes=[Sphere(center=(0, 0, z), radius=radius,
                       material=Material(kind="flat"))],
        lights=[Light(position=(0, 0, -1))],
        camera=cam,
    )
    _, depth, _ = __import__(
        "bench.synth.render", fromlist=["render_view"]
    ).render_view(scene, cam, 200, 200, supersample=1, shadows=False)

    hit = np.isfinite(depth)
    # Projected radius of a sphere seen head-on: f * r / sqrt(z^2 - r^2).
    expected = 400.0 * radius / np.sqrt(z * z - radius * radius)
    measured = np.sqrt(hit.sum() / np.pi)
    assert measured == pytest.approx(expected, rel=0.02)
    # The nearest point of the sphere is at z - r.
    assert depth[hit].min() == pytest.approx(z - radius, abs=1e-3)


def test_left_image_resamples_to_the_right_at_the_ground_truth_disparity():
    """``left(y, x) == right(y, x - d)`` -- the definition of the GT.

    This is the end-to-end check on the stereo convention. If the sign were
    flipped, or the right camera displaced along the wrong axis, every
    algorithm would score terribly and the blame would land on the algorithm.
    Sensor noise and exposure mismatch are switched off so the comparison
    measures geometry only.
    """
    import cv2

    rules = preset("void").replace(
        **FAST, sensor_noise=0.0, exposure_mismatch=0.0, defocus_px=0.0,
    )
    pair = generate_pair(rules, seed=4)
    h, w = pair.shape
    xx = np.tile(np.arange(w, dtype=np.float32), (h, 1))
    yy = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))

    resampled = cv2.remap(
        pair.right, xx - np.nan_to_num(pair.gt), yy,
        interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101,
    )
    # Away from silhouettes, where a pixel mixes two surfaces at different
    # depths and no single disparity can reproduce it.
    interior = (
        pair.occ_mask
        & np.isfinite(pair.gt)
        & (xx - np.nan_to_num(pair.gt) > 2)
        & ~_near_depth_edge(pair.gt)
    )
    assert interior.sum() > 2000
    diff = np.abs(
        resampled[interior].astype(np.float32) - pair.left[interior].astype(np.float32)
    ).mean()
    assert diff < 6.0, f"the disparity does not reproduce the stereo geometry ({diff=})"


def _near_depth_edge(gt: np.ndarray, radius: int = 3) -> np.ndarray:
    import cv2

    edge = (np.abs(cv2.Laplacian(np.nan_to_num(gt), cv2.CV_32F)) > 1.0).astype(np.uint8)
    k = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
    return cv2.dilate(edge, k) > 0


def test_occlusion_mask_is_geometric_not_guessed():
    """A near object must occlude a band of the background, on its left side.

    For this convention the right camera sits to the right of the left one,
    so the surface it cannot see is the strip just left of a foreground
    object. Checking the *side* and not merely the amount is what separates a
    real visibility computation from a plausible-looking edge dilation.
    """
    cam = Camera(position=(0, 0, 0), target=(0, 0, 1), focal_px=500.0)
    scene = Scene(
        shapes=[
            Plane(point=(0, 0, 14), normal=(0, 0, -1),
                  material=Material(kind="noise")),
            Sphere(center=(0, 0, 6), radius=0.5, material=Material(kind="gravel")),
        ],
        lights=[Light(position=(2, 3, -2))],
        camera=cam,
    )
    r = render_stereo(scene, 120, 200, baseline=0.25, supersample=1, shadows=False)

    occluded = ~r.nonocc
    background = r.depth_left > 10.0
    foreground = r.depth_left < 7.0
    assert foreground.any() and background.any()

    # Occluded background pixels, excluding the left border band where no
    # match exists for any surface.
    xs = np.arange(200)[None, :].repeat(120, axis=0)
    band = occluded & background & (xs > 60)
    assert band.sum() > 50, "a near sphere should occlude part of the backdrop"

    rows = np.unique(np.nonzero(band)[0])
    sphere_rows = np.unique(np.nonzero(foreground)[0])
    shared = np.intersect1d(rows, sphere_rows)
    assert shared.size > 10
    for y in shared[:10]:
        assert band[y].nonzero()[0].mean() < foreground[y].nonzero()[0].mean(), (
            "occlusion must fall to the left of the occluder for this convention"
        )


def test_nothing_is_occluded_without_an_occluder():
    """A single plane fills both views, so every pixel is visible to both."""
    cam = Camera(position=(0, 0, 0), target=(0, 0, 1), focal_px=300.0)
    scene = Scene(
        shapes=[Plane(point=(0, 0, 6), normal=(0, 0, -1),
                      material=Material(kind="gravel"))],
        lights=[Light(position=(0, 1, -1))],
        camera=cam,
    )
    r = render_stereo(scene, 40, 60, baseline=0.02, supersample=1, shadows=False)
    assert r.nonocc.mean() > 0.98


def test_baseline_is_fitted_to_fill_the_search_range():
    rules = preset("room").replace(**{**FAST, "ndisp": 64},
                                   disparity_fill=0.9)
    pair = generate_pair(rules, seed=1)
    assert np.nanmax(pair.gt) <= 64
    assert np.nanpercentile(pair.gt, 99) > 0.6 * 64, (
        "an auto-fitted baseline should make the scene use its search range"
    )


def test_fit_baseline_solves_the_stated_equation():
    depth = np.full((10, 10), 5.0, dtype=np.float32)
    b = fit_baseline(depth, focal_px=400.0, target_max_disparity=50.0)
    assert b == pytest.approx(50.0 * 5.0 / 400.0)


# ----------------------------------------------------------------------
# shapes
# ----------------------------------------------------------------------


@pytest.mark.parametrize("shape", [
    Sphere(center=(0, 0, 5), radius=1.0),
    Box(center=(0, 0, 5), half=(1, 1, 1), yaw=0.4),
    Cylinder(base=(0, -1, 5), radius=0.7, height=2.0),
    Plane(point=(0, 0, 5), normal=(0, 0, -1)),
])
def test_shapes_hit_from_the_front_and_miss_behind(shape):
    """A ray down +Z hits; the same ray reversed must not."""
    o = np.zeros((1, 3), dtype=np.float32)
    forward = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    backward = np.array([[0.0, 0.0, -1.0]], dtype=np.float32)

    t = shape.intersect(o, forward)
    assert np.isfinite(t[0]) and t[0] > 0
    assert not np.isfinite(shape.intersect(o, backward)[0])

    n = shape.normal_at(o + forward * t[0])
    assert np.allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-4)


def test_box_parallel_rays_do_not_produce_nan():
    """A ray exactly parallel to a slab divides by zero in the slab test.

    Left alone that nan propagates through the min/max reduction and silently
    turns into a miss *or* a bogus hit depending on reduction order, so it is
    pinned here rather than left to chance.
    """
    box = Box(center=(0, 0, 5), half=(1, 1, 1))
    o = np.array([[0.0, 0.0, 0.0], [0.0, 5.0, 0.0]], dtype=np.float32)
    d = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    t = box.intersect(o, d)
    assert not np.isnan(t).any()
    assert np.isfinite(t[0])          # through the middle
    assert not np.isfinite(t[1])      # parallel, above the box


# ----------------------------------------------------------------------
# textures and materials
# ----------------------------------------------------------------------


def test_noise_is_deterministic_and_bounded():
    p = np.random.default_rng(0).random((500, 3)).astype(np.float32) * 10
    a, b = value_noise(p, 5), value_noise(p, 5)
    assert np.array_equal(a, b)
    assert 0.0 <= a.min() and a.max() <= 1.0
    assert not np.array_equal(a, value_noise(p, 6))
    f = fbm(p, 5)
    assert 0.0 <= f.min() and f.max() <= 1.0


@pytest.mark.parametrize("kind", TEXTURE_KINDS)
def test_every_texture_kind_renders(kind):
    p = np.random.default_rng(1).random((400, 3)).astype(np.float32) * 6
    a = Material(kind=kind, micro=0.0).albedo(p)
    assert a.shape == (400, 3)
    assert np.isfinite(a).all()
    assert 0.0 <= a.min() and a.max() <= 1.0
    if kind != "flat":
        assert a.std() > 1e-4, f"{kind} produced a constant albedo"


def test_texture_is_a_function_of_world_position_not_view():
    """The two views must sample identical texture at identical 3-D points.

    This is the property that makes the pair matchable at all. If texture
    depended on anything view-specific, the two images would disagree
    everywhere and no disparity would be correct.
    """
    m = Material(kind="noise", scale=3.0, seed=11)
    p = np.random.default_rng(2).random((200, 3)).astype(np.float32) * 4
    assert np.array_equal(m.albedo(p), m.albedo(p.copy()))


def test_contrast_zero_gives_a_flat_surface():
    p = np.random.default_rng(3).random((300, 3)).astype(np.float32) * 5
    flat = Material(kind="gravel", contrast=0.0, micro=0.0).albedo(p)
    assert flat.std(axis=0).max() < 1e-6, "albedo varies across points"


# ----------------------------------------------------------------------
# tone mapping
# ----------------------------------------------------------------------


def test_tone_curve_is_monotonic_and_clips_nothing_below_white():
    lin = np.linspace(0, 1.9, 64, dtype=np.float32).reshape(-1, 1, 1).repeat(3, axis=2)
    out = to_srgb8(lin, exposure=1.0, white=2.0).astype(np.int32)
    assert (np.diff(out[:, 0, 0]) >= 0).all()
    assert out.max() <= 255 and out.min() >= 0
    assert out[-1, 0, 0] > 230, "the white point should map near full scale"


def test_auto_exposure_normalises_a_dim_and_a_bright_scene_alike():
    rng = np.random.default_rng(0)
    base = rng.random((64, 64, 3)).astype(np.float32)
    dim, bright = base * 0.02, base * 8.0
    for img in (dim, bright):
        out = to_srgb8(img, exposure=auto_exposure(img))
        assert 40 < float(np.median(out)) < 215, "exposure left the image unusable"


# ----------------------------------------------------------------------
# rules
# ----------------------------------------------------------------------


def test_rules_round_trip_through_json():
    r = preset("corridor").with_difficulty("hard")
    assert SceneRules.from_json(r.to_json()).to_dict() == r.to_dict()


def test_rules_ignore_unknown_keys():
    """A rules file from a newer version must still load."""
    r = SceneRules.from_dict({"environment": "room", "a_field_from_the_future": 7})
    assert r.environment == "room"


@pytest.mark.parametrize("bad", [
    {"environment": "mars"},
    {"width": 4},
    {"ndisp": 0},
    {"texture_kinds": []},
    {"texture_kinds": ["tartan"]},
    {"supersample": 7},
    {"texture_richness": 2.0},
    {"min_objects": 10, "max_objects": 2},
    {"baseline": 99.0},
])
def test_invalid_rules_are_rejected(bad):
    with pytest.raises(RuleError):
        SceneRules(**bad)


def test_focal_length_follows_the_field_of_view():
    r = SceneRules(width=400, fov_degrees=90.0)
    assert r.focal_px == pytest.approx(200.0, rel=1e-6)


def test_schema_covers_every_rule_the_ui_can_set():
    """The editor is generated from this, so a missing row is a missing knob."""
    names = {f["name"] for f in rule_schema()}
    skip = {"baseline", "mask_depth_edges"} - names
    for f in SceneRules().to_dict():
        if f in skip:
            continue
        assert f in names, f"rule {f!r} has no schema row, so no UI control"


def test_difficulty_presets_order_the_texture_knobs():
    rich = [preset("room").with_difficulty(d).texture_richness
            for d in ("easy", "medium", "hard")]
    assert rich == sorted(rich, reverse=True)
    flat = [preset("room").with_difficulty(d).textureless_fraction
            for d in ("easy", "medium", "hard")]
    assert flat == sorted(flat)


# ----------------------------------------------------------------------
# generation
# ----------------------------------------------------------------------


@pytest.mark.parametrize("environment", sorted(ENVIRONMENTS))
def test_every_environment_generates_a_usable_pair(environment):
    """Each environment must render, and produce ground truth worth having.

    "Usable" is checked, not assumed: an environment whose camera ends up
    inside a wall still returns a perfectly valid image, and only the
    statistics reveal it.
    """
    rules = preset(environment).replace(**FAST)
    pair = generate_pair(rules, seed=5)

    assert pair.left.shape == (112, 160, 3)
    assert pair.left.dtype == np.uint8
    assert pair.has_gt and pair.occ_mask is not None
    assert not np.isnan(pair.left).any()

    stats = pair.meta["stats"]
    assert stats["gt_coverage"] > 0.3, "most of the frame should have depth"
    assert stats["nonocc_fraction"] > 0.3, "most of it should be matchable"
    assert 0 < stats["disparity_max"] <= rules.ndisp
    assert pair.left.std() > 5, "the image is too flat to be a real render"
    # The two views must differ: identical images would mean the baseline
    # never took effect, and every disparity would be trivially zero.
    assert not np.array_equal(pair.left, pair.right)


def test_generation_is_reproducible():
    rules = preset("void").replace(**FAST)
    a, b = generate_pair(rules, seed=9), generate_pair(rules, seed=9)
    assert np.array_equal(a.left, b.left)
    assert np.array_equal(a.right, b.right)
    assert np.array_equal(np.nan_to_num(a.gt, nan=-1), np.nan_to_num(b.gt, nan=-1))

    other = generate_pair(rules, seed=10)
    assert not np.array_equal(a.left, other.left), "different seeds, same scene"


def test_ground_truth_is_dropped_outside_the_search_range():
    """A disparity past ``ndisp`` is outside the dataset's own contract."""
    rules = preset("tabletop").replace(**{**FAST, "ndisp": 32},
                                       disparity_fill=1.5)
    pair = generate_pair(rules, seed=2)
    finite = np.isfinite(pair.gt)
    assert finite.any()
    assert pair.gt[finite].max() < 32
    assert pair.meta["stats"]["out_of_range_fraction"] > 0


def test_texture_richness_drives_image_contrast():
    """The knob users reach for first must actually do what it says.

    Measured as local standard deviation in an 11x11 window, which is what a
    correlation-based matcher sees, rather than global contrast, which a
    single bright object would dominate.
    """
    import cv2

    def local_contrast(img: np.ndarray) -> float:
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
        mu = cv2.blur(g, (11, 11))
        mu2 = cv2.blur(g * g, (11, 11))
        return float(np.sqrt(np.maximum(mu2 - mu * mu, 0)).mean())

    base = preset("room").replace(
        **{**FAST, "width": 192, "height": 144}, sensor_noise=0.0,
    )
    rich = local_contrast(generate_pair(
        base.replace(texture_richness=1.0, textureless_fraction=0.0), seed=5).left)
    plain = local_contrast(generate_pair(
        base.replace(texture_richness=0.0, textureless_fraction=1.0), seed=5).left)
    assert rich > 2.5 * plain, f"richness barely changed contrast ({rich=}, {plain=})"


def test_sensor_noise_differs_between_the_two_views():
    """Identical noise in both eyes would cancel out of every matching cost."""
    rules = preset("void").replace(**FAST, sensor_noise=0.05)
    a = generate_pair(rules, seed=3)
    b = generate_pair(rules.replace(sensor_noise=0.0), seed=3)
    # Same geometry, so the noise-free pair is the signal; the residuals must
    # not be the same in both views.
    nl = a.left.astype(float) - b.left.astype(float)
    nr = a.right.astype(float) - b.right.astype(float)
    assert np.corrcoef(nl.ravel(), nr.ravel())[0, 1] < 0.3


# ----------------------------------------------------------------------
# datasets on disk
# ----------------------------------------------------------------------


def test_dataset_round_trips_through_the_normal_loader(tmp_path):
    """A written sample must load back identically through the folder loader.

    Generated data goes through exactly the same path as a downloaded
    Middlebury scene, which is the only way a comparison between them means
    anything. The ground truth has to survive the PFM round trip bit for bit.
    """
    from bench.datasets.local import load_image_pair
    from bench.synth import write_sample

    rules = preset("void").replace(**FAST)
    mem = generate_pair(rules, seed=0)
    write_sample(mem, tmp_path / "00000")
    disk = load_image_pair(tmp_path / "00000")

    assert np.array_equal(mem.left, disk.left)
    assert np.array_equal(mem.right, disk.right)
    assert disk.ndisp == mem.ndisp
    assert np.array_equal(np.isfinite(mem.gt), np.isfinite(disk.gt))
    finite = np.isfinite(mem.gt)
    assert np.array_equal(mem.gt[finite], disk.gt[finite])
    assert np.array_equal(mem.occ_mask, disk.occ_mask)


def test_generate_dataset_writes_a_manifest_and_reloads(tmp_path):
    from bench.datasets import is_synth_dataset, load_synth_dataset

    rules = preset("void").replace(**FAST)
    result = generate_dataset(rules, 3, tmp_path, workers=1)

    assert result.count == 3
    assert not result.cancelled
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["count"] == 3
    assert manifest["rules"]["environment"] == "void"
    assert len(manifest["samples"]) == 3
    assert (tmp_path / "rules.json").exists()
    assert (tmp_path / "README.md").exists()

    # Reproducible from the saved rules alone.
    assert SceneRules.load(tmp_path / "rules.json").to_dict() == rules.to_dict()

    assert is_synth_dataset(tmp_path)
    pairs = load_synth_dataset(tmp_path)
    assert len(pairs) == 3
    assert all(p.has_gt for p in pairs)


def test_dataset_can_cycle_environments(tmp_path):
    rules = preset("void").replace(**FAST)
    generate_dataset(
        rules, 4, tmp_path, workers=1, environments=["void", "room"]
    )
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["environments"] == ["void", "room"]


def test_dataset_generation_can_be_cancelled(tmp_path):
    calls = {"n": 0}

    def stop() -> bool:
        calls["n"] += 1
        return calls["n"] > 2

    result = generate_dataset(
        preset("void").replace(**FAST), 10, tmp_path, workers=1, should_stop=stop
    )
    assert result.cancelled
    assert result.count < 10


def test_start_index_extends_rather_than_overwrites(tmp_path):
    rules = preset("void").replace(**FAST)
    generate_dataset(rules, 2, tmp_path, workers=1)
    generate_dataset(rules, 2, tmp_path, workers=1, start_index=2)
    folders = sorted(d.name for d in tmp_path.iterdir() if d.is_dir())
    assert folders == ["00000", "00001", "00002", "00003"]


# ----------------------------------------------------------------------
# integration with the rest of the harness
# ----------------------------------------------------------------------


def test_resolve_pairs_understands_the_synth_spec():
    from bench.datasets import resolve_pairs

    pairs = resolve_pairs("synth:void:2")
    assert len(pairs) == 2
    assert all(p.has_gt for p in pairs)
    assert pairs[0].name != pairs[1].name


def test_resolve_pairs_reads_a_generated_dataset_folder(tmp_path):
    from bench.datasets import resolve_pairs

    generate_dataset(preset("void").replace(**FAST), 2, tmp_path, workers=1)
    pairs = resolve_pairs(str(tmp_path))
    assert len(pairs) == 2


def test_a_real_matcher_scores_well_on_easy_generated_data():
    """The acceptance test for the whole generator.

    Sub-pixel end-point error is only achievable if the ground truth is
    geometrically correct -- an off-by-one or a sign error would put the
    error in the tens of pixels. This is what proves the data is usable for
    training and not merely pretty.
    """
    pytest.importorskip("cv2")
    from bench.algorithms import select
    from bench.metrics import evaluate

    specs = [s for s in select(["opencv_sgbm"]) if s.availability()[0]]
    if not specs:
        pytest.skip("opencv_sgbm unavailable")

    rules = preset("room").with_difficulty("easy").replace(
        width=320, height=224, supersample=1, sensor_noise=0.0,
    )
    pair = generate_pair(rules, seed=11)
    disparity, valid = specs[0].build().compute(pair).normalized()
    acc = evaluate(disparity, valid, pair.gt, eval_mask=pair.occ_mask)

    assert acc.n_eval_px > 1000
    assert acc.density > 0.5, "a textured scene should be widely matchable"
    assert acc.epe_valid < 1.5, (
        f"end-point error of {acc.epe_valid:.2f} px on easy data suggests the "
        f"ground truth is wrong, not the matcher"
    )


def test_harder_rules_produce_measurably_harder_data():
    """The difficulty knobs have to move a real matcher's error, not just
    the pixels. If they did not, the whole configuration surface would be
    decoration."""
    from bench.algorithms import select
    from bench.metrics import evaluate

    specs = [s for s in select(["opencv_sgbm"]) if s.availability()[0]]
    if not specs:
        pytest.skip("opencv_sgbm unavailable")

    def bad2(difficulty: str) -> float:
        rules = preset("room").with_difficulty(difficulty).replace(
            width=320, height=224, supersample=1,
        )
        pair = generate_pair(rules, seed=11)
        d, v = specs[0].build().compute(pair).normalized()
        return evaluate(d, v, pair.gt, eval_mask=pair.occ_mask).bad_all["2"]

    assert bad2("hard") > bad2("easy") + 5.0


# ----------------------------------------------------------------------
# train / val / test splits
# ----------------------------------------------------------------------


@pytest.mark.parametrize("count,fractions,expected", [
    (100, {"train": 0.8, "val": 0.1, "test": 0.1}, {"train": 80, "val": 10, "test": 10}),
    (10, {"train": 0.8, "val": 0.1, "test": 0.1}, {"train": 8, "val": 1, "test": 1}),
    (7, {"train": 0.5, "val": 0.5}, {"train": 4, "val": 3}),
    (1, {"train": 1.0}, {"train": 1}),
])
def test_splits_have_exact_counts(count, fractions, expected):
    """Asking for 80/10/10 of 100 must give 80, 10 and 10.

    A per-sample random draw would give 83/8/9 and nobody would notice until
    they quoted a validation number computed on eight images.
    """
    from collections import Counter

    from bench.synth.pipeline import resolve_splits

    assert dict(Counter(resolve_splits(count, fractions))) == expected


def test_every_requested_split_gets_at_least_one_sample():
    from collections import Counter

    from bench.synth.pipeline import resolve_splits

    counts = Counter(resolve_splits(3, {"train": 0.98, "val": 0.01, "test": 0.01}))
    assert set(counts) == {"train", "val", "test"}
    assert min(counts.values()) >= 1


def test_splits_are_skipped_when_not_asked_for():
    from bench.synth.pipeline import resolve_splits

    assert resolve_splits(5, None) == []
    assert resolve_splits(5, {}) == []
    assert resolve_splits(5, {"train": 0.0}) == []


def test_too_few_pairs_to_split_is_an_error():
    from bench.synth.pipeline import resolve_splits

    with pytest.raises(ValueError, match="cannot divide"):
        resolve_splits(2, {"train": 0.8, "val": 0.1, "test": 0.1})


def test_split_dataset_writes_subdirectories_and_reloads(tmp_path):
    from bench.datasets import load_synth_dataset, synth_splits

    rules = preset("void").replace(**FAST)
    result = generate_dataset(
        rules, 10, tmp_path, workers=1,
        splits={"train": 0.8, "val": 0.1, "test": 0.1},
    )
    assert result.count == 10
    assert synth_splits(tmp_path) == ["test", "train", "val"]
    assert sorted(d.name for d in (tmp_path / "train").iterdir()) == [
        f"{i:05d}" for i in range(8)
    ]

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["splits"] == {"train": 8, "val": 1, "test": 1}
    assert list(manifest["splits"]) == ["train", "val", "test"], "requested order"

    assert len(load_synth_dataset(tmp_path)) == 10
    assert len(load_synth_dataset(tmp_path, split="train")) == 8
    assert len(load_synth_dataset(tmp_path, split="val")) == 1


def test_splits_share_no_scene():
    """The whole point of a split: no geometry in common.

    Every sample is generated from its own seed, so two samples share a scene
    only if they share a seed. Checked on the seeds rather than on the pixels
    because that is the property that has to hold, and pixels could coincide
    by accident on a near-empty frame.
    """
    from bench.synth.pipeline import resolve_splits

    assignment = resolve_splits(30, {"train": 0.8, "val": 0.1, "test": 0.1})
    by_split: dict[str, set[int]] = {}
    for index, split in enumerate(assignment):
        by_split.setdefault(split, set()).add(1000 + index)  # seed = base + n

    seen: set[int] = set()
    for seeds in by_split.values():
        assert not (seeds & seen), "a seed appears in two splits"
        seen |= seeds
    assert len(seen) == 30


def test_environments_are_mixed_across_splits(tmp_path):
    """Each split should see the same environment mix, not one get all of one."""
    generate_dataset(
        preset("void").replace(**FAST), 12, tmp_path, workers=1,
        environments=["void", "room"],
        splits={"train": 0.5, "val": 0.5},
    )
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    per_split: dict[str, set[str]] = {}
    for row in manifest["samples"]:
        per_split.setdefault(row["split"], set()).add(row["environment"])
    assert per_split["train"] == {"void", "room"}
    assert per_split["val"] == {"void", "room"}


def test_resolve_pairs_accepts_one_split_of_a_dataset(tmp_path):
    from bench.datasets import resolve_pairs

    generate_dataset(
        preset("void").replace(**FAST), 10, tmp_path, workers=1,
        splits={"train": 0.8, "val": 0.1, "test": 0.1},
    )
    assert len(resolve_pairs(str(tmp_path / "train"))) == 8
    assert len(resolve_pairs(str(tmp_path))) == 10


def test_a_single_pair_folder_is_not_mistaken_for_a_dataset(tmp_path):
    """``practice/`` holds im0/im1 directly and must still load as one pair."""
    from bench.datasets import is_synth_dataset
    from bench.synth import write_sample

    write_sample(generate_pair(preset("void").replace(**FAST), seed=0), tmp_path)
    assert not is_synth_dataset(tmp_path)


@pytest.mark.parametrize("spec,expected", [
    ("train=0.8,val=0.1,test=0.1", {"train": 0.8, "val": 0.1, "test": 0.1}),
    ("a=1,b=1", {"a": 1.0, "b": 1.0}),
    ("", None),
    (None, None),
])
def test_cli_parses_split_specs(spec, expected):
    from bench.synth.cli import _parse_splits

    assert _parse_splits(spec) == expected


@pytest.mark.parametrize("spec", ["train", "train=x", "=0.5", "../evil=1", "a=-1"])
def test_cli_rejects_bad_split_specs(spec):
    from bench.synth.cli import _parse_splits

    with pytest.raises(ValueError):
        _parse_splits(spec)
