"""The rule set: everything the user can configure about a generated dataset.

One dataclass holds the whole specification, it round-trips through JSON, and
it validates itself. That matters more than it sounds: a synthetic training
set is only useful if you can say precisely what is in it, and the honest way
to say that is to ship the rules next to the images. Every dataset this
package writes carries its ``rules`` verbatim in ``manifest.json``, so a
result six months later can be traced back to the exact distribution it came
from, and regenerated bit-for-bit from the same seed.

The fields fall into four groups:

**Framing** -- resolution, field of view, and the disparity range the pair
should occupy. ``ndisp`` is the search range an algorithm will be given, and
by default the rig is sized so the scene actually fills it (see
``disparity_fill``), because a dataset whose disparities only ever reach a
third of the search range quietly tests something other than what you meant.

**Content** -- which environment, how much clutter, how big the objects are.

**Appearance** -- texture richness, which texture kinds are allowed, lighting,
reflectivity. These are the knobs that make data *hard*: a matcher's error
rate is driven far more by textureless and repetitive surfaces than by
geometry.

**Sensor** -- noise, exposure mismatch, defocus, and vertical rectification
error. All of them are per-view, i.e. they break the photometric consistency
that real rigs break and naive synthetic data does not.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from .materials import TEXTURE_KINDS

#: The environments a scene can be sampled from. Each is a different *shape*
#: of depth distribution, which is the thing that actually varies between real
#: stereo benchmarks -- not the subject matter.
ENVIRONMENTS: dict[str, str] = {
    "tabletop": (
        "Close-range clutter on a table, walls behind. Large disparities, "
        "many small occlusions. The Middlebury regime."
    ),
    "room": (
        "An enclosed room with furniture-sized blocks and a textured floor. "
        "Mixed depths, strong planar structure, interreflection-free."
    ),
    "corridor": (
        "A long hallway receding from the camera. Wide depth range, strongly "
        "slanted side walls, which punishes fronto-parallel matchers."
    ),
    "warehouse": (
        "Tall repeated shelving units. Deliberately ambiguous: near-identical "
        "structures repeat along the epipolar line."
    ),
    "street": (
        "An outdoor ground plane with building blocks either side and open "
        "sky. Sky pixels have no ground truth, as in real outdoor data."
    ),
    "terrain": (
        "Open rocky ground under a sky, with scattered boulders and haze. "
        "Mostly smooth depth with sparse hard occlusions."
    ),
    "void": (
        "Primitives floating in front of a backdrop. No floor, so objects are "
        "isolated and every silhouette is an occlusion boundary."
    ),
}

#: Bundles of appearance settings, from "a matcher should ace this" to
#: "state of the art struggles". Applied by :meth:`SceneRules.with_difficulty`.
DIFFICULTY_PRESETS: dict[str, dict[str, Any]] = {
    "easy": {
        "texture_richness": 1.0,
        "textureless_fraction": 0.0,
        # Rich and aperiodic: every window has a unique signature, so there is
        # exactly one good match. The periodic kinds are excluded on purpose --
        # they are what makes a scene ambiguous, not what makes it detailed.
        "texture_kinds": ["gravel", "noise", "marble", "speckle"],
        "specular_strength": 0.05,
        "reflective_fraction": 0.0,
        "shadows": True,
        "sensor_noise": 0.002,
        "exposure_mismatch": 0.0,
        "defocus_px": 0.0,
        "vertical_jitter_px": 0.0,
        "fog_density": 0.0,
    },
    "medium": {
        "texture_richness": 0.75,
        "textureless_fraction": 0.15,
        "texture_kinds": ["gravel", "noise", "marble", "speckle", "grid"],
        "specular_strength": 0.18,
        "reflective_fraction": 0.0,
        "shadows": True,
        "sensor_noise": 0.008,
        "exposure_mismatch": 0.03,
        "defocus_px": 0.0,
        "vertical_jitter_px": 0.0,
        "fog_density": 0.0,
    },
    "hard": {
        "texture_richness": 0.45,
        "textureless_fraction": 0.35,
        "texture_kinds": list(TEXTURE_KINDS),
        "specular_strength": 0.35,
        "reflective_fraction": 0.12,
        "shadows": True,
        "sensor_noise": 0.018,
        "exposure_mismatch": 0.08,
        "defocus_px": 0.6,
        "vertical_jitter_px": 0.0,
        "fog_density": 0.015,
    },
}


class RuleError(ValueError):
    """A rule set that cannot produce sensible images."""


@dataclass
class SceneRules:
    """A complete, reproducible specification for a generated dataset."""

    # ---- framing -------------------------------------------------------
    environment: str = "room"
    width: int = 448
    height: int = 320
    ndisp: int = 64
    #: Fraction of ``ndisp`` the scene's largest disparity should reach. The
    #: baseline is solved for per scene to hit it, so every pair uses the
    #: search range instead of leaving most of it empty. Set ``baseline``
    #: explicitly to override and keep a physically fixed rig instead.
    disparity_fill: float = 0.9
    #: Metres between the two camera centres. ``None`` means "fit per scene".
    baseline: float | None = None
    #: Horizontal field of view in degrees; converted to a focal length.
    fov_degrees: float = 55.0

    # ---- content -------------------------------------------------------
    min_objects: int = 8
    max_objects: int = 18
    object_scale: float = 1.0
    #: 0 keeps the camera level and centred; 1 allows strong yaw/pitch and
    #: off-axis placement, so the same environment yields varied framings.
    camera_variation: float = 0.6

    # ---- appearance ----------------------------------------------------
    texture_richness: float = 0.8
    textureless_fraction: float = 0.12
    texture_kinds: list[str] = field(
        default_factory=lambda: ["gravel", "noise", "marble", "speckle", "grid"]
    )
    texture_scale: float = 1.0
    light_count: int = 2
    ambient: float = 0.22
    specular_strength: float = 0.18
    reflective_fraction: float = 0.0
    shadows: bool = True
    fog_density: float = 0.0
    palette_warmth: float = 0.5

    # ---- sensor --------------------------------------------------------
    sensor_noise: float = 0.008
    exposure_mismatch: float = 0.03
    defocus_px: float = 0.0
    #: Sub-pixel vertical misalignment of the right view, i.e. imperfect
    #: rectification. Horizontal ground truth stays valid, so this trains
    #: robustness to a real rig flaw -- but it is off by default, because it
    #: makes the data violate the epipolar assumption the GT is scored under.
    vertical_jitter_px: float = 0.0

    # ---- rendering -----------------------------------------------------
    #: 1 is fast and aliased; 2 is 4x the rays and looks like a real sensor.
    supersample: int = 2
    #: Mask out pixels straddling a depth discontinuity, where no single true
    #: disparity exists. Middlebury does the same; turning it off gives denser
    #: ground truth that is wrong at every silhouette.
    mask_depth_edges: bool = True
    reflection_bounces: int = 1
    seed: int = 0

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        self.validate()

    @property
    def focal_px(self) -> float:
        """Focal length in pixels implied by the horizontal field of view."""
        import math

        half = math.radians(self.fov_degrees) / 2.0
        return float(self.width / (2.0 * math.tan(half)))

    @property
    def target_max_disparity(self) -> float:
        return float(self.ndisp) * float(self.disparity_fill)

    def validate(self) -> None:
        """Raise :class:`RuleError` on a rule set that cannot work."""
        if self.environment not in ENVIRONMENTS:
            raise RuleError(
                f"unknown environment {self.environment!r}; "
                f"choose one of {sorted(ENVIRONMENTS)}"
            )
        if not (32 <= self.width <= 2048 and 32 <= self.height <= 2048):
            raise RuleError("width and height must be between 32 and 2048 pixels")
        if not (8 <= self.ndisp <= 512):
            raise RuleError("ndisp must be between 8 and 512")
        if self.ndisp >= self.width:
            # The left `ndisp` columns of the left image have no counterpart
            # in the right image at any disparity, so they can never be
            # matched. If the search range is as wide as the image, that band
            # is the whole image and the pair carries no usable supervision.
            raise RuleError(
                f"ndisp ({self.ndisp}) must be smaller than the image width "
                f"({self.width}): the leftmost ndisp columns have no match in "
                f"the right view, so nothing would be left to score"
            )
        if not (0.05 <= self.disparity_fill <= 1.5):
            raise RuleError("disparity_fill must be between 0.05 and 1.5")
        if self.baseline is not None and not (1e-4 <= self.baseline <= 10.0):
            raise RuleError("baseline must be between 0.0001 and 10 metres, or null")
        if not (10.0 <= self.fov_degrees <= 140.0):
            raise RuleError("fov_degrees must be between 10 and 140")
        if self.min_objects < 0 or self.max_objects < self.min_objects:
            raise RuleError("need 0 <= min_objects <= max_objects")
        if self.max_objects > 120:
            raise RuleError("max_objects above 120 makes rendering very slow")
        if not self.texture_kinds:
            raise RuleError("texture_kinds must list at least one kind")
        unknown = sorted(set(self.texture_kinds) - set(TEXTURE_KINDS))
        if unknown:
            raise RuleError(
                f"unknown texture kinds {unknown}; choose from {list(TEXTURE_KINDS)}"
            )
        if self.supersample not in (1, 2, 3):
            raise RuleError("supersample must be 1, 2 or 3")
        if not (0 <= self.reflection_bounces <= 2):
            raise RuleError("reflection_bounces must be 0, 1 or 2")
        for name, lo, hi in (
            ("texture_richness", 0.0, 1.0),
            ("textureless_fraction", 0.0, 1.0),
            ("reflective_fraction", 0.0, 1.0),
            ("camera_variation", 0.0, 1.0),
            ("ambient", 0.0, 1.0),
            ("specular_strength", 0.0, 1.0),
            ("sensor_noise", 0.0, 0.2),
            ("exposure_mismatch", 0.0, 0.5),
            ("defocus_px", 0.0, 5.0),
            ("vertical_jitter_px", 0.0, 3.0),
            ("fog_density", 0.0, 0.5),
            ("palette_warmth", 0.0, 1.0),
            ("texture_scale", 0.05, 20.0),
            ("object_scale", 0.1, 5.0),
        ):
            v = float(getattr(self, name))
            if not (lo <= v <= hi):
                raise RuleError(f"{name} must be between {lo} and {hi} (got {v})")
        if not (0 <= self.light_count <= 6):
            raise RuleError("light_count must be between 0 and 6")

    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SceneRules":
        """Build from a dict, ignoring unknown keys and reporting bad types.

        Unknown keys are dropped rather than raising: rule files written by an
        older version of this package should still load, just without whatever
        was added since.
        """
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in (data or {}).items() if k in known}
        try:
            return cls(**kwargs)
        except TypeError as exc:
            raise RuleError(f"bad rule file: {exc}") from exc

    @classmethod
    def from_json(cls, text: str) -> "SceneRules":
        return cls.from_dict(json.loads(text))

    @classmethod
    def load(cls, path: str | Path) -> "SceneRules":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    def replace(self, **changes: Any) -> "SceneRules":
        """A copy with some fields changed, re-validated."""
        return SceneRules.from_dict({**self.to_dict(), **changes})

    def with_difficulty(self, level: str) -> "SceneRules":
        """A copy with the appearance knobs set to a named difficulty bundle."""
        if level not in DIFFICULTY_PRESETS:
            raise RuleError(
                f"unknown difficulty {level!r}; "
                f"choose one of {sorted(DIFFICULTY_PRESETS)}"
            )
        return self.replace(**DIFFICULTY_PRESETS[level])

    def describe(self) -> str:
        """A short human-readable summary, for logs and dataset READMEs."""
        base = "auto" if self.baseline is None else f"{self.baseline:.3f} m"
        return (
            f"{self.environment} {self.width}x{self.height} "
            f"ndisp={self.ndisp} baseline={base} "
            f"fov={self.fov_degrees:g} deg "
            f"objects={self.min_objects}-{self.max_objects} "
            f"texture={self.texture_richness:.2f} "
            f"flat={self.textureless_fraction:.2f} "
            f"noise={self.sensor_noise:.3f} ss={self.supersample}"
        )


def preset(name: str, **overrides: Any) -> SceneRules:
    """A ready-made rule set for one environment, at medium difficulty.

    Resolution, disparity range and object counts are tuned per environment,
    because sensible values genuinely differ: a corridor needs a wide image
    and a long search range, a tabletop needs neither.
    """
    if name not in ENVIRONMENTS:
        raise RuleError(
            f"unknown environment {name!r}; choose one of {sorted(ENVIRONMENTS)}"
        )

    tuned: dict[str, Any] = {
        "tabletop": dict(
            ndisp=96, fov_degrees=48, min_objects=10, max_objects=22,
            object_scale=0.8, ambient=0.26,
        ),
        "room": dict(
            ndisp=64, fov_degrees=58, min_objects=8, max_objects=16,
        ),
        "corridor": dict(
            ndisp=96, fov_degrees=65, min_objects=6, max_objects=14,
            width=512, height=320,
        ),
        "warehouse": dict(
            ndisp=96, fov_degrees=60, min_objects=10, max_objects=20,
            texture_kinds=["grid", "stripes", "checker", "noise", "flat"],
            textureless_fraction=0.3,
        ),
        "street": dict(
            ndisp=128, fov_degrees=70, min_objects=8, max_objects=18,
            width=512, height=288, ambient=0.3,
        ),
        "terrain": dict(
            ndisp=64, fov_degrees=72, min_objects=12, max_objects=28,
            object_scale=1.3, ambient=0.32, fog_density=0.01,
        ),
        "void": dict(
            ndisp=64, fov_degrees=52, min_objects=6, max_objects=14,
        ),
    }[name]

    return SceneRules(environment=name, **tuned).with_difficulty(
        "medium"
    ).replace(**{**tuned, **overrides})


def rule_schema() -> list[dict[str, Any]]:
    """Machine-readable description of every rule, for building a UI.

    Kept beside the dataclass rather than duplicated in the frontend, so the
    editor cannot drift out of step with what the generator accepts.
    """
    def row(
        name: str, kind: str, group: str, label: str, help_: str, **extra: Any
    ) -> dict[str, Any]:
        return {
            "name": name, "type": kind, "group": group, "label": label,
            "help": help_, "default": getattr(SceneRules(), name), **extra,
        }

    return [
        row("environment", "choice", "Framing", "Environment",
            "The kind of scene to build.",
            options=[{"value": k, "help": v} for k, v in ENVIRONMENTS.items()]),
        row("width", "int", "Framing", "Width", "Image width in pixels.",
            min=64, max=1280, step=16),
        row("height", "int", "Framing", "Height", "Image height in pixels.",
            min=64, max=1024, step=16),
        row("ndisp", "int", "Framing", "Disparity range",
            "Search range an algorithm is given, in pixels.",
            min=16, max=256, step=8),
        row("disparity_fill", "float", "Framing", "Disparity fill",
            "Fraction of the search range the nearest surface should reach. "
            "The baseline is solved per scene to hit this.",
            min=0.2, max=1.2, step=0.05),
        row("fov_degrees", "float", "Framing", "Field of view",
            "Horizontal field of view in degrees.", min=20, max=120, step=1),
        row("camera_variation", "float", "Framing", "Camera variation",
            "How much the viewpoint moves and rotates between samples.",
            min=0, max=1, step=0.05),

        row("min_objects", "int", "Content", "Min objects",
            "Fewest clutter objects to place.", min=0, max=80, step=1),
        row("max_objects", "int", "Content", "Max objects",
            "Most clutter objects to place.", min=0, max=80, step=1),
        row("object_scale", "float", "Content", "Object scale",
            "Multiplies the size of every placed object.",
            min=0.2, max=3.0, step=0.1),

        row("texture_richness", "float", "Appearance", "Texture richness",
            "Contrast of surface texture. Low values starve a matcher of "
            "signal and are the single biggest driver of error.",
            min=0, max=1, step=0.05),
        row("textureless_fraction", "float", "Appearance", "Textureless surfaces",
            "Fraction of surfaces made completely flat.",
            min=0, max=1, step=0.05),
        row("texture_kinds", "multichoice", "Appearance", "Texture kinds",
            "Which patterns surfaces may use. Periodic kinds (checker, "
            "stripes, grid) create matching ambiguity on purpose.",
            options=[{"value": k} for k in TEXTURE_KINDS]),
        row("texture_scale", "float", "Appearance", "Texture scale",
            "Multiplies texture frequency; high values make finer detail.",
            min=0.1, max=5.0, step=0.1),
        row("light_count", "int", "Appearance", "Lights",
            "Number of light sources.", min=0, max=6, step=1),
        row("ambient", "float", "Appearance", "Ambient light",
            "Base illumination, which sets how much detail survives in "
            "shadow.", min=0, max=0.8, step=0.02),
        row("specular_strength", "float", "Appearance", "Specular strength",
            "Highlight intensity. Highlights sit in different places in the "
            "two views, which is a real and useful photometric challenge.",
            min=0, max=1, step=0.05),
        row("reflective_fraction", "float", "Appearance", "Reflective surfaces",
            "Fraction of objects given a mirror component.",
            min=0, max=1, step=0.05),
        row("shadows", "bool", "Appearance", "Cast shadows",
            "Trace shadow rays. Off is roughly twice as fast."),
        row("fog_density", "float", "Appearance", "Haze",
            "Atmospheric fade with distance; washes out far texture.",
            min=0, max=0.2, step=0.005),
        row("palette_warmth", "float", "Appearance", "Palette warmth",
            "0 is a cool/blue palette, 1 is warm/red.", min=0, max=1, step=0.05),

        row("sensor_noise", "float", "Sensor", "Sensor noise",
            "Per-view Gaussian noise, as a fraction of full scale. Applied "
            "independently to each eye, as on a real rig.",
            min=0, max=0.08, step=0.002),
        row("exposure_mismatch", "float", "Sensor", "Exposure mismatch",
            "Gain difference between the two cameras.",
            min=0, max=0.3, step=0.01),
        row("defocus_px", "float", "Sensor", "Defocus blur",
            "Gaussian blur radius in pixels.", min=0, max=3, step=0.1),
        row("vertical_jitter_px", "float", "Sensor", "Rectification error",
            "Vertical misalignment of the right view in pixels. Realistic, "
            "but it breaks the epipolar assumption the ground truth is "
            "scored under, so leave at 0 unless that is the point.",
            min=0, max=2, step=0.05),

        row("supersample", "int", "Rendering", "Supersampling",
            "Rays per pixel axis. 2 gives antialiased edges at 4x the cost.",
            min=1, max=3, step=1),
        row("mask_depth_edges", "bool", "Rendering", "Mask depth edges",
            "Drop ground truth at pixels straddling two surfaces, where no "
            "single true disparity exists."),
        row("reflection_bounces", "int", "Rendering", "Reflection bounces",
            "Extra rays for mirror surfaces.", min=0, max=2, step=1),
        row("seed", "int", "Rendering", "Seed",
            "Base seed. Sample n uses seed + n, so a dataset is "
            "reproducible.", min=0, max=999999, step=1),
    ]
