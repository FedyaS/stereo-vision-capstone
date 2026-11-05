"""Training-image synthesizer: 3-D scenes with exact ground-truth disparity.

Generates rectified stereo pairs by ray tracing analytic 3-D scenes on the
CPU, in pure NumPy. No GPU, no display, no download, no new dependency -- and
because the geometry is solved rather than rasterised, the disparity it ships
is exact to float precision rather than quantised by a z-buffer.

Typical use::

    from bench.synth import SceneRules, generate_pair, generate_dataset

    rules = SceneRules(environment="room", width=448, height=320, ndisp=64)
    pair = generate_pair(rules, seed=0)          # one pair, in memory

    generate_dataset(rules, 500, "data/synth/train")   # a dataset on disk

or from the command line::

    python -m bench synth environments
    python -m bench synth preview --environment corridor --samples 4
    python -m bench synth generate --rules rules.json -n 500 --out data/synth/train

See :mod:`bench.synth.rules` for every knob, :mod:`bench.synth.environments`
for the scene types, and :mod:`bench.synth.render` for why this is a ray
tracer rather than a wrapper around Blender or Open3D.
"""
from __future__ import annotations

from .environments import build_scene
from .materials import TEXTURE_KINDS, Light, Material
from .pipeline import (
    DatasetResult,
    generate_dataset,
    generate_pair,
    pair_statistics,
    write_sample,
)
from .preview import contact_sheet, preview_pairs
from .render import Camera, Scene, StereoRender, render_stereo, to_srgb8
from .rules import (
    DIFFICULTY_PRESETS,
    ENVIRONMENTS,
    RuleError,
    SceneRules,
    preset,
    rule_schema,
)
from .shapes import Box, Cylinder, Plane, Sphere

__all__ = [
    "SceneRules",
    "RuleError",
    "ENVIRONMENTS",
    "DIFFICULTY_PRESETS",
    "preset",
    "rule_schema",
    "generate_pair",
    "generate_dataset",
    "DatasetResult",
    "pair_statistics",
    "write_sample",
    "preview_pairs",
    "contact_sheet",
    "build_scene",
    "Scene",
    "Camera",
    "StereoRender",
    "render_stereo",
    "to_srgb8",
    "Material",
    "Light",
    "TEXTURE_KINDS",
    "Plane",
    "Sphere",
    "Box",
    "Cylinder",
]
