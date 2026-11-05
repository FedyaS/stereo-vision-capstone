"""Dataset loaders. Every loader yields :class:`bench.types.StereoPair`."""
from __future__ import annotations

from typing import Callable, Iterator

from ..types import StereoPair
from .middlebury import (
    DEFAULT_MIDDLEBURY_ROOT,
    list_middlebury_scenes,
    load_middlebury_scene,
    read_pfm,
)
from .synthetic import make_synthetic_pair
from .local import load_image_pair

__all__ = [
    "DEFAULT_MIDDLEBURY_ROOT",
    "list_middlebury_scenes",
    "load_middlebury_scene",
    "read_pfm",
    "make_synthetic_pair",
    "load_image_pair",
    "resolve_pairs",
]


def resolve_pairs(spec: str, limit: int | None = None) -> list[StereoPair]:
    """Turn a dataset spec string into concrete stereo pairs.

    Accepted specs::

        middlebury                  all MiddEval3 trainingQ scenes with GT
        middlebury:Teddy,Piano      named scenes only
        synthetic                   procedurally generated pair, exact GT
        synthetic:3                 three generated pairs with varied content
        path/to/dir                 a folder with im0.png / im1.png

    Pairs are returned in a stable order so repeated runs are comparable.
    """
    spec = spec.strip()
    name, _, arg = spec.partition(":")

    if name == "middlebury":
        scenes = [s for s in arg.split(",") if s] or None
        pairs = [load_middlebury_scene(s) for s in (scenes or list_middlebury_scenes())]
    elif name == "synthetic":
        n = int(arg) if arg else 1
        pairs = [make_synthetic_pair(seed=i) for i in range(n)]
    else:
        pairs = [load_image_pair(spec)]

    return pairs[:limit] if limit else pairs
