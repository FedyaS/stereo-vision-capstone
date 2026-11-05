"""Stereo algorithm adapters, grouped by family.

``opencv``   OpenCV's own matchers and ximgproc post-filters
``scratch``  NumPy/SciPy implementations, every stage exposed
``torch``    the same maths as tensor ops, GPU-capable
``learned``  pretrained networks via ONNX Runtime
"""
from __future__ import annotations

from .base import (
    REGISTRY,
    AlgorithmSpec,
    StereoAlgorithm,
    all_specs,
    load_all,
    register,
    select,
)

__all__ = [
    "REGISTRY",
    "AlgorithmSpec",
    "StereoAlgorithm",
    "all_specs",
    "load_all",
    "register",
    "select",
]
