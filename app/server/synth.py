"""Backend for the synthesizer tab: previews and generated datasets.

Previews are held in memory rather than written to disk. A preview is a
throwaway -- you tweak a rule, look, tweak again -- and persisting every
attempt would litter the repository with hundreds of near-identical PNGs for
no benefit. The cache is small and evicted oldest-first.

Generated *datasets* are the opposite: they go to ``data/synth/<name>/`` in
the Middlebury layout and are registered with the scene catalogue, so a
dataset made in the browser can immediately be benchmarked in the same
browser without an export step.
"""
from __future__ import annotations

import re
import shutil
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from bench.datasets import load_synth_dataset
from bench.synth import (
    DIFFICULTY_PRESETS,
    ENVIRONMENTS,
    SceneRules,
    TEXTURE_KINDS,
    preset,
    rule_schema,
)
from bench.synth.preview import pair_views, preview_pairs, preview_report
from bench.types import StereoPair

REPO_ROOT = Path(__file__).resolve().parents[2]
SYNTH_ROOT = REPO_ROOT / "data" / "synth"

_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.-]{0,63}$")


def safe_name(name: str) -> str:
    name = (name or "").strip()
    if not _SAFE_NAME.match(name) or ".." in name:
        raise ValueError(
            "dataset name must be 1-64 characters of letters, digits, "
            "space, dot, dash or underscore"
        )
    return name


# ----------------------------------------------------------------------
# preview cache
# ----------------------------------------------------------------------


@dataclass
class Preview:
    id: str
    rules: dict[str, Any]
    pairs: list[StereoPair] = field(default_factory=list)
    report: dict[str, Any] = field(default_factory=dict)

    def payload(self) -> dict[str, Any]:
        return {
            "preview_id": self.id,
            "rules": self.rules,
            "summary": self.report.get("summary", {}),
            "warnings": self.report.get("warnings", []),
            "samples": [
                {
                    "index": i,
                    "seed": p.meta.get("seed"),
                    "name": p.name,
                    "width": p.shape[1],
                    "height": p.shape[0],
                    "ndisp": p.ndisp,
                    "stats": p.meta.get("stats", {}),
                }
                for i, p in enumerate(self.pairs)
            ],
        }


_previews: dict[str, Preview] = {}
_preview_lock = threading.Lock()
_PREVIEW_LIMIT = 6


def schema() -> dict[str, Any]:
    """Everything a UI needs to build the rules editor."""
    return {
        "fields": rule_schema(),
        "environments": [
            {"value": k, "label": k, "help": v} for k, v in ENVIRONMENTS.items()
        ],
        "difficulties": [
            {"value": k, "settings": v} for k, v in DIFFICULTY_PRESETS.items()
        ],
        "texture_kinds": list(TEXTURE_KINDS),
        "defaults": SceneRules().to_dict(),
        "presets": {name: preset(name).to_dict() for name in ENVIRONMENTS},
    }


def make_preview(rules_dict: dict[str, Any], samples: int = 3) -> Preview:
    """Render a few pairs and cache them for the image endpoints."""
    rules = SceneRules.from_dict(rules_dict)
    pairs = preview_pairs(rules, max(1, min(int(samples), 6)))
    pv = Preview(
        id=uuid.uuid4().hex[:12],
        rules=rules.to_dict(),
        pairs=pairs,
        report=preview_report(pairs),
    )
    with _preview_lock:
        while len(_previews) >= _PREVIEW_LIMIT:
            _previews.pop(next(iter(_previews)))
        _previews[pv.id] = pv
    return pv


def get_preview(preview_id: str) -> Preview:
    with _preview_lock:
        if preview_id not in _previews:
            raise KeyError(
                f"no preview {preview_id!r}; it may have been evicted -- "
                f"render the preview again"
            )
        return _previews[preview_id]


def preview_image(preview_id: str, index: int, view: str) -> np.ndarray:
    pv = get_preview(preview_id)
    if not (0 <= index < len(pv.pairs)):
        raise KeyError(f"sample {index} is out of range")
    views = pair_views(pv.pairs[index])
    if view not in views:
        raise KeyError(f"unknown view {view!r}; expected one of {sorted(views)}")
    return views[view]


# ----------------------------------------------------------------------
# generated datasets on disk
# ----------------------------------------------------------------------


def dataset_dir(name: str) -> Path:
    return SYNTH_ROOT / safe_name(name)


def list_datasets() -> list[dict[str, Any]]:
    """Every generated dataset under ``data/synth``, newest first."""
    import json

    if not SYNTH_ROOT.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for d in sorted(SYNTH_ROOT.iterdir()):
        manifest_path = d / "manifest.json"
        if not (d.is_dir() and manifest_path.is_file()):
            continue
        try:
            m = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            out.append({"name": d.name, "error": str(exc), "count": 0})
            continue
        out.append({
            "name": d.name,
            "path": str(d.relative_to(REPO_ROOT)).replace("\\", "/"),
            "count": m.get("count", 0),
            "created_utc": m.get("created_utc"),
            "elapsed_s": m.get("elapsed_s"),
            "seconds_per_pair": m.get("seconds_per_pair"),
            "environments": m.get("environments", []),
            "splits": m.get("splits", {}),
            "rules": m.get("rules", {}),
            "summary": m.get("summary", {}),
            "cancelled": m.get("cancelled", False),
        })
    out.sort(key=lambda r: r.get("created_utc") or "", reverse=True)
    return out


def sample_ids(name: str, limit: int | None = None) -> list[str]:
    """Scene ids for a dataset's samples, for the catalogue and pickers."""
    d = dataset_dir(name)
    if not d.is_dir():
        return []
    samples = sorted(
        s.name for s in d.iterdir() if s.is_dir() and (s / "im0.png").exists()
    )
    if limit:
        samples = samples[:limit]
    return [f"synth/{name}/{s}" for s in samples]


def load_sample(name: str, sample: str) -> StereoPair:
    """Load one sample of a generated dataset as a :class:`StereoPair`."""
    from bench.datasets.local import load_image_pair

    folder = dataset_dir(name) / Path(sample).name
    if not folder.is_dir():
        raise FileNotFoundError(f"no sample {sample!r} in dataset {name!r}")
    return load_image_pair(folder, name=f"synth/{name}/{folder.name}")


def load_all(name: str, limit: int | None = None) -> list[StereoPair]:
    d = dataset_dir(name)
    if not d.is_dir():
        raise FileNotFoundError(f"no generated dataset named {name!r}")
    return load_synth_dataset(d, limit)


def delete_dataset(name: str) -> None:
    d = dataset_dir(name)
    if not d.is_dir():
        raise FileNotFoundError(f"no generated dataset named {name!r}")
    shutil.rmtree(d)
