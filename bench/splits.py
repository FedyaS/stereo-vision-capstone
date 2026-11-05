"""Tune / held-out scene splits.

Tuning parameters on every scene and then reporting accuracy on those same
scenes measures how well the search ran, not how well the parameters work.
The sweep therefore only ever sees the ``tune`` scenes, and the headline
numbers are re-measured on ``heldout``, which no parameter ever touched.

**The split is by scene, not by file.** Middlebury ships several views of the
same physical scene: ``Motorcycle`` / ``MotorcycleE`` differ only in exposure,
``Piano`` / ``PianoL`` in lighting, ``Playtable`` / ``PlaytableP`` in whether
rectification was done perfectly. Putting ``Motorcycle`` in the tune set and
``MotorcycleE`` in the held-out set would leak -- the geometry is identical, so
a block size fitted to one is fitted to the other. Variants are kept together
in :data:`SCENE_GROUPS` and split as units.

Both sides deliberately keep at least one photometrically awkward pair, since
that is where cost functions separate most: the tune side has ``MotorcycleE``
and ``PianoL``, the held-out side has ``ArtL`` and ``PlaytableP``.
"""
from __future__ import annotations

from .datasets import list_middlebury_scenes

#: Scenes that are different captures of the same subject, grouped so a split
#: can never put two views of one scene on opposite sides.
SCENE_GROUPS: tuple[tuple[str, ...], ...] = (
    ("Adirondack",),
    ("ArtL",),
    ("Jadeplant",),
    ("Motorcycle", "MotorcycleE"),
    ("Piano", "PianoL"),
    ("Pipes",),
    ("Playroom",),
    ("Playtable", "PlaytableP"),
    ("Recycle",),
    ("Shelves",),
    ("Teddy",),
    ("Vintage",),
)

#: Groups the parameter sweep is allowed to see. 7 scenes, 5 groups.
TUNE_GROUPS: tuple[str, ...] = (
    "Motorcycle",  # + MotorcycleE: exposure mismatch
    "Piano",  # + PianoL: lighting mismatch
    "Teddy",
    "Recycle",
    "Shelves",
)

TUNE_SCENES: tuple[str, ...] = tuple(
    s for g in SCENE_GROUPS if g[0] in TUNE_GROUPS for s in g
)

#: Everything else. 8 scenes, 7 groups. Never seen during tuning.
HELDOUT_SCENES: tuple[str, ...] = tuple(
    s for g in SCENE_GROUPS if g[0] not in TUNE_GROUPS for s in g
)

SPLITS: dict[str, tuple[str, ...]] = {
    "tune": TUNE_SCENES,
    "heldout": HELDOUT_SCENES,
    "all": TUNE_SCENES + HELDOUT_SCENES,
}


def split_scenes(name: str, available: list[str] | None = None) -> list[str]:
    """Scene names in split ``name``, restricted to what is downloaded."""
    if name not in SPLITS:
        raise KeyError(f"unknown split {name!r}; known: {', '.join(SPLITS)}")
    have = set(available if available is not None else list_middlebury_scenes())
    return [s for s in SPLITS[name] if s in have]


def split_spec(name: str) -> str:
    """A dataset spec string for a split, e.g. ``middlebury:Teddy,Piano``."""
    scenes = split_scenes(name)
    if not scenes:
        raise FileNotFoundError(
            "No Middlebury scenes found -- run `python scripts/get_dataset.py`."
        )
    return "middlebury:" + ",".join(scenes)


def describe() -> str:
    return (
        f"tune    ({len(TUNE_SCENES)} scenes): {', '.join(TUNE_SCENES)}\n"
        f"heldout ({len(HELDOUT_SCENES)} scenes): {', '.join(HELDOUT_SCENES)}"
    )
