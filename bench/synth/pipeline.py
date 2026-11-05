"""From rules to a :class:`~bench.types.StereoPair`, and to a dataset on disk.

Two entry points matter:

:func:`generate_pair`
    One pair, in memory, with exact ground truth. This is what the preview
    uses and what the benchmark can consume directly.

:func:`generate_dataset`
    Many pairs written to disk in the Middlebury layout, in parallel, with a
    manifest recording exactly how they were made.

The on-disk layout is deliberately the one
:func:`bench.datasets.local.load_image_pair` already reads -- ``im0.png``,
``im1.png``, ``disp0GT.pfm``, ``mask0nocc.png``, ``calib.txt`` per sample.
A generated dataset is therefore not a special case anywhere downstream: it
loads, scores and reports through exactly the same code path as a downloaded
Middlebury scene, which is the only way to be sure the comparison between
them is apples to apples.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from ..types import StereoPair
from .environments import build_scene
from .render import StereoRender, auto_exposure, render_stereo, to_srgb8
from .rules import SceneRules

#: Bumped when the output format or the generator's semantics change, so a
#: manifest always says which version produced it.
FORMAT_VERSION = 1


def _apply_sensor(
    rgb_left: np.ndarray,
    rgb_right: np.ndarray,
    rules: SceneRules,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Turn linear radiance into two 8-bit images from two imperfect cameras.

    Order matters and follows the physical chain: metering and gain are
    applied to linear radiance (an exposure difference is a gain, not a
    brightness offset), then the tone curve, then optical blur, then read
    noise -- which is added after everything else because it is the sensor's,
    not the scene's.

    The *base* exposure is metered once, off the left view, and used for both.
    That is what a stereo rig with locked exposure does, and metering the two
    views separately would inject a brightness difference that the rules did
    not ask for -- precisely the error ``exposure_mismatch`` exists to model
    deliberately and in a known amount.

    The per-view terms, though, are drawn independently. That is the whole
    point: identical noise in both eyes would cancel out of every matching
    cost and make the data easier than a real rig, which is the mistake that
    makes naive synthetic training sets fail to transfer.
    """
    base = auto_exposure(rgb_left)
    mismatch = 1.0 + float(rules.exposure_mismatch) * float(rng.uniform(-1.0, 1.0))
    left = to_srgb8(rgb_left, exposure=base).astype(np.float32)
    right = to_srgb8(rgb_right, exposure=base * mismatch).astype(np.float32)

    if rules.defocus_px > 0:
        import cv2

        for img in (left, right):
            sigma = float(rules.defocus_px) * float(rng.uniform(0.85, 1.15))
            k = max(3, int(sigma * 6) | 1)
            cv2.GaussianBlur(img, (k, k), sigma, dst=img)

    if rules.vertical_jitter_px > 0:
        import cv2

        dy = float(rules.vertical_jitter_px) * float(rng.uniform(-1.0, 1.0))
        h, w = right.shape[:2]
        map_x = np.tile(np.arange(w, dtype=np.float32), (h, 1))
        map_y = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w)) + dy
        right = cv2.remap(
            right, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
        )

    if rules.sensor_noise > 0:
        s = float(rules.sensor_noise) * 255.0
        left = left + rng.normal(0.0, s, left.shape).astype(np.float32)
        right = right + rng.normal(0.0, s, right.shape).astype(np.float32)

    return (
        np.clip(left, 0, 255).astype(np.uint8),
        np.clip(right, 0, 255).astype(np.uint8),
    )


def generate_pair(rules: SceneRules, seed: int | None = None) -> StereoPair:
    """Render one stereo pair with exact, physically derived ground truth.

    The returned pair carries:

    ``gt``
        float32 disparity in left-image pixels, ``nan`` where unknown -- sky,
        pixels straddling a depth discontinuity (when ``mask_depth_edges`` is
        on), and anything beyond the declared search range.
    ``occ_mask``
        True where the surface is genuinely visible to *both* cameras,
        computed by comparing the two independently ray-traced depth maps.
    ``meta['stats']``
        Measured properties of this pair, so a dataset can be audited rather
        than assumed.
    """
    rules.validate()
    s = int(rules.seed if seed is None else seed)
    scene = build_scene(rules, s)
    scene.ambient = rules.ambient

    render: StereoRender = render_stereo(
        scene,
        rules.height,
        rules.width,
        baseline=rules.baseline,
        target_max_disparity=rules.target_max_disparity,
        supersample=rules.supersample,
        shadows=rules.shadows,
        bounces=rules.reflection_bounces,
    )

    rng = np.random.default_rng(s * 7919 + 13)
    left, right = _apply_sensor(render.left_rgb, render.right_rgb, rules, rng)

    gt = render.disparity.copy()
    if rules.mask_depth_edges:
        gt[render.edge_pixels] = np.nan
    # A disparity past the declared search range is outside the dataset's own
    # contract: no algorithm given this ndisp can find it, so scoring against
    # it would measure the generator's framing, not the matcher.
    out_of_range = np.isfinite(gt) & (gt >= float(rules.ndisp))
    gt[out_of_range] = np.nan

    occ = render.nonocc & np.isfinite(gt)

    pair = StereoPair(
        name=f"synth/{rules.environment}/{s}",
        left=left,
        right=right,
        ndisp=int(rules.ndisp),
        gt=gt,
        occ_mask=occ,
        focal_length_px=render.focal_px,
        baseline_mm=render.baseline * 1000.0,
        doffs=0.0,
        meta={
            "source": "synth",
            "seed": s,
            "environment": rules.environment,
            "rules": rules.to_dict(),
            "format_version": FORMAT_VERSION,
            "stats": pair_statistics(
                gt, occ, render, out_of_range_fraction=float(out_of_range.mean())
            ),
        },
    )
    return pair


def pair_statistics(
    gt: np.ndarray,
    occ: np.ndarray,
    render: StereoRender,
    *,
    out_of_range_fraction: float = 0.0,
) -> dict[str, Any]:
    """Measured properties of one generated pair.

    Reported rather than assumed, because the interesting failure modes of a
    generator are quiet ones: a rule set that produces 3 px of disparity, or
    ground truth covering 12% of the frame, still renders a perfectly
    attractive picture.
    """
    finite = np.isfinite(gt)
    vals = gt[finite]
    depth = render.depth_left[np.isfinite(render.depth_left)]
    return {
        "gt_coverage": round(float(finite.mean()), 4),
        "nonocc_fraction": round(float(occ.mean()), 4),
        "occluded_fraction": round(float((finite & ~occ).mean()), 4),
        "out_of_range_fraction": round(out_of_range_fraction, 4),
        "disparity_min": round(float(vals.min()), 3) if vals.size else None,
        "disparity_max": round(float(vals.max()), 3) if vals.size else None,
        "disparity_mean": round(float(vals.mean()), 3) if vals.size else None,
        "depth_min_m": round(float(depth.min()), 3) if depth.size else None,
        "depth_max_m": round(float(depth.max()), 3) if depth.size else None,
        "sky_fraction": round(float((~np.isfinite(render.depth_left)).mean()), 4),
        "edge_fraction": round(float(render.edge_pixels.mean()), 4),
        "baseline_m": round(float(render.baseline), 5),
        "focal_px": round(float(render.focal_px), 2),
    }


# ----------------------------------------------------------------------
# writing a dataset
# ----------------------------------------------------------------------


def write_sample(pair: StereoPair, folder: Path) -> dict[str, Any]:
    """Write one pair in the Middlebury layout and return its manifest row."""
    import cv2

    from ..datasets.middlebury import write_pfm

    folder.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(folder / "im0.png"), cv2.cvtColor(pair.left, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(folder / "im1.png"), cv2.cvtColor(pair.right, cv2.COLOR_RGB2BGR))

    # PFM with inf for unknown is the Middlebury convention; the loader turns
    # it back into nan. Writing nan would be read back as "a number".
    disp = np.where(np.isfinite(pair.gt), pair.gt, np.inf).astype(np.float32)
    write_pfm(folder / "disp0GT.pfm", disp)

    if pair.occ_mask is not None:
        cv2.imwrite(
            str(folder / "mask0nocc.png"),
            np.where(pair.occ_mask, 255, 0).astype(np.uint8),
        )

    h, w = pair.shape
    (folder / "calib.txt").write_text(
        f"width={w}\nheight={h}\nndisp={pair.ndisp}\n"
        f"baseline={pair.baseline_mm}\n"
        f"cam0=[{pair.focal_length_px} 0 {w / 2}; 0 {pair.focal_length_px} "
        f"{h / 2}; 0 0 1]\ndoffs=0\n",
        encoding="utf-8",
    )

    row = {"name": folder.name, "seed": pair.meta.get("seed")}
    row.update(pair.meta.get("stats", {}))
    return row


def _worker(args: tuple[dict[str, Any], int, str, str]) -> dict[str, Any]:
    """Render and write one sample. Top-level so it survives pickling.

    ``name`` may carry a split prefix (``"train/00007"``), which is simply
    part of the path; the worker needs no other knowledge of splitting.
    """
    rules_dict, seed, out_dir, name = args
    t0 = time.perf_counter()
    rules = SceneRules.from_dict(rules_dict)
    pair = generate_pair(rules, seed)
    row = write_sample(pair, Path(out_dir) / name)
    row["name"] = name
    row["split"] = name.rpartition("/")[0] or None
    row["environment"] = rules.environment
    row["render_s"] = round(time.perf_counter() - t0, 3)
    return row


@dataclass
class DatasetResult:
    """What :func:`generate_dataset` produced."""

    out_dir: Path
    rows: list[dict[str, Any]]
    elapsed_s: float
    cancelled: bool = False

    @property
    def count(self) -> int:
        return len(self.rows)


#: Default partition for a training set. Deliberately present as a default
#: rather than an option you have to discover: a generated set that is one
#: undivided pile invites evaluating on the pairs you trained on, and because
#: every pair here comes from the same generator, that mistake produces
#: excellent numbers and no warning at all.
DEFAULT_SPLITS: dict[str, float] = {"train": 0.8, "val": 0.1, "test": 0.1}


def resolve_splits(count: int, splits: dict[str, float] | None) -> list[str]:
    """Assign each sample index to a split, returning a list of length ``count``.

    Exact counts, not a per-sample coin flip: asking for 80/10/10 of 100 pairs
    should give 80, 10 and 10, not 83/8/9. Fractions are apportioned largest
    remainder first, and every non-empty split is guaranteed at least one
    sample so a small run cannot silently produce an empty validation set.

    Returns an empty list when ``splits`` is ``None``, meaning "one flat
    directory", which is what the single-environment smoke tests want.
    """
    if not splits:
        return []
    clean = {k: float(v) for k, v in splits.items() if float(v) > 0}
    if not clean:
        return []
    if len(clean) > count:
        raise ValueError(
            f"cannot divide {count} pair(s) into {len(clean)} splits; "
            f"generate at least {len(clean)}"
        )

    total = sum(clean.values())
    exact = {k: v / total * count for k, v in clean.items()}
    sizes = {k: max(1, int(v)) for k, v in exact.items()}

    # Hand out or claw back the rounding difference, largest remainder first,
    # never taking a split below one sample.
    diff = count - sum(sizes.values())
    order = sorted(clean, key=lambda k: exact[k] - int(exact[k]), reverse=True)
    i = 0
    while diff != 0 and order:
        k = order[i % len(order)]
        if diff > 0:
            sizes[k] += 1
            diff -= 1
        elif sizes[k] > 1:
            sizes[k] -= 1
            diff += 1
        i += 1
        if i > 4 * count + len(order):  # pragma: no cover - defensive
            break

    out: list[str] = []
    for name in clean:
        out.extend([name] * sizes[name])
    return out[:count]


def generate_dataset(
    rules: SceneRules,
    count: int,
    out_dir: str | Path,
    *,
    workers: int | None = None,
    start_index: int = 0,
    progress: Callable[[int, int, dict[str, Any]], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    environments: Iterable[str] | None = None,
    splits: dict[str, float] | None = None,
) -> DatasetResult:
    """Generate ``count`` pairs into ``out_dir``.

    Sample *n* uses seed ``rules.seed + start_index + n``, so the dataset is
    reproducible, and extending it later with a larger ``count`` reproduces
    the earlier samples rather than reshuffling them.

    ``environments`` cycles through several environments instead of using the
    one in ``rules`` -- the usual way to build a training set, since a model
    trained on a single scene type generalises poorly.

    ``splits`` partitions the output into subdirectories, e.g.
    ``{"train": 0.8, "val": 0.1, "test": 0.1}``, giving
    ``<out_dir>/train/00000`` and so on. Each split gets whole samples with
    disjoint seeds, so no scene appears in two of them. Pass ``{}`` or
    ``None`` for a single flat directory. When environments are being cycled,
    the cycle runs across the whole set before splitting, so each split sees
    the same mix rather than one split getting all the corridors.

    Rendering is CPU-bound pure NumPy, so it fans out across processes. One
    worker per core by default; ``workers=1`` stays in-process, which is what
    the tests and the single-sample preview want.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    rules.validate()
    if count < 1:
        raise ValueError("count must be at least 1")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    env_cycle = list(environments) if environments else None
    if env_cycle:
        for name in env_cycle:
            rules.replace(environment=name)  # validates each one up front

    assignment = resolve_splits(count, splits)

    jobs: list[tuple[dict[str, Any], int, str, str]] = []
    for n in range(count):
        seed = int(rules.seed) + start_index + n
        r = rules
        if env_cycle:
            r = rules.replace(environment=env_cycle[n % len(env_cycle)])
        # The split name is part of the sample's relative path, so a worker
        # needs nothing beyond it to know where its output belongs.
        name = f"{start_index + n:05d}"
        if assignment:
            name = f"{assignment[n]}/{name}"
        jobs.append((r.to_dict(), seed, str(out), name))

    if workers is None:
        workers = max(1, min(count, (os.cpu_count() or 2)))
    workers = max(1, int(workers))

    t0 = time.perf_counter()
    rows: list[dict[str, Any]] = []
    cancelled = False

    def run_serial(todo: list[tuple[dict[str, Any], int, str, str]]) -> bool:
        """Render ``todo`` in this process. Returns True if cancelled."""
        for job in todo:
            if should_stop is not None and should_stop():
                return True
            row = _worker(job)
            rows.append(row)
            if progress:
                progress(len(rows), count, row)
        return False

    if workers == 1:
        cancelled = run_serial(jobs)
    else:
        from concurrent.futures.process import BrokenProcessPool

        done_names: set[str] = set()
        try:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_worker, job): job for job in jobs}
                try:
                    for fut in as_completed(futures):
                        if should_stop is not None and should_stop():
                            cancelled = True
                            for f in futures:
                                f.cancel()
                            break
                        row = fut.result()
                        rows.append(row)
                        done_names.add(row["name"])
                        if progress:
                            progress(len(rows), count, row)
                finally:
                    if cancelled:
                        pool.shutdown(wait=False, cancel_futures=True)
        except (BrokenProcessPool, OSError) as exc:
            # On Windows a worker process re-imports the parent's __main__ to
            # reconstruct globals. A caller whose entry point is not guarded
            # by ``if __name__ == "__main__":`` therefore re-executes its own
            # script inside every worker, and the pool dies on the spot.
            # That is the caller's bug, but losing a long generation run over
            # it is a poor trade, so finish the work in this process instead
            # and say why it is slower.
            import warnings

            warnings.warn(
                f"parallel rendering failed ({type(exc).__name__}: {exc}); "
                f"falling back to a single process, which is slower. On "
                f"Windows this usually means the calling script needs its "
                f"entry point guarded by `if __name__ == \"__main__\":`.",
                RuntimeWarning,
                stacklevel=2,
            )
            remaining = [j for j in jobs if j[3] not in done_names]
            cancelled = run_serial(remaining)

    rows.sort(key=lambda r: r["name"])
    elapsed = time.perf_counter() - t0
    _write_manifest(out, rules, rows, elapsed, env_cycle, cancelled,
                    split_order=list(splits) if splits else None)
    return DatasetResult(out, rows, elapsed, cancelled)


def split_counts(
    rows: list[dict[str, Any]], order: Iterable[str] | None = None
) -> dict[str, int]:
    """How many samples ended up in each split (``{}`` for a flat dataset).

    ``order`` keeps the splits in the order they were requested, so a
    manifest and the CLI both read "train, val, test" rather than the
    alphabetical "test, train, val" that sorting by sample path would give.
    """
    counts: dict[str, int] = {}
    for r in rows:
        s = r.get("split")
        if s:
            counts[s] = counts.get(s, 0) + 1
    if order:
        wanted = [k for k in order if k in counts]
        return {k: counts[k] for k in wanted + sorted(set(counts) - set(wanted))}
    return counts


def _write_manifest(
    out: Path,
    rules: SceneRules,
    rows: list[dict[str, Any]],
    elapsed: float,
    environments: list[str] | None,
    cancelled: bool,
    split_order: list[str] | None = None,
) -> None:
    """Record the rules and the measured result next to the images."""
    from datetime import datetime, timezone

    def agg(key: str) -> dict[str, float] | None:
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        if not vals:
            return None
        return {
            "min": round(float(np.min(vals)), 3),
            "mean": round(float(np.mean(vals)), 3),
            "max": round(float(np.max(vals)), 3),
        }

    manifest = {
        "format_version": FORMAT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "count": len(rows),
        "cancelled": cancelled,
        "elapsed_s": round(elapsed, 2),
        "seconds_per_pair": round(elapsed / max(len(rows), 1), 3),
        "environments": environments or [rules.environment],
        "splits": split_counts(rows, split_order),
        "rules": rules.to_dict(),
        "summary": {
            k: agg(k)
            for k in (
                "gt_coverage", "nonocc_fraction", "occluded_fraction",
                "out_of_range_fraction", "disparity_min", "disparity_max",
                "disparity_mean", "sky_fraction", "render_s",
            )
        },
        "samples": rows,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    rules.save(out / "rules.json")
    (out / "README.md").write_text(_dataset_readme(manifest), encoding="utf-8")


def _dataset_readme(manifest: dict[str, Any]) -> str:
    """A README that says what this data is and how to reproduce it."""
    s = manifest["summary"]

    def rng(key: str) -> str:
        v = s.get(key)
        return "n/a" if not v else f"{v['min']:g} .. {v['max']:g} (mean {v['mean']:g})"

    rules = manifest["rules"]
    splits = manifest.get("splits") or {}
    if splits:
        listing = "\n".join(
            f"- `{name}/` — {n} pairs" for name, n in splits.items()
        )
        first = next(iter(splits))
        layout = f"""This set is split into disjoint subsets. No scene appears in more than one,
and every sample has its own seed, so there is no leakage between them:

{listing}

Each numbered folder inside a split is one sample in the Middlebury layout,
so it loads with the harness as-is:

    python -m bench run -d <this folder>/{first} -a opencv_sgbm
"""
    else:
        layout = """Each numbered folder is one sample in the Middlebury layout, so it loads with
the harness as-is:

    python -m bench run -d <this folder> -a opencv_sgbm
"""

    return f"""# Generated stereo dataset

{manifest['count']} rectified pairs with exact ground-truth disparity,
produced by `bench.synth` on {manifest['created_utc']}.

{layout}

| file | meaning |
| --- | --- |
| `im0.png` | left (reference) image |
| `im1.png` | right image |
| `disp0GT.pfm` | ground-truth disparity in left-image pixels; `inf` = unknown |
| `mask0nocc.png` | 255 where the surface is visible to **both** cameras |
| `calib.txt` | focal length, baseline and the search range `ndisp` |

## How the ground truth was made

Both views are ray traced independently from analytic surfaces, so depth is
the exact distance to the true surface rather than a z-buffer approximation,
and disparity is `focal_px * baseline / Z`. Occlusion is physical: the
`mask0nocc` region is computed by checking each left pixel's match against the
right camera's own depth map, so it marks the surfaces the right camera really
cannot see, not an estimate of them.

Ground truth is `inf` (unknown) at sky pixels, at pixels straddling a depth
discontinuity where no single true disparity exists, and beyond `ndisp`.

## What is in it

- environments: {', '.join(manifest['environments'])}
- ground-truth coverage: {rng('gt_coverage')}
- non-occluded fraction: {rng('nonocc_fraction')}
- disparity range: {rng('disparity_min')} to {rng('disparity_max')}
- render time per pair: {rng('render_s')} s

## Reproducing it

`rules.json` holds the exact rule set, and sample *n* used seed
`rules.seed + n`. The same rules and seeds regenerate these images bit for
bit:

    python -m bench synth generate --rules rules.json -n {manifest['count']} --out <dir>

Key settings: environment `{rules['environment']}`, {rules['width']}x{rules['height']},
ndisp {rules['ndisp']}, texture richness {rules['texture_richness']},
textureless fraction {rules['textureless_fraction']}, sensor noise
{rules['sensor_noise']}.
"""
