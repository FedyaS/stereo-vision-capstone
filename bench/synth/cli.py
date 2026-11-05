"""``python -m bench synth`` -- inspect rules, preview them, generate datasets.

Three steps, in the order you actually work in them:

``environments``
    What can be generated, and what each environment is for.

``preview``
    Render a handful of pairs from a rule set and write a contact sheet plus
    the measured statistics. This is the step that catches a rule set which
    renders beautifully and supervises nothing.

``generate``
    Produce the dataset, in parallel, with a manifest.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .rules import DIFFICULTY_PRESETS, ENVIRONMENTS, RuleError, SceneRules, preset

#: Rule fields that can be set straight from the command line. Everything
#: else lives in a rules file, which is the right place for a long
#: specification anyway.
_OVERRIDES: dict[str, type] = {
    "width": int,
    "height": int,
    "ndisp": int,
    "disparity_fill": float,
    "fov_degrees": float,
    "min_objects": int,
    "max_objects": int,
    "object_scale": float,
    "camera_variation": float,
    "texture_richness": float,
    "textureless_fraction": float,
    "texture_scale": float,
    "light_count": int,
    "ambient": float,
    "specular_strength": float,
    "reflective_fraction": float,
    "fog_density": float,
    "palette_warmth": float,
    "sensor_noise": float,
    "exposure_mismatch": float,
    "defocus_px": float,
    "vertical_jitter_px": float,
    "supersample": int,
    "reflection_bounces": int,
    "seed": int,
}


def _rules_from_args(args: argparse.Namespace) -> SceneRules:
    """Build the rule set: file, then environment, then difficulty, then flags.

    Later sources win, so ``--rules base.json --texture-richness 0.2`` means
    what it looks like it means.
    """
    if getattr(args, "rules", None):
        rules = SceneRules.load(args.rules)
        if getattr(args, "environment", None):
            rules = rules.replace(environment=args.environment)
    else:
        rules = preset(getattr(args, "environment", None) or "room")

    if getattr(args, "difficulty", None):
        rules = rules.with_difficulty(args.difficulty)

    changes = {}
    for name in _OVERRIDES:
        value = getattr(args, name, None)
        if value is not None:
            changes[name] = value
    if getattr(args, "texture_kinds", None):
        changes["texture_kinds"] = list(args.texture_kinds)
    if getattr(args, "no_shadows", False):
        changes["shadows"] = False
    if getattr(args, "keep_depth_edges", False):
        changes["mask_depth_edges"] = False
    return rules.replace(**changes) if changes else rules


def _add_rule_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--rules", default=None, metavar="FILE",
                   help="a rules JSON file to start from")
    p.add_argument("-e", "--environment", default=None, choices=sorted(ENVIRONMENTS),
                   help="which kind of scene to build")
    p.add_argument("--difficulty", default=None, choices=sorted(DIFFICULTY_PRESETS),
                   help="apply a bundle of appearance settings")
    p.add_argument("--texture-kinds", nargs="+", default=None, metavar="KIND",
                   help="restrict which texture patterns may be used")
    p.add_argument("--no-shadows", action="store_true",
                   help="skip shadow rays (about twice as fast)")
    p.add_argument("--keep-depth-edges", action="store_true",
                   help="keep ground truth at depth discontinuities, where no "
                        "single true disparity exists")
    g = p.add_argument_group("rule overrides")
    for name, kind in _OVERRIDES.items():
        g.add_argument(f"--{name.replace('_', '-')}", dest=name, type=kind,
                       default=None, metavar=kind.__name__.upper()[0])


# ----------------------------------------------------------------------


def _cmd_environments(args: argparse.Namespace) -> int:
    print(f"{len(ENVIRONMENTS)} environments\n")
    for name, blurb in ENVIRONMENTS.items():
        print(f"  {name}")
        for line in _wrap(blurb, 72):
            print(f"      {line}")
        print()
    print("difficulty presets")
    for name, settings in DIFFICULTY_PRESETS.items():
        print(
            f"  {name:7} texture={settings['texture_richness']:.2f} "
            f"flat={settings['textureless_fraction']:.2f} "
            f"noise={settings['sensor_noise']:.3f} "
            f"specular={settings['specular_strength']:.2f}"
        )
    print("\nExamples")
    print("  python -m bench synth preview -e corridor --difficulty hard")
    print("  python -m bench synth generate -e room -n 200 --out data/synth/room")
    print("  python -m bench synth generate --rules my.json -n 1000 "
          "--environments room corridor tabletop --out data/synth/mixed")
    return 0


def _wrap(text: str, width: int) -> list[str]:
    import textwrap

    return textwrap.wrap(" ".join(text.split()), width)


def _cmd_rules(args: argparse.Namespace) -> int:
    """Print a rule set as JSON, so it can be saved and edited."""
    rules = _rules_from_args(args)
    if args.out:
        rules.save(args.out)
        print(f"wrote {args.out}")
    else:
        print(rules.to_json())
    return 0


def _cmd_preview(args: argparse.Namespace) -> int:
    import cv2

    from .preview import contact_sheet, preview_pairs, preview_report

    rules = _rules_from_args(args)
    print(f"rules: {rules.describe()}")
    print(f"rendering {args.samples} sample(s)...")

    def tick(done: int, total: int) -> None:
        print(f"  [{done}/{total}]", end="\r", flush=True)

    pairs = preview_pairs(rules, args.samples, progress=tick)
    print(" " * 24, end="\r")

    report = preview_report(pairs)
    s = report["summary"]
    print("\nmeasured over the samples")
    print(f"  ground-truth coverage   {s['gt_coverage']:.1%}")
    print(f"  visible in both views   {s['nonocc_fraction']:.1%}")
    print(f"  occluded                {s['occluded_fraction']:.1%}")
    print(f"  sky (no depth)          {s['sky_fraction']:.1%}")
    print(f"  disparity               {s['disparity_min']:.1f} .. "
          f"{s['disparity_max']:.1f} px  (search range {rules.ndisp})")
    print(f"  fitted baseline         {s['baseline_m']:.4f} m")

    if report["warnings"]:
        print("\nwarnings")
        for w in report["warnings"]:
            for i, line in enumerate(_wrap(w, 70)):
                print(f"  {'-' if i == 0 else ' '} {line}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet = contact_sheet(pairs, scale=args.scale)
    cv2.imwrite(str(out), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    print(f"\ncontact sheet: {out}  ({sheet.shape[1]}x{sheet.shape[0]})")

    if args.save_rules:
        rules.save(args.save_rules)
        print(f"rules:         {args.save_rules}")
    if args.open:
        import webbrowser

        webbrowser.open(out.resolve().as_uri())
    return 0


def _parse_splits(spec: str | None) -> dict[str, float] | None:
    """``"train=0.8,val=0.2"`` -> ``{"train": 0.8, "val": 0.2}``.

    An empty string means "no splitting", which is how the flat layout is
    requested from a shell where you cannot pass ``None``.
    """
    if spec is None:
        return None
    spec = spec.strip()
    if not spec:
        return None
    out: dict[str, float] = {}
    for part in spec.split(","):
        name, sep, value = part.partition("=")
        name = name.strip()
        if not sep or not name:
            raise ValueError(
                f"bad split {part.strip()!r}; expected name=fraction, "
                f"e.g. train=0.8,val=0.1,test=0.1"
            )
        if "/" in name or "\\" in name or name in (".", ".."):
            raise ValueError(f"bad split name {name!r}")
        try:
            out[name] = float(value)
        except ValueError:
            raise ValueError(f"split {name!r} needs a number, got {value!r}") from None
        if out[name] < 0:
            raise ValueError(f"split {name!r} cannot be negative")
    return out


def _cmd_generate(args: argparse.Namespace) -> int:
    import time

    from .pipeline import generate_dataset

    rules = _rules_from_args(args)
    envs = args.environments or None
    try:
        splits = _parse_splits(args.splits)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"rules: {rules.describe()}")
    if envs:
        print(f"cycling environments: {', '.join(envs)}")
    print(
        "splits: "
        + (", ".join(f"{k}={v:g}" for k, v in splits.items()) if splits
           else "none (one flat directory)")
    )
    print(f"generating {args.count} pair(s) into {args.out}")

    t0 = time.perf_counter()
    last = [0.0]

    def tick(done: int, total: int, row: dict) -> None:
        now = time.perf_counter()
        if now - last[0] < 0.2 and done != total:
            return
        last[0] = now
        rate = done / max(now - t0, 1e-6)
        eta = (total - done) / max(rate, 1e-6)
        print(
            f"  [{done}/{total}] {rate:.2f} pair/s  eta {eta:5.0f}s  ",
            end="\r", flush=True,
        )

    try:
        result = generate_dataset(
            rules, args.count, args.out,
            workers=args.workers, start_index=args.start_index,
            progress=tick, environments=envs, splits=splits,
        )
    except RuleError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(" " * 60, end="\r")

    print(f"\n{result.count} pair(s) in {result.elapsed_s:.1f}s "
          f"({result.elapsed_s / max(result.count, 1):.2f}s each)")
    print(f"  {result.out_dir}")
    print("  manifest.json, rules.json, README.md")

    manifest = json.loads((result.out_dir / "manifest.json").read_text())
    if manifest.get("splits"):
        print("  splits: " + ", ".join(
            f"{k} {v}" for k, v in manifest["splits"].items()))
    summary = manifest["summary"]
    for key, label in (
        ("gt_coverage", "ground-truth coverage"),
        ("nonocc_fraction", "visible in both views"),
        ("disparity_max", "max disparity"),
        ("out_of_range_fraction", "dropped, past ndisp"),
    ):
        v = summary.get(key)
        if v:
            print(f"  {label:24} {v['min']:g} .. {v['max']:g} (mean {v['mean']:g})")

    # Point at a split when there is one: the dataset root of a split set has
    # no samples directly under it, so the obvious command would not work.
    first = next(iter(manifest.get("splits") or {}), None)
    target = f"{result.out_dir}/{first}" if first else str(result.out_dir)
    print(f"\nUse it:  python -m bench run -d {target} -a opencv_sgbm")
    return 0


def add_subparser(sub: argparse._SubParsersAction) -> None:
    """Attach the ``synth`` command tree to the main parser."""
    p = sub.add_parser(
        "synth",
        help="generate training images with exact ground-truth disparity",
        description="Ray trace 3-D scenes into rectified stereo pairs whose "
        "disparity is exact by construction. Pure NumPy -- no GPU, no "
        "display, no download.",
    )
    inner = p.add_subparsers(dest="synth_command")

    pe = inner.add_parser("environments", help="list environments and presets")
    pe.set_defaults(func=_cmd_environments)

    pr = inner.add_parser(
        "rules", help="print a rule set as JSON, to save and edit"
    )
    _add_rule_flags(pr)
    pr.add_argument("-o", "--out", default=None, help="write to this file")
    pr.set_defaults(func=_cmd_rules)

    pp = inner.add_parser(
        "preview",
        help="render a few examples and report what they contain",
        description="Render a handful of pairs and write a contact sheet of "
        "left, right, ground truth and occlusion mask, with the measured "
        "statistics and any warnings.",
    )
    _add_rule_flags(pp)
    pp.add_argument("-n", "--samples", type=int, default=3,
                    help="how many example pairs (default: 3)")
    pp.add_argument("-o", "--out", default="figures/synth_preview.png",
                    help="where to write the contact sheet")
    pp.add_argument("--scale", type=float, default=1.0,
                    help="scale the tiles in the contact sheet")
    pp.add_argument("--save-rules", default=None, metavar="FILE",
                    help="also write the rule set that produced this")
    pp.add_argument("--open", action="store_true",
                    help="open the contact sheet when it is written")
    pp.set_defaults(func=_cmd_preview)

    pg = inner.add_parser(
        "generate",
        help="generate a dataset on disk",
        description="Write N pairs in the Middlebury layout, in parallel, "
        "with a manifest recording the rules and the measured result.",
    )
    _add_rule_flags(pg)
    pg.add_argument("-n", "--count", type=int, required=True,
                    help="how many pairs to generate")
    pg.add_argument("-o", "--out", required=True,
                    help="output directory for the dataset")
    pg.add_argument("--workers", type=int, default=None,
                    help="parallel processes (default: one per core)")
    pg.add_argument("--start-index", type=int, default=0,
                    help="first sample number, for extending a dataset")
    pg.add_argument("--environments", nargs="+", default=None, metavar="ENV",
                    choices=sorted(ENVIRONMENTS),
                    help="cycle through these environments instead of one")
    pg.add_argument(
        "--splits", default="train=0.8,val=0.1,test=0.1", metavar="SPEC",
        help="partition the output into subdirectories, as "
             "name=fraction pairs (default: train=0.8,val=0.1,test=0.1). "
             "Use --splits '' for one flat directory.",
    )
    pg.set_defaults(func=_cmd_generate)

    def _no_sub(args: argparse.Namespace) -> int:
        p.print_help()
        return 0

    p.set_defaults(func=_no_sub)
