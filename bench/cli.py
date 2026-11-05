"""Command line entry point.

::

    python -m bench list
    python -m bench run                                   # everything, Middlebury
    python -m bench run -a opencv scratch -d middlebury --limit 3
    python -m bench run -a opencv_sgbm_3way scratch_census_sgm4 --repeats 5
    python -m bench run -d synthetic:3 --threads 1
    python -m bench run -d practice --downscale 6 -a opencv learned
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .algorithms import all_specs, select
from .datasets import list_middlebury_scenes, resolve_pairs
from .datasets.local import load_image_pair
from .profiling import machine_info
from .report import write_all
from .runner import RunConfig, run_benchmark


def _cmd_list(args: argparse.Namespace) -> int:
    specs = all_specs()
    by_family: dict[str, list] = {}
    for s in specs:
        by_family.setdefault(s.family, []).append(s)

    print(f"{len(specs)} algorithms registered\n")
    for family, items in by_family.items():
        print(f"[{family}]")
        for s in items:
            ok, why = s.availability()
            flag = "  " if ok else "! "
            print(f"  {flag}{s.name:32s} {s.description}")
            if not ok:
                print(f"     unavailable: {why}")
            if args.verbose and s.tags:
                print(f"     tags: {', '.join(s.tags)}")
        print()

    scenes = list_middlebury_scenes()
    print(f"[datasets]")
    print(f"   middlebury        {len(scenes)} scenes with ground truth")
    if scenes:
        print(f"                     {', '.join(scenes)}")
    else:
        print("                     not downloaded -- run scripts/get_dataset.py")
    print("   synthetic[:N]     generated pairs with exact ground truth")
    print("   <folder>          a folder containing im0/im1 (e.g. `practice`)")
    print()

    if args.verbose:
        print("[machine]")
        for k, v in machine_info().items():
            print(f"   {k:24s} {v}")
    return 0


def _dataset_spec(dataset: str) -> str:
    """Expand the split shorthands ``tune`` / ``heldout`` into scene lists."""
    from .splits import SPLITS, split_spec

    return split_spec(dataset) if dataset in SPLITS else dataset


def _cmd_run(args: argparse.Namespace) -> int:
    try:
        specs = select(args.algorithms)
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    args.dataset = _dataset_spec(args.dataset)
    if args.dataset.split(":")[0] in ("middlebury", "synthetic"):
        pairs = resolve_pairs(args.dataset, limit=args.limit)
    else:
        pairs = [
            load_image_pair(
                args.dataset, downscale=args.downscale, ndisp=args.ndisp
            )
        ]

    if not pairs:
        print(
            f"error: dataset {args.dataset!r} produced no pairs. "
            "Run `python scripts/get_dataset.py` to fetch Middlebury, "
            "or use `-d synthetic`.",
            file=sys.stderr,
        )
        return 2

    available = [s for s in specs if s.availability()[0]]
    print(
        f"{len(pairs)} pair(s) x {len(specs)} algorithm(s) "
        f"({len(specs) - len(available)} unavailable) = "
        f"{len(pairs) * len(specs)} cells\n"
    )

    config = RunConfig(
        repeats=args.repeats,
        warmup=args.warmup,
        threads=args.threads,
        eval_region=args.eval_region,
        save_maps=args.save_maps,
        save_figures=not args.no_figures,
        max_volume_gb=args.max_volume_gb,
        out_dir=Path(args.out),
    )

    results, out_dir = run_benchmark(pairs, specs, config)
    summary = write_all(results, out_dir)

    print(f"\nSummary: {summary}")
    print(f"Figures: {out_dir / 'figures'}")
    n_bad = sum(c.status != "ok" for c in results)
    if n_bad:
        print(f"({n_bad} cell(s) skipped or failed -- see the summary)")
    return 0


def _cmd_sweep(args: argparse.Namespace) -> int:
    from datetime import datetime, timezone

    from . import sweep as sweep_mod
    from . import sweep_report
    from .profiling import set_thread_limit
    from .splits import describe

    try:
        specs = select(args.algorithms)
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    grids = sweep_mod.GRID_SETS[args.grid]
    have_grid = [s for s in specs if grids.get(s.name)]
    available = [s for s in have_grid if s.availability()[0]]
    if not available:
        print(
            f"error: none of the selected algorithms has a {args.grid} grid. "
            f"Grids exist for: {', '.join(sorted(grids))}",
            file=sys.stderr,
        )
        return 2

    pairs = resolve_pairs(_dataset_spec(args.dataset), limit=args.limit)
    if not pairs:
        print(f"error: dataset {args.dataset!r} produced no pairs", file=sys.stderr)
        return 2

    set_thread_limit(args.threads)
    n_cfg = sum(len(sweep_mod.iter_configs(grids[s.name])) for s in available)
    print(describe())
    print(
        f"\nSweeping {len(available)} algorithm(s) with the {args.grid} grids, "
        f"{n_cfg} configurations "
        f"x {len(pairs)} scene(s) = {n_cfg * len(pairs)} runs\n"
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out) / stamp
    all_cells, out_dir = sweep_mod.run_sweeps(
        [s.name for s in available],
        pairs,
        out_dir,
        grids=grids,
        eval_region=args.eval_region,
        max_volume_gb=args.max_volume_gb,
    )
    summary = sweep_report.write_all(
        all_cells,
        out_dir,
        tune_scenes=[p.name for p in pairs],
        tolerance_pp=args.tolerance,
        grids=grids,
    )
    print(f"\nSummary: {summary}")
    print(f"Figures: {out_dir / 'figures'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m bench",
        description="Benchmark stereo matching algorithms on accuracy, time "
        "and compute resources.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="command")

    pl = sub.add_parser("list", help="list algorithms, datasets and the machine")
    pl.add_argument("-v", "--verbose", action="store_true")
    pl.set_defaults(func=_cmd_list)

    pr = sub.add_parser("run", help="run the benchmark")
    pr.add_argument(
        "-a", "--algorithms", nargs="+", default=None,
        metavar="SELECTOR",
        help="names, families (opencv/scratch/torch/learned), tags, or globs. "
        "Default: all.",
    )
    pr.add_argument(
        "-d", "--dataset", default="middlebury",
        help="middlebury[:Scene,...] | synthetic[:N] | a folder path. "
        "Default: middlebury.",
    )
    pr.add_argument("--limit", type=int, default=None,
                    help="use at most this many scenes")
    pr.add_argument("--repeats", type=int, default=1,
                    help="timed runs per cell; the median is reported")
    pr.add_argument("--warmup", type=int, default=1,
                    help="untimed runs before timing (keep >=1)")
    pr.add_argument("--threads", type=int, default=None,
                    help="pin every backend to N threads for a fair compute "
                    "comparison. Default: library defaults.")
    pr.add_argument("--eval-region", choices=("nonocc", "all"), default="nonocc",
                    help="score the non-occluded region (Middlebury standard) "
                    "or everything with ground truth")
    pr.add_argument("--downscale", type=float, default=1.0,
                    help="for folder datasets: divide image dimensions by this")
    pr.add_argument("--ndisp", type=int, default=None,
                    help="override the disparity search range")
    pr.add_argument("--max-volume-gb", type=float, default=4.0,
                    help="skip a cell whose cost volume would exceed this")
    pr.add_argument("--save-maps", action="store_true",
                    help="also write each disparity map as a .pfm")
    pr.add_argument("--no-figures", action="store_true",
                    help="skip PNG rendering (faster)")
    pr.add_argument("-o", "--out", default="results",
                    help="output root; a timestamped subfolder is created")
    pr.set_defaults(func=_cmd_run)

    ps = sub.add_parser(
        "sweep",
        help="sweep algorithm parameters on the tune split",
        description="Grid-search each algorithm's parameters on the tune "
        "scenes only. The held-out scenes are never loaded, so the benchmark "
        "run afterwards still measures generalisation.",
    )
    ps.add_argument(
        "-a", "--algorithms", nargs="+", default=None, metavar="SELECTOR",
        help="names, families, tags or globs. Default: everything with a grid.",
    )
    ps.add_argument(
        "-d", "--dataset", default="tune",
        help="tune | heldout | all | middlebury[:Scene,...] | synthetic[:N]. "
        "Default: tune.",
    )
    ps.add_argument("--limit", type=int, default=None,
                    help="use at most this many scenes")
    ps.add_argument(
        "--grid", choices=("default", "refined"), default="default",
        help="which grid set to sweep. 'refined' is the coarse-to-fine second "
        "pass: it extends past the ranges the first sweep bottomed or topped "
        "out on, and bisects around the ones it landed inside.",
    )
    ps.add_argument("--threads", type=int, default=None,
                    help="pin every backend to N threads")
    ps.add_argument("--eval-region", choices=("nonocc", "all"), default="nonocc")
    ps.add_argument("--max-volume-gb", type=float, default=4.0)
    ps.add_argument(
        "--tolerance", type=float, default=0.5, metavar="PP",
        help="among configs within this many percentage points of the best "
        "bad2_all, pick the fastest (default: 0.5)",
    )
    ps.add_argument("-o", "--out", default="results/sweeps",
                    help="output root; a timestamped subfolder is created")
    ps.set_defaults(func=_cmd_sweep)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
