"""Copy a benchmark run's files into ``results/keep/`` so they can be committed.

    python scripts/keep_result.py                      # the most recent run
    python scripts/keep_result.py 20261105-035941      # a specific run
    python scripts/keep_result.py --figures            # bring figures/ too
    python scripts/keep_result.py --all                # the whole folder
    python scripts/keep_result.py --list               # what is already kept

``results/`` is gitignored because a run is reproducible and a full one is
tens of megabytes. ``results/keep/`` is un-ignored (see ``.gitignore``), so
anything placed there is committed with a plain ``git add`` -- no ``-f``.

By default only the small, durable files travel: ``summary.md``,
``results.csv``, ``run.json``, and ``report.html`` / ``report.pdf`` when they
exist. ``maps/`` and ``figures/`` are the bulk and are left behind unless you
ask for them.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "results"
KEEP_ROOT = RESULTS_ROOT / "keep"

#: Small files worth committing. Order is the order they are reported in.
DEFAULT_FILES = ("summary.md", "results.csv", "run.json", "report.html", "report.pdf")

#: Roughly where a commit stops being a good idea. Not a hard limit -- the
#: script warns and keeps going, because sometimes you really do want it.
WARN_MB = 20.0


def _human(n_bytes: int) -> str:
    mb = n_bytes / 1e6
    return f"{n_bytes / 1e3:.0f} kB" if mb < 1 else f"{mb:.1f} MB"


def _dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _is_run(path: Path) -> bool:
    return path.is_dir() and (path / "results.csv").is_file()


def _runs() -> list[Path]:
    """Live run folders, newest first. ``keep/`` is an archive, not a run."""
    if not RESULTS_ROOT.is_dir():
        return []
    return sorted(
        (d for d in RESULTS_ROOT.iterdir() if d.name != "keep" and _is_run(d)),
        key=lambda d: d.name,
        reverse=True,
    )


def _resolve(run: str | None) -> Path | None:
    """Find the run the user means, by name or as 'the latest one'."""
    if run is None:
        runs = _runs()
        return runs[0] if runs else None
    # Accept '20261105-035941', 'results/20261105-035941' or a full path.
    candidate = Path(run)
    for guess in (candidate, RESULTS_ROOT / candidate.name, REPO_ROOT / candidate):
        if _is_run(guess):
            return guess.resolve()
    return None


def _cmd_list() -> int:
    if not KEEP_ROOT.is_dir():
        print(f"nothing kept yet ({KEEP_ROOT} does not exist)")
        return 0
    kept = sorted(d for d in KEEP_ROOT.iterdir() if d.is_dir())
    if not kept:
        print("nothing kept yet")
        return 0
    print(f"{len(kept)} kept under results/keep/:\n")
    for d in kept:
        files = sorted(p.name for p in d.iterdir() if p.is_file())
        print(f"  {d.name:24s} {_human(_dir_size(d)):>9s}  {', '.join(files) or '(empty)'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("run", nargs="?", default=None,
                    help="run id or path; default is the most recent run")
    ap.add_argument("--as", dest="alias", default=None, metavar="NAME",
                    help="keep it under this name instead of the run id")
    ap.add_argument("--figures", action="store_true",
                    help="also copy figures/ (the PNGs, not the .pfm maps)")
    ap.add_argument("--all", action="store_true",
                    help="copy the entire run folder, maps included")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an already-kept copy of the same name")
    ap.add_argument("--list", action="store_true",
                    help="list what is already kept and exit")
    args = ap.parse_args(argv)

    if args.list:
        return _cmd_list()

    src = _resolve(args.run)
    if src is None:
        if args.run is None:
            print(f"error: no runs found under {RESULTS_ROOT}. "
                  "Run `python -m bench run` first.", file=sys.stderr)
        else:
            print(f"error: {args.run!r} is not a run folder "
                  "(no results.csv in it).", file=sys.stderr)
            if runs := _runs():
                print(f"available: {', '.join(d.name for d in runs[:8])}",
                      file=sys.stderr)
        return 2

    dest = KEEP_ROOT / (args.alias or src.name)
    if dest.exists() and not args.force:
        print(f"error: {dest.relative_to(REPO_ROOT)} already exists; "
              "pass --force to overwrite.", file=sys.stderr)
        return 2
    if dest.resolve() == src.resolve():
        print(f"error: {src} is already the destination.", file=sys.stderr)
        return 2

    if args.all:
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
        copied = sorted(
            str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file()
        )
    else:
        dest.mkdir(parents=True, exist_ok=True)
        copied = []
        for name in DEFAULT_FILES:
            if (src / name).is_file():
                shutil.copy2(src / name, dest / name)
                copied.append(name)
        if args.figures and (src / "figures").is_dir():
            shutil.copytree(src / "figures", dest / "figures", dirs_exist_ok=True)
            copied += [
                f"figures/{p.name}" for p in sorted((dest / "figures").glob("*.png"))
            ]

    if not copied:
        print(f"error: nothing to copy from {src} -- expected one of "
              f"{', '.join(DEFAULT_FILES)}.", file=sys.stderr)
        return 2

    rel = dest.relative_to(REPO_ROOT).as_posix()
    total = _dir_size(dest)
    print(f"{src.name} -> {rel}  ({len(copied)} file(s), {_human(total)})")
    for name in copied[:12]:
        print(f"  {name}")
    if len(copied) > 12:
        print(f"  ... and {len(copied) - 12} more")

    if total / 1e6 > WARN_MB:
        print(f"\nnote: {_human(total)} is large for a commit. Drop --all/--figures "
              "to keep just the tables and the report.")

    print(f"\nCommit it with:\n  git add {rel}\n  git commit -m \"Keep run {dest.name}\"")
    print("\nAnd record why in results/keep/README.md -- a kept result with no "
          "reason looks like one nobody cleaned up.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
