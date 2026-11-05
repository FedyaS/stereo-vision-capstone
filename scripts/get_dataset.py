"""Download the Middlebury MiddEval3 stereo dataset (quarter resolution).

    python scripts/get_dataset.py            # training split + ground truth
    python scripts/get_dataset.py --test     # also the 15 test scenes (no GT)
    python scripts/get_dataset.py --force    # re-download

Fetches ~45 MB from vision.middlebury.edu into ``data/middlebury/``:

* ``MiddEval3-data-Q.zip`` -- image pairs and calibration for 30 scenes
* ``MiddEval3-GT0-Q.zip``  -- float ground-truth disparity for the 15
  training scenes, plus their non-occluded masks

Quarter resolution (~450x375 to ~750x500) is deliberate: it is the resolution
the Middlebury leaderboard's "Q" tables use, and it keeps a from-scratch
NumPy matcher at a second or two per pair rather than minutes.

The data is released for research use; see
https://vision.middlebury.edu/stereo/data/scenes2014/ for terms and the
citation (Scharstein et al., GCPR 2014).
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD_DIR = REPO_ROOT / "data" / "downloads"
EXTRACT_DIR = REPO_ROOT / "data" / "middlebury"

BASE = "https://vision.middlebury.edu/stereo/submit3/zip"
ARCHIVES = {
    "data": "MiddEval3-data-Q.zip",
    "gt": "MiddEval3-GT0-Q.zip",
}


def _download(url: str, dest: Path, force: bool = False) -> Path:
    if dest.exists() and not force:
        print(f"  have {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  fetching {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as f:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while chunk := resp.read(1 << 16):
            f.write(chunk)
            done += len(chunk)
            if total:
                pct = 100.0 * done / total
                print(f"\r    {done / 1e6:6.1f} / {total / 1e6:.1f} MB ({pct:5.1f}%)",
                      end="", flush=True)
        print()
    tmp.replace(dest)
    return dest


def _safe_extract(archive: Path, dest: Path) -> int:
    """Extract, refusing any entry that would escape ``dest``."""
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    with zipfile.ZipFile(archive) as zf:
        for name in zf.namelist():
            target = (root / name).resolve()
            if not str(target).startswith(str(root)):
                raise RuntimeError(f"{archive.name}: unsafe path {name!r}")
        zf.extractall(root)
        return len(zf.namelist())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="re-download archives")
    ap.add_argument("--test", action="store_true",
                    help="keep the test split too (no ground truth)")
    ap.add_argument("--keep-archives", action="store_true",
                    help="do not delete the zips after extracting")
    args = ap.parse_args(argv)

    print(f"Middlebury MiddEval3 (quarter resolution) -> {EXTRACT_DIR}")
    for key, name in ARCHIVES.items():
        path = _download(f"{BASE}/{name}", DOWNLOAD_DIR / name, force=args.force)
        n = _safe_extract(path, EXTRACT_DIR)
        print(f"  extracted {n} entries from {name}")

    if not args.test:
        test_dir = EXTRACT_DIR / "MiddEval3" / "testQ"
        if test_dir.exists():
            shutil.rmtree(test_dir)
            print("  removed testQ (no ground truth); pass --test to keep it")

    if not args.keep_archives:
        shutil.rmtree(DOWNLOAD_DIR, ignore_errors=True)
        print("  removed archives; pass --keep-archives to keep them")

    training = EXTRACT_DIR / "MiddEval3" / "trainingQ"
    scenes = sorted(p.name for p in training.iterdir() if p.is_dir()) if training.is_dir() else []
    print(f"\n{len(scenes)} training scenes ready: {', '.join(scenes)}")
    if not scenes:
        print("No scenes found -- something went wrong.", file=sys.stderr)
        return 1
    print("\nNext: python -m bench run -a opencv --limit 3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
