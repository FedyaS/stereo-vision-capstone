"""Download the pretrained ONNX stereo networks into ``models/``.

    python scripts/get_models.py             # all known models (~70 MB)
    python scripts/get_models.py --list
    python scripts/get_models.py raft_stereo_384x1280_r4

Models come from the ``AXERA-TECH/RAFT-stereo`` mirror on Hugging Face, which
publishes ONNX exports of RAFT-Stereo (Lipson, Teed & Deng, 3DV 2021) at fixed
input sizes. They are KITTI/SceneFlow-trained, so on Middlebury they are being
applied out of domain -- see ``bench/algorithms/learned.py``.

Without these files the ``learned_*`` algorithms report themselves as
unavailable and the rest of the benchmark runs normally.
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "models"
HF = "https://huggingface.co"

# key -> (local filename, repo-relative source path)
MODELS: dict[str, tuple[str, str]] = {
    "raft_stereo_384x1280_r4": (
        "raft_steoro384x1280_r4.onnx",
        "AXERA-TECH/RAFT-stereo/resolve/main/ax650/raft_steoro384x1280_r4.onnx",
    ),
    "raft_stereo_256x640_r1": (
        "raft_steoro256x640_r1.onnx",
        "AXERA-TECH/RAFT-stereo/resolve/main/ax630c/raft_steoro256x640_r1.onnx",
    ),
}


def _download(url: str, dest: Path, force: bool = False) -> None:
    if dest.exists() and not force:
        print(f"  have {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  fetching {dest.name}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "stereo-bench/0.1"})
    with urllib.request.urlopen(req, timeout=180) as resp, open(tmp, "wb") as f:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while chunk := resp.read(1 << 16):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r    {done / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", flush=True)
        print()
    tmp.replace(dest)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="*", default=None,
                    help="model keys to fetch; default is all")
    ap.add_argument("--list", action="store_true", help="list known models and exit")
    ap.add_argument("--force", action="store_true", help="re-download")
    args = ap.parse_args(argv)

    if args.list:
        for key, (fname, src) in MODELS.items():
            have = (MODEL_DIR / fname).exists()
            print(f"  {'[have]' if have else '[    ]'} {key:28s} {fname}")
            print(f"           {HF}/{src}")
        return 0

    keys = args.models or list(MODELS)
    unknown = [k for k in keys if k not in MODELS]
    if unknown:
        print(f"unknown model(s): {', '.join(unknown)}", file=sys.stderr)
        print(f"known: {', '.join(MODELS)}", file=sys.stderr)
        return 2

    print(f"ONNX stereo models -> {MODEL_DIR}")
    for key in keys:
        fname, src = MODELS[key]
        _download(f"{HF}/{src}", MODEL_DIR / fname, force=args.force)

    print("\nVerifying with onnxruntime...")
    try:
        import onnxruntime as ort
    except ImportError:
        print("  onnxruntime not installed; run `pip install -r requirements.txt`")
        return 1

    for key in keys:
        fname, _ = MODELS[key]
        path = MODEL_DIR / fname
        try:
            opts = ort.SessionOptions()
            opts.log_severity_level = 3
            sess = ort.InferenceSession(str(path), opts,
                                        providers=["CPUExecutionProvider"])
            shapes = [i.shape for i in sess.get_inputs()]
            print(f"  ok {key}: inputs {shapes}")
        except Exception as exc:
            print(f"  FAILED {key}: {exc}", file=sys.stderr)
            return 1

    print("\nNext: python -m bench run -a learned --limit 3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
