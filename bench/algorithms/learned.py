"""Learned stereo networks, run through ONNX Runtime.

The reference point the classical methods are measured against. Weights are
not in the repo -- fetch them with ``python scripts/get_models.py``, and any
model whose file is missing stays registered but reports itself unavailable
rather than crashing the run.

Currently wired up: **RAFT-Stereo** (Lipson et al., 3DV 2021), the recurrent
all-pairs-correlation matcher, in two ONNX exports taken from the
`AXERA-TECH/RAFT-stereo <https://huggingface.co/AXERA-TECH/RAFT-stereo>`_
mirror -- one at 384x1280 with 4 refinement iterations, one at 256x640 with
1. The iteration count is the network's own accuracy/time dial, so having
both makes that trade-off visible in the same table as SGBM's path count.

Two things to keep in mind when reading these rows.

*Fixed input size.* Both exports have a frozen input resolution, so the pair
is resized in, and the disparity is resized out and **rescaled by the width
ratio** -- disparity is a horizontal length, so halving the width halves it.
That resampling costs real accuracy at Middlebury resolutions and is a
property of the export, not of RAFT-Stereo.

*Domain.* These exports are KITTI/SceneFlow-trained. Middlebury is indoor,
higher-disparity and out of domain for them, so treat the numbers as "a
pretrained network applied cold", not as RAFT-Stereo's published accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..types import DisparityResult, StereoPair
from ..paths import MODELS_DIR as MODEL_DIR
from .base import register

_FAMILY = "learned"

# name -> (filename, huggingface repo path)
MODEL_SOURCES: dict[str, tuple[str, str]] = {
    "raft_stereo_384x1280_r4": (
        "raft_steoro384x1280_r4.onnx",
        "AXERA-TECH/RAFT-stereo/resolve/main/ax650/raft_steoro384x1280_r4.onnx",
    ),
    "raft_stereo_256x640_r1": (
        "raft_steoro256x640_r1.onnx",
        "AXERA-TECH/RAFT-stereo/resolve/main/ax630c/raft_steoro256x640_r1.onnx",
    ),
}


def _check(filename: str):
    def inner() -> tuple[bool, str]:
        try:
            import onnxruntime  # noqa: F401
        except Exception as exc:
            return False, f"onnxruntime not installed ({exc})"
        if not (MODEL_DIR / filename).exists():
            return False, f"missing weights {filename} -- run scripts/get_models.py"
        return True, ""

    return inner


@dataclass
class OnnxStereoMatcher:
    """Run a fixed-input-size ONNX stereo network on an arbitrary pair.

    ``normalize`` selects the input scaling:

    ``"raw"``
        pass ``0..255`` through unchanged. **The default**, because these
        exports fold RAFT-Stereo's own normalisation into the graph; feeding
        them pre-normalised input triples the error (``bad 2.0`` on Teddy goes
        from 5.3% to 23.3%). Verified against ground truth rather than assumed
        -- see ``tests/test_learned.py``.
    ``"raft"``
        ``2 * (x / 255) - 1``, RAFT-Stereo's upstream preprocessing. Needed
        for exports that do *not* bake it in.
    ``"unit"``
        ``x / 255``.
    """

    filename: str
    normalize: str = "raw"
    providers: tuple[str, ...] = ("CPUExecutionProvider",)
    intra_op_threads: int | None = None
    _session: object | None = field(default=None, repr=False, compare=False)
    _init_s: float = field(default=0.0, repr=False, compare=False)

    # ------------------------------------------------------------------
    def _ensure_session(self):
        import time

        import onnxruntime as ort

        if self._session is not None:
            return self._session
        path = MODEL_DIR / self.filename
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run `python scripts/get_models.py`."
            )
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        if self.intra_op_threads:
            opts.intra_op_num_threads = int(self.intra_op_threads)
        t0 = time.perf_counter()
        self._session = ort.InferenceSession(
            str(path), opts, providers=list(self.providers)
        )
        self._init_s = time.perf_counter() - t0
        return self._session

    def _prep(self, img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        import cv2

        h, w = size
        resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        x = resized.astype(np.float32)
        if self.normalize == "raft":
            x = 2.0 * (x / 255.0) - 1.0
        elif self.normalize == "unit":
            x = x / 255.0
        elif self.normalize != "raw":
            raise ValueError(f"unknown normalize mode {self.normalize!r}")
        return np.ascontiguousarray(x.transpose(2, 0, 1)[None])

    # ------------------------------------------------------------------
    def compute(self, pair: StereoPair) -> DisparityResult:
        import cv2

        sess = self._ensure_session()
        inputs = sess.get_inputs()
        if len(inputs) < 2:
            raise RuntimeError(
                f"{self.filename}: expected two image inputs, got "
                f"{[i.name for i in inputs]}"
            )

        shape = [d if isinstance(d, int) else -1 for d in inputs[0].shape]
        mh, mw = shape[-2], shape[-1]
        h, w = pair.shape
        if mh <= 0 or mw <= 0:  # dynamic axes: feed the native size
            mh, mw = h, w

        feed = {
            inputs[0].name: self._prep(pair.left, (mh, mw)),
            inputs[1].name: self._prep(pair.right, (mh, mw)),
        }
        out = np.asarray(sess.run(None, feed)[0], dtype=np.float32)
        while out.ndim > 2:
            out = out[0]

        # RAFT-Stereo emits a negative horizontal flow; disparity is its
        # magnitude. Decide by sign rather than hardcoding, so a differently
        # exported model still works.
        disp_model = -out if float(np.nanmean(out)) < 0 else out

        # Resize back, then rescale: disparity is a horizontal length, so it
        # scales with the width ratio, not with area.
        disp = cv2.resize(disp_model, (w, h), interpolation=cv2.INTER_LINEAR)
        disp = disp * (float(w) / float(mw))

        valid = np.isfinite(disp) & (disp > 0)
        return DisparityResult(
            np.where(valid, disp, np.nan).astype(np.float32),
            valid,
            {
                "session_init_s": round(self._init_s, 3),
                "model_input": f"{mw}x{mh}",
                "resize_factor": round(float(w) / float(mw), 5),
                "providers": list(self.providers),
            },
        )


@register(
    "learned_raft_stereo_r4",
    family=_FAMILY,
    description="RAFT-Stereo ONNX, 384x1280 input, 4 refinement iterations",
    tags=("learned", "raft", "accurate"),
    check=_check(MODEL_SOURCES["raft_stereo_384x1280_r4"][0]),
)
def _raft_r4() -> OnnxStereoMatcher:
    return OnnxStereoMatcher(filename=MODEL_SOURCES["raft_stereo_384x1280_r4"][0])


@register(
    "learned_raft_stereo_r1",
    family=_FAMILY,
    description="RAFT-Stereo ONNX, 256x640 input, 1 refinement iteration",
    tags=("learned", "raft", "fast"),
    check=_check(MODEL_SOURCES["raft_stereo_256x640_r1"][0]),
)
def _raft_r1() -> OnnxStereoMatcher:
    return OnnxStereoMatcher(filename=MODEL_SOURCES["raft_stereo_256x640_r1"][0])
