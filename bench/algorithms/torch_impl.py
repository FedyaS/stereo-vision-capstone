"""The same matching maths in PyTorch, so it can run on a GPU.

These are not learned models -- they are the classical cost-volume matchers
re-expressed as tensor ops. The point is the resource comparison: identical
mathematics, different backend, so the ``wall_s`` / ``parallelism`` /
``gpu_peak_mb`` columns show what the hardware contributes rather than what
the algorithm contributes.

On a CUDA machine the cost-volume methods here should beat the NumPy versions
by a large factor. On a CPU-only install they will land close to NumPy, since
both end up in a blocked BLAS-free elementwise loop; ``device`` is recorded in
the result's ``extra`` so the numbers are never ambiguous about which ran.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..types import DisparityResult, StereoPair
from .base import register, to_gray

_FAMILY = "torch"
BIG = 1e4


def _check_torch() -> tuple[bool, str]:
    try:
        import torch  # noqa: F401
    except Exception as exc:
        return False, f"torch not importable: {exc}"
    return True, ""


def pick_device(prefer: str = "auto") -> str:
    """``"cuda"`` when a GPU is usable, else ``"cpu"``."""
    import torch

    if prefer != "auto":
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _box_sum(x, size: int):
    """Window sum over the last two dims, via a separable average pool."""
    import torch.nn.functional as f

    pad = size // 2
    x = f.pad(x, (pad, pad, pad, pad), mode="reflect")
    return f.avg_pool2d(x, kernel_size=size, stride=1, divisor_override=1)


def _census_bits(gray, win_h: int = 7, win_w: int = 9):
    """Census transform as a ``(1, bits, H, W)`` float tensor of 0/1 values.

    Kept as separate channels rather than packed into an integer, because
    Hamming distance then becomes an L1 distance over channels, which is a
    single fused tensor op instead of a bit-twiddle plus popcount.
    """
    import torch
    import torch.nn.functional as f

    pad_h, pad_w = win_h // 2, win_w // 2
    padded = f.pad(gray, (pad_w, pad_w, pad_h, pad_h), mode="replicate")
    planes = []
    for dy in range(win_h):
        for dx in range(win_w):
            if dy == pad_h and dx == pad_w:
                continue
            neigh = padded[..., dy : dy + gray.shape[-2], dx : dx + gray.shape[-1]]
            planes.append((neigh > gray).to(gray.dtype))
    return torch.cat(planes, dim=1)


@dataclass
class TorchMatcher:
    """Cost volume + optional SGM aggregation, in torch.

    ``metric``: ``sad`` | ``ssd`` | ``zncc`` | ``census``.
    ``aggregation``: ``none`` | ``sgm4``.
    """

    metric: str = "sad"
    block: int = 9
    #: Census neighbourhood as ``(height, width)``; mirrors ScratchMatcher.
    census_win: tuple[int, int] = (7, 9)
    aggregation: str = "none"
    p1: float = 10.0
    p2: float = 120.0
    subpixel: bool = True
    device: str = "auto"
    dtype: str = "float32"

    # ---------------------------------------------------------------- cost
    def _cost_volume(self, pair: StereoPair, dev, dt):
        import torch

        nd = pair.ndisp
        if self.metric == "census":
            gl = torch.from_numpy(to_gray(pair.left)).to(dev, dt)[None, None]
            gr = torch.from_numpy(to_gray(pair.right)).to(dev, dt)[None, None]
            fl = _census_bits(gl, *self.census_win)
            fr = _census_bits(gr, *self.census_win)
        elif self.metric == "zncc":
            fl = torch.from_numpy(to_gray(pair.left)).to(dev, dt)[None, None]
            fr = torch.from_numpy(to_gray(pair.right)).to(dev, dt)[None, None]
        else:
            fl = torch.from_numpy(
                np.ascontiguousarray(pair.left.transpose(2, 0, 1))
            ).to(dev, dt)[None]
            fr = torch.from_numpy(
                np.ascontiguousarray(pair.right.transpose(2, 0, 1))
            ).to(dev, dt)[None]

        h, w = pair.shape
        cost = torch.full((nd, h, w), BIG, device=dev, dtype=dt)
        n_win = float(self.block * self.block)

        if self.metric == "zncc":
            mu_l = _box_sum(fl, self.block) / n_win
            var_l = (_box_sum(fl * fl, self.block) / n_win - mu_l * mu_l).clamp_min(0)

        for d in range(nd):
            if d == 0:
                shifted = fr
            else:
                # Replicate column 0 into the vacated strip, matching
                # `scratch._shift_right`. Zero-filling here would make windows
                # straddling the boundary compare against black and diverge
                # from the NumPy backend -- which is what
                # `test_numpy_and_torch_backends_agree` exists to catch.
                shifted = torch.empty_like(fr)
                shifted[..., d:] = fr[..., :-d]
                shifted[..., :d] = fr[..., :1]

            if self.metric == "zncc":
                mu_r = _box_sum(shifted, self.block) / n_win
                var_r = (
                    _box_sum(shifted * shifted, self.block) / n_win - mu_r * mu_r
                ).clamp_min(0)
                cross = _box_sum(fl * shifted, self.block) / n_win - mu_l * mu_r
                c = 1.0 - (cross / (torch.sqrt(var_l * var_r) + 1e-6)).clamp(-1, 1)
            else:
                diff = fl - shifted
                per_px = diff.abs() if self.metric in ("sad", "census") else diff * diff
                c = _box_sum(per_px.sum(dim=1, keepdim=True), self.block) / n_win

            sl = c[0, 0]
            if d > 0:
                cost[d, :, d:] = sl[:, d:]
            else:
                cost[d] = sl
        return cost

    # --------------------------------------------------------- aggregation
    def _normalize_cost(self, cost, scale: float = 64.0):
        """Rescale to ``0..scale`` so P1/P2 mean the same for every metric.

        Mirrors :func:`bench.algorithms.scratch.normalize_cost`; the two
        backends have to agree or the cross-check in ``tests/`` is pointless.
        """
        import torch

        valid = cost < BIG * 0.5
        sample = cost[valid].float()
        if sample.numel() == 0:
            return cost
        stride = max(1, sample.numel() // 200_000)
        q = torch.quantile(
            sample[::stride], torch.tensor([0.01, 0.99], device=sample.device)
        )
        lo, hi = float(q[0]), float(q[1])
        if hi <= lo:
            hi = lo + 1.0
        out = ((cost - lo) / (hi - lo)).clamp(0.0, 1.0) * scale
        return torch.where(valid, out, torch.full_like(cost, BIG))

    def _sgm4(self, cost, p1: float, p2: float):
        import torch

        nd, h, w = cost.shape
        agg = torch.zeros_like(cost)

        def step(c, prev):
            m = prev.min(dim=0).values
            best = prev.clone()
            best[1:] = torch.minimum(best[1:], prev[:-1] + p1)
            best[:-1] = torch.minimum(best[:-1], prev[1:] + p1)
            best = torch.minimum(best, m + p2)
            return c + best - m

        for horizontal in (True, False):
            n = w if horizontal else h
            for forward in (True, False):
                prev = torch.zeros(
                    (nd, h if horizontal else w), device=cost.device, dtype=cost.dtype
                )
                idx = range(n) if forward else range(n - 1, -1, -1)
                for i in idx:
                    sl = cost[:, :, i] if horizontal else cost[:, i, :]
                    prev = step(sl, prev)
                    if horizontal:
                        agg[:, :, i] += prev
                    else:
                        agg[:, i, :] += prev
        return agg

    # ------------------------------------------------------------- driver
    def compute(self, pair: StereoPair) -> DisparityResult:
        import torch

        dev = torch.device(pick_device(self.device))
        dt = getattr(torch, self.dtype)

        with torch.inference_mode():
            cost = self._cost_volume(pair, dev, dt)

            if self.aggregation == "sgm4":
                cost = self._sgm4(
                    self._normalize_cost(cost), self.p1, self.p2
                )
            elif self.aggregation != "none":
                raise ValueError(f"unknown aggregation {self.aggregation!r}")

            best_c, best_d = cost.min(dim=0)
            disp = best_d.to(torch.float32)

            if self.subpixel and pair.ndisp >= 3:
                nd = cost.shape[0]
                dm = (best_d - 1).clamp(0, nd - 1)
                dp = (best_d + 1).clamp(0, nd - 1)
                c_m = torch.gather(cost, 0, dm[None])[0]
                c_p = torch.gather(cost, 0, dp[None])[0]
                denom = c_m - 2 * best_c + c_p
                delta = torch.where(
                    denom.abs() > 1e-6,
                    (c_m - c_p) / (2 * denom),
                    torch.zeros_like(denom),
                )
                delta = torch.where(
                    delta.abs() <= 0.5, delta, torch.zeros_like(delta)
                )
                interior = (best_d > 0) & (best_d < nd - 1)
                disp = disp + torch.where(
                    interior, delta, torch.zeros_like(delta)
                ).to(torch.float32)

            # Geometric validity: the match x - d must land inside the image.
            # A cost-magnitude test is wrong after aggregation, where BIG has
            # been smeared along the paths.
            w = cost.shape[-1]
            xx = torch.arange(w, device=disp.device, dtype=disp.dtype)[None, :]
            valid = (disp <= xx) & (disp >= 0) & torch.isfinite(disp)
            disp_np = disp.detach().cpu().numpy().astype(np.float32)
            valid_np = valid.detach().cpu().numpy()

        return DisparityResult(
            np.where(valid_np, disp_np, np.nan).astype(np.float32),
            valid_np,
            {"device": str(dev)},
        )


def _entry(name: str, blurb: str, tags: tuple[str, ...], **kw):
    @register(name, family=_FAMILY, description=blurb, tags=tags, check=_check_torch)
    def _build(_kw=kw) -> TorchMatcher:
        return TorchMatcher(**_kw)

    return _build


# These mirror their `scratch_` counterparts exactly, including the tuned
# defaults. They compute the same function, so giving them different
# parameters would quietly turn the backend comparison into a comparison of
# two different algorithms. The sweep's torch grids are confirmatory only --
# `test_torch_defaults_mirror_scratch` is what keeps them in step.
_entry(
    "torch_sad_wta",
    "SAD cost volume + winner-takes-all, as tensor ops (GPU if available)",
    ("local", "wta", "sad", "gpu-capable"),
    metric="sad",
    block=29,
)
_entry(
    "torch_zncc_wta",
    "ZNCC cost volume + winner-takes-all, as tensor ops (GPU if available)",
    ("local", "wta", "zncc", "gpu-capable"),
    metric="zncc",
    block=13,
)
_entry(
    "torch_census_sgm4",
    "Census cost + 4-path SGM, as tensor ops (GPU if available)",
    ("global", "sgm", "census", "gpu-capable"),
    metric="census",
    block=7,
    aggregation="sgm4",
    p1=20.0,
)
