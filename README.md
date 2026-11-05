# stereo-vision-capstone

Stereo Vision 2026-2027 Capstone for Math and Data Science.

A test harness for comparing stereo-matching algorithms on the same images,
with the same disparity range, under the same measurement, reporting
**accuracy**, **time** and **compute resources** side by side.

23 algorithms across four families, 15 ground-truth scenes, one command:

```bash
python -m bench run
```

---

## Quickstart

```bash
python -m venv --system-site-packages .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate on POSIX
pip install -r requirements.txt
```

```bash
python scripts/get_dataset.py
```

```bash
python scripts/get_models.py
```

```bash
python -m bench list
```

```bash
python -m bench run -d middlebury:Teddy,Piano --repeats 3
```

Results land in `results/<timestamp>/`: read `summary.md`, look at
`figures/`, load `results.csv` for your own analysis.

Nothing above is strictly required to get started -- with no dataset and no
weights, `python -m bench run -d synthetic:3` still works, because the
synthetic generator produces pairs with exact ground truth in-process.

---

## What is included

### Algorithms

Four families. `python -m bench list` prints the live set with availability.

**`opencv`** -- the library implementations most projects actually reach for.

| name | what it is |
|---|---|
| `opencv_bm` | StereoBM: SAD block matching after an x-Sobel prefilter |
| `opencv_sgbm` | StereoSGBM, 5-path aggregation (the default mode) |
| `opencv_sgbm_hh` | SGBM, full 8-path whole-image DP |
| `opencv_sgbm_hh4` | SGBM, 4-path HH |
| `opencv_sgbm_3way` | SGBM, 3-path -- the speed-tuned mode |
| `opencv_sgbm_wls` | SGBM + right matcher + ximgproc WLS edge-aware filter |
| `opencv_bm_wls` | StereoBM + WLS filter |
| `opencv_quasi_dense` | QuasiDenseStereo: sparse feature seeds, then region growing |

The four SGBM modes differ only in how many aggregation paths the solver
walks, which is the accuracy/time dial this harness exists to measure, so they
are separate entries rather than one entry with a flag.

**`scratch`** -- NumPy/SciPy, every stage exposed and swappable. These are the
ones to modify when the question is *what does changing the cost function do?*

| name | cost | aggregation |
|---|---|---|
| `scratch_sad_wta` | SAD | none (box + winner-takes-all) |
| `scratch_ssd_wta` | SSD | none |
| `scratch_zncc_wta` | zero-mean normalised cross-correlation | none |
| `scratch_census_wta` | census + Hamming | none |
| `scratch_sad_wta_lrcheck` | SAD | none, + uniqueness + left-right check |
| `scratch_census_dp` | census | exact per-scanline Viterbi (Potts prior) |
| `scratch_census_sgm4` | census | semi-global, 4 paths |
| `scratch_census_sgm8` | census | semi-global, 8 paths |
| `scratch_census_sgm8_lrcheck` | census | 8 paths + left-right check |
| `scratch_sad_sgm8` | SAD | semi-global, 8 paths |

The pipeline is four replaceable functions in
[`bench/algorithms/scratch.py`](bench/algorithms/scratch.py):
`cost_volume` → `sgm_aggregate` / `potts_scanline_dp` → `winner_takes_all` →
`left_right_check`.

**`torch`** -- the same mathematics as tensor ops, so it runs on a GPU when one
is present: `torch_sad_wta`, `torch_zncc_wta`, `torch_census_sgm4`. Not learned
models. The point is the resource comparison: identical maths, different
backend, so the timing columns isolate the hardware's contribution. The device
actually used is recorded per run.

**`learned`** -- `learned_raft_stereo_r4` and `learned_raft_stereo_r1`:
pretrained **RAFT-Stereo** through ONNX Runtime, at 384x1280 with 4 refinement
iterations and at 256x640 with 1. The iteration count is the network's own
accuracy/time dial, so it sits in the same table as SGBM's path count.

### Datasets

| spec | what you get |
|---|---|
| `middlebury` | all 15 MiddEval3 quarter-res training scenes, dense float ground truth, non-occluded masks, per-scene `ndisp` |
| `middlebury:Teddy,Piano` | just those scenes |
| `synthetic[:N]` | generated pairs, **exact** ground truth, no download |
| `practice` (any folder) | a folder with `im0`/`im1`; add `--downscale 6` for the 2964x2000 pair already in the repo |

Each Middlebury scene brings its own disparity range from `calib.txt`, and the
harness uses it: Jadeplant needs 160 disparities, Teddy needs 64. Giving every
scene the same oversized range would make the easy scenes look needlessly slow.

---

## How the numbers are produced

### Accuracy

The headline metric is Middlebury's: **`bad N`** is the percentage of evaluated
pixels whose absolute disparity error exceeds `N` pixels, scored on the
non-occluded region by default (`--eval-region all` for everything with ground
truth).

One wrinkle decides whether a stereo comparison is honest. Some matchers return
a value everywhere; `opencv_bm` and `opencv_quasi_dense` leave large holes.
Scoring only where a method *chose* to answer rewards methods that answer
rarely, so every error metric is reported twice:

- **`bad2_all`** -- over the whole evaluation region, with missing estimates
  counted as errors. **This is the ranking metric**, and the only one directly
  comparable between dense and sparse methods.
- **`*_valid`** (`bad2_valid`, `epe_valid`, `rmse_valid`) -- only over pixels
  the method answered. Read these next to **`density`**, never alone.
- **`epe_filled`** -- over the whole region after cheap hole filling, which is
  what the Middlebury leaderboard effectively requires.

Also reported: end-point error, RMSE, the 90th/95th/99th error percentiles,
and signed bias (which catches a systematic offset that an absolute error hides).

`tests/test_metrics.py` pins this down, including a case where a method that
answers one perfect pixel out of a hundred must not outrank a dense method.

### Time and compute

`bench/profiling.py` samples the process while each algorithm runs. Beyond wall
time:

- **`cpu_s` and `parallelism`** (`cpu_s / wall_s`) -- roughly how many cores the
  method kept busy. OpenCV's SGBM multithreads internally; a hand-written NumPy
  loop does not. A NumPy method that looks 8x slower in wall time may be using
  8x *less* compute, which wall time alone hides completely. Use `--threads 1`
  to pin every backend and compare algorithms rather than threading quality.
- **`peak_rss_mb` / `rss_delta_mb`** -- peak resident memory, sampled at 200 Hz.
  The cost-volume methods are `D x H x W` floats and it shows.
- **`gpu_peak_mb`** -- peak CUDA allocation, when a GPU is used.
- **`mdps`** -- mega disparity-hypotheses per second,
  `H * W * ndisp / wall / 1e6`. Stereo cost scales with the search range, so
  this normalises away both image size and how wide a range the scene needed.

Timing is the **median** of `--repeats` runs after `--warmup` untimed ones
(keep warmup at 1 or more: ONNX session construction and OpenCV's thread pool
both initialise lazily, and would otherwise land inside the first measurement).

### Outputs

```
results/<timestamp>/
  summary.md                       leaderboard, per-scene table, resource table, failures
  results.csv                      one row per (scene, algorithm), every metric
  run.json                         config + full machine/library versions
  figures/
    accuracy_vs_time.png           the trade-off, with the Pareto frontier marked
    accuracy_by_scene.png          heatmap -- who is merely average vs who collapses
    qualitative_<scene>.png        disparity maps on a shared colour scale
```

---

## Tuning

Every algorithm ships one configuration, so the honest question about any
result is whether a method lost because it is worse or because its defaults
were worse. `python -m bench sweep` answers that:

```bash
python -m bench sweep
```

765 configurations across 21 algorithms, ~20 minutes. Output in
`results/sweeps/<timestamp>/`: `sweep_summary.md` (pre-sweep vs tuned, the
recommended diff, top configs), `sweep.csv`, and a parameter-response figure
per algorithm.

Two rules keep it from being self-congratulatory.

**The sweep only sees half the scenes.** `bench/splits.py` splits the 15
scenes into 7 `tune` and 8 `heldout`, **by scene rather than by file**:
Middlebury ships `Motorcycle`/`MotorcycleE` (exposure), `Piano`/`PianoL`
(lighting) and `Playtable`/`PlaytableP` (rectification) as separate files of
the *same geometry*, so splitting those apart would leak. Both sides keep a
photometrically awkward pair so the held-out set can still test the
cost-function finding.

**Accuracy first, speed only when it is real.** The rule takes the lowest
`bad2_all`, keeps everything within 0.5 points of it, keeps those within 5% of
the fastest, and takes the most accurate of *those*. The 5% band matters: the
first version sorted near-ties by raw wall time and picked the worse of two
block sizes whose runtimes differed by 3 ms of jitter.

### Did it generalise?

| | tune (7 scenes, seen) | held-out (8 scenes, unseen) |
|---|---|---|
| mean gain | **+1.42 pp** | **+1.45 pp** |

The held-out gain matches the tune gain, so the grid found real settings
rather than scene-specific ones. 21 of 21 tuned algorithms improved on
held-out data; the two untuned ones are the ONNX exports, which have no
parameters to move.

Biggest held-out gains: `scratch_sad_wta_lrcheck` +4.33, `scratch_sad_wta`
+3.54, `opencv_sgbm_wls` +2.80, `opencv_quasi_dense` +2.65,
`opencv_bm` +2.59, `scratch_census_wta` +2.57.

One clean pattern came out of it. **Winner-takes-all methods wanted much
larger windows than the conventional 9x9** -- 13 to 15 px, worth 3-4 points --
while the **aggregating methods wanted small ones** (3 to 5 px). Without an
aggregation step the only thing suppressing an ambiguous match is window
support, so more is better; once SGM supplies that support globally, a large
window only blurs depth edges. The SGBM sweep also moved OpenCV's documented
`P1`/`P2` recommendation from 8/32 to 4/64: a wider ratio holds a surface
together through texture while still paying heavily for a true depth jump.

The ranking barely moved, which is the reassuring outcome -- the conclusions
below were not an artifact of untuned baselines.

---

## Results

All 15 scenes x 23 algorithms = 345 cells, no failures, on an i9-13900HX
(24 physical cores, no CUDA), with tuned defaults. Full output in
`results/20261105-023941/`. Reproduce with `python -m bench run`.

| rank | algorithm | bad2_all | wall_s | cpu_s | cores busy |
|---|---|---|---|---|---|
| 1 | `learned_raft_stereo_r4` | 10.51% | 1.90 | 45.6 | 24.1 |
| 2 | `scratch_census_sgm8` | 12.12% | 3.13 | 3.1 | 1.0 |
| 3 | `torch_census_sgm4` | 12.29% | 4.38 | 103.5 | 23.7 |
| 4 | `scratch_census_sgm4` | 12.32% | 1.83 | 1.9 | 1.0 |
| 6 | `scratch_census_dp` | 16.34% | 1.05 | 1.1 | 1.1 |
| 8 | `opencv_sgbm_hh4` | 23.31% | 0.04 | 0.1 | 2.8 |
| 9 | `opencv_sgbm_3way` | 23.50% | 0.02 | 0.0 | 2.4 |

Pareto frontier: `opencv_bm` -> `opencv_sgbm_3way` -> `opencv_sgbm_hh4` ->
`scratch_census_wta` -> `scratch_census_dp` -> `scratch_census_sgm4` ->
`learned_raft_stereo_r4`.

Four things this says:

**The matching cost matters more than the aggregation.** Same
winner-takes-all pipeline, each at its own tuned window, only the cost
function changed: census 17.6%, ZNCC 23.9%, SSD 35.5%, SAD 36.4%. Swapping SAD
for census buys more than adding 8-path semi-global aggregation to SAD does
(`scratch_sad_sgm8`, 29.6%).

Most of that gap is three scenes. `ArtL`, `MotorcycleE` and `PianoL` are
deliberately mis-exposed between the two views, and an intensity-difference
cost has no defence:

| scene | census | ZNCC | SAD |
|---|---|---|---|
| MotorcycleE | **5.9%** | 12.3% | 70.9% |
| ArtL | **13.5%** | 28.1% | 54.7% |
| PianoL | **32.3%** | 40.9% | 73.2% |
| Motorcycle (normal exposure) | **6.4%** | 13.2% | 17.3% |

Census keeps only the *ordering* of neighbouring pixels, so a gain-and-offset
change leaves it untouched. This is the clearest result in the run, and it is
visible precisely because the dataset includes scenes built to expose it.

**A from-scratch SGM lands within 2 points of a pretrained network.**
`scratch_census_sgm8` at 12.12% against RAFT-Stereo's 10.51% -- roughly 300
lines of NumPy against a trained model, though the ONNX export is being
applied out of domain and at a resampled resolution, so this is not
RAFT-Stereo at its best.

**Wall time hides what compute is being spent.** `torch_census_sgm4` and
`scratch_census_sgm4` compute the same thing to within 0.03 points, from
identical parameters -- but torch burns **103 CPU-seconds to NumPy's 1.9**,
keeping 24 cores busy to end up *slower* in wall time (4.38s vs 1.83s). The
tensor formulation is written for a GPU this machine does not have. Wall time
alone would have called these close; CPU-seconds show one of them costing 55x
the compute.

**And the classical library is still two orders of magnitude faster.**
`opencv_sgbm_3way` is 23.5% bad2 in 0.017s. Getting to 10.5% costs ~110x the
wall time and ~1000x the CPU time. Which end of that trade is right depends
entirely on the application.

Hardest scenes, averaged over all 23 algorithms: `PianoL` 46.6%, `Shelves`
37.3%, `Jadeplant` 35.2%. Easiest: `Teddy` 14.5%, `Motorcycle` 14.7%.

### A caveat on the timings

The pre-tuning run (`results/20261105-013249/`) and the tuned one are **not
comparable on time**, only on accuracy. Nearly every algorithm appears ~2.5x
faster in the later run, including the two ONNX models whose parameters never
changed (2.10x and 2.70x, bracketing the 2.47x median across all 23). Running
both parameter sets back to back confirms it: `scratch_census_sgm8` takes
1.655s with the old values and 1.636s with the new ones. The difference is
machine state across a 27-minute sustained load on a laptop, not the sweep.

Both runs used `--repeats 1`. For any timing claim, use `--repeats 5` or more
and keep the machine otherwise idle; accuracy is deterministic and unaffected.

---

## Project layout

```
bench/
  types.py              StereoPair, DisparityResult -- the two shared contracts
  metrics.py            bad N, EPE, RMSE, density, hole filling
  profiling.py          timing, CPU time, peak RSS, GPU memory, machine info
  runner.py             runs the grid; one failure never aborts the sweep
  report.py             summary.md and the figures
  splits.py             tune / held-out scene split, grouped by scene
  sweep.py              parameter grids, the selection rule, the sweep runner
  sweep_report.py       sweep tables and parameter-response figures
  cli.py                python -m bench
  datasets/
    middlebury.py       MiddEval3 loader + PFM reader/writer
    synthetic.py        generated pairs with exact ground truth
    local.py            any folder of rectified images
  algorithms/
    base.py             the registry and the output contract
    opencv_classical.py
    scratch.py          NumPy/SciPy, every stage exposed
    torch_impl.py       the same maths as tensor ops
    learned.py          ONNX networks
scripts/
  get_dataset.py        download Middlebury MiddEval3 (quarter resolution)
  get_models.py         download the ONNX weights
tests/                  193 tests
practice/               the original hand-written notebook and its image pair
```

### Adding an algorithm

Write something with a `compute(pair) -> DisparityResult` method and register it:

```python
from .base import register

@register("my_matcher", family="scratch", description="what it does",
          tags=("local", "wta"), check=lambda: (True, ""))
def _build():
    return MyMatcher(block=7)
```

It is then selectable by name, family, tag or glob, and the parametrised
contract tests in `tests/test_algorithms.py` pick it up automatically -- shape,
dtype, nan-for-invalid, determinism, and a sanity floor against a
constant-disparity baseline.

To make it tunable, add a grid to `GRIDS` in `bench/sweep.py` naming any of
its dataclass fields. Tests check that every grid key is a real parameter and
that each grid brackets and contains the shipped default -- a typo would
otherwise produce a flat sweep, and a default outside the grid leaves nothing
to compare the winner against. Both mistakes happened while building this.

The contract: return **float32 disparity in left-image pixels**, with `nan`
wherever the method declined to estimate. That last part is load-bearing.
OpenCV returns fixed-point `int16` scaled by 16 with `-16` meaning "no match",
and a naive `/16` turns every unmatched pixel into a confident `-1`, then into
an apparently-correct `0` after clipping. `base.from_opencv_fixed_point` does
it properly.

---

## Notes and caveats

- **No GPU on the development machine.** `torch` is a CPU-only build here, so
  the `torch_*` rows show CPU numbers. The code is device-agnostic and reports
  which device ran; `run.json` records it.
- **The ONNX exports are out of domain.** They are KITTI/SceneFlow-trained with
  a frozen input resolution, so the pair is resized in and the disparity
  resized out and rescaled by the width ratio (disparity is a horizontal
  length). Treat them as "a pretrained network applied cold to indoor scenes",
  not as RAFT-Stereo's published accuracy.
- **Input normalisation was measured, not assumed.** These exports fold
  RAFT-Stereo's normalisation into the graph, so they want raw 0-255 input;
  feeding them pre-normalised input quietly triples the error rather than
  failing. `tests/test_learned.py` pins the choice against ground truth.
- **Aggregation penalties are scale-free.** SAD over colour runs 0-765, ZNCC
  0-2, census Hamming 0-63. Cost volumes are rescaled to a common 0-64 range
  before aggregation so the literature's `P1=10 / P2=120` applies to all of
  them. Skipping this was a real bug during development: penalties ~100x the
  cost range collapsed SGM to a near-constant disparity map that still *looked*
  plausible.
- **Defaults are per-algorithm, not per scene.** Every method runs at one
  fixed configuration across all scenes, which is the fair comparison but not
  the best each could do. Those configurations come from `python -m bench
  sweep` on the tune split; see the Tuning section and the comments on each
  adapter's dataclass defaults.
- **The two torch/NumPy pairs must stay in step.** `torch_*` entries exist to
  isolate the backend, which only means anything if both run the same
  configuration, so their tuned values are copied from the `scratch_*` side
  rather than searched independently (`test_torch_defaults_mirror_scratch`
  enforces it). Their sweep grids are confirmatory only.
- **Timings need `--repeats`.** The headline runs used `--repeats 1`, which is
  fine for accuracy but not for time: see the caveat under Results, where a
  whole run drifted ~2.5x on machine state alone.
- **Cost-volume memory** is `D x H x W` float32 -- about 210 MB for the widest
  scene at `ndisp=160`. `--max-volume-gb` skips a cell rather than letting the
  machine swap, and the skip is recorded in the summary.

---

## References

- D. Scharstein, H. Hirschmuller, Y. Kitajima, G. Krathwohl, N. Nesic, X. Wang,
  P. Westling. *High-resolution stereo datasets with subpixel-accurate ground
  truth.* GCPR 2014. [Middlebury stereo data](https://vision.middlebury.edu/stereo/data/scenes2014/)
- H. Hirschmuller. *Stereo processing by semiglobal matching and mutual
  information.* IEEE TPAMI 30(2), 2008.
- R. Zabih, J. Woodfill. *Non-parametric local transforms for computing visual
  correspondence.* ECCV 1994. (the census transform)
- L. Lipson, Z. Teed, J. Deng. *RAFT-Stereo: Multilevel recurrent field
  transforms for stereo matching.* 3DV 2021.
  [arXiv:2109.07547](https://arxiv.org/abs/2109.07547)
- J. Kowalczuk, E. T. Psota, L. C. Perez. *Real-time stereo matching on CUDA
  using an iterative refinement method for adaptive support-weight
  correspondences.* IEEE TCSVT 23(1), 2013. (OpenCV's QuasiDenseStereo)
- D. Scharstein, R. Szeliski. *A taxonomy and evaluation of dense two-frame
  stereo correspondence algorithms.* IJCV 47, 2002. (where `bad N` comes from)
