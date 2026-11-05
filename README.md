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

```bash
python -m bench explain --open
```

That writes `report.html` next to the latest run: one self-contained page that
explains what the numbers mean rather than only listing them -- the controlled
comparisons behind the leaderboard, a glossary of every metric, and the
caveats. Add `--pdf` for a printable copy.

Nothing above is strictly required to get started -- with no dataset and no
weights, `python -m bench run -d synthetic:3` still works, because the
synthetic generator produces pairs with exact ground truth in-process.

Need more data than Middlebury's 15 scenes? Generate it:

```bash
python -m bench synth preview -e room --difficulty medium --open
```

```bash
python -m bench synth generate -n 500 --out data/synth/set1 --environments room corridor tabletop
```

Ray-traced 3-D scenes with exact ground truth and true occlusion, split into
`train`/`val`/`test`, with no download and no extra dependency. See
[Generating training images](#generating-training-images).

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
| `synthetic[:N]` | warp-generated pairs, **exact** ground truth, no download |
| `synth:<env>[:N[:difficulty]]` | ray-traced 3-D scenes: exact ground truth **and** true occlusion, no download |
| `notebooks/practice` (any folder) | a folder with `im0`/`im1`; add `--downscale 6` for the 2964x2000 pair already in the repo |
| a generated dataset folder | every sample of a set written by `bench synth generate` |

Each Middlebury scene brings its own disparity range from `calib.txt`, and the
harness uses it: Jadeplant needs 160 disparities, Teddy needs 64. Giving every
scene the same oversized range would make the easy scenes look needlessly slow.

### Generating training images

Middlebury has 15 scenes. That is enough to benchmark on and nowhere near
enough to train on, and there is no more of it. `bench/synth/` closes that gap:
it ray traces 3-D scenes into rectified stereo pairs whose disparity is exact
by construction, as many as you want, on the CPU.

```bash
python -m bench synth environments                      # what can be generated
python -m bench synth preview -e corridor --difficulty hard
python -m bench synth generate -n 500 --out data/synth/set1 --environments room corridor tabletop
```

or in the app's **Synthesize** tab: edit the rules, render examples, then
generate. Generated sets appear in the scene picker immediately, so you can
benchmark them without an export step.

**Splits.** Output is divided into `train/`, `val/` and `test/` by default —
override with `--splits train=0.9,val=0.1`, or `--splits ''` for one flat
directory. Counts are exact, and every pair has its own seed and its own
scene, so the splits share no geometry. That matters more here than usual:
every pair comes from the same generator, so evaluating on pairs you trained
on would give excellent numbers and no warning that anything was wrong. Any
split loads on its own:

```bash
python -m bench run -d data/synth/set1/test -a opencv_sgbm
```

**Why a ray tracer and not Blender or Open3D.** Every off-the-shelf option
costs more than it gives here. `bpy` is a ~1 GB wheel pinned to one exact
CPython version that can only be imported once per process, which fights the
multiprocess fan-out mass generation needs. Open3D and pyrender render through
an offscreen OpenGL context, which on headless Windows has a long history of
failing *silently* — returning a plausible buffer of zeros. That is the worst
possible failure mode for a ground-truth generator, because the dataset looks
fine until you train on it. And all of them hand back depth from a z-buffer:
quantised, interpolated across triangles, clipped by the near and far planes.

Solving the ray–surface equation instead gives the exact distance to an
analytic surface, in pure NumPy, with **no new dependency**, no GPU, no
display, and identical output on every platform. For tens of primitives rather
than millions of triangles it is also fast enough — roughly 0.1–0.5 s per pair
at 320x224, across all cores.

**What makes the ground truth trustworthy.** Both views are traced
independently from the same geometry, so:

- disparity is `focal_px * baseline / Z` from an exact `Z`, not a z-buffer read;
- occlusion is **physical**. The `mask0nocc` region comes from comparing the
  two ray-traced depth maps, so it marks the surfaces the right camera really
  cannot see. The older `synthetic` generator warps one image into the other,
  which cannot model occlusion at all and masks a guessed band instead;
- ground truth is `nan` where it is genuinely unknown — sky, pixels straddling
  a depth discontinuity where no single true disparity exists, and anything
  past `ndisp`;
- the baseline is **fitted per scene** so disparities fill the search range. A
  fixed rig gives 8 px of disparity in a corridor and 400 on a tabletop, so
  half a generated set would otherwise waste the range or overflow it.

**Seven environments**, each a different *shape* of depth distribution, which
is the axis real benchmarks actually differ along: `tabletop`, `room`,
`corridor`, `warehouse` (deliberately repetitive, to punish local matchers),
`street` and `terrain` (open sky, so some pixels legitimately have no ground
truth), and `void`.

**The rules** control framing, clutter, texture, lighting and sensor
behaviour, and are saved next to the images, so a dataset can be audited and
regenerated bit for bit. The knobs that matter most are the texture ones —
`texture_richness` and `textureless_fraction` move a real matcher's error far
more than geometry does. Sensor noise, exposure mismatch and defocus are drawn
*independently per view*, because identical noise in both eyes cancels out of
every matching cost and is what makes naive synthetic data fail to transfer.

Previewing reports measured statistics rather than assumed ones, and warns
about the rule sets that render beautifully and supervise nothing — a search
range the scene never fills, ground truth covering a third of the frame.

Sanity check, with `opencv_sgbm` on 320x224 generated pairs:

| difficulty | bad2 (all) | EPE (answered) |
|---|---|---|
| easy | 11% | 0.31 px |
| hard | 42% | 2.27 px |

Sub-pixel end-point error on the easy set is only achievable if the geometry
is right; an off-by-one or a sign error would put it in the tens of pixels.

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
  report.html                      written by `python -m bench explain`
  report.pdf                       the same, with `--pdf`
```

### The explained report

`summary.md` answers *what* the numbers were. `python -m bench explain
<run>` answers *what they mean*, as one self-contained HTML page -- no CDN, no
sibling assets, so it can be mailed, committed or opened on a machine with no
Python on it. The print stylesheet turns the browser's "Save as PDF" into a
document rather than a screenshot of a web app, and `--pdf` does that for you
through a headless Chrome or Edge.

What it adds over the summary:

- **Controlled comparisons.** A leaderboard says `scratch_census_sgm4` beat
  `scratch_sad_wta`; it cannot say whether that was the cost function or the
  aggregation, because both changed. The report groups algorithms by their
  recipe and reports every comparison where exactly one component differs.
- **Derived prose.** Each statement is assembled from this run's measurements
  with the supporting numbers in it, so re-running the benchmark changes the
  sentences and not just the tables. Nothing is written in advance.
- **A metric glossary** attached to the columns it describes, so the two
  framings of every error figure -- counting blanks as errors, and not -- are
  explained where they are used.
- **Interaction**: sortable leaderboard, a scatter whose Pareto frontier is
  recomputed for whichever metric you put on the y axis, and a per-scene
  heatmap you can switch between metrics.
- **The caveats**, stated in the document rather than left to the reader.

```bash
python -m bench explain results/<timestamp> --pdf --open
```

With no argument it explains the most recent run. It is also reachable from
the app: *Runs* -> pick a run -> **open report**. A pre-built copy for the
headline run is committed at
[`results/keep/20261105-035941/report.html`](results/keep/20261105-035941/report.html).

---

## Tuning

Every algorithm ships one configuration, so the honest question about any
result is whether a method lost because it is worse or because its defaults
were worse. `python -m bench sweep` answers that:

```bash
python -m bench sweep
```

```bash
python -m bench sweep --grid refined
```

Three passes were run, all on the tune split only:

| pass | grid set | configs | what it was for |
|---|---|---|---|
| 1 | `default` | 765 | coarse search around the shipped defaults |
| 2 | `refined` | 1068 | extend past round one's truncated ranges, bisect elsewhere |
| 3 | `wta-extended` | 20 | push the block-matching window until it turned over |

Output lands in `results/sweeps/<timestamp>/`: `sweep_summary.md` (pre-sweep
vs tuned, the recommended diff, truncated searches, top configs),
`sweep.csv`, and a parameter-response figure per algorithm.

### What keeps it honest

**The sweep only sees half the scenes.** `bench/splits.py` splits the 15
scenes into 7 `tune` and 8 `heldout`, **by scene rather than by file**:
Middlebury ships `Motorcycle`/`MotorcycleE` (exposure), `Piano`/`PianoL`
(lighting) and `Playtable`/`PlaytableP` (rectification) as separate files of
the *same geometry*, so splitting those apart would leak. Both sides keep a
photometrically awkward pair so the held-out set can still test the
cost-function finding.

**Held-out is reported, never acted on.** Selection happens on `tune` in all
three passes. Reverting a change because held-out disliked it would be
selection on held-out, and would forfeit exactly the property that makes it
worth measuring. So the regressions below are reported and kept, not quietly
undone.

**Accuracy first, speed only when it is real.** The rule takes the lowest
`bad2_all`, then moves to a configuration within 0.5 points of it only if that
one is at least 1.3x faster. An earlier version worked the other way round --
near-ties first, then everything within 5% of the *fastest*, then the most
accurate of those -- which let one quick measurement define the tier and hand
back 0.4 points for a 1.11x "speedup" inside measurement noise.

That fix has a documented limit, which is worth stating because it bit:
`torch_zncc_wta` measured a 17 px box filter 1.39x *faster* than a 13 px one,
which cannot be true, and both the old and the new rule take that trade. A
selection rule cannot repair a wrong measurement. What protects the result is
distrusting it -- the `torch_*` entries copy their parameters from the
matching `scratch_*` entry rather than using their own sweep rows, enforced by
`test_torch_defaults_mirror_scratch`.

**Truncated searches are flagged, not reported as results.**
`boundary_hits()` lists every parameter whose winner landed on the edge of its
range, which means the grid ran out before the response did. Round one had 33
of them -- that diagnostic is what round two was built from, and it is why the
block-matching window got a third pass of its own.

### Did it generalise?

| | tune (7 scenes, seen) | held-out (8 scenes, unseen) | ratio |
|---|---|---|---|
| round 1 over untuned | +1.42 pp | **+1.45 pp** | 1.02 |
| round 2 over round 1 | +1.06 pp | **+0.26 pp** | 0.25 |
| cumulative | +2.52 pp | **+1.83 pp** | 0.73 |

**This is the most useful result in the whole exercise.** The first pass
generalised completely -- the held-out gain even edged past the tune gain, and
21 of 21 tunable algorithms improved on scenes the sweep had never seen. The
second pass did not: it found another 1.06 points on the tune scenes and kept
only a quarter of them, with 10 of 19 algorithms actually *regressing* on
held-out.

Nothing changed between the passes except how finely the same 7 scenes were
searched. That is diminishing returns turning into overfitting, visible only
because the held-out half was never touched. A sweep reporting tune numbers
alone would have called round two a 1.06-point win.

Where round two did hold up: `scratch_sad_wta_lrcheck` +3.42 held-out,
`opencv_sgbm_hh` +1.50, `opencv_sgbm_wls` +1.19, `opencv_bm` +1.69. Where it
did not: every entry whose matching window grew (`scratch_ssd_wta` -2.16,
`scratch_census_wta` -0.73, `scratch_sad_wta` -0.66).

The shipped defaults are round two's, per the protocol above. Cumulatively
they are still clearly better than untuned (+1.83 held-out, 19 of 21
improving) -- just not by as much as the tune column claims.

### Two things the sweep revealed about the metric

**The window wants to be enormous, and `bad2_all` is why.** Round one moved
the winner-takes-all window from 9 px to 13-15, round two to 25, and the
extension pass found it finally turning over at 21-29 px -- on a 450 px-wide
image. Splitting the evaluation region explains it exactly. For
`scratch_sad_wta` over four tune scenes, going from a 9 px to a 29 px window:

| region | share of pixels | change in bad 2.0 |
|---|---|---|
| near a depth discontinuity (±9 px) | 30% | **7.97 pp worse** |
| away from one | 70% | **6.27 pp better** |
| overall | 100% | 2.01 pp better |

0.70 x 6.27 - 0.30 x 7.97 = +2.0, the whole of the overall gain. The window
buys flat regions and pays for it at every depth edge, and since the metric
counts all pixels equally it books that as an improvement. The held-out
regressions above suggest the trade does not even transfer between scenes.

**The sweep switched the quality filters off.** Every SGBM mode and
`scratch_sad_wta_lrcheck` chose `uniqueness_ratio = 0`, and StereoBM also went
to `texture_threshold = 0` and QuasiDenseStereo to its loosest correlation
threshold. Those parameters exist to let a matcher *abstain* when the match is
ambiguous -- and under `bad2_all` an unanswered pixel scores exactly as badly
as a wrong one, so abstaining is never worth anything. The metric rewards
guessing. Read `bad2_valid` and `density` alongside it before copying any of
these defaults into an application where a confident wrong depth is worse than
no depth.

---

## Results

All 15 scenes x 23 algorithms = 345 cells, no failures, on an i9-13900HX
(24 physical cores, no CUDA), with round-2 defaults. Full output in
[`results/keep/20261105-035941/`](results/keep/20261105-035941/) — the one run
committed to the repository. Reproduce with `python -m bench run`.

| rank | algorithm | bad2_all | wall_s | cpu_s | cores busy |
|---|---|---|---|---|---|
| 1 | `learned_raft_stereo_r4` | 10.51% | 4.43 | 62.0 | 14.4 |
| 2 | `torch_census_sgm4` | 12.34% | 6.26 | 71.1 | 11.9 |
| 3 | `scratch_census_sgm4` | 12.37% | 4.80 | 4.9 | 1.0 |
| 4 | `scratch_census_sgm8` | 12.42% | 10.01 | 9.6 | 1.0 |
| 6 | `scratch_census_dp` | 15.63% | 2.57 | 2.8 | 1.1 |
| 7 | `scratch_census_wta` | 17.34% | 1.45 | 1.4 | 1.0 |
| 8 | `opencv_sgbm_wls` | 22.66% | 0.10 | 0.4 | 3.6 |
| 11 | `opencv_sgbm_3way` | 22.98% | 0.04 | 0.1 | 3.8 |

Pareto frontier: `opencv_bm` -> `opencv_sgbm_3way` -> `opencv_sgbm_hh4` ->
`opencv_sgbm_wls` -> `scratch_census_wta` -> `scratch_census_dp` ->
`learned_raft_stereo_r4`.

Four things this says:

**The matching cost matters more than the aggregation.** Same
winner-takes-all pipeline, each at its own tuned window, only the cost
function changed: census 17.3%, ZNCC 23.9%, SAD 35.8%, SSD 36.0%. Swapping SAD
for census buys more than adding 8-path semi-global aggregation to SAD does
(`scratch_sad_sgm8`, 29.6%).

Most of that gap is three scenes. `ArtL`, `MotorcycleE` and `PianoL` are
deliberately mis-exposed between the two views, and an intensity-difference
cost has no defence:

| scene | census | ZNCC | SAD |
|---|---|---|---|
| MotorcycleE | **6.5%** | 12.3% | 60.9% |
| ArtL | **15.6%** | 28.1% | 47.7% |
| PianoL | **27.7%** | 40.9% | 68.1% |
| Motorcycle (normal exposure) | **6.9%** | 13.2% | 17.0% |

Census keeps only the *ordering* of neighbouring pixels, so a gain-and-offset
change leaves it untouched. This is the clearest result in the run, and it is
visible precisely because the dataset includes scenes built to expose it.

**A from-scratch SGM lands within 2 points of a pretrained network.**
`scratch_census_sgm4` at 12.37% against RAFT-Stereo's 10.51% -- roughly 300
lines of NumPy against a trained model, though the ONNX export is being
applied out of domain and at a resampled resolution, so this is not
RAFT-Stereo at its best.

**Wall time hides what compute is being spent.** `torch_census_sgm4` and
`scratch_census_sgm4` compute the same thing to within 0.03 points, from
mirrored parameters -- but torch burns **71 CPU-seconds to NumPy's 4.9**,
keeping 12 cores busy to end up *slower* in wall time (6.26s vs 4.80s). The
tensor formulation is written for a GPU this machine does not have. Wall time
alone would have called these close; CPU-seconds show one of them costing 14x
the compute.

**And the classical library is still two orders of magnitude faster.**
`opencv_sgbm_3way` is 23.0% bad2 in 0.039s. Getting to 10.5% costs ~110x the
wall time and ~440x the CPU time. Which end of that trade is right depends
entirely on the application.

Hardest scenes, averaged over all 23 algorithms: `PianoL` 44.8%, `Jadeplant`
36.4%, `Shelves` 35.9%. Easiest: `Motorcycle` 14.5%, `Teddy` 14.9%.

### A caveat on the timings

The three full runs are comparable on accuracy but **not on time**. The two
ONNX models have never had a parameter changed, and their wall time still
moved like this:

| | untuned run | round-1 run | round-2 run |
|---|---|---|---|
| `learned_raft_stereo_r4` | 3.99s | 1.90s | 4.43s |
| `learned_raft_stereo_r1` | 0.91s | 0.34s | 0.93s |

A 2.3x spread on identical work. Running two parameter sets back to back
confirms the same thing from the other direction:
`scratch_census_sgm8` took 1.655s with its old values and 1.636s with its new
ones in the same session. The differences between runs are machine state over
a ~25-minute sustained load on a laptop, not the sweep.

All three runs used `--repeats 1`. For any timing claim use `--repeats 5` or
more on an otherwise idle machine; accuracy is deterministic and unaffected.

---

## Project layout

```
bench/          ✓ git   the harness -- pure Python, no web, no UI
app/            ✓ git   the explorer -- a local web UI onto bench/
scripts/        ✓ git   get_dataset.py · get_models.py · keep_result.py
tests/          ✓ git   282 tests, ~65 s
docs/           ✓ git   architecture.md · repository.md · figures/ · presentation/
notebooks/      ✓ git   using bench from a notebook, and the original matcher
data/           ✗ git   inputs: downloaded + uploaded image pairs
models/         ✗ git   inputs: ONNX weights
results/        ✗ git   outputs -- except results/keep/, which is committed
```

There is one benchmark harness and everything else is a front end onto it:
`app/` imports `bench` and calls the same functions the CLI does, so a run
triggered from the browser lands in `results/<timestamp>/` with the same
`summary.md`. `bench/` has no dependency on `app/`.

Three documents cover the rest:

| | |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | **what runs where** -- the four processes, the module map, how one cell flows from loader to `results.csv`, with diagrams |
| [`docs/repository.md`](docs/repository.md) | **what is committed** -- the full tree, the ignore rules and why, and how to commit a result that matters |
| [`app/README.md`](app/README.md) | the explorer's four tabs, and what to try in them |

Within the harness:

```
bench/
  paths.py              every directory constant, in one place
  types.py              StereoPair, DisparityResult -- the two shared contracts
  metrics.py            bad N, EPE, RMSE, density, hole filling
  profiling.py          timing, CPU time, peak RSS, GPU memory, machine info
  runner.py             runs the grid; one failure never aborts the sweep
  report.py             summary.md and the figures
  explain.py            derives the findings a report states -- pure, testable
  explain_html.py       renders them as one self-contained page (and a PDF)
  splits.py             tune / held-out scene split, grouped by scene
  sweep.py              parameter grids, the selection rule, the sweep runner
  sweep_report.py       sweep tables and parameter-response figures
  cli.py                python -m bench
  datasets/
    middlebury.py       MiddEval3 loader + PFM reader/writer
    synthetic.py        warp-generated pairs with exact ground truth
    local.py            any folder of rectified images
  synth/                the training-image synthesizer (pure NumPy, no new deps)
    rules.py            SceneRules: every knob, validated, JSON round-trips
    environments.py     seven scene samplers, one per depth distribution
    shapes.py           analytic primitives with vectorised ray intersection
    materials.py        hashed value noise, solid textures, materials
    render.py           the ray tracer and the rectified stereo rig
    pipeline.py         sensor simulation, StereoPair, parallel dataset writing
    preview.py          contact sheets and the "is this data usable?" report
    cli.py              python -m bench synth
  algorithms/
    base.py             the registry and the output contract
    opencv_classical.py
    scratch.py          NumPy/SciPy, every stage exposed
    torch_impl.py       the same maths as tensor ops
    learned.py          ONNX networks
```

### Keeping a result

`results/` is ignored -- a run is reproducible and a full one is tens of
megabytes. When one does matter, `results/keep/` is un-ignored, so a plain
`git add` works there and nowhere else under `results/`:

```bash
python scripts/keep_result.py 20261105-035941
```

That copies the run's small files (`summary.md`, `results.csv`, `run.json`,
the report) and prints the `git add` line. `--figures` brings the PNGs too.
Record *why* in [`results/keep/README.md`](results/keep/README.md).

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
- **The defaults are tuned for `bad2_all`, and it shows.** The sweep turned
  every uniqueness and texture test off and grew the matching window far past
  what a depth map should use, because an unanswered pixel and a wrong one
  score identically and edges are only ~30% of the image. They are the right
  defaults for this benchmark and the wrong ones for an application that would
  rather have a hole than a confident mistake -- retune against `bad2_valid`
  and `density` for that.
- **Round two of the sweep overfit.** The second, finer pass over the same 7
  tune scenes found +1.06 points there and kept +0.26 on held-out, with 10 of
  19 algorithms regressing. Held-out is reported, never used to revert a
  change, so those regressions are still in the shipped defaults; the
  cumulative held-out gain over untuned is +1.83. Treat a third pass over the
  same split as very unlikely to be worth running.
- **The two torch/NumPy pairs must stay in step.** `torch_*` entries exist to
  isolate the backend, which only means anything if both run the same
  configuration, so their tuned values are copied from the `scratch_*` side
  rather than searched independently (`test_torch_defaults_mirror_scratch`
  enforces it). Their sweep grids are confirmatory only.
- **Timings need `--repeats`.** All three full runs used `--repeats 1`, which
  is fine for accuracy but not for time: see the caveat under Results, where
  the two untuned ONNX models drifted 2.3x across runs on machine state alone.
  The same noise reached the sweep's selection rule once, which is why it now
  demands a 1.3x speedup before trading any accuracy away.
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
