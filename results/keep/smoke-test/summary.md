# Stereo benchmark results

- **when** 2026-11-05T20:18:47.862212+00:00
- **machine** 13th Gen Intel(R) Core(TM) i9-13900HX / 24 physical cores / 34.03 GB RAM
- **gpu** none detected (CUDA unavailable)
- **libraries** opencv 4.13.0, numpy 2.3.0, torch 2.11.0+cpu, onnxruntime 1.31.1
- **timing** median of 1 run(s) after 1 warmup; threads = library default
- **evaluated on** the `nonocc` region

`bad2_all` is the share of evaluated pixels off by more than 2 px, counting pixels the method left blank as errors -- the one column that is fair to both dense and sparse methods. `epe_valid` and `density` describe where a method did answer. See `bench/metrics.py`.

## Leaderboard (mean over scenes)

| # | algorithm        | family | bad2_all | bad1_all | EPE  | density | wall_s | cpu_s | par  | RSS+MB | pareto |
|---|------------------|--------|----------|----------|------|---------|--------|-------|------|--------|--------|
| 1 | opencv_sgbm_3way | opencv | 3.03%    | 3.03%    | 0.13 | 97.0%   | 0.001  | 0.000 | 0.0x | 0      | yes    |

### Pareto frontier (accuracy vs wall time)

Nothing in the table is both faster and more accurate than these:

- `opencv_sgbm_3way`

## `bad 2.0` by scene (%)

| algorithm        | seed0 |
|------------------|-------|
| opencv_sgbm_3way | 3.0   |

## Compute cost

`cores busy` is `cpu_s / wall_s`: how many cores the method kept busy. A method near 1.0 is single-threaded, so its wall time understates how much cheaper it is than a method of the same speed running on 16 threads. `Mdisp/s` normalises throughput by both image size and disparity range.

**On the memory columns.** `RSS growth` is the rise in resident memory while the cell ran, and is the figure attributable to the algorithm. `process peak` is the whole interpreter's peak at that moment, so it carries everything earlier cells allocated and should not be read as this method's footprint. Both are under-estimates for a method that happens to reuse memory an allocator already holds, so compare within a family before comparing across one, and re-run a single algorithm on its own if you need a clean number.

| algorithm        | wall_s | cpu_s | cores busy | Mpix/s | Mdisp/s | RSS growth MB | process peak MB |
|------------------|--------|-------|------------|--------|---------|---------------|-----------------|
| opencv_sgbm_3way | 0.001  | 0.000 | 0.0        | 31.32  | 1002.3  | 0             | 519             |
