# Stereo benchmark results

- **when** 2026-11-05T04:26:54.569262+00:00
- **machine** 13th Gen Intel(R) Core(TM) i9-13900HX / 24 physical cores / 34.03 GB RAM
- **gpu** none detected (CUDA unavailable)
- **libraries** opencv 4.13.0, numpy 2.3.0, torch 2.11.0+cpu, onnxruntime 1.31.1
- **timing** median of 1 run(s) after 1 warmup; threads = library default
- **evaluated on** the `nonocc` region

`bad2_all` is the share of evaluated pixels off by more than 2 px, counting pixels the method left blank as errors -- the one column that is fair to both dense and sparse methods. `epe_valid` and `density` describe where a method did answer. See `bench/metrics.py`.

## Leaderboard (mean over scenes)

| #  | algorithm                   | family  | bad2_all | bad1_all | EPE   | density | wall_s | cpu_s  | par   | RSS+MB | pareto |
|----|-----------------------------|---------|----------|----------|-------|---------|--------|--------|-------|--------|--------|
| 1  | learned_raft_stereo_r4      | learned | 10.51%   | 17.95%   | 1.14  | 100.0%  | 4.430  | 62.024 | 14.4x | 236    | yes    |
| 2  | torch_census_sgm4           | torch   | 12.34%   | 17.28%   | 1.91  | 100.0%  | 6.264  | 71.105 | 11.9x | 1257   |        |
| 3  | scratch_census_sgm4         | scratch | 12.37%   | 17.32%   | 1.91  | 100.0%  | 4.803  | 4.923  | 1.0x  | 1872   |        |
| 4  | scratch_census_sgm8         | scratch | 12.42%   | 17.31%   | 1.96  | 100.0%  | 10.005 | 9.588  | 1.0x  | 1858   |        |
| 5  | scratch_census_sgm8_lrcheck | scratch | 15.25%   | 19.31%   | 1.28  | 92.5%   | 9.988  | 10.046 | 1.0x  | 1848   |        |
| 6  | scratch_census_dp           | scratch | 15.63%   | 23.25%   | 2.64  | 100.0%  | 2.569  | 2.775  | 1.1x  | 1840   | yes    |
| 7  | scratch_census_wta          | scratch | 17.34%   | 22.80%   | 3.06  | 100.0%  | 1.451  | 1.429  | 1.0x  | 530    | yes    |
| 8  | opencv_sgbm_wls             | opencv  | 22.66%   | 27.84%   | 2.18  | 89.2%   | 0.104  | 0.360  | 3.6x  | 15     | yes    |
| 9  | opencv_sgbm_hh              | opencv  | 22.70%   | 26.74%   | 1.75  | 87.3%   | 0.208  | 0.307  | 1.5x  | 199    |        |
| 10 | opencv_sgbm_hh4             | opencv  | 22.74%   | 26.52%   | 1.81  | 87.6%   | 0.068  | 0.264  | 4.2x  | 197    | yes    |
| 11 | opencv_sgbm_3way            | opencv  | 22.98%   | 27.08%   | 1.90  | 87.5%   | 0.039  | 0.138  | 3.8x  | 8      | yes    |
| 12 | opencv_sgbm                 | opencv  | 23.69%   | 27.94%   | 1.91  | 87.2%   | 0.129  | 0.184  | 1.4x  | 5      |        |
| 13 | scratch_zncc_wta            | scratch | 23.88%   | 30.30%   | 4.73  | 100.0%  | 2.201  | 2.158  | 1.0x  | 529    |        |
| 14 | torch_zncc_wta              | torch   | 23.88%   | 30.31%   | 4.73  | 100.0%  | 1.970  | 23.454 | 12.5x | 3      |        |
| 15 | opencv_bm_wls               | opencv  | 27.27%   | 36.63%   | 2.36  | 87.0%   | 0.071  | 0.287  | 4.5x  | 15     |        |
| 16 | scratch_sad_sgm8            | scratch | 29.60%   | 39.49%   | 4.24  | 100.0%  | 10.835 | 10.883 | 1.0x  | 1865   |        |
| 17 | scratch_sad_wta             | scratch | 35.81%   | 46.18%   | 5.72  | 100.0%  | 1.787  | 1.793  | 1.0x  | 529    |        |
| 18 | torch_sad_wta               | torch   | 35.82%   | 46.20%   | 5.72  | 100.0%  | 2.189  | 29.575 | 13.5x | 7      |        |
| 19 | scratch_ssd_wta             | scratch | 36.01%   | 45.94%   | 5.77  | 99.9%   | 1.847  | 1.856  | 1.0x  | 529    |        |
| 20 | opencv_bm                   | opencv  | 36.39%   | 39.27%   | 1.93  | 70.9%   | 0.026  | 0.052  | 2.0x  | 3      | yes    |
| 21 | opencv_quasi_dense          | opencv  | 38.34%   | 46.03%   | 5.15  | 85.5%   | 1.552  | 2.219  | 1.5x  | 29     |        |
| 22 | scratch_sad_wta_lrcheck     | scratch | 43.18%   | 52.50%   | 4.48  | 77.9%   | 2.062  | 2.085  | 1.0x  | 794    |        |
| 23 | learned_raft_stereo_r1      | learned | 61.24%   | 70.74%   | 25.15 | 100.0%  | 0.930  | 13.896 | 15.5x | 81     |        |

### Pareto frontier (accuracy vs wall time)

Nothing in the table is both faster and more accurate than these:

- `opencv_bm`, `opencv_sgbm_3way`, `opencv_sgbm_hh4`, `opencv_sgbm_wls`, `scratch_census_wta`, `scratch_census_dp`, `learned_raft_stereo_r4`

## `bad 2.0` by scene (%)

| algorithm                   | Adirondack | ArtL | Jadeplant | Motorcycle | MotorcycleE | Piano | PianoL | Pipes | Playroom | Playtable | PlaytableP | Recycle | Shelves | Teddy | Vintage |
|-----------------------------|------------|------|-----------|------------|-------------|-------|--------|-------|----------|-----------|------------|---------|---------|-------|---------|
| learned_raft_stereo_r4      | 2.8        | 9.1  | 32.3      | 5.8        | 5.8         | 7.7   | 17.4   | 9.8   | 9.8      | 6.6       | 7.0        | 4.9     | 16.5    | 5.3   | 16.8    |
| torch_census_sgm4           | 3.3        | 12.2 | 14.7      | 4.7        | 4.4         | 12.1  | 24.4   | 9.9   | 12.2     | 18.4      | 8.9        | 5.1     | 28.3    | 5.2   | 21.3    |
| scratch_census_sgm4         | 3.3        | 12.2 | 14.7      | 4.7        | 4.4         | 12.1  | 24.5   | 10.0  | 12.4     | 18.4      | 8.9        | 5.1     | 28.3    | 5.1   | 21.4    |
| scratch_census_sgm8         | 3.5        | 12.2 | 16.1      | 4.6        | 4.3         | 11.8  | 24.4   | 10.4  | 12.1     | 16.2      | 9.0        | 5.2     | 28.3    | 5.7   | 22.4    |
| scratch_census_sgm8_lrcheck | 5.8        | 15.0 | 19.8      | 5.8        | 5.6         | 14.5  | 29.3   | 12.8  | 16.2     | 20.3      | 11.6       | 7.3     | 32.2    | 6.9   | 25.6    |
| scratch_census_dp           | 6.7        | 13.6 | 17.7      | 5.6        | 5.0         | 16.7  | 29.4   | 12.5  | 18.1     | 25.0      | 12.6       | 7.4     | 28.6    | 6.0   | 29.3    |
| scratch_census_wta          | 5.7        | 15.6 | 22.1      | 6.9        | 6.5         | 15.4  | 27.7   | 14.6  | 19.0     | 33.0      | 19.0       | 7.8     | 32.1    | 8.6   | 26.0    |
| opencv_sgbm_wls             | 13.8       | 24.4 | 35.8      | 11.5       | 19.4        | 24.3  | 45.9   | 14.2  | 18.3     | 16.4      | 14.1       | 13.0    | 32.3    | 14.5  | 41.9    |
| opencv_sgbm_hh              | 14.4       | 23.9 | 37.6      | 12.8       | 16.2        | 23.5  | 41.9   | 16.8  | 20.0     | 15.9      | 15.3       | 15.3    | 35.2    | 16.7  | 35.0    |
| opencv_sgbm_hh4             | 14.9       | 24.3 | 36.8      | 12.7       | 15.2        | 23.6  | 40.4   | 17.0  | 18.9     | 17.6      | 15.9       | 15.3    | 36.8    | 16.2  | 35.6    |
| opencv_sgbm_3way            | 15.1       | 25.2 | 36.4      | 12.8       | 15.8        | 25.1  | 42.1   | 17.5  | 19.1     | 19.3      | 16.3       | 15.7    | 32.6    | 15.2  | 36.6    |
| opencv_sgbm                 | 15.5       | 26.3 | 39.3      | 13.1       | 15.7        | 25.3  | 42.9   | 17.8  | 21.4     | 18.5      | 17.8       | 16.0    | 31.5    | 17.1  | 37.2    |
| scratch_zncc_wta            | 13.4       | 28.1 | 30.7      | 13.2       | 12.3        | 21.1  | 40.9   | 21.5  | 27.2     | 33.2      | 23.2       | 14.3    | 34.2    | 12.5  | 32.5    |
| torch_zncc_wta              | 13.4       | 28.2 | 30.7      | 13.2       | 12.3        | 21.1  | 40.9   | 21.5  | 27.2     | 33.2      | 23.2       | 14.3    | 34.2    | 12.5  | 32.5    |
| opencv_bm_wls               | 21.5       | 23.2 | 48.7      | 15.4       | 17.2        | 20.0  | 37.7   | 18.7  | 26.6     | 33.3      | 22.0       | 21.5    | 33.7    | 18.7  | 50.7    |
| scratch_sad_sgm8            | 24.1       | 46.4 | 35.5      | 15.3       | 55.6        | 29.1  | 60.9   | 22.9  | 24.8     | 18.8      | 15.8       | 14.2    | 39.0    | 12.1  | 29.7    |
| scratch_sad_wta             | 34.7       | 47.7 | 40.3      | 17.0       | 60.9        | 29.2  | 68.1   | 27.6  | 35.9     | 34.6      | 29.2       | 22.2    | 42.8    | 17.6  | 29.3    |
| torch_sad_wta               | 34.8       | 47.9 | 40.3      | 16.9       | 60.9        | 29.2  | 68.1   | 27.6  | 35.9     | 34.5      | 29.2       | 22.2    | 42.8    | 17.6  | 29.3    |
| scratch_ssd_wta             | 32.0       | 46.3 | 43.4      | 17.5       | 52.8        | 30.3  | 66.1   | 30.1  | 35.6     | 37.1      | 35.0       | 22.5    | 42.7    | 18.4  | 30.3    |
| opencv_bm                   | 26.9       | 36.8 | 52.9      | 22.6       | 23.3        | 31.5  | 48.8   | 27.5  | 37.2     | 48.2      | 38.0       | 29.3    | 45.6    | 25.8  | 51.6    |
| opencv_quasi_dense          | 30.7       | 51.9 | 72.9      | 25.3       | 25.1        | 31.1  | 44.8   | 47.1  | 38.9     | 40.7      | 34.3       | 25.2    | 38.4    | 21.1  | 47.4    |
| scratch_sad_wta_lrcheck     | 41.8       | 55.3 | 49.4      | 21.6       | 75.0        | 33.0  | 78.6   | 33.7  | 43.5     | 47.1      | 39.5       | 28.1    | 45.5    | 20.1  | 35.6    |
| learned_raft_stereo_r1      | 47.5       | 28.5 | 68.3      | 54.0       | 53.1        | 74.8  | 84.7   | 71.1  | 53.3     | 84.9      | 84.4       | 57.0    | 63.4    | 42.8  | 50.8    |

## Compute cost

`cores busy` is `cpu_s / wall_s`: how many cores the method kept busy. A method near 1.0 is single-threaded, so its wall time understates how much cheaper it is than a method of the same speed running on 16 threads. `Mdisp/s` normalises throughput by both image size and disparity range.

**On the memory columns.** `RSS growth` is the rise in resident memory while the cell ran, and is the figure attributable to the algorithm. `process peak` is the whole interpreter's peak at that moment, so it carries everything earlier cells allocated and should not be read as this method's footprint. Both are under-estimates for a method that happens to reuse memory an allocator already holds, so compare within a family before comparing across one, and re-run a single algorithm on its own if you need a clean number.

| algorithm                   | wall_s | cpu_s  | cores busy | Mpix/s | Mdisp/s | RSS growth MB | process peak MB |
|-----------------------------|--------|--------|------------|--------|---------|---------------|-----------------|
| opencv_bm                   | 0.026  | 0.052  | 2.0        | 14.33  | 1159.8  | 3             | 2251            |
| opencv_sgbm_3way            | 0.039  | 0.138  | 3.8        | 8.91   | 704.2   | 8             | 2273            |
| opencv_sgbm_hh4             | 0.068  | 0.264  | 4.2        | 5.58   | 421.2   | 197           | 2462            |
| opencv_bm_wls               | 0.071  | 0.287  | 4.5        | 5.22   | 438.1   | 15            | 2263            |
| opencv_sgbm_wls             | 0.104  | 0.360  | 3.6        | 3.46   | 267.0   | 15            | 2276            |
| opencv_sgbm                 | 0.129  | 0.184  | 1.4        | 2.64   | 207.3   | 5             | 2268            |
| opencv_sgbm_hh              | 0.208  | 0.307  | 1.5        | 1.70   | 130.2   | 199           | 2463            |
| learned_raft_stereo_r1      | 0.930  | 13.896 | 15.5       | 0.39   | 30.8    | 81            | 2478            |
| scratch_census_wta          | 1.451  | 1.429  | 1.0        | 0.25   | 19.0    | 530           | 2797            |
| opencv_quasi_dense          | 1.552  | 2.219  | 1.5        | 0.23   | 18.8    | 29            | 2281            |
| scratch_sad_wta             | 1.787  | 1.793  | 1.0        | 0.21   | 16.0    | 529           | 2797            |
| scratch_ssd_wta             | 1.847  | 1.856  | 1.0        | 0.22   | 16.0    | 529           | 2798            |
| torch_zncc_wta              | 1.970  | 23.454 | 12.5       | 0.19   | 14.4    | 3             | 3320            |
| scratch_sad_wta_lrcheck     | 2.062  | 2.085  | 1.0        | 0.19   | 14.3    | 794           | 3062            |
| torch_sad_wta               | 2.189  | 29.575 | 13.5       | 0.16   | 12.3    | 7             | 3323            |
| scratch_zncc_wta            | 2.201  | 2.158  | 1.0        | 0.20   | 14.8    | 529           | 2798            |
| scratch_census_dp           | 2.569  | 2.775  | 1.1        | 0.13   | 10.3    | 1840          | 4105            |
| learned_raft_stereo_r4      | 4.430  | 62.024 | 14.4       | 0.08   | 6.2     | 236           | 2773            |
| scratch_census_sgm4         | 4.803  | 4.923  | 1.0        | 0.08   | 5.9     | 1872          | 4139            |
| torch_census_sgm4           | 6.264  | 71.105 | 11.9       | 0.06   | 4.4     | 1257          | 3293            |
| scratch_census_sgm8_lrcheck | 9.988  | 10.046 | 1.0        | 0.04   | 2.7     | 1848          | 4115            |
| scratch_census_sgm8         | 10.005 | 9.588  | 1.0        | 0.04   | 2.9     | 1858          | 4125            |
| scratch_sad_sgm8            | 10.835 | 10.883 | 1.0        | 0.03   | 2.5     | 1865          | 4132            |
