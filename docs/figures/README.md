# Figures

Scratch space for figures used in write-ups and presentations. Committed,
unlike anything under `results/` — put a figure here when you have decided it
is worth keeping.

Benchmark runs do **not** write here. Each run creates its own
`results/<timestamp>/figures/` so results are never silently overwritten:

- `accuracy_vs_time.png` — the accuracy/runtime trade-off, Pareto frontier marked
- `accuracy_by_scene.png` — `bad 2.0` heatmap across scenes
- `qualitative_<scene>.png` — disparity maps on a shared colour scale
- `<scene>__<algorithm>.png` — the individual maps behind that grid

Copy whichever you want to keep into this folder:

```bash
cp results/20261105-035941/figures/accuracy_vs_time.png docs/figures/
```

To keep a whole run's figures alongside its tables instead, so they stay
together with the numbers that produced them, use `results/keep/`:

```bash
python scripts/keep_result.py 20261105-035941 --figures
```

See [repository.md](../repository.md) for which is which.

The presentation's own figures live beside its source in
[`../presentation/`](../presentation/), because LaTeX resolves
`\includegraphics` relative to the `.tex` file.
