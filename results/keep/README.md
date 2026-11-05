# Kept results

`results/` is ignored — a run is reproducible and a full one is tens of
megabytes of disparity maps and figures. **This folder is the exception.**
Everything under `results/keep/` is committed, so plain `git add` works here
and nowhere else under `results/`:

```bash
python scripts/keep_result.py 20261105-035941
```

```bash
git add results/keep/20261105-035941 && git commit -m "Keep the headline run"
```

`scripts/keep_result.py` copies a run's small, durable files — `summary.md`,
`results.csv`, `run.json`, and `report.html` / `report.pdf` if they exist —
and leaves behind the `maps/` and `figures/` bulk. Pass `--figures` to bring
the figures too, or `--all` for the whole folder, and it will tell you how
large that is before you commit it.

Keeping a result by hand works just as well; the script only saves you the
copying and the size check:

```bash
mkdir -p results/keep/my-run && cp results/20261105-035941/summary.md results/keep/my-run/
```

## What is in here

| run | why it is kept |
|---|---|
| `20261105-035941` | the headline run quoted in the root README: all 15 scenes x 23 algorithms, round-2 defaults |

Add a row when you add a run. A kept result with no reason recorded is
indistinguishable from one somebody forgot to delete.

## A kept run is an archive, not a live run

`python -m bench explain` and the app's **Runs** tab scan `results/` one level
deep for folders containing `results.csv`, so these nested copies deliberately
do not show up there. To browse a kept run in the app, copy it back up:

```bash
cp -r results/keep/20261105-035941 results/20261105-035941
```
