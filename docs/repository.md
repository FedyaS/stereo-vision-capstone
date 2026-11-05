# Repository layout

Where everything lives, and what is committed.

For *what the pieces do and what runs in which process*, see
[architecture.md](architecture.md).

---

## The rule

> **Anything a command can regenerate stays out of git. Anything a human
> wrote, or deliberately chose to keep, goes in.**

That single rule explains every entry in [`.gitignore`](../.gitignore). The
repository is ~25 MB of source, documentation and one committed result;
a working checkout with the dataset, the model weights and a few runs is
closer to 600 MB, and none of the difference is worth versioning.

---

## The tree

```
stereo-vision-capstone/
│
├── README.md               what this is, the results, the caveats
├── requirements.txt        the harness's dependencies
├── .gitignore              annotated; the rule above in enforceable form
│
├── bench/          ✓ git   THE HARNESS — pure Python, no web, no UI
│   ├── paths.py            every directory constant, in one place
│   ├── types.py            StereoPair, DisparityResult
│   ├── runner.py           the (scene × algorithm) grid
│   ├── metrics.py          bad N, EPE, RMSE, density
│   ├── profiling.py        wall, CPU, RSS, GPU
│   ├── report.py           summary.md + figures
│   ├── explain*.py         the narrative HTML/PDF report
│   ├── sweep*.py           parameter search and its report
│   ├── splits.py           tune / held-out scene split
│   ├── cli.py              python -m bench
│   ├── paths.py            where everything lives on disk
│   ├── datasets/           middlebury · synthetic · local
│   ├── synth/              the training-image synthesizer (ray tracer)
│   └── algorithms/         opencv · scratch · torch · learned
│
├── app/            ✓ git   THE EXPLORER — a local web UI onto bench/
│   ├── requirements.txt    extra deps, needed only for the app
│   ├── README.md           the five tabs, and what to try in them
│   ├── server/             FastAPI: routes, compute, catalog, jobs, images
│   └── web/                React + Vite frontend
│       ├── src/            ✓ git
│       ├── node_modules/   ✗ npm install
│       └── dist/           ✗ npm run build
│
├── scripts/        ✓ git   STANDALONE UTILITIES — no bench import needed
│   ├── get_dataset.py      download Middlebury MiddEval3 (quarter res)
│   ├── get_models.py       download the RAFT-Stereo ONNX exports
│   └── keep_result.py      copy a run into results/keep/ to commit it
│
├── tests/          ✓ git   pytest: 282 tests, ~65 s
│
├── docs/           ✓ git   DOCUMENTATION
│   ├── architecture.md     what runs where, with diagrams
│   ├── repository.md       this file
│   ├── figures/            scratch space for write-up figures
│   └── presentation/       the LaTeX beamer midterm presentation
│
├── notebooks/      ✓ git   EXPLORATION
│   ├── 01_initial_exploration.ipynb   using bench from a notebook
│   └── practice/           the original hand-written matcher + its image pair
│
├── data/           ✗ git   INPUTS, fetched or uploaded
│   ├── README.md           ✓ git (the one exception)
│   ├── middlebury/         scripts/get_dataset.py        ~45 MB
│   ├── downloads/          the zips, deleted after extraction
│   ├── custom/             pairs uploaded through the app's Datasets tab
│   └── synth/              generated training sets; each carries the
│                           rules.json that reproduces it bit for bit
│
├── models/         ✗ git   INPUTS, fetched
│   ├── README.md           ✓ git (the one exception)
│   └── *.onnx              scripts/get_models.py          ~70 MB
│
└── results/        ✗ git   OUTPUTS, generated
    ├── keep/               ✓ git — the opt-in escape hatch, see below
    ├── <timestamp>/        python -m bench run
    └── sweeps/<timestamp>/ python -m bench sweep
```

`✓ git` = committed · `✗ git` = ignored, with how to regenerate it.

---

## Committing a file that lands in an ignored directory

This is the case the layout above is built around: a run is throwaway, but
*occasionally one matters* — the numbers quoted in a write-up, the report sent
to a supervisor — and that one needs to be in git.

**`results/keep/` is un-ignored.** Anything you put there is committed with a
plain `git add`. No `git add -f`, which is the usual answer and a bad one: a
force-add leaves no trace of *why* the file was special, and the next person
to glance at `.gitignore` concludes the file should not be there.

```bash
python scripts/keep_result.py 20261105-035941
```

```bash
git add results/keep/20261105-035941 && git commit -m "Keep the headline run"
```

`scripts/keep_result.py` copies the small, durable files — `summary.md`,
`results.csv`, `run.json`, and `report.html` / `report.pdf` if present — and
leaves the `maps/` and `figures/` bulk behind. `--figures` brings the PNGs,
`--all` takes the whole folder and warns you how large it is. `--list` shows
what is already kept.

Doing it by hand is equally fine; the script only saves the copying and the
size check. Either way, **add a row to
[`results/keep/README.md`](../results/keep/README.md) saying why** — a kept
result with no recorded reason is indistinguishable from one somebody forgot
to delete.

A kept run is an archive, not a live run: `bench explain` and the app's Runs
tab scan `results/` one level deep for `results.csv`, so nested copies under
`keep/` deliberately do not appear as runs. Copy one back up to browse it.

### Why `results/*` and not `results/`

```gitignore
results/*
!results/keep/
```

Git does not descend into an ignored **directory**, so `results/` — the
obvious spelling — would make the un-ignore on the next line dead, silently.
Ignoring the directory's *contents* with `results/*` leaves git free to
descend and honour the exception. This is the single most common way an
un-ignore rule fails to work.

---

## Where paths are defined

[`bench/paths.py`](../bench/paths.py) is the only place the layout is written
down in code. Six modules used to each recompute the repository root with
`Path(__file__).resolve().parents[N]`, which meant moving a file silently
changed what `N` had to be.

```python
from bench.paths import RESULTS_DIR, MIDDLEBURY_DIR, PRACTICE_DIR
```

Read these constants **at call time**, not bound at import time, so a test can
redirect one with `monkeypatch.setattr` — which `tests/conftest.py` does for
`RESULTS_DIR`, to keep test output out of the real `results/`.

`scripts/` deliberately does *not* import `bench`: the downloaders must work
on a fresh checkout before anything is installed, so they keep their own
root constant.

---

## Adding a file, by kind

| you have | put it | committed? |
|---|---|---|
| a new algorithm | `bench/algorithms/` (or a new module, registered) | yes |
| a new dataset loader | `bench/datasets/` | yes |
| a new generated environment | `bench/synth/environments.py` | yes |
| a test | `tests/` | yes |
| a notebook | `notebooks/` | yes |
| documentation | `docs/` | yes |
| a figure for a write-up | `docs/figures/` | yes |
| a result worth keeping | `results/keep/<name>/` | yes |
| a downloaded dataset | `data/` | no — add a fetch script |
| model weights | `models/` | no — add it to `get_models.py` |
| a benchmark run | `results/<timestamp>/` | no — unless kept |

If something does not fit this table, that is worth a conversation rather
than a `git add -f`.

---

## Generated files that are *not* in `.gitignore` by extension

Two deliberate scoping choices, both to avoid a pattern that is too greedy:

- **LaTeX artifacts** are ignored as `docs/presentation/*.aux` and so on,
  rather than a repo-wide `*.aux`. A bare `*.log` or `*.out` would silently
  swallow an output file somebody meant to commit.
- **`*.tsbuildinfo`** is ignored under `app/web/` only, for the same reason.

Both sets of files were tracked at one point, because `.gitignore` does not
retroactively untrack anything it starts matching. If you ever add a pattern
for something already committed, finish the job:

```bash
git rm --cached <paths> && git commit -m "Untrack generated files"
```

To check for a recurrence — tracked files that the current rules claim to
ignore, which `git check-ignore` hides by default without `--no-index`. The
`grep` drops the `!` un-ignore rules, which are matches by design:

```bash
git ls-files | git check-ignore --no-index --stdin -v | grep -v ':[0-9]*:!'
```

No output means the tracked set and `.gitignore` agree.
