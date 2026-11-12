# Stereo Playground

A local web app for poking at the benchmark: run any algorithm on any scene,
drag its parameters and watch the disparity map change, compare matchers side
by side, trigger real benchmark runs, add your own image pairs, and generate
training data with exact ground truth.

Python backend (FastAPI) + TypeScript/React frontend (Vite). Everything runs
on your machine; nothing is uploaded anywhere.

This app is a front end onto the `bench` harness — it imports it and calls the
same functions the CLI does, rather than reimplementing a stereo run. See
[`docs/architecture.md`](../docs/architecture.md) for the process boundaries
(which server, which port, which thread, which cache) and
[`docs/repository.md`](../docs/repository.md) for what is committed.

---

## Run it

From the repository root, with the project's virtualenv active:

```bash
pip install -r app/requirements.txt
```

```bash
cd app/web && npm install && npm run build
```

```bash
python -m app.server
```

That serves the UI and the API together on <http://127.0.0.1:8000> and opens a
browser. Interactive API docs are at `/docs`.

### While editing the frontend

```bash
python -m app.server --reload --no-open
```

```bash
cd app/web && npm run dev
```

Vite serves on <http://localhost:5173> with hot reload and proxies `/api` to
the backend on port 8000.

---

## The five tabs

### Playground

One algorithm, one scene, live parameters. Pick a matcher, drag a slider and
the map re-renders — auto-run is on by default and debounced, so dragging does
not fire a run per pixel.

Six views share one colour scale, taken from the scene's ground truth, so two
algorithms can be compared by eye without one being rescaled into looking
better:

| view | what it shows |
|---|---|
| left image | the input |
| ground truth | what the answer should be |
| disparity | what the matcher produced |
| bad pixels | red where the error exceeds the threshold |
| error magnitude | how wrong, not just whether |
| where it answered | dark where the method declined to estimate |

The last one matters more than it looks. A sparse matcher that answers only
where it is confident scores beautifully on "error over answered pixels" and
terribly on anything fair, so the UI always shows density next to accuracy.

**Scoring is separate from matching.** Changing the evaluation region or the
bad-pixel threshold re-scores the *stored* disparity map — milliseconds, no
re-matching. That split is deliberate: it makes it obvious which numbers come
from the algorithm and which come from how you chose to score it.

Things worth trying:

- Switch `metric` between `sad` and `census` on **MotorcycleE**. The scene is
  deliberately mis-exposed between the two views; SAD collapses and census
  barely notices, because census keeps only the *ordering* of neighbouring
  pixels.
- Set `aggregation` to `dp` and look for horizontal streaking. The scanline DP
  solves each row exactly and independently, so nothing ties one row to the
  next.
- Raise `uniqueness_ratio` and watch density fall while "bad, answered"
  improves and `bad 2px` gets worse. That is the metric telling you it counts
  a blank as an error.
- Turn `subpixel` off and watch `bad 0.5px` jump while `bad 4px` barely moves.

### Compare

Several algorithms, one scene, ranked. The presets each isolate one variable —
`cost functions` holds the pipeline fixed and changes only how patches are
compared; `backends` runs the same maths on NumPy and on torch. Runs are
sequential on purpose: the table reports wall time, and matchers competing for
the same cores would make it meaningless.

### Runs

Browse stored benchmark runs — the sortable leaderboard, the generated
figures, and the full `summary.md`.

**open report** builds the narrative report for that run and opens it in a new
tab: the controlled comparisons behind the leaderboard, a glossary of every
metric, and the caveats, all derived from that run's own numbers. It is the
same page `python -m bench explain` writes, cached in the run folder after the
first click — a few seconds, most of it resampling the gallery thumbnails.
**PDF** prints it through a headless Chrome or Edge, and **regenerate**
rebuilds it from the run's data — useful if the run folder changed under it.
Note that it rebuilds with the code the *running server* imported, so an edit
to `bench/explain.py` needs `--reload` or a restart before it shows up.

You can also trigger a new batch run. Pick **whole datasets** — one checkbox
per folder, with each `train`/`val`/`test` subset selectable on its own — or
switch to **individual pairs** and tick them one at a time. *pairs per
dataset* caps how many to take from each set: the first N of a fixed order,
so a quick pass over 20 and the full run later cover the same pairs in the
same sequence.

The button states the real cell count, resolved by the server rather than
guessed: picking a set and a subset of it does not run those pairs twice.
Per-pair PNGs are written automatically for small runs and skipped for large
ones — a hundred pairs across five algorithms is six hundred images nobody
looks at — and the checkbox overrides either way.

Output goes to `results/<timestamp>/` exactly like `python -m bench run` does,
with the same summary and figures. Progress polls while it works and it can be
cancelled.

### Datasets

**One pair:** upload a rectified image pair, optionally with a `.pfm`
ground-truth disparity map. It lands in `data/custom/<name>/` and appears
everywhere the built-in scenes do.

**A whole dataset:** copy the folder into `data/custom/` and press **rescan**.
Any shape works — a single pair, a flat folder of samples, or splits with
samples under them — because a *pair* is just a folder holding `im0`/`im1`
(or `left`/`right`) images, optionally with `disp0GT.pfm`, `mask0nocc.png` and
`calib.txt` beside them, and a *dataset* is any folder with pairs below it.
The **Dataset folders** table lists what was found, how many pairs each holds
and what its subsets are; **browse** walks the samples.

A dataset folder contributes only its first twelve samples to the scene
picker — listing twenty thousand would mean decoding twenty thousand images
to describe them. The rest are reached by the dataset browser at the top of
the Playground and Compare tabs, which steps through every pair in a set, and
the whole set is one checkbox in the Runs tab.

You can also generate a synthetic pair from a seed — textured scene warped by
a known disparity field, so the ground truth is exact by construction rather
than measured.

Pairs must already be **rectified**: matching only searches along a row, so
any vertical offset between the views is unrecoverable here.

### Synthesize

Build a training set. Pick an environment and a difficulty, adjust the rules,
render a few examples, then generate as many pairs as you want.

The example panel shows the left image, the right image, the ground truth and
the occlusion mask together, on purpose: a rule set can render a handsome
picture and supervise almost nothing, and you cannot tell from the left image
alone. Alongside them are the *measured* statistics — ground-truth coverage,
how much of the frame is visible to both cameras, the disparity range against
the search range — plus warnings for the mistakes that are easy to make and
hard to see, such as a search range the scene only fills a third of.

Generation runs in the background across all cores and writes to
`data/synth/<name>/` in the MiddEval3 layout, with a manifest recording the
exact rules and seeds. The pairs appear in the scene picker as soon as the job
finishes, so you can benchmark what you just made without an export step.

Output is split into `train/`, `val/` and `test/` by default, with exact
counts and no scene shared between them; untick "split" for one flat folder.
On by default deliberately — every pair here comes from the same generator,
so evaluating on pairs you trained on gives excellent numbers and no warning
that anything is wrong.

"Mix environments" cycles through several scene types instead of one; a model
trained on a single environment generalises poorly, so that is usually what
you want.

The ground truth is exact rather than estimated — both views are ray traced
independently from analytic 3-D geometry, so disparity comes from a solved
`focal * baseline / Z` and the occlusion mask is a real visibility
computation. See the main README for why this is a ray tracer and not a
wrapper around Blender or Open3D.

---

## Notes

- **Use `downscale` on the from-scratch matchers.** A census SGM at full
  Middlebury resolution is a few seconds per run, which is too slow to drag a
  slider against. At 2x or 4x it is interactive, and the *relative* ordering of
  algorithms barely changes.
- **Timing in the UI is indicative.** It defaults to one repeat. Raise
  `timing repeats` in the Playground, or use the CLI with `--repeats 5`, before
  quoting a number — a laptop under sustained load drifts by more than the
  differences you are measuring.
- **Results are cached in memory**, the last 48 runs. A `result_id` older than
  that returns 404 and the UI re-runs it.
- The server binds to `127.0.0.1` and has no authentication, because it can
  read and write files under the repository. Do not expose it to a network.

## Layout

```
app/
  server/
    main.py       FastAPI routes
    compute.py    single-pair runs, caching, re-scoring
    catalog.py    scene discovery, custom dataset upload
    params.py     turns adapter dataclasses into UI controls
    images.py     PNG rendering of disparity, error and mask views
    jobs.py       background batch runs and dataset generation
    synth.py      the synthesizer: preview cache, generated-dataset listing
  web/
    src/
      App.tsx       tab shell
      api.ts        typed client
      components.tsx shared controls
      views/        Playground, Compare, Runs, Datasets, Synthesize
```

The parameter controls are generated, not hand-written: every adapter is a
dataclass, so its fields are its parameter list, and the sweep grids in
`bench/sweep.py` supply the ranges worth searching. Add a parameter to an
algorithm and a control for it appears here automatically.

The Synthesize tab works the same way. Its editor is built from
`/api/synth/schema`, which is served straight from `bench.synth.rules`, so a
new rule gets a control with its range and help text without the frontend
knowing it exists -- and the editor cannot drift away from what the generator
will actually accept.
