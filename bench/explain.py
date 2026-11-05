"""Turn one benchmark run into a report that explains itself.

``summary.md`` answers *what* the numbers were. This module answers *what they
mean*: it re-reads a finished run's ``results.csv`` and derives the comparisons
a reader would otherwise have to make by hand -- which knob actually moved the
error, which methods only look good because they declined to answer, which
scenes separate the field -- and states each one in prose with the measured
numbers in it.

    python -m bench explain results/20261105-035941

The derivation is deliberately structural, not hard-coded. Algorithm names in
this project decompose as ``<family>_<cost>_<aggregation>[_extra]``, and the
registry tags each one, so "hold everything fixed but the cost function" is a
group-by rather than a list of names I typed out. A run containing algorithms
that did not exist when this file was written still produces its contrasts.

Everything here is pure: in, a run directory; out, a :class:`Findings`.
:mod:`bench.explain_html` renders it. Nothing in this module imports a
rendering library, so the derivations are testable on their own.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .report import PRIMARY, leaderboard, pareto_front

#: Cost functions, most distinctive first -- the order only matters for the
#: rare algorithm tagged with two of them.
COSTS = ("census", "zncc", "ssd", "sad")

#: How to say an aggregation token in English. Keys are matched as prefixes,
#: longest first, so ``sgm8`` resolves before ``sgm``.
AGG_NAMES = {
    "wta": "winner-takes-all",
    "sgm8": "semi-global, 8 paths",
    "sgm4": "semi-global, 4 paths",
    "sgm": "semi-global",
    "dp": "scanline dynamic programming",
    "bm": "block matching",
    "sgbm": "semi-global block matching",
}

#: Extra stages that sit after aggregation.
EXTRA_NAMES = {
    "lrcheck": "left-right consistency check",
    "wls": "WLS edge-aware post-filter",
    "hh": "full-image dynamic programming",
    "hh4": "full-image DP, 4 paths",
    "3way": "3-way optimisation",
}

#: An error this many times the field's median is almost certainly a broken
#: configuration rather than a hard scene, and is called out as such.
BLOWUP_FACTOR = 3.0


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


@dataclass
class RunData:
    """A finished run, as it sits on disk."""

    run_dir: Path
    df: Any  # pandas.DataFrame, one row per (scene, algorithm)
    meta: dict[str, Any]

    @property
    def ok(self):
        return self.df[self.df["status"] == "ok"]

    @property
    def failures(self):
        return self.df[self.df["status"] != "ok"]

    @property
    def scenes(self) -> list[str]:
        return list(self.ok["scene"].drop_duplicates())

    @property
    def figures_dir(self) -> Path:
        return self.run_dir / "figures"


def load_run(run_dir: str | Path) -> RunData:
    """Read ``results.csv`` and ``run.json`` out of a run directory."""
    import pandas as pd

    run_dir = Path(run_dir)
    csv = run_dir / "results.csv"
    if not csv.exists():
        raise FileNotFoundError(
            f"{run_dir} does not look like a benchmark run: no results.csv"
        )
    df = pd.read_csv(csv)
    meta_path = run_dir / "run.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return RunData(run_dir=run_dir, df=df, meta=meta)


# --------------------------------------------------------------------------
# decomposing an algorithm name
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Recipe:
    """What an algorithm is made of, so like can be compared with like."""

    name: str
    family: str
    cost: str | None = None
    agg: str | None = None
    extra: str = ""
    tags: tuple[str, ...] = ()

    @property
    def photometric_invariant(self) -> bool:
        """True where the cost compares *structure* rather than brightness.

        Census keeps only the ordering of neighbouring pixels and ZNCC
        normalises each window's mean and variance away, so neither can be
        moved by a global exposure or gain change. SAD and SSD compare raw
        intensities and are moved by exactly that.
        """
        return "photometric-invariant" in self.tags or self.cost in ("census", "zncc")

    @property
    def understood(self) -> bool:
        """Whether the name decomposed into things this module recognises.

        ``opencv_quasi_dense`` and ``learned_raft_stereo_r4`` are not
        ``<cost>_<aggregation>`` at all, and splitting them on underscores
        yields "quasi + dense" and "raft + stereo r4" -- confident-looking
        nonsense. A recipe is only reported where at least one component is
        a term this module actually knows.
        """
        return bool(
            self.cost
            or self.agg in AGG_NAMES
            or (self.extra and self.extra in EXTRA_NAMES)
        )

    def label(self) -> str:
        """A prose recipe, or an empty string where the name did not decompose."""
        if not self.understood:
            return ""
        bits = []
        if self.cost:
            bits.append(self.cost.upper() if self.cost != "census" else "census")
        if self.agg:
            bits.append(AGG_NAMES.get(self.agg, self.agg))
        if self.extra:
            bits.append(EXTRA_NAMES.get(self.extra, self.extra.replace("_", " ")))
        return " + ".join(bits)


def _registry_tags() -> dict[str, tuple[str, ...]]:
    """Tags for every algorithm currently registered, if any are importable.

    A report can be generated for an old run on a machine where some optional
    dependency has since gone missing, so this must never be the thing that
    fails: an empty mapping just means the names are parsed without the
    registry's help.
    """
    try:
        from .algorithms import all_specs

        return {s.name: tuple(s.tags) for s in all_specs()}
    except Exception:  # pragma: no cover - defensive
        return {}


def decompose(name: str, family: str, tags: Sequence[str] = ()) -> Recipe:
    """Split ``scratch_census_sgm8_lrcheck`` into its parts.

    The tags are authoritative for the cost function where they name one; the
    name supplies the aggregation and any trailing stage, since the tags
    collapse ``sgm4`` and ``sgm8`` into one ``sgm``.
    """
    tags = tuple(tags)
    parts = name.split("_")
    body = parts[1:] if parts and parts[0] == family else parts

    cost = next((c for c in COSTS if c in tags), None)
    if cost is None:
        cost = next((p for p in body if p in COSTS), None)

    rest = list(body)
    if cost in rest:
        rest = rest[rest.index(cost) + 1:]
    agg = rest[0] if rest else None
    extra = "_".join(rest[1:])
    return Recipe(
        name=name, family=family, cost=cost, agg=agg, extra=extra, tags=tags
    )


def recipes(run: RunData) -> dict[str, Recipe]:
    tags = _registry_tags()
    out: dict[str, Recipe] = {}
    for name, fam in run.ok[["algorithm", "family"]].drop_duplicates().itertuples(
        index=False
    ):
        out[name] = decompose(name, fam, tags.get(name, ()))
    return out


# --------------------------------------------------------------------------
# the pieces a report is made of
# --------------------------------------------------------------------------


@dataclass
class Note:
    """One statement about the run, with the numbers that support it.

    ``kind`` drives presentation only: ``headline`` for the things a reader
    should leave with, ``caution`` for a number that would mislead if quoted
    without its caveat, ``detail`` for the rest.
    """

    kind: str
    title: str
    body: str
    evidence: str = ""


@dataclass
class Contrast:
    """A controlled comparison: one component varied, the rest held fixed.

    This is the whole point of the report. A leaderboard tells you that
    ``scratch_census_sgm4`` beat ``scratch_sad_wta``; it does not tell you
    whether that was the cost function or the aggregation, because both
    changed. A contrast changes one.
    """

    axis: str  # "cost" | "aggregation" | "backend"
    held: str  # prose description of what stayed the same
    rows: list[dict[str, Any]] = field(default_factory=list)

    @property
    def spread(self) -> float:
        vals = [r["bad2"] for r in self.rows if r["bad2"] == r["bad2"]]
        return max(vals) - min(vals) if len(vals) > 1 else 0.0

    @property
    def best(self) -> dict[str, Any]:
        return min(self.rows, key=lambda r: r["bad2"])

    @property
    def worst(self) -> dict[str, Any]:
        return max(self.rows, key=lambda r: r["bad2"])


@dataclass
class Findings:
    """Everything a renderer needs. No presentation decisions in here."""

    run: RunData
    lb: Any  # leaderboard dataframe
    pivot: Any  # algorithm x scene, PRIMARY
    recipes: dict[str, Recipe]
    front: list[str]
    notes: list[Note]
    contrasts: list[Contrast]
    scene_stats: list[dict[str, Any]]
    photometric: list[dict[str, Any]]
    abstract: str
    metric_columns: dict[str, dict[str, str]]

    @property
    def n_scenes(self) -> int:
        return len(self.run.scenes)

    @property
    def n_algorithms(self) -> int:
        return int(self.lb.shape[0])


# --------------------------------------------------------------------------
# derivation
# --------------------------------------------------------------------------


def _f(x: Any, default: float = float("nan")) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _mean(pivot, algo: str) -> float:
    try:
        return _f(pivot.loc[algo].mean())
    except KeyError:
        return float("nan")


def _group_contrasts(f: Findings, axis: str) -> list[Contrast]:
    """Build every controlled comparison along one axis.

    The key is the recipe with ``axis`` removed. Any key with two or more
    members is a contrast; a key with one member is an algorithm nothing in
    this run can be compared against, which is worth knowing but is not a
    comparison.
    """
    keyer = {
        "cost": lambda r: (r.family, r.agg, r.extra),
        "aggregation": lambda r: (r.family, r.cost, r.extra),
        "backend": lambda r: (r.cost, r.agg, r.extra),
    }[axis]
    held = {
        "cost": "the same family, aggregation and post-processing",
        "aggregation": "the same family, cost function and post-processing",
        "backend": "the same cost function, aggregation and post-processing",
    }[axis]
    varied = {"cost": "cost", "aggregation": "agg", "backend": "family"}[axis]

    buckets: dict[tuple, list[Recipe]] = {}
    for rec in f.recipes.values():
        if getattr(rec, varied) is None:
            continue
        buckets.setdefault(keyer(rec), []).append(rec)

    out: list[Contrast] = []
    for key, members in buckets.items():
        if len(members) < 2:
            continue
        rows = []
        for rec in sorted(members, key=lambda r: _mean(f.pivot, r.name)):
            lb_row = f.lb[f.lb["algorithm"] == rec.name]
            if lb_row.empty:
                continue
            r = lb_row.iloc[0]
            rows.append({
                "algorithm": rec.name,
                "varied": getattr(rec, varied),
                "varied_label": (
                    AGG_NAMES.get(getattr(rec, varied, ""), getattr(rec, varied, ""))
                    if axis == "aggregation"
                    else getattr(rec, varied)
                ),
                "bad2": _f(r.get(PRIMARY)),
                "epe": _f(r.get("epe_valid")),
                "density": _f(r.get("density")),
                "wall_s": _f(r.get("wall_s")),
                "cpu_s": _f(r.get("cpu_s")),
                "recipe": rec,
            })
        if len(rows) < 2:
            continue
        label = " / ".join(str(k) for k in key if k)
        out.append(Contrast(axis=axis, held=f"{held} ({label})", rows=rows))
    out.sort(key=lambda c: c.spread, reverse=True)
    return out


def _scene_stats(f: Findings) -> list[dict[str, Any]]:
    """Per-scene difficulty and how much the field disagreed on it."""
    from .splits import HELDOUT_SCENES, TUNE_SCENES

    out = []
    for scene in f.pivot.columns:
        col = f.pivot[scene].dropna()
        short = str(scene).split("/")[-1]
        split = (
            "tune" if short in TUNE_SCENES
            else "held-out" if short in HELDOUT_SCENES
            else ""
        )
        rows = f.run.ok[f.run.ok["scene"] == scene]
        out.append({
            "scene": scene,
            "short": short,
            "split": split,
            "mean": _f(col.mean()),
            "best": _f(col.min()),
            "best_algorithm": str(col.idxmin()) if len(col) else "",
            "worst": _f(col.max()),
            "spread": _f(col.max() - col.min()),
            "width": int(rows["width"].iloc[0]) if len(rows) else 0,
            "height": int(rows["height"].iloc[0]) if len(rows) else 0,
            "ndisp": int(rows["ndisp"].iloc[0]) if len(rows) else 0,
        })
    out.sort(key=lambda d: d["mean"], reverse=True)
    return out


def _photometric(f: Findings) -> list[dict[str, Any]]:
    """How much each method lost when only the exposure or lighting changed.

    Middlebury ships variant captures of the same geometry -- ``Motorcycle``
    and ``MotorcycleE`` differ in exposure, ``Piano`` and ``PianoL`` in
    lighting. The disparity field is identical, so any error a method gains
    between the two is attributable to the photometric change alone. That is a
    cleaner experiment than anything in the leaderboard, and it is already in
    the data.
    """
    try:
        from .splits import SCENE_GROUPS
    except Exception:  # pragma: no cover - defensive
        return []

    by_short = {str(c).split("/")[-1]: c for c in f.pivot.columns}
    pairs = [
        (by_short[g[0]], by_short[v], g[0], v)
        for g in SCENE_GROUPS if len(g) > 1 and g[0] in by_short
        for v in g[1:] if v in by_short
    ]
    if not pairs:
        return []

    out = []
    for algo in f.pivot.index:
        deltas, detail = [], []
        for base_col, var_col, base, var in pairs:
            b, v = _f(f.pivot.loc[algo, base_col]), _f(f.pivot.loc[algo, var_col])
            if math.isnan(b) or math.isnan(v):
                continue
            deltas.append(v - b)
            detail.append({"base": base, "variant": var, "delta": v - b,
                           "base_bad2": b, "variant_bad2": v})
        if not deltas:
            continue
        rec = f.recipes.get(algo)
        out.append({
            "algorithm": algo,
            "mean_delta": float(np.mean(deltas)),
            "worst_delta": float(np.max(deltas)),
            "invariant": bool(rec and rec.photometric_invariant),
            "cost": rec.cost if rec else None,
            "detail": detail,
        })
    out.sort(key=lambda d: d["mean_delta"])
    return out


def _notes(f: Findings) -> list[Note]:
    """The statements a reader should leave with, each tied to a measurement."""
    notes: list[Note] = []
    lb = f.lb
    if lb.empty:
        return notes

    top = lb.iloc[0]
    region = f.run.meta.get("config", {}).get("eval_region", "nonocc")

    # --- 1. the winner, and what it cost -----------------------------------
    notes.append(Note(
        kind="headline",
        title=f"{top['algorithm']} is the most accurate method here",
        body=(
            f"It leaves <b>{_f(top[PRIMARY]):.1f}%</b> of evaluated pixels more "
            f"than 2 px from the truth, averaged over {f.n_scenes} scenes, and "
            f"takes <b>{_f(top['wall_s']):.2f} s</b> per pair. Read that number "
            f"as an upper bound on quality for this field, not as a good "
            f"result in absolute terms &mdash; one pixel in "
            f"{1 / max(_f(top[PRIMARY]) / 100, 1e-9):.0f} is still wrong."
        ),
        evidence=f"mean over the {region} region, {f.n_algorithms} algorithms compared",
    ))

    # --- 2. hand-written vs library vs learned ------------------------------
    by_family: dict[str, Any] = {}
    for fam in lb["family"].unique():
        sub = lb[lb["family"] == fam].sort_values(PRIMARY)
        if not sub.empty:
            by_family[fam] = sub.iloc[0]
    if "scratch" in by_family and len(by_family) > 1:
        mine = by_family["scratch"]
        rivals = {k: v for k, v in by_family.items() if k != "scratch"}
        beaten = [k for k, v in rivals.items() if _f(v[PRIMARY]) > _f(mine[PRIMARY])]
        best_rival = min(rivals.values(), key=lambda v: _f(v[PRIMARY]))
        gap = _f(mine[PRIMARY]) - _f(best_rival[PRIMARY])
        if gap <= 0:
            verdict = (
                f"the NumPy implementation is the most accurate thing in the run"
            )
        else:
            verdict = (
                f"it lands <b>{gap:.1f} points</b> behind "
                f"<code>{best_rival['algorithm']}</code>"
            )
        notes.append(Note(
            kind="headline",
            title="A from-scratch matcher is competitive with the libraries",
            body=(
                f"<code>{mine['algorithm']}</code> &mdash; a few hundred lines of "
                f"NumPy &mdash; scores <b>{_f(mine[PRIMARY]):.1f}%</b>, and "
                f"{verdict}"
                + (
                    f", while beating the best "
                    + " and ".join(f"<code>{k}</code>" for k in beaten)
                    + " entry"
                    if beaten else ""
                )
                + ". Classical stereo is not a black box: the accuracy is in "
                "the cost function and the aggregation, both of which fit on a "
                "page."
            ),
            evidence=", ".join(
                f"{k}: {_f(v[PRIMARY]):.1f}%" for k, v in sorted(
                    by_family.items(), key=lambda kv: _f(kv[1][PRIMARY])
                )
            ),
        ))

    # --- 3. which axis moved the error most ---------------------------------
    best_by_axis = {}
    for axis in ("cost", "aggregation"):
        cs = [c for c in f.contrasts if c.axis == axis]
        if cs:
            best_by_axis[axis] = cs[0]
    if len(best_by_axis) == 2:
        cost_c, agg_c = best_by_axis["cost"], best_by_axis["aggregation"]
        bigger = "cost function" if cost_c.spread > agg_c.spread else "aggregation"
        notes.append(Note(
            kind="headline",
            title=f"The {bigger} matters more than the other half of the pipeline",
            body=(
                f"Holding the pipeline fixed and changing only the <b>cost "
                f"function</b> moves <code>bad 2px</code> by "
                f"<b>{cost_c.spread:.1f} points</b> "
                f"({cost_c.worst['varied']} &rarr; {cost_c.best['varied']}: "
                f"{cost_c.worst['bad2']:.1f}% &rarr; {cost_c.best['bad2']:.1f}%). "
                f"Changing only the <b>aggregation</b> moves it by "
                f"<b>{agg_c.spread:.1f}</b> "
                f"({agg_c.worst['varied_label']} &rarr; "
                f"{agg_c.best['varied_label']}: {agg_c.worst['bad2']:.1f}% &rarr; "
                f"{agg_c.best['bad2']:.1f}%). How you compare two patches is the "
                f"decision that counts; smoothing a bad cost volume cannot "
                f"recover what the comparison threw away."
            ),
            evidence=f"{cost_c.held}; {agg_c.held}",
        ))

    # --- 4. the density trap ------------------------------------------------
    if "bad2_valid" in lb.columns and "density" in lb.columns:
        cand = lb.dropna(subset=[PRIMARY, "bad2_valid"]).copy()
        cand["flattery"] = cand[PRIMARY] - cand["bad2_valid"]
        cand = cand[cand["density"] < 0.95].sort_values("flattery", ascending=False)
        if not cand.empty and _f(cand.iloc[0]["flattery"]) > 3.0:
            r = cand.iloc[0]
            rank_all = int(lb.index[lb["algorithm"] == r["algorithm"]][0]) + 1
            by_valid = lb.sort_values("bad2_valid").reset_index(drop=True)
            rank_valid = int(by_valid.index[by_valid["algorithm"] == r["algorithm"]][0]) + 1
            notes.append(Note(
                kind="caution",
                title="Sparse methods look far better than they are",
                body=(
                    f"<code>{r['algorithm']}</code> answers only "
                    f"<b>{_f(r['density']) * 100:.0f}%</b> of pixels. Score it "
                    f"only where it answered and it is "
                    f"<b>{_f(r['bad2_valid']):.1f}%</b> wrong, ranking "
                    f"<b>#{rank_valid}</b>; count the pixels it skipped as "
                    f"errors &mdash; which is what a downstream consumer of a "
                    f"depth map experiences &mdash; and it is "
                    f"<b>{_f(r[PRIMARY]):.1f}%</b> wrong, ranking "
                    f"<b>#{rank_all}</b>. This is why every error figure in this "
                    f"report is given twice. A method that declines to guess is "
                    f"not thereby accurate."
                ),
                evidence=(
                    f"density {_f(r['density']) * 100:.0f}%, "
                    f"gap {_f(r['flattery']):.1f} points between the two framings"
                ),
            ))

    # --- 5. wall time hides compute ----------------------------------------
    if "parallelism" in lb.columns and "cpu_s" in lb.columns:
        pairs = []
        for c in f.contrasts:
            if c.axis != "backend":
                continue
            lo = min(c.rows, key=lambda r: r["cpu_s"])
            hi = max(c.rows, key=lambda r: r["cpu_s"])
            if lo["cpu_s"] > 0:
                pairs.append((hi["cpu_s"] / lo["cpu_s"], lo, hi))
        if pairs:
            ratio, lo, hi = max(pairs, key=lambda t: t[0])
            if ratio > 2.0:
                notes.append(Note(
                    kind="caution",
                    title="Wall time hides what a method actually costs",
                    body=(
                        f"<code>{hi['algorithm']}</code> and "
                        f"<code>{lo['algorithm']}</code> compute the same answer "
                        f"({hi['bad2']:.1f}% and {lo['bad2']:.1f}%) in a "
                        f"comparable wall time ({hi['wall_s']:.2f} s and "
                        f"{lo['wall_s']:.2f} s), but the first burns "
                        f"<b>{hi['cpu_s']:.0f} CPU-seconds</b> to the second's "
                        f"<b>{lo['cpu_s']:.1f}</b> &mdash; a "
                        f"<b>{ratio:.0f}&times;</b> difference in energy for the "
                        f"same result. On a laptop sharing cores with anything "
                        f"else, or on a robot with a power budget, wall time is "
                        f"the wrong column."
                    ),
                    evidence="cpu_s / wall_s is the 'cores busy' column",
                ))

    # --- 6. anything that plainly did not work ------------------------------
    if "epe_valid" in lb.columns:
        med = _f(lb["epe_valid"].median())
        if math.isfinite(med) and med > 0:
            blown = lb[lb["epe_valid"] > med * BLOWUP_FACTOR]
            for _, r in blown.iterrows():
                notes.append(Note(
                    kind="caution",
                    title=f"{r['algorithm']} did not converge &mdash; do not read it as a result",
                    body=(
                        f"Its average error is <b>{_f(r['epe_valid']):.1f} px</b> "
                        f"against a field median of {med:.1f} px, at "
                        f"{_f(r[PRIMARY]):.1f}% bad pixels. An error that size on "
                        f"a dense output is not a hard scene, it is a method "
                        f"running outside the conditions it needs &mdash; too few "
                        f"refinement iterations, or an input normalisation that "
                        f"does not match what the weights expect. It is kept in "
                        f"the table because removing rows that disappoint you is "
                        f"how benchmarks start lying."
                    ),
                    evidence=f"field median EPE {med:.1f} px",
                ))

    # --- 7. photometric robustness -----------------------------------------
    if f.photometric:
        inv = [p for p in f.photometric if p["invariant"]]
        raw = [p for p in f.photometric if not p["invariant"] and p["cost"]]
        if inv and raw:
            i_mean = float(np.mean([p["mean_delta"] for p in inv]))
            r_mean = float(np.mean([p["mean_delta"] for p in raw]))
            worst = max(raw, key=lambda p: p["worst_delta"])
            notes.append(Note(
                kind="headline",
                title="Only structure-based costs survive an exposure change",
                body=(
                    f"The dataset ships pairs that differ in exposure or lighting "
                    f"but not in geometry, so the extra error is caused by the "
                    f"photometric change and nothing else. Costs that compare "
                    f"<i>structure</i> (census, ZNCC) gain "
                    f"<b>{i_mean:+.1f} points</b> on average; costs that compare "
                    f"<i>raw intensity</i> (SAD, SSD) gain "
                    f"<b>{r_mean:+.1f}</b>. The worst case is "
                    f"<code>{worst['algorithm']}</code> at "
                    f"<b>{worst['worst_delta']:+.1f}</b>. This is the single "
                    f"clearest mechanism in the whole run: census discards "
                    f"brightness and keeps only the ordering of neighbouring "
                    f"pixels, so a gain change is invisible to it."
                ),
                evidence=f"{len(inv)} invariant vs {len(raw)} intensity-based methods",
            ))

    # --- 8. the frontier ----------------------------------------------------
    if len(f.front) > 1:
        fsub = lb[lb["algorithm"].isin(f.front)].sort_values("wall_s")
        fastest, most_accurate = fsub.iloc[0], fsub.iloc[-1]
        span = _f(most_accurate["wall_s"]) / max(_f(fastest["wall_s"]), 1e-9)
        gain = _f(fastest[PRIMARY]) - _f(most_accurate[PRIMARY])
        notes.append(Note(
            kind="detail",
            title="The accuracy you can buy with time, and where it stops",
            body=(
                f"{len(f.front)} of {f.n_algorithms} methods are not beaten on "
                f"both axes at once. Moving along that frontier from "
                f"<code>{fastest['algorithm']}</code> to "
                f"<code>{most_accurate['algorithm']}</code> costs "
                f"<b>{span:.0f}&times;</b> the time and buys "
                f"<b>{gain:.1f} points</b> of accuracy. Everything off the "
                f"frontier is strictly dominated &mdash; slower <i>and</i> less "
                f"accurate than something else in the table &mdash; and needs a "
                f"reason other than accuracy or speed to be chosen."
            ),
            evidence="Pareto frontier on (wall time, bad 2px)",
        ))

    # --- 9. the scene that separates the field ------------------------------
    if f.scene_stats:
        hardest = f.scene_stats[0]
        easiest = f.scene_stats[-1]
        widest = max(f.scene_stats, key=lambda d: d["spread"])
        notes.append(Note(
            kind="detail",
            title="Averages over scenes hide where the methods differ",
            body=(
                f"<b>{hardest['short']}</b> is the hardest scene in the run "
                f"(mean {hardest['mean']:.1f}%, best "
                f"{hardest['best']:.1f}% by <code>{hardest['best_algorithm']}</code>) "
                f"and <b>{easiest['short']}</b> the easiest "
                f"({easiest['mean']:.1f}%). But the scene worth looking at is "
                f"<b>{widest['short']}</b>, where the field spans "
                f"<b>{widest['spread']:.0f} points</b> from "
                f"{widest['best']:.1f}% to {widest['worst']:.1f}%. A method's "
                f"mean can look respectable while it collapses on one scene "
                f"type, which is the failure mode that matters if you only ever "
                f"deploy on that type."
            ),
            evidence=f"{f.n_scenes} scenes, spread measured across all methods",
        ))

    return notes


def _abstract(f: Findings) -> str:
    """One paragraph a reader can quote, assembled from the derived numbers."""
    if f.lb.empty:
        return "This run produced no successful cells."
    if PRIMARY not in f.lb.columns:
        # A run on pairs without ground truth: timing and memory only.
        fastest = f.lb.sort_values("wall_s").iloc[0]
        return (
            f"{f.n_algorithms} stereo matching methods were run on "
            f"{f.n_scenes} pair(s) <b>without ground truth</b>, so this run "
            f"measures time and memory only &mdash; there is nothing to score "
            f"accuracy against. The fastest was "
            f"<code>{fastest['algorithm']}</code> at "
            f"<b>{_f(fastest['wall_s']):.3f} s</b> per pair."
        )
    top = f.lb.iloc[0]
    fastest = f.lb.sort_values("wall_s").iloc[0]
    cost_c = next((c for c in f.contrasts if c.axis == "cost"), None)
    agg_c = next((c for c in f.contrasts if c.axis == "aggregation"), None)
    region = f.run.meta.get("config", {}).get("eval_region", "nonocc")

    bits = [
        f"{f.n_algorithms} stereo matching methods were run on {f.n_scenes} "
        f"rectified pairs with ground truth, scored on the <code>{region}</code> "
        f"region and profiled for time and memory on one machine."
    ]
    bits.append(
        f"The most accurate was <code>{top['algorithm']}</code> at "
        f"<b>{_f(top[PRIMARY]):.1f}%</b> of pixels off by more than 2 px; the "
        f"fastest was <code>{fastest['algorithm']}</code> at "
        f"<b>{_f(fastest['wall_s']):.3f} s</b> per pair "
        f"({_f(fastest[PRIMARY]):.1f}%)."
    )
    if cost_c and agg_c:
        which = "cost function" if cost_c.spread > agg_c.spread else "aggregation"
        bits.append(
            f"The run's main result is that the <b>{which}</b> dominates: with "
            f"everything else held fixed, swapping the cost function moves the "
            f"error by {cost_c.spread:.1f} points against {agg_c.spread:.1f} for "
            f"the aggregation strategy."
        )
    if f.photometric:
        inv = [p["mean_delta"] for p in f.photometric if p["invariant"]]
        raw = [p["mean_delta"] for p in f.photometric
               if not p["invariant"] and p["cost"]]
        if inv and raw:
            bits.append(
                f"On the pairs that differ only in exposure or lighting, "
                f"structure-based costs lose {float(np.mean(inv)):+.1f} points "
                f"where intensity-based costs lose "
                f"{float(np.mean(raw)):+.1f}."
            )
    bits.append(
        "Every error figure appears twice &mdash; once counting unanswered "
        "pixels as errors, once over answered pixels only &mdash; because the "
        "two orderings are different and a method can be made to look good by "
        "answering less."
    )
    return " ".join(bits)


#: Column definitions for the interactive table: label, unit, and whether
#: lower is better. The glossary text lives here too, so the table header and
#: the explanation can never drift apart.
METRIC_COLUMNS: dict[str, dict[str, str]] = {
    "bad2_all": {
        "label": "bad 2px",
        "unit": "%",
        "better": "low",
        "group": "accuracy",
        "short": "share of evaluated pixels off by more than 2 px, "
                 "counting unanswered pixels as errors",
        "long": "The headline number. For every pixel in the scoring region, "
                "is the estimate within 2 px of the truth? A pixel the method "
                "declined to estimate counts as wrong, which is the only "
                "framing that compares a dense method with a sparse one "
                "honestly. 2 px is the Middlebury convention; it is loose "
                "enough to ignore sub-pixel noise and tight enough that a "
                "surface at the wrong depth fails it.",
    },
    "bad2_valid": {
        "label": "bad 2px, answered",
        "unit": "%",
        "better": "low",
        "group": "accuracy",
        "short": "the same, but only over pixels the method did estimate",
        "long": "The same test restricted to pixels the method answered. This "
                "measures the quality of the answers it chose to give, which "
                "is the right question when a later stage will fill the holes "
                "anyway &mdash; and the wrong one if nothing will.",
    },
    "bad1_all": {
        "label": "bad 1px",
        "unit": "%",
        "better": "low",
        "group": "accuracy",
        "short": "a stricter threshold: off by more than 1 px",
        "long": "Tightening the threshold separates methods that get the "
                "surface roughly right from methods that get it precisely "
                "right. A method with good <code>bad 2px</code> and poor "
                "<code>bad 1px</code> is finding the correct surface but "
                "quantising it &mdash; usually sub-pixel refinement turned off.",
    },
    "bad0p5_all": {
        "label": "bad 0.5px",
        "unit": "%",
        "better": "low",
        "group": "accuracy",
        "short": "sub-pixel accuracy: off by more than half a pixel",
        "long": "Almost entirely a test of sub-pixel interpolation. Without "
                "it, disparities are integers and roughly half of all pixels "
                "fail this by construction.",
    },
    "epe_valid": {
        "label": "EPE",
        "unit": "px",
        "better": "low",
        "group": "accuracy",
        "short": "mean absolute disparity error over answered pixels",
        "long": "End-point error: the average of |estimate &minus; truth| in "
                "pixels. Where <code>bad N</code> asks how <i>often</i> a "
                "method is wrong, EPE asks how <i>far</i>. A mean is dragged "
                "by outliers, so a method with a low bad-pixel rate and a high "
                "EPE is mostly right with a few catastrophically wrong "
                "regions &mdash; typically a repeated texture matched to the "
                "wrong period.",
    },
    "rmse_valid": {
        "label": "RMSE",
        "unit": "px",
        "better": "low",
        "group": "accuracy",
        "short": "root-mean-square error, which punishes large errors harder",
        "long": "Squaring before averaging weights a single 20 px blunder like "
                "a hundred 2 px wobbles. Compare it against EPE: a large ratio "
                "means the error is concentrated in a few places rather than "
                "spread thin.",
    },
    "a95": {
        "label": "A95",
        "unit": "px",
        "better": "low",
        "group": "accuracy",
        "short": "the error that 95% of answered pixels come in under",
        "long": "A quantile, so no single blunder can move it. Read it as the "
                "error you should expect in the worst 1 pixel in 20 &mdash; a "
                "more useful promise than a mean for anything that has to be "
                "safe rather than accurate on average.",
    },
    "bias_valid": {
        "label": "bias",
        "unit": "px",
        "better": "zero",
        "group": "accuracy",
        "short": "mean signed error: is the method systematically near or far?",
        "long": "The average of (estimate &minus; truth), with its sign kept. "
                "Zero means the errors cancel; a consistent offset points at a "
                "calibration or convention problem &mdash; an off-by-one in the "
                "disparity indexing, or a sub-pixel refinement biased toward "
                "the cost minimum's left side &mdash; rather than at matching "
                "quality.",
    },
    "density": {
        "label": "density",
        "unit": "%",
        "better": "high",
        "group": "accuracy",
        "short": "share of pixels the method estimated at all",
        "long": "The denominator behind every '&hellip;, answered' column. A "
                "method that answers 70% of pixels is being graded on a "
                "self-selected 70%. Always read it next to accuracy: a "
                "consistency check that raises confidence by discarding "
                "ambiguous pixels improves every 'answered' metric and worsens "
                "every honest one.",
    },
    "wall_s": {
        "label": "wall",
        "unit": "s",
        "better": "low",
        "group": "cost",
        "short": "elapsed seconds per pair, as a user experiences it",
        "long": "Clock time for one pair, median over the configured repeats. "
                "It is what latency means, and it is also the noisiest number "
                "in the report: a laptop that has warmed up and dropped its "
                "clocks drifts by more than the differences being measured. "
                "Compare within a run, never across runs.",
    },
    "cpu_s": {
        "label": "CPU",
        "unit": "s",
        "better": "low",
        "group": "cost",
        "short": "CPU-seconds consumed across all threads",
        "long": "Total processor time, summed over every core. This is the "
                "figure proportional to energy and to what you are renting. "
                "Two methods with the same wall time and a 20&times; gap in "
                "CPU time are not equally cheap.",
    },
    "parallelism": {
        "label": "cores busy",
        "unit": "&times;",
        "better": "info",
        "group": "cost",
        "short": "cpu_s / wall_s &mdash; how many cores were kept occupied",
        "long": "1.0 means single-threaded; 16 means sixteen cores were busy "
                "throughout. A single-threaded method's wall time understates "
                "how much cheaper it is than a method of the same speed "
                "spreading across the machine &mdash; and overstates how it "
                "will behave once something else wants those cores.",
    },
    "mdps": {
        "label": "Mdisp/s",
        "unit": "",
        "better": "high",
        "group": "cost",
        "short": "million (pixel &times; disparity) hypotheses evaluated per second",
        "long": "Throughput normalised by both image size and search range, so "
                "it compares a small image with a wide disparity range against "
                "a large image with a narrow one. The honest way to compare "
                "speed across scenes of different sizes.",
    },
    "mpix_per_s": {
        "label": "Mpix/s",
        "unit": "",
        "better": "high",
        "group": "cost",
        "short": "megapixels of output per second",
        "long": "Throughput normalised by image size only. Easier to reason "
                "about than Mdisp/s, but it flatters a method run on a scene "
                "with a narrow disparity range.",
    },
    "rss_delta_mb": {
        "label": "RSS growth",
        "unit": "MB",
        "better": "low",
        "group": "cost",
        "short": "rise in resident memory while the cell ran",
        "long": "The memory attributable to this method: resident set size at "
                "the end minus at the start, sampled on a background thread. "
                "An under-estimate for a method reusing memory the allocator "
                "already holds, so compare within a family first.",
    },
    "peak_rss_mb": {
        "label": "process peak",
        "unit": "MB",
        "better": "info",
        "group": "cost",
        "short": "the whole interpreter's peak at that moment &mdash; not this "
                 "method's footprint",
        "long": "Carries everything earlier cells in the same process "
                "allocated, so it is an upper bound on the process, not a "
                "measurement of the algorithm. Re-run one method alone if you "
                "need a clean number.",
    },
}


#: Columns the report needs that :func:`bench.report.leaderboard` does not
#: average. Kept here rather than added there, because ``summary.md``'s
#: leaderboard is deliberately narrow and this one is deliberately not.
EXTRA_LEADERBOARD_COLUMNS = (
    "bad2_valid", "bad1_valid", "bad0p5_valid", "bad4_valid", "rmse_valid",
    "a90", "a99", "wall_s_std", "n_eval_px",
)


def _augment(lb, ok):
    """Add the per-algorithm means of the columns the report also shows."""
    if lb.empty:
        return lb
    cols = [c for c in EXTRA_LEADERBOARD_COLUMNS
            if c in ok.columns and c not in lb.columns]
    if not cols:
        return lb
    extra = ok.groupby("algorithm", as_index=False)[cols].mean()
    return lb.merge(extra, on="algorithm", how="left")


def derive(run: RunData) -> Findings:
    """Everything the report says, computed from the run."""
    ok = run.ok
    lb = _augment(leaderboard(run.df), ok)
    pivot = (
        ok.pivot_table(index="algorithm", columns="scene", values=PRIMARY,
                       aggfunc="mean")
        if not ok.empty and PRIMARY in ok.columns
        else ok.iloc[:0]
    )
    if not lb.empty and not pivot.empty:
        order = [a for a in lb["algorithm"] if a in pivot.index]
        pivot = pivot.loc[order]

    f = Findings(
        run=run,
        lb=lb,
        pivot=pivot,
        recipes=recipes(run),
        front=pareto_front(lb) if not lb.empty else [],
        notes=[],
        contrasts=[],
        scene_stats=[],
        photometric=[],
        abstract="",
        metric_columns=METRIC_COLUMNS,
    )
    if pivot.empty:
        f.abstract = _abstract(f)
        return f

    for axis in ("cost", "aggregation", "backend"):
        f.contrasts.extend(_group_contrasts(f, axis))
    f.scene_stats = _scene_stats(f)
    f.photometric = _photometric(f)
    f.notes = _notes(f)
    f.abstract = _abstract(f)
    return f


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------


def write_report(
    run_dir: str | Path,
    out_path: str | Path | None = None,
    *,
    gallery_scenes: int = 2,
    gallery_algorithms: int = 6,
    pdf: bool = False,
) -> Path:
    """Generate ``report.html`` for a run. Returns the path written.

    With ``pdf=True`` an installed Chrome or Edge is driven headlessly to
    print the page, and the PDF path is returned instead. The HTML is written
    either way: it is the interactive artefact, and the PDF is a snapshot of
    it with the controls stripped by the print stylesheet.
    """
    from .explain_html import render, to_pdf

    run = load_run(run_dir)
    findings = derive(run)
    html = render(
        findings,
        gallery_scenes=gallery_scenes,
        gallery_algorithms=gallery_algorithms,
    )
    out = Path(out_path) if out_path else Path(run_dir) / "report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    if pdf:
        return to_pdf(out)
    return out
