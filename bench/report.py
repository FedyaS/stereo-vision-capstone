"""Turn raw cell results into tables and figures.

Writes, into the run's output folder:

``summary.md``
    Leaderboard averaged over scenes, a per-scene breakdown, a resource table
    and the list of anything that failed. This is the file to read.
``results.csv``
    One row per (scene, algorithm), every metric. Written by the runner.
``figures/accuracy_vs_time.png``
    The accuracy/time trade-off, with the Pareto frontier marked.
``figures/accuracy_by_scene.png``
    Heatmap of ``bad 2.0`` across scenes -- shows which methods are merely
    average everywhere versus which collapse on specific scenes.
``figures/qualitative_<scene>.png``
    Side-by-side disparity maps, so the numbers can be checked against what
    the output actually looks like.

Colour choices follow the project's data-viz reference palette: a single-hue
blue sequential ramp for magnitude, orange reserved as the one accent, and
identity carried by marker shape and direct labels rather than by hue -- a
scatter with more than three categorical colours cannot stay colourblind-safe.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .runner import CellResult, _slug

# --- reference palette (light surface) ------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8982"
ACCENT = "#eb6834"  # orange, slot 2 -- the single accent
# Sequential blue ramp, steps 100..700 of the reference palette.
BLUE_RAMP = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
# Marker shape carries family identity; it survives any colour-vision deficit.
FAMILY_MARKERS = {
    "opencv": "o",
    "scratch": "s",
    "torch": "^",
    "learned": "D",
}

# The metric used for every ranking. See bench.metrics for why "_all".
PRIMARY = "bad2_all"


def _blue_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("ref_blue", BLUE_RAMP)


def _halo(width: float = 2.5):
    """A surface-coloured outline, so a label stays readable over a mark."""
    import matplotlib.patheffects as pe

    return [pe.withStroke(linewidth=width, foreground=SURFACE)]


def _style(ax) -> None:
    """Recessive axes and grid, so the marks dominate."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color="#e3e2dd", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d6d5cf")
    ax.tick_params(colors=INK_SECONDARY, labelsize=9, length=3)


def to_dataframe(results: Sequence[CellResult]):
    import pandas as pd

    return pd.DataFrame([c.row() for c in results])


# --------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------


def leaderboard(df):
    """Average each algorithm over the scenes it completed."""
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        return ok

    agg: dict = {
        "wall_s": "mean",
        "cpu_s": "mean",
        "parallelism": "mean",
        "peak_rss_mb": "max",
        "rss_delta_mb": "max",
        "mdps": "mean",
        "mpix_per_s": "mean",
        "scene": "count",
    }
    for col in (
        PRIMARY, "bad0p5_all", "bad1_all", "bad4_all",
        "epe_valid", "epe_filled", "rmse_valid", "density", "a95", "bias_valid",
    ):
        if col in ok.columns:
            agg[col] = "mean"

    out = ok.groupby(["family", "algorithm"], as_index=False).agg(agg)
    out = out.rename(columns={"scene": "n_scenes"})
    sort_key = PRIMARY if PRIMARY in out.columns else "wall_s"
    return out.sort_values(sort_key, na_position="last").reset_index(drop=True)


def pareto_front(df, x: str = "wall_s", y: str = PRIMARY) -> list[str]:
    """Algorithms not beaten on both time and accuracy by any other.

    The only rows worth defending in a trade-off discussion: everything else
    is strictly dominated -- slower *and* less accurate than something else in
    the table.
    """
    if df.empty or y not in df.columns:
        return []
    pts = df[["algorithm", x, y]].dropna().sort_values(x)
    front: list[str] = []
    best = float("inf")
    for _, r in pts.iterrows():
        if r[y] < best - 1e-12:
            best = r[y]
            front.append(r["algorithm"])
    return front


def _md_table(rows: list[list[str]], header: list[str]) -> str:
    widths = [
        max(len(header[i]), *(len(r[i]) for r in rows)) if rows else len(header[i])
        for i in range(len(header))
    ]
    def line(cells: Iterable[str]) -> str:
        return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"

    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    return "\n".join([line(header), sep, *(line(r) for r in rows)])


def write_summary(
    results: Sequence[CellResult], out_dir: Path, run_meta: dict | None = None
) -> Path:
    """Write ``summary.md``. Returns its path."""
    import json

    df = to_dataframe(results)
    lb = leaderboard(df)
    front = set(pareto_front(lb))
    meta = run_meta or {}
    if not meta and (out_dir / "run.json").exists():
        meta = json.loads((out_dir / "run.json").read_text())

    parts: list[str] = ["# Stereo benchmark results", ""]

    machine = meta.get("machine", {})
    cfg = meta.get("config", {})
    parts += [
        f"- **when** {meta.get('created_utc', 'n/a')}",
        f"- **machine** {machine.get('cpu_brand') or machine.get('processor', '?')}"
        f" / {machine.get('cpu_count_physical', '?')} physical cores"
        f" / {machine.get('ram_total_gb', '?')} GB RAM",
        f"- **gpu** {machine.get('gpu', 'none detected (CUDA unavailable)')}",
        f"- **libraries** opencv {machine.get('opencv', '?')},"
        f" numpy {machine.get('numpy', '?')}, torch {machine.get('torch', '?')},"
        f" onnxruntime {machine.get('onnxruntime', '?')}",
        f"- **timing** median of {cfg.get('repeats', '?')} run(s)"
        f" after {cfg.get('warmup', '?')} warmup;"
        f" threads = {cfg.get('threads') or 'library default'}",
        f"- **evaluated on** the `{cfg.get('eval_region', '?')}` region",
        "",
        "`bad2_all` is the share of evaluated pixels off by more than 2 px,"
        " counting pixels the method left blank as errors -- the one column"
        " that is fair to both dense and sparse methods. `epe_valid` and"
        " `density` describe where a method did answer. See `bench/metrics.py`.",
        "",
    ]

    # --- leaderboard ------------------------------------------------------
    if not lb.empty:
        parts += ["## Leaderboard (mean over scenes)", ""]
        header = [
            "#", "algorithm", "family", "bad2_all", "bad1_all", "EPE", "density",
            "wall_s", "cpu_s", "par", "RSS+MB", "pareto",
        ]
        rows = []
        for i, r in lb.iterrows():
            rows.append([
                str(i + 1),
                str(r["algorithm"]),
                str(r["family"]),
                f"{r.get(PRIMARY, float('nan')):.2f}%",
                f"{r.get('bad1_all', float('nan')):.2f}%",
                f"{r.get('epe_valid', float('nan')):.2f}",
                f"{r.get('density', float('nan')) * 100:.1f}%",
                f"{r['wall_s']:.3f}",
                f"{r['cpu_s']:.3f}",
                f"{r['parallelism']:.1f}x",
                f"{r['rss_delta_mb']:.0f}",
                "yes" if r["algorithm"] in front else "",
            ])
        parts += [_md_table(rows, header), ""]

        parts += [
            "### Pareto frontier (accuracy vs wall time)", "",
            "Nothing in the table is both faster and more accurate than these:",
            "",
        ]
        parts += [
            "- "
            + ", ".join(f"`{a}`" for a in pareto_front(lb))
            if front
            else "- (none)",
            "",
        ]

    # --- per scene --------------------------------------------------------
    ok = df[df["status"] == "ok"]
    if not ok.empty and PRIMARY in ok.columns:
        pivot = ok.pivot_table(
            index="algorithm", columns="scene", values=PRIMARY, aggfunc="mean"
        )
        order = [a for a in lb["algorithm"] if a in pivot.index]
        pivot = pivot.loc[order]
        parts += ["## `bad 2.0` by scene (%)", ""]
        header = ["algorithm", *[c.split("/")[-1] for c in pivot.columns]]
        rows = [
            [idx, *[("-" if np.isnan(v) else f"{v:.1f}") for v in row]]
            for idx, row in zip(pivot.index, pivot.to_numpy())
        ]
        parts += [_md_table(rows, header), ""]

    # --- resources --------------------------------------------------------
    if not lb.empty:
        parts += [
            "## Compute cost", "",
            "`cores busy` is `cpu_s / wall_s`: how many cores the method kept"
            " busy. A method near 1.0 is single-threaded, so its wall time"
            " understates how much cheaper it is than a method of the same"
            " speed running on 16 threads. `Mdisp/s` normalises throughput by"
            " both image size and disparity range.",
            "",
            "**On the memory columns.** `RSS growth` is the rise in resident"
            " memory while the cell ran, and is the figure attributable to the"
            " algorithm. `process peak` is the whole interpreter's peak at that"
            " moment, so it carries everything earlier cells allocated and"
            " should not be read as this method's footprint. Both are"
            " under-estimates for a method that happens to reuse memory an"
            " allocator already holds, so compare within a family before"
            " comparing across one, and re-run a single algorithm on its own"
            " if you need a clean number.",
            "",
        ]
        header = [
            "algorithm", "wall_s", "cpu_s", "cores busy", "Mpix/s", "Mdisp/s",
            "RSS growth MB", "process peak MB",
        ]
        rows = []
        for _, r in lb.sort_values("wall_s").iterrows():
            rows.append([
                str(r["algorithm"]),
                f"{r['wall_s']:.3f}",
                f"{r['cpu_s']:.3f}",
                f"{r['parallelism']:.1f}",
                f"{r['mpix_per_s']:.2f}",
                f"{r['mdps']:.1f}",
                f"{r['rss_delta_mb']:.0f}",
                f"{r['peak_rss_mb']:.0f}",
            ])
        parts += [_md_table(rows, header), ""]

    # --- problems ---------------------------------------------------------
    bad = df[df["status"] != "ok"]
    if not bad.empty:
        parts += ["## Skipped and failed cells", ""]
        rows = [
            [str(r["algorithm"]), str(r["scene"]), str(r["status"]),
             str(r["message"])[:110]]
            for _, r in bad.iterrows()
        ]
        parts += [_md_table(rows, ["algorithm", "scene", "status", "message"]), ""]

    path = out_dir / "summary.md"
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------


def plot_accuracy_vs_time(lb, path: Path) -> None:
    """Scatter of error against wall time, with the Pareto frontier drawn.

    Identity is marker shape (family) plus direct labels on the frontier only
    -- labelling all two dozen points would collide, and colour is spent on
    ``density`` instead, which is the third thing you need to read a stereo
    result honestly.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if lb.empty or PRIMARY not in lb.columns:
        return

    fig, ax = plt.subplots(figsize=(10, 6.5), facecolor=SURFACE)
    _style(ax)

    front = pareto_front(lb)
    cmap = _blue_cmap()

    for family, marker in FAMILY_MARKERS.items():
        sub = lb[lb["family"] == family]
        if sub.empty:
            continue
        ax.scatter(
            sub["wall_s"], sub[PRIMARY],
            s=95, marker=marker, c=sub["density"] * 100.0,
            cmap=cmap, vmin=0, vmax=100,
            edgecolors=SURFACE, linewidths=1.6, zorder=3, label=family,
        )

    fsub = lb[lb["algorithm"].isin(front)].sort_values("wall_s")
    if len(fsub) > 1:
        ax.plot(
            fsub["wall_s"], fsub[PRIMARY],
            color=ACCENT, linewidth=2, alpha=0.9, zorder=2,
            label="Pareto frontier",
        )

    ax.set_xscale("log")
    # Widen the limits before placing labels so none of them runs off the edge.
    xs = lb["wall_s"].dropna()
    ys = lb[PRIMARY].dropna()
    if len(xs) and xs.min() > 0:
        ax.set_xlim(xs.min() / 3.4, xs.max() * 2.6)
    if len(ys):
        span = max(ys.max() - ys.min(), 1.0)
        ax.set_ylim(ys.min() - 0.1 * span, ys.max() + 0.14 * span)

    # Only the frontier is labelled -- two dozen labels would be unreadable.
    # Successive labels alternate corner (up-right, then down-left), which
    # separates them on both axes at once; the frontier's points get close
    # together at the accurate end, and a purely vertical offset is not enough
    # there. A surface-coloured halo keeps text legible where it crosses the
    # frontier line.
    for n, (_, r) in enumerate(fsub.iterrows()):
        right = n % 2 == 0
        ax.annotate(
            r["algorithm"],
            (r["wall_s"], r[PRIMARY]),
            textcoords="offset points",
            xytext=(10, 9) if right else (-10, -17),
            ha="left" if right else "right",
            fontsize=8.5, color=INK, zorder=5,
            path_effects=_halo(),
        )
    ax.set_xlabel("wall time per pair (s, log scale)", color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel("bad 2.0 px, blanks counted as errors (%)",
                  color=INK_SECONDARY, fontsize=10)
    ax.set_title(
        "Stereo accuracy vs runtime", color=INK, fontsize=13, fontweight="bold",
        loc="left", pad=14,
    )
    ax.text(
        0.0, 1.015,
        "lower and further left is better  ·  marker shape = library family"
        "  ·  fill = result density",
        transform=ax.transAxes, fontsize=9, color=INK_MUTED,
    )

    cb = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 100)),
        ax=ax, pad=0.015, fraction=0.04,
    )
    cb.set_label("density of estimated pixels (%)", color=INK_SECONDARY, fontsize=9)
    cb.ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    cb.outline.set_visible(False)

    leg = ax.legend(
        frameon=False, fontsize=9, labelcolor=INK_SECONDARY, loc="upper right"
    )
    leg.set_zorder(5)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_accuracy_by_scene(df, lb, path: Path) -> None:
    """Heatmap of ``bad 2.0`` per algorithm per scene (sequential, one hue)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = df[df["status"] == "ok"]
    if ok.empty or PRIMARY not in ok.columns or lb.empty:
        return

    pivot = ok.pivot_table(
        index="algorithm", columns="scene", values=PRIMARY, aggfunc="mean"
    )
    order = [a for a in lb["algorithm"] if a in pivot.index]
    if not order:
        return
    pivot = pivot.loc[order]
    data = pivot.to_numpy(dtype=float)

    h = max(4.0, 0.33 * len(pivot.index) + 2.2)
    w = max(7.0, 0.72 * len(pivot.columns) + 4.5)
    fig, ax = plt.subplots(figsize=(w, h), facecolor=SURFACE)

    vmax = float(np.nanpercentile(data, 95)) if np.isfinite(data).any() else 1.0
    im = ax.imshow(
        data, cmap=_blue_cmap(), aspect="auto", vmin=0, vmax=max(vmax, 1.0)
    )

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(
        [c.split("/")[-1] for c in pivot.columns], rotation=45, ha="right", fontsize=8.5
    )
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8.5)
    ax.tick_params(colors=INK_SECONDARY, length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    # Direct values: the contrast WARN on light fills obliges visible labels.
    # Switch to the surface colour before the midpoint -- the ramp darkens
    # faster than its position suggests, and dark ink on a mid-blue cell is
    # the weakest pairing on the chart.
    mid = max(vmax, 1.0) * 0.45
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if not np.isfinite(v):
                continue
            ax.text(
                j, i, f"{v:.0f}", ha="center", va="center", fontsize=7,
                color=SURFACE if v > mid else INK_SECONDARY,
            )

    ax.set_title(
        "bad 2.0 px by scene (%), best methods at the top",
        color=INK, fontsize=12, fontweight="bold", loc="left", pad=12,
    )
    cb = fig.colorbar(im, ax=ax, pad=0.012, fraction=0.025)
    cb.set_label("bad 2.0 (%)", color=INK_SECONDARY, fontsize=9)
    cb.ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    cb.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_qualitative(
    scene: str, figures_dir: Path, algorithms: Sequence[str], path: Path
) -> None:
    """Grid of the left image, the ground truth, and each method's output."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    base = _slug(scene)
    tiles: list[tuple[str, Path]] = []
    for label, suffix in (("left image", "00_left"), ("ground truth", "00_gt")):
        p = figures_dir / f"{base}__{suffix}.png"
        if p.exists():
            tiles.append((label, p))
    for algo in algorithms:
        p = figures_dir / f"{base}__{algo}.png"
        if p.exists():
            tiles.append((algo, p))
    if len(tiles) < 2:
        return

    ncols = min(5, len(tiles))
    nrows = int(np.ceil(len(tiles) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.1 * ncols, 2.7 * nrows), facecolor=SURFACE
    )
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, (label, p) in zip(axes, tiles):
        ax.imshow(mpimg.imread(p))
        ax.set_title(label, fontsize=9, color=INK, pad=4)

    fig.suptitle(
        f"{scene} -- disparity maps on a shared colour scale",
        fontsize=12, color=INK, fontweight="bold", x=0.01, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def write_all(
    results: Sequence[CellResult],
    out_dir: Path,
    *,
    qualitative_scenes: Sequence[str] | None = None,
) -> Path:
    """Write the summary and every figure. Returns the summary path."""
    df = to_dataframe(results)
    lb = leaderboard(df)
    figures = out_dir / "figures"
    figures.mkdir(exist_ok=True)

    summary = write_summary(results, out_dir)
    plot_accuracy_vs_time(lb, figures / "accuracy_vs_time.png")
    plot_accuracy_by_scene(df, lb, figures / "accuracy_by_scene.png")

    scenes = list(
        qualitative_scenes
        if qualitative_scenes is not None
        else df["scene"].drop_duplicates()
    )
    algos = list(lb["algorithm"]) if not lb.empty else []
    for scene in scenes[:3]:
        plot_qualitative(
            scene, figures, algos, figures / f"qualitative_{_slug(scene)}.png"
        )
    return summary
