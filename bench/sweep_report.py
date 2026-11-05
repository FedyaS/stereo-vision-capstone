"""Tables and figures for parameter sweeps.

The interesting output of a sweep is not the winning configuration -- it is
the shape of the response. A parameter whose best and worst settings differ by
0.2 points does not need tuning and should be left alone; one that swings 30
points is where the method's behaviour actually lives. Both the summary table
and the figures are built to show that distinction.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .algorithms import select
from .report import ACCENT, BLUE_RAMP, INK, INK_MUTED, INK_SECONDARY, SURFACE, _halo, _md_table, _style
from .sweep import GRIDS, SweepCell, best_config, current_defaults, iter_configs


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _params_str(params: Mapping[str, Any]) -> str:
    return ", ".join(f"{k}={_fmt(v)}" for k, v in params.items() if not k.startswith("_"))


def summarise(
    all_cells: Mapping[str, Sequence[SweepCell]],
    tolerance_pp: float = 0.5,
    baseline: Mapping[str, Mapping[str, Any]] | None = None,
    grids: Mapping[str, Mapping[str, list]] | None = None,
) -> list[dict[str, Any]]:
    """One row per algorithm: default vs best, and how much the grid mattered.

    ``baseline`` supplies the defaults to compare against. Pass the values
    recorded in the run's ``sweep.json``; reading them from live code gives a
    meaningless comparison once the tuned values have been applied.
    """
    out: list[dict[str, Any]] = []
    for name, cells in all_cells.items():
        ok = [c for c in cells if c.status == "ok" and np.isfinite(c.bad2_all)]
        if not ok:
            out.append({"algorithm": name, "status": "no successful configs"})
            continue

        g = GRIDS if grids is None else grids
        best = best_config(cells, tolerance_pp=tolerance_pp)
        if baseline is not None:
            defaults = dict(baseline.get(name, {}))
        else:
            try:
                spec = select([name])[0]
                defaults = current_defaults(spec, (g.get(name) or {}).keys())
            except Exception:
                defaults = {}

        # The shipped configuration, if the grid happened to contain it.
        default_cell = next(
            (c for c in ok if all(c.params.get(k) == v for k, v in defaults.items())),
            None,
        )
        scores = [c.bad2_all for c in ok]
        out.append({
            "algorithm": name,
            "status": "ok",
            "n_configs": len(ok),
            "default_params": defaults,
            "default_bad2": default_cell.bad2_all if default_cell else float("nan"),
            "default_wall": default_cell.wall_s if default_cell else float("nan"),
            "best_params": best.params if best else {},
            "best_bad2": best.bad2_all if best else float("nan"),
            "best_wall": best.wall_s if best else float("nan"),
            "grid_spread": float(max(scores) - min(scores)),
            "changed": bool(best and defaults and best.params != defaults),
        })
    return out


def boundary_hits(
    all_cells: Mapping[str, Sequence[SweepCell]],
    tolerance_pp: float = 0.5,
    baseline: Mapping[str, Mapping[str, Any]] | None = None,
    grids: Mapping[str, Mapping[str, list]] | None = None,
) -> list[dict[str, Any]]:
    """Parameters whose winning value sits at the edge of the swept range.

    A boundary hit means the search was *truncated*, not that it converged:
    the grid ran out before the response did, so the real optimum may lie
    outside. These are the parameters a second, finer pass should extend --
    bisecting an interior optimum refines an answer, but extending a boundary
    is the only way to find out whether there was a better one all along.

    Only numeric parameters are considered; a categorical value is at an
    "edge" of nothing.
    """
    out: list[dict[str, Any]] = []
    for name, cells in all_cells.items():
        best = best_config(cells, tolerance_pp=tolerance_pp)
        grid = (GRIDS if grids is None else grids).get(name) or {}
        if best is None:
            continue
        for key, values in grid.items():
            if key not in best.params:
                continue
            numeric = [v for v in values if isinstance(v, (int, float))]
            if len(numeric) < 2:
                continue
            chosen = best.params[key]
            if not isinstance(chosen, (int, float)):
                continue
            lo, hi = min(numeric), max(numeric)
            edge = "min" if chosen <= lo else ("max" if chosen >= hi else None)
            if edge is None:
                continue
            out.append({
                "algorithm": name,
                "parameter": key,
                "value": chosen,
                "edge": edge,
                "range": (lo, hi),
            })
    return out


def recommended_changes(
    all_cells: Mapping[str, Sequence[SweepCell]],
    tolerance_pp: float = 0.5,
    baseline: Mapping[str, Mapping[str, Any]] | None = None,
    grids: Mapping[str, Mapping[str, list]] | None = None,
) -> list[dict[str, Any]]:
    """Parameters whose swept best differs from the shipped default.

    The auditable half of a sweep: the exact edits to make, with the gain each
    one buys, so updating the defaults is a reviewable diff rather than a
    claim.
    """
    changes: list[dict[str, Any]] = []
    for row in summarise(all_cells, tolerance_pp, baseline=baseline, grids=grids):
        if row.get("status") != "ok" or not row.get("changed"):
            continue
        for key, new in row["best_params"].items():
            old = row["default_params"].get(key, "(unset)")
            if old == new:
                continue
            changes.append({
                "algorithm": row["algorithm"],
                "parameter": key,
                "old": old,
                "new": new,
                "default_bad2": row["default_bad2"],
                "best_bad2": row["best_bad2"],
                "gain_pp": row["default_bad2"] - row["best_bad2"],
            })
    return changes


def write_sweep_summary(
    all_cells: Mapping[str, Sequence[SweepCell]],
    out_dir: Path,
    *,
    tune_scenes: Sequence[str] = (),
    tolerance_pp: float = 0.5,
    baseline: Mapping[str, Mapping[str, Any]] | None = None,
    grids: Mapping[str, Mapping[str, list]] | None = None,
) -> Path:
    rows = summarise(all_cells, tolerance_pp, baseline=baseline, grids=grids)
    parts: list[str] = [
        "# Parameter sweep",
        "",
        f"- **tune scenes** ({len(tune_scenes)}): "
        + (", ".join(s.split("/")[-1] for s in tune_scenes) or "n/a"),
        "- **selection rule**: lowest mean `bad2_all`, then the fastest"
        f" configuration within {tolerance_pp} percentage points of it",
        "- **held-out scenes were not loaded here** -- see `bench/splits.py`",
        "",
        "`grid spread` is the gap between the best and worst configuration"
        " tried. A small spread means the parameter barely matters for that"
        " method and the default was already fine; a large one means the"
        " method's published accuracy depends heavily on tuning.",
        "",
    ]

    table: list[list[str]] = []
    for r in rows:
        if r.get("status") != "ok":
            table.append([r["algorithm"], "-", "-", "-", "-", str(r.get("status"))])
            continue
        delta = r["default_bad2"] - r["best_bad2"]
        table.append([
            r["algorithm"],
            f"{r['default_bad2']:.2f}%" if np.isfinite(r["default_bad2"]) else "n/a",
            f"{r['best_bad2']:.2f}%",
            f"{delta:+.2f}" if np.isfinite(delta) else "n/a",
            f"{r['grid_spread']:.1f}",
            _params_str(r["best_params"]) or "(defaults)",
        ])
    parts += [
        _md_table(
            table,
            ["algorithm", "pre-sweep", "tuned", "gain (pp)", "grid spread", "tuned config"],
        ),
        "",
    ]

    # The actionable diff.
    changes = recommended_changes(
        all_cells, tolerance_pp, baseline=baseline, grids=grids
    )
    parts += ["## Recommended default changes", ""]
    if changes:
        parts += [
            "Each row is one edit to an adapter's dataclass default. `gain`"
            " is for the whole configuration, not the single parameter.",
            "",
            _md_table(
                [
                    [
                        c["algorithm"],
                        c["parameter"],
                        _fmt(c["old"]),
                        _fmt(c["new"]),
                        f"{c['gain_pp']:+.2f}" if np.isfinite(c["gain_pp"]) else "n/a",
                    ]
                    for c in changes
                ],
                ["algorithm", "parameter", "from", "to", "gain (pp)"],
            ),
            "",
        ]
    else:
        parts += ["The sweep confirmed every shipped default. No changes.", ""]

    # Searches that ran out of grid rather than converging.
    edges = boundary_hits(all_cells, tolerance_pp, grids=grids)
    parts += ["## Truncated searches", ""]
    if edges:
        parts += [
            "The winning value sits at the edge of the swept range for these,"
            " so the grid ran out before the response did and the true optimum"
            " may lie further out. A second pass should *extend* these rather"
            " than refine around them.",
            "",
            _md_table(
                [
                    [
                        e["algorithm"],
                        e["parameter"],
                        _fmt(e["value"]),
                        e["edge"],
                        f"{_fmt(e['range'][0])}..{_fmt(e['range'][1])}",
                    ]
                    for e in edges
                ],
                ["algorithm", "parameter", "chosen", "at", "swept range"],
            ),
            "",
        ]
    else:
        parts += ["Every winning value is interior to its swept range.", ""]

    # Per-algorithm detail: the top handful of configurations.
    parts += ["## Top configurations per algorithm", ""]
    for name, cells in all_cells.items():
        ok = [c for c in cells if c.status == "ok" and np.isfinite(c.bad2_all)]
        if not ok:
            continue
        ok = sorted(ok, key=lambda c: c.bad2_all)[:5]
        parts += [f"### `{name}`", ""]
        parts += [
            _md_table(
                [
                    [
                        _params_str(c.params) or "(defaults)",
                        f"{c.bad2_all:.2f}%",
                        f"{c.bad1_all:.2f}%",
                        f"{c.epe_valid:.2f}",
                        f"{c.density * 100:.1f}%",
                        f"{c.wall_s:.3f}",
                    ]
                    for c in ok
                ],
                ["config", "bad2_all", "bad1_all", "EPE", "density", "wall_s"],
            ),
            "",
        ]

    failed = [
        (name, c)
        for name, cells in all_cells.items()
        for c in cells
        if c.status != "ok"
    ]
    if failed:
        parts += ["## Skipped and failed configurations", ""]
        parts += [
            _md_table(
                [
                    [name, _params_str(c.params), c.status, c.message[:90]]
                    for name, c in failed[:40]
                ],
                ["algorithm", "config", "status", "message"],
            ),
            "",
        ]

    path = out_dir / "sweep_summary.md"
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def plot_parameter_response(
    name: str,
    cells: Sequence[SweepCell],
    path: Path,
    *,
    default: Mapping | None = None,
    grids: Mapping[str, Mapping[str, list]] | None = None,
) -> None:
    """Small multiples: ``bad2_all`` against each swept parameter.

    For each value of a parameter, two things are drawn -- the *best*
    ``bad2_all`` achievable with that value (what the parameter is worth once
    everything else is tuned around it) and the *mean* over the other
    parameters (how forgiving that value is). A parameter whose best-case line
    is flat is not worth tuning, whatever its mean does.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = [c for c in cells if c.status == "ok" and np.isfinite(c.bad2_all)]
    if not ok:
        return
    grid = (GRIDS if grids is None else grids).get(name) or {}
    keys = [k for k in grid if any(k in c.params for c in ok)]
    if not keys:
        return

    ncols = min(3, len(keys))
    nrows = int(np.ceil(len(keys) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.3 * ncols, 3.3 * nrows), facecolor=SURFACE, squeeze=False
    )
    flat_axes = axes.ravel()
    for ax in flat_axes[len(keys):]:
        ax.axis("off")

    for ax, key in zip(flat_axes, keys):
        _style(ax)
        values = sorted({c.params[key] for c in ok if key in c.params}, key=_sort_key)
        labels = [_fmt(v) for v in values]
        xs = np.arange(len(values))
        best = [min(c.bad2_all for c in ok if c.params.get(key) == v) for v in values]
        mean = [float(np.mean([c.bad2_all for c in ok if c.params.get(key) == v]))
                for v in values]

        ax.plot(xs, mean, color=BLUE_RAMP[4], linewidth=2, marker="o",
                markersize=6, label="mean over other params", zorder=3)
        ax.plot(xs, best, color=BLUE_RAMP[9], linewidth=2.2, marker="o",
                markersize=7, label="best achievable", zorder=4)

        if default is not None and key in default:
            try:
                di = values.index(default[key])
                ax.axvline(di, color=ACCENT, linewidth=1.6, linestyle="--",
                           zorder=2, label="shipped default")
            except ValueError:
                pass

        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_xlabel(key, color=INK_SECONDARY, fontsize=9.5)
        ax.set_ylabel("bad 2.0 (%)", color=INK_SECONDARY, fontsize=9)
        ax.set_xlim(-0.35, len(values) - 0.65)

    spread = max(c.bad2_all for c in ok) - min(c.bad2_all for c in ok)
    fig.suptitle(
        f"{name} -- parameter response  (grid spread {spread:.1f} points)",
        fontsize=12.5, color=INK, fontweight="bold", x=0.01, ha="left",
    )
    handles, labels_ = flat_axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels_, frameon=False, fontsize=9, labelcolor=INK_SECONDARY,
        loc="upper right", ncol=3, bbox_to_anchor=(0.995, 1.0),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _sort_key(v: Any):
    if isinstance(v, (int, float)):
        return (0, float(v), "")
    return (1, 0.0, str(v))


def write_all(
    all_cells: Mapping[str, Sequence[SweepCell]],
    out_dir: Path,
    *,
    tune_scenes: Sequence[str] = (),
    tolerance_pp: float = 0.5,
    baseline: Mapping[str, Mapping[str, Any]] | None = None,
    grids: Mapping[str, Mapping[str, list]] | None = None,
) -> Path:
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for name, cells in all_cells.items():
        if baseline is not None:
            default = dict(baseline.get(name, {}))
        else:
            try:
                spec = select([name])[0]
                default = current_defaults(
                spec,
                ((GRIDS if grids is None else grids).get(name) or {}).keys(),
            )
            except Exception:
                default = None
        plot_parameter_response(
            name, cells, figures / f"{name}.png", default=default, grids=grids
        )
    return write_sweep_summary(
        all_cells, out_dir, tune_scenes=tune_scenes, tolerance_pp=tolerance_pp,
        baseline=baseline, grids=grids,
    )
