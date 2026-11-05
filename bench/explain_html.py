"""Render a :class:`~bench.explain.Findings` as one self-contained HTML page.

Self-contained is the requirement that drives every other decision here: one
file, no network, no CDN, no sibling assets. It can be mailed, committed,
opened on a machine that has never had Python on it, and printed to PDF
without anything going missing. Images are inlined as base64 JPEG, the charts
are drawn in a few hundred lines of vanilla JS from a JSON blob embedded in the
page, and the print stylesheet strips the controls so a printed copy is a
coherent document rather than a screenshot of a web app.

Colour follows the project's data-viz reference palette, which is also what
:mod:`bench.report` uses for its PNGs, so the page and the figures in the run
folder agree:

* **sequential** single-hue blue ramp wherever a number is a magnitude (every
  heatmap cell, the scatter's density fill);
* **two categorical hues at most**, blue and orange, and only where two series
  genuinely coexist (wall vs CPU time, invariant vs intensity-based costs);
* identity otherwise carried by marker shape and direct labels, because a
  scatter with two dozen categorical colours cannot stay colourblind-safe;
* both light and dark are selected, not flipped -- the dark steps are their own
  steps of the same ramps against the dark surface.
"""
from __future__ import annotations

import base64
import html
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .explain import METRIC_COLUMNS, Findings
from .report import PRIMARY, BLUE_RAMP
from .runner import _slug

#: Metrics offered in the chart selectors. Everything else stays in the table.
CHART_METRICS = (
    PRIMARY, "bad2_valid", "bad1_all", "bad0p5_all", "epe_valid", "rmse_valid",
    "a95", "density", "wall_s", "cpu_s", "mdps", "rss_delta_mb",
)

#: Order the leaderboard's columns are shown in.
TABLE_COLUMNS = (
    PRIMARY, "bad2_valid", "bad1_all", "bad0p5_all", "epe_valid", "a95",
    "bias_valid", "density", "wall_s", "cpu_s", "parallelism", "mdps",
    "rss_delta_mb",
)

#: Marker shapes, one per family. Shape survives any colour-vision deficit and
#: survives being printed in greyscale, which hue does not.
FAMILY_SHAPE = {
    "opencv": "circle",
    "scratch": "square",
    "torch": "triangle",
    "learned": "diamond",
}

FAMILY_BLURB = {
    "opencv": "OpenCV's own matchers, C++ and SIMD-optimised",
    "scratch": "written here in NumPy, to be read rather than to be fast",
    "torch": "the same maths as <code>scratch</code>, re-expressed as tensor "
             "operations so it can move to a GPU",
    "learned": "a pretrained deep network, run through ONNX Runtime",
}


def esc(x: Any) -> str:
    return html.escape(str(x), quote=True)


def _num(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v == v and abs(v) != float("inf") else None


def _fmt(key: str, value: Any) -> str:
    """Format one metric for display, in the unit its column header promises."""
    v = _num(value)
    if v is None:
        return "&mdash;"
    if key == "density":
        return f"{v * 100:.0f}%"
    if key.startswith("bad"):
        return f"{v:.1f}%"
    if key in ("wall_s", "cpu_s"):
        return f"{v:.3f}" if v < 1 else f"{v:.2f}"
    if key == "parallelism":
        return f"{v:.1f}&times;"
    if key in ("rss_delta_mb", "peak_rss_mb"):
        return f"{v:.0f}"
    if key == "bias_valid":
        return f"{v:+.2f}"
    return f"{v:.2f}"


# --------------------------------------------------------------------------
# the data the page's JavaScript draws from
# --------------------------------------------------------------------------


def _payload(f: Findings) -> dict[str, Any]:
    """Everything the charts need, as plain JSON-able Python."""
    algos: list[dict[str, Any]] = []
    for _, r in f.lb.iterrows():
        name = str(r["algorithm"])
        rec = f.recipes.get(name)
        algos.append({
            "name": name,
            "family": str(r["family"]),
            "shape": FAMILY_SHAPE.get(str(r["family"]), "circle"),
            "front": name in f.front,
            "recipe": rec.label() if rec else "",
            "metrics": {k: _num(r.get(k)) for k in set(CHART_METRICS) | set(TABLE_COLUMNS)},
        })

    matrix: dict[str, dict[str, dict[str, float | None]]] = {}
    ok = f.run.ok
    for metric in CHART_METRICS:
        if metric not in ok.columns:
            continue
        piv = ok.pivot_table(index="algorithm", columns="scene", values=metric,
                             aggfunc="mean")
        matrix[metric] = {
            str(a): {str(s): _num(piv.loc[a, s]) for s in piv.columns}
            for a in piv.index
        }

    return {
        "algorithms": algos,
        "scenes": [
            {"id": s["scene"], "short": s["short"], "split": s["split"]}
            for s in sorted(f.scene_stats, key=lambda d: d["short"])
        ],
        "matrix": matrix,
        "metrics": {
            k: {kk: vv for kk, vv in METRIC_COLUMNS[k].items()}
            for k in CHART_METRICS if k in METRIC_COLUMNS
        },
        "chartMetrics": [m for m in CHART_METRICS if m in METRIC_COLUMNS],
        "ramp": BLUE_RAMP,
        "primary": PRIMARY,
    }


# --------------------------------------------------------------------------
# image gallery
# --------------------------------------------------------------------------


def _inline_image(
    path: Path, width: int = 300, quality: int = 80
) -> dict[str, Any] | None:
    """Downscale one figure and return it as a ``data:`` URI with its size.

    The run folder holds ~95 MB of PNGs at full resolution. Inlining those
    would produce a report no browser enjoys, so each tile is resampled to
    thumbnail width and JPEG-encoded; the gallery exists to show the *character*
    of each method's failure (streaks, blobs, holes), which survives that.

    The intrinsic size is returned with it and written onto the ``<img>``. A
    data URI still decodes asynchronously, so without it the gallery lays out
    at zero height, and anything that measures or prints the page before the
    decode finishes -- the headless browser that produces the PDF, for one --
    captures empty boxes.
    """
    try:
        import cv2
        import numpy as np

        img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
        h, w = img.shape[:2]
        if w > width:
            h, w = max(1, round(h * width / w)), width
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            return None
        return {
            "uri": "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode(),
            "w": int(w),
            "h": int(h),
        }
    except Exception:
        return None


def _gallery(f: Findings, n_scenes: int, n_algos: int) -> list[dict[str, Any]]:
    """Qualitative tiles for a few scenes: the inputs, then selected methods.

    The selection is the top ``n_algos - 1`` of the leaderboard plus the
    *worst* entry. Showing only the winners makes a gallery that teaches
    nothing: the point of looking at the maps is to see what each kind of
    failure looks like.
    """
    if f.lb.empty or not f.scene_stats:
        return []
    ranked = [str(a) for a in f.lb["algorithm"]]
    chosen = ranked[: max(1, n_algos - 1)]
    if len(ranked) > len(chosen):
        chosen = chosen + [ranked[-1]]

    # Hardest and easiest, so the contrast between them is visible.
    picks = []
    stats = f.scene_stats
    for s in (stats[0], stats[-1], *stats[1:-1]):
        if s not in picks:
            picks.append(s)
    picks = picks[:n_scenes]

    out = []
    for s in picks:
        base = _slug(s["scene"])
        tiles = []
        for label, suffix, note in (
            ("left image", "00_left", "the input"),
            ("ground truth", "00_gt", "what the answer should be"),
        ):
            img = _inline_image(f.run.figures_dir / f"{base}__{suffix}.png")
            if img:
                tiles.append({"label": label, "note": note, **img})
        for algo in chosen:
            img = _inline_image(f.run.figures_dir / f"{base}__{algo}.png")
            if not img:
                continue
            v = None
            try:
                v = _num(f.pivot.loc[algo, s["scene"]])
            except Exception:
                v = None
            tiles.append({
                "label": algo,
                "note": f"bad 2px {v:.1f}%" if v is not None else "",
                **img,
            })
        if len(tiles) > 2:
            out.append({"scene": s, "tiles": tiles})
    return out


# --------------------------------------------------------------------------
# HTML fragments
# --------------------------------------------------------------------------


def _bar_rows(
    rows: list[dict[str, Any]], key: str, *, labels: list[str] | None = None,
    series: list[str] | None = None, max_value: float | None = None,
) -> str:
    """A CSS bar chart: one row per entry, value label outside the bar.

    CSS rather than SVG because these print correctly, scale with the reader's
    font size, and need no JavaScript to exist.
    """
    vals = [_num(r.get(key)) or 0.0 for r in rows]
    top = max_value if max_value is not None else (max(vals) if vals else 1.0)
    top = max(top, 1e-9)
    out = ['<div class="bars">']
    for i, r in enumerate(rows):
        v = _num(r.get(key)) or 0.0
        pct = max(1.0, 100.0 * v / top)
        label = labels[i] if labels else str(r.get("algorithm", ""))
        cls = f" s{series[i]}" if series else ""
        out.append(
            f'<div class="bar-row">'
            f'<div class="bar-label">{label}</div>'
            f'<div class="bar-track"><div class="bar-fill{cls}" '
            f'style="width:{pct:.2f}%"></div></div>'
            f'<div class="bar-value">{_fmt(key, v)}</div>'
            f"</div>"
        )
    out.append("</div>")
    return "\n".join(out)


def _notes_html(f: Findings) -> str:
    order = {"headline": 0, "caution": 1, "detail": 2}
    notes = sorted(f.notes, key=lambda n: order.get(n.kind, 9))
    parts = []
    for n in notes:
        icon = {"headline": "&#9679;", "caution": "&#9888;"}.get(n.kind, "&#9675;")
        parts.append(
            f'<div class="finding {esc(n.kind)}">'
            f'<div class="finding-head"><span class="finding-icon" aria-hidden="true">'
            f"{icon}</span>"
            f'<h3>{n.title}</h3>'
            f'<span class="finding-kind">{esc(n.kind)}</span></div>'
            f"<p>{n.body}</p>"
            + (f'<p class="evidence">{esc(n.evidence)}</p>' if n.evidence else "")
            + "</div>"
        )
    return "\n".join(parts)


def _table_html(f: Findings) -> str:
    cols = [c for c in TABLE_COLUMNS if c in f.lb.columns]
    head = [
        '<th class="sticky-col" data-key="algorithm" data-type="text" '
        'title="click to sort">algorithm</th>',
        '<th data-key="family" data-type="text">family</th>',
    ]
    for c in cols:
        m = METRIC_COLUMNS.get(c, {})
        arrow = {"low": "&darr; lower is better", "high": "&uarr; higher is better"}.get(
            m.get("better", ""), ""
        )
        head.append(
            f'<th class="num" data-key="{esc(c)}" data-type="num" '
            f'data-tip="<b>{esc(m.get("label", c))}</b> ({esc(c)})<br>'
            f'{m.get("short", "")}<br><i>{arrow}</i>">'
            f'{m.get("label", c)}<span class="unit">{m.get("unit", "")}</span></th>'
        )
    body = []
    for i, (_, r) in enumerate(f.lb.iterrows()):
        name = str(r["algorithm"])
        rec = f.recipes.get(name)
        front = name in f.front
        cells = [
            f'<td class="sticky-col" data-val="{esc(name)}">'
            f'<span class="rank">{i + 1}</span> <code>{esc(name)}</code>'
            + (' <span class="pill front" data-tip="On the Pareto frontier: '
               'nothing in this run is both faster and more accurate.">frontier</span>'
               if front else "")
            + (f'<div class="recipe">{esc(rec.label())}</div>' if rec and rec.label() else "")
            + "</td>",
            f'<td data-val="{esc(r["family"])}">{esc(r["family"])}</td>',
        ]
        for c in cols:
            v = _num(r.get(c))
            cells.append(
                f'<td class="num" data-val="{"" if v is None else v}">'
                f"{_fmt(c, r.get(c))}</td>"
            )
        body.append(f"<tr>{''.join(cells)}</tr>")
    return (
        '<div class="table-wrap"><table id="lb-table"><thead><tr>'
        + "".join(head)
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def _contrast_section(f: Findings) -> str:
    """One block per axis: the widest controlled comparison, then the rest."""
    intro = {
        "cost": (
            "How two candidate patches are compared. SAD and SSD add up raw "
            "intensity differences; ZNCC normalises each window's mean and "
            "variance away; census throws the intensities out entirely and "
            "keeps only whether each neighbour is brighter or darker than the "
            "centre."
        ),
        "aggregation": (
            "What happens to the cost volume before a winner is picked. "
            "Winner-takes-all picks per pixel independently. Scanline DP "
            "solves each row optimally but ties nothing between rows. "
            "Semi-global matching accumulates along 4 or 8 directions, so a "
            "pixel's choice is informed by the whole image."
        ),
        "backend": (
            "The same arithmetic, executed differently. These should agree on "
            "the answer to within floating-point noise; where they differ in "
            "accuracy, one of them has a bug. What they do differ in is how "
            "much of the machine they use to get there."
        ),
    }
    title = {
        "cost": "Changing only the cost function",
        "aggregation": "Changing only the aggregation",
        "backend": "Changing only the backend",
    }
    parts = []
    for axis in ("cost", "aggregation", "backend"):
        cs = [c for c in f.contrasts if c.axis == axis and c.spread > 0.0]
        if not cs:
            continue
        c = cs[0]
        rows = sorted(c.rows, key=lambda r: r["bad2"])
        labels = [
            f'<b>{esc(r["varied_label"] or r["varied"])}</b>'
            f'<span class="sub">{esc(r["algorithm"])}</span>'
            for r in rows
        ]
        extra = ""
        if axis == "backend":
            extra = (
                '<div class="aside"><h4>The same answer at very different '
                "cost</h4>"
                + _bar_rows(
                    [r for r in rows for _ in (0,)][: len(rows)],
                    "cpu_s",
                    labels=[f'<b>{esc(r["algorithm"])}</b>' for r in rows],
                    series=["2"] * len(rows),
                )
                + '<p class="caption">CPU-seconds, summed over every core. '
                "Wall time is nearly identical; this is not.</p></div>"
            )
        others = ""
        if len(cs) > 1:
            items = []
            for o in cs[1:4]:
                items.append(
                    f'<li><b>{o.spread:.1f} pt</b> spread &mdash; '
                    f'<code>{esc(o.worst["algorithm"])}</code> '
                    f'{o.worst["bad2"]:.1f}% &rarr; '
                    f'<code>{esc(o.best["algorithm"])}</code> '
                    f'{o.best["bad2"]:.1f}%</li>'
                )
            others = (
                '<details class="more"><summary>the same comparison in '
                f"{len(cs) - 1} other pipeline(s)</summary><ul>"
                + "".join(items)
                + "</ul></details>"
            )
        parts.append(
            f'<div class="card contrast">'
            f"<h3>{title[axis]}</h3>"
            f'<p class="lede">{intro[axis]}</p>'
            f'<p class="held">Held fixed: {esc(c.held)}.</p>'
            + _bar_rows(rows, "bad2", labels=labels)
            + f'<p class="caption">bad 2px, mean over {f.n_scenes} scenes. '
            f"Spread across this axis: <b>{c.spread:.1f} percentage points</b>.</p>"
            + extra
            + others
            + "</div>"
        )
    return "\n".join(parts)


def _photometric_section(f: Findings) -> str:
    if not f.photometric:
        return ""
    rows = f.photometric
    labels = [
        f'<b>{esc(p["algorithm"])}</b>'
        f'<span class="sub">{"structure-based" if p["invariant"] else "intensity-based" if p["cost"] else "cost unknown"}</span>'
        for p in rows
    ]
    series = ["1" if p["invariant"] else "2" if p["cost"] else "n" for p in rows]
    shifted = [{"v": p["mean_delta"]} for p in rows]
    lo = min(0.0, min(p["mean_delta"] for p in rows))
    span = max(p["mean_delta"] for p in rows) - lo
    out = ['<div class="bars diverge">']
    zero = (-lo / span * 100) if span > 0 else 0.0
    for p, lab, s in zip(rows, labels, series):
        v = p["mean_delta"]
        width = abs(v) / span * 100 if span > 0 else 0
        left = zero if v >= 0 else zero - width
        out.append(
            f'<div class="bar-row"><div class="bar-label">{lab}</div>'
            f'<div class="bar-track"><div class="bar-zero" style="left:{zero:.2f}%"></div>'
            f'<div class="bar-fill s{s}" style="left:{left:.2f}%;width:{max(width, 0.6):.2f}%">'
            f"</div></div>"
            f'<div class="bar-value">{v:+.1f}</div></div>'
        )
    out.append("</div>")
    pairs = rows[0]["detail"] if rows else []
    pair_text = ", ".join(
        f'<code>{esc(d["base"])}</code> &rarr; <code>{esc(d["variant"])}</code>'
        for d in pairs
    )
    return (
        '<div class="card">'
        "<h3>The cleanest experiment in the run</h3>"
        '<p class="lede">The dataset ships repeat captures of the same scene '
        "that differ only in exposure or lighting. The geometry is identical, "
        "so the ground truth is identical, so every extra error a method makes "
        "on the variant is caused by the photometric change and nothing else. "
        "No other comparison here isolates a single cause this cleanly.</p>"
        f'<p class="held">Pairs used: {pair_text}.</p>'
        + "\n".join(out)
        + '<p class="caption">Change in bad 2px from the base capture to its '
        "variant, averaged over the pairs. "
        '<span class="key"><span class="swatch s1"></span>structure-based cost '
        "(census, ZNCC)</span> "
        '<span class="key"><span class="swatch s2"></span>intensity-based cost '
        "(SAD, SSD)</span> "
        '<span class="key"><span class="swatch sn"></span>cost not identifiable '
        "from the registry</span></p>"
        "</div>"
    )


def _scene_table(f: Findings) -> str:
    rows = []
    for s in f.scene_stats:
        split = (
            f'<span class="pill {"tune" if s["split"] == "tune" else "held"}">'
            f'{esc(s["split"])}</span>' if s["split"] else ""
        )
        rows.append(
            f"<tr><td><b>{esc(s['short'])}</b> {split}</td>"
            f'<td class="num">{s["width"]}&times;{s["height"]}</td>'
            f'<td class="num">{s["ndisp"]}</td>'
            f'<td class="num">{s["mean"]:.1f}%</td>'
            f'<td class="num">{s["best"]:.1f}%</td>'
            f"<td><code>{esc(s['best_algorithm'])}</code></td>"
            f'<td class="num">{s["spread"]:.0f}</td></tr>'
        )
    return (
        '<div class="table-wrap"><table class="plain"><thead><tr>'
        "<th>scene</th><th>size</th><th>ndisp</th>"
        '<th class="num" data-tip="mean bad 2px across every method">mean</th>'
        '<th class="num">best</th><th>by</th>'
        '<th class="num" data-tip="best to worst, in percentage points -- how '
        'much the choice of method matters on this scene">spread</th>'
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def _glossary(f: Findings) -> str:
    groups: dict[str, list[tuple[str, dict]]] = {"accuracy": [], "cost": []}
    for k, m in METRIC_COLUMNS.items():
        groups.setdefault(m.get("group", "other"), []).append((k, m))
    parts = []
    titles = {
        "accuracy": "Accuracy &mdash; was it right?",
        "cost": "Cost &mdash; what did it take?",
    }
    for g, items in groups.items():
        if not items:
            continue
        parts.append(f"<h3>{titles.get(g, esc(g))}</h3><dl>")
        for k, m in items:
            better = {
                "low": "lower is better", "high": "higher is better",
                "zero": "closer to zero is better", "info": "neither good nor bad",
            }.get(m.get("better", ""), "")
            # Each term is wrapped so it cannot be split down the middle by a
            # column or page break.
            parts.append(
                f'<div class="gloss"><dt><code>{esc(k)}</code> '
                f'<span class="gloss-label">{m["label"]}'
                + (f' <span class="unit">{m["unit"]}</span>' if m.get("unit") else "")
                + "</span>"
                + (f'<span class="better">{better}</span>' if better else "")
                + f'</dt><dd>{m["long"]}</dd></div>'
            )
        parts.append("</dl>")
    return "\n".join(parts)


def _gallery_html(gallery: list[dict[str, Any]]) -> str:
    if not gallery:
        return ""
    parts = []
    for g in gallery:
        s = g["scene"]
        tiles = "".join(
            f'<figure><img src="{t["uri"]}" width="{t["w"]}" height="{t["h"]}" '
            f'alt="{esc(t["label"])} for {esc(s["short"])}">'
            f'<figcaption><b>{esc(t["label"])}</b>'
            + (f'<span class="sub">{esc(t["note"])}</span>' if t["note"] else "")
            + "</figcaption></figure>"
            for t in g["tiles"]
        )
        parts.append(
            f'<div class="card">'
            f'<h3>{esc(s["short"])} <span class="muted">&mdash; mean '
            f'{s["mean"]:.1f}%, spread {s["spread"]:.0f} points</span></h3>'
            f'<div class="gallery">{tiles}</div></div>'
        )
    return "\n".join(parts)


def _appendix(f: Findings) -> str:
    meta = f.run.meta
    machine, cfg = meta.get("machine", {}), meta.get("config", {})
    def rows(d: dict) -> str:
        return "".join(
            f"<tr><td><code>{esc(k)}</code></td><td>{esc(v)}</td></tr>"
            for k, v in d.items()
        )
    fails = f.run.failures
    fail_html = ""
    if not fails.empty:
        fr = "".join(
            f"<tr><td><code>{esc(r['algorithm'])}</code></td>"
            f"<td>{esc(str(r['scene']).split('/')[-1])}</td>"
            f"<td>{esc(r['status'])}</td><td>{esc(str(r['message'])[:160])}</td></tr>"
            for _, r in fails.iterrows()
        )
        fail_html = (
            f"<h3>Skipped and failed cells ({len(fails)})</h3>"
            '<div class="table-wrap"><table class="plain"><thead><tr>'
            "<th>algorithm</th><th>scene</th><th>status</th><th>why</th>"
            f"</tr></thead><tbody>{fr}</tbody></table></div>"
        )
    else:
        fail_html = (
            f'<p class="lede">All {meta.get("n_cells", "?")} cells completed. '
            "A failure here would be recorded as a row rather than aborting the "
            "run, so an empty list means an empty list.</p>"
        )
    return (
        f"{fail_html}"
        "<h3>Run configuration</h3>"
        f'<div class="table-wrap"><table class="plain kv"><tbody>{rows(cfg)}'
        "</tbody></table></div>"
        "<h3>Machine</h3>"
        f'<div class="table-wrap"><table class="plain kv"><tbody>{rows(machine)}'
        "</tbody></table></div>"
    )


# --------------------------------------------------------------------------
# the page
# --------------------------------------------------------------------------

_CSS = """
:root{
  --surface:#fcfcfb; --plane:#f9f9f7; --ink:#0b0b0b; --ink2:#52514e;
  --muted:#898781; --grid:#e1e0d9; --border:#e3e2dd;
  --s1:#2a78d6; --s2:#eb6834; --sn:#898781;
  --good:#0ca30c; --warn:#fab219; --crit:#d03b3b;
  --cell-ink:#ffffff;
  color-scheme:light;
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --surface:#1a1a19; --plane:#0d0d0d; --ink:#f0efec; --ink2:#c3c2b7;
    --muted:#898781; --grid:#2c2c2a; --border:#2c2c2a;
    --s1:#3987e5; --s2:#d95926; --sn:#898781;
    color-scheme:dark;
  }
}
:root[data-theme="dark"]{
  --surface:#1a1a19; --plane:#0d0d0d; --ink:#f0efec; --ink2:#c3c2b7;
  --muted:#898781; --grid:#2c2c2a; --border:#2c2c2a;
  --s1:#3987e5; --s2:#d95926; --sn:#898781;
  color-scheme:dark;
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{
  margin:0; background:var(--plane); color:var(--ink);
  font:15px/1.62 ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  font-variant-numeric:tabular-nums;
}
code,.mono{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;font-size:.88em}
code{background:color-mix(in srgb,var(--ink) 7%,transparent);padding:.08em .32em;border-radius:3px}
.wrap{max-width:1080px;margin:0 auto;padding:0 20px 80px}
h1{font-size:30px;line-height:1.2;margin:0 0 6px;letter-spacing:-.015em}
h2{font-size:21px;margin:0 0 4px;letter-spacing:-.01em}
h3{font-size:16px;margin:0 0 8px}
h4{font-size:13.5px;margin:14px 0 6px;color:var(--ink2)}
p{margin:0 0 10px}
.muted{color:var(--muted);font-weight:400}
.lede{color:var(--ink2)}
.caption{font-size:12.5px;color:var(--muted);margin:8px 0 0}
.held{font-size:12.5px;color:var(--muted);margin:-2px 0 12px}
.evidence{font-size:12px;color:var(--muted);margin:6px 0 0;font-style:italic}

/* --- header ----------------------------------------------------------- */
header{border-bottom:1px solid var(--border);background:var(--surface);padding:26px 0 18px;margin-bottom:26px}
.eyebrow{font-size:11.5px;letter-spacing:.09em;text-transform:uppercase;color:var(--muted);margin:0 0 8px}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:14px}
.chip{font-size:11.5px;border:1px solid var(--border);border-radius:999px;padding:3px 10px;color:var(--ink2);background:var(--plane)}
.chip b{color:var(--ink)}
.toolbar{display:flex;gap:8px;align-items:center;margin-top:16px;flex-wrap:wrap}
button,select{
  font:inherit;font-size:13px;padding:5px 11px;border-radius:7px;
  border:1px solid var(--border);background:var(--surface);color:var(--ink);cursor:pointer;
}
button:hover,select:hover{border-color:var(--muted)}
.abstract{
  background:var(--surface);border:1px solid var(--border);border-left:3px solid var(--s1);
  border-radius:4px;padding:16px 18px;margin:20px 0 0;font-size:15.5px;line-height:1.66;
}

/* --- sections --------------------------------------------------------- */
section{margin:38px 0 0;scroll-margin-top:16px}
.sec-head{display:flex;align-items:baseline;gap:10px;margin-bottom:14px;padding-bottom:8px;border-bottom:1px solid var(--border)}
.sec-num{font-size:12px;color:var(--muted);font-variant-numeric:tabular-nums}
.sec-head h2{flex:0 0 auto}
.sec-head p{margin:0;font-size:13px;color:var(--muted);flex:1 1 280px;min-width:0}
.card{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:16px 18px;margin:0 0 14px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media (max-width:820px){.grid2{grid-template-columns:1fr}}

/* --- findings --------------------------------------------------------- */
.finding{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:15px 18px;margin:0 0 12px;border-left:3px solid var(--s1)}
.finding.caution{border-left-color:var(--s2)}
.finding.detail{border-left-color:var(--grid)}
.finding-head{display:flex;align-items:baseline;gap:9px;margin-bottom:6px}
.finding-head h3{margin:0;flex:1}
.finding-icon{color:var(--s1);font-size:11px}
.finding.caution .finding-icon{color:var(--s2)}
.finding.detail .finding-icon{color:var(--muted)}
.finding-kind{font-size:10.5px;letter-spacing:.07em;text-transform:uppercase;color:var(--muted)}
.finding p{margin:0;color:var(--ink2)}
.finding b{color:var(--ink)}

/* --- tables ----------------------------------------------------------- */
.table-wrap{overflow:auto;border:1px solid var(--border);border-radius:8px;background:var(--surface)}
table{border-collapse:separate;border-spacing:0;width:100%;font-size:12.5px}
th,td{padding:7px 10px;text-align:left;border-bottom:1px solid var(--border);white-space:nowrap}
thead th{position:sticky;top:0;background:var(--surface);z-index:2;font-weight:600;color:var(--ink2);border-bottom:1px solid var(--grid)}
#lb-table thead th{cursor:pointer;user-select:none}
#lb-table thead th:hover{color:var(--ink)}
#lb-table thead th.sorted::after{content:" \\2193";color:var(--s1)}
#lb-table thead th.sorted.asc::after{content:" \\2191"}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.unit{color:var(--muted);font-weight:400;margin-left:3px;font-size:10.5px}
tbody tr:hover td{background:color-mix(in srgb,var(--s1) 6%,transparent)}
.sticky-col{position:sticky;left:0;background:var(--surface);z-index:1}
tbody tr:hover .sticky-col{background:color-mix(in srgb,var(--s1) 6%,var(--surface))}
.rank{color:var(--muted);font-size:11px;display:inline-block;min-width:16px}
.recipe{color:var(--muted);font-size:11px;margin-top:1px;padding-left:20px}
table.plain tbody tr:hover td{background:color-mix(in srgb,var(--s1) 5%,transparent)}
table.kv td:first-child{width:220px;color:var(--ink2)}
.pill{font-size:10px;letter-spacing:.04em;text-transform:uppercase;padding:1px 6px;border-radius:999px;border:1px solid var(--border);color:var(--ink2)}
.pill.front{border-color:var(--s2);color:var(--s2)}
.pill.tune{border-color:var(--s1);color:var(--s1)}
.pill.held{border-color:var(--muted);color:var(--muted)}

/* --- bars ------------------------------------------------------------- */
.bars{margin:10px 0 0}
.bar-row{display:grid;grid-template-columns:minmax(120px,210px) 1fr 62px;gap:10px;align-items:center;padding:3px 0}
.bar-label{font-size:12.5px;line-height:1.3;overflow:hidden;text-overflow:ellipsis}
.bar-label .sub{display:block;color:var(--muted);font-size:10.5px;font-family:ui-monospace,Menlo,Consolas,monospace}
.bar-track{position:relative;height:16px;background:color-mix(in srgb,var(--ink) 5%,transparent);border-radius:3px}
.bar-fill{position:absolute;top:0;bottom:0;left:0;background:var(--s1);border-radius:0 3px 3px 0;border:2px solid var(--surface);border-left-width:0}
.bar-fill.s1{background:var(--s1)} .bar-fill.s2{background:var(--s2)} .bar-fill.sn{background:var(--sn)}
.diverge .bar-fill{border-radius:3px;border-left-width:2px}
.bar-zero{position:absolute;top:-2px;bottom:-2px;width:1px;background:var(--muted);opacity:.6}
.bar-value{font-size:12px;text-align:right;font-variant-numeric:tabular-nums;color:var(--ink)}
.key{display:inline-flex;align-items:center;gap:4px;margin-right:12px}
.swatch{width:9px;height:9px;border-radius:2px;display:inline-block;background:var(--s1)}
.swatch.s2{background:var(--s2)} .swatch.sn{background:var(--sn)}
.aside{margin-top:16px;padding-top:12px;border-top:1px solid var(--border)}
details.more{margin-top:12px;font-size:12.5px;color:var(--ink2)}
details.more summary{cursor:pointer;color:var(--s1)}
details.more ul{margin:8px 0 0;padding-left:20px}
details.more li{margin:3px 0}

/* --- charts ----------------------------------------------------------- */
.chart-head{display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:6px}
svg.chart{display:block;width:100%;height:auto;overflow:visible}
svg text{fill:var(--ink2);font-size:10px}
svg .axis-line{stroke:var(--grid)}
svg .gridline{stroke:var(--grid);stroke-width:1}
svg .mark{stroke:var(--surface);stroke-width:1.6;cursor:pointer}
svg .mark.dim{opacity:.22}
svg .front-path{fill:none;stroke:var(--s2);stroke-width:2}
svg .label{fill:var(--ink);font-size:9.5px;paint-order:stroke;stroke:var(--surface);stroke-width:2.6px;stroke-linejoin:round}
svg .axis-title{fill:var(--ink2);font-size:10.5px}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:11.5px;color:var(--ink2);margin-top:8px}
.legend .k{display:inline-flex;align-items:center;gap:5px}
.legend svg{overflow:visible}
.rampbar{display:flex;align-items:center;gap:6px;font-size:11px;color:var(--muted)}
.rampbar .strip{display:flex;height:9px;width:112px;border-radius:2px;overflow:hidden}
.rampbar .strip i{flex:1}

/* --- heatmap ---------------------------------------------------------- */
.heat{border-collapse:separate;border-spacing:2px;font-size:11px}
.heat th{position:static;background:transparent;border:0;padding:3px 5px;font-weight:500;color:var(--ink2)}
.heat thead th{writing-mode:vertical-rl;transform:rotate(180deg);text-align:left;height:104px;font-size:10.5px}
.heat thead th .split{color:var(--muted);font-size:9.5px}
.heat tbody th{text-align:right;white-space:nowrap;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:10.5px}
.heat td{border:0;padding:0;text-align:center;border-radius:3px;width:40px;height:21px;font-size:10px;cursor:default}
.heat td.hl,.heat th.hl{outline:1.5px solid var(--s2);outline-offset:-1.5px}
.heat th.hl{color:var(--s2)}

/* --- gallery ---------------------------------------------------------- */
.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px;margin-top:10px}
.gallery figure{margin:0}
.gallery img{width:100%;height:auto;border-radius:4px;border:1px solid var(--border);display:block;background:var(--plane)}
.gallery figcaption{font-size:10.5px;color:var(--ink2);margin-top:4px;line-height:1.35;font-family:ui-monospace,Menlo,Consolas,monospace}
.gallery figcaption .sub{display:block;color:var(--muted)}

/* --- glossary --------------------------------------------------------- */
dl{margin:0}
.gloss{break-inside:avoid;page-break-inside:avoid}
dt{margin:14px 0 3px;display:flex;align-items:baseline;gap:8px;flex-wrap:wrap}
.gloss:first-child dt{margin-top:4px}
.gloss-label{font-weight:600;font-size:13.5px}
.better{font-size:10.5px;color:var(--muted);letter-spacing:.03em}
dd{margin:0;color:var(--ink2);font-size:13.5px;max-width:74ch}

/* --- tooltip ---------------------------------------------------------- */
#tip{
  position:fixed;z-index:50;pointer-events:none;opacity:0;transition:opacity .08s;
  background:var(--ink);color:var(--surface);font-size:11.5px;line-height:1.5;
  padding:7px 10px;border-radius:6px;max-width:300px;box-shadow:0 4px 16px rgba(0,0,0,.22);
}
#tip code{background:color-mix(in srgb,var(--surface) 22%,transparent);color:inherit}
#tip b{color:inherit}
#tip table{font-size:11px;color:inherit;margin-top:4px;width:auto}
#tip td{border:0;padding:0 0 0 8px;color:inherit;white-space:nowrap}
#tip td:first-child{padding:0;opacity:.72}

/* --- toc -------------------------------------------------------------- */
nav.toc{font-size:12.5px;margin:18px 0 0;display:flex;flex-wrap:wrap;gap:4px 14px}
nav.toc a{color:var(--s1);text-decoration:none}
nav.toc a:hover{text-decoration:underline}
footer{margin-top:44px;padding-top:16px;border-top:1px solid var(--border);font-size:12px;color:var(--muted)}

/* --- print ------------------------------------------------------------ */
@media print{
  :root{
    --surface:#ffffff; --plane:#ffffff; --ink:#000000; --ink2:#333333;
    --muted:#666666; --grid:#d8d8d4; --border:#d8d8d4;
    --s1:#2a78d6; --s2:#c8501f; color-scheme:light;
  }
  @page{margin:14mm 13mm}
  body{background:#fff;font-size:10.5pt;line-height:1.5}
  .no-print{display:none !important}
  .wrap{max-width:none;padding:0}
  header{border:0;padding:0 0 10px;margin-bottom:14px;border-bottom:1px solid var(--border)}
  section{margin-top:18px;page-break-inside:auto}
  /* Only small blocks avoid a break. Making every .card unbreakable looks
     tidy until a card is taller than the space left on the page, at which
     point it jumps to the next one and leaves a third of a page blank. */
  .finding,.bar-row,.gallery figure,tr{page-break-inside:avoid;break-inside:avoid}
  .card{page-break-inside:auto;break-inside:auto}
  dl{column-count:2;column-gap:22px}
  dd{max-width:none}
  h2,h3{page-break-after:avoid;break-after:avoid}
  .table-wrap{overflow:visible;max-height:none}
  thead th{position:static}
  .sticky-col{position:static}
  table{font-size:8pt}
  th,td{padding:3px 5px}
  details.more{display:none}
  details[open] summary ~ *{display:revert}
  .abstract{border-left-width:2px}
  .gallery{grid-template-columns:repeat(5,1fr)}
  a[href^="http"]::after{content:" (" attr(href) ")";font-size:8pt;color:var(--muted)}
}
"""

_JS = r"""
(function(){
'use strict';
const D = window.__REPORT__;
const tip = document.getElementById('tip');

/* ---------- tooltips: one node, driven by data-tip ------------------- */
function showTip(html, x, y){
  tip.innerHTML = html;
  tip.style.opacity = '1';
  const r = tip.getBoundingClientRect();
  let left = x + 14, top = y + 14;
  if (left + r.width > window.innerWidth - 8) left = x - r.width - 14;
  if (top + r.height > window.innerHeight - 8) top = y - r.height - 14;
  tip.style.left = Math.max(8, left) + 'px';
  tip.style.top = Math.max(8, top) + 'px';
}
function hideTip(){ tip.style.opacity = '0'; }
document.addEventListener('mouseover', e => {
  const el = e.target.closest('[data-tip]');
  if (el) showTip(el.getAttribute('data-tip'), e.clientX, e.clientY);
});
document.addEventListener('mousemove', e => {
  const el = e.target.closest('[data-tip]');
  if (el) showTip(el.getAttribute('data-tip'), e.clientX, e.clientY);
  else hideTip();
});
document.addEventListener('mouseleave', hideTip);

/* ---------- formatting, shared with the server-rendered tables ------- */
function fmt(key, v){
  if (v === null || v === undefined || Number.isNaN(v)) return '\u2014';
  if (key === 'density') return (v*100).toFixed(0) + '%';
  if (key.indexOf('bad') === 0) return v.toFixed(1) + '%';
  if (key === 'wall_s' || key === 'cpu_s') return v < 1 ? v.toFixed(3) : v.toFixed(2);
  if (key === 'parallelism') return v.toFixed(1) + '\u00d7';
  if (key === 'rss_delta_mb' || key === 'peak_rss_mb') return v.toFixed(0);
  if (key === 'bias_valid') return (v>=0?'+':'') + v.toFixed(2);
  return v.toFixed(2);
}
function label(key){ const m = D.metrics[key]; return m ? m.label : key; }
function unit(key){ const m = D.metrics[key]; return m && m.unit ? m.unit : ''; }

/* ---------- sortable leaderboard ------------------------------------ */
(function(){
  const table = document.getElementById('lb-table');
  if (!table) return;
  const tbody = table.tBodies[0];
  const original = Array.from(tbody.rows);
  table.querySelectorAll('thead th').forEach((th, idx) => {
    th.addEventListener('click', () => {
      const numeric = th.dataset.type === 'num';
      const wasSorted = th.classList.contains('sorted');
      const asc = wasSorted ? !th.classList.contains('asc') : true;
      table.querySelectorAll('thead th').forEach(o => o.classList.remove('sorted','asc'));
      th.classList.add('sorted'); if (asc) th.classList.add('asc');
      const rows = Array.from(tbody.rows);
      rows.sort((a, b) => {
        const av = a.cells[idx].dataset.val, bv = b.cells[idx].dataset.val;
        if (numeric){
          const x = av === '' ? Infinity : parseFloat(av);
          const y = bv === '' ? Infinity : parseFloat(bv);
          return asc ? x - y : y - x;   /* blanks last either way */
        }
        return asc ? String(av).localeCompare(bv) : String(bv).localeCompare(av);
      });
      rows.forEach(r => tbody.appendChild(r));
    });
  });
  const reset = document.getElementById('reset-sort');
  if (reset) reset.addEventListener('click', () => {
    table.querySelectorAll('thead th').forEach(o => o.classList.remove('sorted','asc'));
    original.forEach(r => tbody.appendChild(r));
  });
})();

/* ---------- colour ramp --------------------------------------------- */
function rampColor(t){                 /* t in [0,1] -> sequential blue */
  if (!(t >= 0)) return 'transparent';
  const n = D.ramp.length;
  const i = Math.min(n - 1, Math.max(0, Math.round(t * (n - 1))));
  return D.ramp[i];
}
/* Ink on a ramp fill: the ramp darkens faster than its position suggests, so
   switch to the light colour before the midpoint. Both are literals, not
   theme tokens -- the fill comes from the light ramp in either theme, so a
   token would put the dark theme's pale ink on a pale cell. */
function rampInk(t){ return t > 0.45 ? '#fcfcfb' : '#0b0b0b'; }

/* ---------- scatter: accuracy vs time ------------------------------- */
const SHAPES = {
  circle: (x,y,s) => `<circle cx="${x}" cy="${y}" r="${s}"/>`,
  square: (x,y,s) => `<rect x="${x-s}" y="${y-s}" width="${2*s}" height="${2*s}" rx="1"/>`,
  triangle:(x,y,s)=> `<polygon points="${x},${y-s*1.15} ${x+s*1.1},${y+s*0.8} ${x-s*1.1},${y+s*0.8}"/>`,
  diamond:(x,y,s) => `<polygon points="${x},${y-s*1.2} ${x+s*1.2},${y} ${x},${y+s*1.2} ${x-s*1.2},${y}"/>`,
};
function drawScatter(){
  const yKey = document.getElementById('scatter-metric').value;
  const host = document.getElementById('scatter');
  const W = Math.max(560, host.clientWidth || 860), H = 430;
  const m = {l: 52, r: 18, t: 14, b: 46};
  const pts = D.algorithms
    .map(a => ({a, x: a.metrics.wall_s, y: a.metrics[yKey], d: a.metrics.density}))
    .filter(p => p.x !== null && p.y !== null && p.x > 0);
  if (!pts.length){ host.innerHTML = '<p class="caption">no data</p>'; return; }

  const xs = pts.map(p => p.x), ys = pts.map(p => p.y);
  const x0 = Math.min(...xs) / 2.2, x1 = Math.max(...xs) * 2.2;
  const pad = Math.max((Math.max(...ys) - Math.min(...ys)) * 0.12, 0.5);
  const y0 = Math.max(0, Math.min(...ys) - pad), y1 = Math.max(...ys) + pad;
  const sx = v => m.l + (Math.log10(v) - Math.log10(x0)) / (Math.log10(x1) - Math.log10(x0)) * (W - m.l - m.r);
  const sy = v => m.t + (1 - (v - y0) / (y1 - y0)) * (H - m.t - m.b);

  const lower = (D.metrics[yKey] || {}).better !== 'high';
  const parts = [];

  /* x gridlines at decades, y at 5 ticks */
  for (let e = Math.floor(Math.log10(x0)); e <= Math.ceil(Math.log10(x1)); e++){
    for (const mul of [1,2,5]){
      const v = mul * Math.pow(10, e);
      if (v < x0 || v > x1) continue;
      parts.push(`<line class="gridline" x1="${sx(v)}" y1="${m.t}" x2="${sx(v)}" y2="${H-m.b}"/>`);
      parts.push(`<text x="${sx(v)}" y="${H-m.b+15}" text-anchor="middle">${v < 1 ? v : v.toFixed(0)}</text>`);
    }
  }
  for (let i = 0; i <= 4; i++){
    const v = y0 + (y1 - y0) * i / 4;
    parts.push(`<line class="gridline" x1="${m.l}" y1="${sy(v)}" x2="${W-m.r}" y2="${sy(v)}"/>`);
    parts.push(`<text x="${m.l-7}" y="${sy(v)+3.5}" text-anchor="end">${fmt(yKey, v)}</text>`);
  }
  parts.push(`<text class="axis-title" x="${(m.l+W-m.r)/2}" y="${H-6}" text-anchor="middle">wall time per pair (s, log scale) \u2014 further left is faster</text>`);
  parts.push(`<text class="axis-title" transform="translate(13,${(m.t+H-m.b)/2}) rotate(-90)" text-anchor="middle">${label(yKey)} ${unit(yKey)} \u2014 ${lower ? 'lower' : 'higher'} is better</text>`);

  /* frontier, recomputed for whichever metric is on screen */
  const sorted = pts.slice().sort((p,q) => p.x - q.x);
  const front = [];
  let best = lower ? Infinity : -Infinity;
  for (const p of sorted){
    if (lower ? p.y < best - 1e-12 : p.y > best + 1e-12){ best = p.y; front.push(p); }
  }
  if (front.length > 1){
    parts.push('<path class="front-path" d="' +
      front.map((p,i) => (i ? 'L' : 'M') + sx(p.x) + ' ' + sy(p.y)).join(' ') + '"/>');
  }

  for (const p of pts){
    const t = p.d === null ? 0.5 : p.d;
    const shape = SHAPES[p.a.shape] || SHAPES.circle;
    const rows = D.chartMetrics
      .filter(k => p.a.metrics[k] !== null && p.a.metrics[k] !== undefined)
      .map(k => `<tr><td>${label(k)}</td><td>${fmt(k, p.a.metrics[k])}</td></tr>`).join('');
    const tipHtml = `<b>${p.a.name}</b><br>${p.a.recipe || p.a.family}` +
      (p.a.front ? ' &middot; <i>on the frontier</i>' : '') + `<table>${rows}</table>`;
    parts.push(`<g class="mark" fill="${rampColor(t)}" data-algo="${p.a.name}" data-tip="${tipHtml.replace(/"/g,'&quot;')}">${shape(sx(p.x), sy(p.y), 6.2)}</g>`);
  }
  /* Label the frontier only -- two dozen labels would be unreadable. Even
     seven collide at the accurate end, where the points bunch up, so after
     placing them (alternating corners, which separates on both axes at once)
     a second pass pushes any two that still overlap vertically apart. The
     labels carry a surface-coloured halo, so a small nudge is enough. */
  const placed = front.map((p, i) => {
    const right = i % 2 === 0;
    return {p, right, x: sx(p.x) + (right ? 10 : -10), y: sy(p.y) + (right ? -8 : 15)};
  });
  const MIN_GAP = 12;
  placed.slice().sort((a, b) => a.y - b.y).forEach((lab, i, all) => {
    if (i === 0) return;
    const prev = all[i - 1];
    /* only a real overlap: labels on opposite sides never collide */
    if (prev.right === lab.right && lab.y - prev.y < MIN_GAP) lab.y = prev.y + MIN_GAP;
  });
  placed.forEach(l => {
    parts.push(`<text class="label" x="${l.x}" y="${l.y}" text-anchor="${l.right ? 'start' : 'end'}">${l.p.a.name}</text>`);
  });

  host.innerHTML = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${label(yKey)} against wall time for each algorithm">${parts.join('')}</svg>`;
}

/* ---------- heatmap: algorithm x scene ------------------------------ */
function drawHeat(){
  const key = document.getElementById('heat-metric').value;
  const host = document.getElementById('heat');
  const mat = D.matrix[key];
  if (!mat){ host.innerHTML = '<p class="caption">not recorded in this run</p>'; return; }
  const algos = D.algorithms.map(a => a.name).filter(n => mat[n]);
  const scenes = D.scenes.filter(s => algos.some(a => mat[a][s.id] !== null && mat[a][s.id] !== undefined));
  const vals = [];
  for (const a of algos) for (const s of scenes){
    const v = mat[a][s.id]; if (v !== null && v !== undefined) vals.push(v);
  }
  if (!vals.length){ host.innerHTML = '<p class="caption">no data</p>'; return; }
  /* clip at the 95th percentile so one outlier does not flatten the ramp */
  const sortedV = vals.slice().sort((p,q) => p - q);
  const lo = sortedV[0];
  const hi = sortedV[Math.floor(0.95 * (sortedV.length - 1))] || sortedV[sortedV.length-1];
  const lower = (D.metrics[key] || {}).better !== 'high';
  const norm = v => {
    const t = (v - lo) / Math.max(hi - lo, 1e-9);
    return Math.min(1, Math.max(0, lower ? t : 1 - t));
  };

  let h = '<table class="heat"><thead><tr><th></th>';
  scenes.forEach((s, j) => {
    const split = s.split ? (s.split === 'tune' ? 'tune' : 'held') : '';
    h += `<th data-col="${j}">${s.short}${split ? ' <span class="split">\u00b7 ' + split + '</span>' : ''}</th>`;
  });
  h += '</tr></thead><tbody>';
  algos.forEach((a, i) => {
    h += `<tr><th data-row="${i}">${a}</th>`;
    scenes.forEach((s, j) => {
      const v = mat[a][s.id];
      if (v === null || v === undefined){
        h += `<td data-row="${i}" data-col="${j}" style="background:transparent;color:var(--muted)">\u2014</td>`;
      } else {
        const t = norm(v);
        const tipHtml = `<b>${a}</b><br>${s.short}<br>${label(key)} <b>${fmt(key, v)}</b>`;
        h += `<td data-row="${i}" data-col="${j}" style="background:${rampColor(t)};color:${rampInk(t)}" data-tip="${tipHtml.replace(/"/g,'&quot;')}">${fmt(key, v).replace('%','')}</td>`;
      }
    });
    h += '</tr>';
  });
  h += '</tbody></table>';
  const ramp = D.ramp.map(c => `<i style="background:${c}"></i>`).join('');
  h += `<div class="caption"><span class="rampbar">${lower ? 'better' : 'worse'}
        <span class="strip">${ramp}</span> ${lower ? 'worse' : 'better'}
        &nbsp;(${fmt(key, lo)} \u2013 ${fmt(key, hi)}, clipped at the 95th percentile)</span></div>`;
  host.innerHTML = h;

  host.querySelectorAll('[data-row]').forEach(cell => {
    cell.addEventListener('mouseenter', () => {
      const r = cell.dataset.row, c = cell.dataset.col;
      host.querySelectorAll('.hl').forEach(e => e.classList.remove('hl'));
      host.querySelectorAll(`[data-row="${r}"]`).forEach(e => e.classList.add('hl'));
      if (c !== undefined) host.querySelectorAll(`[data-col="${c}"]`).forEach(e => e.classList.add('hl'));
    });
  });
  host.addEventListener('mouseleave', () => {
    host.querySelectorAll('.hl').forEach(e => e.classList.remove('hl'));
  });
}

/* ---------- wiring --------------------------------------------------- */
function fillSelect(id, metrics, selected){
  const sel = document.getElementById(id);
  if (!sel) return;
  sel.innerHTML = metrics.map(k =>
    `<option value="${k}"${k === selected ? ' selected' : ''}>${label(k)}${unit(k) ? ' (' + unit(k).replace('&times;','\u00d7') + ')' : ''}</option>`
  ).join('');
}
const accuracyish = D.chartMetrics.filter(k => (D.metrics[k]||{}).group === 'accuracy');
fillSelect('scatter-metric', accuracyish, D.primary);
fillSelect('heat-metric', D.chartMetrics, D.primary);
document.getElementById('scatter-metric').addEventListener('change', drawScatter);
document.getElementById('heat-metric').addEventListener('change', drawHeat);
drawScatter(); drawHeat();
let t = null;
window.addEventListener('resize', () => { clearTimeout(t); t = setTimeout(drawScatter, 150); });

/* theme toggle: explicit, so a dark-mode reader can force light for printing */
const btn = document.getElementById('theme');
if (btn) btn.addEventListener('click', () => {
  const root = document.documentElement;
  const now = root.getAttribute('data-theme');
  const dark = now ? now === 'dark'
                   : window.matchMedia('(prefers-color-scheme: dark)').matches;
  root.setAttribute('data-theme', dark ? 'light' : 'dark');
  drawHeat();
});
const pb = document.getElementById('printbtn');
if (pb) pb.addEventListener('click', () => {
  document.querySelectorAll('details').forEach(d => d.open = true);
  window.print();
});
})();
"""


def _section(num: str, anchor: str, title: str, sub: str, body: str) -> str:
    return (
        f'<section id="{anchor}"><div class="sec-head">'
        f'<span class="sec-num">{num}</span><h2>{title}</h2>'
        + (f"<p>{sub}</p>" if sub else "")
        + f"</div>{body}</section>"
    )


def render(
    f: Findings, *, gallery_scenes: int = 2, gallery_algorithms: int = 6
) -> str:
    """Build the whole page as one string."""
    meta = f.run.meta
    machine, cfg = meta.get("machine", {}), meta.get("config", {})
    payload = json.dumps(_payload(f)).replace("<", "\\u003c")
    gallery = _gallery(f, gallery_scenes, gallery_algorithms)

    chips = [
        ("run", f.run.run_dir.name),
        ("when", str(meta.get("created_utc", "unknown"))[:19].replace("T", " ") + " UTC"),
        ("cells", f"{meta.get('n_ok', '?')} of {meta.get('n_cells', '?')} succeeded"),
        ("scenes", str(f.n_scenes)),
        ("algorithms", str(f.n_algorithms)),
        ("scored on", f"the <code>{cfg.get('eval_region', '?')}</code> region"),
        ("timing", f"median of {cfg.get('repeats', '?')} run(s), "
                   f"{cfg.get('warmup', '?')} warmup"),
        ("threads", str(cfg.get("threads") or "library default")),
        ("cpu", str(machine.get("cpu_brand") or machine.get("processor", "?"))),
        ("gpu", str(machine.get("gpu") or "none (CUDA unavailable)")),
    ]
    chip_html = "".join(
        f'<span class="chip">{esc(k)} <b>{v if k in ("scored on",) else esc(v)}</b></span>'
        for k, v in chips
    )

    families = sorted(set(f.lb["family"])) if not f.lb.empty else []
    fam_html = "".join(
        f"<li><b>{esc(fam)}</b> &mdash; {FAMILY_BLURB.get(fam, '')} "
        f'<span class="muted">(marker: {FAMILY_SHAPE.get(fam, "circle")})</span></li>'
        for fam in families
    )

    toc = [
        ("summary", "Summary"),
        ("findings", "What the numbers mean"),
        ("metrics", "How to read the metrics"),
        ("leaderboard", "Leaderboard"),
        ("tradeoff", "Accuracy vs time"),
        ("why", "What moved the numbers"),
        ("photometric", "Exposure robustness"),
        ("scenes", "Scene by scene"),
        ("cost", "Compute cost"),
        ("maps", "Looking at the output"),
        ("caveats", "What this does not show"),
        ("appendix", "Appendix"),
    ]
    if not gallery:
        toc = [t for t in toc if t[0] != "maps"]
    if not f.photometric:
        toc = [t for t in toc if t[0] != "photometric"]
    toc_html = "".join(f'<a href="#{a}">{esc(t)}</a>' for a, t in toc)

    # --- compute-cost block ------------------------------------------------
    cost_rows = (
        f.lb.sort_values("cpu_s", ascending=False).head(14).to_dict("records")
        if "cpu_s" in f.lb.columns else []
    )
    cost_html = ""
    if cost_rows:
        top = max(
            max(_num(r.get("cpu_s")) or 0 for r in cost_rows),
            max(_num(r.get("wall_s")) or 0 for r in cost_rows),
        )
        pairs = []
        for r in cost_rows:
            pairs.append(
                '<div class="bar-row">'
                f'<div class="bar-label"><b>{esc(r["algorithm"])}</b>'
                f'<span class="sub">{(_num(r.get("parallelism")) or 0):.1f}&times; cores busy</span></div>'
                '<div class="bar-track">'
                f'<div class="bar-fill s2" style="width:{100 * (_num(r.get("cpu_s")) or 0) / max(top, 1e-9):.2f}%" '
                f'data-tip="<b>{esc(r["algorithm"])}</b><br>CPU {(_num(r.get("cpu_s")) or 0):.2f} s across all cores"></div>'
                f'<div class="bar-fill s1" style="width:{100 * (_num(r.get("wall_s")) or 0) / max(top, 1e-9):.2f}%;height:7px;top:auto;bottom:0;border-bottom-width:0" '
                f'data-tip="<b>{esc(r["algorithm"])}</b><br>wall {(_num(r.get("wall_s")) or 0):.2f} s elapsed"></div>'
                "</div>"
                f'<div class="bar-value">{(_num(r.get("cpu_s")) or 0):.1f}</div></div>'
            )
        cost_html = (
            '<div class="card"><h3>What each method actually consumes</h3>'
            '<p class="lede">The thin bar is wall time &mdash; what you wait. '
            "The thick bar is CPU time &mdash; what you pay for, summed over "
            "every core. Where the thick bar dwarfs the thin one, the method "
            "bought its speed by occupying the whole machine, and will not keep "
            "that speed when something else wants a core.</p>"
            '<div class="bars">' + "".join(pairs) + "</div>"
            '<p class="caption">'
            '<span class="key"><span class="swatch s2"></span>CPU-seconds</span>'
            '<span class="key"><span class="swatch s1"></span>wall seconds</span>'
            "Ordered by CPU time, top 14. Values in CPU-seconds.</p></div>"
        )

    # --- caveats ------------------------------------------------------------
    repeats = cfg.get("repeats", 1)
    caveats = [
        (
            "Timing is only comparable <i>within</i> this run",
            f"Every cell here was timed as the median of {repeats} run(s) on one "
            "machine in one session. A laptop that has warmed up and dropped its "
            "clocks drifts by more than the differences between neighbouring "
            "rows, and two runs of this benchmark hours apart have been measured "
            "to differ by over 2&times; on identical code. Ratios inside this "
            "table are meaningful; absolute seconds quoted against another run "
            "are not."
            + ("" if (repeats or 1) > 2 else " With a single repeat there is no "
               "variance estimate at all &mdash; re-run with <code>--repeats 5</code> "
               "before quoting any individual number."),
        ),
        (
            "Memory is an under-estimate",
            "<code>RSS growth</code> is sampled on a background thread, so a "
            "short-lived peak between samples is missed, and a method reusing "
            "memory the allocator already holds appears to allocate nothing. "
            "<code>process peak</code> carries everything earlier cells "
            "allocated and is not this method's footprint. Compare within a "
            "family first.",
        ),
        (
            "Every pair here is already rectified",
            "Matching only ever searches along a row. Rectification &mdash; and "
            "the calibration behind it &mdash; is assumed perfect and is not "
            "measured, yet it is where a real system's error usually comes from. "
            "A method that wins here can still lose on a rig whose epipolar "
            "lines are a pixel off.",
        ),
        (
            "Indoor scenes, one dataset",
            "These are controlled Middlebury captures: static, textured, "
            "evenly lit, indoors. Nothing here predicts behaviour on a wet road "
            "at night, on a specular surface, or on anything moving. The "
            "<i>ordering</i> of methods is reasonably stable across datasets; "
            "the absolute numbers are not.",
        ),
        (
            "Defaults were tuned, and tuning can overfit",
            "Per-algorithm defaults in this project came from a grid search "
            "restricted to a tune split, with the remaining scenes held out. "
            "This run scores both splits together, which is the honest headline "
            "number but mixes scenes the parameters saw with scenes they did "
            "not. The scene table marks which is which.",
        ),
        (
            "A failed method is still a method",
            "Rows that did not converge are kept in the table and flagged. "
            "Dropping them would raise every average and make the field look "
            "tighter than it is.",
        ),
    ]
    caveat_html = '<div class="grid2">' + "".join(
        f'<div class="card"><h3>{t}</h3><p class="lede">{b}</p></div>'
        for t, b in caveats
    ) + "</div>"

    title = f"Stereo benchmark \u2014 {f.run.run_dir.name}"
    return f"""<!doctype html>
<html lang="en" >
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)}</title>
<meta name="description" content="What {f.n_algorithms} stereo matching methods scored on {f.n_scenes} scenes, and what the numbers mean.">
<style>{_CSS}</style>
</head>
<body>
<div id="tip" role="tooltip"></div>
<header>
  <div class="wrap">
    <p class="eyebrow">Stereo vision capstone &middot; benchmark report</p>
    <h1>{f.n_algorithms} stereo matching methods, {f.n_scenes} scenes,<br>and what separated them</h1>
    <div class="chips">{chip_html}</div>
    <div class="toolbar no-print">
      <button id="printbtn">Save as PDF</button>
      <button id="theme">Toggle dark mode</button>
      <span class="muted" style="font-size:12px">Charts and tables are interactive &mdash; hover a mark or a column header, click a header to sort.</span>
    </div>
    <nav class="toc">{toc_html}</nav>
  </div>
</header>

<div class="wrap">

{_section("01", "summary", "Summary", "", f'<div class="abstract">{f.abstract}</div>')}

{_section("02", "findings", "What the numbers mean",
          "Each statement below is derived from this run's data, not written in "
          "advance. The supporting measurement is given under each one.",
          _notes_html(f))}

{_section("03", "metrics", "How to read the metrics",
          "A stereo result cannot be summarised by one number, because "
          "&quot;wrong&quot; has at least three meanings: how often, how far, "
          "and how much of the image was answered at all.",
          '<div class="card">' + _glossary(f) + '</div>'
          + '<div class="card"><h3>The families</h3><ul class="lede">'
          + fam_html + "</ul></div>")}

{_section("04", "leaderboard", "Leaderboard",
          f"Every algorithm, averaged over the {f.n_scenes} scenes it completed. "
          "Click any header to sort; hover one for what the column means.",
          '<div class="chart-head no-print"><span class="muted" style="font-size:12px">'
          'Sorted by bad 2px. Blanks sort last.</span>'
          '<button id="reset-sort">reset order</button></div>'
          + _table_html(f))}

{_section("05", "tradeoff", "Accuracy vs time",
          "Lower and further left is better. Marker shape is the family; fill is "
          "how much of the image the method answered, which is the third thing "
          "you need to read a stereo result honestly.",
          '<div class="card"><div class="chart-head">'
          '<h3 style="margin:0">The trade-off</h3>'
          '<label class="no-print" style="font-size:12.5px;color:var(--ink2)">'
          'y axis <select id="scatter-metric"></select></label></div>'
          '<div id="scatter"></div>'
          '<div class="legend">'
          + "".join(
              f'<span class="k"><svg width="13" height="13" viewBox="0 0 13 13">'
              f'<g fill="var(--muted)">'
              + {
                  "circle": '<circle cx="6.5" cy="6.5" r="5"/>',
                  "square": '<rect x="1.5" y="1.5" width="10" height="10" rx="1"/>',
                  "triangle": '<polygon points="6.5,1 12,11.5 1,11.5"/>',
                  "diamond": '<polygon points="6.5,0.8 12.2,6.5 6.5,12.2 0.8,6.5"/>',
              }.get(FAMILY_SHAPE.get(fam, "circle"), "")
              + f"</g></svg>{esc(fam)}</span>"
              for fam in families
          )
          + '<span class="k"><svg width="22" height="8"><line x1="0" y1="4" x2="22" y2="4" stroke="var(--s2)" stroke-width="2"/></svg>Pareto frontier</span>'
          + '<span class="rampbar">0% answered<span class="strip">'
          + "".join(f'<i style="background:{c}"></i>' for c in BLUE_RAMP)
          + "</span>100%</span></div>"
          '<p class="caption">The frontier is recomputed for whichever metric is '
          "on the y axis, so switching to <code>bad 2px, answered</code> shows a "
          "different &mdash; and more flattering &mdash; set of winners. That the "
          "set changes is the point.</p></div>")}

{_section("06", "why", "What moved the numbers",
          "A leaderboard cannot tell you <i>why</i> one method beat another, "
          "because several things changed at once. These comparisons change one "
          "thing and hold the rest fixed.",
          _contrast_section(f))}

{_section("07", "photometric", "Exposure robustness", "",
          _photometric_section(f)) if f.photometric else ""}

{_section("08", "scenes", "Scene by scene",
          "An average over scenes hides a method that is merely mediocre "
          "everywhere behind one that is excellent except where it collapses.",
          '<div class="card"><div class="chart-head">'
          '<h3 style="margin:0">Every method on every scene</h3>'
          '<label class="no-print" style="font-size:12.5px;color:var(--ink2)">'
          'metric <select id="heat-metric"></select></label></div>'
          '<div id="heat" style="overflow:auto"></div></div>'
          + '<div class="card"><h3>Scene difficulty</h3>'
          '<p class="lede">The <b>spread</b> column is the one to read: it is '
          "how much your choice of method matters on that scene. A scene with a "
          "small spread is telling you that this decision does not matter "
          "there.</p>" + _scene_table(f) + "</div>")}

{_section("09", "cost", "Compute cost",
          "Three different questions: how long you wait, how much processor you "
          "consume, and how much memory it takes.", cost_html)}

{_section("10", "maps", "Looking at the output",
          "Numbers rank; pictures explain. Each method's failure has a "
          "characteristic look &mdash; horizontal streaks from a row-independent "
          "solver, speckle from unaggregated winner-takes-all, black holes from "
          "a consistency check that discarded what it could not verify.",
          _gallery_html(gallery)) if gallery else ""}

{_section("11", "caveats", "What this does not show",
          "The limits of the experiment, stated so a number from it is not "
          "quoted further than it can carry.", caveat_html)}

{_section("12", "appendix", "Appendix", "Everything needed to reproduce the run.",
          '<div class="card">' + _appendix(f) + "</div>")}

<footer>
  Generated from <code>{esc(f.run.run_dir.as_posix())}</code> by
  <code>python -m bench explain</code>. The prose is assembled from this run's
  own measurements, so re-running the benchmark and regenerating this page will
  change the sentences, not just the tables. Source data:
  <code>results.csv</code>, <code>run.json</code>.
</footer>
</div>
<script>window.__REPORT__ = {payload};</script>
<script>{_JS}</script>
</body>
</html>
"""


# --------------------------------------------------------------------------
# PDF
# --------------------------------------------------------------------------

_BROWSERS = (
    "chrome", "google-chrome", "chromium", "msedge",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
)


def find_browser() -> str | None:
    """A Chromium-family binary that can print to PDF, or ``None``."""
    for cand in _BROWSERS:
        if Path(cand).exists():
            return cand
        found = shutil.which(cand)
        if found:
            return found
    return None


def to_pdf(html_path: Path, pdf_path: Path | None = None) -> Path:
    """Print the generated page to PDF with a headless Chromium.

    There is no Python PDF engine in this project's dependencies, and adding
    one would mean maintaining a second rendering path that drifts from the
    HTML. Driving the browser that already renders the page correctly keeps
    one source of truth: the print stylesheet decides what the PDF looks like.
    """
    html_path = Path(html_path).resolve()
    pdf_path = Path(pdf_path) if pdf_path else html_path.with_suffix(".pdf")
    browser = find_browser()
    if not browser:
        raise RuntimeError(
            "No Chrome, Chromium or Edge found to print with. The HTML report "
            f"is at {html_path} -- open it and use the 'Save as PDF' button, "
            "which prints through the same stylesheet."
        )
    cmd = [
        browser, "--headless=new", "--disable-gpu", "--no-sandbox",
        "--no-pdf-header-footer",
        "--virtual-time-budget=8000",
        f"--print-to-pdf={pdf_path}",
        html_path.as_uri(),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if not pdf_path.exists():
        raise RuntimeError(
            f"{Path(browser).name} did not produce a PDF "
            f"(exit {proc.returncode}). stderr:\n{proc.stderr[-1500:]}"
        )
    return pdf_path
