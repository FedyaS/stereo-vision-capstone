/* An interactive version of figures/accuracy_vs_time.png.
 *
 * The PNG from bench/report.py answers exactly one question: bad 2.0 px
 * against wall time, averaged over scenes. Every other trade-off in the same
 * table -- error against throughput, against memory, against density, or on
 * one scene instead of the mean -- needed a code edit and a re-run. This
 * component plots any metric against any metric from the rows the run already
 * returned, so the whole results.csv is reachable from the controls.
 *
 * Design follows the project's data-viz reference palette, like the PNG does:
 * family identity is carried by marker shape *and* hue (both validated for
 * colour-vision deficiency against the light and dark surfaces), orange is
 * reserved for the Pareto frontier, magnitude colouring uses the single-hue
 * blue ramp, and the leaderboard table above is the table view that the one
 * sub-3:1 contrast slot (torch green on light) obliges us to keep.
 */
import { useEffect, useMemo, useRef, useState } from 'react'
import type { RunRow } from './types'

/* ------------------------------------------------------------------ metrics */

type Dir = 'lower' | 'higher'

export interface MetricMeta {
  key: string
  /** Shown in the pickers and the tooltip. */
  label: string
  group: 'accuracy' | 'speed' | 'resources'
  unit: string
  /** Raw value is multiplied by this before display (density is 0..1). */
  mul: number
  /** Which way is better; undefined means "no direction", so no frontier. */
  better?: Dir
  /** Scale this metric gets when it is picked. */
  scale: 'log' | 'linear'
}

const m = (
  key: string,
  label: string,
  group: MetricMeta['group'],
  unit: string,
  better?: Dir,
  scale: 'log' | 'linear' = 'linear',
  mul = 1,
): MetricMeta => ({ key, label, group, unit, better, scale, mul })

/** Every numeric column bench/runner.py writes that is worth an axis. */
export const METRICS: MetricMeta[] = [
  // accuracy -- "_all" counts blank pixels as errors, "_valid" does not
  m('bad2_all', 'bad 2.0 px, blanks = errors', 'accuracy', '%', 'lower'),
  m('bad1_all', 'bad 1.0 px, blanks = errors', 'accuracy', '%', 'lower'),
  m('bad0p5_all', 'bad 0.5 px, blanks = errors', 'accuracy', '%', 'lower'),
  m('bad4_all', 'bad 4.0 px, blanks = errors', 'accuracy', '%', 'lower'),
  m('bad2_valid', 'bad 2.0 px, answered only', 'accuracy', '%', 'lower'),
  m('bad1_valid', 'bad 1.0 px, answered only', 'accuracy', '%', 'lower'),
  m('epe_valid', 'EPE, answered only', 'accuracy', 'px', 'lower'),
  m('epe_filled', 'EPE, hole-filled', 'accuracy', 'px', 'lower'),
  m('rmse_valid', 'RMSE, answered only', 'accuracy', 'px', 'lower'),
  m('rmse_filled', 'RMSE, hole-filled', 'accuracy', 'px', 'lower'),
  m('a90', '90th percentile error', 'accuracy', 'px', 'lower'),
  m('a95', '95th percentile error', 'accuracy', 'px', 'lower'),
  m('a99', '99th percentile error', 'accuracy', 'px', 'lower'),
  m('bias_valid', 'signed bias', 'accuracy', 'px'),
  m('density', 'density of estimated pixels', 'accuracy', '%', 'higher', 'linear', 100),
  // speed
  m('wall_s', 'wall time per pair', 'speed', 's', 'lower', 'log'),
  m('wall_s_min', 'fastest wall time', 'speed', 's', 'lower', 'log'),
  m('cpu_s', 'cpu time per pair', 'speed', 's', 'lower', 'log'),
  m('mdps', 'throughput, disparity hypotheses', 'speed', 'M/s', 'higher', 'log'),
  m('mpix_per_s', 'throughput, pixels', 'speed', 'MP/s', 'higher', 'log'),
  m('parallelism', 'cores busy', 'speed', 'x', 'higher'),
  // resources
  m('peak_rss_mb', 'peak RSS', 'resources', 'MB', 'lower', 'log'),
  m('rss_delta_mb', 'RSS growth', 'resources', 'MB', 'lower', 'log'),
  m('gpu_peak_mb', 'peak GPU memory', 'resources', 'MB', 'lower', 'log'),
]

const BY_KEY = new Map(METRICS.map((x) => [x.key, x]))

/** One click each for the trade-offs worth looking at first. */
const PRESETS: [string, string, string][] = [
  ['accuracy vs time', 'wall_s', 'bad2_all'],
  ['accuracy vs throughput', 'mpix_per_s', 'bad2_all'],
  ['error vs density', 'density', 'epe_valid'],
  ['accuracy vs memory', 'peak_rss_mb', 'bad2_all'],
]

const FAMILIES = ['opencv', 'scratch', 'torch', 'learned'] as const
const SHAPES: Record<string, Shape> = {
  opencv: 'circle', scratch: 'square', torch: 'triangle', learned: 'diamond',
}
type Shape = 'circle' | 'square' | 'triangle' | 'diamond'

const ALL_SCENES = '__mean__'

/* -------------------------------------------------------------------- theme */

/* Colours are read off the stylesheet rather than hard-coded so the chart
   tracks light/dark with the rest of the app -- and so the exported SVG has
   real hex values in it instead of var() references that resolve nowhere. */
function readVars() {
  const s = getComputedStyle(document.documentElement)
  const g = (n: string, fallback: string) => s.getPropertyValue(n).trim() || fallback
  return {
    opencv: g('--family-opencv', '#2a78d6'),
    scratch: g('--family-scratch', '#eb6834'),
    torch: g('--family-torch', '#1baf7a'),
    learned: g('--family-learned', '#6250d6'),
    surface: g('--surface-1', '#fcfcfb'),
    border: g('--border', '#dddbd3'),
    borderStrong: g('--border-strong', '#c8c6bc'),
    ink: g('--text-primary', '#0b0b0b'),
    ink2: g('--text-secondary', '#52514e'),
    muted: g('--text-muted', '#8a8982'),
    frontier: g('--warn', '#eb6834'),
    dark:
      document.documentElement.dataset.theme !== 'light' &&
      window.matchMedia('(prefers-color-scheme: dark)').matches,
  }
}
type Vars = ReturnType<typeof readVars>

function useThemeVars(): Vars {
  const [v, setV] = useState<Vars>(readVars)
  useEffect(() => {
    const mq = window.matchMedia('(prefers-color-scheme: dark)')
    const on = () => setV(readVars())
    mq.addEventListener('change', on)
    return () => mq.removeEventListener('change', on)
  }, [])
  return v
}

/* Single-hue blue ramp, steps 100..700 of the reference palette -- the same
   ramp bench/report.py colours its scatter with. Reversed on a dark surface so
   that lightness still rises with magnitude. */
const BLUE = [
  '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b',
]
function rampColor(t: number, dark: boolean): string {
  const stops = dark ? [...BLUE].reverse() : BLUE
  const u = Math.min(1, Math.max(0, t)) * (stops.length - 1)
  const i = Math.floor(u)
  const a = stops[i]
  const b = stops[Math.min(stops.length - 1, i + 1)]
  const f = u - i
  const mix = (o: number) => {
    const ca = parseInt(a.slice(1 + o, 3 + o), 16)
    const cb = parseInt(b.slice(1 + o, 3 + o), 16)
    return Math.round(ca + (cb - ca) * f).toString(16).padStart(2, '0')
  }
  return `#${mix(0)}${mix(2)}${mix(4)}`
}

/* ------------------------------------------------------------------- numbers */

const num = (v: unknown): number | null =>
  typeof v === 'number' && Number.isFinite(v) ? v : null

function mean(rows: RunRow[], key: string): number | null {
  let sum = 0
  let n = 0
  for (const r of rows) {
    const v = num(r[key])
    if (v != null) { sum += v; n += 1 }
  }
  return n ? sum / n : null
}

/** Compact, scale-aware formatting: 0.0014 and 470 both stay readable. */
function sig(v: number): string {
  const a = Math.abs(v)
  if (a === 0) return '0'
  if (a >= 1000) return v.toFixed(0)
  if (a >= 100) return v.toFixed(0)
  if (a >= 10) return v.toFixed(1)
  if (a >= 1) return v.toFixed(2)
  if (a >= 0.01) return v.toFixed(3)
  if (a >= 0.001) return v.toFixed(4)
  return v.toExponential(1)
}
const withUnit = (v: number, mm: MetricMeta) =>
  `${sig(v)}${mm.unit === '%' || mm.unit === 'x' ? '' : ' '}${mm.unit}`

/* --------------------------------------------------------------------- ticks */

function linTicks(lo: number, hi: number, target = 6): number[] {
  if (!(hi > lo)) return [lo]
  const raw = (hi - lo) / target
  const mag = Math.pow(10, Math.floor(Math.log10(raw)))
  const norm = raw / mag
  const step = (norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1) * mag
  const out: number[] = []
  for (let t = Math.ceil(lo / step) * step; t <= hi + step * 1e-9; t += step) {
    out.push(Math.abs(t) < step * 1e-9 ? 0 : t)
  }
  return out
}

function logTicks(lo: number, hi: number): number[] {
  if (!(hi > lo) || lo <= 0) return [lo]
  const decades = Math.log10(hi) - Math.log10(lo)
  const mantissas = decades > 4 ? [1] : decades > 2 ? [1, 3] : decades > 1 ? [1, 2, 5] : [1, 2, 3, 5, 7]
  const out: number[] = []
  for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e += 1) {
    for (const mn of mantissas) {
      const t = mn * Math.pow(10, e)
      if (t >= lo && t <= hi) out.push(t)
    }
  }
  return out.length >= 2 ? out : [lo, hi]
}

/* -------------------------------------------------------------------- points */

interface Pt {
  id: string
  algorithm: string
  family: string
  n: number
  vals: Record<string, number | null>
}

/** Points not beaten on both axes at once -- the only rows worth defending. */
function paretoIds(pts: Pt[], xm: MetricMeta, ym: MetricMeta): Set<string> {
  const out = new Set<string>()
  if (!xm.better || !ym.better) return out
  const good = pts.filter((p) => p.vals[xm.key] != null && p.vals[ym.key] != null)
  const xSign = xm.better === 'lower' ? 1 : -1
  const ySign = ym.better === 'lower' ? 1 : -1
  good.sort((a, b) => xSign * (a.vals[xm.key]! - b.vals[xm.key]!))
  let best = Infinity
  for (const p of good) {
    const y = ySign * p.vals[ym.key]!
    if (y < best - 1e-12) { best = y; out.add(p.id); }
  }
  return out
}

/* ----------------------------------------------------------------- component */

export default function TradeoffChart({ rows }: { rows: RunRow[] }) {
  const v = useThemeVars()

  const okRows = useMemo(
    () => rows.filter((r) => String(r.status ?? 'ok') === 'ok'),
    [rows],
  )

  /** Only offer axes the run actually produced numbers for. */
  const available = useMemo(
    () => METRICS.filter((mm) => okRows.some((r) => num(r[mm.key]) != null)),
    [okRows],
  )
  const has = (k: string) => available.some((mm) => mm.key === k)

  const scenes = useMemo(() => {
    const s = new Set<string>()
    for (const r of okRows) if (r.scene != null) s.add(String(r.scene))
    return Array.from(s).sort()
  }, [okRows])

  const [xKey, setXKey] = useState('wall_s')
  const [yKey, setYKey] = useState('bad2_all')
  const [xScale, setXScale] = useState<'log' | 'linear'>('log')
  const [yScale, setYScale] = useState<'log' | 'linear'>('linear')
  const [scene, setScene] = useState<string>(ALL_SCENES)
  const [colorBy, setColorBy] = useState<'family' | 'density'>('family')
  const [showFrontier, setShowFrontier] = useState(true)
  const [labelMode, setLabelMode] = useState<'frontier' | 'all' | 'none'>('frontier')
  const [hiddenFams, setHiddenFams] = useState<string[]>([])
  const [hiddenAlgos, setHiddenAlgos] = useState<string[]>([])
  const [hover, setHover] = useState<string | null>(null)
  const [pinned, setPinned] = useState<string[]>([])

  /* Fall back to whatever the run does have if the defaults are missing. */
  useEffect(() => {
    if (!available.length) return
    if (!has(xKey)) {
      setXKey((available.find((mm) => mm.group === 'speed') ?? available[0]).key)
    }
    if (!has(yKey)) {
      setYKey((available.find((mm) => mm.group === 'accuracy') ?? available[0]).key)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [available])

  const xm = BY_KEY.get(xKey) ?? METRICS[0]
  const ym = BY_KEY.get(yKey) ?? METRICS[0]

  const applyPreset = (px: string, py: string) => {
    setXKey(px)
    setYKey(py)
    setXScale(BY_KEY.get(px)?.scale ?? 'linear')
    setYScale(BY_KEY.get(py)?.scale ?? 'linear')
  }
  const pickX = (k: string) => { setXKey(k); setXScale(BY_KEY.get(k)?.scale ?? 'linear') }
  const pickY = (k: string) => { setYKey(k); setYScale(BY_KEY.get(k)?.scale ?? 'linear') }

  /* One point per algorithm: the mean over scenes, or a single scene's row. */
  const allPts = useMemo<Pt[]>(() => {
    const sel = scene === ALL_SCENES
      ? okRows
      : okRows.filter((r) => String(r.scene) === scene)
    const by = new Map<string, RunRow[]>()
    for (const r of sel) {
      const k = String(r.algorithm ?? '?')
      const list = by.get(k)
      if (list) list.push(r)
      else by.set(k, [r])
    }
    return Array.from(by.entries()).map(([algorithm, rs]) => ({
      id: algorithm,
      algorithm,
      family: String(rs[0].family ?? ''),
      n: rs.length,
      vals: Object.fromEntries(
        METRICS.map((mm) => [mm.key, mean(rs, mm.key)]),
      ) as Record<string, number | null>,
    })).sort((a, b) => a.algorithm.localeCompare(b.algorithm))
  }, [okRows, scene])

  const famsPresent = useMemo(
    () => FAMILIES.filter((f) => allPts.some((p) => p.family === f)),
    [allPts],
  )

  const kept = useMemo(
    () => allPts.filter(
      (p) => !hiddenFams.includes(p.family) && !hiddenAlgos.includes(p.id),
    ),
    [allPts, hiddenFams, hiddenAlgos],
  )
  /* A point with a null on either axis cannot be placed -- say so, don't drop
     it silently. */
  const pts = useMemo(
    () => kept.filter((p) => p.vals[xKey] != null && p.vals[yKey] != null),
    [kept, xKey, yKey],
  )
  const nUnplottable = kept.length - pts.length

  /* Log needs strictly positive data on that axis; otherwise force linear. */
  const xLogOk = pts.every((p) => (p.vals[xKey] ?? 0) > 0) && pts.length > 0
  const yLogOk = pts.every((p) => (p.vals[yKey] ?? 0) > 0) && pts.length > 0
  const xLog = xScale === 'log' && xLogOk
  const yLog = yScale === 'log' && yLogOk

  const front = useMemo(
    () => (showFrontier ? paretoIds(pts, xm, ym) : new Set<string>()),
    [pts, xm, ym, showFrontier],
  )

  /* ------------------------------------------------------------- geometry */

  const wrapRef = useRef<HTMLDivElement>(null)
  const svgRef = useRef<SVGSVGElement>(null)
  const [width, setWidth] = useState(760)
  useEffect(() => {
    const el = wrapRef.current
    if (!el) return
    const ro = new ResizeObserver(([e]) => setWidth(e.contentRect.width))
    ro.observe(el)
    setWidth(el.clientWidth)
    return () => ro.disconnect()
  }, [])

  const H = 430
  const pad = { t: 16, r: 16, b: 46, l: 66 }
  const W = Math.max(320, width)
  const iw = Math.max(40, W - pad.l - pad.r)
  const ih = H - pad.t - pad.b

  const domain = (key: string, log: boolean): [number, number] => {
    const vs = pts.map((p) => p.vals[key]!).filter((n) => n != null)
    if (!vs.length) return log ? [1, 10] : [0, 1]
    let lo = Math.min(...vs)
    let hi = Math.max(...vs)
    if (log) {
      if (hi / lo < 1.2) { lo /= 2; hi *= 2 }
      return [lo / 1.6, hi * 1.6]
    }
    if (hi === lo) { const d = Math.abs(hi) || 1; return [lo - d * 0.5, hi + d * 0.5] }
    const span = hi - lo
    return [lo - span * 0.09, hi + span * 0.12]
  }

  const [x0, x1] = domain(xKey, xLog)
  const [y0, y1] = domain(yKey, yLog)
  const mulX = xm.mul
  const mulY = ym.mul

  const sx = (raw: number) => {
    const a = xLog ? Math.log10(x0) : x0
    const b = xLog ? Math.log10(x1) : x1
    const t = ((xLog ? Math.log10(raw) : raw) - a) / (b - a || 1)
    return pad.l + t * iw
  }
  const sy = (raw: number) => {
    const a = yLog ? Math.log10(y0) : y0
    const b = yLog ? Math.log10(y1) : y1
    const t = ((yLog ? Math.log10(raw) : raw) - a) / (b - a || 1)
    return pad.t + ih - t * ih
  }

  const xTicks = xLog ? logTicks(x0, x1) : linTicks(x0 * mulX, x1 * mulX).map((t) => t / mulX)
  const yTicks = yLog ? logTicks(y0, y1) : linTicks(y0 * mulY, y1 * mulY).map((t) => t / mulY)

  const famColor = (f: string) => (v as unknown as Record<string, string>)[f] ?? v.ink2

  /* Magnitude colouring uses density, the third number you need to read a
     stereo result honestly. */
  const densRange = useMemo(() => {
    const ds = pts.map((p) => p.vals.density).filter((d): d is number => d != null)
    return ds.length ? [Math.min(...ds), Math.max(...ds)] : [0, 1]
  }, [pts])
  const fillOf = (p: Pt) => {
    if (colorBy === 'family') return famColor(p.family)
    const d = p.vals.density
    if (d == null) return v.muted
    const [lo, hi] = densRange
    return rampColor(hi > lo ? (d - lo) / (hi - lo) : 0.6, v.dark)
  }

  const frontierPath = useMemo(() => {
    const fp = pts
      .filter((p) => front.has(p.id))
      .sort((a, b) => a.vals[xKey]! - b.vals[xKey]!)
    if (fp.length < 2) return ''
    return fp.map((p, i) =>
      `${i ? 'L' : 'M'}${sx(p.vals[xKey]!).toFixed(1)},${sy(p.vals[yKey]!).toFixed(1)}`,
    ).join(' ')
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pts, front, xKey, yKey, x0, x1, y0, y1, iw, ih, xLog, yLog])

  const labelled = (p: Pt) =>
    labelMode === 'all' ||
    pinned.includes(p.id) ||
    hover === p.id ||
    (labelMode === 'frontier' && front.has(p.id))

  /* Labels are placed greedily against the boxes already taken, trying each
     corner in turn and giving up rather than overprinting. The frontier gets
     close-packed at the accurate end, where a fixed offset always collides;
     a label the user explicitly asked for (pinned or hovered) is drawn even
     if nothing is free. */
  const labels = useMemo(() => {
    type Box = [number, number, number, number]
    const overlaps = (a: Box, b: Box) =>
      !(a[2] < b[0] || a[0] > b[2] || a[3] < b[1] || a[1] > b[3])

    const taken: Box[] = pts.map((p) => {
      const cx = sx(p.vals[xKey]!)
      const cy = sy(p.vals[yKey]!)
      return [cx - 8, cy - 8, cx + 8, cy + 8] as Box
    })

    const want = pts.filter(labelled)
    /* Explicit picks first, then the frontier: whoever is placed first wins
       the space. */
    const rank = (p: Pt) =>
      (pinned.includes(p.id) || hover === p.id ? 0 : front.has(p.id) ? 1 : 2)
    want.sort((a, b) => rank(a) - rank(b) || a.algorithm.localeCompare(b.algorithm))

    const CANDS: [number, number][] = [
      [10, -7], [-10, -7], [10, 14], [-10, 14], [0, -15], [0, 21],
      [16, 4], [-16, 4], [14, -20], [-14, -20], [14, 27], [-14, 27],
      [0, -28], [0, 34], [0, -41], [0, 47],
    ]
    const out: {
      id: string; x: number; y: number; anchor: 'start' | 'end' | 'middle'; text: string
    }[] = []

    for (const p of want) {
      const cx = sx(p.vals[xKey]!)
      const cy = sy(p.vals[yKey]!)
      const w = p.algorithm.length * 5.45 + 3
      const forced = pinned.includes(p.id) || hover === p.id
      let done = false
      for (const [dx, dy] of CANDS) {
        const anchor = dx > 0 ? 'start' : dx < 0 ? 'end' : 'middle'
        const lx = cx + dx
        const ly = cy + dy
        const left = anchor === 'start' ? lx : anchor === 'end' ? lx - w : lx - w / 2
        const box: Box = [left - 2, ly - 10, left + w + 2, ly + 3]
        /* A label may spill into the right margin -- better there than
           dropped -- but never over the y-axis ticks or off the canvas. */
        if (box[0] < pad.l - 2 || box[2] > W - 4) continue
        if (box[1] < pad.t + 1 || box[3] > pad.t + ih - 1) continue
        if (taken.some((t) => overlaps(box, t))) continue
        taken.push(box)
        out.push({ id: p.id, x: lx, y: ly, anchor, text: p.algorithm })
        done = true
        break
      }
      if (!done && forced) {
        out.push({ id: p.id, x: cx + 10, y: cy - 7, anchor: 'start', text: p.algorithm })
      }
    }
    return out
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pts, labelMode, pinned, hover, front, xKey, yKey, x0, x1, y0, y1, iw, ih, xLog, yLog])

  const hoverPt = pts.find((p) => p.id === hover) ?? null

  const reset = () => {
    applyPreset('wall_s', 'bad2_all')
    setScene(ALL_SCENES)
    setColorBy('family')
    setShowFrontier(true)
    setLabelMode('frontier')
    setHiddenFams([])
    setHiddenAlgos([])
    setPinned([])
  }

  const downloadSvg = () => {
    const el = svgRef.current
    if (!el) return
    const clone = el.cloneNode(true) as SVGSVGElement
    clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg')
    clone.querySelectorAll('[data-strip]').forEach((n) => n.remove())
    const blob = new Blob(
      ['<?xml version="1.0" encoding="UTF-8"?>\n', new XMLSerializer().serializeToString(clone)],
      { type: 'image/svg+xml' },
    )
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = `${yKey}_vs_${xKey}.svg`
    a.click()
    URL.revokeObjectURL(a.href)
  }

  const axisLabel = (mm: MetricMeta, log: boolean) =>
    `${mm.label}${mm.unit ? ` (${mm.unit}` : ' ('}${log ? ', log scale' : ''})`

  const groups: MetricMeta['group'][] = ['accuracy', 'speed', 'resources']
  const picker = (value: string, onPick: (k: string) => void, id: string) => (
    <select
      aria-label={id} value={value}
      onChange={(e) => onPick(e.target.value)}
    >
      {groups.map((g) => {
        const items = available.filter((mm) => mm.group === g)
        if (!items.length) return null
        return (
          <optgroup key={g} label={g}>
            {items.map((mm) => (
              <option key={mm.key} value={mm.key}>
                {mm.label}{mm.unit ? ` (${mm.unit})` : ''}
              </option>
            ))}
          </optgroup>
        )
      })}
    </select>
  )

  const scaleToggle = (
    value: 'log' | 'linear',
    set: (s: 'log' | 'linear') => void,
    logOk: boolean,
  ) => (
    <div className="seg" role="group">
      {(['linear', 'log'] as const).map((s) => (
        <button
          key={s}
          className={value === s ? 'on' : ''}
          disabled={s === 'log' && !logOk}
          title={s === 'log' && !logOk
            ? 'needs every value to be greater than zero'
            : undefined}
          onClick={() => set(s)}
        >{s}</button>
      ))}
    </div>
  )

  if (!available.length) {
    return (
      <div className="card">
        <h3>Trade-off explorer</h3>
        <div className="muted" style={{ fontSize: 12.5 }}>
          This run has no completed cells with metrics to plot.
        </div>
      </div>
    )
  }

  return (
    <div className="card">
      <div className="row-between" style={{ marginBottom: 10 }}>
        <h3 style={{ margin: 0 }}>Trade-off explorer</h3>
        <div className="card-row">
          <button className="ghost small" onClick={downloadSvg}>download svg</button>
          <button className="ghost small" onClick={reset}>reset</button>
        </div>
      </div>

      {/* presets first: the common questions should cost one click */}
      <div className="card-row" style={{ marginBottom: 10 }}>
        {PRESETS.filter(([, px, py]) => has(px) && has(py)).map(([label, px, py]) => {
          const on = xKey === px && yKey === py
          return (
            <button
              key={label}
              className={`chip-btn ${on ? 'on' : ''}`}
              onClick={() => applyPreset(px, py)}
            >{label}</button>
          )
        })}
      </div>

      <div className="ctrl-grid">
        <label className="ctrl">
          <span>y axis</span>
          {picker(yKey, pickY, 'y axis metric')}
        </label>
        <label className="ctrl narrow">
          <span>y scale</span>
          {scaleToggle(yScale, setYScale, yLogOk)}
        </label>
        <label className="ctrl">
          <span>x axis</span>
          {picker(xKey, pickX, 'x axis metric')}
        </label>
        <label className="ctrl narrow">
          <span>x scale</span>
          {scaleToggle(xScale, setXScale, xLogOk)}
        </label>
        <label className="ctrl">
          <span>scenes</span>
          <select value={scene} onChange={(e) => setScene(e.target.value)}>
            <option value={ALL_SCENES}>mean over all scenes</option>
            {scenes.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
        </label>
        <label className="ctrl narrow">
          <span>colour</span>
          <div className="seg" role="group">
            {(['family', 'density'] as const).map((c) => (
              <button
                key={c} className={colorBy === c ? 'on' : ''}
                disabled={c === 'density' && !has('density')}
                onClick={() => setColorBy(c)}
              >{c}</button>
            ))}
          </div>
        </label>
        <label className="ctrl narrow">
          <span>labels</span>
          <div className="seg" role="group">
            {(['frontier', 'all', 'none'] as const).map((l) => (
              <button
                key={l} className={labelMode === l ? 'on' : ''}
                onClick={() => setLabelMode(l)}
              >{l}</button>
            ))}
          </div>
        </label>
        <label className="ctrl narrow">
          <span>pareto</span>
          <div className="card-row" style={{ height: 29 }}>
            <input
              type="checkbox" checked={showFrontier}
              disabled={!xm.better || !ym.better}
              onChange={(e) => setShowFrontier(e.target.checked)}
            />
            <span className="muted" style={{ fontSize: 11.5 }}>
              {xm.better && ym.better ? 'frontier' : 'no direction'}
            </span>
          </div>
        </label>
      </div>

      {/* legend: identity is shape + hue, never hue alone */}
      <div className="chart-legend">
        {famsPresent.map((f) => {
          const off = hiddenFams.includes(f)
          return (
            <button
              key={f}
              className={`legend-item ${off ? 'off' : ''}`}
              title={off ? `show ${f}` : `hide ${f}`}
              onClick={() => setHiddenFams((h) =>
                h.includes(f) ? h.filter((x) => x !== f) : [...h, f])}
            >
              <svg width="13" height="13" viewBox="-7 -7 14 14" aria-hidden>
                <MarkerShape
                  shape={SHAPES[f] ?? 'circle'} cx={0} cy={0} r={5}
                  fill={colorBy === 'family' ? famColor(f) : v.muted}
                  stroke={v.surface}
                />
              </svg>
              {f}
            </button>
          )
        })}
        {showFrontier && front.size > 0 && (
          <span className="legend-item static">
            <svg width="18" height="13" viewBox="0 0 18 13" aria-hidden>
              <path d="M1 9 L7 5 L17 3" stroke={v.frontier} strokeWidth="2" fill="none" />
            </svg>
            pareto frontier
          </span>
        )}
        {colorBy === 'density' && (
          <span className="legend-item static">
            density
            <svg width="86" height="11" viewBox="0 0 86 11" aria-hidden>
              <defs>
                <linearGradient id="tdx-ramp-legend">
                  {(v.dark ? [...BLUE].reverse() : BLUE).map((c, i, arr) => (
                    <stop key={c} offset={`${(i / (arr.length - 1)) * 100}%`} stopColor={c} />
                  ))}
                </linearGradient>
              </defs>
              <rect x="0" y="1" width="86" height="9" rx="2" fill="url(#tdx-ramp-legend)" />
            </svg>
            <span className="mono" style={{ fontSize: 10.5 }}>
              {sig(densRange[0] * 100)}–{sig(densRange[1] * 100)}%
            </span>
          </span>
        )}
        <span className="spacer" />
        <span className="muted" style={{ fontSize: 11.5 }}>
          {pts.length} of {allPts.length} algorithms
          {nUnplottable > 0 && ` · ${nUnplottable} have no value on these axes`}
        </span>
      </div>

      <div className="chart-wrap" ref={wrapRef}>
        <svg
          ref={svgRef} width={W} height={H} viewBox={`0 0 ${W} ${H}`}
          role="img"
          aria-label={`${ym.label} against ${xm.label} for ${pts.length} algorithms`}
        >
          <rect x="0" y="0" width={W} height={H} fill={v.surface} />

          {/* recessive grid */}
          {xTicks.map((t) => (
            <line
              key={`gx${t}`} x1={sx(t)} x2={sx(t)} y1={pad.t} y2={pad.t + ih}
              stroke={v.border} strokeWidth="1"
            />
          ))}
          {yTicks.map((t) => (
            <line
              key={`gy${t}`} x1={pad.l} x2={pad.l + iw} y1={sy(t)} y2={sy(t)}
              stroke={v.border} strokeWidth="1"
            />
          ))}
          <line
            x1={pad.l} x2={pad.l + iw} y1={pad.t + ih} y2={pad.t + ih}
            stroke={v.borderStrong} strokeWidth="1"
          />
          <line
            x1={pad.l} x2={pad.l} y1={pad.t} y2={pad.t + ih}
            stroke={v.borderStrong} strokeWidth="1"
          />

          {xTicks.map((t) => (
            <text
              key={`tx${t}`} x={sx(t)} y={pad.t + ih + 15} textAnchor="middle"
              fontSize="10.5" fill={v.muted} fontFamily="ui-monospace, monospace"
            >{sig(t * mulX)}</text>
          ))}
          {yTicks.map((t) => (
            <text
              key={`ty${t}`} x={pad.l - 8} y={sy(t) + 3.5} textAnchor="end"
              fontSize="10.5" fill={v.muted} fontFamily="ui-monospace, monospace"
            >{sig(t * mulY)}</text>
          ))}

          <text
            x={pad.l + iw / 2} y={H - 8} textAnchor="middle"
            fontSize="11.5" fill={v.ink2}
          >
            {axisLabel(xm, xLog)}{xm.better ? ` · ${xm.better} is better` : ''}
          </text>
          <text
            transform={`translate(13 ${pad.t + ih / 2}) rotate(-90)`}
            textAnchor="middle" fontSize="11.5" fill={v.ink2}
          >
            {axisLabel(ym, yLog)}{ym.better ? ` · ${ym.better} is better` : ''}
          </text>

          {frontierPath && (
            <path
              d={frontierPath} fill="none" stroke={v.frontier}
              strokeWidth="2" strokeLinejoin="round" opacity="0.9"
            />
          )}

          {pts.map((p) => {
            const cx = sx(p.vals[xKey]!)
            const cy = sy(p.vals[yKey]!)
            const on = hover === p.id || pinned.includes(p.id)
            return (
              <g key={p.id}>
                <MarkerShape
                  shape={SHAPES[p.family] ?? 'circle'}
                  cx={cx} cy={cy} r={on ? 7 : 5.5}
                  fill={fillOf(p)}
                  stroke={on ? v.ink : v.surface}
                  strokeWidth={2}
                />
                {/* hit target larger than the mark */}
                <circle
                  data-strip cx={cx} cy={cy} r={13} fill="transparent"
                  style={{ cursor: 'pointer' }}
                  onMouseEnter={() => setHover(p.id)}
                  onMouseLeave={() => setHover((h) => (h === p.id ? null : h))}
                  onClick={() => setPinned((q) =>
                    q.includes(p.id) ? q.filter((x) => x !== p.id) : [...q, p.id])}
                />
              </g>
            )
          })}

          {/* direct labels, collision-placed; a halo keeps them legible where
              they cross the frontier line */}
          {labels.map((l) => (
            <text
              key={`l${l.id}`}
              x={l.x} y={l.y} textAnchor={l.anchor}
              fontSize="10.5" fill={v.ink}
              stroke={v.surface} strokeWidth="2.6" paintOrder="stroke"
            >{l.text}</text>
          ))}

          {pts.length === 0 && (
            <text
              x={pad.l + iw / 2} y={pad.t + ih / 2} textAnchor="middle"
              fontSize="12.5" fill={v.muted}
            >
              nothing to plot — every algorithm is hidden or missing these metrics
            </text>
          )}
        </svg>

        {hoverPt && (
          <div
            className="chart-tip"
            style={{
              left: Math.min(Math.max(sx(hoverPt.vals[xKey]!), 90), W - 90),
              top: sy(hoverPt.vals[yKey]!) - 14,
            }}
          >
            <div className="tip-head">
              <span className={`dot fam-${hoverPt.family}`} />
              <b>{hoverPt.algorithm}</b>
            </div>
            <div className="tip-row">
              <span>{ym.label}</span>
              <span className="mono">{withUnit(hoverPt.vals[yKey]! * mulY, ym)}</span>
            </div>
            <div className="tip-row">
              <span>{xm.label}</span>
              <span className="mono">{withUnit(hoverPt.vals[xKey]! * mulX, xm)}</span>
            </div>
            {(['bad2_all', 'epe_valid', 'density', 'wall_s'] as const)
              .filter((k) => k !== xKey && k !== yKey && hoverPt.vals[k] != null)
              .map((k) => {
                const mm = BY_KEY.get(k)!
                return (
                  <div className="tip-row sub" key={k}>
                    <span>{mm.label}</span>
                    <span className="mono">{withUnit(hoverPt.vals[k]! * mm.mul, mm)}</span>
                  </div>
                )
              })}
            <div className="tip-foot">
              {scene === ALL_SCENES
                ? `mean over ${hoverPt.n} scene${hoverPt.n === 1 ? '' : 's'}`
                : scene}
              {front.has(hoverPt.id) && ' · on the frontier'}
              {' · click to pin the label'}
            </div>
          </div>
        )}
      </div>

      <details className="algo-filter">
        <summary>
          algorithms ({allPts.length - hiddenAlgos.length}/{allPts.length})
        </summary>
        <div className="card-row" style={{ margin: '6px 0' }}>
          <button className="ghost small" onClick={() => setHiddenAlgos([])}>all</button>
          <button
            className="ghost small"
            onClick={() => setHiddenAlgos(allPts.map((p) => p.id))}
          >none</button>
          {front.size > 0 && (
            <button
              className="ghost small"
              onClick={() => setHiddenAlgos(
                allPts.filter((p) => !front.has(p.id)).map((p) => p.id),
              )}
            >frontier only</button>
          )}
        </div>
        <div className="pick-list" style={{ maxHeight: 180, overflow: 'auto' }}>
          {allPts.map((p) => {
            const on = !hiddenAlgos.includes(p.id)
            return (
              <label key={p.id} className={`pick ${on ? 'on' : ''}`}>
                <input
                  type="checkbox" checked={on}
                  onChange={() => setHiddenAlgos((h) =>
                    h.includes(p.id) ? h.filter((x) => x !== p.id) : [...h, p.id])}
                />
                <span className={`dot fam-${p.family}`} /> {p.algorithm}
              </label>
            )
          })}
        </div>
      </details>

      <div className="hint" style={{ marginTop: 8 }}>
        Hover a point for its numbers, click to pin its label, click a legend
        entry to hide a family. The static{' '}
        <span className="mono">figures/accuracy_vs_time.png</span> is still
        written into the run folder by <span className="mono">bench/report.py</span>.
      </div>
    </div>
  )
}

/* Marker shape carries family identity; it survives any colour-vision
   deficit, and it is what the generated PNG uses too. */
function MarkerShape({
  shape, cx, cy, r, fill, stroke, strokeWidth = 2,
}: {
  shape: Shape
  cx: number
  cy: number
  r: number
  fill: string
  stroke: string
  strokeWidth?: number
}) {
  const common = { fill, stroke, strokeWidth, strokeLinejoin: 'round' as const }
  if (shape === 'square') {
    const s = r * 1.78
    return <rect x={cx - s / 2} y={cy - s / 2} width={s} height={s} rx={1.5} {...common} />
  }
  if (shape === 'triangle') {
    const a = r * 1.3
    return (
      <polygon
        points={`${cx},${cy - a} ${cx + a * 0.92},${cy + a * 0.72} ${cx - a * 0.92},${cy + a * 0.72}`}
        {...common}
      />
    )
  }
  if (shape === 'diamond') {
    const a = r * 1.32
    return (
      <polygon
        points={`${cx},${cy - a} ${cx + a},${cy} ${cx},${cy + a} ${cx - a},${cy}`}
        {...common}
      />
    )
  }
  return <circle cx={cx} cy={cy} r={r} {...common} />
}
