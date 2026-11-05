import { useEffect, useRef, useState } from 'react'
import type { Algorithm, ComputeResult, Metrics, ParamSpec, Scene } from './types'

export const fmt = (v: number | null | undefined, digits = 2, suffix = '') =>
  v == null || !Number.isFinite(v) ? '—' : v.toFixed(digits) + suffix

export function FamilyDot({ family }: { family: string }) {
  return <span className={`dot fam-${family}`} title={family} />
}

/* ------------------------------------------------------------------ images */

export function ImageTile({
  title, note, src, empty,
}: { title: string; note?: string; src?: string; empty?: string }) {
  return (
    <div className="img-tile">
      <div className="cap">
        <b>{title}</b>
        {note && <span className="note">{note}</span>}
      </div>
      {src ? (
        <img src={src} alt={title} loading="lazy" />
      ) : (
        <div className="img-placeholder">{empty ?? 'nothing to show yet'}</div>
      )}
    </div>
  )
}

/* -------------------------------------------------------------- parameters */

function ParamField({
  spec, value, onChange,
}: {
  spec: ParamSpec
  value: unknown
  onChange: (v: unknown) => void
}) {
  const label = (
    <span className="lbl">
      <b>{spec.name}</b>
      {spec.tuned && <span className="chip tuned">swept</span>}
      <span className="val">
        {Array.isArray(value) ? value.join('x') : String(value)}
      </span>
    </span>
  )

  if (spec.kind === 'bool') {
    return (
      <label className="field">
        <span className="lbl">
          <b>{spec.name}</b>
          {spec.tuned && <span className="chip tuned">swept</span>}
        </span>
        <div className="card-row">
          <input
            type="checkbox"
            checked={Boolean(value)}
            onChange={(e) => onChange(e.target.checked)}
          />
          <span className="muted" style={{ fontSize: 12 }}>
            {value ? 'on' : 'off'}
          </span>
        </div>
        {spec.hint && <div className="hint">{spec.hint}</div>}
      </label>
    )
  }

  if (spec.kind === 'choice') {
    const options = spec.choices ?? []
    const encode = (v: unknown) => JSON.stringify(v)
    return (
      <label className="field">
        {label}
        <select
          value={encode(value)}
          onChange={(e) => onChange(JSON.parse(e.target.value))}
        >
          {options.map((o) => (
            <option key={encode(o)} value={encode(o)}>
              {Array.isArray(o) ? o.join(' x ') : String(o)}
            </option>
          ))}
        </select>
        {spec.hint && <div className="hint">{spec.hint}</div>}
      </label>
    )
  }

  const num = Number(value)
  const step = spec.step ?? (spec.kind === 'int' ? 1 : 0.05)
  return (
    <label className="field">
      {label}
      <input
        type="range"
        min={spec.min}
        max={spec.max}
        step={step}
        value={num}
        onChange={(e) => {
          let v = Number(e.target.value)
          if (spec.odd_only && v % 2 === 0) v += 1
          onChange(spec.kind === 'int' ? Math.round(v) : v)
        }}
      />
      <div className="card-row" style={{ justifyContent: 'space-between' }}>
        <span className="muted" style={{ fontSize: 11 }}>{spec.min}</span>
        {spec.swept && (
          <span className="muted" style={{ fontSize: 11 }}>
            swept: {spec.swept.join(', ')}
          </span>
        )}
        <span className="muted" style={{ fontSize: 11 }}>{spec.max}</span>
      </div>
      {spec.hint && <div className="hint">{spec.hint}</div>}
    </label>
  )
}

export function ParamControls({
  algorithm, values, onChange, onReset,
}: {
  algorithm: Algorithm
  values: Record<string, unknown>
  onChange: (name: string, v: unknown) => void
  onReset: () => void
}) {
  const changed = algorithm.params.filter(
    (p) => JSON.stringify(values[p.name]) !== JSON.stringify(p.value),
  ).length
  return (
    <div className="card">
      <div className="row-between" style={{ marginBottom: 10 }}>
        <h3 style={{ margin: 0 }}>Parameters</h3>
        <button className="ghost small" onClick={onReset} disabled={!changed}>
          reset{changed ? ` (${changed})` : ''}
        </button>
      </div>
      {algorithm.params.length === 0 && (
        <div className="muted" style={{ fontSize: 12.5 }}>
          This algorithm has no tunable parameters — the ONNX export has frozen
          weights, a frozen input size and a frozen iteration count.
        </div>
      )}
      {algorithm.params.map((p) => (
        <ParamField
          key={p.name}
          spec={p}
          value={values[p.name] ?? p.value}
          onChange={(v) => onChange(p.name, v)}
        />
      ))}
    </div>
  )
}

/* ----------------------------------------------------------------- metrics */

export function MetricsPanel({
  result, threshold,
}: { result: ComputeResult | null; threshold: number }) {
  if (!result) return null
  const m: Metrics | null = result.metrics
  const t = result.timing
  const key = String(threshold)
  const bad = m?.bad?.[key] ?? null
  const badValid = m?.bad_valid?.[key] ?? null

  return (
    <>
      {m ? (
        <div className="card">
          <h3>Accuracy &middot; {m.eval_region} region</h3>
          <div className="stat-grid">
            <div className="stat">
              <div className="k">bad {threshold}px</div>
              <div className={`v ${bad != null && bad < 15 ? 'good' : 'bad'}`}>
                {fmt(bad, 2, '%')}
              </div>
              <div className="sub">blanks count as errors</div>
            </div>
            <div className="stat">
              <div className="k">bad, answered</div>
              <div className="v">{fmt(badValid, 2, '%')}</div>
              <div className="sub">where it did answer</div>
            </div>
            <div className="stat">
              <div className="k">density</div>
              <div className="v">{fmt((m.density ?? 0) * 100, 1, '%')}</div>
              <div className="sub">pixels estimated</div>
            </div>
            <div className="stat">
              <div className="k">EPE</div>
              <div className="v">{fmt(m.epe_valid)}</div>
              <div className="sub">mean abs error, px</div>
            </div>
            <div className="stat">
              <div className="k">RMSE</div>
              <div className="v">{fmt(m.rmse_valid)}</div>
              <div className="sub">answered pixels</div>
            </div>
            <div className="stat">
              <div className="k">bias</div>
              <div className="v">{fmt(m.bias_valid, 2)}</div>
              <div className="sub">signed, px</div>
            </div>
            <div className="stat">
              <div className="k">A95</div>
              <div className="v">{fmt(m.a95)}</div>
              <div className="sub">95th pct error</div>
            </div>
            <div className="stat">
              <div className="k">EPE filled</div>
              <div className="v">{fmt(m.epe_filled)}</div>
              <div className="sub">after hole filling</div>
            </div>
          </div>
        </div>
      ) : (
        <div className="card">
          <div className="banner">
            This scene has no ground truth, so only timing and the disparity map
            are available. Add a dataset with ground truth, or use a synthetic
            scene, to get accuracy numbers.
          </div>
        </div>
      )}

      <div className="card">
        <h3>Compute</h3>
        <div className="stat-grid">
          <div className="stat">
            <div className="k">wall time</div>
            <div className="v">{fmt(t.wall_s, 3, 's')}</div>
            <div className="sub">median of {t.repeats}</div>
          </div>
          <div className="stat">
            <div className="k">cpu time</div>
            <div className="v">{fmt(t.cpu_s, 3, 's')}</div>
            <div className="sub">user + system</div>
          </div>
          <div className="stat">
            <div className="k">cores busy</div>
            <div className="v">{fmt(t.parallelism, 1, 'x')}</div>
            <div className="sub">cpu / wall</div>
          </div>
          <div className="stat">
            <div className="k">rss growth</div>
            <div className="v">{fmt(t.rss_delta_mb, 0, ' MB')}</div>
            <div className="sub">during the run</div>
          </div>
          <div className="stat">
            <div className="k">throughput</div>
            <div className="v">{fmt(t.mdps, 1)}</div>
            <div className="sub">M disp-hyp / s</div>
          </div>
          <div className="stat">
            <div className="k">resolution</div>
            <div className="v" style={{ fontSize: 14 }}>
              {result.width}x{result.height}
            </div>
            <div className="sub">ndisp {result.ndisp}</div>
          </div>
        </div>
        {Object.keys(result.extra).length > 0 && (
          <div className="hint" style={{ marginTop: 9 }}>
            {Object.entries(result.extra)
              .map(([k, v]) => `${k}: ${JSON.stringify(v)}`)
              .join('  ·  ')}
          </div>
        )}
      </div>
    </>
  )
}

/* ------------------------------------------------------------------ pickers */

export function ScenePicker({
  scenes, value, onChange,
}: { scenes: Scene[]; value: string; onChange: (v: string) => void }) {
  const groups = Array.from(new Set(scenes.map((s) => s.kind)))
  return (
    <select value={value} onChange={(e) => onChange(e.target.value)}>
      {groups.map((g) => (
        <optgroup key={g} label={g}>
          {scenes.filter((s) => s.kind === g).map((s) => (
            <option key={s.id} value={s.id}>
              {s.name} — {s.width}x{s.height}, ndisp {s.ndisp}
              {s.has_gt ? '' : ' (no GT)'}
            </option>
          ))}
        </optgroup>
      ))}
    </select>
  )
}

export function AlgorithmPicker({
  algorithms, value, onChange,
}: { algorithms: Algorithm[]; value: string; onChange: (v: string) => void }) {
  const families = ['opencv', 'scratch', 'torch', 'learned']
  return (
    <select value={value} onChange={(e) => onChange(e.target.value)}>
      {families.map((f) => (
        <optgroup key={f} label={f}>
          {algorithms.filter((a) => a.family === f).map((a) => (
            <option key={a.name} value={a.name} disabled={!a.available}>
              {a.name}{a.available ? '' : ' (unavailable)'}
            </option>
          ))}
        </optgroup>
      ))}
    </select>
  )
}

/* -------------------------------------------------------------------- misc */

export function useDebounced<T>(value: T, ms: number): T {
  const [v, setV] = useState(value)
  useEffect(() => {
    const t = setTimeout(() => setV(value), ms)
    return () => clearTimeout(t)
  }, [value, ms])
  return v
}

/** Poll a function while `active`, cleaning up on unmount. */
export function usePoll(fn: () => void, ms: number, active: boolean) {
  const ref = useRef(fn)
  ref.current = fn
  useEffect(() => {
    if (!active) return
    const t = setInterval(() => ref.current(), ms)
    return () => clearInterval(t)
  }, [ms, active])
}
