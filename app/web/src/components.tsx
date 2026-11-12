import { useEffect, useRef, useState } from 'react'
import { api } from './api'
import type {
  Algorithm, ComputeResult, DatasetCollection, Metrics, ParamSpec, Scene,
} from './types'

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

/** Metadata for the selected pair, whether or not it is in the catalogue.
 *
 *  Scenes reached by browsing a dataset are deliberately absent from
 *  `/api/scenes`, so a view that looked them up there alone would show "no
 *  GT" for a pair that has ground truth and blank chips for its size. Falls
 *  back to asking for that one scene, and remembers the answer: the
 *  Playground re-renders on every slider drag.
 */
const sceneCache = new Map<string, Scene>()

export function useScene(scenes: Scene[], sceneId: string): Scene | undefined {
  const listed = scenes.find((s) => s.id === sceneId)
  const [fetched, setFetched] = useState<Scene | undefined>(
    () => sceneCache.get(sceneId),
  )

  useEffect(() => {
    if (listed) return
    const cached = sceneCache.get(sceneId)
    if (cached) { setFetched(cached); return }
    let live = true
    setFetched(undefined)
    api.scene(sceneId)
      .then((s) => {
        sceneCache.set(sceneId, s)
        if (live) setFetched(s)
      })
      .catch(() => { /* a bad id surfaces through the run itself */ })
    return () => { live = false }
  }, [sceneId, listed])

  return listed ?? (fetched?.id === sceneId ? fetched : undefined)
}

/** Indentation for a dataset that sits inside another selected one, so a
 *  set and its train/val/test subsets read as a tree rather than as four
 *  unrelated rows with confusingly similar names. */
function nesting(id: string, ids: string[]): number {
  let depth = 0
  for (const other of ids) {
    if (other !== id && id.startsWith(other + '/')) depth += 1
  }
  return depth
}

export function collectionLabel(c: DatasetCollection): string {
  const splits = Object.keys(c.splits)
  return (
    `${c.count} pair${c.count === 1 ? '' : 's'}`
    + (splits.length > 1 ? ` · ${splits.length} subsets` : '')
    + (c.has_gt ? '' : ' · no GT')
  )
}

/** Multi-select over whole datasets. The counterpart to picking pairs one by
 *  one: one click selects every pair in a folder, which is the only workable
 *  way to benchmark a generated set of a hundred. */
export function DatasetPicker({
  collections, picked, onChange,
}: {
  collections: DatasetCollection[]
  picked: string[]
  onChange: (next: string[]) => void
}) {
  const ids = collections.map((c) => c.id)
  const toggle = (id: string) =>
    onChange(picked.includes(id) ? picked.filter((x) => x !== id) : [...picked, id])

  if (collections.length === 0) {
    return (
      <div className="muted" style={{ fontSize: 12.5, padding: '6px 0' }}>
        No datasets found. Drop a folder of stereo pairs into
        <span className="mono"> data/custom/</span> — any depth of
        subfolders is fine — or generate one in the Synthesize tab.
      </div>
    )
  }

  return (
    <div className="scroll-list pick-list" style={{ maxHeight: 190 }}>
      {collections.map((c) => (
        <label
          key={c.id}
          className={`pick ${picked.includes(c.id) ? 'on' : ''}`}
          style={{ paddingLeft: 8 + nesting(c.id, ids) * 14 }}
          title={c.path}
        >
          <input
            type="checkbox"
            checked={picked.includes(c.id)}
            onChange={() => toggle(c.id)}
          />
          <span>
            {c.name} <span className="muted">{c.kind}</span>
            <div className="muted" style={{ fontSize: 11 }}>
              {collectionLabel(c)}
            </div>
          </span>
        </label>
      ))}
    </div>
  )
}

/** Pick one pair: either from the catalogue of individually listed scenes,
 *  or by stepping through a dataset.
 *
 *  A dataset folder contributes only its first handful of samples to
 *  `/api/scenes` — listing twenty thousand would mean decoding twenty
 *  thousand images to describe them. Browsing by dataset asks for that set's
 *  ids on demand instead, so every pair in it is reachable here, not just
 *  the ones the catalogue happens to list.
 */
export function SceneSelect({
  scenes, collections, value, onChange,
}: {
  scenes: Scene[]
  collections: DatasetCollection[]
  value: string
  onChange: (v: string) => void
}) {
  const [dataset, setDataset] = useState('')
  const [samples, setSamples] = useState<string[]>([])
  const [loading, setLoading] = useState(false)

  // Held in a ref so switching dataset does not re-run on every parent
  // render just because the callback identity changed.
  const onChangeRef = useRef(onChange)
  onChangeRef.current = onChange

  useEffect(() => {
    if (!dataset) { setSamples([]); return }
    let live = true
    setLoading(true)
    api.datasetScenes(dataset, 5000)
      .then((r) => {
        if (!live) return
        setSamples(r.scene_ids)
        if (r.scene_ids.length) onChangeRef.current(r.scene_ids[0])
      })
      .catch(() => { if (live) setSamples([]) })
      .finally(() => { if (live) setLoading(false) })
    return () => { live = false }
  }, [dataset])

  const index = samples.indexOf(value)
  const step = (by: number) => {
    if (!samples.length) return
    const next = (index + by + samples.length) % samples.length
    onChange(samples[next])
  }

  const browsable = collections.filter((c) => c.count > 1)

  return (
    <div style={{ display: 'grid', gap: 6 }}>
      {browsable.length > 0 && (
        <select value={dataset} onChange={(e) => setDataset(e.target.value)}>
          <option value="">browse: all listed scenes</option>
          {browsable.map((c) => (
            <option key={c.id} value={c.id}>
              {c.kind}/{c.name} — {c.count} pairs
            </option>
          ))}
        </select>
      )}

      {dataset ? (
        <div className="card-row">
          <select
            value={value}
            style={{ flex: 1, minWidth: 0 }}
            onChange={(e) => onChange(e.target.value)}
          >
            {!samples.includes(value) && <option value={value}>{value}</option>}
            {samples.map((id) => (
              <option key={id} value={id}>{id}</option>
            ))}
          </select>
          <button className="ghost small" onClick={() => step(-1)}
                  disabled={!samples.length}>◀</button>
          <button className="ghost small" onClick={() => step(1)}
                  disabled={!samples.length}>▶</button>
          <span className="muted" style={{ fontSize: 11.5, whiteSpace: 'nowrap' }}>
            {loading ? 'loading…' : `${index + 1}/${samples.length}`}
          </span>
        </div>
      ) : (
        <ScenePicker scenes={scenes} value={value} onChange={onChange} />
      )}
    </div>
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
