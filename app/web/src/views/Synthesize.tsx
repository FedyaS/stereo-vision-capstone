import { useCallback, useEffect, useMemo, useState } from 'react'
import { api, synthPreviewUrl } from '../api'
import { ImageTile, fmt, usePoll } from '../components'
import type {
  Job, RuleField, SceneRules, SynthDataset, SynthPreview, SynthSchema,
} from '../types'

/** The four things worth looking at for every generated pair. Ground truth
 *  and the occlusion mask are shown by default alongside the images, because
 *  a rule set that produces a handsome picture and useless supervision looks
 *  completely fine if you only ever look at the left image. */
const VIEWS: [string, string][] = [
  ['left', 'left image'],
  ['right', 'right image'],
  ['disparity', 'ground truth'],
  ['nonocc', 'visible in both'],
]

const GROUP_ORDER = ['Framing', 'Content', 'Appearance', 'Sensor', 'Rendering']

function RuleControl({
  field, value, onChange,
}: {
  field: RuleField
  value: unknown
  onChange: (v: unknown) => void
}) {
  if (field.type === 'bool') {
    return (
      <label className="field">
        <span className="lbl"><b>{field.label}</b></span>
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
        <div className="hint">{field.help}</div>
      </label>
    )
  }

  if (field.type === 'choice') {
    const opts = field.options ?? []
    const current = opts.find((o) => o.value === value)
    return (
      <label className="field">
        <span className="lbl"><b>{field.label}</b></span>
        <select value={String(value)} onChange={(e) => onChange(e.target.value)}>
          {opts.map((o) => (
            <option key={o.value} value={o.value}>{o.value}</option>
          ))}
        </select>
        <div className="hint">{current?.help ?? field.help}</div>
      </label>
    )
  }

  if (field.type === 'multichoice') {
    const selected = Array.isArray(value) ? (value as string[]) : []
    const toggle = (v: string) => {
      const next = selected.includes(v)
        ? selected.filter((x) => x !== v)
        : [...selected, v]
      // Never let the list empty out: the generator rejects it, and the
      // error would arrive as a failed preview rather than as a disabled box.
      if (next.length) onChange(next)
    }
    return (
      <div className="field">
        <span className="lbl">
          <b>{field.label}</b>
          <span className="val">{selected.length} of {field.options?.length}</span>
        </span>
        <div className="card-row" style={{ flexWrap: 'wrap', gap: 4 }}>
          {(field.options ?? []).map((o) => (
            <button
              key={o.value}
              className="ghost small"
              onClick={() => toggle(o.value)}
              title={o.help}
              style={selected.includes(o.value)
                ? undefined
                : { opacity: 0.45, textDecoration: 'line-through' }}
            >
              {o.value}
            </button>
          ))}
        </div>
        <div className="hint">{field.help}</div>
      </div>
    )
  }

  const num = Number(value)
  const step = field.step ?? (field.type === 'int' ? 1 : 0.05)
  return (
    <label className="field">
      <span className="lbl">
        <b>{field.label}</b>
        <span className="val">{field.type === 'int' ? num : num.toFixed(3)}</span>
      </span>
      <input
        type="range"
        min={field.min} max={field.max} step={step} value={num}
        onChange={(e) => {
          const v = Number(e.target.value)
          onChange(field.type === 'int' ? Math.round(v) : v)
        }}
      />
      <div className="hint">{field.help}</div>
    </label>
  )
}

function StatRow({ summary, ndisp }: {
  summary: Record<string, unknown>; ndisp: number
}) {
  const n = (k: string) => {
    const v = summary[k]
    return typeof v === 'number' ? v : null
  }
  return (
    <div className="stat-grid">
      <div className="stat">
        <div className="k">ground truth</div>
        <div className="v">{fmt((n('gt_coverage') ?? 0) * 100, 1, '%')}</div>
        <div className="sub">pixels with a known disparity</div>
      </div>
      <div className="stat">
        <div className="k">visible in both</div>
        <div className="v">{fmt((n('nonocc_fraction') ?? 0) * 100, 1, '%')}</div>
        <div className="sub">the scorable region</div>
      </div>
      <div className="stat">
        <div className="k">occluded</div>
        <div className="v">{fmt((n('occluded_fraction') ?? 0) * 100, 1, '%')}</div>
        <div className="sub">hidden from the right eye</div>
      </div>
      <div className="stat">
        <div className="k">sky</div>
        <div className="v">{fmt((n('sky_fraction') ?? 0) * 100, 1, '%')}</div>
        <div className="sub">no depth, as outdoors</div>
      </div>
      <div className="stat">
        <div className="k">disparity</div>
        <div className="v" style={{ fontSize: 15 }}>
          {fmt(n('disparity_min'), 0)}–{fmt(n('disparity_max'), 0)}
        </div>
        <div className="sub">of {ndisp} px search range</div>
      </div>
      <div className="stat">
        <div className="k">baseline</div>
        <div className="v" style={{ fontSize: 15 }}>
          {fmt(n('baseline_m'), 3, ' m')}
        </div>
        <div className="sub">fitted to the scene</div>
      </div>
    </div>
  )
}

export default function Synthesize({ onChanged }: { onChanged: () => void }) {
  const [schema, setSchema] = useState<SynthSchema | null>(null)
  const [rules, setRules] = useState<SceneRules>({})
  const [samples, setSamples] = useState(2)
  const [preview, setPreview] = useState<SynthPreview | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [name, setName] = useState('train')
  const [count, setCount] = useState(100)
  const [mixEnvs, setMixEnvs] = useState<string[]>([])
  // Splitting is on by default. An undivided pile of generated pairs invites
  // evaluating on what you trained on, and because every pair comes from the
  // same generator that mistake produces excellent numbers and no warning.
  const [useSplits, setUseSplits] = useState(true)
  const [splits, setSplits] = useState<Record<string, number>>({
    train: 0.8, val: 0.1, test: 0.1,
  })
  const [job, setJob] = useState<Job | null>(null)
  const [datasets, setDatasets] = useState<SynthDataset[]>([])

  const loadDatasets = useCallback(() => {
    api.synthDatasets().then(setDatasets).catch(() => undefined)
  }, [])

  useEffect(() => {
    api.synthSchema()
      .then((s) => {
        setSchema(s)
        setRules(s.presets.room ?? s.defaults)
      })
      .catch((e) => setError(String(e)))
    loadDatasets()
  }, [loadDatasets])

  const set = (k: string, v: unknown) => setRules((r) => ({ ...r, [k]: v }))

  const render = useCallback(async () => {
    if (!schema) return
    setBusy(true)
    setError(null)
    try {
      setPreview(await api.synthPreview(rules, samples))
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setPreview(null)
    } finally {
      setBusy(false)
    }
  }, [rules, samples, schema])

  // Render once as soon as the schema lands, so the tab is never a blank
  // form. Rule edits are explicit from then on: a render is seconds of CPU,
  // so firing one on every slider drag would make the UI feel broken.
  useEffect(() => {
    if (schema && !preview) void render()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [schema])

  usePoll(() => {
    if (!job) return
    api.job(job.id).then((j) => {
      setJob(j)
      if (j.status !== 'running') {
        loadDatasets()
        onChanged()
      }
    }).catch(() => undefined)
  }, 700, job?.status === 'running')

  const generate = async () => {
    setError(null)
    try {
      setJob(await api.startSynthesis({
        name, rules, count,
        environments: mixEnvs.length ? mixEnvs : null,
        splits: useSplits ? splits : null,
      }))
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  const groups = useMemo(() => {
    if (!schema) return []
    const by = new Map<string, RuleField[]>()
    for (const f of schema.fields) {
      if (!by.has(f.group)) by.set(f.group, [])
      by.get(f.group)!.push(f)
    }
    return GROUP_ORDER.filter((g) => by.has(g)).map((g) => [g, by.get(g)!] as const)
  }, [schema])

  if (error && !schema) return <div className="banner error">{error}</div>
  if (!schema) return <div className="banner busy"><span className="spin" /> loading…</div>

  const ndisp = Number(rules.ndisp ?? 64)
  const pixels = Number(rules.width ?? 0) * Number(rules.height ?? 0)
  return (
    <div className="split">
      {/* ------------------------------------------------------- controls */}
      <div>
        <div className="card">
          <h3>Start from</h3>
          <div className="field">
            <span className="lbl"><b>environment preset</b></span>
            <select
              value={String(rules.environment ?? 'room')}
              onChange={(e) => setRules(schema.presets[e.target.value] ?? rules)}
            >
              {schema.environments.map((e) => (
                <option key={e.value} value={e.value}>{e.value}</option>
              ))}
            </select>
            <div className="hint">
              {schema.environments.find((e) => e.value === rules.environment)?.help}
            </div>
          </div>
          <div className="field">
            <span className="lbl"><b>difficulty</b></span>
            <div className="card-row" style={{ gap: 4 }}>
              {schema.difficulties.map((d) => (
                <button
                  key={d.value}
                  className="ghost small"
                  onClick={() => setRules((r) => ({ ...r, ...d.settings }))}
                >
                  {d.value}
                </button>
              ))}
            </div>
            <div className="hint">
              Applies a bundle of appearance and sensor settings. Texture is
              what drives matching error far more than geometry does.
            </div>
          </div>
        </div>

        {groups.map(([group, fields]) => (
          <details className="card" key={group} open={group === 'Framing'}>
            <summary><b>{group}</b></summary>
            <div style={{ marginTop: 8 }}>
              {fields.map((f) => (
                <RuleControl
                  key={f.name}
                  field={f}
                  value={rules[f.name] ?? f.default}
                  onChange={(v) => set(f.name, v)}
                />
              ))}
            </div>
          </details>
        ))}
      </div>

      {/* -------------------------------------------------------- results */}
      <div style={{ display: "grid", gap: 16 }}>
        <div className="card">
          <div className="row-between">
            <h3 style={{ margin: 0 }}>Examples</h3>
            <div className="card-row">
              <label className="muted" style={{ fontSize: 12 }}>
                samples{' '}
                <input
                  type="number" min={1} max={6} value={samples}
                  style={{ width: 52 }}
                  onChange={(e) => setSamples(Number(e.target.value))}
                />
              </label>
              <button onClick={render} disabled={busy}>
                {busy ? 'rendering…' : 'Render examples'}
              </button>
            </div>
          </div>
          <div className="hint" style={{ marginTop: 6 }}>
            {Number(rules.width)}×{Number(rules.height)}, ndisp {ndisp},{' '}
            {(pixels / 1e3).toFixed(0)}k px per view
            {Number(rules.supersample) > 1 &&
              `, ${Number(rules.supersample) ** 2} rays per pixel`}
            . These are the first {samples} pairs the dataset would contain —
            same seeds, same scenes.
          </div>
        </div>

        {error && <div className="banner error">{error}</div>}

        {preview?.warnings.map((w, i) => (
          <div className="banner" key={i}>{w}</div>
        ))}

        {preview && (
          <div className="card">
            <h3>Measured over {preview.samples.length} sample(s)</h3>
            <StatRow summary={preview.summary as Record<string, unknown>} ndisp={ndisp} />
          </div>
        )}

        {preview?.samples.map((s) => (
          <div className="card" key={s.index}>
            <div className="row-between" style={{ marginBottom: 8 }}>
              <h3 style={{ margin: 0 }}>seed {s.seed}</h3>
              <span className="muted" style={{ fontSize: 11.5 }}>
                disparity {fmt(s.stats.disparity_min, 1)}–
                {fmt(s.stats.disparity_max, 1)} px ·{' '}
                depth {fmt(s.stats.depth_min_m, 1)}–
                {fmt(s.stats.depth_max_m, 1)} m ·{' '}
                {fmt(s.stats.occluded_fraction * 100, 1, '%')} occluded
              </span>
            </div>
            <div className="img-grid">
              {VIEWS.map(([view, label]) => (
                <ImageTile
                  key={view}
                  title={label}
                  src={synthPreviewUrl(preview.preview_id, s.index, view)}
                />
              ))}
            </div>
          </div>
        ))}

        {/* ------------------------------------------------ mass generation */}
        <div className="card">
          <h3>Generate a dataset</h3>
          <div className="hint" style={{ marginBottom: 10 }}>
            Writes pairs to <span className="mono">data/synth/&lt;name&gt;/</span> in
            the Middlebury layout, with a manifest recording these exact rules.
            Generated scenes appear in the scene picker straight away, so you
            can benchmark them without an export step.
          </div>
          <div className="card-row" style={{ flexWrap: 'wrap', gap: 10 }}>
            <label className="muted" style={{ fontSize: 12 }}>
              name{' '}
              <input
                value={name} style={{ width: 150 }}
                onChange={(e) => setName(e.target.value)}
              />
            </label>
            <label className="muted" style={{ fontSize: 12 }}>
              pairs{' '}
              <input
                type="number" min={1} max={20000} value={count}
                style={{ width: 90 }}
                onChange={(e) => setCount(Number(e.target.value))}
              />
            </label>
            <button
              onClick={generate}
              disabled={job?.status === 'running' || !name.trim()}
            >
              Generate {count}
            </button>
            {job?.status === 'running' && (
              <button className="ghost" onClick={() => api.cancelJob(job.id)}>
                cancel
              </button>
            )}
          </div>

          <div className="field" style={{ marginTop: 12 }}>
            <span className="lbl">
              <b>splits</b>
              <span className="val">
                {useSplits
                  ? Object.entries(splits)
                      .filter(([, f]) => f > 0)
                      .map(([k, f]) => `${k} ${Math.round(f * count)}`)
                      .join(' · ')
                  : 'one flat folder'}
              </span>
            </span>
            <div className="card-row" style={{ gap: 10, flexWrap: 'wrap' }}>
              <label className="muted" style={{ fontSize: 12 }}>
                <input
                  type="checkbox"
                  checked={useSplits}
                  onChange={(e) => setUseSplits(e.target.checked)}
                />{' '}
                split
              </label>
              {useSplits && Object.keys(splits).map((k) => (
                <label key={k} className="muted" style={{ fontSize: 12 }}>
                  {k}{' '}
                  <input
                    type="number" min={0} max={1} step={0.05}
                    value={splits[k]} style={{ width: 66 }}
                    onChange={(e) => setSplits((s) => ({
                      ...s, [k]: Math.max(0, Number(e.target.value)),
                    }))}
                  />
                </label>
              ))}
            </div>
            <div className="hint">
              Written as <span className="mono">train/</span>,{' '}
              <span className="mono">val/</span> and{' '}
              <span className="mono">test/</span> subfolders. Every pair has
              its own seed and scene, so no geometry is shared between them.
              Fractions are normalised, and each non-empty split gets at least
              one pair.
            </div>
          </div>

          <div className="field" style={{ marginTop: 12 }}>
            <span className="lbl">
              <b>mix environments</b>
              <span className="val">
                {mixEnvs.length ? mixEnvs.join(', ') : `only ${rules.environment}`}
              </span>
            </span>
            <div className="card-row" style={{ flexWrap: 'wrap', gap: 4 }}>
              {schema.environments.map((e) => (
                <button
                  key={e.value}
                  className="ghost small"
                  style={mixEnvs.includes(e.value)
                    ? undefined
                    : { opacity: 0.45 }}
                  onClick={() => setMixEnvs((m) =>
                    m.includes(e.value)
                      ? m.filter((x) => x !== e.value)
                      : [...m, e.value])}
                >
                  {e.value}
                </button>
              ))}
            </div>
            <div className="hint">
              Cycle through several environments instead of one. A model
              trained on a single scene type generalises poorly, so a mixed
              set is usually what you want for training.
            </div>
          </div>

          {job && (
            <div style={{ marginTop: 12 }}>
              <div className="row-between">
                <span
                  className="chip"
                  style={job.status === 'error'
                    ? { borderColor: 'var(--bad)', color: 'var(--bad)' }
                    : job.status === 'done'
                      ? { borderColor: 'var(--good)', color: 'var(--good)' }
                      : undefined}
                >
                  {job.status}
                </span>
                <span className="muted" style={{ fontSize: 12 }}>
                  {job.done}/{job.total} · {job.elapsed_s.toFixed(0)}s
                  {job.done > 0 && job.status === 'running' &&
                    ` · eta ${(
                      (job.total - job.done) * (job.elapsed_s / job.done)
                    ).toFixed(0)}s`}
                </span>
              </div>
              <div className="progress" style={{ marginTop: 6 }}>
                <div
                  style={{ width: `${(100 * job.done) / Math.max(job.total, 1)}%` }}
                />
              </div>
              {job.message && (
                <div className="hint" style={{ marginTop: 6 }}>{job.message}</div>
              )}
            </div>
          )}
        </div>

        {datasets.length > 0 && (
          <div className="card">
            <h3>Generated datasets</h3>
            <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>name</th><th>pairs</th><th>splits</th>
                  <th>environments</th><th>s / pair</th><th>created</th><th />
                </tr>
              </thead>
              <tbody>
                {datasets.map((d) => (
                  <tr key={d.name}>
                    <td className="mono">{d.name}</td>
                    <td>{d.count}</td>
                    <td className="muted">
                      {Object.keys(d.splits ?? {}).length
                        ? Object.entries(d.splits!)
                            .map(([k, n]) => `${k} ${n}`).join(' · ')
                        : 'flat'}
                    </td>
                    <td className="muted">{(d.environments ?? []).join(', ')}</td>
                    <td>{fmt(d.seconds_per_pair ?? null, 2)}</td>
                    <td className="muted" style={{ fontSize: 11.5 }}>
                      {d.created_utc?.replace('T', ' ').replace('+00:00', '') ?? '—'}
                    </td>
                    <td>
                      <button
                        className="ghost small"
                        onClick={async () => {
                          if (!confirm(
                            `Delete the generated dataset "${d.name}" and all ` +
                            `${d.count} of its pairs? This cannot be undone.`,
                          )) return
                          await api.deleteSynthDataset(d.name)
                          loadDatasets()
                          onChanged()
                        }}
                      >
                        delete
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
