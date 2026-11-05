import { useCallback, useEffect, useMemo, useState } from 'react'
import { api, runFigureUrl } from '../api'
import { FamilyDot, fmt, usePoll } from '../components'
import type { Algorithm, Job, RunDetail, RunRow, RunSummary, Scene } from '../types'

type SortKey = string

export default function Runs({
  scenes, algorithms,
}: { scenes: Scene[]; algorithms: Algorithm[] }) {
  const [runs, setRuns] = useState<RunSummary[]>([])
  const [selected, setSelected] = useState<string | null>(null)
  const [detail, setDetail] = useState<RunDetail | null>(null)
  const [figures, setFigures] = useState<string[]>([])
  const [sortKey, setSortKey] = useState<SortKey>('bad2_all')
  const [asc, setAsc] = useState(true)
  const [jobs, setJobs] = useState<Job[]>([])
  const [error, setError] = useState<string | null>(null)

  // batch-run form
  const [pickScenes, setPickScenes] = useState<string[]>([])
  const [pickAlgos, setPickAlgos] = useState<string[]>([])
  const [downscale, setDownscale] = useState(2)
  const [repeats, setRepeats] = useState(1)

  const refreshRuns = useCallback(() => {
    api.runs().then(setRuns).catch((e) => setError(String(e)))
  }, [])
  const refreshJobs = useCallback(() => {
    api.jobs().then(setJobs).catch(() => { /* transient */ })
  }, [])

  useEffect(() => { refreshRuns(); refreshJobs() }, [refreshRuns, refreshJobs])

  const anyRunning = jobs.some((j) => j.status === 'running')
  usePoll(() => {
    refreshJobs()
    if (!anyRunning) refreshRuns()
  }, 1500, anyRunning)

  useEffect(() => {
    if (!selected) { setDetail(null); setFigures([]); return }
    api.run(selected).then(setDetail).catch((e) => setError(String(e)))
    api.runFigures(selected).then(setFigures).catch(() => setFigures([]))
  }, [selected])

  const leaderboard = useMemo(() => {
    if (!detail) return []
    const byAlgo = new Map<string, RunRow[]>()
    for (const r of detail.rows) {
      if (r.status !== 'ok') continue
      const k = String(r.algorithm)
      if (!byAlgo.has(k)) byAlgo.set(k, [])
      byAlgo.get(k)!.push(r)
    }
    const mean = (rows: RunRow[], key: string) => {
      const vals = rows.map((r) => r[key]).filter((v): v is number =>
        typeof v === 'number' && Number.isFinite(v))
      return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null
    }
    return Array.from(byAlgo.entries()).map(([algorithm, rows]) => ({
      algorithm,
      family: String(rows[0].family ?? ''),
      n: rows.length,
      bad2_all: mean(rows, 'bad2_all'),
      bad1_all: mean(rows, 'bad1_all'),
      epe_valid: mean(rows, 'epe_valid'),
      density: mean(rows, 'density'),
      wall_s: mean(rows, 'wall_s'),
      cpu_s: mean(rows, 'cpu_s'),
      parallelism: mean(rows, 'parallelism'),
    }))
  }, [detail])

  const sorted = useMemo(() => {
    const rows = [...leaderboard]
    rows.sort((a, b) => {
      const av = (a as Record<string, unknown>)[sortKey]
      const bv = (b as Record<string, unknown>)[sortKey]
      if (typeof av === 'string' || typeof bv === 'string') {
        return asc
          ? String(av).localeCompare(String(bv))
          : String(bv).localeCompare(String(av))
      }
      const an = typeof av === 'number' ? av : Infinity
      const bn = typeof bv === 'number' ? bv : Infinity
      return asc ? an - bn : bn - an
    })
    return rows
  }, [leaderboard, sortKey, asc])

  const sortBy = (k: SortKey) => {
    if (k === sortKey) setAsc(!asc)
    else { setSortKey(k); setAsc(true) }
  }

  const startRun = async () => {
    setError(null)
    try {
      await api.startBenchmark({
        scene_ids: pickScenes,
        algorithms: pickAlgos,
        repeats,
        downscale,
        save_figures: true,
      })
      refreshJobs()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  const cols: [string, string][] = [
    ['algorithm', 'algorithm'], ['bad2_all', 'bad 2px'], ['bad1_all', 'bad 1px'],
    ['epe_valid', 'EPE'], ['density', 'density'], ['wall_s', 'wall s'],
    ['cpu_s', 'cpu s'], ['parallelism', 'cores'], ['n', 'scenes'],
  ]

  return (
    <div className="split">
      <div>
        <div className="card">
          <h3>Trigger a benchmark</h3>
          <div className="row-between" style={{ marginBottom: 6 }}>
            <span className="muted" style={{ fontSize: 12 }}>
              scenes ({pickScenes.length})
            </span>
            <div className="card-row">
              <button
                className="ghost small"
                onClick={() => setPickScenes(scenes.filter((s) => s.has_gt).map((s) => s.id))}
              >all with GT</button>
              <button className="ghost small" onClick={() => setPickScenes([])}>
                clear
              </button>
            </div>
          </div>
          <div className="scroll-list pick-list" style={{ maxHeight: 150 }}>
            {scenes.map((s) => (
              <label key={s.id} className={`pick ${pickScenes.includes(s.id) ? 'on' : ''}`}>
                <input
                  type="checkbox"
                  checked={pickScenes.includes(s.id)}
                  onChange={() => setPickScenes((p) =>
                    p.includes(s.id) ? p.filter((x) => x !== s.id) : [...p, s.id])}
                />
                {s.name} <span className="muted">{s.kind}</span>
              </label>
            ))}
          </div>

          <div className="row-between" style={{ margin: '10px 0 6px' }}>
            <span className="muted" style={{ fontSize: 12 }}>
              algorithms ({pickAlgos.length})
            </span>
            <div className="card-row">
              <button
                className="ghost small"
                onClick={() => setPickAlgos(
                  algorithms.filter((a) => a.available).map((a) => a.name),
                )}
              >all</button>
              <button className="ghost small" onClick={() => setPickAlgos([])}>
                clear
              </button>
            </div>
          </div>
          <div className="scroll-list pick-list" style={{ maxHeight: 150 }}>
            {algorithms.filter((a) => a.available).map((a) => (
              <label key={a.name} className={`pick ${pickAlgos.includes(a.name) ? 'on' : ''}`}>
                <input
                  type="checkbox"
                  checked={pickAlgos.includes(a.name)}
                  onChange={() => setPickAlgos((p) =>
                    p.includes(a.name) ? p.filter((x) => x !== a.name) : [...p, a.name])}
                />
                <FamilyDot family={a.family} /> {a.name}
              </label>
            ))}
          </div>

          <label className="field" style={{ marginTop: 10 }}>
            <span className="lbl">
              <b>downscale</b><span className="val">{downscale}x</span>
            </span>
            <input
              type="range" min={1} max={8} step={1} value={downscale}
              onChange={(e) => setDownscale(Number(e.target.value))}
            />
          </label>
          <label className="field">
            <span className="lbl">
              <b>repeats</b><span className="val">{repeats}</span>
            </span>
            <input
              type="range" min={1} max={5} step={1} value={repeats}
              onChange={(e) => setRepeats(Number(e.target.value))}
            />
          </label>
          <button
            className="primary" style={{ width: '100%' }}
            onClick={() => void startRun()}
            disabled={!pickScenes.length || !pickAlgos.length}
          >
            run {pickScenes.length * pickAlgos.length} cells
          </button>
          <div className="hint" style={{ marginTop: 7 }}>
            Writes to <span className="mono">results/&lt;timestamp&gt;/</span> exactly
            like the CLI, with the same summary and figures.
          </div>
        </div>

        {jobs.length > 0 && (
          <div className="card">
            <h3>Jobs</h3>
            {jobs.slice(0, 4).map((j) => (
              <div key={j.id} style={{ marginBottom: 10 }}>
                <div className="row-between">
                  <span style={{ fontSize: 12.5 }}>
                    {j.status === 'running' && <span className="spin" />} {j.status}
                    <span className="muted"> · {j.done}/{j.total} · {j.elapsed_s}s</span>
                  </span>
                  {j.status === 'running' && (
                    <button
                      className="ghost small danger"
                      onClick={() => void api.cancelJob(j.id).then(refreshJobs)}
                    >cancel</button>
                  )}
                </div>
                <div className="progress" style={{ marginTop: 5 }}>
                  <div style={{ width: `${(j.done / Math.max(1, j.total)) * 100}%` }} />
                </div>
                {j.current && (
                  <div className="hint mono" style={{ marginTop: 4 }}>{j.current}</div>
                )}
                {j.message && <div className="hint">{j.message}</div>}
              </div>
            ))}
          </div>
        )}

        <div className="card">
          <div className="row-between" style={{ marginBottom: 8 }}>
            <h3 style={{ margin: 0 }}>Stored runs</h3>
            <button className="ghost small" onClick={refreshRuns}>refresh</button>
          </div>
          <div className="scroll-list pick-list">
            {runs.map((r) => (
              <div
                key={r.id}
                className={`pick ${selected === r.id ? 'on' : ''}`}
                onClick={() => setSelected(r.id)}
              >
                <div>
                  <div className="mono">{r.id}</div>
                  <div className="muted" style={{ fontSize: 11 }}>
                    {r.n_ok}/{r.n_cells} cells · {r.algorithms.length} algos ·{' '}
                    {r.scenes.length} scenes
                  </div>
                </div>
              </div>
            ))}
            {runs.length === 0 && (
              <div className="muted" style={{ fontSize: 12.5 }}>
                No runs yet. Trigger one above, or run
                <span className="mono"> python -m bench run</span>.
              </div>
            )}
          </div>
        </div>
      </div>

      <div>
        {error && <div className="banner error" style={{ marginBottom: 12 }}>{error}</div>}
        {!selected && (
          <div className="card">
            <div className="muted">Pick a run on the left to see its leaderboard.</div>
          </div>
        )}
        {detail && (
          <>
            <div className="card">
              <h3>Leaderboard &middot; {detail.id} &middot; mean over scenes</h3>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      {cols.map(([k, label]) => (
                        <th key={k} onClick={() => sortBy(k)}>
                          {label}{sortKey === k ? (asc ? ' ▲' : ' ▼') : ''}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {sorted.map((r) => (
                      <tr key={r.algorithm}>
                        <td><FamilyDot family={r.family} /> {r.algorithm}</td>
                        <td className="num">{fmt(r.bad2_all, 2, '%')}</td>
                        <td className="num">{fmt(r.bad1_all, 2, '%')}</td>
                        <td className="num">{fmt(r.epe_valid)}</td>
                        <td className="num">{fmt((r.density ?? 0) * 100, 1, '%')}</td>
                        <td className="num">{fmt(r.wall_s, 3)}</td>
                        <td className="num">{fmt(r.cpu_s, 2)}</td>
                        <td className="num">{fmt(r.parallelism, 1, 'x')}</td>
                        <td className="num">{r.n}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {figures.filter((f) => !f.includes('__')).length > 0 && (
              <div className="card">
                <h3>Figures</h3>
                <div className="img-grid">
                  {figures.filter((f) => !f.includes('__')).map((f) => (
                    <div key={f} className="img-tile">
                      <div className="cap"><b>{f.replace('.png', '')}</b></div>
                      <img src={runFigureUrl(detail.id, f)} alt={f} loading="lazy"
                           style={{ background: '#fcfcfb' }} />
                    </div>
                  ))}
                </div>
              </div>
            )}

            {detail.summary && (
              <div className="card">
                <h3>summary.md</h3>
                <pre className="log" style={{ maxHeight: 420 }}>{detail.summary}</pre>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}
