import { useCallback, useMemo, useState } from 'react'
import { api, resultImageUrl, sceneImageUrl } from '../api'
import { FamilyDot, ImageTile, SceneSelect, fmt, useScene } from '../components'
import type {
  Algorithm, ComputeResult, DatasetCollection, Scene,
} from '../types'

const THRESHOLDS = [0.5, 1, 2, 4]
const PRESETS: Record<string, string[]> = {
  'cost functions': [
    'scratch_sad_wta', 'scratch_ssd_wta', 'scratch_zncc_wta', 'scratch_census_wta',
  ],
  'aggregation': [
    'scratch_census_wta', 'scratch_census_dp', 'scratch_census_sgm4',
    'scratch_census_sgm8',
  ],
  'opencv modes': [
    'opencv_bm', 'opencv_sgbm', 'opencv_sgbm_3way', 'opencv_sgbm_hh4',
    'opencv_sgbm_wls',
  ],
  'backends': ['scratch_census_sgm4', 'torch_census_sgm4'],
  'best of each family': [
    'opencv_sgbm_hh4', 'scratch_census_sgm8', 'torch_census_sgm4',
    'learned_raft_stereo_r4',
  ],
}

export default function Compare({
  scenes, algorithms, collections,
}: {
  scenes: Scene[]
  algorithms: Algorithm[]
  collections: DatasetCollection[]
}) {
  const [sceneId, setSceneId] = useState('middlebury/Teddy')
  const [downscale, setDownscale] = useState(2)
  const [evalRegion, setEvalRegion] = useState('nonocc')
  const [threshold, setThreshold] = useState(2)
  const [picked, setPicked] = useState<string[]>(PRESETS['cost functions'])
  const [results, setResults] = useState<Record<string, ComputeResult>>({})
  const [errors, setErrors] = useState<Record<string, string>>({})
  const [busy, setBusy] = useState(false)
  const [progress, setProgress] = useState(0)

  const scene = useScene(scenes, sceneId)
  const available = useMemo(
    () => algorithms.filter((a) => a.available),
    [algorithms],
  )

  const toggle = (name: string) =>
    setPicked((p) => (p.includes(name) ? p.filter((x) => x !== name) : [...p, name]))

  const runAll = useCallback(async () => {
    setBusy(true)
    setResults({})
    setErrors({})
    setProgress(0)
    // Sequential on purpose: these numbers include wall time, and running
    // several matchers at once would have them competing for the same cores.
    for (let i = 0; i < picked.length; i++) {
      const name = picked[i]
      try {
        const res = await api.compute({
          scene_id: sceneId,
          algorithm: name,
          params: {},
          downscale,
          eval_region: evalRegion,
          thresholds: THRESHOLDS,
        })
        setResults((r) => ({ ...r, [name]: res }))
      } catch (e) {
        setErrors((x) => ({ ...x, [name]: e instanceof Error ? e.message : String(e) }))
      }
      setProgress(((i + 1) / picked.length) * 100)
    }
    setBusy(false)
  }, [picked, sceneId, downscale, evalRegion])

  const rows = picked
    .map((name) => ({ name, res: results[name] }))
    .filter((r) => r.res)
    .sort((a, b) => {
      const av = a.res!.metrics?.bad[String(threshold)] ?? Infinity
      const bv = b.res!.metrics?.bad[String(threshold)] ?? Infinity
      return av - bv
    })

  return (
    <div className="split">
      <div>
        <div className="card">
          <h3>Scene</h3>
          <label className="field">
            <span className="lbl"><b>image pair</b></span>
            <SceneSelect
              scenes={scenes} collections={collections}
              value={sceneId} onChange={setSceneId}
            />
          </label>
          <label className="field">
            <span className="lbl">
              <b>downscale</b><span className="val">{downscale}x</span>
            </span>
            <input
              type="range" min={1} max={8} step={1} value={downscale}
              onChange={(e) => setDownscale(Number(e.target.value))}
            />
            <div className="hint">
              2x or more keeps a multi-algorithm comparison responsive.
            </div>
          </label>
          <div className="card-row">
            <label className="card-row" style={{ gap: 6, fontSize: 12.5 }}>
              region
              <select
                value={evalRegion} style={{ width: 'auto' }}
                onChange={(e) => setEvalRegion(e.target.value)}
              >
                <option value="nonocc">non-occluded</option>
                <option value="all">all</option>
              </select>
            </label>
            <label className="card-row" style={{ gap: 6, fontSize: 12.5 }}>
              bad
              <select
                value={threshold} style={{ width: 'auto' }}
                onChange={(e) => setThreshold(Number(e.target.value))}
              >
                {THRESHOLDS.map((t) => <option key={t} value={t}>{t} px</option>)}
              </select>
            </label>
          </div>
        </div>

        <div className="card">
          <h3>Presets</h3>
          <div className="card-row">
            {Object.entries(PRESETS).map(([label, names]) => (
              <button
                key={label} className="small"
                onClick={() => setPicked(names.filter(
                  (n) => available.some((a) => a.name === n),
                ))}
              >
                {label}
              </button>
            ))}
          </div>
          <div className="hint" style={{ marginTop: 8 }}>
            Each preset isolates one variable. "cost functions" keeps the
            pipeline fixed and changes only how patches are compared.
          </div>
        </div>

        <div className="card">
          <div className="row-between" style={{ marginBottom: 8 }}>
            <h3 style={{ margin: 0 }}>Algorithms ({picked.length})</h3>
            <button className="ghost small" onClick={() => setPicked([])}>
              clear
            </button>
          </div>
          <div className="scroll-list pick-list">
            {available.map((a) => (
              <label
                key={a.name}
                className={`pick ${picked.includes(a.name) ? 'on' : ''}`}
              >
                <input
                  type="checkbox"
                  checked={picked.includes(a.name)}
                  onChange={() => toggle(a.name)}
                />
                <FamilyDot family={a.family} />
                <span>{a.name}</span>
              </label>
            ))}
          </div>
          <button
            className="primary"
            style={{ width: '100%', marginTop: 10 }}
            onClick={() => void runAll()}
            disabled={busy || picked.length === 0}
          >
            {busy ? `running… ${progress.toFixed(0)}%` : `run ${picked.length}`}
          </button>
          {busy && (
            <div className="progress" style={{ marginTop: 8 }}>
              <div style={{ width: `${progress}%` }} />
            </div>
          )}
        </div>
      </div>

      <div>
        {Object.entries(errors).map(([name, msg]) => (
          <div key={name} className="banner error" style={{ marginBottom: 8 }}>
            {name}: {msg}
          </div>
        ))}

        {rows.length > 0 && (
          <div className="card">
            <h3>Ranking at bad {threshold}px</h3>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>algorithm</th>
                    <th>bad {threshold}px</th>
                    <th>EPE</th>
                    <th>density</th>
                    <th>wall s</th>
                    <th>cpu s</th>
                    <th>cores</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map(({ name, res }) => (
                    <tr key={name}>
                      <td>
                        <FamilyDot family={res!.family} /> {name}
                      </td>
                      <td className="num">
                        {fmt(res!.metrics?.bad[String(threshold)], 2, '%')}
                      </td>
                      <td className="num">{fmt(res!.metrics?.epe_valid)}</td>
                      <td className="num">
                        {fmt((res!.metrics?.density ?? 0) * 100, 1, '%')}
                      </td>
                      <td className="num">{fmt(res!.timing.wall_s, 3)}</td>
                      <td className="num">{fmt(res!.timing.cpu_s, 2)}</td>
                      <td className="num">{fmt(res!.timing.parallelism, 1, 'x')}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        <div className="card">
          <h3>Disparity maps &middot; shared colour scale</h3>
          <div className="img-grid">
            <ImageTile
              title="left image"
              src={sceneImageUrl(sceneId, 'left', downscale)}
            />
            {scene?.has_gt && (
              <ImageTile
                title="ground truth"
                src={sceneImageUrl(sceneId, 'gt', downscale)}
              />
            )}
            {rows.map(({ name, res }) => (
              <ImageTile
                key={name}
                title={name}
                note={fmt(res!.metrics?.bad[String(threshold)], 1, '%')}
                src={resultImageUrl(res!.result_id, 'disparity')}
              />
            ))}
          </div>
        </div>

        {rows.length > 0 && scene?.has_gt && (
          <div className="card">
            <h3>Where each one is wrong &middot; &gt; {threshold}px</h3>
            <div className="img-grid">
              {rows.map(({ name, res }) => (
                <ImageTile
                  key={name}
                  title={name}
                  note={fmt(res!.metrics?.bad[String(threshold)], 1, '%')}
                  src={resultImageUrl(res!.result_id, 'bad', { threshold })}
                />
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
