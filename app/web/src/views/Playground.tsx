import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api, resultImageUrl, sceneImageUrl } from '../api'
import {
  AlgorithmPicker, ImageTile, MetricsPanel, ParamControls, SceneSelect,
  useDebounced, useScene,
} from '../components'
import type {
  Algorithm, ComputeResult, DatasetCollection, Scene,
} from '../types'

const THRESHOLDS = [0.5, 1, 2, 4]

export default function Playground({
  scenes, algorithms, collections,
}: {
  scenes: Scene[]
  algorithms: Algorithm[]
  collections: DatasetCollection[]
}) {
  const [sceneId, setSceneId] = useState('middlebury/Teddy')
  const [algoName, setAlgoName] = useState('scratch_census_sgm4')
  const [params, setParams] = useState<Record<string, unknown>>({})
  const [downscale, setDownscale] = useState(1)
  const [repeats, setRepeats] = useState(1)
  const [evalRegion, setEvalRegion] = useState('nonocc')
  const [threshold, setThreshold] = useState(2)
  const [autoRun, setAutoRun] = useState(true)

  const [result, setResult] = useState<ComputeResult | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const algo = useMemo(
    () => algorithms.find((a) => a.name === algoName),
    [algorithms, algoName],
  )
  const scene = useScene(scenes, sceneId)

  // Reset parameters to the algorithm's shipped defaults when it changes.
  const resetParams = useCallback(() => {
    if (!algo) return
    const next: Record<string, unknown> = {}
    for (const p of algo.params) next[p.name] = p.value
    setParams(next)
  }, [algo])
  useEffect(() => { resetParams() }, [resetParams])

  // Debounce so dragging a slider does not fire a run per pixel.
  const debouncedParams = useDebounced(params, 350)
  const runToken = useRef(0)

  const run = useCallback(async (overrideParams?: Record<string, unknown>) => {
    if (!algo?.available) return
    const token = ++runToken.current
    setBusy(true)
    setError(null)
    try {
      const res = await api.compute({
        scene_id: sceneId,
        algorithm: algoName,
        params: overrideParams ?? params,
        downscale,
        repeats,
        eval_region: evalRegion,
        thresholds: THRESHOLDS,
      })
      if (token === runToken.current) setResult(res)
    } catch (e) {
      if (token === runToken.current) {
        setError(e instanceof Error ? e.message : String(e))
        setResult(null)
      }
    } finally {
      if (token === runToken.current) setBusy(false)
    }
    // `params` is intentionally excluded: the auto-run effect passes the
    // debounced copy explicitly, so including it would double-fire.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [algo, algoName, sceneId, downscale, repeats, evalRegion])

  useEffect(() => {
    if (!autoRun || !algo?.available) return
    if (Object.keys(debouncedParams).length === 0) return
    void run(debouncedParams)
  }, [autoRun, algo, debouncedParams, run])

  // Re-scoring is a different operation from re-running: it reuses the stored
  // disparity map, so changing the region or threshold is instant.
  useEffect(() => {
    if (!result) return
    let cancelled = false
    api.rescore({
      result_id: result.result_id,
      eval_region: evalRegion,
      thresholds: THRESHOLDS,
    })
      .then((r) => { if (!cancelled) setResult(r) })
      .catch(() => { /* the result expired; the next run will refresh it */ })
    return () => { cancelled = true }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [evalRegion])

  const rid = result?.result_id
  const stale = result
    && JSON.stringify(result.params) !== JSON.stringify(
      Object.fromEntries(Object.entries(params).filter(([k]) => k in result.params)),
    )

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
              <b>downscale</b>
              <span className="val">{downscale}x</span>
            </span>
            <input
              type="range" min={1} max={8} step={1}
              value={downscale}
              onChange={(e) => setDownscale(Number(e.target.value))}
            />
            <div className="hint">
              Shrinks the pair and its disparity range together. Use it to make
              the from-scratch matchers fast enough to tweak interactively.
            </div>
          </label>
          {scene && (
            <div className="card-row">
              <span className="chip">{scene.width}x{scene.height}</span>
              <span className="chip">ndisp {scene.ndisp}</span>
              <span className="chip">
                {scene.has_gt ? `GT ${fmtPct(scene.gt_coverage)}` : 'no GT'}
              </span>
            </div>
          )}
        </div>

        <div className="card">
          <h3>Algorithm</h3>
          <label className="field">
            <span className="lbl"><b>matcher</b></span>
            <AlgorithmPicker
              algorithms={algorithms} value={algoName} onChange={setAlgoName}
            />
          </label>
          {algo && <div className="hint">{algo.description}</div>}
          {algo && !algo.available && (
            <div className="banner error" style={{ marginTop: 8 }}>
              {algo.reason}
            </div>
          )}
          <div className="card-row" style={{ marginTop: 10 }}>
            {algo?.tags.map((t) => <span key={t} className="chip">{t}</span>)}
          </div>
        </div>

        {algo && (
          <ParamControls
            algorithm={algo}
            values={params}
            onChange={(n, v) => setParams((p) => ({ ...p, [n]: v }))}
            onReset={resetParams}
          />
        )}

        <div className="card">
          <h3>Run</h3>
          <div className="card-row">
            <button
              className="primary"
              onClick={() => void run()}
              disabled={busy || !algo?.available}
            >
              {busy ? 'running…' : stale ? 'run (changed)' : 'run again'}
            </button>
            <label className="card-row" style={{ gap: 6, fontSize: 12.5 }}>
              <input
                type="checkbox"
                checked={autoRun}
                onChange={(e) => setAutoRun(e.target.checked)}
              />
              auto-run
            </label>
          </div>
          <label className="field" style={{ marginTop: 11 }}>
            <span className="lbl">
              <b>timing repeats</b><span className="val">{repeats}</span>
            </span>
            <input
              type="range" min={1} max={9} step={1}
              value={repeats}
              onChange={(e) => setRepeats(Number(e.target.value))}
            />
            <div className="hint">
              Wall time is the median of this many runs. One repeat is fine for
              accuracy but noisy for timing.
            </div>
          </label>
        </div>
      </div>

      <div>
        {error && <div className="banner error" style={{ marginBottom: 12 }}>{error}</div>}

        <div className="card">
          <div className="row-between" style={{ marginBottom: 10 }}>
            <h3 style={{ margin: 0 }}>Scoring</h3>
            <span className="muted" style={{ fontSize: 11.5 }}>
              changes here re-score the stored map — no re-matching
            </span>
          </div>
          <div className="card-row">
            <label className="card-row" style={{ gap: 6, fontSize: 12.5 }}>
              region
              <select
                value={evalRegion}
                onChange={(e) => setEvalRegion(e.target.value)}
                style={{ width: 'auto' }}
              >
                <option value="nonocc">non-occluded</option>
                <option value="all">all with ground truth</option>
              </select>
            </label>
            <label className="card-row" style={{ gap: 6, fontSize: 12.5 }}>
              bad threshold
              <select
                value={threshold}
                onChange={(e) => setThreshold(Number(e.target.value))}
                style={{ width: 'auto' }}
              >
                {THRESHOLDS.map((t) => (
                  <option key={t} value={t}>{t} px</option>
                ))}
              </select>
            </label>
            {busy && <span className="spin" />}
          </div>
        </div>

        <div className="card">
          <h3>Views</h3>
          <div className="img-grid">
            <ImageTile
              title="left image"
              src={sceneImageUrl(sceneId, 'left', downscale)}
            />
            <ImageTile
              title="ground truth"
              note={scene?.has_gt ? 'shared colour scale' : undefined}
              src={scene?.has_gt ? sceneImageUrl(sceneId, 'gt', downscale) : undefined}
              empty="no ground truth for this scene"
            />
            <ImageTile
              title="disparity"
              note={result ? result.algorithm : undefined}
              src={rid ? resultImageUrl(rid, 'disparity') : undefined}
              empty="run the matcher to see its output"
            />
            <ImageTile
              title={`bad pixels > ${threshold}px`}
              note="red = wrong, dark = no estimate"
              src={rid && result?.has_gt
                ? resultImageUrl(rid, 'bad', { threshold })
                : undefined}
              empty="needs ground truth"
            />
            <ImageTile
              title="error magnitude"
              note="clipped at 4px"
              src={rid && result?.has_gt
                ? resultImageUrl(rid, 'error', { maxError: 4 })
                : undefined}
              empty="needs ground truth"
            />
            <ImageTile
              title="where it answered"
              note="dark = declined"
              src={rid ? resultImageUrl(rid, 'valid') : undefined}
              empty="run the matcher first"
            />
          </div>
        </div>

        <MetricsPanel result={result} threshold={threshold} />
      </div>
    </div>
  )
}

function fmtPct(v: number | null | undefined) {
  return v == null ? '—' : `${(v * 100).toFixed(0)}%`
}
