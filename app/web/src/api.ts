import type {
  Algorithm, ComputeResult, DatasetCollection, Job, RunDetail, RunSummary,
  Scene, SceneRules, SynthDataset, SynthPreview, SynthSchema,
} from './types'

const BASE = '/api'

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(BASE + path, {
    headers: init?.body instanceof FormData
      ? undefined
      : { 'Content-Type': 'application/json' },
    ...init,
  })
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`
    try {
      const body = await res.json()
      if (body?.detail) detail = String(body.detail)
    } catch { /* non-JSON error body */ }
    throw new Error(detail)
  }
  return res.json() as Promise<T>
}

export const api = {
  health: () => req<{ status: string; machine: Record<string, unknown> }>('/health'),
  algorithms: () => req<Algorithm[]>('/algorithms'),
  scenes: () => req<Scene[]>('/scenes'),

  /** One scene by id, for pairs the catalogue listing does not include. */
  scene: (id: string) =>
    req<Scene>(`/scene?${new URLSearchParams({ id })}`),

  /** Whole datasets, as opposed to the individual pairs `/scenes` lists. */
  datasets: () => req<DatasetCollection[]>('/datasets'),

  /** The pairs inside one dataset. */
  datasetScenes: (id: string, limit?: number) => {
    const q = new URLSearchParams({ id })
    if (limit != null) q.set('limit', String(limit))
    return req<{ id: string; scene_ids: string[]; total: number }>(
      `/dataset/scenes?${q}`,
    )
  },

  /** Exactly how many pairs a mixed selection comes to. Resolved by the
   *  server so the cell count the button promises is the one that runs:
   *  picking a dataset and a subset of it must not double-count. */
  resolveSelection: (body: {
    scene_ids?: string[]
    dataset_ids?: string[]
    limit?: number | null
  }) => req<{ total: number; scene_ids: string[] }>('/datasets/resolve', {
    method: 'POST', body: JSON.stringify(body),
  }),

  compute: (body: {
    scene_id: string
    algorithm: string
    params: Record<string, unknown>
    downscale?: number
    repeats?: number
    eval_region?: string
    thresholds?: number[]
    reuse?: boolean
  }) => req<ComputeResult>('/compute', { method: 'POST', body: JSON.stringify(body) }),

  rescore: (body: {
    result_id: string
    eval_region: string
    thresholds?: number[]
  }) => req<ComputeResult>('/rescore', { method: 'POST', body: JSON.stringify(body) }),

  runs: () => req<RunSummary[]>('/runs'),
  run: (id: string) => req<RunDetail>(`/runs/${encodeURIComponent(id)}`),
  runFigures: (id: string) => req<string[]>(`/runs/${encodeURIComponent(id)}/figures`),

  startBenchmark: (body: {
    scene_ids?: string[]
    dataset_ids?: string[]
    limit?: number | null
    algorithms: string[]
    repeats?: number
    downscale?: number
    eval_region?: string
    save_figures?: boolean
  }) => req<Job>('/jobs/benchmark', { method: 'POST', body: JSON.stringify(body) }),
  jobs: () => req<Job[]>('/jobs'),
  job: (id: string) => req<Job>(`/jobs/${encodeURIComponent(id)}`),
  cancelJob: (id: string) =>
    req<Job>(`/jobs/${encodeURIComponent(id)}/cancel`, { method: 'POST' }),

  addDataset: (form: FormData) =>
    req<Scene>('/datasets', { method: 'POST', body: form }),
  deleteDataset: (name: string) =>
    req<{ deleted: string }>(`/datasets/${encodeURIComponent(name)}`, {
      method: 'DELETE',
    }),
  synthetic: (seed: number) =>
    req<Scene>('/datasets/synthetic', {
      method: 'POST', body: JSON.stringify({ seed }),
    }),

  /* ---------------------------------------------------- the synthesizer */

  synthSchema: () => req<SynthSchema>('/synth/schema'),

  synthPreview: (rules: SceneRules, samples = 3) =>
    req<SynthPreview>('/synth/preview', {
      method: 'POST', body: JSON.stringify({ rules, samples }),
    }),

  synthDatasets: () => req<SynthDataset[]>('/synth/datasets'),

  deleteSynthDataset: (name: string) =>
    req<{ deleted: string }>(`/synth/datasets/${encodeURIComponent(name)}`, {
      method: 'DELETE',
    }),

  startSynthesis: (body: {
    name: string
    rules: SceneRules
    count: number
    workers?: number | null
    environments?: string[] | null
    splits?: Record<string, number> | null
    append?: boolean
  }) => req<Job>('/jobs/synth', { method: 'POST', body: JSON.stringify(body) }),
}

/** A preview image. Previews are regenerated on every rule change, so the
 *  URL carries the preview id and the server sends `no-store`. */
export function synthPreviewUrl(previewId: string, index: number, view: string) {
  const q = new URLSearchParams({
    preview_id: previewId, index: String(index), view,
  })
  return `${BASE}/synth/preview/image?${q}`
}

/** Image URLs are plain GETs so the browser caches and streams them. */
export function sceneImageUrl(id: string, view: string, downscale = 1) {
  const q = new URLSearchParams({ id, view, downscale: String(downscale) })
  return `${BASE}/scene/image?${q}`
}

export function resultImageUrl(
  resultId: string,
  view: string,
  opts: { threshold?: number; maxError?: number } = {},
) {
  const q = new URLSearchParams({ result_id: resultId, view })
  if (opts.threshold != null) q.set('threshold', String(opts.threshold))
  if (opts.maxError != null) q.set('max_error', String(opts.maxError))
  return `${BASE}/result/image?${q}`
}

export function runFigureUrl(runId: string, name: string) {
  return `${BASE}/runs/${encodeURIComponent(runId)}/figure?name=${
    encodeURIComponent(name)
  }`
}

/** The narrative report for a run. Opened in a new tab rather than fetched:
 *  it is a whole self-contained document, not data for this UI to render. */
export function runReportUrl(
  runId: string, format: 'html' | 'pdf' = 'html', refresh = false,
) {
  return `${BASE}/runs/${encodeURIComponent(runId)}/report?format=${format}${
    refresh ? '&refresh=1' : ''
  }`
}
