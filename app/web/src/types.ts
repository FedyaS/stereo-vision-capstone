export type ParamKind = 'int' | 'float' | 'bool' | 'choice'

export interface ParamSpec {
  name: string
  kind: ParamKind
  value: number | boolean | string | number[]
  tuned: boolean
  hint?: string
  choices?: (string | number | number[])[]
  min?: number
  max?: number
  step?: number
  odd_only?: boolean
  swept?: number[]
}

export interface Algorithm {
  name: string
  family: 'opencv' | 'scratch' | 'torch' | 'learned'
  description: string
  tags: string[]
  available: boolean
  reason: string
  params: ParamSpec[]
}

export interface Scene {
  id: string
  kind: string
  name: string
  width: number
  height: number
  ndisp: number
  has_gt: boolean
  megapixels: number
  gt_coverage: number | null
  error?: string
}

export interface Timing {
  wall_s: number | null
  wall_s_min: number | null
  cpu_s: number | null
  parallelism: number | null
  peak_rss_mb: number | null
  rss_delta_mb: number | null
  mdps: number | null
  mpix_per_s: number | null
  repeats: number
}

export interface Metrics {
  bad: Record<string, number | null>
  bad_valid: Record<string, number | null>
  epe_valid: number | null
  epe_filled: number | null
  rmse_valid: number | null
  rmse_filled: number | null
  bias_valid: number | null
  a90: number | null
  a95: number | null
  a99: number | null
  density: number | null
  n_eval_px: number
  eval_region: string
}

export interface ComputeResult {
  result_id: string
  scene_id: string
  algorithm: string
  family: string
  params: Record<string, unknown>
  downscale: number
  width: number
  height: number
  ndisp: number
  has_gt: boolean
  timing: Timing
  disparity_range: [number | null, number | null]
  metrics: Metrics | null
  extra: Record<string, unknown>
}

export interface RunSummary {
  id: string
  n_cells: number
  n_ok: number
  algorithms: string[]
  scenes: string[]
  created_utc: string | null
  elapsed_s: number | null
  config: Record<string, unknown> | null
  has_summary: boolean
}

export type RunRow = Record<string, string | number | null>

export interface RunDetail {
  id: string
  rows: RunRow[]
  summary: string | null
}

export interface Job {
  id: string
  kind: string
  total: number
  done: number
  status: 'running' | 'done' | 'error' | 'cancelled'
  message: string
  current: string
  out_dir: string | null
  elapsed_s: number
  log: string[]
}
