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

/* ------------------------------------------------------- the synthesizer */

/** A rule set, as the generator accepts it. Values stay loosely typed: the
 *  editor is built from the server's schema, so the frontend never needs to
 *  know the field list and cannot drift out of step with the generator. */
export type SceneRules = Record<string, unknown>

export type RuleFieldType = 'int' | 'float' | 'bool' | 'choice' | 'multichoice'

export interface RuleField {
  name: string
  type: RuleFieldType
  group: string
  label: string
  help: string
  default: unknown
  min?: number
  max?: number
  step?: number
  options?: { value: string; help?: string }[]
}

export interface SynthSchema {
  fields: RuleField[]
  environments: { value: string; label: string; help: string }[]
  difficulties: { value: string; settings: Record<string, unknown> }[]
  texture_kinds: string[]
  defaults: SceneRules
  presets: Record<string, SceneRules>
}

/** Measured properties of one generated pair — reported, not assumed. */
export interface SynthStats {
  gt_coverage: number
  nonocc_fraction: number
  occluded_fraction: number
  out_of_range_fraction: number
  disparity_min: number | null
  disparity_max: number | null
  disparity_mean: number | null
  depth_min_m: number | null
  depth_max_m: number | null
  sky_fraction: number
  edge_fraction: number
  baseline_m: number
  focal_px: number
}

export interface SynthSample {
  index: number
  seed: number
  name: string
  width: number
  height: number
  ndisp: number
  stats: SynthStats
}

export interface SynthPreview {
  preview_id: string
  rules: SceneRules
  summary: Partial<SynthStats>
  warnings: string[]
  samples: SynthSample[]
}

export interface SynthDataset {
  name: string
  path?: string
  count: number
  created_utc?: string | null
  elapsed_s?: number | null
  seconds_per_pair?: number | null
  environments?: string[]
  splits?: Record<string, number>
  rules?: SceneRules
  summary?: Record<string, { min: number; mean: number; max: number } | null>
  cancelled?: boolean
  error?: string
}
