export const PLUGIN = "lerobot-data-curation";

export interface ProfileScore {
  score: number | null;
  n_flags: number;
  driver: string | null;
  verdict: "pass" | "warn" | "fail" | "unknown";
}

export interface Row {
  id: string;
  episode: string;
  source: string;
  task: string;
  robot: string;
  group: string;
  group_basis: "task" | "pooled";
  group_n: number;
  integrity: string;
  language: string;
  profiles: Record<string, ProfileScore>;
  values: Record<string, number | null>;
  raw: Record<string, number | null>;
  z: Record<string, number | null>;
  notes: Record<string, string>;
  /** metric -> {signal: value} for per-arm metrics */
  by_signal: Record<string, Record<string, number | null>>;
}

export interface MetricMeta {
  family: string;
  group: string | null;
  description: string;
  higher_is_worse: boolean;
  scored: boolean;
  kind: "value" | "signed_z" | "flag";
  per_signal: boolean;
  opt_in: boolean;
  /** "raw" when the panel shows the raw value of a signed-z metric (for example lag in ms) */
  display: "raw" | "value";
  /** integrity checks only: [level, threshold]; the verdict gets `level` when the value exceeds it */
  check: [string, number] | null;
}

export interface Balance {
  tasks: Record<string, number>;
  sources: Record<string, number>;
  under_covered: string[];
  dominant_sources: string[];
}

export interface PanelData {
  scored: boolean;
  rows: Row[];
  metrics: Record<string, MetricMeta>;
  computed: string[];
  signals: string[];
  profiles: { id: string; label: string }[];
  default_profile: string;
  tasks: { task: string; n: number }[];
  warn_thresholds: Record<string, Record<string, number>>;
  balance: Balance;
  min_group: number;
  pooled_count: number;
  mixed_runs: boolean;
  under_covered_below: number;
  warn_z: number;
  fail_z: number;
  config_version_mismatch: boolean;
}

export interface HistogramBar {
  x: number;
  x0: number;
  x1: number;
  /** episode count per series (signal); "" for a single series */
  counts: Record<string, number>;
}

/** Human-readable titles and short column headers per metric. */
export const METRIC_LABELS: Record<string, { title: string; short: string }> = {
  sparc: { title: "Smoothness (SPARC)", short: "SPARC" },
  ldlj: { title: "Normalized jerk (LDLJ)", short: "LDLJ" },
  sparc_phase: { title: "Phase-aware smoothness", short: "SPARC ph." },
  jerk_rms: { title: "Jerk intensity (RMS)", short: "Jerk RMS" },
  psd_lf_hf: { title: "Low/high frequency ratio", short: "PSD LF/HF" },
  idle_lead_s: { title: "Idle at start (s)", short: "Idle start" },
  idle_trail_s: { title: "Idle at end (s)", short: "Idle end" },
  longest_pause_s: { title: "Longest mid-task pause (s)", short: "Pause" },
  end_motion_ratio: { title: "Motion at the end", short: "End motion" },
  idle_frac: { title: "Idle fraction", short: "Idle frac" },
  length_z: { title: "Episode length (z)", short: "Length z" },
  path_length_z: { title: "Path length (z)", short: "Path z" },
  track_lag_ms: { title: "Tracking lag (ms)", short: "Lag ms" },
  track_resid: { title: "Tracking residual", short: "Track res." },
  accel_spike_frac: { title: "Acceleration spikes", short: "Spikes" },
  joint_limit_frac: { title: "Near joint limits", short: "Limits" },
  gripper_flips_per_s: { title: "Gripper flips per second", short: "Flips/s" },
  missed_grasp_frac: { title: "Missed grasps", short: "Missed" },
  recovery_count: { title: "Regrasp recoveries", short: "Recov." },
  action_divergence: { title: "Action divergence", short: "Divergence" },
  stats_leverage: { title: "Stats leverage", short: "Leverage" },
  iforest_score: { title: "Isolation-forest score", short: "iForest" },
  novelty_knn: { title: "Novelty (kNN distance)", short: "Novelty" },
  blur: { title: "Sharpness (log Laplacian variance)", short: "Sharpness" },
  exposure_err: { title: "Exposure error", short: "Exposure" },
  clipped_frac: { title: "Clipped pixels", short: "Clipped" },
  frozen_frac: { title: "Frozen camera feed", short: "Frozen" },
  video_action_lag_ms: { title: "Video-action lag (ms)", short: "V-A lag" },
  frame_gaps: { title: "Frame gaps", short: "Gaps" },
  timestamp_dev: { title: "Timestamp deviation (s)", short: "Time dev" },
  length_mismatch: { title: "Length mismatch", short: "Length" },
  video_window_mismatch: { title: "Video window mismatch (frames)", short: "Video win." },
  nonfinite_values: { title: "Non-finite values", short: "NaN/inf" },
  too_short: { title: "Too short", short: "Short" },
  schema_mismatch: { title: "Schema mismatch", short: "Schema" },
  task_missing: { title: "Task missing", short: "No task" },
  task_generic: { title: "Task too generic", short: "Generic" },
};

export const label = (metric: string) => METRIC_LABELS[metric] ?? { title: metric, short: metric };

export interface Span {
  start_s: number;
  end_s: number;
  label: string;
  kind: "idle" | "pause" | "rough" | "spike" | "recovery";
}

export interface EpisodeDetail {
  error?: string;
  sample_id: string;
  episode: string;
  task: string;
  fps: number;
  length: number;
  duration_s: number;
  stride: number;
  t: number[];
  joint_names: string[];
  action: number[][] | null;
  state: number[][] | null;
  speed: { t: number[]; v: number[] } | null;
  idle_threshold: number | null;
  grippers: Record<string, { t: number[]; v: number[]; events: { t: number; kind: "open" | "close" }[] }>;
  spans: Span[];
  thumbnails: { camera: string; t: number; jpeg: string }[];
  missing: string[];
}
