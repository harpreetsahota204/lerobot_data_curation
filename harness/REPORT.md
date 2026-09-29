# Validation report

Generated 2026-09-29 13:47 on `lr_dev` (102 episodes) in 18 s. Regenerate with `python -m lerobot_data_curation.harness.run_harness --dataset lr_dev`.

## Corruption suite

Each applicable episode is paired with a corrupted copy of itself. A metric passes when the copy ranks worse than its own original in at least 80% of pairs (flags: when the flag fires on the copy and not on the original). The identity control must move in at most 5% of pairs.

| Corruption | Metric | Pairs | Worse than original | Status |
|---|---|---|---|---|
| identity | `(every metric, worst case)` | 102 | 0% | pass |
| jitter | `ldlj` | 100 | 88% | pass |
| stop_and_go | `sparc` | 100 | 91% | pass |
| stop_and_go | `ldlj` | 100 | 83% | pass |
| wander | `path_length_z` | 100 | 96% | pass |
| idle_start | `idle_lead_s` | 100 | 100% | pass |
| idle_end | `idle_trail_s` | 100 | 100% | pass |
| mid_pause | `longest_pause_s` | 100 | 93% | pass |
| cut_off | `end_motion_ratio` | 72 | 100% | pass |
| abandoned | `length_z` | 100 | 100% | pass |
| duplicate_frame | `frame_gaps` | 102 | 100% | pass |
| nan | `nonfinite_values` | 100 | 100% | pass |
| one_frame | `too_short` | 100 | 100% | pass |
| generic_task | `task_generic` | 102 | 98% | pass |
| missing_task | `task_missing` | 102 | 100% | pass |
| length_mismatch | `length_mismatch` | 102 | 100% | pass |
| video_off | `video_window_mismatch` | 102 | 100% | pass |
| stalled_joint | `track_resid` | 86 | 95% | pass |
| lagged_state | `track_lag_ms` | 86 | 100% | pass |
| jolts | `accel_spike_frac` | 100 | 97% | pass |
| at_limits | `joint_limit_frac` | 98 | 100% | pass |
| gripper_chatter | `gripper_flips_per_s` | 70 | 100% | pass |

## False-alarm rate of the flags on unmodified episodes

| Flag | Fires on |
|---|---|
| `frame_gaps` | 0% of episodes |
| `length_mismatch` | 0% of episodes |
| `nonfinite_values` | 0% of episodes |
| `recovery_count` | 13% of episodes |
| `schema_mismatch` | 0% of episodes |
| `task_generic` | 2% of episodes |
| `task_missing` | 0% of episodes |
| `too_short` | 0% of episodes |
| `video_window_mismatch` | 1% of episodes |

## Length control

Every episode longer than 11.0 s is truncated to 11.0 s and re-scored (91 episodes). The table is the Spearman correlation between a metric's z-score before and after truncation. A metric whose ranking collapses (below 0.5) is length-confounded: it measures how long an episode is more than how good it is.

| Metric | Episodes | Rank correlation before vs after | Correlation with duration | Status |
|---|---|---|---|---|
| `sparc` | 91 | 0.81 | 0.17 | ok |
| `ldlj` | 91 | 0.95 | 0.01 | ok |
| `sparc_phase` | 57 | 0.82 | 0.54 | ok |
| `idle_lead_s` | 91 | 0.93 | 0.07 | ok |
| `idle_trail_s` | 91 | 0.39 | 0.02 | not judged: measures the end, which truncation removes |
| `longest_pause_s` | 91 | 0.60 | 0.19 | ok |
| `end_motion_ratio` | 91 | 0.14 | 0.05 | not judged: measures the end, which truncation removes |
| `track_resid` | 79 | 0.88 | -0.01 | ok |
| `accel_spike_frac` | 91 | 0.96 | -0.07 | ok |
| `joint_limit_frac` | 90 | 0.80 | -0.09 | ok |
| `gripper_flips_per_s` | 57 | 0.80 | -0.38 | ok |
| `missed_grasp_frac` | 47 | 0.83 | -0.08 | ok |
| `action_divergence` | 0 | n/a | n/a | too few episodes |

## Not testable on this data

| Metric | Why |
|---|---|
| `action_divergence` | needs at least 5 episodes per source; lr_dev has 2 |
| `stats_leverage` | needs at least 3 episodes per source; lr_dev has 2 |
| `sparc_phase` | synthetic tests only (needs a named gripper and clear phases) |
| `missed_grasp_frac` | synthetic tests only (needs a known gripper convention) |
| `recovery_count` | synthetic tests only (needs a known gripper convention) |
| `schema_mismatch` | a dataset-level comparison, tested synthetically |
| `iforest_score` | outlier models, tested synthetically |
| `novelty_knn` | outlier models, tested synthetically |
| `is_outlier` | outlier models, tested synthetically |
| `jerk_rms` | never scored; covered by the jitter direction test |
| `psd_lf_hf` | never scored; covered by the jitter direction test |
| `idle_frac` | never scored; context only |
| `timestamp_dev` | info only |

## Caveats

- The data is `lr_dev`: 2 episodes from each of 51 sources, so every episode is normalized against the pooled view. Pairing each episode with its own corrupted copy removes the cross-robot variance, but this is a test of metric direction, not of ranking quality on a real curation task.
- Corruptions are synthetic. They show a metric responds to the defect it targets, not that the defect predicts a worse policy (see the evidence notes in METRICS.md).
