# LeRobot Data Curation: Metrics Guide

This guide explains every metric the plugin computes for a LeRobot episode, how to read it, and how the metrics combine into one ranking score.

Three rules apply everywhere:

- Scores rank your review queue. Nothing is deleted, hidden or rewritten automatically.
- Every score is compared against a group of similar episodes (see [Normalization](#normalization)), never against the whole dataset unless a fallback says so.
- Inside a score, every metric is oriented so that higher means worse. The tables below show each metric's raw direction.

## Dataset-level checks

These run once before any episode metric. Each one infers a value from `info.json`, joint names and value ranges, shows it to you, and lets you confirm or override it. A metric that depends on an assumption you have not confirmed switches off and says why.

| Check | What it decides | Why it matters |
|---|---|---|
| `feature_map` | Which columns are state, action and cameras | Some datasets use non-standard keys. Every metric reads its arrays through this map, and a metric whose array is missing switches off |
| `action_semantics` | Whether `action` is leader joint positions, deltas, end-effector pose or unknown | Decides whether `track_lag_ms` and `track_resid` can run |
| `joint_units` | Degrees, radians or normalized (-100..100) | Range normalization and thresholds depend on it |
| `gripper_convention` | Which direction is closed, and which dimensions are grippers | Needed by the gripper metrics and `sparc_phase` |
| `camera_roles` | Wrist, top, side or ego camera, from the feature key | Used to label cameras |
| `balance` | Episodes per canonical task and per source | Shows under-covered tasks and dominant sources. Reported, not scored |

## Signals: which array each metric reads

| Metric family | Reads | Why |
|---|---|---|
| Smoothness (`sparc`, `ldlj`, `sparc_phase`, `jerk_rms`, `psd_lf_hf`) | `action` by default, with a signal picker | `action` is what a policy learns to output. On leader-follower rigs (SO-100, Koch, ALOHA) it is the operator's hand, not the robot. `observation.state` is selectable |
| Contact and jolt (`accel_spike_frac`, `joint_limit_frac`, `track_resid`, `track_lag_ms`) | `observation.state`. Tracking metrics use both | Contact, jolts and limits happen to the follower robot, not the leader |
| Time and gripper metrics | `action` for speed, gripper dimensions by name | Idle and pause are properties of commanded motion |

Smoothness processing:

- Each joint is range-normalized before speeds are combined, so a joint with a large range does not dominate and units do not matter.
- Gripper dimensions are excluded from arm speed.
- The low-pass cutoff is `min(10 Hz, 0.4 x fps)`, so 30 fps data is not analyzed near Nyquist.
- Metrics run on fixed-length windows, because LDLJ is duration-sensitive.

## Motion

| Metric | What it measures | How to read it |
|---|---|---|
| `sparc` | Spectral arc length of the speed profile: how many corrections the motion contains | Closer to 0 is smoother. Very negative is fragmented, hesitant motion. Scale-invariant, so it is the primary smoothness metric. It responds to stop-and-go motion, not to additive white noise |
| `ldlj` | Log dimensionless jerk of the speed profile | Closer to 0 is smoother. Responds to white noise. Noisier and duration-sensitive, so it carries half weight |
| `sparc_phase` (opt-in) | SPARC inside each gripper-delimited phase, rolled up by median | Same reading as `sparc`. Stops contact transitions from counting as roughness. Needs a confirmed gripper dimension. Phases with too few samples are skipped |
| `jerk_rms` (opt-in, never scored) | RMS jerk after a low-pass filter | Lower is smoother. Noise-sensitive and correlated with the others |
| `psd_lf_hf` (opt-in, never scored) | Log ratio of low- to high-frequency power in the speed profile | Higher is smoother. Unreliable at LeRobot frame rates, where tremor sits near Nyquist |

Each scored motion metric also has a worst-window value, so one bad stretch is not averaged away.

## Time efficiency

| Metric | What it measures | How to read it |
|---|---|---|
| `idle_frac` (never scored) | Fraction of frames below the idle speed (a multiple of the episode's own moving speed) | Context only. A high value can be a legitimate pause |
| `idle_lead_s` | Seconds of stillness at the start | High means dead time before the task begins. Also stores a suggested trim start (`keep_from_s`) |
| `idle_trail_s` | Seconds of stillness at the end | High means a long dead tail. Half weight, since fixed-duration recordings produce this from a timer. Also stores a suggested trim end (`keep_to_s`) |
| `longest_pause_s` | Longest idle run that is not at either end | High means a mid-task stall or hesitation. A pause next to a regrasp counts as recovery and is excluded |
| `end_motion_ratio` | Speed in the last ~1 s relative to the episode's typical moving speed | High means the episode probably ended mid-motion. Half weight, same timer caveat as `idle_trail_s` |
| `length_z` | Robust z of episode duration within its group, stored signed | Scored as `abs(z)`. Positive means unusually long (struggle), negative means unusually short (abandoned) |
| `path_length_z` (opt-in, never scored) | Robust z of total joint-space path length | High means a longer, wandering route than peers. Correlated with `length_z`, so shown for review only |

`length_z`, `path_length_z` and `idle_frac` depend on episode length by construction. Only `length_z` is scored, and it shares the `time` group's single vote with the idle metrics.

## Tracking and contact

| Metric | What it measures | How to read it |
|---|---|---|
| `track_lag_ms` | Delay between the commanded `action` and the achieved `state`, from cross-correlation | Mostly a hardware property, so it is summarized per dataset or session. Only episodes far from the dataset median are flagged, and it never enters a score. Runs only when `action_semantics` says action and state share a joint space |
| `track_resid` | Per-joint normalized gap between action and state after lag alignment, with each joint's median offset removed first (arm joints only) | High means the robot is not reaching what was commanded (slip, collision, stall). Also raises interval flags |
| `accel_spike_frac` | Fraction of frames where any joint's `state` acceleration exceeds median + k x MAD, with an absolute floor | High means sudden jolts, a proxy for collisions or contact. The floor stops idle joints from making every frame a spike |
| `joint_limit_frac` (opt-in) | Share of (frame, joint) pairs within a margin (default 2% of range) of a joint limit, averaged over the joints that move in the episode | High means the operator worked near the robot's limits. Uses your limits when given, otherwise `meta/stats.json` min and max, which is a weak proxy because observed extremes always sit at the edge of the observed range |

## Gripper

| Metric | What it measures | How to read it |
|---|---|---|
| `gripper_flips_per_s` | Open/close transitions per second on gripper dimensions, with hysteresis so noise near the threshold does not count | High means chatter or a hesitant operator. Several clean cycles are often recoveries, which is why `recovery_count` exists |
| `missed_grasp_frac` (opt-in) | Fraction of close commands where the follower closes fully with no stall residual (nothing was held) | High means empty grasps. Needs a confirmed gripper convention and is heuristic |
| `recovery_count` (never scored) | Number of detected regrasps (open, then close again near the same place) | Neutral or positive. Writes a `recovery` tag. Recovery is never penalized |

## Consistency

| Metric | What it measures | How to read it |
|---|---|---|
| `action_divergence` | Variance of `action` among the nearest `state` neighbors in other episodes of the same group | High means the demonstration takes a different action from peers in a similar state. Needs enough episodes in the group. Similar states can legitimately need different actions, so treat it as a review signal |
| `stats_leverage` (never scored) | How many feature dimensions this episode alone stretches beyond the rest of the dataset's `meta/stats.json` bounds | High means this episode distorts the normalization every training run will use |

## Integrity

Pass, warn or fail flags plus counts. They mean "this episode is broken", not "this episode is worse", and they never enter the score. Integrity produces its own verdict.

| Metric | What it measures | How to read it |
|---|---|---|
| `frame_gaps` | Non-contiguous or duplicated `frame_index` values | Any nonzero count is a broken episode |
| `timestamp_dev` (info only) | Largest deviation of consecutive timestamps from `1/fps` | Usually about 0. Does not affect the verdict |
| `length_mismatch` | Episode `length` vs actual parquet rows | Any mismatch means metadata and data disagree |
| `video_window_mismatch` | Per camera, the video window duration vs `length/fps` | Mismatch means video and actions differ in length. Cannot see a constant offset |
| `nonfinite_values` | Count of NaN or inf in `action` or `state` | Any nonzero count breaks training |
| `too_short` | Episode under a minimum duration or frame count | Catches one-frame and near-empty episodes |
| `schema_mismatch` | Differences from the other merged sources in feature shapes and names, `fps`, units, camera keys and resolution | Flags a dataset that will not collate cleanly |

## Language

Static checks on the task string. Task strings are canonicalized (case, whitespace, punctuation) before they are used for grouping or balance.

| Metric | What it measures | How to read it |
|---|---|---|
| `task_missing` | Empty, whitespace-only or null task string | A flag |
| `task_generic` | Fewer than 3 tokens, a placeholder ("task desc", "Hold"), or no verb | A flag. The instruction is too weak to say what the demonstration shows |

## Outliers

Never scored. Unusual does not mean bad.

| Metric | What it measures | How to read it |
|---|---|---|
| `iforest_score` | Isolation-forest anomaly score over the episode's metric vector, fit within its group | Higher means more unusual than peers. Can rank defective episodes as normal, so it is information only |
| `novelty_knn` | Mean distance to the nearest same-group episodes in metric-vector space | Two-sided. Very high means unlike the rest, very low means near-duplicate |
| `is_outlier` | Either score at z >= 2 | Coarse "worth a second look" |

## Normalization

- **Grouping.** Task strings are canonicalized, then episodes are grouped by task. An episode with several tasks is grouped by its sorted set of tasks. A task group with at least 20 episodes is used as is. Otherwise the scores pool over the view. A view with fewer than 20 episodes overall is pooled and marked low-confidence in the panel. The 20-episode threshold is configurable.
- **Scale.** Robust z-scores with a floored scale. Zero-inflated metrics (`idle_lead_s`, `frame_gaps`) use percentile ranks or a tail-based scale.

## Scoring profiles

A profile is a named set of groups that produces one ranking score.

**Groups.** Correlated metrics share a group, and a group counts as one vote.

| Group | Scored members |
|---|---|
| `motion` | `sparc`, `ldlj`, `sparc_phase` |
| `time` | `idle_lead_s`, `idle_trail_s`, `longest_pause_s`, `end_motion_ratio`, `length_z` |
| `tracking` | `track_resid`, `accel_spike_frac`, `joint_limit_frac` |
| `gripper` | `gripper_flips_per_s`, `missed_grasp_frac` |
| `consistency` | `action_divergence` |

**Aggregation.**

- Inside a group: the maximum of the weighted, oriented member z-scores. Weights are 1.0, except `ldlj`, `idle_trail_s` and `end_motion_ratio` at 0.5.
- Across groups: the profile score is the maximum group value, with the weighted mean of group values as a tie-break. Default group weights are equal.
- `n_flags` counts flagged groups, and `driver` names the group behind each episode's score.

| Profile | Uses | Intended for |
|---|---|---|
| `policy` | All groups above | Training low-level imitation policies (ACT, Diffusion Policy), where geometry and timing matter and language does not |
| `vla` | `policy` plus the language flags | Fine-tuning vision-language-action models, where the instruction must match the demonstration |

Rules that hold for every profile:

- Never scored: Integrity, Outliers, `idle_frac`, `path_length_z`, `jerk_rms`, `psd_lf_hf`, `stats_leverage`, `recovery_count` and `track_lag_ms`.
- Language flags do not enter the weighted mean. They force the `vla` verdict to at least "review" and add to `n_flags`.
- Trimming is a suggestion only: `keep_from_s` and `keep_to_s` store a proposed range, and nothing is written back to the dataset.
- Marking an episode `exclude-candidate` is always a separate human action.
- `config_version` is written on every episode, so scores from different formula versions are never compared.
- Review worst-first.

## How the metrics are checked

Each scored metric has a corruption it must catch: added jitter, an inserted idle run, a truncated episode, values pushed against joint limits, and so on. The corruption is applied to real episodes, and the test asserts that the corrupted copy ranks worse than its own original. Every metric is also re-tested after truncating all episodes to a common length, to catch metrics that only measure duration. Results are in `harness/REPORT.md`.

## Evidence

Scores are triage signals. Detection accuracy has not been shown to predict policy quality, so validate a curation run with a training run before dropping data.

- **Smoothness helps, but is not sufficient.** RINSE (arXiv:2604.23000) found SPARC-based filtering gave 16% higher RoboMimic success with one-sixth of the data, and reports SPARC as about 10 times more noise-robust than LDLJ. It assumes episodes were already filtered for task success, and it was tested on behavior cloning with proprioceptive signals, not VLAs. `sparc_phase` adapts its gripper-phase idea and is not the paper's exact method.
- **Length can fake a good detector.** A curation-metrics audit (arXiv:2606.10229) found 5 of 7 metrics exploited episode length, and that detection accuracy and policy success were uncorrelated. Isolation-forest curation matched no curation. It is a single-author preprint on 80 scripted demonstrations, so read it as a caution. This is why outlier metrics are never scored and why the length check exists.
- **Time-alignment matters.** RoboDrop (arXiv:2609.10021) shows observation-action temporal misalignment is a major corruption that action-only methods handle poorly.
- **Stalls, jerk and limits.** A teleoperation paper (arXiv:2605.26349) flags stalls, LDLJ on joint state and operation near joint limits. Its user study had 3 operators, so it is weak evidence.

Not validated:

- Whether generic instructions such as "Hold" hurt VLA language grounding is not measured in anything read. `task_generic` rests on a Hugging Face list of such strings as real defects.
- The gripper and tracking metrics have no published downstream validation. Treat them as sanity checks.
- Default thresholds (2% limit margin, minimum phase length, 20-episode group size, warn at z >= 2, weights) are starting points to tune on your data.
- `action_divergence` and `stats_leverage` cannot be tested on datasets with very few episodes per source. `sparc_phase`, `missed_grasp_frac`, `recovery_count`, `schema_mismatch` and the outlier metrics are covered by synthetic tests only.

## Not computed yet

Camera metrics (blur, exposure, clipped pixels, frozen frames, video-action lag), visual duplicate detection, and a vision-language check of the task string are not computed. The Vision tab is a placeholder.
