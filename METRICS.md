# LeRobot Data Curation: Planned Metrics

Status: draft v3.1, pending final sign-off. 41 per-episode metrics and 6 dataset-level checks, grouped by the build phase that delivers them. Nothing here is implemented yet.

Conventions that apply to everything:

- Scores rank the review queue. Nothing is deleted, hidden or rewritten automatically.
- Every score is normalized within a group of comparable episodes (see [Normalization](#normalization)). It is never pooled across tasks unless a fallback says so.
- Raw directions are shown in the tables below. Inside a profile, every scored metric is oriented so that higher is worse.
- Each metric is a plain function plus one dict entry (`family`, `group`, `requires`, `signal`, `higher_is_worse`, `scored`, `description`). A fork adds a metric by writing the function and the entry, following `EXTENDING.md`. There is no registry or auto-discovery.
- Detection accuracy does not predict policy quality (see [Evidence](#evidence-and-caveats)). Treat every metric as a triage signal, and validate a curation run with a policy run before dropping data.

## Signals: which array each metric reads

| Metric family | Reads | Why |
|---|---|---|
| Smoothness (`sparc`, `ldlj`, `sparc_phase`, `jerk_rms`, `psd_lf_hf`) | `action` by default, with a signal picker | `action` is what a policy is trained to output. On leader-follower rigs (SO-100, Koch, ALOHA) it is the operator's hand, not the robot. `observation.state` is selectable. |
| Contact and jolt (`accel_spike_frac`, `joint_limit_frac`, `track_resid`, `track_lag_ms`) | `observation.state` (tracking metrics use both) | Contact, jolts and limits happen to the follower robot, not the leader. |
| Time and gripper metrics | `action` for speed, gripper dimensions by name | Idle and pause are properties of commanded motion. |

Smoothness processing rules:

- Each joint is range-normalized (using `meta/stats.json` or a robust per-task scale) before speeds are combined, so a joint with a large range does not dominate and units (degrees, radians, -100..100) do not matter.
- Gripper dimensions are excluded from arm speed.
- The low-pass cutoff is `min(10 Hz, 0.4 x fps)`, so 30 fps data is not analyzed near Nyquist.
- Metrics run on fixed-length windows, because LDLJ is duration-sensitive.

## Dataset-level checks (6)

These run once per dataset before any episode metric. Each infers a value from `info.json`, joint names and value ranges, shows it, and lets the user confirm or override. Metrics that depend on a wrong assumption switch off instead of running on bad input.

| Check | What it decides | Why it matters |
|---|---|---|
| `feature_map` | Which columns are state, action and cameras | Some datasets use non-standard keys (for example no `observation.state` or `action` columns). Every other metric reads its arrays through this map, and metrics whose array is missing switch off |
| `action_semantics` | Whether `action` is leader joint positions, deltas, end-effector pose or unknown | Decides whether `track_lag_ms` and `track_resid` can run, and how smoothness is read |
| `joint_units` | Degrees, radians or normalized (-100..100) | Thresholds and range normalization depend on it |
| `gripper_convention` | Which direction is closed, and which dimensions are grippers | Needed by `gripper_flips_per_s`, `missed_grasp_frac`, `recovery_count`, `sparc_phase` |
| `camera_roles` | Wrist, top, side, ego, from feature keys | Vision metrics are normalized per camera |
| `balance` | Episodes per canonical task and per source/session | Shows under-covered tasks and dominant sources. Reported, not scored |

## Phase 1: no video decoding (33 metrics)

### Motion (5)

| Metric | What it measures | How to read it |
|---|---|---|
| `sparc` | Spectral arc length of the speed profile: how many "corrections" the motion contains | Closer to 0 is smoother. Very negative is fragmented, hesitant motion. Scale-invariant, so it is the primary smoothness metric |
| `ldlj` | Log dimensionless jerk of the speed profile | Closer to 0 is smoother. Noisier and duration-sensitive, so it is de-weighted inside the motion group |
| `sparc_phase` (opt-in) | SPARC computed inside each gripper-delimited phase (between open/close transitions), rolled up by median | Same reading as `sparc`. Stops contact transitions from being penalized as roughness. Needs a confirmed gripper dimension. Phases under a minimum sample count are skipped |
| `jerk_rms` (opt-in, never scored) | RMS jerk after a low-pass filter | Lower is smoother. Noise-sensitive and correlated with the others |
| `psd_lf_hf` (opt-in, never scored) | Log ratio of low- to high-frequency power in the speed profile | Higher is smoother. Tremor sits near or above Nyquist at LeRobot frame rates, so this is unreliable there |

Each scored motion metric has a worst-window variant, so a bad stretch is not averaged away.

### Time efficiency (7)

| Metric | What it measures | How to read it |
|---|---|---|
| `idle_frac` (never scored) | Fraction of frames below the idle speed (a multiple of this episode's own moving speed) | Context only. A high value may be a legitimate pause |
| `idle_lead_s` | Seconds of stillness at the start | High means dead time before the task begins (no-op frames). High priority. Also stores a suggested trim start (`keep_from_s`) |
| `idle_trail_s` | Seconds of stillness at the end | High means a long dead tail. Low weight, since fixed-duration recordings produce this from a timer. Also stores a suggested trim end (`keep_to_s`) |
| `longest_pause_s` | Longest idle run that is not at either end | High means a mid-task stall or hesitation. A pause next to a regrasp is treated as recovery and excluded |
| `end_motion_ratio` | Speed in the last ~1 s relative to the episode's typical moving speed | High means the episode probably ended mid-motion (cut off). Low weight, same timer caveat as `idle_trail_s` |
| `length_z` | Robust z of episode duration within its normalization group, stored signed | Profiles score `abs(z)`. Positive means unusually long (struggle), negative means unusually short (abandoned) |
| `path_length_z` (opt-in, never scored) | Robust z of total joint-space path length | High means a longer, wandering route than peers. Correlated with `length_z`, so shown only for review |

`length_z`, `path_length_z` and `idle_frac` are length-confounded by construction. Only `length_z` is scored, and it shares the `time` group's single vote with the idle metrics.

### Tracking and contact (4)

| Metric | What it measures | How to read it |
|---|---|---|
| `track_lag_ms` | Delay between the commanded `action` and the achieved `state`, from cross-correlation | Mostly a hardware property, so it is summarized per dataset or session. Only episodes far from the dataset median are flagged, and it never enters a score. Runs only when `action_semantics` says action and state share a joint space |
| `track_resid` | Per-joint normalized gap between action and state after lag alignment, with the per-joint median offset subtracted first (arm joints only) | High means the robot is not reaching what was commanded (slip, collision, stall). Ranked continuously, and also raises interval flags |
| `accel_spike_frac` | Fraction of frames where any joint's `state` acceleration exceeds a robust threshold: median + k x MAD, with an absolute floor | High means sudden jolts, a proxy for collisions or contact. The floor stops idle joints (median near 0) from making every frame a spike. Reported for review, never used to mask other metrics |
| `joint_limit_frac` (opt-in) | Fraction of `state` frames within a small margin (default 2% of range) of a joint limit | High means the operator worked near the robot's limits. Uses user-supplied limits when given, otherwise `meta/stats.json` min/max. The fallback is a weak proxy, because observed extremes always sit at the edge of the observed range |

### Gripper (3)

| Metric | What it measures | How to read it |
|---|---|---|
| `gripper_flips_per_s` | Open/close transitions per second on gripper dimensions, with hysteresis so noise near the threshold does not count | High means chatter or a hesitant operator. Several clean cycles are often recoveries, which is why `recovery_count` exists |
| `missed_grasp_frac` (opt-in) | Fraction of close commands where the follower closes fully with no stall residual (nothing was held) | High means empty grasps. Needs a confirmed gripper convention and is heuristic |
| `recovery_count` (never scored) | Number of detected regrasp recoveries (open then close again near the same place) | Neutral or positive. Writes a `recovery` tag. Recovery behavior is never penalized |

### Consistency (2)

| Metric | What it measures | How to read it |
|---|---|---|
| `action_divergence` | Variance of `action` among the nearest `state` neighbors in other same-group episodes | High means the demonstration takes a different action from peers in a similar state. Needs enough episodes in the group. Without images, similar states can legitimately need different actions, so read it as a review signal |
| `stats_leverage` (never scored) | How many feature dimensions this episode alone stretches beyond the rest of the dataset's `meta/stats.json` bounds | High means this episode distorts the normalization every training run will use |

### Integrity (7): pass/fail flags plus counts

| Metric | What it measures | How to read it |
|---|---|---|
| `frame_gaps` | Non-contiguous or duplicated `frame_index` values | Any nonzero count is a broken episode |
| `timestamp_dev` (info only) | Largest deviation of consecutive timestamps from `1/fps` | Native recordings derive timestamps from `fps`, so this is usually about 0. It does not affect the verdict. Video timestamps are checked by `video_window_mismatch` |
| `length_mismatch` | Episode `length` vs actual parquet rows | Any mismatch means metadata and data disagree |
| `video_window_mismatch` | Per camera, the video window duration vs `length/fps`, using file-relative timestamps | Mismatch means video and actions differ in length. It cannot see a constant offset, which is what `video_action_lag_ms` is for |
| `nonfinite_values` | Count of NaN or inf in `action` or `state` | Any nonzero count breaks training |
| `too_short` | Episode under a minimum duration or frame count | Catches one-frame and near-empty episodes |
| `schema_mismatch` | Differences from the rest of the merged sources in feature shapes and names, `fps`, units, camera keys and resolution | Flags a dataset that will not collate cleanly |

Integrity produces its own pass/warn/fail verdict. These flags mean "this episode is broken", not "this episode is worse".

### Language, static checks (2)

| Metric | What it measures | How to read it |
|---|---|---|
| `task_missing` | Empty, whitespace-only or null task string | A flag |
| `task_generic` | Fewer than 3 tokens, a placeholder ("task desc", "Hold"), or no verb | A flag. The instruction is too weak to say what the demonstration shows |

Task strings are canonicalized (case, whitespace, punctuation) before they are used for grouping or balance.

### Outliers (3): never scored

| Metric | What it measures | How to read it |
|---|---|---|
| `iforest_score` | Isolation-forest anomaly score over the episode's metric vector, fit within its group | Higher means more unusual than peers. It can rank defective episodes as normal, so it is information only |
| `novelty_knn` | Mean distance to the nearest same-group episodes, first in metric-vector space and, once phase 2 runs, in a combined visual + action embedding | Two-sided. Very high means unlike the rest, very low means near-duplicate. Replaces the old `knn_dist` |
| `is_outlier` | Either score at z >= 2 | Coarse "worth a second look". Unusual does not mean bad |

## Phase 2: needs video decoding (6 metrics)

Per camera, using 8 to 16 sparsely sampled frames (`video_action_lag_ms` needs a denser pass). Values are normalized per (group, camera key) before the worst camera is taken, so a wrist camera does not always win. Every per-camera value is stored.

### Camera (5)

| Metric | What it measures | How to read it |
|---|---|---|
| `blur` | Low-percentile Laplacian variance across sampled frames | Lower is blurrier |
| `exposure_err` | Distance of mean luminance from mid-gray | High means too dark or too bright |
| `clipped_frac` | Fraction of pixels at 0 or 255 | High means blown highlights or crushed shadows |
| `frozen_frac` | Fraction of short sampled bursts where frames are nearly identical while `state` is moving | High means a frozen or dropped camera feed. Static scenes with a still robot are not flagged |
| `video_action_lag_ms` | Lag that best aligns camera motion energy (frame-difference magnitude) with action speed, by cross-correlation | Near 0 means video and actions are in sync. A large constant lag means an observation-action temporal offset. Our own design: no paper validates this exact method, so the harness tests it before it is trusted |

### Redundancy (1)

| Metric | What it measures | How to read it |
|---|---|---|
| `dup_group` (never scored) | Group id shared by episodes above a cosine-similarity threshold in the visual + action embedding | Same id means redundant episodes. Keep one and review the rest. Suggested, never applied automatically |

## Phase 3: local VLM, Qwen3-VL (2 metrics, on hold)

Status: on hold. The runtime (own loader, zoo model or endpoint) is undecided, and no phase 3 work starts until phases 1 and 2 are done.

| Metric | What it measures | How to read it |
|---|---|---|
| `vlm_match` (+ `vlm_conf`) | Whether a short clip of the episode shows what the task string says: yes, no or unsure | "No" means the instruction likely mismatches the video. It never drops an episode. It writes a `relabel` tag and a `proposed_task`. Model choice is set by a measured check, not a size rule (see below) |
| `proposed_task` | A VLM-suggested rewrite of the instruction | A string proposal only. It never overwrites the original. A human accepts (writing `accepted_task`) or rejects it |

Calibration: before a model is trusted for `vlm_match`, the harness swaps task strings between episodes and reports how often the model catches the swap. Published progress-ordering results vary widely across open models regardless of size, and they measure a different task than yes/no matching, so we measure our own. The swap needs at least two distinct tasks, so on single-task datasets `vlm_match` runs uncalibrated and the panel says so. The swap also measures only coarse mismatch, not subtle errors such as a wrong object color.

## Normalization

- **Grouping.** Task strings are canonicalized, then episodes are grouped by task. An episode that lists several tasks is grouped by its canonicalized, sorted set of tasks. Use the task group when it has at least 20 episodes. Otherwise pool over the view. A view with fewer than 20 episodes overall is still pooled and is marked low-confidence in the panel. The threshold is configurable. (An earlier draft added a robot-type rung for small datasets. It could never trigger, because a robot group inside a view of fewer than 20 episodes is itself smaller than 20.)
- **Vision.** Normalized per (group, camera key).
- **Scale.** Robust z-scores with a floored scale. Zero-inflated metrics (`idle_lead_s`, `frame_gaps`) use percentile ranks or the tail-based scale from the MCAP plugin. This needs a check against `normalize.py` when it is ported.
- **Small groups.** Groups too small to normalize fall back to absolute thresholds and say so in the panel.

## Scoring profiles

A profile is a named set of groups, weights and thresholds that produces one ranking score. Presets ship with the plugin, and forks add their own.

**Groups.** Correlated metrics share a group, and a group counts as one vote.

| Group | Members (scored) |
|---|---|
| `motion` | `sparc`, `ldlj`, `sparc_phase` |
| `time` | `idle_lead_s`, `idle_trail_s`, `longest_pause_s`, `end_motion_ratio`, `length_z` |
| `tracking` | `track_resid`, `accel_spike_frac`, `joint_limit_frac` |
| `gripper` | `gripper_flips_per_s`, `missed_grasp_frac` |
| `consistency` | `action_divergence` |
| `camera` (phase 2) | `blur`, `exposure_err`, `clipped_frac`, `frozen_frac`, `video_action_lag_ms` |

**Aggregation.**

- Inside a group: the maximum of the weighted, oriented member z-scores. This is triage semantics, as in the MCAP plugin's worst-signal rule. Provisional weights: 1.0, except `ldlj`, `idle_trail_s` and `end_motion_ratio` at 0.5.
- Across groups: the profile score is the maximum group value (worst-of, the same triage rule as the MCAP plugin), with the weighted mean of group values as a tie-break. Group weights scale each group's value before the max. Default group weights are equal and not final.
- `n_flags` is counted per group, and the profile shows which group drives each episode's score.
- Output goes to `lr_quality.scores.<profile>` as `score`, `n_flags`, `driver` and `verdict`.

| Profile | Uses | Intended for |
|---|---|---|
| `policy` | `motion`, `time`, `tracking`, `gripper`, `consistency`, `camera` | Training low-level imitation policies (ACT, Diffusion Policy), where geometry and timing matter and language does not |
| `vla` | Everything in `policy`, plus the language rules below | Fine-tuning vision-language-action models, where the instruction must match the demonstration |
| custom | Any groups, metrics and weights | Forks and specific robots |

Rules that hold for every profile:

- Never scored: Integrity, Outliers, `idle_frac`, `path_length_z`, `jerk_rms`, `psd_lf_hf`, `dup_group`, `stats_leverage`, `recovery_count`, and `track_lag_ms` (flag only).
- Language flags and `vlm_match = no` do not enter the weighted mean. They force the `vla` verdict to at least "review" and add to `n_flags`.
- Nothing is applied automatically. A `vlm_match = no` episode gets a `relabel` tag and a `proposed_task`. If the human rejects the rewrite, the `relabel` tag is cleared and the episode keeps its original task. Marking an episode `exclude-candidate` is always a separate human action.
- Trimming and deduplication are suggestions only. Trimming stores a suggested range (`keep_from_s`, `keep_to_s`), and `dup_group` stores a group id. Writing trimmed episodes is out of scope for now, because v3 videos are concatenated and `lerobot-edit-dataset` has no trim operation as far as the Compass notes say.
- `config_version` is written on every sample so scores from different formula versions are never compared.
- Default action is to review worst-first.

## How metrics are validated

Each metric ships with a corruption it must catch, applied in memory to real episodes with the affected episodes recorded: added jitter, an inserted idle run, frozen frames, a truncated episode, a swapped task string, a video time-shift of k frames, values pushed against joint limits, an early gripper release. The test asserts that the metric ranks corrupted episodes worse than clean ones. Forks adding a metric add its corruption too.

Two extra rules:

1. **Length control.** Every metric is re-tested after truncating all episodes to a common length. A metric whose ranking collapses is marked length-confounded and documented as such.
2. **Policy validation.** Detection results are not evidence of a better policy. A later recipe trains on curated and on held-out-checked subsets and compares them on episodes the scorer did not choose. It is not part of v1.

## Candidates, not committed

- `traj_align`: distance of an episode's state trajectory to the mean trajectory of a user-tagged clean reference set, and not computed without one. In the cited benchmark the reference was the ground-truth clean set, which is not available in practice, and a task median would be the defective trajectory under heavy contamination. It scored a low detection AUROC but a strong policy in one benchmark, on one task and one defect, so treat it as a lead only.
- Task-progress metrics (`success_prob`, `progress_stall_frac`, TOPReward-style): cut from v1. Revisit after a prototype shows that token probabilities are available on this hardware.
- Writing trimmed episodes back to LeRobot format.
- The policy-validation recipe above.

## Validation status

`harness/REPORT.md` holds the current results. On `lr_dev` (102 episodes):

- All 22 corruption checks pass: the corrupted copy ranks worse than its own original in at least 83% of pairs, and the identity control moves nothing.
- No scored metric collapses when every episode is truncated to a common length. `idle_trail_s` and `end_motion_ratio` measure the end of an episode, which truncation removes, so that check does not judge them.
- Not testable on `lr_dev` (2 episodes per source): `action_divergence`, `stats_leverage`. Synthetic tests only: `sparc_phase`, `missed_grasp_frac`, `recovery_count`, `schema_mismatch` and the outlier metrics.

Definitions changed by what the harness found:

- `joint_limit_frac` is now the share of (frame, joint) pairs near a limit, averaged over the joints that move in the episode. The earlier "any joint near a limit" fired on nearly every frame of robots with a joint parked at its dataset minimum.
- `sparc` is insensitive to additive white noise, by design. It responds to intermittent, stop-and-go motion, which is what its corruption test uses. `ldlj` responds to white noise.

## Evidence and caveats

Verified against the papers:

- **RINSE** (arXiv:2604.23000): SPARC-based filtering gave 16% higher RoboMimic success with one-sixth of the data. Its TED metric partitions demonstrations by gripper contact status, because raw jerk "penalizes contact transitions indiscriminately". It reports SPARC as about 10 times more noise-robust than LDLJ. Smoothness is necessary but not sufficient, and the paper assumes episodes were already filtered for task success. It was evaluated on behavior cloning, not VLAs, and on proprioceptive signals. `sparc_phase` adapts the phase idea and is not RINSE's exact method.
- **Curation-metrics audit** (arXiv:2606.10229): on one LIBERO pick-and-place task with one synthetic defect (early gripper release, 80% contamination, 3 seeds), 5 of 7 metrics exploited episode length. Smoothness detection AUROC fell from 0.979 to 0.447 after truncating to a common length, yet smoothness-curated policies still reached 63.3% success (±30.7) against 3.3% uncurated. Detection AUROC and policy success were uncorrelated (Spearman -0.14). Isolation forest curation gave 3.3% success, the same as no curation. A trajectory-alignment metric with AUROC 0.638 reached 90.0%. This is why outliers are never scored and why length control is a validation rule. The benchmark is a single-author preprint using 80 scripted demonstrations (16 clean, 64 defective), so treat it as a caution, not proof.
- **RoboDrop** (arXiv:2609.10021): a gradient-based VLA curation method that needs a one-epoch warm-up run, so it is policy-in-the-loop and out of scope here. It shows observation-action temporal misalignment is a major corruption that action-only methods handle poorly. It does not test language-mismatch corruption in the parts read.
- **Siemens teleoperation paper** (arXiv:2605.26349): flags stalls (adaptive static fraction), LDLJ on joint state, and operation near joint limits. Its VLM progress analysis was sensitive to occlusion and imperfect progress, which is why VLM results never gate. The user study had 3 operators, so it is weak evidence.
- **OpenGVL** (arXiv:2509.17321): value-order correlation for progress scoring. Qwen2.5-VL-3B scored about 0 on every dataset, Qwen2.5-VL-7B scored between -0.05 and 0.14, and MiMo-VL-7B reached 0.53 to 0.60 on the first two datasets. Open models reached about 60-70% of proprietary upper bounds. A high score is "necessary but not sufficient".
- **TOPReward** (arXiv:2602.19313): only the abstract was read. It reports training-free progress rewards from token probabilities, success detection and reward-weighted behavior cloning. Its metrics are cut from v1 (see Candidates), and its implementation details are not checked.

Not verified:

- SCIZOR, OpenVLA no-op removal, the Hugging Face community-dataset defect list, and the action-consistency theory behind `action_divergence` come from `compass_artifact_wf-...md` and have not been checked against the sources.
- Whether generic instructions such as "Hold" degrade VLA language grounding is not measured in anything read. `task_generic` rests on the Hugging Face list of such strings as real defects.
- Gripper, tracking and vision metrics have no published downstream validation. Treat them as sanity checks.
- Default thresholds (2% limit margin, minimum phase length, 20-episode group size, warn at z >= 2, provisional weights) are starting points and will be tuned on real datasets.
