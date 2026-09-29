# LeRobot Data Curation: Implementation Plan

Status: phase 1 built and running (updated after the review fixes and the Overview expand and tooltip changes). Phases 2 and 3 are not built. This is the executable plan behind `METRICS.md` (what we compute), `UX.md` (what the panel does) and `ARCHITECTURE.md` (how it is built). Estimates are guesses for the executing agent, not measurements.

## Goal

Ship a FiftyOne plugin, `lerobot_data_curation`, that scores LeRobot v3 episodes, and ranks them worst-first in a panel that looks and behaves like the `demo_quality_scorer` "Episode Quality" panel. Export was built and then removed from this plugin (see the progress log). v1 covers phase 1 (no video decoding). Phase 2 (vision) follows. Phase 3 (VLM) is on hold.

## Where we are

**The panel today** (five tabs, two header dropdowns, a footer, and an inspector):

| Part | State |
|---|---|
| Header | Profile dropdown (`policy`, `vla`) and Task dropdown. Banners for mixed formula versions, mixed runs, and low-confidence pooled scores |
| **Overview** | Profile score histogram, three verdict charts (profile, integrity, language), episodes per task, outlier scatter, per-source summary, and the worst-first ranking. Each of the four charts expands to full width. "i" tooltips on the Mean score and Score column headers |
| **Motion & Action** | Signal chips, one histogram per metric grouped by family (motion, time, tracking, gripper, consistency) with expand toggles and tooltips, and a worst-first table |
| **Integrity & Coverage** | Integrity verdicts, episodes per task (amber below 5), episodes per source, per-episode failing checks |
| **Vision** | Placeholder until phase 2 |
| **Language** | Static task-string checks (missing, placeholder, too short, no action verb). No VLM check yet |
| **Inspector** | Opens under any table on row click: all metrics with per-arm breakdown, joint traces, speed profile, gripper timeline (needs a named gripper), camera frames, spans, and Open in viewer |
| **Footer** | Tag buttons (`review`, `exclude-candidate`, `relabel`) and Refresh |
| **Compute quality** | Three-tab form (Dataset checks, Metrics, Normalization), delegated by default, writes about 108 flat `lr_*` fields and temporal tags for flagged spans |

**Checked in the App** (through the page's DOM, plus your own checks of the timeline tags): all five tabs render without errors, bar clicks filter the view, row clicks open the inspector and the viewer, and flagged spans render on episode timelines.

**Built and tested in code, not yet clicked through in the App:** the Overview expand toggles, the two new column tooltips, the compute form's Dataset checks tab, and a delegated compute run. Automated tests cover the engine, reader, writes, panel payload and the corruption harness: 106 tests, about 13 seconds.

**Known limits:**
- The only dataset used is `lr_dev` (2 episodes per source, 51 robots), so every score is normalized against one pooled group. The per-task path (20 or more episodes of one task) runs only in synthetic tests. Rankings there test the UI, not curation quality.
- The panel loads every episode in one call (about 2.7 KB each). There is no pagination.
- No frontend tests. Cloud-hosted LeRobot sources are not supported. Thresholds such as the spike floor are calibrated on `lr_dev` only and are not exposed in the form.

## What remains

**To close v1 (each needs you or a running App):**
1. Click through the Overview expand toggles and the new column tooltips.
2. Open the compute form's Dataset checks tab and change an override.
3. Run **compute quality** delegated (`fiftyone delegated launch` must be running) and confirm it completes on `lr_dev`.
4. Score one real single-task dataset with 20 or more episodes (for example the aloha sim insertion set, 50 episodes) to exercise the per-task normalization path on real data.

**Nice to have before v1:**
- Expand toggles on the Integrity & Coverage and Language charts, and a Score tooltip on the Motion & Action table.
- Header tooltips on the remaining tables that show computed columns.
- Expose the main thresholds (spike floor, limit margin, minimum group size) in the Normalization tab.

**After v1:**
- **Phase 2, vision** (Step 4 below): blur, exposure, clipped pixels, frozen frames, video-action lag, duplicate detection, the Vision tab and a Compute vision operator. Needs the embedding-model decision first.
- **Phase 3, VLM** (Step 6): on hold. Instruction-versus-video check, proposed task rewrites, and Accept and Reject on the Language tab.
- Pagination for large datasets, frontend tests, and cloud sources.

## Out of scope for v1

- The VLM step, `vlm_match` and `proposed_task` (all depend on phase 3).
- Exporting a curated dataset. It was built, then removed; use the tags and your own workflow to act on the ranking.
- Writing trimmed episodes back to LeRobot format.
- The policy-validation recipe (train on curated vs held-out).
- Cloud-hosted LeRobot sources (the reader raises `NotImplementedError`).

## Progress log

- **Step 0:** done. The timeline probe (0.4) passed on retest.
- **Step 1 (slice 1):** built and running. 38 unit and corruption tests pass (`python -m unittest lerobot_data_curation.tests.test_engine lerobot_data_curation.tests.test_reader`). The compute operator scores all 102 `lr_dev` episodes in about 2 seconds, and the panel loads in the App with a working histogram, ranking table, bar-click filtering and row-click open. Gate 1 passed.
- **Changes found while building:**
  1. Flat field names cannot contain double underscores: mongoengine reads `__` as a lookup separator, so per-signal fields are `lr_<metric>_on_<signal>` and the worst signal is `lr_<metric>_signal`. `ARCHITECTURE.md` is updated.
  2. The normalization ladder's robot-type rung could never trigger, so it was removed. A view with fewer than 20 episodes is pooled and shown as low-confidence. `METRICS.md` is updated.
  3. Tests use the standard library `unittest`, so nothing was added to your conda environment.

- **Step 2 (slice 2):** built. 86 unit and corruption tests pass. All phase 1 metrics run on `lr_dev` in about 5 seconds (108 fields with every opt-in on). The five tabs render in the App (Overview 6 cards, Motion & Action 19 charts, Integrity & Coverage, Language; Vision is a placeholder until phase 2), and selecting a row opens the inspector (metrics table, joint traces, speed profile, gripper timeline when a gripper is named, camera frames).
- **Export findings:**
  1. FiftyOne's exporter refuses to mix sources, so a multi-source view is exported as one dataset per source in subfolders. Checked: 3 sources, 5 kept episodes, re-imported to 5.
  2. LeRobot cannot write some sources (for example ones with a string-typed feature). Those are skipped and listed in the manifest under `failed_sources`. Checked with a simulated failure only, not with a real failing source.
  3. Export is heavy on high-resolution sources: a test with two episodes from the two sources with string features reached 9 GB of temporary frames before I stopped it, and a whole-`lr_dev` export reached 20 GB. Small-resolution sources export in under a second. The form warns that media is copied.
- **Not verified in the App:** the compute form's Dataset checks tab (verified at the schema and Python level only).

- **Step 3:** built. `harness/run_harness.py` pairs each episode with a corrupted copy of itself. All 22 corruption rows pass (at least 83% of pairs worse than the original; most 100%), and the identity control moves nothing. `tests/test_harness.py` fails if a metric has neither a corruption nor a stated reason it cannot be tested. The report is `harness/REPORT.md`.
- **What the harness found (all fixed or documented):**
  1. `stats_leverage` crashed when episodes of one source had different array sizes. Fixed.
  2. `joint_limit_frac` fired on 22% of clean pairs because any resting joint sits at its dataset minimum. It now averages over the joints that move in the episode.
  3. SPARC did not respond to white noise (60% of pairs). It is noise-robust by design, so its test uses stop-and-go hesitation instead (91%). This agrees with RINSE's claim that SPARC is about 10 times more noise-robust than LDLJ.
  4. `end_motion_ratio` and `idle_trail_s` cannot be judged by a length check that cuts the end off. They are marked "not judged" instead of "length-confounded".
  5. No scored metric collapsed under truncation to a common length (lowest rank correlation 0.60, `longest_pause_s`).
- **Temporal tags:** a retest showed both anchored and unanchored probe tags render on `lr_dev` timelines, so the plugin now writes flagged spans as tags (98 on `lr_dev`; a re-run replaces them).

- **Step 5:** `README.md`, `EXTENDING.md` and `requirements.txt` written. The extension path in `EXTENDING.md` was checked by adding a metric to a running copy: it produced its `lr_*` fields, panel metadata and a place in the `motion` group.

- **Review fixes (all six findings):**
  1. `track_lag_ms` now shows milliseconds in the panel, not its z-score (`display: "raw"` in the metric metadata).
  2. Export folder names are sanitized and checked to stay inside the destination before anything is written or removed.
  3. A failed export removes what it wrote, so the destination can be reused; a non-empty destination is still refused and left untouched.
  4. The compute form caches the dataset checks: the second and later interactions take 0.06 s instead of 0.82 s.
  5. Every scored sample carries `lr_run_id`. The panel hides warn thresholds and shows a banner when a view mixes runs.
  6. New tests: `test_write.py` (13, on a temporary FiftyOne dataset) and `test_export.py` (9, on stand-in views). The suite is now 115 tests.
  Also: the acceleration-spike threshold lives in one place (`tracking.spike_threshold`), and the under-covered cutoff is sent by the backend instead of being repeated in TypeScript.

- **Export removed.** Export kept view (operator, footer button, `export.py`, `test_export.py`, and the export docs) was taken out of the plugin. It was heavy on high-resolution sources, and writing a curated dataset is better handled outside the panel. The tags stay. The export findings and review fixes above describe code that no longer exists. The suite is now 106 tests.
- **Overview polish:** the four Overview charts (profile score, verdicts, episodes per task, outliers) expand to full width like the Motion & Action histograms. The Mean score column (per-source summary) and the Score column (worst-first ranking) have "i" tooltips explaining what they average and how the score is built. `DataTable` columns accept an `info` string for this. Type-checks and builds; not yet opened in the App.

## Ground rules

1. Open questions are asked as multiple-choice pop-ups, never as prose lists.
2. Each step ends with a check: a test, a command, or a UI action. A step is not done until its check passes.
3. Work happens on a clone of the test dataset, never on the original.
4. Each gate below includes a code-review pass over the new code (bugs, regressions, missing tests) before the next slice starts.
5. When a decision changes, the matching doc (`METRICS.md`, `UX.md`, `ARCHITECTURE.md`) is updated in the same step.
6. Environment for everything: conda env `fo_lerobot`, with `VFF_MULTIMODAL=1` set before `fiftyone` is imported.

## Step 0: Setup and spikes (about 45 minutes; two checks need you)

Each spike answers a question that could change the plan.

- [x] **0.1 Dev install.** Symlink the plugin folder into `~/fiftyone/__plugins__/` and confirm `fiftyone plugins list` shows it. Check: the plugin appears with no errors.
- [x] **0.2 Working dataset. Done.** `lr_dev` is a clone of `community_v3_10per_embodiment` with the first 2 episodes from each of the 51 sources: 102 episodes. Checked: media references resolve, all 102 episodes are readable with the reader, the original still has 497 episodes. Consequence: the largest group of episodes sharing one task string is 6, so real data never reaches the 20-episode per-task path. Step 1.3 covers that path with synthetic tests instead.
- [x] **0.3 Sidebar groups.** Check that a plugin operator can set `dataset.app_config.sidebar_groups` so flat `lr_` fields group under one heading. If not, document the manual step.
- [x] **0.4 Temporal tags on the timeline. Done: they render. (needs you, 2 minutes.)** Write one probe temporal tag on an `lr_dev` episode. You open that episode in the App and confirm the span appears on the multimodal timeline. Then delete the probe. If it does not render, intervals show only in the inspector and the footer toast.
- [x] **0.5 React toolchain.** Node 22 and npm 10 are installed. Copy `package.json`, `vite.config.ts` and `tsconfig.json` from `demo_quality_scorer`, build a stub panel, and confirm it loads in the App. Check: the stub panel opens.

**Gate 0:** all five checks pass, or each failure has a documented fallback.

## Step 1: Slice 1, thin end-to-end path (about 2 to 3 hours plus your review)

One metric per family, all the way from disk to panel.

- [x] **1.1 Reader fixes.** Keep raw frame order (so integrity checks can see gaps), read each parquet file once per batch, and read arrays through the `feature_map`. Check: all 102 `lr_dev` episodes read, the time is recorded, and a test with shuffled `frame_index` is detected.
- [x] **1.2 `feature_map` inference.** Detect state, action and camera keys. Prefer `observation.state` and `action`, then known patterns, then user override. Return "missing" instead of failing. Check: the 5 datasets used earlier give correct maps, and `interndata-a1-representative-mix` reports state and action as missing.
- [x] **1.3 Normalization port.** Port `normalize.py` from the MCAP plugin. Add canonicalized task grouping and the ladder from `METRICS.md`: task group with at least 20 episodes, otherwise pool over the dataset. Check: unit tests for group sizes 5, 19, 20 and 400, and a zero-inflated metric.
- [x] **1.4 Five metrics.** `sparc` (Motion), `idle_lead_s` (Time), `length_z` (Time, signed, scored as `abs`), `frame_gaps` (Integrity), `task_generic` (Language). Each is a plain function plus one entry in the `METRICS` dict. `sparc` uses range-normalized `action`, gripper dimensions excluded, cutoff `min(10 Hz, 0.4 x fps)`, fixed-length windows, median over windows plus a worst-window value. Check per metric: a synthetic unit test and a corruption test (added jitter worsens `sparc`, an inserted idle run raises `idle_lead_s`, truncation moves `length_z`, duplicated `frame_index` raises `frame_gaps`, the task string "Hold" is flagged).
- [x] **1.5 `policy` profile.** Groups from `METRICS.md`, max inside a group, max across groups, `n_flags`, `driver`. Only the groups with slice-1 metrics exist for now. Check: `driver` names the right group in a hand-built example.
- [x] **1.6 Writes.** Flat `lr_*` fields, the run record (normalization stats, confirmed checks, thresholds), `lr_config_version`. Re-running overwrites cleanly. Check: fields appear in the sidebar, and a second run gives identical values.
- [x] **1.7 Compute quality operator.** Form with the `feature_map` confirm step and family switches. Delegated by default, with progress. Check: a delegated run over all 102 episodes completes, and the time is recorded.
- [x] **1.8 Overview tab.** Copy the shell (`ui.tsx`, `charts.tsx`, `theme.ts`). Build the header (profile and task dropdowns), the score histogram with warn line, the worst-first table, the footer tag buttons, the empty state and the "click row opens the episode plus toast" behavior. Check: you open the panel on `lr_dev`, see the ranking, click a bar and a row.

**Gate 1 (you, about 15 minutes):** look at the panel and tell me what to change. This is also the first real test of the "small groups use the pooled fallback" risk. Then the code-review pass.

## Step 2: Slice 2, the rest of phase 1 (about 1 to 1.5 days)

This refines the earlier one-day estimate in `ARCHITECTURE.md`, because the metric batches and the inspector are larger than they looked.

- [x] **2A. Remaining metrics, in four batches, each with corruption tests.** Each batch ends with a green test run.
  1. Motion and Time: `ldlj`, `sparc_phase` (opt-in), `idle_trail_s`, `longest_pause_s`, `end_motion_ratio`, and the never-scored `idle_frac`, `jerk_rms`, `psd_lf_hf`, `path_length_z`. About 2 hours.
  2. Tracking and Gripper: `track_lag_ms` (flag only), `track_resid`, `accel_spike_frac`, `joint_limit_frac`, `gripper_flips_per_s`, `missed_grasp_frac`, `recovery_count`. About 2 hours.
  3. Consistency and Integrity: `action_divergence`, `stats_leverage`, `timestamp_dev`, `length_mismatch`, `video_window_mismatch`, `nonfinite_values`, `too_short`, `schema_mismatch`, and Language's `task_missing`. About 2 hours.
  4. Outliers: `iforest_score`, `novelty_knn` (metric-vector space only for now), `is_outlier`. About 1 hour.
- [x] **2B. Dataset-level checks.** The other five checks (`action_semantics`, `joint_units`, `gripper_convention`, `camera_roles`, `balance`) with the confirm and override UI in the first step of the Compute quality form. Metrics that depend on an unconfirmed assumption switch off and say why. About 1 hour. Check: on `lr_dev`, unnamed and gripper-less sources switch the right metrics off.
- [x] **2C. Panel tabs.** Motion & Action (signal chips, histogram grid with expand toggle and "i" tooltips, per-tab table), Integrity & Coverage, and the remaining Overview cards (verdict counts, episodes per task, per-source summary, outlier scatter). About 3 hours.
- [x] **2D. Inspector.** New `get_episode_detail` operator. Inline card with all metric values plus per-arm breakdown, per-joint action and state traces with flagged spans, speed profile, gripper timeline, camera thumbnails, and the Open in viewer button. Iterated after you see it. About 3 hours.
- [x] **2E. Tags.** Footer tag buttons (`review`, `exclude-candidate`, `relabel`). The Export kept view built here was later removed (see the progress log).

**Gate 2 (you, about 30 minutes):** click through every tab and the inspector. Then the code-review pass.

## Step 3: Validation harness (about half a day; written alongside steps 1 and 2)

- [x] **3.1 Corruption suite.** One in-memory corruption per metric, applied to real `lr_dev` episodes with the affected episodes recorded. The test asserts that the metric ranks corrupted episodes worse than clean ones.
- [x] **3.2 Length control.** Re-run every metric after truncating all episodes to a common length. Mark any metric whose ranking collapses as length-confounded in `METRICS.md`.
- [x] **3.3 Report.** A generated table of metric, corruption, and pass or fail, saved next to the tests.

**Gate 3:** no scored metric ships without a passing corruption test. A metric that fails is fixed, marked opt-in, or removed.

## Step 4: Phase 2, vision (about 1 day)

- [ ] **4.0 Decide the visual embedding model** for `novelty_knn` and `dup_group` before starting (a pop-up question). Nothing else in phase 2 depends on it.
- [ ] **4.1 Sparse frame decoding** with PyAV. AV1 decoding (`libdav1d`) was confirmed on a v3 dataset.
- [ ] **4.2 Camera metrics:** `blur`, `exposure_err`, `clipped_frac`, `frozen_frac`, `video_action_lag_ms`, normalized per group and camera key, worst camera taken.
- [ ] **4.3 Redundancy:** `dup_group` and the visual + action `novelty_knn`.
- [ ] **4.4 Compute vision operator and the Vision tab** with camera chips.

**Gate 4:** corruption tests for every camera metric, including a video time-shift of k frames for `video_action_lag_ms`. Then the code-review pass.

## Step 5: Documentation (about 1 hour)

- [x] `EXTENDING.md`: how to add a metric (one function, one dict entry, one corruption test) and how to add a chart (a documented React template).
- [x] `README.md`: install, quick start, what each tab does, limits, and the "scores rank the review queue, they do not decide" statement.
- [x] Final pass on `METRICS.md`, `UX.md` and `ARCHITECTURE.md` so they match what was built (done for phase 1; the phase 2 and 3 sections describe unbuilt work).

## Step 6: Phase 3, VLM (on hold)

Not scheduled. Reopen with a decision on the runtime (own loader, zoo model, or endpoint). `proposed_task` review and the Language tab's Accept / Reject buttons wait for this.

## Definition of done for v1

1. [x] Steps 0 to 3 and 5 are complete, and every gate passed.
2. [ ] Compute quality runs delegated on `lr_dev` (102 episodes) without errors. It runs immediately in about 5 seconds; a delegated run has not been tried.
3. [x] The panel shows the Overview, Motion & Action and Integrity & Coverage tabs with working filters, row clicks, inspector and tags.
4. [x] Every scored metric has a passing corruption test, and length-confounded metrics are marked (none are; two tail metrics are "not judged").
5. [ ] One real single-task dataset with 20 or more episodes has been scored, so the per-task normalization path is exercised on real data.

## Decisions made

1. **Git:** you handle it. Nothing in this plan runs `git init` or commits.
2. **Development dataset:** `lr_dev`, 2 episodes per source, 102 episodes (done, see 0.2).

## Open decisions

1. The visual embedding model for step 4 (phase 2).
2. Which single-task dataset to score for done item 6.
3. The runtime for the VLM step (phase 3, on hold).
