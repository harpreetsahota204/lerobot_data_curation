# LeRobot Data Curation: Panel and Operator UX

Status: draft v1. Decisions below came from three rounds of multiple-choice questions. Items marked **(inferred)** are my mapping onto `METRICS.md` v3 and have not been confirmed. Unresolved items are listed at the end. Nothing here is implemented yet.

The target is the exact look and feel of the `demo_quality_scorer` "Episode Quality" panel, extended to cover LeRobot episodes and the larger metric set in `METRICS.md` (v3.1).

## Principles carried over from the MCAP panel

1. **Worst-first triage.** Rows arrive pre-sorted so a reviewer reads down from the top and stops when episodes look fine.
2. **Everything clickable.** A chart element either filters the samples view to the matching episodes or opens an episode.
3. **The panel follows the view.** It re-fetches and re-ranks whenever the FiftyOne view changes.
4. **Every chart explains itself.** An "i" tooltip on each card says what the metric means and how to read it.
5. **Missing data is stated, never hidden.** "Unknown" is not "pass". A metric that was not computed says so.
6. **Tags only.** The panel writes sample tags. It never deletes or hides data.

## Architecture

- A pure React panel with no Python `Panel` class, backed by unlisted operators, as in the MCAP plugin.
- One data operator serves the whole payload (rows, histogram bins, thresholds, verdict counts, which families were scored). It re-fires on every view change.
- One new operator returns a single episode's arrays and frames on demand, for the inspector. It is not part of the main payload.
- The look is unchanged: dark theme, FiftyOne orange accent (`#ff6d04`), `#1f1f1f` cards, Recharts, inline styles, the same `Card`, `Tabs`, `Button`, `Banner`, `Chip`, `DataTable` and hover "i" tooltip components.

## Panel layout

Top to bottom:

1. **Banner** (orange) only when the view mixes `config_version` values.
2. **Header controls:** a **profile dropdown** (`policy`, `vla`, custom) and a **task dropdown**.
   - Switching profile re-ranks the Overview table and recolors scores.
   - The task dropdown defaults to "all tasks", with each episode ranked within its own task's normalization group.
3. **Tab bar:** Overview, Motion & Action, Integrity & Coverage, Vision, Language. It opens on the first tab that has data and never overrides a tab the user picked.
4. **Scrolling body** of cards.
5. **Footer** (see below).

## Tabs

### Overview

Cards:

- **Profile score histogram** for the selected profile, with a dashed warn line. Clicking a bar filters the view.
- **Verdict counts:** Integrity and Language pass/warn/fail/unknown bars. Clicking a bar filters the view.
- **Episodes per task** bar chart (coverage). Clicking a bar filters the view.
- **Per-source summary:** episodes, mean score and flagged share per source or robot.
- **Outlier scatter:** `iforest_score` against `novelty_knn`, red for `is_outlier`. Clicking a point opens the episode. Information only, never scored.

Below the cards: the **worst-first ranking table** for the selected profile **(inferred columns):** Episode, Task, Score, Flags, Driver (the metric group behind the score), Integrity verdict, Language verdict.

### Motion & Action **(inferred contents)**

Covers the Motion, Time efficiency, Tracking and contact, Gripper and Consistency families.

- **Signal chips** (arm groups such as `arm:left`, `arm:right`) act as a shared legend. Clicking a chip isolates that signal across all charts and re-ranks the table, as in the MCAP Motion tab.
- A grid of histogram cards, one per metric. Each card has a title, a subtitle with the smoother/better direction and the warn threshold, an "i" tooltip, an expand toggle (grid to full-width chart) and a dashed warn line. Clicking a bar filters the view.
- Metrics that were not computed show "Not computed in the last run". Never-scored metrics (`idle_frac`, `path_length_z`, `recovery_count`, ...) still get a chart, labeled "not scored".
- A worst-first table for this tab's groups, with a column per metric.

### Integrity & Coverage **(inferred contents)**

- Pass/warn/fail/unknown verdict bar for Integrity. Clicking a bar filters the view.
- A per-episode table with a colored verdict chip and a "Worst check" column naming which integrity check failed.
- Coverage: the dataset-level `balance` result (episodes per canonical task and per source or session).

### Vision

- **Camera chips** isolate one camera across all charts, like the signal chips.
- Histogram cards for `blur`, `exposure_err`, `clipped_frac`, `frozen_frac` and `video_action_lag_ms`.
- A `dup_group` card **(inferred)** listing duplicate groups, where clicking a group filters the view to its members.
- A worst-first table.

### Language

Until the VLM step is built (on hold), this tab shows only the static flags (`task_missing`, `task_generic`). The proposed-task columns and the Accept / Reject buttons appear once phase 3 exists.

- A table with: Episode, original task, proposed task, `vlm_match` chip, confidence and the static flags (`task_missing`, `task_generic`).
- **Accept** and **Reject** buttons on each row, plus a bulk "accept all in view" button.
- Accept writes `accepted_task`. Reject clears the `relabel` tag and keeps the original task. It does not mark the episode for exclusion; that stays a separate action with the footer tag button.
- `vlm_match = no` writes the `relabel` tag and a `proposed_task`. Nothing is dropped automatically.

## Inspector

Clicking a row in any table selects it and shows an inline card **(position below the table is inferred)** with all of these:

1. All metric values for the episode, with per-arm and per-camera breakdown.
2. Per-joint action and state traces overlaid, with flagged spans shaded.
3. Speed profile with idle and pause spans marked.
4. Gripper timeline: open/close events and phase boundaries.
5. Camera thumbnails at the worst flagged times.
6. An **Open in viewer** button, which opens the episode in the multimodal viewer and toasts the worst flagged interval's timecode (there is still no playhead-seek API).

All five contents were chosen so we can see them and adjust. Expect this to be the most iterated part. Frames are decoded lazily and cached.

## Footer

Buttons: `Tag N selected: review`, `Tag N selected: exclude-candidate`, `Tag N selected: relabel`, a "N scored episode(s) in view" counter, and **Refresh**. With nothing selected, the label reads "all N in view".

## Compute operators

Three operators, each delegated by default with an immediate-run option and an advisory line above Run (as in the MCAP form):

| Operator | Phase | Cost |
|---|---|---|
| Compute quality | 1 | seconds |
| Compute vision | 2 | minutes |
| Check instructions with VLM (on hold) | 3 | up to hours |

Form conventions from the MCAP plugin:

- **Compute quality starts with the dataset-level checks.** The first step of the form shows each inferred value (`feature_map` first, then `action_semantics`, `joint_units`, `gripper_convention`, `camera_roles`, `balance`) with confirm or override. Metrics that depend on an unconfirmed or wrong assumption switch off.
- One tab per family with an on/off switch and per-metric checkboxes. A family switches itself off and says why when its requirement is missing (for example, no named gripper dimension).
- Opt-in metrics are off by default.
- **Thresholds and weights** are fixed defaults in the panel. They are configurable per profile in the Compute form and stored with `config_version`.
- Re-running overwrites the affected `lr_*` fields and replaces this plugin's own temporal tags (anchor `lerobot_data_curation`), never touching tags drawn by hand. Flagged spans appear as temporal tags on the multimodal timeline, and the inspector computes the same spans on demand.

## Extensibility in the panel

- The five tabs are handwritten React.
- There is no auto-generated section for extra metrics. A fork adds a metric in Python (one function and one dict entry) and, to chart it, copies a documented chart template in the React. `EXTENDING.md` covers both. The panel is not meant to support every future case.

## States

- Loading.
- **No scores yet:** a primary orange button that opens Compute quality.
- **Tab not computed:** "X wasn't scored in the last run" with the operator to run.
- **Unknown is not pass.**
- **`config_version` mismatch** banner.
- **Small-group fallback:** when a task group has fewer than 20 episodes and scores fall back to a pooled or absolute basis, the panel says so, per `METRICS.md`.

## Open items

1. **Table columns per tab and inspector position** are my inference and untested. We adjust these once the panel is on screen.
