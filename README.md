# LeRobot Data Curation

A FiftyOne plugin that ranks the episodes of a LeRobot v3 dataset worst-first, so you know which to look at before training a policy or fine-tuning a VLA on them. It scores each episode on motion smoothness, time efficiency, tracking, gripper behavior, consistency, integrity and language, shows the results in a panel, and exports the episodes you keep.

**Use it to decide where to look first. Do not use it as an automatic accept/reject gate.** A smooth, well-timed episode can still show the wrong task, and detection accuracy has not been shown to predict policy quality (see [METRICS.md](METRICS.md), Evidence). The scores order your review queue. A person decides.

Status: phase 1 (everything that needs no video decoding) is built. Vision metrics (phase 2) and the vision-language instruction check (phase 3) are not.

## Contents

- [Install](#install)
- [Score a dataset](#score-a-dataset)
- [The panel](#the-panel)
- [Tag, then export](#tag-then-export)
- [Limits](#limits)
- [Develop and validate](#develop-and-validate)
- [Design documents](#design-documents)

## Install

Requires FiftyOne 1.22 or later (the native `fo.types.LeRobotDataset` type) and Python packages:

```bash
pip install "fiftyone>=1.22" numpy scipy scikit-learn pyarrow av opencv-python-headless
```

Install the plugin, or link a checkout into your plugins folder:

```bash
ln -s /path/to/lerobot_data_curation ~/fiftyone/__plugins__/lerobot-data-curation
fiftyone plugins list          # lerobot-data-curation should appear
```

Set `VFF_MULTIMODAL=1` before `fiftyone` is imported, in every process that touches a LeRobot dataset (your script, `fiftyone app launch`, a notebook). It enables the App's multimodal episode viewer.

The panel ships prebuilt in `dist/index.umd.js`. To change the frontend: `npm install && npm run build`.

## Score a dataset

1. Import a LeRobot v3 dataset with FiftyOne's native type. One sample is one episode:

   ```python
   import fiftyone as fo
   dataset = fo.Dataset.from_dir(
       dataset_dir="/data/lerobot/my_dataset",
       dataset_type=fo.types.LeRobotDataset,
   )
   dataset.add_dir(dataset_dir="/data/lerobot/other_dataset", dataset_type=fo.types.LeRobotDataset)  # optional
   ```

2. Open the dataset in the App and run **LeRobot curation: compute quality** from the operator browser, or click the button in the empty panel. It runs delegated by default. The form has three tabs:
   - **Dataset checks:** the state and action columns the plugin found, and what it assumed about action semantics, joint units, gripper direction, camera roles and balance. Override the two that matter if it guessed wrong. Metrics that depend on an unknown assumption switch off instead of guessing.
   - **Metrics:** one checkbox per metric, grouped by family. Opt-in metrics are off by default.
   - **Normalization:** the smallest task group that is normalized on its own (default 20 episodes).

3. Open **New panel, LeRobot Curation**. It refreshes whenever the view changes, so filtering the grid re-ranks the panel.

Scores are batch-relative: an episode is compared with the other episodes of its task in the view you scored, or with the whole view when its task has fewer than 20 episodes. The panel says so when that happens. Score episodes of the same task together when you want a clean ranking.

## The panel

Two dropdowns sit above the tabs: **Profile** (`policy` for low-level imitation policies, `vla` for VLA fine-tuning, where a weak task string also raises the verdict to warn) and **Task**.

| Tab | What it shows |
|---|---|
| **Overview** | Score histogram, verdict counts, episodes per task, outlier scatter, per-source summary and the worst-first ranking. |
| **Motion & Action** | One histogram per metric (smoothness, time, tracking, gripper, consistency), signal chips to isolate one arm, and a worst-first table. |
| **Integrity & Coverage** | Integrity verdicts, episodes per task and source, and a per-episode list of which check failed. |
| **Vision** | Placeholder until phase 2. |
| **Language** | Static checks on the task string: missing, placeholder, too short, no action verb. |

Click any bar to filter the samples grid to those episodes. Click any row to open the **inspector** under the tables: every metric with its per-arm breakdown, joint traces (action solid, state dashed), the speed profile with the idle threshold, the gripper timeline, and camera frames at the flagged moments. **Open in viewer** opens the episode in the multimodal viewer.

Flagged spans (idle stretches, the longest pause, the roughest smoothness window, acceleration spikes, regrasp recoveries) are also written as temporal tags on each episode's timeline, so you can scrub straight to them.

## Tag, then export

The footer tags episodes `review`, `exclude-candidate` or `relabel` (the selection, or everything in view when nothing is selected). Nothing is deleted or hidden; `exclude-candidate` is a tag you act on later.

**Export kept view** writes every episode not tagged `exclude-candidate` back to LeRobot v3 format, plus `curation_manifest.json` (scores, settings, excluded ids, original task strings). The source dataset is never modified.

- A view spanning several sources is exported as one dataset per source, in subfolders, because FiftyOne's exporter will not mix sources.
- Export copies frames through LeRobot, which is slow and needs disk space on high-resolution data (gigabytes of temporary frames for a couple of episodes).
- A source LeRobot cannot write is skipped and listed in the manifest.

## Limits

- **Scores are triage, not verdicts.** Smoothness and timing say nothing about whether the demonstration did the right thing.
- **Small groups.** Per-task normalization needs about 20 episodes per task. With fewer, episodes are compared across tasks and robots, and the panel shows a low-confidence banner.
- **Unnamed joints.** Per-arm scoring, gripper metrics and the gripper timeline need named joints. Without names the plugin uses one signal over every dimension and turns the gripper metrics off.
- **Tracking metrics** only run when the action is joint positions in the same space as the state (leader-follower teleoperation).
- **Local datasets only.** Cloud-hosted LeRobot sources are not supported yet.
- **Temporal tags** use FiftyOne's internal `fiftyone.core.tags` API, which has no stability guarantee.
- **Not built:** vision metrics, duplicate detection, the instruction-versus-video check, task rewrites, and trimming.

## Develop and validate

Run from the folder that contains `lerobot_data_curation/`:

```bash
python -m unittest discover -s lerobot_data_curation/tests -t .    # 115 tests, about 12 seconds
python -m lerobot_data_curation.harness.run_harness --dataset <your dataset>   # writes harness/REPORT.md
```

The harness pairs each real episode with a corrupted copy of itself (added hesitation, inserted idle time, a duplicated frame, a blanked task, a stalled joint, and so on) and checks that each metric ranks the copy worse than the original. It also re-scores every episode truncated to a common length to find length-confounded metrics. `harness/REPORT.md` holds the latest results.

To add your own metric or chart, see [EXTENDING.md](EXTENDING.md).

## Design documents

- [METRICS.md](METRICS.md): every metric, what it means, how it is scored, and the evidence behind it.
- [UX.md](UX.md): the panel and operator design.
- [ARCHITECTURE.md](ARCHITECTURE.md): module layout, the two-pass pipeline, field names and known risks.
- [PLAN.md](PLAN.md): the build plan and progress log.
