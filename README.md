# LeRobot Data Curation

A FiftyOne plugin that ranks the episodes of a LeRobot v3 dataset worst-first, so you know which to look at before training a policy or fine-tuning a VLA on them. It scores each episode on motion smoothness, time efficiency, tracking, gripper behavior, consistency, integrity and language, optionally adds camera quality (blur, exposure, clipping, frozen feeds, video-action lag), and shows the results in a panel.

**Use it to decide where to look first. Do not use it as an automatic accept/reject gate.** A smooth, well-timed episode can still show the wrong task, and detection accuracy has not been shown to predict policy quality (see [METRICS.md](METRICS.md), Evidence). The scores order your review queue. A person decides.

The plugin has no model-based metrics by design, so there is no model to download or install.

## Contents

- [Install](#install)
- [Score a dataset](#score-a-dataset)
- [The panel](#the-panel)
- [Tagging](#tagging)
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

2. Open the dataset in the App and run **LeRobot curation: compute quality** from the operator browser, or click the button in the empty panel. It runs delegated by default. The form has four tabs:
   - **Data:** one line of what the dataset declares, then three short sections. **Robot:** Single arm or Dual arm, and a **Has multi-joint hands** checkbox (off by default). **Arrays:** the **state** and **action** arrays, which start on `observation.state` and `action` when the dataset uses those LeRobot standard names and at Not set otherwise. When both are picked and the same size, a yes/no question asks whether the action is absolute joint positions; when they differ in size, a note says the action cannot be joint positions. **Joints:** chip fields for the Arm and the Gripper (a left and a right set for dual arm, plus Hand fields if the checkbox is on), picked the same way as cameras. Each joint is offered by name with its observed range; you pick every joint yourself, and a picked joint leaves every field's list. The gripper's open direction appears right under the Gripper field once gripper joints are picked. Nothing is guessed from the data. A list at the bottom of the form shows which metric families will be scored and what each switched-off family still needs. Your picks are remembered for the next run.
   - **Metrics:** one checkbox per metric, in a collapsible section per family. Each section title shows how many of its metrics are selected, with a line under it on what the family is for. Opt-in metrics are off by default.
   - **Camera:** the video cameras to score (with their resolution), and the five camera metrics. No camera is selected for you, because decoding is the slow part (about 0.4 s per camera per episode).
   - **Normalization:** the smallest task group that is normalized on its own (default 20 episodes).

3. Optional: pick cameras on the **Camera** tab (or click **Compute camera metrics** on the Vision tab, which opens the form there). Every metric is recomputed together, so the profile score includes the camera group. The cameras and metrics you used stay selected on the next run.

4. Open a new panel and choose **LeRobot Curation**. It refreshes whenever the view changes, so filtering the grid re-ranks the panel.

Scores are batch-relative: an episode is compared with the other episodes of its task in the view you scored, or with the whole view when its task has fewer than 20 episodes. The panel says so when that happens. Score episodes of the same task together when you want a clean ranking.

## The panel

A **Task** dropdown sits above the tabs. Every episode gets one score from the motion, time, tracking, gripper, consistency and (when computed) camera groups. The task-string check is a separate Language verdict with its own column: if you are fine-tuning a VLA, read it alongside the score, because a weak instruction hurts a VLA even when the motion is clean.

| Tab | What it shows |
|---|---|
| **Overview** | Score histogram, verdict counts, episodes per task, outlier scatter and the worst-first ranking. |
| **Motion & Action** | One histogram per metric (smoothness, time, tracking, gripper, consistency), signal chips to isolate one arm, and a worst-first table. |
| **Integrity & Coverage** | Integrity verdicts, episodes per task, and a per-episode list of which check failed. |
| **Vision** | The five camera metrics (blur, exposure, clipped pixels, frozen feed, video-action lag): one histogram each, camera chips to isolate one camera, and a worst-first table. Empty until the camera metrics are computed. |
| **Language** | Static checks on the task string: missing, placeholder, too short, no action verb. |

Click any bar to filter the samples grid to those episodes. Most charts have an expand button that gives one chart the full width, and an "i" icon that explains it. Click any row to open the **inspector** under the tables: every metric with its per-arm or per-camera breakdown, joint traces (action solid, state dashed), the speed profile with the idle threshold, the gripper timeline, and a few frames from the first two cameras, picked around the flagged spans. The icons in the inspector's top-right corner open the episode in the multimodal viewer (the square with an arrow) and close the inspector (the X).

Flagged spans (idle stretches, the longest pause, the roughest smoothness window, acceleration spikes) are also written as temporal tags on each episode's timeline when their metric reaches warn, so you can scrub straight to them. Regrasp recoveries are tagged too, as information. A re-run replaces the plugin's tags and leaves tags you drew yourself alone.

## Tagging

The footer has three tag buttons, `review`, `exclude-candidate` and `relabel`, which tag the selection (or everything in view when nothing is selected), and a **Refresh** button. Nothing is deleted or hidden. The tags are ordinary FiftyOne sample tags, so you can filter on them, or act on them with your own code or another FiftyOne workflow.

## Limits

- **Scores are triage, not verdicts.** Smoothness and timing say nothing about whether the demonstration did the right thing.
- **Small groups.** Per-task normalization needs about 20 episodes per task. With fewer, episodes are compared across tasks and robots, and the panel shows a low-confidence banner.
- **You decide what the arrays mean.** With no arm joints picked there is no smoothness, idle or pause score, and with no gripper joints there are no gripper metrics. If the dataset gives no joint names, you pick joints by position (`action[0]`, `action[1]`, ...) in the same picker, using the range beside each one to tell them apart.
- **Tracking, acceleration-spike and joint-limit metrics** only run when you declare the action to be joint positions in the state's space (leader-follower teleoperation, for example). They read the state with the action's arm dimensions, so they need the two arrays to have the same shape.
- **Local datasets only.** Cloud-hosted LeRobot sources are not supported yet.
- **Large datasets.** The panel loads every episode in one call, about 3 KB each, with no pagination.
- **Tuned on one dataset.** Default thresholds (the acceleration-spike floor, the 2% joint-limit margin, the idle threshold, the camera thresholds) were calibrated on a 102-episode development set and are not yet adjustable in the form. Check them on your own data.
- **Temporal tags** use FiftyOne's internal `fiftyone.core.tags` API, which has no stability guarantee.
- **Camera metrics are relative.** Each camera is compared with the same camera in other episodes, so a view that pools many robots (or a task group under 20 episodes) will flag more episodes than a single-task dataset does. A camera that cannot see the robot gets no video-action lag.
- **At most two arms.** A robot with more arms, or with a torso, head or legs, leaves those joints unpicked.
- **Cameras stored as images** (instead of video) cannot be read by the camera metrics and are not offered.
- **Not built, by design:** anything that needs a model (duplicate detection by embedding, the instruction-versus-video check, task rewrites), and trimming.

## Develop and validate

Run from the folder that contains `lerobot_data_curation/`:

```bash
python -m unittest discover -s lerobot_data_curation/tests -t .    # 248 tests, about 16 seconds
python -m lerobot_data_curation.harness.run_harness --dataset <your dataset>   # writes harness/REPORT.md and report.json
```

The harness pairs each real episode with a corrupted copy of itself (added hesitation, inserted idle time, a duplicated frame, a blanked task, a stalled joint, a blurred or frozen video, and so on) and checks that each metric ranks the copy worse than the original. It also re-scores every episode truncated to a common length to find length-confounded metrics. It needs a FiftyOne LeRobot dataset, and takes about 5 minutes on 102 episodes with the camera metrics included. The report is generated, so run it to see the results for your data.

To add your own metric or chart, see [EXTENDING.md](EXTENDING.md).

## Design documents

- [METRICS.md](METRICS.md): every metric, what it means, how it is scored, and the evidence behind it.
- [UX.md](UX.md): the panel and operator design.
