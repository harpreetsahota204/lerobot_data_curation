# Extending the plugin

Adding a metric takes a function, one table entry, one corruption test and, to score it, one line naming its group. In an existing family it appears in the form, the panel charts and the ranking table without touching the frontend. This page walks through one worked example, then the other extension points.

Everything below was checked against the code: the example metric was added to a running copy and produced its `lr_*` fields, its panel metadata and a place in the `motion` group.

## Add a metric

Example: `peak_speed`, the 95th percentile arm speed. High means abrupt motion.

**1. Write the function** in a module under `engine/metrics/` (a new file, or one of the existing family files). It takes an `EpisodeData` and returns `{signal: MV}`:

```python
import numpy as np

from ..signals import all_arm_speed
from .base import MV


def peak_speed(ep):
    """The 95th percentile arm speed, in range-fractions per second."""
    speed = all_arm_speed(ep)
    if speed is None:
        return {}          # no result: the metric is simply absent for this episode
    return {"": MV(float(np.percentile(speed, 95)))}
```

- `signal` is `""` for one value per episode, or a slug such as `arm_left` for one value per arm. Set `per_signal=True` in the entry if you return several.
- Return `{}` when the episode cannot carry the metric. Never raise for missing data; an exception is caught and recorded, but the metric then shows as failed.
- An `EpisodeData` has `action`, `state` (each `(T, D)` or `None`), `fps`, `length`, `tasks`, `action_names`, `state_names`, `videos`, `groups` (the joint groups the user picked), `assumptions` and more. See `engine/reader.py`. The helpers in `engine/signals.py` give you range-normalized speeds per arm (`arm_speeds`, `all_arm_speed`), gripper signals and joint-space checks.
- Use `MV(value, worst=None, note=None)`. `worst` is a bad-tail value for windowed metrics; `note` is a short reason shown in the panel, mostly for flags.

**2. Add one entry** to `METRICS` in `engine/metrics/__init__.py`:

```python
"peak_speed": spec(
    motion.peak_speed, "motion", group="motion", requires=("action",),
    description="95th percentile arm speed. High means abrupt, fast motion.",
),
```

`spec()` fills in the defaults. Its arguments:

| Argument | Meaning |
|---|---|
| `fn`, `family`, `description` | The function, the family tab it belongs to, and one plain sentence shown in tooltips |
| `group` | The profile group it votes in. Leave it `None` for a metric that is shown but never scored |
| `requires` | Arrays the episode needs: `"action"`, `"state"`. The metric is skipped without them |
| `kind` | `"value"` (z-scored), `"signed_z"` (turned into a signed z within its group, scored as \|z\|) or `"flag"` (0 or 1, never z-scored) |
| `higher_is_worse` | The metric's raw direction (default `True`) |
| `per_signal` | Whether it returns one value per arm or camera |
| `opt_in` | Off unless the user ticks it |
| `batch` | `True` if `fn(raws, label_of)` compares an episode with its peers |
| `check` | Integrity metrics only: `("fail" or "warn", threshold)` sets the integrity verdict |

**3. Score it.** A metric only affects a profile if it is a member of a group. Add it to `GROUPS` in `engine/profiles.py` with a weight (1.0 is a full vote):

```python
"motion": {"sparc": 1.0, "ldlj": 0.5, "sparc_phase": 1.0, "peak_speed": 1.0},
```

Correlated metrics share a group, and a group counts as one vote. Inside a group the profile takes the worst weighted z-score, and across groups the worst group, so one bad group is never diluted.

**4. Add a corruption test.** Every metric ships with a defect it must catch. Add a corruption to `harness/corruptions.py` and list the metric in its `expect`:

```python
def abrupt(ep, rng):
    """Speed up a stretch of the episode 4 times."""
    if ep.action is None or len(ep.action) < 120:
        return None                       # None means "this episode cannot carry the defect"
    ...                                   # return a copy built with dataclasses.replace(ep, action=...)

Corruption("abrupt", "Speed up a stretch 4x", abrupt, ("peak_speed",)),
```

`tests/test_harness.py` fails if a metric has neither a corruption nor an entry in `NOT_TESTABLE` explaining why it cannot be tested, so a new metric cannot ship untested. Run `python -m lerobot_data_curation.harness.run_harness --dataset <name>` to see it pass or fail on real episodes.

**5. Bump the formula version** if the change moves scores already written: increase `CONFIG_VERSION` in `engine/score.py`. The panel then warns when a view mixes scores from different versions.

### Where it appears

- **Existing family** (`motion`, `time`, `tracking`, `gripper`, `consistency`): the Motion & Action tab picks it up automatically, one histogram and one table column. It falls back to the metric's name for its title; add a nicer title in `METRIC_LABELS` in `src/types.ts` and rebuild.
- **New family:** add the family name to the `families` list for the Motion & Action tab in `src/CurationPanel.tsx`, add a title in `FAMILY_TITLES` in `src/MetricGrid.tsx`, and a label in `FAMILY_LABELS` in `operators.py`. Or give it its own tab (see below).
- **Fields written:** `lr_peak_speed` (the value), `lr_z_peak_speed` (the robust z), plus `lr_peak_speed_on_<signal>` and `lr_peak_speed_signal` for per-signal metrics.

## Add a chart or a tab

The panel is React (`src/`), built with Vite. The tabs are handwritten; there is no auto-generated section for extra metrics.

- **A chart on an existing tab:** the family tabs use `MetricGrid.tsx`, which already draws a histogram per metric. To draw something different, copy one card from `MetricGrid.tsx` and use the components in `charts.tsx` (`MetricHistogram`, `VerdictBar`, `CountBars`, `OutlierScatter`) or Recharts directly.
- **A new tab:** copy `LanguageTab.tsx` (the smallest), add it to `TABS` in `CurationPanel.tsx`, and render it in the tab switch. Every row the panel receives is in `data.rows`, with `values`, `z`, `by_signal` and `notes` per metric (`src/types.ts`).
- **New data for the panel:** add it to `build_panel_data` in `panel_data.py` and to `PanelData` in `src/types.ts`. The panel makes one backend call per refresh.
- **Rebuild:** `npm install && npm run build`, then hard-refresh the browser. `npm run dev` rebuilds on save.

## Add a scoring profile

A profile is a named set of groups. Add one to `PROFILES` in `engine/profiles.py`:

```python
"my_robot": {"label": "My robot", "groups": ["motion", "tracking"]},
```

It is scored and written like the built-in ones (`lr_score_my_robot`, `lr_verdict_my_robot`, ...) and a Profile dropdown appears in the panel header (it is hidden while there is only one profile).

## Add something the user must pick

The plugin never infers what a dataset means. What the user tells it lives in the picks dict (`engine/picks.py`). To add a pick: add the key to `empty_picks` and `clean_picks`, add a dropdown for it in the Data tab of `resolve_input` in `operators.py` (`_dropdown` gives you a "Not set" first choice) and read it in `_collect_picks`, then expose it to metrics through `assumptions_for` and read it from `ep.assumptions`. Add a sentence to `explain_off` saying what switches off without it. A metric that needs the pick must return `{}` when it is missing, never a default. Joint groups work differently: they are attached to the episode as `ep.groups` by the reader.

## Tests

- Unit tests use synthetic episodes (`tests/helpers.py`: `make_episode`, `smooth_action`). Use the standard library `unittest`; no extra packages are needed.
- Run everything from the folder that contains `lerobot_data_curation/` with `python -m unittest discover -s lerobot_data_curation/tests -t .`. Most tests use synthetic episodes; `test_write.py` uses a temporary FiftyOne dataset of plain samples.

## Two gotchas

- **Field names cannot contain double underscores.** mongoengine reads `__` as a lookup separator, so `lr_<metric>__<signal>` breaks. Per-signal fields use `_on_`.
- **Restart the App server after any Python change.** Operators register once per server start. Stop the server by its process id (a `pkill -f` on a command line that contains the process name also kills the shell you ran it from).
