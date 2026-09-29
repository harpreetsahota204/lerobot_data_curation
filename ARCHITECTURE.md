# LeRobot Data Curation: Architecture

Status: v1, built through slice 2 (all phase 1 metrics, the five-tab panel, and the inspector). Phases 2 (vision) and 3 (VLM) are not built. Decisions below came from the architecture round of multiple-choice questions.

## Decisions

| Topic | Decision |
|---|---|
| Layout | Mirrors `demo_quality_scorer`: `operators.py`, `panel_ops.py`, `panel_data.py`, `write.py`, an `engine/` package and a React `src/` |
| Extensibility | Scaled back. A metric is a plain function plus one dict entry. No registry, no auto-discovery, no auto-generated panel section. `EXTENDING.md` documents how to add a metric and how to add a chart |
| Execution | Sequential raw per-episode pass inside a delegated operator with progress, like the MCAP plugin. Keep a clean seam so a pool can be added later |
| Storage | Flat fields on each episode sample (naming below), plus a run record for normalization stats and confirmed settings |
| Build order | Thin vertical slice first, then widen family by family |
| Test data | `lr_dev`: a clone of `community_v3_10per_embodiment` with 2 episodes from each of its 51 sources (102 episodes) |
| Normalization | Unchanged from `METRICS.md`: task group when it has at least 20 episodes, otherwise pool over the dataset |
| VLM (phase 3) | On hold. Runtime undecided |

## Module layout

```
lerobot_data_curation/
  fiftyone.yml, __init__.py        # registers operators + panel
  operators.py                     # Compute quality, Compute vision (VLM check on hold)
  panel_ops.py, panel_data.py      # unlisted backend, one-call payload, get_episode_detail for the inspector
  write.py                         # flat fields, intervals, temporal tags
  engine/
    reader.py                      # EpisodeData (exists)
    dataset_checks.py              # the 6 infer/confirm checks
    normalize.py                   # ported from the MCAP plugin, plus group + fallback ladder
    metrics/                       # __init__.py holds the METRICS dict; one module per family
    profiles.py                    # groups, worst-of aggregation, presets
    score.py                       # raw per-episode pass, then finalize-batch pass
  src/                             # React: 5 tabs, Inspector, charts, ui, theme (shell reused)
  harness/                         # corruption injection + tests
  EXTENDING.md                     # how to add a metric and a chart
```

## Pipeline

1. **Dataset checks** run once and are confirmed or overridden in the first step of the Compute quality form. They produce the `feature_map` and the other assumptions every metric reads.
2. **Raw pass:** for each episode, read arrays through the `feature_map`, then compute every enabled metric. Independent per episode.
3. **Finalize pass:** fit robust normalization within each group, apply the fallback ladder, compute group values and profile scores (max across groups), outlier scores and the spans worth a timeline tag.
4. **Write:** flat fields, plus flagged spans as temporal tags (anchor `lerobot_data_curation`). The normalization stats, confirmed dataset checks, profile settings and thresholds go in the run record.

## Storage: flat fields

All fields start with `lr_`. Signal slugs are sanitized (`arm_left`, `arm_right`, `arm_all`, `cam_<key>`).

| Field | Meaning |
|---|---|
| `lr_<metric>` | Worst-of value for that metric (e.g. `lr_sparc`, `lr_idle_lead_s`) |
| `lr_<metric>_on_<signal>` | Per-arm or per-camera value (e.g. `lr_sparc_on_arm_left`). No double underscores: mongoengine reads `__` as a lookup separator |
| `lr_<metric>_signal` | Which signal produced the worst-of value |
| `lr_z_<metric>` | Oriented robust z (higher is worse) |
| `lr_<metric>_note` | Reason text for a flag |
| `lr_<metric>_worst` | Worst-window tail for windowed metrics |
| `lr_score_<profile>`, `lr_nflags_<profile>`, `lr_driver_<profile>`, `lr_verdict_<profile>` | Profile outputs |
| `lr_integrity_verdict`, `lr_integrity_reason`, `lr_language_verdict` | Verdicts for the never-scored families |
| `lr_keep_from_s`, `lr_keep_to_s`, `lr_dup_group` | Suggestions only, never applied |
| `lr_config_version` | Formula version |

A plugin-managed sidebar group keeps this many fields readable in the App. Whether the App lets a plugin set sidebar groups this way still needs to be checked.

## Build order

| Slice | Contents | Estimate |
|---|---|---|
| 1 | Reader fixes and `feature_map` inference. One metric per family: `sparc`, `idle_lead_s`, `length_z`, `frame_gaps`, `task_generic`. Normalization port with the ladder. Flat writes. Compute quality operator. Overview tab: worst-first table and score histogram | 2 to 3 hours of implementation, plus your review |
| 2 | Rest of phase 1 metrics, dataset-check confirm UI, Motion & Action and Integrity & Coverage tabs, inspector, tag buttons | about 1 day |
| 3 | Harness: corruption tests per metric, length control. Written alongside slices 1 and 2, then completed here | half a day |
| 4 | Phase 2: Compute vision operator, Vision tab | about 1 day |
| 5 | `EXTENDING.md` | 1 hour |
| on hold | Phase 3 VLM | not scheduled |

The estimates are my guesses for the executing agent, not measurements.

## Known risks

1. **Small test groups.** `lr_dev` has 2 episodes per source and 47 distinct task strings, and the largest group sharing one task string is 6. With the 20-episode threshold kept, every group uses the pooled fallback across 51 different robots and `fps` values (10 to 50). The primary per-task path is covered only by synthetic tests.
2. **Field volume.** Flat per-signal fields multiply. Metrics times arms or cameras can reach hundreds of fields. The sidebar group is the mitigation.
3. **Test data gaps.** 10 of 50 sources have unnamed action dimensions and 6 have no gripper, so per-arm grouping and gripper metrics switch off there. One source has no action feature. That is good coverage of the "metric switches off" path and weaker coverage of named per-arm scoring.
4. **Temporal tags use an internal FiftyOne API** (`fiftyone.core.tags`), as in the MCAP plugin. It has no stability guarantee. Tags written this way render on native LeRobot episodes (see risk 7).
5. **`engine/reader.py` fixes:** done in slice 1 (raw frame order kept, one read per parquet file per batch, arrays read through `feature_map`).
6. **Payload size.** The panel loads every episode's metrics in one call: about 2.7 KB per episode, so roughly 13 MB at 5,000 episodes. Pagination is not built.
7. **Temporal tags render on LeRobot episodes.** A first test said no; a retest with one unanchored and one anchored probe tag showed both on the timelines of `lr_dev` episodes, and the cause of the first negative result is unknown. The plugin writes flagged spans as tags anchored `lerobot_data_curation`: only spans whose metric is at warn or worse, plus regrasp recoveries as information. A re-run replaces them and never touches tags drawn by hand.
8. **Restarting the App server.** `pkill -f` on a command line that contains the process name also kills the calling shell. Stop the server by PID, and restart it after any Python change because operators register once per server start.
