"""Writes engine results onto FiftyOne samples as flat ``lr_*`` fields.

Kept separate from the engine so the engine has no FiftyOne dependency and stays
independently testable. Field naming (see ARCHITECTURE.md):

- ``lr_<metric>``: the metric's worst-of value (for ``length_z``, the signed z)
- ``lr_<metric>_raw``: the raw value behind a signed-z metric
- ``lr_<metric>_worst``: worst-window tail, for windowed metrics
- ``lr_<metric>_on_<signal>``: per-arm or per-camera value
- ``lr_<metric>_signal``: which signal produced the worst-of value

No double underscores: mongoengine reads ``__`` as a lookup separator.
- ``lr_<metric>_note``: reason text for a flag
- ``lr_z_<metric>``: oriented robust z (higher is worse)
- ``lr_score_<profile>``, ``lr_nflags_<profile>``, ``lr_driver_<profile>``,
  ``lr_verdict_<profile>``: profile outputs
- ``lr_integrity_verdict``, ``lr_language_verdict``
- ``lr_group``, ``lr_group_basis``, ``lr_group_n``, ``lr_config_version``, ``lr_run_id``
"""

import math

import fiftyone as fo
import fiftyone.core.tags as fota

from .engine.score import CONFIG_VERSION

RUN_KEY = "lerobot_data_curation"
# Every temporal tag this plugin writes carries this anchor, so a re-run clears
# exactly its own tags and never touches one drawn by hand on the timeline.
TEMPORAL_TAG_ANCHOR = "lerobot_data_curation"
RECOVERY_TAG = "recovery"
SIDEBAR_GROUP = "LeRobot curation"


def _finite(value):
    return value is not None and not (isinstance(value, float) and (math.isnan(value) or math.isinf(value)))


def flatten(result):
    """One episode's result as ``{field_name: value}``. Missing values are omitted."""
    out = {}
    for name, m in result.metrics.items():
        out["lr_%s" % name] = float(m["value"])
        if _finite(m.get("raw")):
            out["lr_%s_raw" % name] = float(m["raw"])
        if _finite(m.get("worst")):
            out["lr_%s_worst" % name] = float(m["worst"])
        if _finite(m.get("z")):
            out["lr_z_%s" % name] = float(m["z"])
        if m.get("worst_signal"):
            out["lr_%s_signal" % name] = m["worst_signal"]
        if m.get("note"):
            out["lr_%s_note" % name] = m["note"]
        for signal, sv in (m.get("by_signal") or {}).items():
            out["lr_%s_on_%s" % (name, signal)] = float(sv["value"])

    for profile, p in result.profiles.items():
        if _finite(p["score"]):
            out["lr_score_%s" % profile] = float(p["score"])
        out["lr_nflags_%s" % profile] = int(p["n_flags"])
        if p["driver"]:
            out["lr_driver_%s" % profile] = p["driver"]
        out["lr_verdict_%s" % profile] = p["verdict"]

    out["lr_integrity_verdict"] = result.integrity_verdict
    out["lr_language_verdict"] = result.language_verdict
    out["lr_group"] = result.group_label
    out["lr_group_basis"] = result.group_basis
    out["lr_group_n"] = int(result.group_n)
    out["lr_config_version"] = int(result.config_version)
    return out


def _field_type(value):
    if isinstance(value, bool):
        return fo.BooleanField
    if isinstance(value, int):
        return fo.IntField
    if isinstance(value, float):
        return fo.FloatField
    return fo.StringField


def declare_fields(dataset, flat_by_id):
    """Adds every field the run will write, with an explicit type.

    Declared up front so a field whose first value happens to be missing is not
    left without a schema entry.
    """
    schema = dataset.get_field_schema()
    types = {}
    for flat in flat_by_id.values():
        for name, value in flat.items():
            types.setdefault(name, _field_type(value))
    for name, ftype in types.items():
        if name not in schema:
            dataset.add_sample_field(name, ftype)
    return sorted(types)


def previous_fields(dataset):
    """Field names written by the previous run, from its run record."""
    if not dataset.has_run(RUN_KEY):
        return []
    try:
        results = dataset.load_run_results(RUN_KEY)
        return list(getattr(results, "fields", []) or [])
    except Exception:  # noqa: BLE001 - a corrupt record must not block a re-run
        return []


def write_results(dataset, view, results, run_id=None):
    """Writes results for the samples in `view`. Returns the sorted field names.

    `run_id` is stamped on every scored sample (``lr_run_id``) so the panel can tell
    when a view mixes episodes scored in different runs.

    Fields the previous run wrote but this run did not are set to None on the
    scored samples, so a re-run with different settings leaves no stale values.
    Samples outside `view` are untouched.
    """
    flat_by_id = {sid: flatten(r) for sid, r in results.items()}
    if run_id is not None:
        for flat in flat_by_id.values():
            flat["lr_run_id"] = run_id
    fields = declare_fields(dataset, flat_by_id)
    stale = set(previous_fields(dataset)) - set(fields)
    schema = dataset.get_field_schema()
    stale = {f for f in stale if f in schema}

    for sample in view.select(list(results), ordered=False).iter_samples(autosave=True):
        flat = flat_by_id[sample.id]
        for name, value in flat.items():
            sample[name] = value
        # Recovery is neutral information, surfaced as a tag so it can be filtered.
        if flat.get("lr_recovery_count", 0) > 0:
            sample.tags = sorted(set(sample.tags) | {RECOVERY_TAG})
        elif RECOVERY_TAG in sample.tags:
            sample.tags = [t for t in sample.tags if t != RECOVERY_TAG]
        for name in fields:
            if name not in flat:
                sample[name] = None
        for name in stale:
            sample[name] = None
    return fields


def set_sidebar_group(dataset, fields):
    """Puts the written fields under one sidebar heading, headline fields first."""
    groups = dataset.app_config.sidebar_groups
    if groups is None:
        groups = fo.DatasetAppConfig.default_sidebar_groups(dataset)

    headline = [
        f
        for f in fields
        if f.startswith(("lr_score_", "lr_verdict_", "lr_nflags_", "lr_driver_", "lr_integrity", "lr_language"))
    ]
    ordered = headline + [f for f in fields if f not in headline]

    for g in groups:
        g.paths = [p for p in g.paths if p not in ordered]
    existing = next((g for g in groups if g.name == SIDEBAR_GROUP), None)
    if existing is None:
        groups.append(fo.SidebarGroupDocument(name=SIDEBAR_GROUP, paths=ordered, expanded=True))
    else:
        existing.paths = ordered
    dataset.app_config.sidebar_groups = groups
    dataset.save()


def register_run(dataset, config, norm_stats, fields, feature_maps, dataset_checks=None, balance=None, run_id=None):
    """Records what produced this run's scores and the stats needed to reuse them.

    Overwrites on every run: this is the dataset's current scoring state, not a
    history.
    """
    cfg = dataset.init_run(config_version=CONFIG_VERSION, **config)
    dataset.register_run(RUN_KEY, cfg, overwrite=True)
    results = dataset.init_run_results(RUN_KEY)
    results.norm_stats = norm_stats
    results.fields = list(fields)
    results.feature_maps = feature_maps
    results.dataset_checks = dataset_checks or {}
    results.balance = balance or {}
    results.run_id = run_id
    dataset.save_run_results(RUN_KEY, results, overwrite=True)


def build_temporal_tags(sample_id, spans):
    """Flagged spans as multimodal temporal tags (they render on the episode's timeline).

    Uses ``fiftyone.core.tags``, an internal and undocumented API with no stability
    guarantee. Verified to render on native LeRobot episodes.
    """
    tags = []
    for span in spans:
        start_ns = round(span["start_s"] * 1e9)
        end_ns = round(span["end_s"] * 1e9)
        if end_ns <= start_ns:
            end_ns = start_ns + 1  # temporal tags require start < end
        tags.append(
            fota.TemporalTag(
                sample_id=sample_id,
                start=start_ns,
                end=end_ns,
                tag="%s:%s" % (span["label"], span["severity"]),
                anchor=TEMPORAL_TAG_ANCHOR,
            )
        )
    return tags


def clear_temporal_tags(view):
    """Deletes this plugin's own temporal tags in the given scope."""
    view.temporal_tags.delete(filter=fota.TemporalTagFilter(anchors=TEMPORAL_TAG_ANCHOR))


def write_temporal_tags(view, results):
    """Replaces this plugin's temporal tags on the scored samples. Returns the tag count."""
    clear_temporal_tags(view)
    tags = []
    for sid, result in results.items():
        tags.extend(build_temporal_tags(sid, result.spans))
    if tags:
        fota.add_temporal_tags(view, tags)
    return len(tags)
