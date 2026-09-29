"""Builds the panel's data payload in one call.

Everything the frontend renders (rows, metric values, per-signal breakdowns,
profile scores, verdicts, warn thresholds, group sizes) is assembled here so the
panel needs exactly one backend call per refresh.
"""

import math

from .engine import normalize
from .engine.dataset_checks import UNDER_COVERED
from .engine.metrics import METRICS
from .engine.profiles import DEFAULT_PROFILE, GROUPS, PROFILES
from .write import RUN_KEY

SCORED_METRICS = [name for name, spec in METRICS.items() if spec["scored"]]


def _num(value):
    if value is None:
        return None
    value = float(value)
    return None if (math.isnan(value) or math.isinf(value)) else value


def _display(name, spec):
    """Signed-z metrics are stored as a z-score. Show the raw value unless the
    metric is itself a z (its name ends in ``_z``), so a lag reads in ms."""
    return "raw" if spec["kind"] == "signed_z" and not name.endswith("_z") else "value"


def metric_meta():
    """What the panel needs to know about each metric (labels, direction, grouping)."""
    out = {}
    for name, spec in METRICS.items():
        if spec["fn"] is None and spec["family"] != "outliers":
            continue
        member_of = next((g for g, members in GROUPS.items() if name in members), None)
        out[name] = {
            "family": spec["family"],
            "group": member_of,
            "description": spec["description"],
            "higher_is_worse": spec["higher_is_worse"],
            "scored": spec["scored"],
            "kind": spec["kind"],
            "per_signal": spec["per_signal"],
            "opt_in": spec["opt_in"],
            "check": list(spec["check"]) if spec["check"] else None,
            "display": _display(name, spec),
        }
    return out


def _run_results(dataset):
    if not dataset.has_run(RUN_KEY):
        return None, {}
    try:
        info = dataset.get_run_info(RUN_KEY)
        return dataset.load_run_results(RUN_KEY), info.config
    except Exception:  # noqa: BLE001 - the panel still renders without the run record
        return None, {}


def _min_group(config):
    try:
        return int(config.min_group)
    except Exception:  # noqa: BLE001
        return 20


def _warn_thresholds(results, group_labels, run_ids):
    """Raw-unit warn thresholds, only when every episode shares one group and came from the recorded run.

    The normalization stats belong to the last run. Rows scored in an earlier run
    were measured against different stats, so a threshold drawn from these would mislead.
    """
    if results is None or len(set(group_labels)) != 1:
        return {}
    if len(run_ids) != 1 or run_ids != {getattr(results, "run_id", None)}:
        return {}
    stats_by_key = (getattr(results, "norm_stats", None) or {}).get(group_labels[0])
    if not stats_by_key:
        return {}
    out = {}
    for key, stats in stats_by_key.items():
        metric, _, signal = key.partition("|")
        if metric not in METRICS or METRICS[metric]["kind"] != "value":
            continue
        out.setdefault(metric, {})[signal] = normalize.raw_value_at_z(
            normalize.WARN_Z, stats, METRICS[metric]["higher_is_worse"]
        )
    return out


def build_panel_data(dataset, view):
    """The full JSON-serializable payload for one panel refresh."""
    schema = dataset.get_field_schema()
    if "lr_config_version" not in schema:
        return {"scored": False}

    profile_fields = []
    for p in PROFILES:
        profile_fields += ["lr_score_%s" % p, "lr_nflags_%s" % p, "lr_driver_%s" % p, "lr_verdict_%s" % p]

    signal_fields = {}  # metric -> {signal: field}
    for name in METRICS:
        prefix = "lr_%s_on_" % name
        found = {f[len(prefix) :]: f for f in schema if f.startswith(prefix)}
        if found:
            signal_fields[name] = found

    value_fields = ["lr_%s" % m for m in METRICS if "lr_%s" % m in schema]
    raw_fields = ["lr_%s_raw" % m for m in METRICS if "lr_%s_raw" % m in schema]
    z_fields = ["lr_z_%s" % m for m in METRICS if "lr_z_%s" % m in schema]
    note_fields = ["lr_%s_note" % m for m in METRICS if "lr_%s_note" % m in schema]
    per_signal_fields = [f for found in signal_fields.values() for f in found.values()]

    base = [
        "id",
        "episode_index",
        "task",
        "robot_type",
        "media_reference.key",
        "lr_group",
        "lr_group_basis",
        "lr_group_n",
        "lr_config_version",
        "lr_run_id",
        "lr_integrity_verdict",
        "lr_language_verdict",
    ]
    base = [f for f in base if f == "id" or f.split(".")[0] in schema]
    fields = base + [
        f
        for f in profile_fields + value_fields + raw_fields + z_fields + note_fields + per_signal_fields
        if f in schema
    ]
    columns = dict(zip(fields, view.values(fields)))

    source_dir = {s["id"]: s.get("dir") or s["id"][-6:] for s in (dataset.media_sources or [])}
    n = len(columns["id"])
    rows, versions = [], set()
    for i in range(n):
        get = lambda f, i=i: columns[f][i] if f in columns else None  # noqa: E731
        if get("lr_config_version") is None:
            continue  # never scored
        versions.add(get("lr_config_version"))
        source_id = (get("media_reference.key") or "").rpartition("/")[0]
        rows.append(
            {
                "id": get("id"),
                "run_id": get("lr_run_id"),
                "episode": "%s / ep %s" % (source_dir.get(source_id, "?"), get("episode_index")),
                "source": source_dir.get(source_id, "?"),
                "task": get("task") or "",
                "robot": get("robot_type") or "",
                "group": get("lr_group"),
                "group_basis": get("lr_group_basis"),
                "group_n": get("lr_group_n"),
                "integrity": get("lr_integrity_verdict") or "unknown",
                "language": get("lr_language_verdict") or "unknown",
                "profiles": {
                    p: {
                        "score": _num(get("lr_score_%s" % p)),
                        "n_flags": get("lr_nflags_%s" % p) or 0,
                        "driver": get("lr_driver_%s" % p),
                        "verdict": get("lr_verdict_%s" % p) or "unknown",
                    }
                    for p in PROFILES
                },
                "values": {
                    m: _num(get("lr_%s_raw" % m) if _display(m, METRICS[m]) == "raw" and get("lr_%s_raw" % m) is not None else get("lr_%s" % m))
                    for m in METRICS
                    if "lr_%s" % m in columns and get("lr_%s" % m) is not None
                },
                "raw": {
                    m: _num(get("lr_%s_raw" % m)) for m in METRICS if "lr_%s_raw" % m in columns and get("lr_%s_raw" % m) is not None
                },
                "z": {
                    m: _num(get("lr_z_%s" % m)) for m in METRICS if "lr_z_%s" % m in columns and get("lr_z_%s" % m) is not None
                },
                "notes": {
                    m: get("lr_%s_note" % m) for m in METRICS if "lr_%s_note" % m in columns and get("lr_%s_note" % m)
                },
                "by_signal": {
                    m: {
                        sig: _num(get(field))
                        for sig, field in found.items()
                        if field in columns and get(field) is not None
                    }
                    for m, found in signal_fields.items()
                    if any(f in columns and get(f) is not None for f in found.values())
                },
            }
        )

    task_counts = {}
    for r in rows:
        task_counts[r["task"]] = task_counts.get(r["task"], 0) + 1

    results, config = _run_results(dataset)
    min_group = _min_group(config)
    computed = sorted({m for r in rows for m in r["values"]})
    signals = sorted({s for r in rows for sigs in r["by_signal"].values() for s in sigs})

    return {
        "scored": True,
        "rows": rows,
        "metrics": metric_meta(),
        "computed": computed,
        "signals": signals,
        "profiles": [{"id": p, "label": spec["label"]} for p, spec in PROFILES.items()],
        "default_profile": DEFAULT_PROFILE,
        "tasks": [{"task": t, "n": c} for t, c in sorted(task_counts.items(), key=lambda kv: -kv[1])],
        "warn_thresholds": _warn_thresholds(results, [r["group"] for r in rows], {r["run_id"] for r in rows}),
        "mixed_runs": len({r["run_id"] for r in rows}) > 1,
        "under_covered_below": UNDER_COVERED,
        "balance": getattr(results, "balance", None) or {},
        "min_group": min_group,
        "pooled_count": sum(1 for r in rows if r["group_basis"] == "pooled"),
        "warn_z": normalize.WARN_Z,
        "fail_z": normalize.FAIL_Z,
        "config_version_mismatch": len(versions) > 1,
    }
