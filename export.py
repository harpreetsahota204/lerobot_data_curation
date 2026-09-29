"""Exports the kept episodes back to LeRobot format, with a curation manifest.

Builds on FiftyOne's native ``LeRobotDatasetExporter``: the exported dataset is a
self-contained v3 dataset with episodes renumbered contiguously. Nothing in the
source dataset is modified. Episodes tagged ``exclude-candidate`` are left out.
"""

import datetime
import json
import os
import shutil

import fiftyone.types as fot

from .engine.score import CONFIG_VERSION
from .write import RUN_KEY

EXCLUDE_TAG = "exclude-candidate"
MANIFEST_NAME = "curation_manifest.json"


def split_kept(view):
    """``(kept_view, excluded_ids)``: the view without `exclude-candidate`, and what was left out."""
    excluded = view.match_tags(EXCLUDE_TAG)
    return view.match_tags(EXCLUDE_TAG, bool=False), excluded.values("id")


def build_manifest(dataset, view, kept, excluded_ids, export_dir):
    """Everything needed to audit or reproduce this curation run."""
    config = {}
    try:
        info = dataset.get_run_info(RUN_KEY)
        config = json.loads(json.dumps(info.config.to_dict(), default=str))
    except Exception:  # noqa: BLE001 - a manifest without the run config is still useful
        pass

    fields = [
        "id", "episode_index", "task", "tasks", "tags", "lr_group", "lr_group_basis", "lr_config_version",
        "lr_score_policy", "lr_verdict_policy", "lr_driver_policy",
        "lr_score_vla", "lr_verdict_vla", "lr_driver_vla",
        "lr_integrity_verdict", "lr_language_verdict",
    ]
    schema = dataset.get_field_schema()
    fields = [f for f in fields if f in schema or f in ("id",)]
    columns = dict(zip(fields, view.values(fields)))
    kept_ids = set(kept.values("id"))

    episodes = []
    for i, sid in enumerate(columns["id"]):
        episodes.append(
            {
                **{f: columns[f][i] for f in fields if f != "id"},
                "id": sid,
                "kept": sid in kept_ids,
            }
        )

    return {
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "plugin": "lerobot-data-curation",
        "source_dataset": dataset.name,
        "export_dir": export_dir,
        "config_version": CONFIG_VERSION,
        "run_config": config,
        "counts": {"in_view": len(view), "kept": len(kept), "excluded": len(excluded_ids)},
        "excluded_ids": excluded_ids,
        "episodes": episodes,
    }


def _source_of(dataset, view):
    """``{sample_id: (source_id, source_dir)}`` for every episode in the view."""
    dirs = {m["id"]: m.get("dir") or m["id"][-6:] for m in (dataset.media_sources or [])}
    out = {}
    for sid, key in zip(view.values("id"), view.values("media_reference.key")):
        source_id = (key or "").rpartition("/")[0]
        out[sid] = (source_id, dirs.get(source_id, source_id[-6:]))
    return out


def export_kept(dataset, view, export_dir):
    """Exports the kept episodes and writes the manifest. Returns a summary dict.

    FiftyOne's exporter refuses to mix episodes from different sources, so a view
    spanning several sources is exported as one LeRobot dataset per source, in
    subfolders named after the source. A single-source view is exported directly
    into `export_dir`.

    A source that LeRobot cannot write (for example one with a string-typed feature)
    is skipped and reported under ``failed`` and in the manifest, so the rest is
    still exported.

    Raises:
        ValueError: if the destination already has files, nothing is left to
            export, or no source could be written
    """
    export_dir = os.path.abspath(os.path.expanduser(export_dir))
    if os.path.exists(export_dir) and os.listdir(export_dir):
        raise ValueError("Destination %s already exists and is not empty" % export_dir)

    kept, excluded_ids = split_kept(view)
    if len(kept) == 0:
        raise ValueError("Nothing to export: every episode in the view is tagged %s" % EXCLUDE_TAG)

    kept_ids = kept.values("id")
    source_of = _source_of(dataset, kept)
    by_source = {}
    for sid in kept_ids:
        by_source.setdefault(source_of[sid], []).append(sid)

    exports, failed = [], []
    used = set()
    single = len(by_source) == 1
    for (source_id, source_dir), ids in by_source.items():
        name = "." if single else source_dir
        while name in used and not single:  # two sources can share a directory name
            name = "%s_%s" % (source_dir, source_id[-4:])
        used.add(name)
        target = export_dir if single else os.path.join(export_dir, name)
        try:
            kept.select(ids).export(export_dir=target, dataset_type=fot.LeRobotDataset, export_media=True)
            exports.append({"path": name, "source": source_dir, "episodes": len(ids)})
        except Exception as e:  # noqa: BLE001 - one unwritable source must not lose the others
            if not single:
                shutil.rmtree(target, ignore_errors=True)
            failed.append({"source": source_dir, "episodes": len(ids), "error": "%s: %s" % (type(e).__name__, str(e)[:200])})

    if not exports:
        raise ValueError(
            "LeRobot could not write any of the %d source(s). First error: %s"
            % (len(failed), failed[0]["error"])
        )

    manifest = build_manifest(dataset, view, kept, excluded_ids, export_dir)
    manifest["exports"] = exports
    manifest["failed_sources"] = failed
    manifest["counts"]["exported"] = sum(e["episodes"] for e in exports)
    os.makedirs(export_dir, exist_ok=True)
    with open(os.path.join(export_dir, MANIFEST_NAME), "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return {
        "export_dir": export_dir,
        "kept": sum(e["episodes"] for e in exports),
        "excluded": len(excluded_ids),
        "datasets": len(exports),
        "failed": failed,
        "manifest": MANIFEST_NAME,
    }
