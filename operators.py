"""Compute quality: scores every LeRobot episode in the target view.

Phase 1 (no video decoding). Runs delegated by default. Scores are batch-relative
by construction (see ``engine.score``), so the unit of work is the whole target
view: scoring a filtered subset gives z-scores relative to that subset.
"""

import copy
import hashlib
import logging
import uuid
from collections import Counter

import fiftyone.operators as foo
import fiftyone.operators.types as types

from . import write
from .engine import dataset_checks
from .engine.groups import DEFAULT_MIN_GROUP, canonical_task
from .engine.metrics import METRICS
from .engine.reader import EpisodeReadError, LeRobotReader
from .engine.score import CONFIG_VERSION, compute_raw, finalize

logger = logging.getLogger(__name__)

PROGRESS_EVERY = 5

FAMILY_LABELS = {
    "motion": "Motion smoothness",
    "time": "Time efficiency",
    "tracking": "Tracking and contact",
    "gripper": "Gripper",
    "consistency": "Consistency",
    "integrity": "Integrity",
    "language": "Language",
    "outliers": "Outliers",
}


def _is_lerobot(dataset):
    return dataset is not None and any(
        s.get("kind") == "lerobot-episode" for s in (dataset.media_sources or [])
    )


def _feature_map_summary(reader):
    """Markdown summary of the inferred feature maps, one line per distinct map."""
    counts = Counter()
    notes = {}
    for sid in reader.source_ids:
        fm = reader.feature_map(sid)
        key = (fm.state_key, fm.action_key)
        counts[key] += 1
        if fm.notes:
            notes[key] = "; ".join(fm.notes)
    lines = []
    for (state, action), n in counts.most_common():
        line = "- **%d source(s)**: state `%s`, action `%s`" % (n, state or "missing", action or "missing")
        if (state, action) in notes:
            line += " (%s)" % notes[(state, action)]
        lines.append(line)
    return "\n".join(lines)


_CHECKS_CACHE = {}
_CHECKS_CACHE_SIZE = 4


def _cached_checks(dataset, reader, samples, overrides):
    """The inferred dataset checks, cached per (dataset state, episodes, column overrides).

    The compute form re-resolves on every click or keystroke, and inferring the
    checks reads a few episodes per source. The key includes the dataset's last
    modification time, so a compute run (which saves the dataset) invalidates it.
    Returns a copy, because overrides are applied to the result in place.
    """
    ids = ",".join(s.id for s in samples)
    key = (
        dataset.name,
        str(getattr(dataset, "last_modified_at", None)),
        hashlib.md5(ids.encode()).hexdigest(),
        overrides.get("state_key"),
        overrides.get("action_key"),
    )
    if key not in _CHECKS_CACHE:
        if len(_CHECKS_CACHE) >= _CHECKS_CACHE_SIZE:
            _CHECKS_CACHE.pop(next(iter(_CHECKS_CACHE)))
        _CHECKS_CACHE[key] = dataset_checks.infer_checks(reader, samples)
    return copy.deepcopy(_CHECKS_CACHE[key])


def _metric_rows():
    return [(name, spec) for name, spec in METRICS.items()]


def _selected_metrics(ctx):
    cfg = ctx.params.get("metrics_cfg") or {}
    return [
        name
        for name, spec in _metric_rows()
        if cfg.get("metric_%s" % name, not spec["opt_in"])
    ]


class ComputeQuality(foo.Operator):
    """Scores the episodes of the target view and writes flat ``lr_*`` fields."""

    @property
    def config(self):
        return foo.OperatorConfig(
            name="lr_compute_quality",
            label="LeRobot curation: compute quality",
            description=(
                "Scores each LeRobot episode for motion smoothness, time "
                "efficiency, integrity and language, then ranks episodes worst-first."
            ),
            dynamic=True,
            execute_as_generator=True,
            allow_immediate_execution=True,
            allow_delegated_execution=True,
            default_choice_to_delegated=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        if not _is_lerobot(ctx.dataset):
            inputs.view(
                "not_lerobot",
                types.Warning(
                    label="This dataset has no LeRobot episodes. Import it with "
                    "fo.types.LeRobotDataset (a multimodal dataset)."
                ),
            )
            return types.Property(inputs)

        view = ctx.target_view()
        if len(view) == 0:
            inputs.view("empty_view", types.Warning(label="The current view has no samples."))
            return types.Property(inputs)

        overrides = ctx.params.get("overrides") or {}
        reader = LeRobotReader(
            ctx.dataset,
            overrides={
                "state_key": overrides.get("state_key"),
                "action_key": overrides.get("action_key"),
            },
        )

        checks = _cached_checks(
            ctx.dataset, reader, list(view.select_fields(["media_reference"])), overrides
        )
        dataset_checks.apply_overrides(
            checks,
            action_semantics=overrides.get("action_semantics"),
            gripper_open_is=overrides.get("gripper_open_is"),
        )

        tabs = types.TabsView()
        tabs.add_choice("CHECKS", label="Dataset checks")
        tabs.add_choice("METRICS", label="Metrics")
        tabs.add_choice("NORMALIZATION", label="Normalization")
        inputs.enum("tab", tabs.values(), default="CHECKS", view=tabs)
        tab = ctx.params.get("tab", "CHECKS")

        if tab == "CHECKS":
            task_keys = [canonical_task(t) for t in view.values("tasks")]
            source_ids = [k.rpartition("/")[0] for k in view.values("media_reference.key")]
            inputs.view("checks_header", types.Header(label="Feature map"))
            inputs.md(
                "The plugin found these state and action columns. Metrics that need "
                "a missing array switch off for those episodes.\n\n"
                + _feature_map_summary(reader),
                name="feature_map_summary",
            )
            inputs.view("assumptions_header", types.Header(label="What the plugin assumed"))
            inputs.md(
                "Inferred from a few episodes per source. Override below if it is wrong. "
                "Metrics that depend on an unknown assumption switch off instead of guessing.\n\n"
                + dataset_checks.summarize(checks, dataset_checks.balance(task_keys, source_ids)),
                name="assumptions_summary",
            )
            section = inputs.obj("overrides", view=types.GridView(orientation="vertical", gap=2))
            section.type.enum(
                "action_semantics",
                ["auto", "joint_positions", "other"],
                default="auto",
                label="Action semantics",
                description=(
                    "Are actions joint positions in the same space as the state? 'other' turns the "
                    "tracking metrics off."
                ),
            )
            section.type.enum(
                "gripper_open_is",
                ["auto", "high", "low"],
                default="auto",
                label="Gripper open direction",
                description="Which value means an open gripper. Needed by missed_grasp_frac and recovery_count.",
            )
            section.type.str(
                "state_key",
                label="State column override (optional)",
                description="Used for every source that has a column with this name.",
            )
            section.type.str(
                "action_key",
                label="Action column override (optional)",
                description="Used for every source that has a column with this name.",
            )
        elif tab == "METRICS":
            section = inputs.obj("metrics_cfg", view=types.GridView(orientation="vertical", gap=2))
            seen = []
            for name, spec in _metric_rows():
                if spec["fn"] is None and spec["family"] != "outliers":
                    continue
                if spec["family"] not in seen:
                    seen.append(spec["family"])
                    section.type.view(
                        "header_%s" % spec["family"],
                        types.Header(label=FAMILY_LABELS.get(spec["family"], spec["family"].title()), divider=True),
                    )
                section.type.bool(
                    "metric_%s" % name,
                    label="%s%s%s"
                    % (
                        name,
                        " (opt-in)" if spec["opt_in"] else "",
                        "" if spec["scored"] else " (not scored)",
                    ),
                    description=spec["description"],
                    default=not spec["opt_in"],
                    view=types.CheckboxView(),
                )
        else:
            section = inputs.obj("normalization", view=types.GridView(orientation="vertical", gap=2))
            section.type.int(
                "min_group",
                label="Minimum episodes per task group",
                description=(
                    "A task with at least this many episodes is normalized on its own. "
                    "Smaller tasks are normalized against the whole view."
                ),
                default=DEFAULT_MIN_GROUP,
                min=2,
            )

        min_group = int((ctx.params.get("normalization") or {}).get("min_group", DEFAULT_MIN_GROUP))
        task_counts = Counter(canonical_task(t) for t in view.values("tasks"))
        pooled = sum(n for key, n in task_counts.items() if not key or n < min_group)
        selected = _selected_metrics(ctx)
        inputs.view(
            "advisory",
            types.Notice(
                label=(
                    "%d episode(s) · %d metric(s) selected · %d episode(s) belong to a task "
                    "group under %d and will be normalized against the whole view "
                    "(low-confidence ranking)."
                )
                % (len(view), len(selected), pooled, min_group)
            ),
        )
        return types.Property(inputs, view=types.View(label="LeRobot curation: compute quality"))

    def execute(self, ctx):
        overrides = ctx.params.get("overrides") or {}
        min_group = int((ctx.params.get("normalization") or {}).get("min_group", DEFAULT_MIN_GROUP))
        metric_names = _selected_metrics(ctx)

        reader = LeRobotReader(
            ctx.dataset,
            overrides={
                "state_key": overrides.get("state_key"),
                "action_key": overrides.get("action_key"),
            },
        )
        view = ctx.target_view()
        samples = list(view.select_fields(["media_reference"]))
        n = len(samples)

        checks = _cached_checks(ctx.dataset, reader, samples, overrides)
        dataset_checks.apply_overrides(
            checks,
            action_semantics=overrides.get("action_semantics"),
            gripper_open_is=overrides.get("gripper_open_is"),
        )
        assumptions = {sid: dataset_checks.assumptions_for(c) for sid, c in checks.items()}

        raws = {}
        read_errors = Counter()
        metric_failures = Counter()
        for i, (sample, episode) in enumerate(reader.read_many(samples)):
            if isinstance(episode, EpisodeReadError):
                read_errors[str(episode)[:80]] += 1
            else:
                raw = compute_raw(episode, sample.id, metric_names, assumptions.get(episode.source_id))
                raws[sample.id] = raw
                for name in raw.failures:
                    metric_failures[name] += 1

            if (i + 1) % PROGRESS_EVERY == 0 or i + 1 == n:
                label = "Scored %d/%d" % (i + 1, n)
                if ctx.delegated:
                    ctx.set_progress(progress=(i + 1) / n, label=label)
                else:
                    yield ctx.trigger("set_progress", {"progress": (i + 1) / n, "label": label})

        if not raws:
            raise RuntimeError(
                "No episode could be read: %s" % (dict(read_errors) or "unknown error")
            )

        results, norm_stats = finalize(raws, metric_names, min_group=min_group)
        run_id = uuid.uuid4().hex[:12]
        fields = write.write_results(ctx.dataset, view, results, run_id=run_id)
        write.set_sidebar_group(ctx.dataset, fields)
        n_tags = write.write_temporal_tags(view.select(list(results), ordered=False), results)
        write.register_run(
            ctx.dataset,
            config={
                "metrics": list(metric_names),
                "min_group": min_group,
                "overrides": {k: v for k, v in overrides.items() if v and v != "auto"},
            },
            norm_stats=norm_stats,
            fields=fields,
            feature_maps={sid: reader.feature_map(sid).as_dict() for sid in reader.source_ids},
            dataset_checks={sid: c.as_dict() for sid, c in checks.items()},
            balance=dataset_checks.balance(
                [r.task_key for r in raws.values()], [r.source_id for r in raws.values()]
            ),
            run_id=run_id,
        )

        verdicts = Counter(r.profiles["policy"]["verdict"] for r in results.values())
        pooled = sum(1 for r in results.values() if r.group_basis == "pooled")
        if not ctx.delegated:
            yield ctx.trigger("reload_dataset")
        yield {
            "scored": len(results),
            "skipped": n - len(results),
            "verdicts": dict(verdicts),
            "pooled": pooled,
            "metric_failures": dict(metric_failures),
            "timeline_tags": n_tags,
            "config_version": CONFIG_VERSION,
        }

    def resolve_output(self, ctx):
        outputs = types.Object()
        r = ctx.results or {}
        verdicts = r.get("verdicts", {})
        lines = [
            "**Scored %d episode(s)**%s." % (r.get("scored", 0), " (%d skipped)" % r["skipped"] if r.get("skipped") else ""),
            "Policy profile: %d fail, %d warn, %d pass."
            % (verdicts.get("fail", 0), verdicts.get("warn", 0), verdicts.get("pass", 0)),
        ]
        if r.get("pooled"):
            lines.append(
                "%d episode(s) were normalized against the whole view (their task group was too small)."
                % r["pooled"]
            )
        if r.get("timeline_tags"):
            lines.append("%d flagged span(s) written as temporal tags on the episode timelines." % r["timeline_tags"])
        if r.get("metric_failures"):
            lines.append("Metric errors: %s" % r["metric_failures"])
        outputs.str("summary", label="Result", view=types.MarkdownView(), default="\n\n".join(lines))
        return types.Property(outputs)
