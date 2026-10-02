"""Compute quality: scores every LeRobot episode in the target view.

Runs the metrics you choose. The camera metrics are opt-in, on their own tab of the
form, because they decode video. Runs delegated by default. Scores are batch-relative
by construction (see ``engine.score``), so the unit of work is the whole target
view: scoring a filtered subset gives z-scores relative to that subset.
"""

import json
import logging
import uuid
from collections import Counter

import numpy as np

import fiftyone.operators as foo
import fiftyone.operators.types as types

from . import write
from .engine import dataset_checks
from .engine import picks as pk
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
    "camera": "Camera (decodes video, about 1 s per episode)",
    "integrity": "Integrity",
    "language": "Language",
    "outliers": "Outliers",
}
FAMILY_HELP = {
    "motion": "How fluid the commanded motion is. Jerky or jittery demonstrations teach a policy to jitter.",
    "time": "Hesitation and wasted time: idle starts and ends, long pauses, and episodes much longer or more "
    "roundabout than their peers.",
    "tracking": "How closely the robot followed its commands, and whether it slammed into joint limits. Large "
    "errors suggest a collision, unexpected contact or a struggling motor.",
    "gripper": "Grasp quality: fumbled grasps, regrasps after a miss, and a gripper that flickers open and shut.",
    "consistency": "Whether this demonstration does something different from its peers in a similar situation, "
    "and whether it alone stretches the dataset's normalization stats.",
    "integrity": "Broken recordings: dropped frames, bad timestamps, NaNs, mismatched video. Flags a broken "
    "episode rather than ranking it, and never enters the score.",
    "language": "Whether the task instruction a VLA conditions on is missing or too generic to be useful.",
    "outliers": "Episodes unlike the rest of their group, for a second look. Unusual does not mean bad, so "
    "these are never scored.",
}


def _is_lerobot(dataset):
    return dataset is not None and any(
        s.get("kind") == "lerobot-episode" for s in (dataset.media_sources or [])
    )


def _last_run_config(dataset):
    """``{"metrics": [...], "min_group": int, "picks": {...}}`` from the last compute run, or {}."""
    if not dataset.has_run(write.RUN_KEY):
        return {}
    try:
        cfg = dataset.get_run_info(write.RUN_KEY).config
        return {
            "metrics": list(getattr(cfg, "metrics", None) or []),
            "min_group": getattr(cfg, "min_group", None),
            "picks": pk.clean_picks(_plain(getattr(cfg, "picks", None) or {})),
        }
    except Exception:  # noqa: BLE001 - a damaged run record must not block a new run
        return {}


def _plain(value):
    """A run-record value as plain dicts and lists."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _default_on(name, last_metrics, has_cameras=True):
    """Whether a metric's checkbox starts ticked.

    Opt-in metrics start off. A camera metric starts ticked once a camera is picked,
    or stays as the last run left it: re-running Compute quality would otherwise clear
    camera scores an earlier run wrote, because every run replaces the fields of the
    one before it.
    """
    spec = METRICS[name]
    if spec["family"] == "camera":
        if not has_cameras:
            return False
        had_camera_metrics = any(METRICS[m]["family"] == "camera" for m in (last_metrics or []) if m in METRICS)
        return name in (last_metrics or []) if had_camera_metrics else True
    return not spec["opt_in"]


SECONDS_PER_CAMERA = 0.4  # measured: about 0.4 s per camera per episode with every camera metric on


def _duration(seconds):
    if seconds < 90:
        return "%d s" % max(1, round(seconds))
    return "%d min" % round(seconds / 60.0)


def _decode_time(n_cameras, n_episodes):
    return _duration(n_cameras * n_episodes * SECONDS_PER_CAMERA)


def camera_metric_names():
    return [name for name, spec in METRICS.items() if spec["family"] == "camera"]


# -- the user's picks ---------------------------------------------------------------

UNSET = "unset"

ACTION_MEANINGS = [
    ("joint_positions", "Yes, absolute joint positions"),
    ("other", "No (deltas, end-effector poses, ...)"),
]
OPEN_DIRECTIONS = [("high", "High value = open"), ("low", "Low value = open")]


def _radio(section, name, options, default, label, description=None):
    """A horizontal radio with nothing selected until the user picks, unless `default` is given."""
    radio = types.RadioGroup(orientation="horizontal")
    for value, text in options:
        radio.add_choice(value, label=text)
    section.type.enum(name, radio.values(), default=default, view=radio, label=label, description=description)


def _dropdown(section, name, options, default, label, description=None):
    """A dropdown whose first choice is "Not set". `options` are ``(value, label)``; `default` a value or None."""
    dd = types.Dropdown()
    dd.add_choice(UNSET, label="Not set")
    for value, text in options:
        dd.add_choice(value, label=text)
    section.type.enum(name, dd.values(), default=default or UNSET, view=dd, label=label, description=description)


def _pick_value(cfg, name, last, valid=None):
    """A dropdown's value: what the form holds if the user touched it, else what the last run used."""
    raw = cfg[name] if name in cfg else (last or {}).get(name)
    if raw in (None, "", UNSET) or (valid is not None and raw not in valid):
        return None
    return raw


def _first_source(reader):
    sid = reader.source_ids[0]
    return sid, reader.info(sid)


def _action_layout(reader, sid, info, action_key):
    """How the picked action array can be described: by joint names, or by dimension numbers."""
    if not action_key:
        return None
    dim = int(np.prod(info["features"][action_key]["shape"]))
    names = reader.feature_names(sid, action_key)
    usable = pk.names_usable(names, action_key)
    return {
        "dim": dim,
        "names": names,
        "usable": usable,
        "bounds": reader.feature_bounds(sid, action_key),
    }


def _size_mismatch(facts, picks):
    """A sentence when the picked state and action differ in size (so the action cannot be joint positions), else None."""
    sizes = dict(facts["arrays"])
    state, action = picks["state_key"], picks["action_key"]
    if state and action and sizes.get(state) != sizes.get(action):
        return "The action (%d) and the state (%d) differ in size, so the action is not joint positions in the state's space." % (
            sizes[action],
            sizes[state],
        )
    return None


def _shown_fields(layout_choice, hands):
    return [f for f in pk.joint_fields(layout_choice) if hands or f["role"] != "hand"]


def _joint_dims(ctx, layout_choice, hands, layout, last_picks, action_key):
    """``({field key: [dims]}, errors)`` from the form's joint fields, falling back to the last run's groups."""
    gcfg = ctx.params.get("groups_cfg") or {}
    same = (last_picks or {}).get("action_key") == action_key and (last_picks or {}).get("layout") == layout_choice
    fallback = pk.dims_by_field(layout_choice, (last_picks or {}).get("groups") if same else [])
    out, errors, used = {}, [], {}
    for f in _shown_fields(layout_choice, hands):
        dims = fallback[f["key"]]
        if f["key"] in gcfg:
            if layout["usable"]:
                dims = pk.dims_from_names(gcfg[f["key"]], layout["names"])
            else:
                dims, errs = pk.parse_dim_list(gcfg[f["key"]], layout["dim"])
                errors += ["%s: %s" % (f["label"], e) for e in errs]
        if not layout["usable"]:
            clash = sorted(d for d in dims if d in used)
            if clash:
                errors.append("%s: dimension %s is already in %s." % (f["label"], pk.describe_dims(clash), used[clash[0]]))
        for d in dims:
            used.setdefault(d, f["label"])
        out[f["key"]] = dims
    return out, errors


def _picked_cameras(ctx, last_picks):
    ccfg = ctx.params.get("camera_cfg") or {}
    return list(ccfg["cameras"] if "cameras" in ccfg else (last_picks or {}).get("cameras") or [])


def _pick_array(cfg, name, last, arrays):
    """A state or action pick: the form, else the last run, else the LeRobot standard name if the dataset has it."""
    if name in cfg or (last or {}).get(name):
        return _pick_value(cfg, name, last, arrays)
    standard = pk.STANDARD_ARRAYS[name]
    return standard if standard in arrays else None


def _wants_hands(rcfg, last):
    """Whether the hand fields are shown: the checkbox, else whether the last run picked a hand."""
    if "hands" in rcfg:
        return bool(rcfg["hands"])
    return any(g.get("role") == "hand" for g in (last or {}).get("groups") or [])


def _collect_picks(ctx, reader):
    """``(picks, errors, layout, field_dims)`` from the form, defaulting to the last run's picks.

    Nothing is inferred from the data. `layout` describes the picked action array (None
    until one is picked) and `field_dims` is ``{joint field key: [dims]}`` for the
    fields the form shows.
    """
    sid, info = _first_source(reader)
    facts = dataset_checks.declared_facts(info)
    arrays = [k for k, _ in facts["arrays"]]
    last = _last_run_config(ctx.dataset).get("picks") or {}
    rcfg = ctx.params.get("robot_cfg") or {}
    cfg = ctx.params.get("data_cfg") or {}
    gcfg = ctx.params.get("groups_cfg") or {}

    picks = pk.empty_picks()
    picks["layout"] = _pick_value(rcfg, "layout", last, [v for v, _ in pk.LAYOUTS]) or pk.DEFAULT_LAYOUT
    picks["state_key"] = _pick_array(cfg, "state_key", last, arrays)
    picks["action_key"] = _pick_array(cfg, "action_key", last, arrays)
    if _size_mismatch(facts, picks):
        picks["action_semantics"] = "other"
    elif picks["state_key"] and picks["action_key"]:
        picks["action_semantics"] = _pick_value(cfg, "action_semantics", last, [v for v, _ in ACTION_MEANINGS])

    layout = _action_layout(reader, sid, info, picks["action_key"])
    errors, field_dims = [], {}
    if layout is not None:
        hands = _wants_hands(rcfg, last)
        field_dims, errors = _joint_dims(ctx, picks["layout"], hands, layout, last, picks["action_key"])
        picks["groups"] = pk.assemble_groups(picks["layout"], field_dims)
    if any(g["role"] == "gripper" for g in picks["groups"]):
        picks["gripper_open_is"] = _pick_value(gcfg, "gripper_open_is", last, [v for v, _ in OPEN_DIRECTIONS])
    picks["cameras"] = [c for c in _picked_cameras(ctx, last) if c in facts["cameras"]]
    return picks, errors, layout, field_dims


def _facts_line(reader):
    """One line of what the dataset declares: robot, fps, arrays and cameras."""
    sid, info = _first_source(reader)
    facts = dataset_checks.declared_facts(info)
    parts = ["**%s**" % (facts["robot_type"] or "robot not stated"), "%s fps" % (facts["fps"] or "?")]
    parts.append(", ".join("`%s` (%d)" % a for a in facts["arrays"]) or "no arrays")
    cameras = "%d video camera(s)" % len(facts["cameras"])
    if facts["image_cameras"]:
        cameras += ", %d stored as images (not scorable)" % len(facts["image_cameras"])
    parts.append(cameras)
    line = " · ".join(parts)
    if len(reader.source_ids) > 1:
        line += "\n\n%d sources: choices are listed from the first and apply to every source with the same arrays." % len(
            reader.source_ids
        )
    return line


def _readiness_markdown(picks, camera_metrics_selected):
    ready, off = pk.readiness(picks, camera_metrics_selected)
    lines = ["**Will be scored:** " + ", ".join(ready)]
    if off:
        lines.append("\n**Off:**")
        lines += ["- %s: %s" % (family, need) for family, need in off]
    return "\n".join(lines)


def _joint_label(layout, d):
    """A joint's choice label: its name and, when the dataset has stats, its observed range."""
    label = str(layout["names"][d])
    bounds = layout.get("bounds")
    if bounds is not None and d < len(bounds[0]):
        label += " · %.3g to %.3g" % (bounds[0][d], bounds[1][d])
    return label


def _joint_field(gsec, f, layout, field_dims):
    """One joint field: chips for a named array, a text box of dimension numbers for an unnamed one."""
    dims = field_dims.get(f["key"], [])
    if not layout["usable"]:
        gsec.type.str(
            f["key"],
            default=pk.describe_dims(dims) if dims else "",
            label=f["label"],
            description="%s Dimension numbers like '0-6' or '0-2, 5'." % f["help"],
        )
        return
    # A joint picked in any field leaves every field's list, so no joint can be in two fields.
    taken = {d for ds in field_dims.values() for d in ds}
    choices = types.AutocompleteView()
    for d in range(layout["dim"]):
        if d not in taken:
            choices.add_choice(str(layout["names"][d]), label=_joint_label(layout, d))
    gsec.type.list(
        f["key"], types.String(), default=pk.names_for_dims(dims, layout["names"]), view=choices,
        label=f["label"], description=f["help"],
    )


def _card_sx(subtitle):
    """MUI ``sx`` that turns the App's collapsible ObjectView into a card: title and subtitle left, chevron right.

    The App renders the section as ``container > div > [div[role=button] (chevron, title), div (content)]``
    and has no subtitle slot, so the subtitle is drawn as the header's ``::after`` content.
    """
    header = "& > div > [role='button']"
    content = header + " ~ div"
    return {
        "border": "1px solid rgba(255, 255, 255, 0.08)",
        "borderRadius": "8px",
        "backgroundColor": "#1f1f1f",
        "overflow": "hidden",
        header: {
            "display": "grid !important",  # the App sets an inline display: flex on the header
            "gridTemplateColumns": "1fr auto",
            "columnGap": "12px",
            "rowGap": "2px",
            "padding": "12px 16px",
            "cursor": "pointer",
        },
        header + ":hover": {"backgroundColor": "rgba(255, 255, 255, 0.04)"},
        header + " > :first-child": {"gridColumn": "2", "gridRow": "1 / span 2", "alignSelf": "center"},
        header + " > :last-child": {"gridColumn": "1", "gridRow": "1", "fontWeight": 500},
        header + "::after": {
            "content": json.dumps(subtitle, ensure_ascii=False),
            "gridColumn": "1",
            "gridRow": "2",
            "fontSize": "0.8rem",
            "color": "rgba(255, 255, 255, 0.55)",
        },
        # The App animates max-height 0 -> 2000px, so closing stalls and then snaps. Animate the real
        # height instead (grid rows 0fr -> 1fr). Vertical padding goes on the inner items so it is
        # clipped while closed; padding on the content itself would show while collapsed.
        content: {
            "display": "grid",
            "gridTemplateRows": "0fr",
            "maxHeight": "none !important",
            "transition": "grid-template-rows 200ms ease-out !important",
            "padding": "0 16px",
        },
        content + "[class*='contentOpen']": {"gridTemplateRows": "1fr"},
        content + " > *": {"minHeight": 0, "overflow": "hidden"},
        content + " > * > :last-child": {"paddingBottom": "12px"},
    }


def _metric_checkbox(section, name, spec, last):
    section.type.bool(
        "metric_%s" % name,
        label="%s%s%s" % (name, " (opt-in)" if spec["opt_in"] else "", "" if spec["scored"] else " (not scored)"),
        description=spec["description"],
        default=_default_on(name, last.get("metrics")),
        view=types.CheckboxView(),
    )


def _selected_metrics(ctx):
    """The metrics ticked in the form. Camera metrics are ticked on their own tab.

    The Metrics tab nests each family's checkboxes under its family key in ``metrics_cfg``.
    """
    cfg = ctx.params.get("metrics_cfg") or {}
    camera = ctx.params.get("camera_cfg") or {}
    last = _last_run_config(ctx.dataset)
    has_cameras = bool(_picked_cameras(ctx, last.get("picks")))
    out = []
    for name, spec in METRICS.items():
        if spec["family"] == "camera":
            ticked = camera.get("metric_%s" % name, _default_on(name, last.get("metrics"), has_cameras))
        else:
            fam = cfg.get(spec["family"]) or {}
            ticked = fam.get("metric_%s" % name, _default_on(name, last.get("metrics")))
        if ticked:
            out.append(name)
    return out


class ComputeQuality(foo.Operator):
    """Scores the episodes of the target view and writes flat ``lr_*`` fields."""

    @property
    def config(self):
        return foo.OperatorConfig(
            name="lr_compute_quality",
            label="LeRobot curation: compute quality",
            description=(
                "Scores each LeRobot episode for motion smoothness, time efficiency, "
                "integrity and language (and camera quality, if you pick cameras), then "
                "ranks episodes worst-first. You say which arrays and joints to use: nothing is guessed."
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

        reader = LeRobotReader(ctx.dataset)  # for what the dataset declares; nothing is inferred from it
        picks, errors, layout, field_dims = _collect_picks(ctx, reader)
        sid, info = _first_source(reader)
        facts = dataset_checks.declared_facts(info)
        last = _last_run_config(ctx.dataset)
        hands = _wants_hands(ctx.params.get("robot_cfg") or {}, last.get("picks"))

        tabs = types.TabsView()
        tabs.add_choice("DATA", label="Data")
        tabs.add_choice("METRICS", label="Metrics")
        tabs.add_choice("CAMERA", label="Camera")
        tabs.add_choice("NORMALIZATION", label="Normalization")
        inputs.enum("tab", tabs.values(), default="DATA", view=tabs)
        tab = ctx.params.get("tab", "DATA")

        if tab == "DATA":
            inputs.md(_facts_line(reader), name="facts")

            rsec = inputs.obj("robot_cfg", view=types.GridView(orientation="vertical", gap=2))
            _radio(rsec, "layout", pk.LAYOUTS, picks["layout"], "Arms", "Select how many arms the robot has.")
            rsec.type.bool(
                "hands",
                default=hands,
                label="Has multi-joint hands",
                description="Adds a Hand field per arm. Leave off for an ordinary gripper.",
                view=types.CheckboxView(),
            )

            inputs.md("---", name="arrays_rule")
            inputs.view(
                "arrays_header",
                types.Header(
                    label="Arrays",
                    description=(
                        "Each episode records rows of numbers over time. The state is where the joints actually "
                        "were, read from the robot's sensors. The action is what the operator or policy commanded, "
                        "which is what a VLA learns to output. Smoothness, idle and gripper metrics judge the action; "
                        "joint limits read the state; tracking compares the two."
                    ),
                ),
            )
            section = inputs.obj("data_cfg", view=types.GridView(orientation="vertical", gap=2))
            array_options = [(k, "%s (%d)" % (k, d)) for k, d in facts["arrays"]]
            _dropdown(
                section, "state_key", array_options, picks["state_key"], "State array",
                "The robot's measured positions. Needed for tracking, acceleration spikes, joint limits and frozen-feed detection.",
            )
            _dropdown(
                section, "action_key", array_options, picks["action_key"], "Action array",
                "What was commanded. Every smoothness, idle, pause and gripper metric reads it.",
            )
            mismatch = _size_mismatch(facts, picks)
            if mismatch:
                inputs.md(mismatch, name="action_note")
            elif picks["state_key"] and picks["action_key"]:
                _radio(
                    section, "action_semantics", ACTION_MEANINGS, picks["action_semantics"],
                    "Does the action command the same joints as the state, as absolute positions?",
                    "Yes turns on tracking, acceleration spikes and joint limits.",
                )

            inputs.md("---", name="groups_rule")
            inputs.view("groups_header", types.Header(label="Joints", description="Joints you leave out are ignored."))
            if layout is None:
                inputs.view("groups_hint", types.Notice(label="Pick an action array above to describe its joints."))
            else:
                if not layout["usable"]:
                    inputs.view(
                        "names_notice",
                        types.Notice(
                            label="This array has no usable joint names, so give dimension numbers instead. "
                            "It has %d dimensions (0 to %d)." % (layout["dim"], layout["dim"] - 1)
                        ),
                    )
                gsec = inputs.obj("groups_cfg", view=types.GridView(orientation="vertical", gap=2))
                fields = _shown_fields(picks["layout"], hands)
                has_gripper = any(g["role"] == "gripper" for g in picks["groups"])
                last_gripper = max((i for i, f in enumerate(fields) if f["role"] == "gripper"), default=None)
                for i, f in enumerate(fields):
                    _joint_field(gsec, f, layout, field_dims)
                    if has_gripper and i == last_gripper:
                        _radio(
                            gsec, "gripper_open_is", OPEN_DIRECTIONS, picks["gripper_open_is"], "Gripper open direction",
                            "Needed for regrasp recovery, missed grasps and gripper phases.",
                        )
                if errors:
                    inputs.view("joint_errors", types.Warning(label=" ".join(errors)))
        elif tab == "METRICS":
            section = inputs.obj("metrics_cfg", view=types.GridView(orientation="vertical", gap=2))
            ticked = set(_selected_metrics(ctx))
            by_family = {}
            for name, spec in METRICS.items():
                if spec["fn"] is None and spec["family"] != "outliers":
                    continue
                if spec["family"] == "camera":
                    continue  # on the Camera tab
                by_family.setdefault(spec["family"], []).append((name, spec))
            for family, rows in by_family.items():
                n_on = sum(name in ticked for name, _ in rows)
                fam = section.type.obj(
                    family,
                    view=types.ObjectView(
                        collapsible=True,
                        default_expanded=False,
                        label="%s · %d of %d selected" % (FAMILY_LABELS.get(family, family.title()), n_on, len(rows)),
                        componentsProps={"container": {"sx": _card_sx(FAMILY_HELP.get(family, ""))}},
                    ),
                )
                for name, spec in rows:
                    _metric_checkbox(fam, name, spec, last)
        elif tab == "CAMERA":
            section = inputs.obj("camera_cfg", view=types.GridView(orientation="vertical", gap=2))
            choices = types.AutocompleteView()
            for cam in facts["cameras"]:
                if cam not in picks["cameras"]:
                    size = facts["camera_sizes"].get(cam)
                    short = cam.split(".")[-1]
                    choices.add_choice(cam, label="%s · %d×%d" % (short, size[1], size[0]) if size else short)
            section.type.list(
                "cameras", types.String(), default=picks["cameras"], view=choices,
                label="Cameras to score",
                description="Pick the cameras the camera metrics should run on. Nothing is selected for you.",
            )
            n_cam = len(picks["cameras"])
            inputs.md(
                "Decoding takes about %s for %d camera(s) over %d episode(s)." % (_decode_time(n_cam, len(view)), n_cam, len(view))
                if n_cam
                else "No camera is picked, so the camera metrics will not run.",
                name="camera_estimate",
            )
            for name in camera_metric_names():
                section.type.bool(
                    "metric_%s" % name,
                    label=name,
                    description=METRICS[name]["description"],
                    default=_default_on(name, last.get("metrics"), n_cam > 0),
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
        has_camera = any(METRICS[m]["family"] == "camera" for m in selected)
        inputs.md(_readiness_markdown(picks, has_camera), name="readiness")
        inputs.view(
            "advisory",
            types.Notice(
                label=(
                    "%d episode(s) · %d metric(s) selected%s · %d episode(s) belong to a task "
                    "group under %d and will be normalized against the whole view "
                    "(low-confidence ranking)."
                )
                % (
                    len(view),
                    len(selected),
                    " (camera metrics: about %s)" % _decode_time(len(picks["cameras"]), len(view))
                    if has_camera and picks["cameras"]
                    else "",
                    pooled,
                    min_group,
                )
            ),
        )
        return types.Property(inputs, view=types.View(label="LeRobot curation: compute quality"))

    def execute(self, ctx):
        min_group = int((ctx.params.get("normalization") or {}).get("min_group", DEFAULT_MIN_GROUP))
        metric_names = _selected_metrics(ctx)

        picks, errors, _, _ = _collect_picks(ctx, LeRobotReader(ctx.dataset))
        if errors:
            raise ValueError("Fix the joint dimensions first: %s" % " ".join(errors))
        reader = LeRobotReader(ctx.dataset, picks=picks)
        view = ctx.target_view()
        samples = list(view.select_fields(["media_reference"]))
        n = len(samples)
        assumptions = {sid: pk.assumptions_for(picks) for sid in reader.source_ids}
        has_camera = any(METRICS[m]["family"] == "camera" for m in metric_names)
        switched_off = pk.explain_off(picks, camera_metrics_selected=has_camera)

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
                "picks": picks,
            },
            norm_stats=norm_stats,
            fields=fields,
            feature_maps={sid: reader.feature_map(sid).as_dict() for sid in reader.source_ids},
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
            "switched_off": switched_off,
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
        if r.get("switched_off"):
            lines.append("**Switched off because nothing is picked for them:**\n" + "\n".join("- %s" % o for o in r["switched_off"]))
        if r.get("metric_failures"):
            lines.append("Metric errors: %s" % r["metric_failures"])
        outputs.str("summary", label="Result", view=types.MarkdownView(), default="\n\n".join(lines))
        return types.Property(outputs)

