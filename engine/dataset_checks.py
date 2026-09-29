"""Dataset-level checks: infer an assumption, show it, let the user confirm or override.

Each check inspects a few episodes per source and infers a fact that metrics
depend on. Metrics that rely on a wrong or unknown assumption switch off instead
of running on bad input. See METRICS.md, "Dataset-level checks".

The checks:

- ``feature_map``: which columns are state, action and cameras (see reader.py)
- ``action_semantics``: whether action is joint positions shared with state
- ``joint_units``: radians, degrees or normalized (informational)
- ``gripper_convention``: which gripper dimensions exist and which direction is open
- ``camera_roles``: wrist, top, side or ego, from feature keys
- ``balance``: episodes per canonical task and per source (reported, not scored)
"""

import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .reader import EpisodeReadError, is_named
from .signals import arm_dims, gripper_signals, named_action_groups, shares_joint_space

EPISODES_PER_SOURCE = 3
UNDER_COVERED = 5  # tasks with fewer episodes than this are reported as under-covered
DOMINANT_SHARE = 0.5


@dataclass
class SourceChecks:
    """What was inferred for one source."""

    action_semantics: str = "unknown"  # joint_positions | other | unknown
    joint_units: str = "unknown"  # radians | degrees | normalized | unknown
    gripper_dims: list = field(default_factory=list)  # names of gripper action dimensions
    gripper_open_is: "str | None" = None  # "high" | "low" | None
    gripper_confident: bool = False
    camera_roles: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def as_dict(self):
        return {
            "action_semantics": self.action_semantics,
            "joint_units": self.joint_units,
            "gripper_dims": list(self.gripper_dims),
            "gripper_open_is": self.gripper_open_is,
            "gripper_confident": self.gripper_confident,
            "camera_roles": dict(self.camera_roles),
            "notes": list(self.notes),
        }


def camera_role(key):
    """Wrist, top, side or ego, guessed from a camera feature key."""
    k = key.lower()
    for role, pattern in (
        ("wrist", r"wrist|hand|gripper|eye_in_hand"),
        ("ego", r"ego|head|first_person|chest"),
        ("top", r"top|overhead|front|high|third|scene|workspace|laptop"),
        ("side", r"side|left|right"),
    ):
        if re.search(pattern, k):
            return role
    return "unknown"


def infer_joint_units(values):
    """Radians, degrees or normalized from arm action values (informational)."""
    if values is None or values.size == 0:
        return "unknown"
    peak = float(np.percentile(np.abs(values), 99))
    if peak <= 2 * np.pi + 0.5:
        return "radians"
    if peak <= 100.5 and np.min(values) < -50:
        return "normalized"
    if peak <= 360.5:
        return "degrees"
    return "unknown"


def infer_gripper_open_is(unit_signals):
    """Which direction is open, from how each gripper behaves at the start and end.

    Manipulation episodes normally start and end with the gripper open. So the
    value near the start is "open". Confident when start and end agree.
    """
    votes, agree = [], []
    for unit in unit_signals.values():
        n = max(1, len(unit) // 20)
        start, end = float(np.median(unit[:n])), float(np.median(unit[-n:]))
        votes.append("high" if start >= 0.5 else "low")
        agree.append((start >= 0.5) == (end >= 0.5))
    if not votes:
        return None, False
    open_is = Counter(votes).most_common(1)[0][0]
    return open_is, all(agree) and len(set(votes)) == 1


def infer_source_checks(episodes):
    """Infers the checks for one source from a few of its episodes."""
    out = SourceChecks()
    if not episodes:
        return out
    ep = episodes[0]
    out.camera_roles = {cam: camera_role(cam) for cam in (ep.feature_map.camera_keys if ep.feature_map else [])}

    if ep.action is None:
        out.notes.append("no action column")
        return out

    dims = arm_dims(ep)
    out.joint_units = infer_joint_units(ep.action[:, dims] if dims else ep.action)

    votes = [shares_joint_space(e) for e in episodes if e.state is not None and e.action is not None]
    if votes:
        out.action_semantics = "joint_positions" if sum(votes) > len(votes) / 2 else "other"
    else:
        out.notes.append("no state column: cannot tell whether action is joint positions")

    groups = named_action_groups(ep)
    out.gripper_dims = [ep.action_names[i] for name, idx in groups.items() if name.startswith("gripper:") for i in idx]
    if out.gripper_dims:
        signals = {}
        for e in episodes:
            for side, unit in gripper_signals(e).items():
                signals.setdefault(side, []).append(unit)
        first = {side: units[0] for side, units in signals.items()}
        out.gripper_open_is, out.gripper_confident = infer_gripper_open_is(first)
        if out.gripper_open_is is None:
            out.notes.append("gripper never moves")
    elif not is_named(ep.action_names, ep.feature_map.action_key or "action"):
        out.notes.append("joint names missing: gripper cannot be identified")
    return out


def infer_checks(reader, samples, per_source=EPISODES_PER_SOURCE):
    """Runs the per-source checks. Returns ``{source_id: SourceChecks}``."""
    by_source = {}
    for sample in samples:
        source_id = sample.media_reference.key.rpartition("/")[0]
        bucket = by_source.setdefault(source_id, [])
        if len(bucket) < per_source:
            bucket.append(sample)

    picked = [s for bucket in by_source.values() for s in bucket]
    episodes = {}
    for sample, ep in reader.read_many(picked):
        if isinstance(ep, EpisodeReadError):
            continue
        episodes.setdefault(ep.source_id, []).append(ep)
    return {sid: infer_source_checks(eps) for sid, eps in episodes.items()}


def apply_overrides(checks, action_semantics=None, gripper_open_is=None):
    """Applies the user's overrides to every source. ``None`` or ``"auto"`` keeps the inferred value."""
    for c in checks.values():
        if action_semantics in ("joint_positions", "other"):
            c.action_semantics = action_semantics
        if gripper_open_is in ("high", "low") and c.gripper_dims:
            c.gripper_open_is = gripper_open_is
            c.gripper_confident = True
    return checks


def assumptions_for(source_checks):
    """The assumptions dict metrics read from ``episode.assumptions``."""
    if source_checks is None:
        return {}
    out = {}
    if source_checks.action_semantics != "unknown":
        out["action_semantics"] = source_checks.action_semantics
    if source_checks.gripper_open_is:
        out["gripper_open_is"] = source_checks.gripper_open_is
    return out


def balance(task_keys, source_ids):
    """Episodes per canonical task and per source, with coverage warnings."""
    tasks = Counter(task_keys)
    sources = Counter(source_ids)
    n = max(1, len(task_keys))
    return {
        "tasks": dict(tasks.most_common()),
        "sources": dict(sources.most_common()),
        "under_covered": sorted(t for t, c in tasks.items() if c < UNDER_COVERED),
        "dominant_sources": sorted(s for s, c in sources.items() if c / n > DOMINANT_SHARE),
    }


def summarize(checks, bal):
    """Markdown for the compute form: one line per check, grouped by distinct value."""

    def grouped(title, values):
        counts = Counter(values)
        parts = ", ".join("%d %s" % (n, v) for v, n in counts.most_common())
        return "- **%s:** %s" % (title, parts or "n/a")

    lines = [
        grouped("Action semantics", [c.action_semantics.replace("_", " ") for c in checks.values()]),
        grouped("Joint units (informational)", [c.joint_units for c in checks.values()]),
        grouped(
            "Gripper",
            [
                ("open is %s%s" % (c.gripper_open_is, "" if c.gripper_confident else " (unsure)"))
                if c.gripper_dims
                else "no named gripper"
                for c in checks.values()
            ],
        ),
        grouped(
            "Camera roles",
            [", ".join(sorted(set(c.camera_roles.values()))) or "none" for c in checks.values()],
        ),
        "- **Balance:** %d task(s), %d source(s); %d under-covered task(s) (fewer than %d episodes)%s"
        % (
            len(bal["tasks"]),
            len(bal["sources"]),
            len(bal["under_covered"]),
            UNDER_COVERED,
            "; dominant source(s): %d" % len(bal["dominant_sources"]) if bal["dominant_sources"] else "",
        ),
    ]
    return "\n".join(lines)
