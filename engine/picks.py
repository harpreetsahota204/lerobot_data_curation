"""What the user told the plugin about a dataset: the "picks".

Nothing here is inferred. A metric that depends on a pick the user has not made
switches off and says why, instead of running on a guess. The compute form builds a
picks dict, the run record stores it, and the engine reads it through
:func:`assumptions_for` and :func:`groups_for_source`.

A picks dict::

    {
      "state_key": "observation.state" | None,   # which array is the state
      "action_key": "action" | None,              # which array is the action
      "action_semantics": "joint_positions" | "other" | None,
      "gripper_open_is": "high" | "low" | None,
      "layout": "single" | "dual",                # one arm, or a left and a right
      "groups": [                                  # over the dimensions of the action
        {"name": "left", "role": "arm", "dims": [0, 1, 2]},
        {"name": "left", "role": "gripper", "dims": [3], "arm": "left"},
      ],
      "cameras": ["observation.images.top"],       # the cameras to score
    }

The form has one field per role, and a field is one signal: every joint put in the Arm
field is one arm, so a five-joint arm is not five signals. A single-arm layout has an
Arm, a Gripper and a Hand field. A dual-arm layout has the same three for the left and
for the right, which also says which gripper belongs to which arm.

Roles:

- ``arm``: scored for smoothness and used for idle and pause detection and tracking
- ``gripper``: an open/close dimension. Feeds the gripper metrics and is left out of
  arm speed
- ``hand``: a multi-joint end effector. Scored for smoothness as its own signal, left
  out of arm speed, and given no gripper metrics

Joints the user puts in no field are ignored.
"""

import re

from .reader import is_named, slug

ROLES = ("arm", "gripper", "hand")
LAYOUTS = (("single", "Single arm"), ("dual", "Dual arm"))
DEFAULT_LAYOUT = "single"
# The LeRobot v3 spec's names for the state and action arrays. The form starts on these
# when a dataset has them: they are the format's convention, not a guess about the data.
STANDARD_ARRAYS = {"state_key": "observation.state", "action_key": "action"}
GROUP_PREFIX = "g:"  # a chip that stands for every joint of a named group

ROLE_HELP = {
    "arm": "The joints that move the arm. Scored for smoothness, idle and pause time, and tracking.",
    "gripper": "The open/close dimension. Several dimensions are averaged into one gripper signal.",
    "hand": "The joints of a multi-joint hand. Scored for smoothness on its own, with no gripper metrics.",
}

_TRAILING_INDEX = re.compile(r"[\s_.\-\[]*\d+\]?$")


def empty_picks():
    return {
        "state_key": None,
        "action_key": None,
        "action_semantics": None,
        "gripper_open_is": None,
        "layout": None,
        "groups": [],
        "cameras": [],
    }


def clean_picks(picks):
    """A picks dict with every key present and empty values turned into None or []."""
    out = empty_picks()
    for key in ("state_key", "action_key", "action_semantics", "gripper_open_is", "layout"):
        out[key] = (picks or {}).get(key) or None
    out["groups"] = [dict(g) for g in (picks or {}).get("groups") or []]
    out["cameras"] = [c for c in (picks or {}).get("cameras") or [] if c]
    return out


# -- the fields of the joint form ---------------------------------------------------


def joint_fields(layout):
    """The joint fields the form shows: ``[{"key", "side", "role", "label", "help"}]``.

    A single-arm layout has one set (side ``all``), a dual-arm layout a left and a right.
    """
    sides = ["left", "right"] if layout == "dual" else ["all"]
    out = []
    for side in sides:
        for role in ROLES:
            label = ("%s %s" % (side.capitalize(), role) if side != "all" else role.capitalize()) + " joints"
            out.append(
                {
                    "key": role if side == "all" else "%s_%s" % (side, role),
                    "side": side,
                    "role": role,
                    "label": label,
                    "help": ROLE_HELP[role],
                }
            )
    return out


def assemble_groups(layout, dims_by_field):
    """Groups from what was put in each field: ``{field key: [dimension indexes]}``.

    One group per non-empty field, so each field is one signal. A dimension used by an
    earlier field is dropped from a later one, and a gripper is paired with the arm of
    its own side when that side has an arm.
    """
    groups, taken, arm_sides = [], set(), set()
    for f in joint_fields(layout):
        dims = [d for d in sorted(set(dims_by_field.get(f["key"]) or [])) if d not in taken]
        if not dims:
            continue
        taken |= set(dims)
        groups.append({"name": f["side"], "role": f["role"], "dims": dims})
        if f["role"] == "arm":
            arm_sides.add(f["side"])
    for g in groups:
        if g["role"] == "gripper" and g["name"] in arm_sides:
            g["arm"] = g["name"]
    return groups


def dims_by_field(layout, groups):
    """The inverse of :func:`assemble_groups`, to refill the form from the last run's groups."""
    by_group = {(g.get("name"), g.get("role")): g.get("dims") or [] for g in groups or []}
    return {f["key"]: list(by_group.get((f["side"], f["role"]), [])) for f in joint_fields(layout)}


# -- choosing joints ----------------------------------------------------------------


def names_usable(names, feature):
    """Whether the dataset gave real, distinct per-dimension names for `feature`."""
    return is_named(names, feature) and len(set(names)) == len(names)


def candidate_groups(names):
    """Groups of dimensions that share a name once a trailing index is removed.

    ``left_arm_q_0 ... left_arm_q_6`` becomes one group ``left_arm_q`` of 7 dimensions.
    Names with no index stay single-dimension groups. These only feed one-click
    shortcuts in the joint pickers: a shortcut adds its joints, it never decides a role.
    """
    order, dims = [], {}
    for i, name in enumerate(names):
        prefix = _TRAILING_INDEX.sub("", str(name)) or str(name)
        if prefix not in dims:
            order.append(prefix)
            dims[prefix] = []
        dims[prefix].append(i)
    return [{"name": p, "dims": dims[p]} for p in order]


def dims_from_chips(values, candidates, dim):
    """Dimension indexes from picked chips: a joint is its index, a shortcut is ``g:<group>``."""
    by_name = {c["name"]: c["dims"] for c in candidates}
    dims = []
    for v in values or []:
        v = str(v)
        if v.startswith(GROUP_PREFIX):
            dims += by_name.get(v[len(GROUP_PREFIX) :], [])
        elif v.isdigit() and int(v) < dim:
            dims.append(int(v))
    return sorted(set(dims))


def chips_for_dims(dims, candidates):
    """The chips that show `dims`: a shortcut for each fully covered multi-joint group, then single joints."""
    left, chips = set(dims or []), []
    for c in candidates:
        if len(c["dims"]) > 1 and set(c["dims"]) <= left:
            chips.append(GROUP_PREFIX + c["name"])
            left -= set(c["dims"])
    return chips + [str(d) for d in sorted(left)]


def describe_dims(dims):
    """``"0-6"`` or ``"3"`` or ``"0-2, 5"`` for a list of dimension indexes."""
    dims = sorted(dims)
    parts, start = [], None
    for i, d in enumerate(dims):
        if start is None:
            start = prev = d
        elif d == prev + 1:
            prev = d
        else:
            parts.append((start, prev))
            start = prev = d
        if i == len(dims) - 1:
            parts.append((start, prev))
    return ", ".join("%d" % a if a == b else "%d-%d" % (a, b) for a, b in parts)


def parse_dim_list(text, dim):
    """Parses ``"0-6"`` or ``"0-2, 5"`` into dimension indexes, for arrays with no usable names.

    Returns ``(dims, errors)``.
    """
    dims, errors = set(), []
    cleaned = re.sub(r"\s*-\s*", "-", text or "")
    for item in re.split(r"[,\s;]+", cleaned):
        if not item:
            continue
        m = re.fullmatch(r"(\d+)(?:-(\d+))?", item)
        if not m:
            errors.append("Could not read %r. Write dimensions like '0-6' or '0-2, 5'." % item)
            continue
        a, b = int(m.group(1)), int(m.group(2) if m.group(2) else m.group(1))
        if b < a or b >= dim:
            errors.append("%r is outside the %d dimensions of this array (0 to %d)." % (item, dim, dim - 1))
            continue
        dims |= set(range(a, b + 1))
    return sorted(dims), errors


# -- what the engine reads ---------------------------------------------------------


def group_signal(group):
    """The signal slug a group's values are stored under: ``arm_left``, ``gripper_left``, ``hand_right``."""
    return "%s_%s" % (group["role"], slug(group["name"]))


def groups_for_source(picks, action_dim):
    """The picked groups that fit an action of `action_dim` dimensions."""
    if action_dim is None:
        return []
    out = []
    for g in (picks or {}).get("groups") or []:
        if g.get("role") in ROLES and g.get("dims") and max(g["dims"]) < action_dim:
            out.append(dict(g))
    return out


def assumptions_for(picks):
    """The per-source assumptions dict metrics read from ``episode.assumptions``."""
    picks = clean_picks(picks)
    out = {}
    if picks["action_semantics"]:
        out["action_semantics"] = picks["action_semantics"]
    if picks["gripper_open_is"]:
        out["gripper_open_is"] = picks["gripper_open_is"]
    out["cameras"] = list(picks["cameras"])
    return out


def explain_off(picks, camera_metrics_selected=False):
    """Plain sentences for what is switched off because a pick is missing.

    Args:
        picks: a picks dict
        camera_metrics_selected (False): whether any camera metric is ticked
    """
    p = clean_picks(picks)
    roles = {g["role"] for g in p["groups"]}
    off = []
    if not p["action_key"]:
        off.append(
            "every metric that reads the action (smoothness, idle and pause, tracking, gripper, camera lag): no action array is picked"
        )
    elif "arm" not in roles:
        off.append("arm smoothness, idle and pause time, and camera lag: no arm joints are picked")
    if not p["state_key"]:
        off.append("tracking, acceleration spikes, joint limits and frozen-feed detection: no state array is picked")
    if p["action_key"] and p["state_key"] and p["action_semantics"] != "joint_positions":
        off.append(
            "tracking, acceleration spikes and joint limits: you have not said the action is joint positions in the state's space"
            if p["action_semantics"] is None
            else "tracking, acceleration spikes and joint limits: you said the action is not joint positions"
        )
    if "gripper" not in roles:
        off.append("gripper metrics: no gripper joints are picked")
    elif not p["gripper_open_is"]:
        off.append("regrasp recovery, missed grasps and gripper phases: the open direction is not picked")
    if camera_metrics_selected and not p["cameras"]:
        off.append("camera metrics: no camera is picked")
    return off


def readiness(picks, camera_metrics_selected=False):
    """``(ready, off)`` for the form's checklist: metric families that will run, and ``(family, what to pick)``.

    Args:
        picks: a picks dict
        camera_metrics_selected (False): whether any camera metric is ticked
    """
    p = clean_picks(picks)
    roles = {g["role"] for g in p["groups"]}
    action, state = bool(p["action_key"]), bool(p["state_key"])
    ready, off = [], []

    def check(family, need):
        if need:
            off.append((family, need))
        else:
            ready.append(family)

    check("smoothness", None if action and roles & {"arm", "hand"} else ("pick the action array" if not action else "pick arm joints"))
    check("idle and pause time", None if action and "arm" in roles else ("pick the action array" if not action else "pick arm joints"))
    if not (state and action):
        need = "pick the state and action arrays"
    elif p["action_semantics"] is None:
        need = "answer whether the action is joint positions"
    elif p["action_semantics"] != "joint_positions":
        need = "only for actions that are joint positions"
    else:
        need = None
    check("tracking, acceleration spikes and joint limits", need)
    if "gripper" not in roles:
        need = "pick gripper joints"
    elif not p["gripper_open_is"]:
        need = "pick the gripper's open direction"
    else:
        need = None
    check("gripper metrics", need)
    if camera_metrics_selected:
        check("camera metrics", None if p["cameras"] else "pick a camera on the Camera tab")
    ready += ["integrity", "language"]
    return ready, off
