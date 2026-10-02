"""Turns an episode's action array into speed profiles per joint group.

Which dimensions are an arm, a gripper or a hand is never guessed: it comes from the
joint groups the user picked (``episode.groups``, see ``engine/picks.py``). With no
groups, every function here returns nothing and the metrics that need them switch off.

Rules from METRICS.md (Signals):

- Each joint is range-normalized before speeds are combined, so a joint with a
  large range does not dominate and units (degrees, radians, -100..100) do not
  matter. Ranges come from ``meta/stats.json`` when present, otherwise a robust
  per-episode spread.
- Gripper and hand dimensions are excluded from arm speed.
"""

import numpy as np

from .picks import group_signal

GRIPPER_HYSTERESIS = (0.4, 0.6)  # of the gripper's own range


def _joint_range(x, stats_range):
    """Per-dimension scale for range normalization. Inactive dims get 1.0."""
    if stats_range is not None and len(stats_range) == x.shape[1]:
        rng = np.asarray(stats_range, dtype=np.float64).copy()
    else:
        rng = np.percentile(x, 95, axis=0) - np.percentile(x, 5, axis=0)
    rng[~np.isfinite(rng) | (rng <= 1e-9)] = 1.0
    return rng


def _groups(ep, *roles):
    return [g for g in (ep.groups or []) if g["role"] in roles and g["dims"]]


def arm_index_groups(ep):
    """``{signal: [dimension indexes]}`` for the groups marked Arm, e.g. ``{"arm_left": [0, 1, 2]}``."""
    if ep.action is None:
        return {}
    return {group_signal(g): list(g["dims"]) for g in _groups(ep, "arm")}


def arm_speeds(ep, roles=("arm", "hand")):
    """Speed profile per group of the given roles: ``{signal: (T-1,) array}``.

    Speeds are in range-fractions per second. Arms and hands are both smoothness
    signals, each on its own, so a 22-joint hand never swamps a 7-joint arm.
    """
    if ep.action is None or len(ep.action) < 3:
        return {}
    x = ep.action / _joint_range(ep.action, ep.action_range)
    d = np.diff(x, axis=0) * ep.fps
    return {group_signal(g): np.linalg.norm(d[:, g["dims"]], axis=1) for g in _groups(ep, *roles)}


def all_arm_speed(ep):
    """One speed profile over every dimension marked Arm, or None. Hands and grippers are not in it."""
    idx = arm_dims(ep)
    if not idx or ep.action is None or len(ep.action) < 3:
        return None
    x = ep.action / _joint_range(ep.action, ep.action_range)
    d = np.diff(x, axis=0) * ep.fps
    return np.linalg.norm(d[:, idx], axis=1)


def gripper_groups(ep):
    """``{group name: [dimension indexes]}`` for the groups marked Gripper."""
    if ep.action is None:
        return {}
    return {g["name"]: list(g["dims"]) for g in _groups(ep, "gripper")}


def _unit_range(x):
    """Rescales a 1-D signal to [0, 1] by its own min and max (None if it never moves)."""
    lo, hi = np.min(x), np.max(x)
    if not np.isfinite(lo) or hi - lo <= 1e-9:
        return None
    return (x - lo) / (hi - lo)


def gripper_signals(ep):
    """Action gripper per group marked Gripper, each rescaled to [0, 1]: ``{group name: array}``.

    A group whose dimensions never move is left out. The direction of "closed" is
    not assumed here.
    """
    out = {}
    for name, idx in gripper_groups(ep).items():
        unit = _unit_range(np.mean(ep.action[:, idx], axis=1))
        if unit is not None:
            out[name] = unit
    return out


def gripper_transitions(unit, low=GRIPPER_HYSTERESIS[0], high=GRIPPER_HYSTERESIS[1]):
    """Indexes where a [0, 1] gripper signal crosses between its two states.

    Uses hysteresis so noise near the midpoint does not count as a transition.
    """
    if len(unit) == 0:
        return []
    state = unit[0] >= 0.5
    events = []
    for i, v in enumerate(unit):
        if state and v < low:
            state = False
            events.append(i)
        elif not state and v > high:
            state = True
            events.append(i)
    return events


def arm_signal_for_gripper(ep, gripper_name):
    """The arm signal a gripper group belongs to, or None when that is not stated.

    The pairing is the one the user gave (the gripper group's ``arm``). With exactly
    one arm group there is nothing to choose, so that arm is used. With several arms
    and no stated pairing there is no answer, and the metrics that need it skip that gripper.
    """
    arms = _groups(ep, "arm")
    for g in _groups(ep, "gripper"):
        if g["name"] == gripper_name and g.get("arm"):
            match = [a for a in arms if a["name"] == g["arm"]]
            return group_signal(match[0]) if match else None
    return group_signal(arms[0]) if len(arms) == 1 else None


def group_signal_for_gripper(ep, gripper_name):
    """The signal slug of a gripper group itself (``gripper_left``)."""
    for g in _groups(ep, "gripper"):
        if g["name"] == gripper_name:
            return group_signal(g)
    return "gripper_%s" % gripper_name


def arm_dims(ep):
    """Sorted dimension indexes of every arm group (grippers and hands excluded)."""
    return sorted({i for idx in arm_index_groups(ep).values() for i in idx})


def normalized(x, stats_range):
    """`x` range-normalized per dimension (see :func:`_joint_range`)."""
    return x / _joint_range(x, stats_range)


def shares_joint_space(ep):
    """Whether the user said the action is joint positions in the state's space (and the shapes agree).

    Tracking, acceleration-spike and joint-limit metrics index the state with the
    action's dimensions, so they only make sense then. Nothing is inferred: with no
    pick, or a pick of "other", this is False and those metrics switch off.
    """
    if ep.action is None or ep.state is None or ep.action.shape != ep.state.shape:
        return False
    return ep.assumptions.get("action_semantics") == "joint_positions"


def gripper_events(unit, low=GRIPPER_HYSTERESIS[0], high=GRIPPER_HYSTERESIS[1]):
    """Gripper transitions as ``(index, went_high)`` pairs, in time order."""
    return [(i, bool(unit[i] > 0.5)) for i in gripper_transitions(unit, low, high)]


def close_and_open_events(unit, open_is):
    """Splits transitions into ``(closes, opens)`` index lists, given which side is open."""
    closes, opens = [], []
    for i, went_high in gripper_events(unit):
        is_open = went_high if open_is == "high" else not went_high
        (opens if is_open else closes).append(i)
    return closes, opens
