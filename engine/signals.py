"""Turns an episode's action array into speed profiles per arm.

Rules from METRICS.md (Signals):

- Each joint is range-normalized before speeds are combined, so a joint with a
  large range does not dominate and units (degrees, radians, -100..100) do not
  matter. Ranges come from ``meta/stats.json`` when present, otherwise a robust
  per-episode spread.
- Gripper dimensions are excluded from arm speed.
- Datasets without joint names use one ``arm:all`` signal over every dimension.
"""

import numpy as np

from .reader import is_named, slug, split_joint_groups

GRIPPER_HYSTERESIS = (0.4, 0.6)  # of the gripper's own range


def _joint_range(x, stats_range):
    """Per-dimension scale for range normalization. Inactive dims get 1.0."""
    if stats_range is not None and len(stats_range) == x.shape[1]:
        rng = np.asarray(stats_range, dtype=np.float64).copy()
    else:
        rng = np.percentile(x, 95, axis=0) - np.percentile(x, 5, axis=0)
    rng[~np.isfinite(rng) | (rng <= 1e-9)] = 1.0
    return rng


def arm_index_groups(ep):
    """``{signal_slug: [dimension indexes]}`` for the arm signals of an episode."""
    if ep.action is None:
        return {}
    dim = ep.action.shape[1]
    if not is_named(ep.action_names, ep.feature_map.action_key if ep.feature_map else "action"):
        return {"arm_all": list(range(dim))}
    groups = split_joint_groups(ep.action_names)
    return {slug(k): idx for k, idx in groups.items() if k.startswith("arm:")}


def arm_speeds(ep):
    """Speed profile per arm signal: ``{signal_slug: (T-1,) array}``.

    Speeds are in range-fractions per second.
    """
    if ep.action is None or len(ep.action) < 3:
        return {}
    x = ep.action / _joint_range(ep.action, ep.action_range)
    d = np.diff(x, axis=0) * ep.fps
    return {
        signal: np.linalg.norm(d[:, idx], axis=1)
        for signal, idx in arm_index_groups(ep).items()
        if idx
    }


def all_arm_speed(ep):
    """One speed profile over every arm dimension, or None."""
    groups = arm_index_groups(ep)
    idx = sorted({i for g in groups.values() for i in g})
    if not idx or ep.action is None or len(ep.action) < 3:
        return None
    x = ep.action / _joint_range(ep.action, ep.action_range)
    d = np.diff(x, axis=0) * ep.fps
    return np.linalg.norm(d[:, idx], axis=1)


def named_action_groups(ep):
    """``{"arm:left": [idx], "gripper:left": [idx], ...}`` or {} when names are missing."""
    if ep.action is None or ep.feature_map is None:
        return {}
    if not is_named(ep.action_names, ep.feature_map.action_key or "action"):
        return {}
    return split_joint_groups(ep.action_names)


def _unit_range(x):
    """Rescales a 1-D signal to [0, 1] by its own min and max (None if it never moves)."""
    lo, hi = np.min(x), np.max(x)
    if not np.isfinite(lo) or hi - lo <= 1e-9:
        return None
    return (x - lo) / (hi - lo)


def gripper_signals(ep):
    """Action gripper per side, each rescaled to [0, 1]: ``{"left": array, ...}``.

    Empty when joint names are missing (the gripper cannot be identified) or the
    gripper never moves. The direction of "closed" is not assumed here.
    """
    out = {}
    for name, idx in named_action_groups(ep).items():
        if not name.startswith("gripper:"):
            continue
        unit = _unit_range(np.mean(ep.action[:, idx], axis=1))
        if unit is not None:
            out[name.split(":", 1)[1]] = unit
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


def arm_signal_for_side(side):
    """The arm signal slug that pairs with a gripper side."""
    return "arm_%s" % side


def arm_dims(ep):
    """Sorted dimension indexes of every arm signal (grippers excluded)."""
    return sorted({i for idx in arm_index_groups(ep).values() for i in idx})


def normalized(x, stats_range):
    """`x` range-normalized per dimension (see :func:`_joint_range`)."""
    return x / _joint_range(x, stats_range)


def shares_joint_space(ep, threshold=0.8):
    """Whether action and state look like the same joint positions.

    True when both exist with the same shape and the median per-dimension
    correlation over arm dimensions is at least `threshold`. Tracking metrics
    only make sense then (leader-follower joint control), so they switch off
    otherwise (delta or end-effector actions, unknown semantics).
    """
    if ep.action is None or ep.state is None or ep.action.shape != ep.state.shape:
        return False
    confirmed = ep.assumptions.get("action_semantics")
    if confirmed == "joint_positions":
        return True
    if confirmed == "other":
        return False
    dims = arm_dims(ep)
    if not dims or len(ep.action) < 10:
        return False
    corrs = []
    for d in dims:
        a, s = ep.action[:, d], ep.state[:, d]
        if np.std(a) < 1e-9 or np.std(s) < 1e-9:
            continue
        corrs.append(np.corrcoef(a, s)[0, 1])
    return bool(corrs) and float(np.nanmedian(corrs)) >= threshold


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
