"""Gripper metrics. They need a joint group marked Gripper in the picks."""

import numpy as np

from ..signals import (
    arm_dims,
    arm_signal_for_gripper,
    close_and_open_events,
    gripper_groups,
    gripper_signals,
    gripper_transitions,
    group_signal_for_gripper,
    normalized,
    shares_joint_space,
)
from .base import MV

RECOVERY_REOPEN_S = 3.0  # a regrasp closes again within this long after opening
RECOVERY_WINDOW_S = 6.0  # ... and within this long after the first close
RECOVERY_DISTANCE = 0.15  # arm positions at the two closes, range-normalized per dimension
HOLD_S = 0.5  # how long after a close the grasp is judged
HELD_GAP = 0.1  # gap between commanded and achieved gripper, as a share of the commanded range


def gripper_flips_per_s(ep):
    """Open/close transitions per second on each gripper, with hysteresis.

    High means chatter or a hesitant operator. Several clean cycles can be
    legitimate regrasp recoveries, which `recovery_count` tracks separately.
    """
    out = {}
    duration = ep.length / ep.fps if ep.fps > 0 else 0.0
    if duration <= 0:
        return out
    for side, unit in gripper_signals(ep).items():
        out["gripper_%s" % side] = MV(len(gripper_transitions(unit)) / duration)
    return out


def recovery_events(ep):
    """Regrasp recoveries as ``(first_close, reopen, second_close)`` index triples.

    A recovery is close, open, close again soon after and near the same arm
    position: the operator missed and tried again. Needs a known gripper
    convention.
    """
    open_is = ep.assumptions.get("gripper_open_is")
    if open_is is None or ep.action is None:
        return []
    dims = arm_dims(ep)
    if not dims:
        return []
    arm = normalized(ep.action, ep.action_range)[:, dims]
    found = []
    for unit in gripper_signals(ep).values():
        closes, opens = close_and_open_events(unit, open_is)
        for c1 in closes:
            reopen = next((o for o in opens if o > c1), None)
            if reopen is None or (reopen - c1) / ep.fps > RECOVERY_WINDOW_S:
                continue
            c2 = next((c for c in closes if c > reopen), None)
            if c2 is None or (c2 - reopen) / ep.fps > RECOVERY_REOPEN_S or (c2 - c1) / ep.fps > RECOVERY_WINDOW_S:
                continue
            distance = np.linalg.norm(arm[c1] - arm[c2]) / np.sqrt(len(dims))
            if distance <= RECOVERY_DISTANCE:
                found.append((c1, reopen, c2))
    return found


def recovery_count(ep):
    """Number of regrasp recoveries. Neutral or positive: recovery is never penalized."""
    if ep.assumptions.get("gripper_open_is") is None or not gripper_signals(ep):
        return {}
    return {"": MV(float(len(recovery_events(ep))))}


def missed_grasp_frac(ep):
    """Fraction of close commands where the follower closes fully (nothing was held).

    A gripper that closes on an object stalls short of the commanded position;
    one that closes on nothing reaches it. Compares commanded and achieved
    gripper position half a second after each close, after subtracting the
    leader-follower offset seen while the gripper was open. Needs a known
    gripper convention, an action declared to be joint positions in the state's
    space, and a joint group marked Gripper. Heuristic.
    """
    open_is = ep.assumptions.get("gripper_open_is")
    if open_is is None or ep.state is None or not shares_joint_space(ep):
        return {}
    out = {}
    groups = gripper_groups(ep)
    hold = max(1, int(round(HOLD_S * ep.fps)))
    for side, unit in gripper_signals(ep).items():
        idx = groups.get(side)
        if not idx:
            continue
        a = np.mean(ep.action[:, idx], axis=1)
        s = np.mean(ep.state[:, idx], axis=1)
        span = float(a.max() - a.min())
        if span <= 1e-9:
            continue
        closes, opens = close_and_open_events(unit, open_is)
        is_open = unit >= 0.5 if open_is == "high" else unit < 0.5
        if is_open.sum() < 3:
            continue
        offset = float(np.median((a - s)[is_open]))
        missed = held = 0
        for c in closes:
            nxt = min([o for o in opens if o > c] + [len(a)])
            j = c + hold
            if j >= nxt or j >= len(a):
                continue  # released before the grasp could be judged
            gap = abs((a[j] - s[j]) - offset) / span
            if gap > HELD_GAP:
                held += 1
            else:
                missed += 1
        if missed + held:
            out[arm_signal_for_gripper(ep, side) or group_signal_for_gripper(ep, side)] = MV(missed / (missed + held))
    return out
