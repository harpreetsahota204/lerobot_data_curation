"""Time-efficiency metrics: dead time at the start, and episode duration."""

import numpy as np

from .. import activity
from ..signals import all_arm_speed
from .base import MV

MIN_SPEED_SAMPLES = 8


def idle_lead_s(ep):
    """Seconds of stillness before the first movement (no-op frames at the start)."""
    speed = all_arm_speed(ep)
    if speed is None or len(speed) < MIN_SPEED_SAMPLES:
        return {}
    filtered = activity.filtered_speed(speed, ep.fps)
    threshold = activity.idle_threshold(filtered)
    return {"": MV(activity.leading_idle_samples(filtered, threshold) / ep.fps)}


def duration_s(ep):
    """Episode duration in seconds. `length_z` standardizes it within a group."""
    if ep.fps <= 0 or ep.length == 0:
        return {}
    return {"": MV(ep.length / ep.fps)}


def _filtered(ep):
    speed = all_arm_speed(ep)
    if speed is None or len(speed) < MIN_SPEED_SAMPLES:
        return None, None
    filtered = activity.filtered_speed(speed, ep.fps)
    return filtered, activity.idle_threshold(filtered)


def idle_trail_s(ep):
    """Seconds of stillness at the end. Low weight: fixed-duration recordings end this way."""
    filtered, threshold = _filtered(ep)
    if filtered is None:
        return {}
    return {"": MV(activity.trailing_idle_samples(filtered, threshold) / ep.fps)}


def longest_pause_s(ep):
    """Longest idle run that touches neither end (a mid-task stall or hesitation)."""
    filtered, threshold = _filtered(ep)
    if filtered is None:
        return {}
    from .gripper import recovery_events

    # A pause next to a regrasp is the operator recovering, not stalling.
    recoveries = [(c1 - int(ep.fps), c2 + int(ep.fps)) for c1, _, c2 in recovery_events(ep)]
    interior = [
        (a, b)
        for a, b in activity.idle_runs(filtered, threshold)
        if a > 0
        and b < len(filtered)
        and not any(a < r1 and b > r0 for r0, r1 in recoveries)
    ]
    longest = max((b - a for a, b in interior), default=0)
    return {"": MV(longest / ep.fps)}


def end_motion_ratio(ep):
    """Speed in the last second relative to moving speed. High means cut off mid-motion."""
    filtered, _ = _filtered(ep)
    if filtered is None:
        return {}
    ratio = activity.end_motion_ratio(filtered, ep.fps)
    return {"": MV(ratio)} if ratio == ratio else {}


def idle_frac(ep):
    """Fraction of frames below the idle speed. Context only, never scored."""
    filtered, threshold = _filtered(ep)
    if filtered is None:
        return {}
    idle = filtered < threshold if threshold > 0 else filtered <= 0
    return {"": MV(float(np.mean(idle)))}


def path_length(ep):
    """Total range-normalized joint-space path length. `path_length_z` standardizes it."""
    speed = all_arm_speed(ep)
    if speed is None:
        return {}
    return {"": MV(float(np.sum(speed) / ep.fps))}
