"""Tracking and contact metrics: how well the robot follows commands, and jolts.

Reads ``observation.state`` (contact, jolts and limits happen to the follower
robot, not the leader) and, for tracking, both state and action.
"""

import numpy as np

from ..signals import arm_dims, arm_index_groups, normalized, shares_joint_space
from .base import MV

MAX_LAG_S = 1.0
MIN_LAG_CORR = 0.3
SPIKE_MAD_K = 8.0
SPIKE_FLOOR = 20.0  # range-fractions / s^2; keeps idle joints from making every frame a spike
LIMIT_MARGIN = 0.02
MIN_ACTIVE_SHARE = 0.05  # a dimension counts if it moves at least this share of its dataset range


def _lag_samples(ep):
    """Best-aligning lag of state behind action, in samples, or None if unreliable.

    Cross-correlates range-normalized velocities over the arm dimensions.
    """
    dims = arm_dims(ep)
    if len(ep.action) < 8 or not dims:
        return None
    a = np.diff(normalized(ep.action, ep.action_range)[:, dims], axis=0)
    s = np.diff(normalized(ep.state, ep.state_range)[:, dims], axis=0)
    a = a - a.mean(axis=0)
    s = s - s.mean(axis=0)
    norm = np.sqrt((a**2).sum(axis=0) * (s**2).sum(axis=0))
    keep = norm > 1e-12
    if not keep.any():
        return None
    a, s, norm = a[:, keep], s[:, keep], norm[keep]
    n = len(a)
    max_lag = min(int(round(MAX_LAG_S * ep.fps)), n // 3)
    scores = np.array(
        [np.mean((a[: n - k] * s[k:]).sum(axis=0) / norm * n / (n - k)) for k in range(max_lag + 1)]
    )
    best = int(np.argmax(scores))
    return best if scores[best] >= MIN_LAG_CORR else None


def state_acceleration(ep, dims):
    """|acceleration| of the range-normalized state per frame and arm joint, in range-fractions per s^2."""
    x = normalized(ep.state, ep.state_range)[:, dims]
    return np.abs(np.diff(x, n=2, axis=0)) * ep.fps**2


def spike_threshold(acc):
    """Per-joint spike threshold: ``median + k x MAD`` of that joint's own |acceleration|, with a floor."""
    median = np.median(acc, axis=0)
    mad = np.median(np.abs(acc - median), axis=0)
    return np.maximum(median + SPIKE_MAD_K * 1.4826 * mad, SPIKE_FLOOR)


def track_lag_ms(ep):
    """Delay between the commanded action and the achieved state, in milliseconds."""
    if not shares_joint_space(ep):
        return {}
    lag = _lag_samples(ep)
    return {} if lag is None else {"": MV(1000.0 * lag / ep.fps)}


def track_resid(ep):
    """Per-arm normalized gap between action and state after lag alignment.

    The per-joint median offset is subtracted first, so a constant calibration
    offset between leader and follower does not count as a tracking error.
    High means the robot is not reaching what was commanded (slip, collision,
    stall).
    """
    if not shares_joint_space(ep):
        return {}
    lag = _lag_samples(ep)
    if lag is None:
        return {}
    a = normalized(ep.action, ep.action_range)
    s = normalized(ep.state, ep.state_range)
    n = len(a)
    diff = a[: n - lag] - s[lag:]
    out = {}
    for signal, idx in arm_index_groups(ep).items():
        d = diff[:, idx]
        d = d - np.median(d, axis=0)
        out[signal] = MV(float(np.mean(np.abs(d))))
    return out


def accel_spike_frac(ep):
    """Fraction of frames where any arm joint's state acceleration is a spike.

    A spike exceeds ``median + k x MAD`` of that joint's own |acceleration|, with
    an absolute floor so idle joints (median near 0) do not turn every frame into
    a spike. A proxy for collisions and contact; reported for review, never used
    to mask other metrics.
    """
    dims = arm_dims(ep)
    if len(ep.state) < 8 or not dims:
        return {}
    acc = state_acceleration(ep, dims)
    return {"": MV(float(np.mean(np.any(acc > spike_threshold(acc), axis=1))))}


def joint_limit_frac(ep):
    """Share of (frame, joint) pairs within a small margin of a joint limit.

    Limits come from ``meta/stats.json`` min and max, which is a weak proxy for
    real joint limits (dataset extremes always sit at the edge of the observed
    range). Averaged over the joints that move in this episode. Skipped when
    stats are missing.
    """
    if ep.state_bounds is None:
        return {}
    dims = arm_dims(ep)
    lo, hi = ep.state_bounds
    if not dims or len(lo) != ep.state.shape[1]:
        return {}
    lo, hi = lo[dims], hi[dims]
    span = hi - lo
    x = ep.state[:, dims]
    # Only dimensions that actually move in this episode: a joint that rests at
    # its dataset minimum (a closed gripper, a parked wrist) sits "at a limit"
    # in every frame and would drown the signal.
    active = (span > 1e-9) & (np.ptp(x, axis=0) >= MIN_ACTIVE_SHARE * span)
    if not active.any():
        return {}
    x, lo, hi, span = x[:, active], lo[active], hi[active], span[active]
    near = (x <= lo + LIMIT_MARGIN * span) | (x >= hi - LIMIT_MARGIN * span)
    return {"": MV(float(np.mean(near)))}
