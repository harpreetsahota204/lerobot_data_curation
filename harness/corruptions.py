"""In-memory corruptions of real episodes, each with the metrics it must move.

Each corruption is a function ``apply(episode, rng) -> EpisodeData | None`` that
returns a corrupted copy, or None when the episode cannot carry the defect (no
state, no named gripper, and so on). ``EXPECT`` lists the metrics that must rank
the corrupted copy worse than the clean episode it came from.

A fork that adds a metric adds one entry here.
"""

from dataclasses import dataclass, replace

import numpy as np

from lerobot_data_curation.engine.metrics import time_metrics, tracking
from lerobot_data_curation.engine.signals import (
    _joint_range,
    all_arm_speed,
    arm_dims,
    gripper_signals,
    named_action_groups,
    shares_joint_space,
)


@dataclass(frozen=True)
class Corruption:
    name: str
    description: str
    apply: callable
    expect: tuple  # metric names that must move the right way
    # For signed-z metrics, which way the signed value must move (+1 up, -1 down).
    directions: tuple = ()


def _rest(ep, n):
    """`n` frames repeating the first action/state (a robot holding still)."""
    return n


def _resample(ep, action=None, state=None):
    n = len(action if action is not None else state)
    fi = np.arange(n)
    return replace(
        ep,
        action=action if action is not None else ep.action,
        state=state if state is not None else ep.state,
        length=n,
        expected_length=n,
        frame_index=fi,
        raw_frame_index=fi,
        timestamps=fi / ep.fps,
        raw_timestamps=fi / ep.fps,
    )


def jitter(ep, rng):
    """White noise worth 5% of each joint's motion in this episode."""
    if ep.action is None or len(ep.action) < 60:
        return None
    spread = np.percentile(ep.action, 95, axis=0) - np.percentile(ep.action, 5, axis=0)
    scale = 0.05 * np.maximum(spread, 1e-6)
    return replace(ep, action=ep.action + rng.normal(0, 1, ep.action.shape) * scale)


def stop_and_go(ep, rng):
    """Hesitation: freeze for 0.3 s after every second of motion, through the middle 60%."""
    if ep.action is None or len(ep.action) < int(8 * ep.fps):
        return None
    n = len(ep.action)
    a, b = int(0.2 * n), int(0.8 * n)
    step, hold = int(ep.fps), max(2, int(0.3 * ep.fps))
    parts_a, parts_s = [ep.action[:a]], [] if ep.state is None else [ep.state[:a]]
    for i in range(a, b, step):
        j = min(i + step, b)
        parts_a.append(ep.action[i:j])
        parts_a.append(np.repeat(ep.action[j - 1 : j], hold, 0))
        if ep.state is not None:
            parts_s.append(ep.state[i:j])
            parts_s.append(np.repeat(ep.state[j - 1 : j], hold, 0))
    parts_a.append(ep.action[b:])
    action = np.vstack(parts_a)
    state = None
    if ep.state is not None:
        state = np.vstack([ep.state[:a]] + parts_s + [ep.state[b:]])
    return _resample(ep, action, state)


def idle_start(ep, rng):
    if ep.action is None or len(ep.action) < 60:
        return None
    pad = int(2 * ep.fps)
    action = np.vstack([np.repeat(ep.action[:1], pad, 0), ep.action])
    state = None if ep.state is None else np.vstack([np.repeat(ep.state[:1], pad, 0), ep.state])
    return _resample(ep, action, state)


def idle_end(ep, rng):
    if ep.action is None or len(ep.action) < 60:
        return None
    pad = int(2 * ep.fps)
    action = np.vstack([ep.action, np.repeat(ep.action[-1:], pad, 0)])
    state = None if ep.state is None else np.vstack([ep.state, np.repeat(ep.state[-1:], pad, 0)])
    return _resample(ep, action, state)


def mid_pause(ep, rng):
    if ep.action is None or len(ep.action) < int(6 * ep.fps):
        return None
    k = len(ep.action) // 2
    pad = int(2 * ep.fps)
    action = np.vstack([ep.action[:k], np.repeat(ep.action[k : k + 1], pad, 0), ep.action[k:]])
    state = None if ep.state is None else np.vstack([ep.state[:k], np.repeat(ep.state[k : k + 1], pad, 0), ep.state[k:]])
    return _resample(ep, action, state)


def cut_off(ep, rng):
    """Truncated mid-motion: the episode stops at a moment when the arm is moving fast."""
    speed = all_arm_speed(ep)
    if speed is None or len(ep.action) < int(6 * ep.fps):
        return None
    ratio = time_metrics.end_motion_ratio(ep)
    if not ratio or ratio[""].value >= 0.3:
        return None  # already ends mid-motion: it cannot be made more cut off
    n, w = len(speed), max(3, int(ep.fps))
    lo, hi = int(0.3 * n), int(0.7 * n)
    # cut where the arm has been moving hardest over the preceding second
    steadiness = [np.median(speed[i - w : i]) for i in range(max(lo, w), hi)]
    k = max(lo, w) + int(np.argmax(steadiness))
    state = None if ep.state is None else ep.state[:k]
    return _resample(ep, ep.action[:k], state)


def abandoned(ep, rng):
    if ep.action is None or len(ep.action) < int(6 * ep.fps):
        return None
    k = max(int(len(ep.action) * 0.25), 12)
    state = None if ep.state is None else ep.state[:k]
    return _resample(ep, ep.action[:k], state)


def duplicate_frame(ep, rng):
    if len(ep.raw_frame_index) < 20:
        return None
    fi = ep.raw_frame_index.copy()
    fi[len(fi) // 2] = fi[len(fi) // 2 - 1]
    return replace(ep, raw_frame_index=fi)


def inject_nan(ep, rng):
    if ep.action is None or len(ep.action) < 20:
        return None
    action = ep.action.copy()
    action[len(action) // 2, 0] = np.nan
    return replace(ep, action=action)


def one_frame(ep, rng):
    if ep.action is None:
        return None
    state = None if ep.state is None else ep.state[:1]
    return _resample(ep, ep.action[:1], state)


def generic_task(ep, rng):
    return replace(ep, tasks=["Hold"])


def missing_task(ep, rng):
    return replace(ep, tasks=[""])


def length_mismatch(ep, rng):
    return replace(ep, expected_length=ep.length + 25)


def video_off(ep, rng):
    if not ep.videos:
        return None
    videos = {}
    for cam, w in ep.videos.items():
        videos[cam] = replace(w, to_timestamp=w.to_timestamp - 1.0)
    return replace(ep, videos=videos)


def stalled_joint(ep, rng):
    """One joint of the state stops following the command for the middle 40%."""
    if not shares_joint_space(ep):
        return None
    dims = arm_dims(ep)
    state = ep.state.copy()
    n = len(state)
    a, b = int(n * 0.3), int(n * 0.7)
    d = dims[int(np.argmax(np.ptp(state[:, dims], axis=0)))]  # the joint that moves most
    state[a:b, d] = state[a, d]
    return replace(ep, state=state)


def lagged_state(ep, rng):
    if not shares_joint_space(ep):
        return None
    lag = max(4, int(0.25 * ep.fps))
    state = np.vstack([np.repeat(ep.state[:1], lag, 0), ep.state[:-lag]])
    return replace(ep, state=state)


def jolts(ep, rng):
    if ep.state is None or arm_dims(ep) == [] or len(ep.state) < 60:
        return None
    state = ep.state.copy()
    span = _joint_range(state, ep.state_range)
    for i in np.linspace(len(state) * 0.2, len(state) * 0.8, 5).astype(int):
        state[i, arm_dims(ep)] += 0.5 * span[arm_dims(ep)]  # a one-frame jump
    return replace(ep, state=state)


def at_limits(ep, rng):
    if ep.state is None or ep.state_bounds is None:
        return None
    dims = arm_dims(ep)
    lo, hi = ep.state_bounds
    if not dims or len(lo) != ep.state.shape[1]:
        return None
    clean = tracking.joint_limit_frac(ep)
    if not clean or clean[""].value >= 0.5:
        return None  # already spends most of its time at the limits
    state = ep.state.copy()
    a, b = int(len(state) * 0.3), int(len(state) * 0.7)
    state[a:b][:, dims] = hi[dims]
    return replace(ep, state=state)


def gripper_chatter(ep, rng):
    groups = named_action_groups(ep)
    idx = [i for name, ix in groups.items() if name.startswith("gripper:") for i in ix]
    if not idx or not gripper_signals(ep):
        return None
    action = ep.action.copy()
    n = len(action)
    a, b = int(n * 0.25), int(n * 0.75)
    lo, hi = action[:, idx].min(axis=0), action[:, idx].max(axis=0)
    toggle = (np.arange(b - a) // 5) % 2
    action[a:b][:, idx] = np.where(toggle[:, None] == 1, hi, lo)
    return replace(ep, action=action)


def wander(ep, rng):
    if ep.action is None or len(ep.action) < 60:
        return None
    dims = arm_dims(ep)
    if not dims:
        return None
    rangeplus = _joint_range(ep.action, ep.action_range)
    t = np.linspace(0, 8 * np.pi, len(ep.action))[:, None]
    action = ep.action.copy()
    action[:, dims] += 0.3 * rangeplus[dims] * np.sin(t)
    return replace(ep, action=action)


CORRUPTIONS = [
    Corruption("identity", "No change. Nothing should move.", lambda ep, rng: replace(ep), ()),
    Corruption("jitter", "Add white noise worth 5% of each joint's motion to the action", jitter, ("ldlj",)),
    Corruption("stop_and_go", "Freeze 0.3 s after every second of motion (hesitation)", stop_and_go, ("sparc", "ldlj")),
    Corruption("wander", "Add a slow sinusoidal detour to every arm joint", wander, ("path_length_z",), (("path_length_z", 1),)),
    Corruption("idle_start", "Prepend 2 s of stillness", idle_start, ("idle_lead_s",)),
    Corruption("idle_end", "Append 2 s of stillness", idle_end, ("idle_trail_s",)),
    Corruption("mid_pause", "Insert a 2 s pause in the middle", mid_pause, ("longest_pause_s",)),
    Corruption("cut_off", "Stop at 50% of the episode, mid-motion", cut_off, ("end_motion_ratio",)),
    Corruption("abandoned", "Keep only the first 25% of the episode", abandoned, ("length_z",), (("length_z", -1),)),
    Corruption("duplicate_frame", "Repeat one frame index", duplicate_frame, ("frame_gaps",)),
    Corruption("nan", "Put a NaN in the action", inject_nan, ("nonfinite_values",)),
    Corruption("one_frame", "Keep a single frame", one_frame, ("too_short",)),
    Corruption("generic_task", "Set the task to 'Hold'", generic_task, ("task_generic",)),
    Corruption("missing_task", "Blank the task string", missing_task, ("task_missing",)),
    Corruption("length_mismatch", "Metadata promises 25 more rows than exist", length_mismatch, ("length_mismatch",)),
    Corruption("video_off", "Shorten every camera window by 1 s", video_off, ("video_window_mismatch",)),
    Corruption("stalled_joint", "State stops following one joint for 40% of the episode", stalled_joint, ("track_resid",)),
    Corruption("lagged_state", "Delay the state by a quarter second", lagged_state, ("track_lag_ms",), (("track_lag_ms", 1),)),
    Corruption("jolts", "Five one-frame jumps in the state", jolts, ("accel_spike_frac",)),
    Corruption("at_limits", "Hold the state at its dataset maximum for 40% of the episode", at_limits, ("joint_limit_frac",)),
    Corruption("gripper_chatter", "Toggle the gripper every 5 frames for half the episode", gripper_chatter, ("gripper_flips_per_s",)),
]

# Metrics that cannot be exercised on this data, and why. Synthetic tests in
# tests/test_engine.py cover them.
NOT_TESTABLE = {
    "action_divergence": "needs at least 5 episodes per source; lr_dev has 2",
    "stats_leverage": "needs at least 3 episodes per source; lr_dev has 2",
    "sparc_phase": "synthetic tests only (needs a named gripper and clear phases)",
    "missed_grasp_frac": "synthetic tests only (needs a known gripper convention)",
    "recovery_count": "synthetic tests only (needs a known gripper convention)",
    "schema_mismatch": "a dataset-level comparison, tested synthetically",
    "iforest_score": "outlier models, tested synthetically",
    "novelty_knn": "outlier models, tested synthetically",
    "is_outlier": "outlier models, tested synthetically",
    "jerk_rms": "never scored; covered by the jitter direction test",
    "psd_lf_hf": "never scored; covered by the jitter direction test",
    "idle_frac": "never scored; context only",
    "timestamp_dev": "info only",
}
