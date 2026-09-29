"""Per-episode detail for the inspector: traces, speed, gripper timeline and flagged spans.

Computed on demand for one episode, never stored. Spans are the same events the
metrics score (idle ends, the longest pause, the roughest window, acceleration
spikes, regrasp recoveries), so the inspector shows where a score comes from.
"""

import numpy as np

from . import activity
from .metrics import gripper as gripper_metrics
from .metrics import motion, tracking
from .signals import (
    all_arm_speed,
    arm_speeds,
    close_and_open_events,
    gripper_signals,
    normalized,
    arm_dims,
)
from . import smoothness

MAX_POINTS = 400


def _stride(n, max_points=MAX_POINTS):
    return max(1, int(np.ceil(n / max_points)))


def _round(a):
    return np.round(np.asarray(a, dtype=np.float64), 4).tolist()


def flagged_spans(ep):
    """Flagged intervals as ``{start_s, end_s, label, kind}`` dicts, in seconds."""
    spans = []
    fps = ep.fps

    speed = all_arm_speed(ep)
    if speed is not None and len(speed) >= 8:
        filtered = activity.filtered_speed(speed, fps)
        thr = activity.idle_threshold(filtered)
        lead = activity.leading_idle_samples(filtered, thr)
        trail = activity.trailing_idle_samples(filtered, thr)
        n = len(filtered)
        if lead >= fps * 0.5:
            spans.append({"start_s": 0.0, "end_s": lead / fps, "label": "idle at start", "kind": "idle"})
        if trail >= fps * 0.5 and n - trail > lead:
            spans.append({"start_s": (n - trail) / fps, "end_s": n / fps, "label": "idle at end", "kind": "idle"})
        recoveries = [(c1 - int(fps), c2 + int(fps)) for c1, _, c2 in gripper_metrics.recovery_events(ep)]
        interior = [
            (a, b)
            for a, b in activity.idle_runs(filtered, thr)
            if a > 0 and b < n and not any(a < r1 and b > r0 for r0, r1 in recoveries)
        ]
        if interior:
            a, b = max(interior, key=lambda r: r[1] - r[0])
            if b - a >= fps * 0.5:
                spans.append({"start_s": a / fps, "end_s": b / fps, "label": "longest pause", "kind": "pause"})

    # the roughest smoothness window per arm
    fc = activity.lowpass_cutoff(fps)
    for signal, sp in arm_speeds(ep).items():
        best = None
        for a, b in motion.window_spans(len(sp), fps):
            v = smoothness.sparc(sp[a:b], fps, fc=fc)
            if np.isfinite(v) and (best is None or v < best[0]):
                best = (v, a, b)
        if best is not None:
            spans.append(
                {"start_s": best[1] / fps, "end_s": best[2] / fps, "label": "roughest window (%s)" % signal, "kind": "rough"}
            )

    # acceleration spikes, clustered
    if ep.state is not None and len(ep.state) >= 8 and arm_dims(ep):
        dims = arm_dims(ep)
        x = normalized(ep.state, ep.state_range)[:, dims]
        acc = np.abs(np.diff(x, n=2, axis=0)) * fps**2
        med = np.median(acc, axis=0)
        mad = np.median(np.abs(acc - med), axis=0)
        thr = np.maximum(med + tracking.SPIKE_MAD_K * 1.4826 * mad, tracking.SPIKE_FLOOR)
        frames = np.flatnonzero(np.any(acc > thr, axis=1))
        if 0 < len(frames) < 0.3 * len(acc):  # a noisy sensor is not a set of jolts
            start = prev = frames[0]
            clusters = []
            for f in frames[1:]:
                if f - prev > 3:
                    clusters.append((start, prev))
                    start = f
                prev = f
            clusters.append((start, prev))
            for a, b in clusters[:12]:
                spans.append({"start_s": (a + 1) / fps, "end_s": (b + 2) / fps, "label": "acceleration spike", "kind": "spike"})

    for c1, reopen, c2 in gripper_metrics.recovery_events(ep):
        spans.append({"start_s": c1 / fps, "end_s": c2 / fps, "label": "regrasp recovery", "kind": "recovery"})
    return sorted(spans, key=lambda s: s["start_s"])


def episode_detail(ep, max_points=MAX_POINTS):
    """Arrays and spans for one episode, subsampled to at most `max_points` samples."""
    n = ep.length
    stride = _stride(n, max_points)
    idx = np.arange(0, n, stride)
    t = idx / ep.fps

    out = {
        "fps": ep.fps,
        "length": n,
        "duration_s": n / ep.fps,
        "stride": stride,
        "t": _round(t),
        "joint_names": list(ep.action_names),
        "action": None,
        "state": None,
        "speed": None,
        "idle_threshold": None,
        "grippers": {},
        "spans": [],
    }
    if ep.action is not None:
        out["action"] = _round(normalized(ep.action, ep.action_range)[idx])
    if ep.state is not None:
        out["state"] = _round(normalized(ep.state, ep.state_range)[idx])

    speed = all_arm_speed(ep)
    if speed is not None and len(speed) >= 8:
        filtered = activity.filtered_speed(speed, ep.fps)
        sidx = np.arange(0, len(filtered), stride)
        out["speed"] = {"t": _round((sidx + 1) / ep.fps), "v": _round(filtered[sidx])}
        out["idle_threshold"] = float(activity.idle_threshold(filtered))

    open_is = ep.assumptions.get("gripper_open_is")
    for side, unit in gripper_signals(ep).items():
        events = []
        if open_is:
            closes, opens = close_and_open_events(unit, open_is)
            events = sorted([{"t": c / ep.fps, "kind": "close"} for c in closes] + [{"t": o / ep.fps, "kind": "open"} for o in opens], key=lambda e: e["t"])
        out["grippers"][side] = {"t": _round(t), "v": _round(unit[idx]), "events": events}

    out["spans"] = flagged_spans(ep)
    return out


def thumbnail_times(ep, spans, max_times=4):
    """Times (seconds) worth showing: the middle of the flagged spans, then the start."""
    priority = {"spike": 0, "rough": 1, "pause": 2, "recovery": 3, "idle": 4}
    times = [
        (s["start_s"] + s["end_s"]) / 2
        for s in sorted(spans, key=lambda s: priority.get(s["kind"], 9))
    ]
    times.append(min(0.2, ep.length / ep.fps))
    times.append(ep.length / ep.fps / 2)
    picked = []
    for t in times:
        t = round(float(t), 1)
        if all(abs(t - p) >= 0.5 for p in picked):
            picked.append(t)
        if len(picked) >= max_times:
            break
    return sorted(picked)
