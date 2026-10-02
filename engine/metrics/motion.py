"""Motion smoothness metrics, per arm or hand signal, on fixed-length windows."""

import numpy as np

from .. import smoothness
from ..activity import lowpass_cutoff
from ..signals import (
    arm_signal_for_gripper,
    arm_speeds,
    gripper_signals,
    gripper_transitions,
)
from .base import MV

WINDOW_S = 3.0
OVERLAP = 0.5
MIN_WINDOW_SAMPLES = 16
WORST_PERCENTILE = 5  # the bad tail for a "closer to 0 is smoother" metric


def window_spans(n, fps):
    """``(start, end)`` sample indexes of the complete fixed-length windows over `n` samples.

    An episode shorter than one window becomes a single window if it has enough
    samples (SPARC is duration-invariant, so this stays comparable).
    """
    win = max(MIN_WINDOW_SAMPLES, int(round(WINDOW_S * fps)))
    if n < win:
        return [(0, n)] if n >= MIN_WINDOW_SAMPLES else []
    step = max(1, int(win * (1 - OVERLAP)))
    return [(i, i + win) for i in range(0, n - win + 1, step)]


def windows(speed, fps):
    """The speed profile cut into the windows of :func:`window_spans`."""
    return [speed[a:b] for a, b in window_spans(len(speed), fps)]


def _windowed(ep, fn, worst_percentile=WORST_PERCENTILE):
    """Median and bad-tail value of `fn` over each arm signal's windows."""
    out = {}
    for signal, speed in arm_speeds(ep).items():
        values = [fn(w, ep.fps) for w in windows(speed, ep.fps)]
        values = np.array([v for v in values if np.isfinite(v)])
        if len(values) == 0:
            continue
        out[signal] = MV(
            value=float(np.median(values)),
            worst=float(np.percentile(values, worst_percentile)),
        )
    return out


def sparc(ep):
    """Spectral arc length per arm signal (closer to 0 is smoother)."""
    fc = lowpass_cutoff(ep.fps)
    return _windowed(ep, lambda w, fps: smoothness.sparc(w, fps, fc=fc))


def ldlj(ep):
    """Log dimensionless jerk per arm signal (closer to 0 is smoother).

    Duration-sensitive, so it runs on fixed-length windows and is de-weighted
    inside the motion group.
    """
    return _windowed(ep, smoothness.ldlj)


def jerk_rms(ep):
    """RMS jerk per arm signal. Opt-in and never scored."""
    cutoff = min(10.0, lowpass_cutoff(ep.fps))
    return _windowed(
        ep, lambda w, fps: smoothness.jerk_rms(w, fps, cutoff_hz=cutoff), worst_percentile=95
    )


def psd_lf_hf(ep):
    """Low/high frequency power ratio per arm signal. Opt-in and never scored."""
    return _windowed(ep, lambda w, fps: smoothness.psd_lf_hf(w, fps))


def sparc_phase(ep):
    """SPARC inside each gripper-delimited phase, rolled up by median.

    A grasp or release is an abrupt event that global smoothness metrics read as
    roughness. Splitting the episode at gripper open/close transitions and
    scoring each phase separately stops contact from being penalized as
    jitter (the idea behind RINSE's contact-aware TED metric, adapted to SPARC).

    Needs a joint group marked Gripper, and the arm it belongs to (stated, or the
    only arm). Phases shorter than the minimum sample count are skipped.
    """
    out = {}
    speeds = arm_speeds(ep, roles=("arm",))
    fc = lowpass_cutoff(ep.fps)
    for side, unit in gripper_signals(ep).items():
        signal = arm_signal_for_gripper(ep, side)
        speed = speeds.get(signal) if signal else None
        if speed is None:
            continue
        bounds = [0] + [i for i in gripper_transitions(unit) if 0 < i < len(speed)] + [len(speed)]
        values = []
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b - a < MIN_WINDOW_SAMPLES:
                continue
            v = smoothness.sparc(speed[a:b], ep.fps, fc=fc)
            if np.isfinite(v):
                values.append(v)
        if values:
            out[signal] = MV(
                value=float(np.median(values)),
                worst=float(np.percentile(values, WORST_PERCENTILE)),
            )
    return out
