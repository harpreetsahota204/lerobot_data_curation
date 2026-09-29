"""Idle detection on a speed profile. Ported from ``demo_quality_scorer.engine.activity``.

The threshold is a fraction (``alpha``) of the episode's own moving speed, so it
needs no units and adapts to each episode.
"""

import numpy as np

from .smoothness import lowpass_filtfilt

IDLE_ALPHA_DEFAULT = 0.05
IDLE_REFERENCE_PERCENTILE = 90


def lowpass_cutoff(fps):
    """`min(10 Hz, 0.4 x fps)`: never analyze near Nyquist."""
    return min(10.0, 0.4 * fps)


def _moving_speed_reference(filtered):
    """An episode's typical speed while it is moving.

    A high percentile is only a provisional split. The reference is the median
    of everything above half of it, which pulls the reference out of the idle
    mass for any episode that spends a few percent of its length moving.
    """
    provisional = float(np.percentile(filtered, IDLE_REFERENCE_PERCENTILE))
    moving = filtered[filtered > provisional / 2.0]
    if len(moving) == 0:
        return provisional
    return float(np.median(moving))


def filtered_speed(speed, fps):
    speed = np.asarray(speed, dtype=np.float64)
    return np.abs(lowpass_filtfilt(speed, lowpass_cutoff(fps), fps))


def idle_threshold(filtered, alpha=IDLE_ALPHA_DEFAULT):
    return float(alpha * _moving_speed_reference(filtered))


def leading_idle_samples(filtered, threshold):
    """Samples before the first one at or above the threshold (all, if none)."""
    moving = np.flatnonzero(filtered >= threshold) if threshold > 0 else np.flatnonzero(filtered > 0)
    return int(moving[0]) if len(moving) else len(filtered)


def trailing_idle_samples(filtered, threshold):
    """Consecutive idle samples at the end (all of them, if the episode never moves)."""
    active = filtered >= threshold if threshold > 0 else filtered > 0
    moving = np.flatnonzero(active)
    return len(filtered) - 1 - int(moving[-1]) if len(moving) else len(filtered)


def idle_runs(filtered, threshold):
    """Runs of idle samples as ``(start, end)`` index pairs, end exclusive."""
    idle = filtered < threshold if threshold > 0 else filtered <= 0
    runs, start = [], None
    for i, flag in enumerate(idle):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(idle)))
    return runs


def end_motion_ratio(filtered, fps, window_s=1.0):
    """Median speed over the last `window_s` relative to the episode's moving speed."""
    n = max(1, int(round(window_s * fps)))
    reference = _moving_speed_reference(filtered)
    if reference <= 0:
        return float("nan")
    return float(np.median(filtered[-n:]) / reference)
