"""Profile groups, weights and aggregation.

A profile is a named set of groups. Correlated metrics share a group and a group
counts as one vote. Inside a group the value is the maximum of the weighted,
oriented member z-scores (triage semantics: any one bad metric matters). Across
groups the profile score is also the maximum (worst-of), so one failing group is
never diluted by the others. ``n_flags`` counts groups at or above the warn
threshold and ``driver`` names the group with the highest value.
"""

import numpy as np

from . import normalize

# group -> {metric: weight}.
GROUPS = {
    "motion": {"sparc": 1.0, "ldlj": 0.5, "sparc_phase": 1.0},
    "time": {
        "idle_lead_s": 1.0,
        "idle_trail_s": 0.5,
        "longest_pause_s": 1.0,
        "end_motion_ratio": 0.5,
        "length_z": 1.0,
    },
    "tracking": {"track_resid": 1.0, "accel_spike_frac": 1.0, "joint_limit_frac": 1.0},
    "gripper": {"gripper_flips_per_s": 1.0, "missed_grasp_frac": 1.0},
    "consistency": {"action_divergence": 1.0},
    "camera": {
        "blur": 1.0,
        "exposure_err": 1.0,
        "clipped_frac": 1.0,
        "frozen_frac": 1.0,
        "video_action_lag_ms": 1.0,
    },
}

PROFILES = {
    "policy": {
        "label": "Policy (ACT, Diffusion Policy)",
        "groups": ["motion", "time", "tracking", "gripper", "consistency", "camera"],
        "language_forces_review": False,
    },
    "vla": {
        "label": "VLA (instruction must match)",
        "groups": ["motion", "time", "tracking", "gripper", "consistency", "camera"],
        "language_forces_review": True,
    },
}

DEFAULT_PROFILE = "policy"


def group_values(metric_z, groups):
    """``{group: max weighted z}`` for the groups that have any metric present."""
    out = {}
    for group in groups:
        values = [
            weight * metric_z[m]
            for m, weight in GROUPS[group].items()
            if m in metric_z and not np.isnan(metric_z[m])
        ]
        if values:
            out[group] = max(values)
    return out


def score_profile(metric_z, profile, language_flagged=False):
    """Scores one episode under one profile.

    Args:
        metric_z: ``{metric: oriented z}`` for this episode (higher is worse)
        profile: a key of :data:`PROFILES`
        language_flagged: whether a language flag is set on this episode

    Returns:
        ``{"score", "n_flags", "driver", "verdict"}``. ``score`` is None and the
        verdict is ``"unknown"`` when no group could be computed
    """
    spec = PROFILES[profile]
    values = group_values(metric_z, spec["groups"])
    if not values:
        verdict = "warn" if (spec["language_forces_review"] and language_flagged) else "unknown"
        return {"score": None, "n_flags": 0, "driver": None, "verdict": verdict}

    driver = max(values, key=values.get)
    score = float(values[driver])
    n_flags = sum(1 for v in values.values() if v >= normalize.WARN_Z)
    verdict = normalize.severity(score) or "pass"
    if spec["language_forces_review"] and language_flagged and verdict == "pass":
        verdict = "warn"
    return {"score": score, "n_flags": n_flags, "driver": driver, "verdict": verdict}
