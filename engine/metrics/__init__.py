"""The metric table.

A metric is a plain function ``fn(episode) -> {signal: MV}`` plus one entry in
:data:`METRICS`. ``signal`` is a per-arm or per-camera slug, or ``""`` for an
episode-level value. To add a metric: write the function, add an entry with
:func:`spec`, and (to chart it) follow EXTENDING.md.

:func:`spec` keys:

- ``fn``: the metric function
- ``family``: the family it belongs to (motion, time, tracking, gripper,
  consistency, camera, integrity, language, outliers)
- ``group``: the profile group it votes in, or None if it is never scored
- ``requires``: arrays the episode must have (``"action"``, ``"state"``); the
  metric is skipped when one is missing
- ``kind``: ``"value"`` (a number, z-scored), ``"signed_z"`` (a number turned
  into a signed z within its group, scored as ``abs``) or ``"flag"`` (0 or 1,
  never z-scored)
- ``higher_is_worse``: the metric's raw direction
- ``per_signal``: whether it returns one value per arm or camera
- ``opt_in``: off unless the user turns it on
- ``batch``: True if ``fn(raws, label_of)`` compares an episode with its peers and
  returns ``{sample_id: {signal: MV}}`` instead of taking one episode
- ``check``: for integrity metrics, ``("fail" | "warn", above)``: the verdict
  gets that level when the value exceeds ``above``. None means info only
- ``description``: one plain-language sentence for tooltips
"""

from . import camera, consistency, gripper, integrity, language, motion, time_metrics, tracking


def spec(
    fn,
    family,
    description,
    group=None,
    requires=(),
    kind="value",
    higher_is_worse=True,
    per_signal=False,
    opt_in=False,
    batch=False,
    check=None,
):
    """Builds one metric entry. ``scored`` is derived: a metric votes iff it has a group."""
    return {
        "fn": fn,
        "family": family,
        "group": group,
        "requires": tuple(requires),
        "kind": kind,
        "higher_is_worse": higher_is_worse,
        "per_signal": per_signal,
        "scored": group is not None,
        "opt_in": opt_in,
        "batch": batch,
        "check": check,
        "description": description,
    }


METRICS = {
    # -- motion ----------------------------------------------------------
    "sparc": spec(
        motion.sparc, "motion", group="motion", requires=("action",), higher_is_worse=False, per_signal=True,
        description=(
            "Spectral arc length of the speed profile. Closer to 0 is smoother; very "
            "negative means fragmented, hesitant motion."
        ),
    ),
    "ldlj": spec(
        motion.ldlj, "motion", group="motion", requires=("action",), higher_is_worse=False, per_signal=True,
        description=(
            "Log dimensionless jerk. Closer to 0 is smoother. Noisier and "
            "duration-sensitive, so it is de-weighted inside the motion group."
        ),
    ),
    "sparc_phase": spec(
        motion.sparc_phase, "motion", group="motion", requires=("action",), higher_is_worse=False,
        per_signal=True, opt_in=True,
        description=(
            "SPARC inside each gripper-delimited phase, rolled up by median, so grasps "
            "and releases are not penalized as roughness. Needs a named gripper dimension."
        ),
    ),
    "jerk_rms": spec(
        motion.jerk_rms, "motion", requires=("action",), per_signal=True, opt_in=True,
        description="RMS jerk after a low-pass filter. Lower is smoother. Noise-sensitive; never scored.",
    ),
    "psd_lf_hf": spec(
        motion.psd_lf_hf, "motion", requires=("action",), higher_is_worse=False, per_signal=True, opt_in=True,
        description=(
            "Log ratio of low- to high-frequency power. Higher is smoother. Unreliable at "
            "LeRobot frame rates; never scored."
        ),
    ),
    # -- time efficiency -------------------------------------------------
    "idle_lead_s": spec(
        time_metrics.idle_lead_s, "time", group="time", requires=("action",),
        description="Seconds of stillness before the first movement. High means dead time at the start.",
    ),
    "idle_trail_s": spec(
        time_metrics.idle_trail_s, "time", group="time", requires=("action",),
        description=(
            "Seconds of stillness at the end. Low weight, because fixed-duration "
            "recordings end this way."
        ),
    ),
    "longest_pause_s": spec(
        time_metrics.longest_pause_s, "time", group="time", requires=("action",),
        description="Longest idle run that touches neither end. High means a mid-task stall or hesitation.",
    ),
    "end_motion_ratio": spec(
        time_metrics.end_motion_ratio, "time", group="time", requires=("action",),
        description=(
            "Speed in the last second relative to moving speed. High means the episode "
            "probably ended mid-motion (cut off). Low weight."
        ),
    ),
    "idle_frac": spec(
        time_metrics.idle_frac, "time", requires=("action",),
        description="Fraction of frames below the idle speed. Context only; never scored.",
    ),
    "length_z": spec(
        time_metrics.duration_s, "time", group="time", kind="signed_z",
        description=(
            "Robust z of episode duration within its normalization group. Positive is "
            "unusually long (struggle), negative unusually short (abandoned). Scored as |z|."
        ),
    ),
    "path_length_z": spec(
        time_metrics.path_length, "time", requires=("action",), kind="signed_z", opt_in=True,
        description="Robust z of total joint-space path length. Correlated with length; shown for review only.",
    ),
    # -- tracking and contact -------------------------------------------
    "track_lag_ms": spec(
        tracking.track_lag_ms, "tracking", requires=("action", "state"), kind="signed_z",
        description=(
            "Delay between the commanded action and the achieved state. Mostly a hardware "
            "property, so only episodes far from the dataset median stand out. Never scored."
        ),
    ),
    "track_resid": spec(
        tracking.track_resid, "tracking", group="tracking", requires=("action", "state"), per_signal=True,
        description=(
            "Normalized gap between action and state after lag alignment. High means the robot "
            "is not reaching what was commanded (slip, collision, stall)."
        ),
    ),
    "accel_spike_frac": spec(
        tracking.accel_spike_frac, "tracking", group="tracking", requires=("state",),
        description="Fraction of frames with a sudden acceleration spike. A proxy for collisions and contact.",
    ),
    "joint_limit_frac": spec(
        tracking.joint_limit_frac, "tracking", group="tracking", requires=("state",), opt_in=True,
        description=(
            "Fraction of frames near a joint limit (from stats.json min and max, a weak proxy). "
            "High means the operator worked near the robot's limits."
        ),
    ),
    # -- gripper ---------------------------------------------------------
    "gripper_flips_per_s": spec(
        gripper.gripper_flips_per_s, "gripper", group="gripper", requires=("action",), per_signal=True,
        description="Open/close transitions per second. High means chatter or a hesitant operator.",
    ),
    "missed_grasp_frac": spec(
        gripper.missed_grasp_frac, "gripper", group="gripper", requires=("action", "state"), per_signal=True,
        opt_in=True,
        description=(
            "Fraction of close commands where the gripper closed fully with no stall (nothing was "
            "held). Needs a known gripper convention. Heuristic."
        ),
    ),
    "recovery_count": spec(
        gripper.recovery_count, "gripper", requires=("action",), kind="flag",
        description=(
            "Detected regrasp recoveries (open then close again near the same place). Neutral or "
            "positive: recovery is never penalized. Never scored."
        ),
    ),
    # -- consistency -----------------------------------------------------
    "action_divergence": spec(
        consistency.action_divergence, "consistency", group="consistency", batch=True,
        description=(
            "How far the action is from what peer episodes did in a similar state. High means a "
            "different action in the same situation. Without images, read it as a review signal."
        ),
    ),
    "stats_leverage": spec(
        consistency.stats_leverage, "consistency", batch=True, kind="flag",
        description=(
            "Feature dimensions this episode alone stretches beyond its source's range. High means "
            "it distorts the normalization statistics every training run will use. Never scored."
        ),
    ),
    # -- camera (decode video; off unless the user turns them on) ---------
    "blur": spec(
        camera.blur, "camera", group="camera", higher_is_worse=False, per_signal=True, opt_in=True,
        description=(
            "Sharpness of the camera image: a low percentile of the Laplacian variance over sampled "
            "frames, on a log scale. Lower is blurrier. Compared with the same camera in other episodes."
        ),
    ),
    "exposure_err": spec(
        camera.exposure_err, "camera", group="camera", per_signal=True, opt_in=True,
        description=(
            "How far the mean brightness is from mid-gray, from 0 (mid-gray) to 1 (black or white). "
            "High means too dark or too bright."
        ),
    ),
    "clipped_frac": spec(
        camera.clipped_frac, "camera", group="camera", per_signal=True, opt_in=True,
        description="Share of pixels at pure black or pure white. High means blown highlights or crushed shadows.",
    ),
    "frozen_frac": spec(
        camera.frozen_frac, "camera", group="camera", requires=("state",), per_signal=True, opt_in=True,
        description=(
            "Share of short bursts, while the robot is moving, in which the frames are nearly "
            "identical. High means a frozen or dropped camera feed."
        ),
    ),
    "video_action_lag_ms": spec(
        camera.video_action_lag_ms, "camera", group="camera", requires=("action",), per_signal=True, opt_in=True,
        description=(
            "Offset in milliseconds between the motion in the video and the commanded motion. Near 0 is "
            "in sync. Relative to other episodes of the same camera, and not reported when the video "
            "does not track the action."
        ),
    ),
    # -- integrity -------------------------------------------------------
    "frame_gaps": spec(
        integrity.frame_gaps, "integrity", kind="flag", check=("fail", 0),
        description="Missing, duplicated or out-of-order frame indexes. Any nonzero count is a broken episode.",
    ),
    "timestamp_dev": spec(
        integrity.timestamp_dev, "integrity", kind="flag",
        description=(
            "Largest deviation of consecutive timestamps from 1/fps, in seconds. Info only: native "
            "timestamps are derived from fps."
        ),
    ),
    "length_mismatch": spec(
        integrity.length_mismatch, "integrity", kind="flag", check=("fail", 0),
        description="Episode length in the metadata versus the rows actually found. Any difference is a fault.",
    ),
    "video_window_mismatch": spec(
        integrity.video_window_mismatch, "integrity", kind="flag", check=("fail", integrity.VIDEO_TOLERANCE_FRAMES),
        description=(
            "Largest gap, in frames, between a camera's video window and the episode length. "
            "Cannot see a constant offset; video_action_lag_ms does."
        ),
    ),
    "nonfinite_values": spec(
        integrity.nonfinite_values, "integrity", kind="flag", check=("fail", 0),
        description="NaN or infinite values in action or state. Any nonzero count breaks training.",
    ),
    "too_short": spec(
        integrity.too_short, "integrity", kind="flag", check=("fail", 0),
        description="Under 10 frames or 1 second: a one-frame or near-empty episode.",
    ),
    "schema_mismatch": spec(
        integrity.schema_mismatch, "integrity", kind="flag", batch=True, check=("warn", 0),
        description=(
            "The source's fps, action or state size, or cameras differ from the most common one in "
            "the view. In a deliberately mixed dataset most episodes differ."
        ),
    ),
    # -- outliers (computed by score.finalize after the z-scores exist) --
    "iforest_score": spec(
        None, "outliers", kind="flag",
        description=(
            "Isolation-forest anomaly score over the episode's metric z-scores, fit within its group. "
            "Higher is more unusual. It can rank defective episodes as normal, so it is information only."
        ),
    ),
    "novelty_knn": spec(
        None, "outliers", kind="flag",
        description=(
            "Mean distance to the 5 nearest episodes of the same group in metric-z space. Two-sided: "
            "very high means unlike the rest, very low means near-duplicate. Information only."
        ),
    ),
    "is_outlier": spec(
        None, "outliers", kind="flag",
        description="Either outlier score at z >= 2. Coarse 'worth a second look'. Unusual does not mean bad.",
    ),
    # -- language --------------------------------------------------------
    "task_missing": spec(
        language.task_missing, "language", kind="flag",
        description="The task string is empty or absent.",
    ),
    "task_generic": spec(
        language.task_generic, "language", kind="flag",
        description=(
            "The task string is a placeholder, has fewer than 3 words, or names no action. "
            "The instruction is too weak to say what the demonstration shows."
        ),
    ),
}


def default_metric_names():
    """Metrics that run unless the user turns them off."""
    return [name for name, s in METRICS.items() if not s["opt_in"]]


def required_arrays_present(metric_spec, episode):
    """Whether the episode has every array the metric needs."""
    for name in metric_spec["requires"]:
        if getattr(episode, name, None) is None:
            return False
    return True
