"""Camera metrics: sharpness, exposure, clipping, frozen feeds and video-action lag.

All five are pixel and signal arithmetic on a few decoded frames, with no model.
Each returns one value per camera (the signal is ``cam_<last part of the key>``),
so normalization is per (group, camera) and a wrist camera is never compared
with a top camera.

Decoding is the cost, so the metrics share it through ``episode.cache``: one
decode of 12 still frames feeds ``blur``, ``exposure_err`` and ``clipped_frac``,
a few short bursts feed ``frozen_frac``, and one dense pass feeds
``video_action_lag_ms``. Only small per-camera numbers are cached, never frames.

A camera whose video cannot be decoded is skipped. If no camera of an episode
decodes, the metric raises ``DecodeError`` and the run records why.
"""

import cv2
import numpy as np

from .. import decode
from ..activity import filtered_speed, idle_threshold
from ..decode import DecodeError, sample_times
from ..reader import slug
from ..signals import all_arm_speed, normalized
from .base import MV

STILL_FRAMES = 12  # frames sampled per video for blur, exposure and clipping
BLUR_MAX_SIDE = 480  # Laplacian variance depends on resolution, so measure at a fixed size
BLUR_PERCENTILE = 10  # a low percentile, so a stretch of blurry frames is not averaged away

FROZEN_BURSTS = 6  # short bursts sampled per video
FROZEN_BURST_FRAMES = 6
FROZEN_MAX_SIDE = 96
FROZEN_DIFF = 0.05  # largest mean gray-level change between frames below which a feed is frozen (duplicated frames read 0; a live static camera reads about 0.3 or more)
MOVING_ALPHA = 0.25  # the robot counts as moving above this share of its own moving speed

LAG_MAX_S = 1.0
LAG_MIN_CORR = 0.3  # below this the motion in the video does not track the action, so no lag is reported
LAG_MAX_SIDE = 96
LAG_SPAN_S = 30.0  # the central stretch of a long episode is enough to line the two up
MIN_LAG_SAMPLES = 20


# -- shared plumbing -------------------------------------------------------------


def camera_windows(ep):
    """``{signal: VideoWindow}``, with the signal ``cam_<last part of the camera key>``."""
    out = {}
    for key, window in ep.videos.items():
        signal = "cam_%s" % slug(key.split(".")[-1])
        if signal in out:  # two cameras share a last part: fall back to the full key
            signal = "cam_%s" % slug(key)
        out[signal] = window
    return out


def _decoder(ep):
    return ep.decoder if ep.decoder is not None else decode


def _per_camera(ep, kind, compute, to_mv):
    """Runs `compute(window)` once per camera (cached under `kind`), then `to_mv(data)`.

    `compute` returns small numbers describing the video, or None when there is
    nothing to say. A DecodeError skips that camera, and is raised only when
    every camera failed.
    """
    out, errors = {}, []
    for signal, window in camera_windows(ep).items():
        key = (kind, signal)
        if key not in ep.cache:
            try:
                ep.cache[key] = compute(window)
            except DecodeError as e:
                ep.cache[key] = e
        data = ep.cache[key]
        if isinstance(data, DecodeError):
            errors.append(data)
        elif data is not None:
            mv = to_mv(data)
            if mv is not None:
                out[signal] = mv
    if not out and errors:
        raise errors[0]
    return out


# -- stills: blur, exposure, clipping ---------------------------------------------


def _laplacian_variance(gray):
    h, w = gray.shape
    scale = BLUR_MAX_SIDE / float(max(h, w))
    if scale < 1.0:
        gray = cv2.resize(gray, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _still_stats(ep, window):
    """Per-frame sharpness, exposure error and clipped share over sampled frames."""
    frames = _decoder(ep).decode_frames(window, sample_times(window, STILL_FRAMES), gray=True)
    sharp, exposure, clipped = [], [], []
    for _, img in frames:
        sharp.append(_laplacian_variance(img))
        exposure.append(abs(float(img.mean()) - 127.5) / 127.5)
        clipped.append(float(np.mean((img == 0) | (img == 255))))
    return {"sharp": np.array(sharp), "exposure": np.array(exposure), "clipped": np.array(clipped)}


def blur(ep):
    """Low-percentile Laplacian variance over sampled frames, on a log scale (lower is blurrier).

    Reported as ``log10(1 + variance)``: raw variance is heavy-tailed, so a blurry
    camera would never stand out from the long tail of sharp ones.
    """

    def to_mv(stats):
        low = float(np.percentile(stats["sharp"], BLUR_PERCENTILE))
        return MV(float(np.log10(1.0 + low)), worst=float(np.log10(1.0 + stats["sharp"].min())))

    return _per_camera(ep, "stills", lambda w: _still_stats(ep, w), to_mv)


def exposure_err(ep):
    """Distance of the mean luminance from mid-gray, as a share of the range (0 is mid-gray, 1 is black or white)."""
    return _per_camera(
        ep, "stills", lambda w: _still_stats(ep, w),
        lambda s: MV(float(np.median(s["exposure"])), worst=float(s["exposure"].max())),
    )


def clipped_frac(ep):
    """Share of pixels at 0 or 255 (blown highlights or crushed shadows), averaged over sampled frames."""
    return _per_camera(
        ep, "stills", lambda w: _still_stats(ep, w),
        lambda s: MV(float(s["clipped"].mean()), worst=float(s["clipped"].max())),
    )


# -- bursts: frozen feeds ----------------------------------------------------------


def _burst_diffs(ep, window):
    """``[(start_s, end_s, largest mean frame-to-frame change)]`` per sampled burst, times relative to the episode."""
    span = FROZEN_BURST_FRAMES / ep.fps
    out = []
    for center in sample_times(window, FROZEN_BURSTS):
        start = max(window.from_timestamp, min(center - span / 2.0, window.to_timestamp - 1.5 * span))
        frames = _decoder(ep).decode_burst(window, start, FROZEN_BURST_FRAMES, gray=True, max_side=FROZEN_MAX_SIDE)
        if len(frames) < 3:
            continue
        arrays = [f.astype(np.float32) for _, f in frames]
        largest = max(float(np.abs(arrays[i + 1] - arrays[i]).mean()) for i in range(len(arrays) - 1))
        out.append((frames[0][0] - window.from_timestamp, frames[-1][0] - window.from_timestamp, largest))
    return out or None


def _state_speed(ep):
    """Filtered speed of the range-normalized state per frame, or None when it cannot be computed."""
    if ep.state is None or len(ep.state) < MIN_LAG_SAMPLES:
        return None
    x = normalized(ep.state, ep.state_range)
    return filtered_speed(np.linalg.norm(np.diff(x, axis=0), axis=1) * ep.fps, ep.fps)


def frozen_frac(ep):
    """Share of bursts, among those where the robot moves, whose frames are nearly identical.

    A static scene with a still robot is not flagged: a burst only counts when the
    state shows the robot moving. A camera that never sees a moving robot is
    reported as nothing to judge, not as frozen.
    """
    speed = _state_speed(ep)
    if speed is None:
        return {}
    threshold = idle_threshold(speed, alpha=MOVING_ALPHA)
    if threshold <= 0:
        return {}

    def to_mv(bursts):
        moving = frozen = 0
        for start, end, largest in bursts:
            seg = speed[int(round(start * ep.fps)) : int(round(end * ep.fps))]
            if len(seg) == 0 or np.median(seg) < threshold:
                continue
            moving += 1
            frozen += largest < FROZEN_DIFF
        return None if moving == 0 else MV(frozen / moving)

    return _per_camera(ep, "bursts", lambda w: _burst_diffs(ep, w), to_mv)


# -- dense pass: video-action lag ---------------------------------------------------


def _motion_energy(ep, window):
    """``(times relative to the episode, mean absolute change between consecutive frames)`` over the central stretch."""
    lo, hi = window.from_timestamp, window.to_timestamp
    if hi - lo > LAG_SPAN_S:
        mid = (lo + hi) / 2.0
        lo, hi = mid - LAG_SPAN_S / 2.0, mid + LAG_SPAN_S / 2.0
    times, energy, prev = [], [], None
    for t, img in _decoder(ep).iter_frames(window, start=lo, stop=hi, gray=True, max_side=LAG_MAX_SIDE):
        cur = img.astype(np.float32)
        if prev is not None:
            times.append(t - window.from_timestamp)
            energy.append(float(np.abs(cur - prev).mean()))
        prev = cur
    if len(times) < MIN_LAG_SAMPLES:
        return None
    return np.array(times), np.array(energy)


def best_lag(action_speed, video_energy, max_lag):
    """Lag in samples at which the video energy best tracks the action speed, with its correlation.

    Positive means the video lags the action. Refined to a fraction of a sample by
    fitting a parabola through the peak. Returns None when either signal is flat.
    """
    a, e = np.asarray(action_speed, dtype=np.float64), np.asarray(video_energy, dtype=np.float64)
    if a.std() < 1e-12 or e.std() < 1e-12:
        return None
    a, e = (a - a.mean()) / a.std(), (e - e.mean()) / e.std()
    n = len(a)
    lags = list(range(-max_lag, max_lag + 1))
    scores = np.array(
        [np.mean(a[: n - k] * e[k:]) if k >= 0 else np.mean(a[-k:] * e[: n + k]) for k in lags]
    )
    i = int(np.argmax(scores))
    lag = float(lags[i])
    if 0 < i < len(scores) - 1:
        denom = scores[i - 1] - 2.0 * scores[i] + scores[i + 1]
        if denom < 0:
            lag += float(np.clip(0.5 * (scores[i - 1] - scores[i + 1]) / denom, -0.5, 0.5))
    return lag, float(scores[i])


def video_action_lag_ms(ep):
    """How far, in ms, the motion in the video is offset from the commanded motion (absolute value).

    Cross-correlates the speed of the arm in the action with the frame-to-frame
    change in each camera. Near 0 means they are in sync. It is a value relative to
    peers: a lag every episode shares sets the baseline, so only the episodes that
    differ from it stand out. Reports nothing when the video does not track the
    action (correlation under 0.3), for example when the robot is out of view.
    """
    speed = all_arm_speed(ep)
    if speed is None or len(ep.timestamps) < MIN_LAG_SAMPLES:
        return {}
    speed_t = ep.timestamps[1:] - ep.timestamps[0]

    def to_mv(data):
        times, energy = data
        t0, t1 = max(times[0], speed_t[0]), min(times[-1], speed_t[-1])
        grid = np.arange(t0, t1, 1.0 / ep.fps)
        if len(grid) < MIN_LAG_SAMPLES:
            return None
        a = filtered_speed(np.interp(grid, speed_t, speed), ep.fps)
        e = filtered_speed(np.interp(grid, times, energy), ep.fps)
        found = best_lag(a, e, max_lag=min(int(round(LAG_MAX_S * ep.fps)), len(grid) // 3))
        if found is None or found[1] < LAG_MIN_CORR:
            return None
        return MV(abs(1000.0 * found[0] / ep.fps))

    return _per_camera(ep, "energy", lambda w: _motion_energy(ep, w), to_mv)
