"""Integrity flags: defects that break training regardless of quality."""

import numpy as np

from .base import MV

VIDEO_TOLERANCE_FRAMES = 2.0
MIN_FRAMES = 10
MIN_SECONDS = 1.0
MIXED_SHARE = 0.5  # below this share for the most common schema, the dataset counts as mixed


def frame_gaps(ep):
    """Count of missing, duplicated and out-of-order frame indexes.

    Frame indexes should run 0..T-1 with no repeats, in order. Any nonzero
    count means a broken episode.
    """
    fi = ep.raw_frame_index
    if fi is None or len(fi) == 0:
        return {}
    unique = np.unique(fi)
    duplicates = len(fi) - len(unique)
    missing = int(unique.max() - unique.min() + 1) - len(unique)
    out_of_order = int(np.sum(np.diff(fi) < 0))
    offset = int(unique.min() != 0)
    count = duplicates + missing + out_of_order + offset
    parts = []
    if duplicates:
        parts.append("%d duplicated" % duplicates)
    if missing:
        parts.append("%d missing" % missing)
    if out_of_order:
        parts.append("%d out of order" % out_of_order)
    if offset:
        parts.append("does not start at 0")
    return {"": MV(float(count), note=", ".join(parts) or None)}


def timestamp_dev(ep):
    """Largest deviation of consecutive timestamps from ``1/fps``, in seconds.

    Info only. Native recordings derive timestamps from ``fps``, so this is
    usually about 0 and does not affect the verdict. Video timing is checked by
    `video_window_mismatch`.
    """
    ts = ep.raw_timestamps
    if ts is None or len(ts) < 2 or ep.fps <= 0:
        return {}
    return {"": MV(float(np.max(np.abs(np.diff(ts) - 1.0 / ep.fps))))}


def length_mismatch(ep):
    """Difference between the episode's promised length and the rows found."""
    return {"": MV(float(abs(ep.length - ep.expected_length)))}


def video_window_mismatch(ep):
    """Largest gap, in frames, between a camera window and the episode length.

    Compares each camera's ``to_timestamp - from_timestamp`` (file-relative in
    v3) with ``length / fps``. A gap of more than a couple of frames means video
    and actions differ in length. It cannot see a constant offset, which
    `video_action_lag_ms` is for.
    """
    if not ep.videos or ep.fps <= 0:
        return {}
    expected = ep.length / ep.fps
    worst = max(abs((w.to_timestamp - w.from_timestamp) - expected) for w in ep.videos.values())
    cam = max(ep.videos, key=lambda c: abs((ep.videos[c].to_timestamp - ep.videos[c].from_timestamp) - expected))
    frames = float(worst * ep.fps)
    return {"": MV(frames, note="%s is %.1f frames off" % (cam.split(".")[-1], frames) if frames > VIDEO_TOLERANCE_FRAMES else None)}


def nonfinite_values(ep):
    """Count of NaN or infinite values in action and state. Any nonzero count breaks training."""
    arrays = [a for a in (ep.action, ep.state) if a is not None]
    if not arrays:
        return {}
    return {"": MV(float(sum(int(np.sum(~np.isfinite(a))) for a in arrays)))}


def too_short(ep):
    """Flags an episode under 10 frames or 1 second (one-frame and near-empty episodes)."""
    if ep.fps <= 0:
        return {}
    short = ep.length < MIN_FRAMES or ep.length / ep.fps < MIN_SECONDS
    return {"": MV(1.0 if short else 0.0, note="%d frames" % ep.length if short else None)}


def schema_mismatch(raws, label_of):
    """Flags episodes whose source schema differs from the most common one.

    A batch metric: it compares each episode's schema signature (fps, action and
    state size, camera keys) against the most common signature in the view. When
    no schema covers at least half the episodes the dataset is deliberately mixed
    and nothing is flagged; the note says so.
    """
    from collections import Counter

    sigs = {sid: r.signature for sid, r in raws.items() if r.signature}
    if not sigs:
        return {}
    frozen = {sid: tuple(sorted(sig.items())) for sid, sig in sigs.items()}
    counts = Counter(frozen.values())
    common, common_n = counts.most_common(1)[0]
    majority = dict(common)
    mixed = common_n / len(sigs) < MIXED_SHARE
    out = {}
    for sid, sig in sigs.items():
        diffs = [k for k in sorted(majority) if sig.get(k) != majority[k]]
        if mixed:
            # No schema dominates: this is a deliberately mixed dataset, so a
            # difference is not a defect. Say so instead of flagging everything.
            note = "mixed dataset: %d distinct schemas" % len(counts) if diffs else None
            out[sid] = {"": MV(0.0, note=note)}
        else:
            out[sid] = {"": MV(1.0 if diffs else 0.0, note=("differs in " + ", ".join(diffs)) if diffs else None)}
    return out
