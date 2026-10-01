"""Sparse video decoding for the camera metrics.

LeRobot v3 stores many episodes back to back in one MP4, so every time here is
*file-relative* seconds and an episode is a ``VideoWindow`` slice of the file
(``from_timestamp`` to ``to_timestamp``). Nothing decodes a whole file: each
function seeks to the nearest earlier keyframe and decodes forward from there.

Three access patterns, one per kind of metric:

- ``decode_frames``: a handful of frames at given times (blur, exposure, clipping).
- ``decode_burst``: a few consecutive frames from one time (frozen-feed check).
- ``iter_frames``: every frame of the window in order, downscaled (video-action lag).

Frames come back as ``(time, array)`` pairs. AV1 (libdav1d) and H.264 both work
through PyAV. A video that cannot be opened or decoded raises ``DecodeError``, so a
caller can mark that camera as not computed instead of failing the episode.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# A frame whose time is within this of the requested time counts as that frame.
_TOLERANCE_S = 1e-3


class DecodeError(Exception):
    """A video could not be opened or decoded."""


def sample_times(window, n=12):
    """`n` evenly spaced file-relative times inside the episode's video window.

    Times sit at the midpoint of `n` equal slices, so the very first and very last
    frames (often black or partially written) are not sampled.
    """
    duration = window.to_timestamp - window.from_timestamp
    if duration <= 0 or n < 1:
        return []
    return [window.from_timestamp + duration * (i + 0.5) / n for i in range(n)]


def decode_frames(window, times, gray=True, max_side=None):
    """Decode the frame at (or just after) each requested time.

    Args:
        window: the episode's ``VideoWindow``
        times: file-relative seconds. Need not be sorted, and are clamped into the window
        gray (True): ``uint8`` ``(H, W)`` arrays if True, else ``uint8`` ``(H, W, 3)`` RGB
        max_side (None): downscale so the longer side is at most this many pixels.
            None keeps the native size, which a sharpness metric needs

    Returns:
        A list of ``(time, array)``, one per requested time that could be decoded, in
        time order. ``time`` is the time of the frame actually decoded.

    Raises:
        DecodeError: the file cannot be opened, or none of the times could be decoded
    """
    times = sorted(_clamp(t, window) for t in times)
    if not times:
        return []
    out = []
    try:
        container, stream = _open(window.path)
        with container:
            for t in times:
                frame = _frame_at(container, stream, t)
                if frame is not None:
                    out.append((float(frame.time), _to_array(frame, gray, max_side)))
    except DecodeError:
        raise
    except Exception as e:  # noqa: BLE001 - PyAV raises many exception types
        raise DecodeError("could not decode %s: %s" % (window.path, e)) from e
    if not out:
        raise DecodeError("no frame could be decoded from %s at the requested times" % window.path)
    return out


def decode_burst(window, start, n_frames, gray=True, max_side=None):
    """Decode `n_frames` consecutive frames beginning at (or just after) `start`.

    The burst stops early at the end of the window. Raises ``DecodeError`` as
    ``decode_frames`` does.
    """
    return list(iter_frames(window, start=start, max_frames=n_frames, gray=gray, max_side=max_side))


def iter_frames(window, start=None, stop=None, step=1, max_frames=None, gray=True, max_side=None):
    """Yield every `step`-th frame of the window in order, as ``(time, array)``.

    Every frame is decoded (video cannot skip inside a group of pictures), but only
    the frames yielded are converted to arrays, which is the expensive part.

    Args:
        window: the episode's ``VideoWindow``
        start (None): first file-relative time to yield. Defaults to the window start
        stop (None): yield frames before this time. Defaults to the window end
        step (1): yield one frame in every `step`
        max_frames (None): stop after yielding this many frames
        gray (True), max_side (None): as in ``decode_frames``

    Raises:
        DecodeError: the file cannot be opened, or a frame fails to decode
    """
    lo = window.from_timestamp if start is None else max(start, window.from_timestamp)
    hi = window.to_timestamp if stop is None else min(stop, window.to_timestamp)
    step = max(1, int(step))
    yielded = seen = 0
    try:
        container, stream = _open(window.path)
        with container:
            _seek(container, stream, lo)
            for frame in container.decode(stream):
                if frame.time is None or frame.time < lo - _TOLERANCE_S:
                    continue  # still before the window, between the keyframe and `lo`
                if frame.time >= hi - _TOLERANCE_S:
                    break
                seen += 1
                if (seen - 1) % step:
                    continue
                yield float(frame.time), _to_array(frame, gray, max_side)
                yielded += 1
                if max_frames is not None and yielded >= max_frames:
                    break
    except DecodeError:
        raise
    except Exception as e:  # noqa: BLE001
        raise DecodeError("could not decode %s: %s" % (window.path, e)) from e


def _clamp(t, window):
    return min(max(t, window.from_timestamp), window.to_timestamp)


def _open(path):
    import av

    try:
        container = av.open(path)
    except Exception as e:  # noqa: BLE001
        raise DecodeError("could not open %s: %s" % (path, e)) from e
    if not container.streams.video:
        container.close()
        raise DecodeError("%s has no video stream" % path)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    if stream.time_base is None:
        container.close()
        raise DecodeError("%s has no usable time base" % path)
    return container, stream


def _seek(container, stream, seconds):
    container.seek(max(0, int(seconds / stream.time_base)), stream=stream, backward=True, any_frame=False)


def _frame_at(container, stream, seconds):
    """The first frame at or after `seconds`, or the last frame decoded if the file ends first."""
    _seek(container, stream, seconds)
    chosen = None
    for frame in container.decode(stream):
        if frame.time is None:
            continue
        chosen = frame
        if frame.time >= seconds - _TOLERANCE_S:
            break
    return chosen


def _to_array(frame, gray, max_side):
    width, height = frame.width, frame.height
    if max_side is not None and max(width, height) > max_side:
        scale = max_side / float(max(width, height))
        width, height = max(1, int(round(width * scale))), max(1, int(round(height * scale)))
    fmt = "gray" if gray else "rgb24"
    return np.ascontiguousarray(frame.reformat(width=width, height=height, format=fmt).to_ndarray())
