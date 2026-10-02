"""Decodes single frames from a (shared) MP4 for thumbnails.

Sparse decoding only: seek near the requested time and decode forward to the
first frame at or after it. Works for AV1 (libdav1d) and H.264 through PyAV.
"""

import logging

import cv2

logger = logging.getLogger(__name__)


def frame_jpeg(path, seconds, width=192, quality=70):
    """A JPEG of the frame at `seconds` (file-relative), or None if it cannot be decoded.

    Args:
        path: the MP4 path
        seconds: time within the file
        width: output width in pixels; height keeps the aspect ratio
        quality: JPEG quality
    """
    import av

    try:
        with av.open(path) as container:
            stream = container.streams.video[0]
            if stream.time_base is None:
                return None
            container.seek(max(0, int(seconds / stream.time_base)), stream=stream, backward=True, any_frame=False)
            chosen = None
            for frame in container.decode(stream):
                if frame.time is None:
                    continue
                chosen = frame
                if frame.time >= seconds - 1e-3:
                    break
            if chosen is None:
                return None
            rgb = chosen.to_ndarray(format="rgb24")
    except Exception as e:  # noqa: BLE001 - a thumbnail must never break the inspector
        logger.warning("Could not decode %s at %.2fs: %s", path, seconds, e)
        return None

    h, w = rgb.shape[:2]
    if w != width:
        rgb = cv2.resize(rgb, (width, max(1, int(round(h * width / w)))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else None
