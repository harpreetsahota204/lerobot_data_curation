"""Facts about a dataset and its balance. Nothing here is inferred.

What the user must decide (which arrays are state and action, what the action means,
which dimensions are an arm, a gripper or a hand, which cameras to score) is picked in
the compute form and stored as picks (see ``engine/picks.py``). This module only
reports what the dataset declares and counts what it contains:

- :func:`declared_facts`: the arrays and cameras a source's ``info.json`` declares
- :func:`balance`: episodes per canonical task and per source, with coverage warnings
"""

from collections import Counter

import numpy as np

from .reader import _vector_keys

UNDER_COVERED = 5  # tasks with fewer episodes than this are reported as under-covered
DOMINANT_SHARE = 0.5


def declared_facts(info):
    """What a source's ``info.json`` declares: robot, fps, vector arrays with their sizes, video cameras."""
    features = info.get("features", {})
    video = [k for k, v in features.items() if v.get("dtype") == "video"]
    return {
        "robot_type": info.get("robot_type"),
        "fps": info.get("fps"),
        "arrays": [(k, int(np.prod(features[k]["shape"]))) for k in _vector_keys(info)],
        # Only video features: the camera metrics decode MP4 windows, so a camera stored as images cannot be scored.
        "cameras": video,
        "camera_sizes": {k: tuple(features[k]["shape"][:2]) for k in video if len(features[k].get("shape") or []) >= 2},
        "image_cameras": [k for k, v in features.items() if v.get("dtype") == "image"],
    }


def balance(task_keys, source_ids):
    """Episodes per canonical task and per source, with coverage warnings."""
    tasks = Counter(task_keys)
    sources = Counter(source_ids)
    n = max(1, len(task_keys))
    return {
        "tasks": dict(tasks.most_common()),
        "sources": dict(sources.most_common()),
        "under_covered": sorted(t for t, c in tasks.items() if c < UNDER_COVERED),
        "dominant_sources": sorted(s for s, c in sources.items() if c / n > DOMINANT_SHARE),
    }
