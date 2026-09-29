"""Normalization groups: which episodes each score is measured against.

The ladder (see METRICS.md, Normalization):

1. Use the episode's task group when it has at least ``min_group`` episodes.
2. Otherwise pool over the whole view.

A view with fewer than ``min_group`` episodes overall cannot be normalized
robustly. It is still pooled, and callers read ``group_n < min_group`` as "low
confidence" and say so. (An earlier draft had a robot-type rung between these,
but it could never be reached: a robot group inside a view smaller than
``min_group`` is itself smaller than ``min_group``.)

Task strings are canonicalized first. An episode listing several tasks is
grouped by its canonicalized, sorted set of tasks.
"""

import re
from collections import Counter

DEFAULT_MIN_GROUP = 20
POOLED = "__pooled__"


def canonical_task(tasks):
    """A stable key for an episode's task list.

    Case, whitespace and punctuation are normalized. Several tasks become their
    sorted set joined with " | ".
    """
    if tasks is None:
        return ""
    if isinstance(tasks, str):
        tasks = [tasks]
    keys = sorted(
        {
            re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", str(t).lower())).strip()
            for t in tasks
            if str(t).strip()
        }
    )
    return " | ".join(k for k in keys if k)


def assign_groups(task_keys, min_group=DEFAULT_MIN_GROUP):
    """Assigns each episode a normalization group.

    Args:
        task_keys: one canonical task key per episode
        min_group: the smallest group that is normalized on its own

    Returns:
        a tuple ``(labels, basis)``: one group label per episode, and one basis
        per episode (``"task"`` or ``"pooled"``)
    """
    task_counts = Counter(task_keys)
    labels, basis = [], []
    for key in task_keys:
        if key and task_counts[key] >= min_group:
            labels.append("task:%s" % key)
            basis.append("task")
        else:
            labels.append(POOLED)
            basis.append("pooled")
    return labels, basis
