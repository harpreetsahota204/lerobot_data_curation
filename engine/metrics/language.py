"""Static language checks on the task string (no model needed)."""

import re

from .base import MV

MIN_TOKENS = 3

PLACEHOLDERS = frozenset(
    {
        "task",
        "task desc",
        "task description",
        "desc",
        "description",
        "test",
        "none",
        "null",
        "n a",
        "na",
        "unknown",
        "default",
        "demo",
        "episode",
        "untitled",
    }
)

# Common manipulation verbs. The "no verb" rule is a heuristic and applies only
# to Latin-script strings; other scripts are checked for length and
# placeholders only, because whitespace tokenization does not fit them.
VERBS = frozenset(
    """
    pick place put move open close push pull stack insert grasp grab hold lift
    pour wipe fold sort hang take press turn rotate slide drop throw sweep clean
    fill empty cut screw unscrew connect plug unplug load unload transfer
    hand give bring carry lower raise flip stir scoop shake tap touch reach
    release align arrange organize remove set store wrap unwrap tie untie zip
    unzip button unbutton peel spread hit knock swap exchange collect gather
    dump tidy straighten rearrange lay align stow dispose serve wash dry
    """.split()
)


def _tokens(text):
    return re.findall(r"[^\W\d_]+", text.lower())


def _has_verb(tokens):
    for t in tokens:
        if t in VERBS:
            return True
        if t.endswith("ing") and len(t) > 4:
            stem = t[:-3]
            if stem in VERBS or stem + "e" in VERBS or (len(stem) > 1 and stem[:-1] in VERBS):
                return True
    return False


def task_generic(ep):
    """Flags a task string too weak to say what the demonstration shows.

    Flagged when the string has fewer than 3 tokens, is a placeholder, or (for
    Latin-script text) contains no manipulation verb. An empty string is left to
    `task_missing`.
    """
    task = (ep.tasks[0] if ep.tasks else "") or ""
    if not task.strip():
        return {}
    tokens = _tokens(task)
    normalized = " ".join(tokens)
    if normalized in PLACEHOLDERS:
        return {"": MV(1.0, note="placeholder text")}
    latin = task.isascii()
    if len(tokens) < MIN_TOKENS and latin:
        return {"": MV(1.0, note="fewer than %d words" % MIN_TOKENS)}
    if latin and not _has_verb(tokens):
        return {"": MV(1.0, note="no action verb")}
    return {"": MV(0.0)}


def task_missing(ep):
    """Flags an empty, whitespace-only or absent task string."""
    task = (ep.tasks[0] if ep.tasks else "") or ""
    missing = not task.strip()
    return {"": MV(1.0 if missing else 0.0, note="no task string" if missing else None)}
