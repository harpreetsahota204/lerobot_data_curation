"""Consistency metrics that compare an episode with its peers (batch metrics).

Each function takes ``(raws, label_of)`` and returns ``{sample_id: {signal: MV}}``.
They read the compact ``aux`` arrays the raw pass keeps for every episode.
"""

from collections import defaultdict

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .base import MV

MIN_PEERS = 5  # episodes needed in a subgroup before action_divergence is defined
K_NEIGHBORS = 5
LEVERAGE_MARGIN = 0.05  # share of the peers' range an episode must stretch beyond


def _subgroups(raws, label_of):
    """Episodes grouped by (normalization group, robot, state size, action size)."""
    groups = defaultdict(list)
    for sid, raw in raws.items():
        aux = raw.aux
        if not aux or aux.get("state") is None or aux.get("action") is None:
            continue
        key = (label_of[sid], raw.robot_type, aux["state"].shape[1], aux["action"].shape[1])
        groups[key].append(sid)
    return groups


def action_divergence(raws, label_of):
    """Variance of action among the nearest state neighbors in other episodes.

    For each sampled frame, finds its nearest neighbors in state space among the
    frames of the other episodes of the same subgroup, and measures how far the
    commanded action is from theirs. High means the demonstration takes a
    different action from its peers in a similar state. Without images, similar
    states can legitimately need different actions, so read it as a review signal.
    Needs at least MIN_PEERS episodes with matching state and action sizes.
    """
    out = {}
    for sids in _subgroups(raws, label_of).values():
        if len(sids) < MIN_PEERS:
            continue
        states = np.vstack([raws[s].aux["state"] for s in sids])
        actions = np.vstack([raws[s].aux["action"] for s in sids])
        owner = np.concatenate([[i] * len(raws[s].aux["state"]) for i, s in enumerate(sids)])
        k = min(K_NEIGHBORS + len(raws[sids[0]].aux["state"]), len(states) - 1)
        nn = NearestNeighbors(n_neighbors=k + 1).fit(states)
        _, idx = nn.kneighbors(states)
        for i, sid in enumerate(sids):
            rows = np.flatnonzero(owner == i)
            scores = []
            for r in rows:
                neighbors = [j for j in idx[r] if owner[j] != i][:K_NEIGHBORS]
                if not neighbors:
                    continue
                scores.append(float(np.mean(np.sum((actions[neighbors] - actions[r]) ** 2, axis=1))))
            if scores:
                out[sid] = {"": MV(float(np.mean(scores)))}
    return out


def stats_leverage(raws, label_of):
    """Number of feature dimensions this episode alone stretches beyond its peers.

    A dimension counts when the episode's minimum or maximum lies beyond every
    other episode of the same source by more than 5% of their range. High means
    this episode distorts the normalization statistics every training run will
    use. Needs at least three episodes from the same source in the view.
    """
    # Grouped by (source, array, size): an episode missing an array, or one whose
    # array has a different size, is compared only with peers that match it.
    groups = defaultdict(list)
    for sid, raw in raws.items():
        if not raw.aux or raw.source_id is None:
            continue
        for key, (lo, hi) in (raw.aux.get("extent") or {}).items():
            groups[(raw.source_id, key, len(lo))].append(sid)

    counts = defaultdict(float)
    seen = set()
    for (source_id, key, _), sids in groups.items():
        if len(sids) < 3:
            continue
        lows = np.array([raws[s].aux["extent"][key][0] for s in sids])
        highs = np.array([raws[s].aux["extent"][key][1] for s in sids])
        for i, sid in enumerate(sids):
            others = np.delete(np.arange(len(sids)), i)
            lo, hi = lows[others].min(axis=0), highs[others].max(axis=0)
            span = np.maximum(hi - lo, 1e-9)
            beyond = (lows[i] < lo - LEVERAGE_MARGIN * span) | (highs[i] > hi + LEVERAGE_MARGIN * span)
            counts[sid] += float(np.sum(beyond))
            seen.add(sid)
    return {sid: {"": MV(counts[sid])} for sid in seen}
