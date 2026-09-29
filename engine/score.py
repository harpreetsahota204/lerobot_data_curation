"""Per-episode and batch-wide orchestration of the metric engine.

Two passes, because normalization needs every episode before any single score
is final:

1. :func:`compute_raw` runs the enabled metrics on one episode, independent of
   every other episode. This is the expensive pass.
2. :func:`finalize` fits normalization within each group, computes z-scores,
   profile scores and verdicts for the whole batch.
"""

import logging
from dataclasses import dataclass, field

import numpy as np

from . import normalize, outliers
from .groups import DEFAULT_MIN_GROUP, assign_groups, canonical_task
from .metrics import METRICS, required_arrays_present
from .profiles import PROFILES, score_profile

logger = logging.getLogger(__name__)

# Bumped whenever a formula change would shift already-written scores, so the
# panel can refuse to blend runs computed under different formulas.
CONFIG_VERSION = 1

SEP = "|"
_MIN_SIGNED_SCALE_FRACTION = 0.05  # floor on the signed-z scale, as a share of the median


@dataclass
class RawEpisode:
    """One episode's raw metrics, before the batch has been seen."""

    episode_id: str
    task_key: str
    robot_type: "str | None"
    source_id: "str | None" = None
    signature: dict = field(default_factory=dict)  # schema signature for schema_mismatch
    aux: "dict | None" = None  # compact arrays for batch metrics
    spans: list = field(default_factory=list)  # candidate flagged intervals (see detail.flagged_spans)
    metrics: dict = field(default_factory=dict)  # metric -> {signal: MV}
    failures: dict = field(default_factory=dict)  # metric -> error text


@dataclass
class EpisodeResult:
    """One episode's finalized, batch-relative scores."""

    metrics: dict  # metric -> {"value", "worst", "worst_signal", "by_signal", "note", "z"}
    group_label: str
    group_basis: str
    group_n: int
    profiles: dict  # profile -> {"score", "n_flags", "driver", "verdict"}
    spans: list  # flagged intervals worth a timeline tag: {start_s, end_s, label, kind, severity}
    integrity_verdict: str
    language_verdict: str
    config_version: int = CONFIG_VERSION


def compute_raw(episode, sample_id, metric_names, assumptions=None):
    """Runs the enabled metrics on one :class:`EpisodeData`.

    `assumptions` are the confirmed dataset-level assumptions for this episode's
    source (see ``dataset_checks.assumptions_for``).

    A metric whose required arrays are missing is skipped. A metric that raises
    is recorded in ``failures`` and skipped, so one bad episode never stops a
    run.
    """
    if assumptions:
        episode.assumptions = dict(assumptions)
    raw = RawEpisode(
        episode_id=sample_id,
        task_key=canonical_task(episode.tasks),
        robot_type=episode.robot_type,
        source_id=episode.source_id,
        signature=_signature(episode),
        aux=_aux(episode),
    )
    try:
        from .detail import flagged_spans

        raw.spans = flagged_spans(episode)
    except Exception as e:  # noqa: BLE001 - spans are a convenience, never fatal
        raw.failures["spans"] = "%s: %s" % (type(e).__name__, e)
    for name in metric_names:
        spec = METRICS[name]
        if spec["batch"] or spec["fn"] is None:
            continue  # computed in finalize, across episodes
        if not required_arrays_present(spec, episode):
            continue
        try:
            values = spec["fn"](episode)
        except Exception as e:  # noqa: BLE001 - recorded, never fatal
            raw.failures[name] = "%s: %s" % (type(e).__name__, e)
            continue
        values = {s: v for s, v in values.items() if v is not None and np.isfinite(v.value)}
        if values:
            raw.metrics[name] = values
    return raw


AUX_POINTS = 32  # frames kept per episode for batch metrics


def _signature(episode):
    """The schema facts `schema_mismatch` compares across sources."""
    return {
        "fps": episode.fps,
        "action_dim": None if episode.action is None else int(episode.action.shape[1]),
        "state_dim": None if episode.state is None else int(episode.state.shape[1]),
        "cameras": tuple(sorted(episode.videos)),
    }


def _aux(episode):
    """A compact, range-normalized sample of state and action for batch metrics."""
    from .signals import normalized

    aux = {"state": None, "action": None, "extent": {}}
    for key, arr, rng in (
        ("state", episode.state, episode.state_range),
        ("action", episode.action, episode.action_range),
    ):
        if arr is None or len(arr) < 2 or not np.all(np.isfinite(arr)):
            continue
        idx = np.linspace(0, len(arr) - 1, min(AUX_POINTS, len(arr))).astype(int)
        aux[key] = normalized(arr, rng)[idx]
        aux["extent"][key] = (arr.min(axis=0), arr.max(axis=0))
    return aux


def _stat_key(metric, signal):
    return "%s%s%s" % (metric, SEP, signal)


def _signed_scale(stats):
    scale = normalize.MAD_TO_STD * stats["mad"]
    floor = _MIN_SIGNED_SCALE_FRACTION * abs(stats["median"])
    return max(scale, floor, 1e-9)


_LEVEL = {"pass": 0, "warn": 1, "fail": 2}

# Which metric decides whether a span kind is worth a timeline tag. A span is
# tagged only when its metric reaches warn, so the timeline shows what was
# flagged rather than an event on every episode. Recoveries are informational.
_SPAN_METRICS = {
    "idle": ("idle_lead_s", "idle_trail_s"),
    "pause": ("longest_pause_s",),
    "rough": ("sparc",),
    "spike": ("accel_spike_frac",),
}


def _timeline_spans(spans, metric_z):
    """Keeps the spans whose metric is at warn or worse, with a severity."""
    out = []
    for span in spans:
        if span["kind"] == "recovery":
            out.append({**span, "severity": "info"})
            continue
        z = max((metric_z.get(m, -np.inf) for m in _SPAN_METRICS.get(span["kind"], ())), default=-np.inf)
        severity = normalize.severity(z)
        if severity:
            out.append({**span, "severity": severity})
    return out


def _integrity_verdict(metrics):
    """Worst level across the integrity metrics that have a check, or "unknown"."""
    seen, worst = False, "pass"
    for name, m in metrics.items():
        check = METRICS[name]["check"]
        if check is None:
            continue
        seen = True
        level, above = check
        if m["value"] > above and _LEVEL[level] > _LEVEL[worst]:
            worst = level
    return worst if seen else "unknown"


def finalize(raws, metric_names, min_group=DEFAULT_MIN_GROUP):
    """Fits normalization and scores every episode as one batch.

    Args:
        raws: ``{sample_id: RawEpisode}``
        metric_names: the metrics that were enabled
        min_group: the smallest group that is normalized on its own

    Returns:
        ``(results, norm_stats)`` where ``results`` is ``{sample_id:
        EpisodeResult}`` and ``norm_stats`` is ``{group_label: {key: stats}}``
        (JSON-serializable, for the run record)
    """
    ids = list(raws)
    labels, basis = assign_groups([raws[i].task_key for i in ids], min_group=min_group)
    label_of = dict(zip(ids, labels))
    basis_of = dict(zip(ids, basis))
    group_n = {}
    for label in labels:
        group_n[label] = group_n.get(label, 0) + 1

    # Batch metrics compare an episode with its peers, so they need every raw
    # episode and the group labels. Their output joins the per-episode metrics.
    for name in metric_names:
        spec = METRICS[name]
        if not spec["batch"]:
            continue
        for sid, values in spec["fn"](raws, label_of).items():
            values = {s: v for s, v in values.items() if v is not None and np.isfinite(v.value)}
            if values:
                raws[sid].metrics[name] = values

    # Fit stats per (group, metric, signal) for every z-scored metric.
    norm_stats = {}
    for name in metric_names:
        spec = METRICS[name]
        if spec["kind"] == "flag":
            continue
        by_key = {}
        for sid in ids:
            for signal, mv in raws[sid].metrics.get(name, {}).items():
                by_key.setdefault((label_of[sid], _stat_key(name, signal)), []).append(mv.value)
        for (label, key), values in by_key.items():
            fitted = normalize.fit({key: values})[key]
            fitted["n"] = len(values)
            norm_stats.setdefault(label, {})[key] = fitted

    results = {}
    for sid in ids:
        raw = raws[sid]
        label = label_of[sid]
        metrics = {}
        metric_z = {}
        for name in metric_names:
            spec = METRICS[name]
            per_signal = raw.metrics.get(name)
            if not per_signal:
                continue

            if spec["kind"] == "flag":
                mv = per_signal[""]
                metrics[name] = {"value": mv.value, "note": mv.note, "by_signal": {}}
                continue

            if spec["kind"] == "signed_z":
                mv = per_signal[""]
                stats = norm_stats[label][_stat_key(name, "")]
                signed = float(
                    np.clip(
                        (mv.value - stats["median"]) / _signed_scale(stats),
                        -normalize.Z_CLIP,
                        normalize.Z_CLIP,
                    )
                )
                metrics[name] = {"value": signed, "raw": mv.value, "z": abs(signed), "by_signal": {}}
                metric_z[name] = abs(signed)
                continue

            zs = {}
            for signal, mv in per_signal.items():
                stats = norm_stats[label][_stat_key(name, signal)]
                zs[signal] = normalize.zscore(mv.value, stats, spec["higher_is_worse"])
            worst_signal = max(zs, key=lambda s: -np.inf if np.isnan(zs[s]) else zs[s])
            chosen = per_signal[worst_signal]
            metrics[name] = {
                "value": chosen.value,
                "worst": chosen.worst,
                "worst_signal": worst_signal if spec["per_signal"] else None,
                "z": zs[worst_signal],
                "by_signal": (
                    {s: {"value": mv.value, "worst": mv.worst, "z": zs[s]} for s, mv in per_signal.items()}
                    if spec["per_signal"]
                    else {}
                ),
            }
            if not np.isnan(zs[worst_signal]):
                metric_z[name] = zs[worst_signal]

        integrity = _integrity_verdict(metrics)
        lang = [metrics[m]["value"] for m in ("task_generic", "task_missing") if m in metrics]
        language = "unknown" if not lang else ("warn" if any(v > 0 for v in lang) else "pass")

        results[sid] = EpisodeResult(
            metrics=metrics,
            group_label=label,
            group_basis=basis_of[sid],
            group_n=group_n[label],
            spans=_timeline_spans(raw.spans, metric_z),
            profiles={
                p: score_profile(metric_z, p, language_flagged=(language == "warn"))
                for p in PROFILES
            },
            integrity_verdict=integrity,
            language_verdict=language,
        )
    outlier_names = {"iforest_score", "novelty_knn", "is_outlier"} & set(metric_names)
    if outlier_names:
        _add_outliers(results, label_of, metric_names, outlier_names)
    return results, norm_stats


MIN_OUTLIER_GROUP = 8  # episodes needed before an outlier model is fit


def _add_outliers(results, label_of, metric_names, outlier_names):
    """Fits the outlier models on each group's metric z-scores. Information only.

    Features are the oriented z-scores of the scored metrics, so a metric in
    thousands cannot drown one in [0, 1] in the distances. Groups smaller than
    MIN_OUTLIER_GROUP are skipped, because "unusual" needs a population.
    """
    scored = [
        m for m in metric_names if METRICS[m]["scored"] and METRICS[m]["kind"] != "flag"
    ]
    by_group = {}
    for sid, label in label_of.items():
        if sid in results:
            by_group.setdefault(label, []).append(sid)

    for sids in by_group.values():
        if len(sids) < MIN_OUTLIER_GROUP:
            continue
        x = np.array(
            [[results[s].metrics.get(m, {}).get("z", np.nan) for m in scored] for s in sids],
            dtype=np.float64,
        )
        x = x[:, ~np.all(np.isnan(x), axis=0)]
        if x.shape[1] == 0:
            continue
        iforest_scores, knn_dists = outliers.fit_and_score(x)
        fit = normalize.fit({"if": iforest_scores, "knn": knn_dists})
        for i, sid in enumerate(sids):
            z_if = normalize.zscore(iforest_scores[i], fit["if"], True)
            z_knn = normalize.zscore(knn_dists[i], fit["knn"], True)
            flagged = float(max(z_if, z_knn) >= normalize.WARN_Z)
            values = {
                "iforest_score": (float(iforest_scores[i]), z_if),
                "novelty_knn": (float(knn_dists[i]), z_knn),
                "is_outlier": (flagged, None),
            }
            for name in outlier_names:
                value, z = values[name]
                results[sid].metrics[name] = {"value": value, "by_signal": {}, **({"z": z} if z is not None else {})}
