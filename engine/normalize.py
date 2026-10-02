"""Dataset-wide robust normalization with median-centered one-sided scales.

Generic over metric names: callers decide which raw per-episode metrics to
normalize, their polarity, and how to combine them. Keeping this module
metric-agnostic means the same normalization logic serves motion, health,
and outlier metrics alike.
"""

import numpy as np

WARN_Z = 2.0
FAIL_Z = 3.0
Z_CLIP = 10.0

MAD_TO_STD = 1.4826  # scales a MAD (or a semi-IQR) to a normal distribution's std
_MIN_SCALE = 1e-12  # numerical guard for a corpus with genuinely no variation


def fit(values_by_metric):
    """Fits robust ``(median, scale)`` stats for each metric across a corpus.

    Args:
        values_by_metric: a dict of metric name -> array-like of raw values,
            one per episode (NaNs are ignored)

    Returns:
        a dict of metric name -> ``{"median", "mad", "scale_high",
        "scale_low"}``. ``scale_high``/``scale_low`` are the std-equivalent
        denominators :func:`zscore` divides by, fit from the values above and
        below the median respectively (see :func:`_one_sided_scale`);
        ``mad`` is the plain two-sided median absolute deviation, kept
        alongside them for reference.
    """
    stats = {}
    for metric, values in values_by_metric.items():
        values = np.asarray(values, dtype=np.float64)
        values = values[~np.isnan(values)]
        if len(values) == 0:
            stats[metric] = {
                "median": 0.0,
                "mad": 0.0,
                "scale_high": _MIN_SCALE,
                "scale_low": _MIN_SCALE,
            }
            continue
        median = float(np.median(values))
        stats[metric] = {
            "median": median,
            "mad": float(np.median(np.abs(values - median))),
            "scale_high": _one_sided_scale(values, median, above=True),
            "scale_low": _one_sided_scale(values, median, above=False),
        }
    return stats


def _one_sided_scale(values, median, above):
    """A z-score denominator fit from one side of the median only.

    The semi-interquartile range on that side (``p75 - p50`` above,
    ``p50 - p25`` below), which for a normal distribution is the same
    0.6745 sigma an ordinary MAD measures -- hence the same
    :data:`MAD_TO_STD` factor, and hence identical behavior on symmetric
    data.

    One-sided because every metric here is read one-sidedly: the question is
    always "how far into the *bad* tail does this episode sit", never "how far
    from typical", so the good side's spread has no business setting the bad
    side's scale. These distributions are strongly asymmetric -- real
    per-window jerk_rms on Voxel51/ABC-130k skews past 20 -- and a two-sided
    MAD averages the narrow good side in, understating the bad side's spread
    badly enough to put a third of every channel's windows past the fail
    threshold. A nominal 0.13% rate was landing at 33% in practice.

    A quantile gap rather than the median of the bad half specifically: both
    converge to the same thing for a continuous distribution, but the quantile
    keeps a 25% breakdown point at small n. Scoring four episodes where one is
    a gross outlier, the median of the bad *half* is computed over two points,
    one of which is the outlier -- so the outlier sets its own scale and hides
    itself.
    """
    quantile = float(np.percentile(values, 75 if above else 25))
    gap = (quantile - median) if above else (median - quantile)
    if gap > 0:
        return gap * MAD_TO_STD
    deviations = values[values > median] - median if above else median - values[values < median]
    return _zero_inflated_scale(deviations)


def _zero_inflated_scale(deviations):
    """Scale for a metric whose selected tail is mostly one repeated value.

    dropout, clock drift and desync all read exactly zero for most episodes
    of a clean dataset, so there's no spread to measure and the choice of
    denominator *is* the policy. Flooring it at a small constant (this used
    to floor it at 1e-6) makes the metric boolean: 0.1% dropout and 30%
    dropout both divide out past the clip into an identical hard "fail" that
    also dumps a maximal term into the composite score. Instead the scale is
    set so a typical member of the nonzero tail lands exactly on the warn
    threshold -- being in the bad tail at all earns a human glance, being
    further out earns proportionally more, and variation on the opposite
    (good) side never sets this denominator.
    """
    tail = deviations[deviations > 0]
    if len(tail) == 0:
        return _MIN_SCALE
    return float(np.median(tail)) / WARN_Z


def _scale(metric_stats, higher_is_worse):
    """The z-score denominator for a metric, taken from its *bad* side.

    One scale serves both directions rather than a piecewise two-sided
    z-score: only the bad direction drives flags and thresholds, and a single
    scale keeps :func:`zscore` monotone and :func:`raw_value_at_z` its exact
    inverse.

    Falls back through the single-``scale`` and then MAD-only shapes this
    dict used to have, so norm_stats cached by an older run still render in
    the panel.
    """
    scale = metric_stats.get("scale_high" if higher_is_worse else "scale_low")
    if scale is None:
        scale = metric_stats.get("scale")
    if scale is None:
        scale = MAD_TO_STD * metric_stats["mad"]
    return max(float(scale), _MIN_SCALE)


def zscore(value, metric_stats, higher_is_worse=True):
    """Robust z-score of one raw value against fitted ``(median, scale)`` stats.

    The sign is flipped when ``higher_is_worse`` is False, so the result is
    always positive-means-worse regardless of the metric's own polarity.
    The result is clipped to +/- :data:`Z_CLIP` as a backstop for a corpus
    with no measurable spread at all (every episode identical, so any
    departure divides by :data:`_MIN_SCALE`); :func:`_one_sided_scale` is what
    keeps an ordinary mostly-zero metric off the clip in the first place.
    """
    if value is None or np.isnan(value):
        return np.nan
    z = (value - metric_stats["median"]) / _scale(metric_stats, higher_is_worse)
    z = z if higher_is_worse else -z
    return float(np.clip(z, -Z_CLIP, Z_CLIP))


def raw_value_at_z(z, metric_stats, higher_is_worse=True):
    """Inverse of :func:`zscore` (ignoring the clip): the raw value whose oriented z-score is ``z``.

    Lets a UI draw thresholds in a metric's own units -- e.g. the raw value
    where warn severity starts is ``raw_value_at_z(WARN_Z, stats, polarity)``.
    """
    delta = z * _scale(metric_stats, higher_is_worse)
    return float(metric_stats["median"] + (delta if higher_is_worse else -delta))


def severity(z_score):
    """Returns "fail"/"warn"/None for an oriented z-score (positive == worse)."""
    if np.isnan(z_score):
        return None
    if z_score >= FAIL_Z:
        return "fail"
    if z_score >= WARN_Z:
        return "warn"
    return None
