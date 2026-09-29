"""Runs the corruption suite and the length-control check on a real dataset.

    python -m lerobot_data_curation.harness.run_harness --dataset lr_dev

For every corruption, each episode that can carry the defect is paired with its
corrupted copy, and both are scored together in one batch. A metric passes when
the corrupted copy ranks worse than its own clean original in at least 80% of
pairs (flags: when they fire on the corrupted copy and not the original). An
identity control checks nothing moves without a corruption.

The length check truncates every episode to a common duration and reports how
much each scored metric's ranking changes. A metric whose ranking collapses is
marked length-confounded (arXiv:2606.10229 found 5 of 7 published metrics were).

Writes REPORT.md and report.json next to this file.
"""

import argparse
import datetime
import json
import os
import time
from dataclasses import replace

import numpy as np
from scipy.stats import spearmanr

import fiftyone as fo

from ..engine import dataset_checks
from ..engine.metrics import METRICS
from ..engine.reader import EpisodeReadError, LeRobotReader
from ..engine.score import compute_raw, finalize
from .corruptions import CORRUPTIONS, NOT_TESTABLE

# Metrics that measure the end of an episode. Truncating every episode to a common
# length cuts off exactly what they measure, so the length check cannot judge them.
TAIL_METRICS = {"idle_trail_s", "end_motion_ratio"}

PASS_RATE = 0.80
MAX_CONTROL_RATE = 0.05
HERE = os.path.dirname(os.path.abspath(__file__))


def load_episodes(dataset_name, limit=None):
    ds = fo.load_dataset(dataset_name)
    reader = LeRobotReader(ds)
    samples = list(ds.select_fields(["media_reference"]))
    if limit:
        samples = samples[:limit]
    checks = dataset_checks.infer_checks(reader, samples)
    assumptions = {sid: dataset_checks.assumptions_for(c) for sid, c in checks.items()}
    episodes = {}
    for sample, ep in reader.read_many(samples):
        if isinstance(ep, EpisodeReadError):
            continue
        ep.assumptions = dict(assumptions.get(ep.source_id, {}))
        episodes[sample.id] = ep
    return episodes


def _outcome(metric, x, c, direction=1):
    """True when the corrupted copy is worse than the clean original on `metric`.

    Compares raw values, not z-scores: a z-score is clipped at 10, and a clipped
    metric shows no movement even when its raw value clearly worsened. Batch
    metrics are skipped, because duplicating an episode changes its peers.
    """
    spec = METRICS[metric]
    if spec["batch"] or metric not in x.metrics or metric not in c.metrics:
        return None
    vx, vc = x.metrics[metric]["value"], c.metrics[metric]["value"]
    if vx is None or vc is None or np.isnan(vx) or np.isnan(vc):
        return None
    if spec["kind"] == "signed_z":
        return (vx - vc) * direction > 1e-9
    if spec["kind"] == "flag":
        return vx > vc
    return (vx > vc + 1e-9) if spec["higher_is_worse"] else (vx < vc - 1e-9)


def run_corruptions(episodes, names, clean_raws, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for corruption in CORRUPTIONS:
        raws, pairs = {}, []
        for sid, ep in episodes.items():
            raws["c:" + sid] = clean_raws[sid]
            corrupted = corruption.apply(ep, rng)
            if corrupted is None:
                continue
            raws["x:" + sid] = compute_raw(corrupted, "x:" + sid, names, ep.assumptions)
            raws["x:" + sid].episode_id = "x:" + sid
            pairs.append(sid)
        if not pairs:
            rows.append({"corruption": corruption.name, "metric": "(none)", "pairs": 0, "rate": None, "status": "not applicable"})
            continue
        results, _ = finalize(raws, names)
        for metric in corruption.expect:
            direction = dict(corruption.directions).get(metric, 1)
            outcomes = [_outcome(metric, results["x:" + s], results["c:" + s], direction) for s in pairs]
            outcomes = [o for o in outcomes if o is not None]
            rate = float(np.mean(outcomes)) if outcomes else None
            rows.append({"corruption": corruption.name, "metric": metric, "pairs": len(outcomes), "rate": rate,
                         "status": "no data" if rate is None else ("pass" if rate >= PASS_RATE else "FAIL")})
        if corruption.name == "identity":
            worse = []
            for metric in names:
                outs = [_outcome(metric, results["x:" + s], results["c:" + s]) for s in pairs]
                outs = [o for o in outs if o is not None]
                if outs:
                    worse.append(float(np.mean(outs)))
            rate = float(np.max(worse)) if worse else 0.0
            rows.append({"corruption": "identity", "metric": "(every metric, worst case)", "pairs": len(pairs), "rate": rate,
                         "status": "pass" if rate <= MAX_CONTROL_RATE else "FAIL"})
    return rows


def flag_baseline(results):
    """Share of clean episodes on which each flag fires (the false-alarm rate).

    A flag fires above its own check threshold when it has one (for example a
    video window more than 2 frames off), otherwise on any value above float noise.
    """
    out = {}
    for metric, spec in METRICS.items():
        if spec["kind"] != "flag" or metric == "timestamp_dev":  # timestamp_dev is info only
            continue
        limit = spec["check"][1] if spec["check"] else 1e-6
        vals = [r.metrics[metric]["value"] for r in results.values() if metric in r.metrics]
        if vals:
            out[metric] = float(np.mean([v > limit for v in vals]))
    return out


def truncate(ep, seconds):
    n = int(seconds * ep.fps)
    if ep.length <= n:
        return None
    fi = np.arange(n)
    return replace(
        ep,
        action=None if ep.action is None else ep.action[:n],
        state=None if ep.state is None else ep.state[:n],
        length=n,
        expected_length=n,
        frame_index=fi,
        raw_frame_index=fi,
        timestamps=fi / ep.fps,
        raw_timestamps=fi / ep.fps,
    )


def run_length_control(episodes, names, clean_results):
    """Spearman of each scored metric's z before and after truncating to a common duration."""
    durations = {s: e.length / e.fps for s, e in episodes.items()}
    common = max(3.0, float(np.percentile(list(durations.values()), 10)))
    kept = {s: e for s, e in episodes.items() if durations[s] > common}
    raws = {}
    for s, e in kept.items():
        t = truncate(e, common)
        if t is not None:
            raws[s] = compute_raw(t, s, names, e.assumptions)
    truncated, _ = finalize(raws, names)

    rows = []
    for metric in names:
        spec = METRICS[metric]
        if not spec["scored"] or spec["kind"] == "flag" or metric == "length_z":
            continue
        pairs = [
            (clean_results[s].metrics[metric]["z"], truncated[s].metrics[metric]["z"], durations[s])
            for s in raws
            if metric in clean_results[s].metrics and metric in truncated[s].metrics
            and clean_results[s].metrics[metric].get("z") is not None and truncated[s].metrics[metric].get("z") is not None
        ]
        if len(pairs) < 8:
            rows.append({"metric": metric, "n": len(pairs), "rho_kept": None, "rho_length": None, "status": "too few episodes"})
            continue
        before, after, dur = (np.array(c) for c in zip(*pairs))
        rho_kept = float(spearmanr(before, after).statistic)
        rho_len = float(spearmanr(before, dur).statistic)
        if metric in TAIL_METRICS:
            status = "not judged: measures the end, which truncation removes"
        else:
            status = "length-confounded" if rho_kept < 0.5 else "ok"
        rows.append({"metric": metric, "n": len(pairs), "rho_kept": rho_kept, "rho_length": rho_len, "status": status})
    return {"common_duration_s": common, "episodes": len(raws), "rows": rows}


def write_report(dataset_name, n_episodes, corruption_rows, baseline, length, seconds):
    lines = [
        "# Validation report",
        "",
        "Generated %s on `%s` (%d episodes) in %.0f s. Regenerate with "
        "`python -m lerobot_data_curation.harness.run_harness --dataset %s`."
        % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), dataset_name, n_episodes, seconds, dataset_name),
        "",
        "## Corruption suite",
        "",
        "Each applicable episode is paired with a corrupted copy of itself. A metric passes when the copy "
        "ranks worse than its own original in at least %d%% of pairs (flags: when the flag fires on the copy "
        "and not on the original). The identity control must move in at most %d%% of pairs."
        % (PASS_RATE * 100, MAX_CONTROL_RATE * 100),
        "",
        "| Corruption | Metric | Pairs | Worse than original | Status |",
        "|---|---|---|---|---|",
    ]
    for r in corruption_rows:
        rate = "n/a" if r["rate"] is None else "%.0f%%" % (100 * r["rate"])
        lines.append("| %s | `%s` | %d | %s | %s |" % (r["corruption"], r["metric"], r["pairs"], rate, r["status"]))

    lines += ["", "## False-alarm rate of the flags on unmodified episodes", "", "| Flag | Fires on |", "|---|---|"]
    for metric, rate in sorted(baseline.items()):
        lines.append("| `%s` | %.0f%% of episodes |" % (metric, 100 * rate))

    lines += [
        "",
        "## Length control",
        "",
        "Every episode longer than %.1f s is truncated to %.1f s and re-scored (%d episodes). The table is the "
        "Spearman correlation between a metric's z-score before and after truncation. A metric whose ranking "
        "collapses (below 0.5) is length-confounded: it measures how long an episode is more than how good it is."
        % (length["common_duration_s"], length["common_duration_s"], length["episodes"]),
        "",
        "| Metric | Episodes | Rank correlation before vs after | Correlation with duration | Status |",
        "|---|---|---|---|---|",
    ]
    for r in length["rows"]:
        fmt = lambda v: "n/a" if v is None else "%.2f" % v  # noqa: E731
        lines.append("| `%s` | %d | %s | %s | %s |" % (r["metric"], r["n"], fmt(r["rho_kept"]), fmt(r["rho_length"]), r["status"]))

    lines += ["", "## Not testable on this data", "", "| Metric | Why |", "|---|---|"]
    for metric, why in NOT_TESTABLE.items():
        lines.append("| `%s` | %s |" % (metric, why))
    lines += [
        "",
        "## Caveats",
        "",
        "- The data is `lr_dev`: 2 episodes from each of 51 sources, so every episode is normalized against the "
        "pooled view. Pairing each episode with its own corrupted copy removes the cross-robot variance, but this is "
        "a test of metric direction, not of ranking quality on a real curation task.",
        "- Corruptions are synthetic. They show a metric responds to the defect it targets, not that the defect "
        "predicts a worse policy (see the evidence notes in METRICS.md).",
    ]
    with open(os.path.join(HERE, "REPORT.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="lr_dev")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    start = time.time()
    episodes = load_episodes(args.dataset, args.limit)
    names = [m for m, s in METRICS.items() if s["fn"] is not None or s["batch"]]
    clean_raws = {sid: compute_raw(ep, sid, names, ep.assumptions) for sid, ep in episodes.items()}
    clean_results, _ = finalize(dict(clean_raws), names)

    corruption_rows = run_corruptions(episodes, names, clean_raws, args.seed)
    baseline = flag_baseline(clean_results)
    length = run_length_control(episodes, names, clean_results)
    seconds = time.time() - start
    write_report(args.dataset, len(episodes), corruption_rows, baseline, length, seconds)
    with open(os.path.join(HERE, "report.json"), "w") as f:
        json.dump({"corruptions": corruption_rows, "flag_baseline": baseline, "length": length}, f, indent=2)

    failed = [r for r in corruption_rows if r["status"] == "FAIL"]
    print("wrote %s" % os.path.join(HERE, "REPORT.md"))
    print("corruption rows: %d, failed: %d, length-confounded: %s" % (
        len(corruption_rows), len(failed), [r["metric"] for r in length["rows"] if r["status"] == "length-confounded"]))
    for r in failed:
        print("  FAIL %s / %s: %s of %d pairs" % (r["corruption"], r["metric"], "n/a" if r["rate"] is None else "%.0f%%" % (100 * r["rate"]), r["pairs"]))


if __name__ == "__main__":
    main()
