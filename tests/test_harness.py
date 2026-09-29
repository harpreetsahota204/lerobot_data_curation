"""The harness itself, on synthetic episodes (no dataset needed).

    python -m unittest lerobot_data_curation.tests.test_harness -v
"""

import unittest

import numpy as np

from lerobot_data_curation.engine.metrics import METRICS
from lerobot_data_curation.engine.score import compute_raw, finalize
from lerobot_data_curation.harness import corruptions as cz
from lerobot_data_curation.harness import run_harness as rh

from .helpers import make_episode, smooth_action


def _episodes(n=14):
    out = {}
    for i in range(n):
        act = smooth_action(n=360, dims=4, seed=i) + np.random.default_rng(900 + i).normal(0, 0.003, (360, 4))
        out["e%d" % i] = make_episode(act, episode_index=i)
    return out


class HarnessTests(unittest.TestCase):
    def test_suite_runs_and_the_metrics_it_can_reach_pass(self):
        names = [m for m, s in METRICS.items() if s["fn"] is not None or s["batch"]]
        episodes = _episodes()
        clean = {sid: compute_raw(ep, sid, names) for sid, ep in episodes.items()}
        rows = rh.run_corruptions(episodes, names, clean, seed=0)
        by = {(r["corruption"], r["metric"]): r for r in rows}
        for key in (
            ("identity", "(every metric, worst case)"),
            ("idle_start", "idle_lead_s"),
            ("mid_pause", "longest_pause_s"),
            ("duplicate_frame", "frame_gaps"),
            ("generic_task", "task_generic"),
            ("abandoned", "length_z"),
        ):
            self.assertEqual(by[key]["status"], "pass", key)
        # a corruption that needs a state array reports not applicable rather than failing
        self.assertEqual(by[("stalled_joint", "(none)")]["status"], "not applicable")

    def test_outcome_uses_raw_values_and_the_metric_polarity(self):
        names = ["sparc", "idle_lead_s"]
        ep = make_episode(smooth_action(n=300, seed=1))
        padded = cz.idle_start(ep, np.random.default_rng(0))
        raws = {"c": compute_raw(ep, "c", names), "x": compute_raw(padded, "x", names)}
        res, _ = finalize(raws, names, min_group=2)
        self.assertTrue(rh._outcome("idle_lead_s", res["x"], res["c"]))
        self.assertFalse(rh._outcome("idle_lead_s", res["c"], res["x"]))

    def test_every_expected_metric_exists(self):
        for corruption in cz.CORRUPTIONS:
            for metric in corruption.expect:
                self.assertIn(metric, METRICS, corruption.name)

    def test_every_metric_is_covered_or_explained(self):
        covered = {m for c in cz.CORRUPTIONS for m in c.expect} | set(cz.NOT_TESTABLE)
        missing = [m for m, s in METRICS.items() if (s["fn"] is not None or s["batch"]) and m not in covered]
        self.assertEqual(missing, [], "add a corruption (or a NOT_TESTABLE reason) for these metrics")


if __name__ == "__main__":
    unittest.main()
