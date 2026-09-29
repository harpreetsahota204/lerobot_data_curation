"""Writing results to FiftyOne samples, and the panel payload built from them.

These tests use a real, temporary, non-persistent FiftyOne dataset of plain
samples: `write.py` and `panel_data.py` only touch sample fields, the run record
and the temporal-tag collection, so no LeRobot media is needed.

    python -m unittest lerobot_data_curation.tests.test_write -v
"""

import unittest

import numpy as np

import fiftyone as fo
import fiftyone.core.tags as fota

from lerobot_data_curation import panel_data, write
from lerobot_data_curation.engine.dataset_checks import UNDER_COVERED
from lerobot_data_curation.engine.metrics import METRICS
from lerobot_data_curation.engine.score import compute_raw, finalize

from .helpers import make_episode, smooth_action
from .test_engine import NAMES, _busy_action, _follower, _grasp_episode

N = 24


def _score(dataset, names=None, only=None, with_state=False, assumptions=None):
    """Scores the dataset's samples with synthetic episodes. Returns ``(results, norm_stats)``."""
    names = names or list(METRICS)
    raws = {}
    for i, sample in enumerate(dataset):
        if only is not None and sample.id not in only:
            continue
        if with_state:
            action = _busy_action(n=300, dims=4, seed=i)
            ep = make_episode(action, state=_follower(action, 3, noise=0.004, seed=i), episode_index=i)
        else:
            action = smooth_action(n=300, dims=4, seed=i) + np.random.default_rng(700 + i).normal(0, 0.002, (300, 4))
            ep = make_episode(action, episode_index=i, tasks=[sample.task])
        raws[sample.id] = compute_raw(ep, sample.id, names, assumptions)
    return finalize(raws, names, min_group=5)


class FiftyOneCase(unittest.TestCase):
    def setUp(self):
        self.dataset = fo.Dataset()
        self.addCleanup(self.dataset.delete)
        self.dataset.add_samples(
            [fo.Sample(filepath="/tmp/ep%d.txt" % i, task="pick up the cube and place it in the box", episode_index=i) for i in range(N)]
        )
        # Filepath-backed samples have no media reference: the panel falls back to "?" for the source.


class WriteTests(FiftyOneCase):
    def test_fields_are_typed_and_named_safely(self):
        results, _ = _score(self.dataset)
        fields = write.write_results(self.dataset, self.dataset, results, run_id="r1")
        schema = self.dataset.get_field_schema()
        self.assertTrue(all("__" not in f for f in fields), "double underscores break mongoengine lookups")
        self.assertIsInstance(schema["lr_score_policy"], fo.FloatField)
        self.assertIsInstance(schema["lr_nflags_policy"], fo.IntField)
        self.assertIsInstance(schema["lr_verdict_policy"], fo.StringField)
        self.assertIsInstance(schema["lr_group_n"], fo.IntField)
        self.assertEqual(set(self.dataset.values("lr_run_id")), {"r1"})
        self.assertEqual(set(self.dataset.values("lr_config_version")), {1})

    def test_values_match_the_engine_results(self):
        results, _ = _score(self.dataset)
        write.write_results(self.dataset, self.dataset, results)
        for sample in self.dataset:
            r = results[sample.id]
            self.assertAlmostEqual(sample["lr_score_policy"], r.profiles["policy"]["score"], places=9)
            self.assertEqual(sample["lr_verdict_policy"], r.profiles["policy"]["verdict"])
            self.assertEqual(sample["lr_integrity_verdict"], r.integrity_verdict)

    def test_a_rerun_gives_identical_values(self):
        results, _ = _score(self.dataset)
        write.write_results(self.dataset, self.dataset, results)
        first = self.dataset.values("lr_score_policy")
        results, _ = _score(self.dataset)
        write.write_results(self.dataset, self.dataset, results)
        self.assertEqual(first, self.dataset.values("lr_score_policy"))

    def test_a_subset_rerun_clears_stale_fields_only_on_the_scored_samples(self):
        results, stats = _score(self.dataset)
        fields = write.write_results(self.dataset, self.dataset, results)
        write.register_run(self.dataset, {"metrics": list(METRICS), "min_group": 5}, stats, fields, {})

        subset_ids = set(self.dataset.take(5, seed=1).values("id"))
        sub_results, _ = _score(self.dataset, names=["sparc"], only=subset_ids)
        write.write_results(self.dataset, self.dataset.select(list(subset_ids)), sub_results)

        for sample in self.dataset:
            idle = sample["lr_idle_lead_s"]
            if sample.id in subset_ids:
                self.assertIsNone(idle)  # not computed this time, so cleared
            else:
                self.assertIsNotNone(idle)  # untouched

    def test_the_recovery_tag_follows_the_metric(self):
        action, state = _grasp_episode(reopen=True)
        assumptions = {"gripper_open_is": "high", "action_semantics": "joint_positions"}
        sample = self.dataset.first()

        def score(action, state):
            ep = make_episode(action, state=state, names=NAMES)
            raws = {sample.id: compute_raw(ep, sample.id, list(METRICS), assumptions)}
            return finalize(raws, list(METRICS), min_group=5)[0]

        write.write_results(self.dataset, self.dataset.select([sample.id]), score(action, state))
        self.assertIn(write.RECOVERY_TAG, self.dataset[sample.id].tags)

        action, state = _grasp_episode(reopen=False)
        write.write_results(self.dataset, self.dataset.select([sample.id]), score(action, state))
        self.assertNotIn(write.RECOVERY_TAG, self.dataset[sample.id].tags)

    def test_temporal_tags_are_replaced_on_rerun_and_hand_drawn_tags_survive(self):
        results, _ = _score(self.dataset)
        sample = self.dataset.first()
        hand = fota.TemporalTag(sample_id=sample.id, start=1_000_000_000, end=2_000_000_000, tag="my own note")
        fota.add_temporal_tags(self.dataset.select([sample.id]), [hand])

        # make sure there is something to tag
        for r in results.values():
            r.spans = [{"start_s": 1.0, "end_s": 2.0, "label": "idle at start", "kind": "idle", "severity": "warn"}]
        n1 = write.write_temporal_tags(self.dataset, results)
        n2 = write.write_temporal_tags(self.dataset, results)
        counts = self.dataset.temporal_tags.count()
        self.assertEqual(n1, n2)
        self.assertEqual(counts.get("idle at start:warn"), N)  # replaced, not doubled
        self.assertEqual(counts.get("my own note"), 1)  # untouched

    def test_the_sidebar_group_holds_the_fields_and_is_idempotent(self):
        results, _ = _score(self.dataset)
        fields = write.write_results(self.dataset, self.dataset, results)
        write.set_sidebar_group(self.dataset, fields)
        write.set_sidebar_group(self.dataset, fields)
        groups = {g.name: g.paths for g in self.dataset.app_config.sidebar_groups}
        self.assertEqual(set(groups[write.SIDEBAR_GROUP]), set(fields))
        others = [p for name, paths in groups.items() if name != write.SIDEBAR_GROUP for p in paths]
        self.assertTrue(set(fields).isdisjoint(others))
        self.assertEqual(sum(1 for g in self.dataset.app_config.sidebar_groups if g.name == write.SIDEBAR_GROUP), 1)

    def test_the_run_record_remembers_the_fields_and_run_id(self):
        results, stats = _score(self.dataset)
        fields = write.write_results(self.dataset, self.dataset, results, run_id="abc")
        write.register_run(self.dataset, {"metrics": ["sparc"], "min_group": 5}, stats, fields, {}, run_id="abc")
        self.assertEqual(set(write.previous_fields(self.dataset)), set(fields))
        self.assertEqual(self.dataset.load_run_results(write.RUN_KEY).run_id, "abc")


class PanelDataTests(FiftyOneCase):
    def _write(self, results, stats, run_id, dataset_view=None):
        view = dataset_view if dataset_view is not None else self.dataset
        fields = write.write_results(self.dataset, view, results, run_id=run_id)
        write.register_run(self.dataset, {"metrics": list(METRICS), "min_group": 5}, stats, fields, {}, run_id=run_id)

    def test_an_unscored_dataset_says_so(self):
        self.assertEqual(panel_data.build_panel_data(self.dataset, self.dataset), {"scored": False})

    def test_rows_carry_scores_verdicts_and_the_under_covered_cutoff(self):
        results, stats = _score(self.dataset)
        self._write(results, stats, "r1")
        payload = panel_data.build_panel_data(self.dataset, self.dataset)
        self.assertTrue(payload["scored"])
        self.assertEqual(len(payload["rows"]), N)
        row = payload["rows"][0]
        self.assertIn("policy", row["profiles"])
        self.assertIn(row["profiles"]["policy"]["verdict"], ("pass", "warn", "fail", "unknown"))
        self.assertEqual(payload["under_covered_below"], UNDER_COVERED)
        self.assertFalse(payload["mixed_runs"])
        self.assertEqual(payload["min_group"], 5)

    def test_signed_z_metrics_show_their_raw_value(self):
        # track_lag_ms is stored as a z-score; the panel must show milliseconds
        results, stats = _score(self.dataset, with_state=True)
        self._write(results, stats, "r1")
        payload = panel_data.build_panel_data(self.dataset, self.dataset)
        self.assertEqual(payload["metrics"]["track_lag_ms"]["display"], "raw")
        self.assertEqual(payload["metrics"]["length_z"]["display"], "value")  # a z by name: keep the z
        lags = [r["values"]["track_lag_ms"] for r in payload["rows"] if "track_lag_ms" in r["values"]]
        self.assertTrue(lags)
        self.assertAlmostEqual(float(np.median(lags)), 3 / 30 * 1000, delta=40)

    def test_warn_thresholds_appear_for_one_run_and_hide_for_mixed_runs(self):
        results, stats = _score(self.dataset)
        self._write(results, stats, "r1")
        self.assertIn("sparc", panel_data.build_panel_data(self.dataset, self.dataset)["warn_thresholds"])

        # rescore a few episodes in a second run: the recorded stats now describe only those
        subset = set(self.dataset.take(6, seed=2).values("id"))
        sub_results, sub_stats = _score(self.dataset, only=subset)
        self._write(sub_results, sub_stats, "r2", self.dataset.select(list(subset)))
        payload = panel_data.build_panel_data(self.dataset, self.dataset)
        self.assertTrue(payload["mixed_runs"])
        self.assertEqual(payload["warn_thresholds"], {})

    def test_metric_meta_describes_every_metric_for_the_panel(self):
        meta = panel_data.metric_meta()
        self.assertEqual(meta["frame_gaps"]["check"], ["fail", 0])
        self.assertEqual(meta["sparc"]["group"], "motion")
        self.assertIn("outliers", {m["family"] for m in meta.values()})
        self.assertTrue(all("display" in m for m in meta.values()))


if __name__ == "__main__":
    unittest.main()
