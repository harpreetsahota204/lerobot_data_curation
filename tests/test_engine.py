"""Unit and corruption tests for the slice-1 engine.

Run from the plugin's parent folder:
    python -m unittest lerobot_data_curation.tests.test_engine -v

Each scored metric ships with a corruption it must catch: a clean episode and a
corrupted copy, and the metric must rank the corrupted one worse.
"""

import unittest

import numpy as np

from lerobot_data_curation.engine import dataset_checks as dc
from lerobot_data_curation.engine import normalize
from lerobot_data_curation.engine.groups import assign_groups, canonical_task
from lerobot_data_curation.engine.metrics import (
    METRICS,
    consistency,
    gripper,
    integrity,
    language,
    motion,
    time_metrics,
    tracking,
)
from lerobot_data_curation.engine.profiles import score_profile
from lerobot_data_curation.engine.reader import VideoWindow, infer_feature_map, slug, split_joint_groups
from lerobot_data_curation.engine.score import compute_raw, finalize

from .helpers import make_episode, smooth_action


class FeatureMapTests(unittest.TestCase):
    def _info(self, keys):
        return {
            "features": {
                k: {"dtype": "float32", "shape": [6]} for k in keys
            }
            | {"observation.images.top": {"dtype": "video", "shape": [480, 640, 3]}}
        }

    def test_standard_keys(self):
        fm = infer_feature_map(self._info(["action", "observation.state"]))
        self.assertEqual((fm.action_key, fm.state_key), ("action", "observation.state"))
        self.assertEqual(fm.camera_keys, ["observation.images.top"])

    def test_ambiguous_keys_are_reported_not_guessed(self):
        fm = infer_feature_map(
            self._info(["arm_action", "hand_action", "observation.arm_state", "observation.hand_state"])
        )
        self.assertIsNone(fm.action_key)
        self.assertIsNone(fm.state_key)
        self.assertTrue(any("ambiguous" in n for n in fm.notes))

    def test_sole_candidate_is_used(self):
        fm = infer_feature_map(self._info(["actions", "observation.states"]))
        self.assertEqual((fm.action_key, fm.state_key), ("actions", "observation.states"))

    def test_override_applies_only_when_key_exists(self):
        info = self._info(["arm_action", "hand_action"])
        fm = infer_feature_map(info, overrides={"action_key": "hand_action", "state_key": "nope"})
        self.assertEqual(fm.action_key, "hand_action")
        self.assertIsNone(fm.state_key)


class GroupingTests(unittest.TestCase):
    def test_canonical_task(self):
        self.assertEqual(canonical_task(["Pick  up the CUP."]), canonical_task(["pick up the cup"]))
        self.assertEqual(canonical_task(["b", "a"]), "a | b")
        self.assertEqual(canonical_task([]), "")

    def _labels(self, n_task, n_other):
        keys = ["a"] * n_task + ["b"] * n_other
        return assign_groups(keys, min_group=20)

    def test_ladder_sizes(self):
        for size, basis in ((5, "pooled"), (19, "pooled"), (20, "task"), (400, "task")):
            labels, bases = self._labels(size, 1)
            self.assertEqual(bases[0], basis, size)
            self.assertEqual(labels[0], "task:a" if basis == "task" else "__pooled__")

    def test_view_smaller_than_min_group_is_pooled(self):
        labels, bases = assign_groups(["t"] * 10, min_group=20)
        self.assertEqual(set(bases), {"pooled"})

    def test_empty_task_is_never_its_own_group(self):
        labels, bases = assign_groups([""] * 30, min_group=20)
        self.assertEqual(set(bases), {"pooled"})

    def test_min_group_is_configurable(self):
        labels, bases = assign_groups(["a"] * 8 + ["b"] * 3, min_group=8)
        self.assertEqual(bases.count("task"), 8)


class NormalizeTests(unittest.TestCase):
    def test_zero_inflated_metric_does_not_saturate(self):
        values = [0.0] * 90 + [1.0] * 5 + [4.0] * 5
        stats = normalize.fit({"m": values})["m"]
        # the median of the nonzero tail (2.5) lands exactly on the warn threshold
        self.assertAlmostEqual(normalize.zscore(2.5, stats, True), normalize.WARN_Z, places=6)
        z1, z4 = normalize.zscore(1.0, stats, True), normalize.zscore(4.0, stats, True)
        self.assertTrue(0 < z1 < normalize.WARN_Z < z4 < normalize.Z_CLIP)


class SparcTests(unittest.TestCase):
    def test_jitter_is_ranked_worse(self):
        clean = smooth_action(n=450, seed=1)
        rng = np.random.default_rng(2)
        jittery = clean + rng.normal(0, 0.04, clean.shape)
        a = motion.sparc(make_episode(clean))["arm_all"].value
        b = motion.sparc(make_episode(jittery))["arm_all"].value
        self.assertGreater(a, b)  # closer to 0 is smoother

    def test_gripper_is_excluded_from_arm_signals(self):
        names = ["left_waist", "left_elbow", "left_gripper", "right_waist", "right_elbow", "right_gripper"]
        ep = make_episode(smooth_action(n=450, dims=6), names=names)
        self.assertEqual(set(motion.sparc(ep)), {"arm_left", "arm_right"})
        self.assertEqual(split_joint_groups(names)["gripper:left"], [2])

    def test_units_do_not_matter(self):
        base = smooth_action(n=450, seed=3)
        rng = np.random.default_rng(4)
        base = base + rng.normal(0, 0.01, base.shape)
        a = motion.sparc(make_episode(base))["arm_all"].value
        b = motion.sparc(make_episode(base * 57.3))["arm_all"].value  # radians -> degrees
        self.assertAlmostEqual(a, b, places=6)


class IdleLeadTests(unittest.TestCase):
    def test_prepended_idle_is_measured(self):
        clean = smooth_action(n=300, seed=5)
        rest = np.repeat(clean[:1], 60, axis=0)  # 2 s at 30 fps
        padded = np.concatenate([rest, clean])
        a = time_metrics.idle_lead_s(make_episode(clean))[""].value
        b = time_metrics.idle_lead_s(make_episode(padded))[""].value
        self.assertGreater(b, a + 1.5)
        self.assertAlmostEqual(b, 2.0, delta=0.6)

    def test_never_moving_reports_full_duration(self):
        still = np.zeros((90, 6))
        self.assertAlmostEqual(time_metrics.idle_lead_s(make_episode(still))[""].value, 3.0, delta=0.1)


class IntegrityTests(unittest.TestCase):
    def test_clean_episode_has_zero_gaps(self):
        self.assertEqual(integrity.frame_gaps(make_episode(smooth_action()))[""].value, 0.0)

    def test_duplicates_are_detected(self):
        fi = np.arange(300)
        fi[100] = 99
        self.assertGreater(integrity.frame_gaps(make_episode(smooth_action(), frame_index=fi))[""].value, 0)

    def test_shuffled_order_is_detected(self):
        fi = np.arange(300)
        fi[[10, 11]] = fi[[11, 10]]
        mv = integrity.frame_gaps(make_episode(smooth_action(), frame_index=fi))[""]
        self.assertGreater(mv.value, 0)
        self.assertIn("out of order", mv.note)

    def test_missing_frames_are_detected(self):
        fi = np.delete(np.arange(301), 150)
        self.assertGreater(integrity.frame_gaps(make_episode(smooth_action(n=300), frame_index=fi))[""].value, 0)


class TaskGenericTests(unittest.TestCase):
    def _flag(self, text):
        ep = make_episode(smooth_action(n=30), tasks=[text])
        out = language.task_generic(ep)
        return out[""].value if out else None

    def test_generic_strings_are_flagged(self):
        for text in ("Hold", "Up", "task desc", "Task", "the red box on the left table"):
            self.assertEqual(self._flag(text), 1.0, text)

    def test_specific_strings_pass(self):
        for text in (
            "Pick up the red cup on the table.",
            "stack cube left",
            "Grab a plastic bottle and put it in the box.",
            "Picking up the cube and placing it in the box",
        ):
            self.assertEqual(self._flag(text), 0.0, text)

    def test_empty_is_left_to_task_missing(self):
        self.assertIsNone(self._flag("   "))

    def test_non_latin_scripts_skip_the_verb_rule(self):
        self.assertEqual(self._flag("把红色杯子放到盘子上"), 0.0)


class ProfileTests(unittest.TestCase):
    def test_worst_group_drives_the_score(self):
        out = score_profile({"sparc": 0.5, "idle_lead_s": 4.0, "length_z": 0.1}, "policy")
        self.assertEqual(out["driver"], "time")
        self.assertAlmostEqual(out["score"], 4.0)
        self.assertEqual(out["n_flags"], 1)
        self.assertEqual(out["verdict"], "fail")

    def test_one_bad_group_is_not_diluted(self):
        out = score_profile({"sparc": 0.0, "idle_lead_s": 3.5, "length_z": 0.0}, "policy")
        self.assertGreaterEqual(out["score"], 3.0)

    def test_language_flag_forces_review_only_for_vla(self):
        z = {"sparc": 0.1, "idle_lead_s": 0.1, "length_z": 0.1}
        self.assertEqual(score_profile(z, "policy", language_flagged=True)["verdict"], "pass")
        self.assertEqual(score_profile(z, "vla", language_flagged=True)["verdict"], "warn")

    def test_no_data_is_unknown_not_pass(self):
        self.assertEqual(score_profile({}, "policy")["verdict"], "unknown")


class BatchTests(unittest.TestCase):
    """Corruptions through the full raw + finalize path."""

    def _batch(self, corrupt):
        raws = {}
        for i in range(30):
            act = smooth_action(n=300, seed=i)
            act = act + np.random.default_rng(100 + i).normal(0, 0.002, act.shape)
            ep = make_episode(act, episode_index=i)
            raws["e%d" % i] = compute_raw(ep, "e%d" % i, list(METRICS))
        # the corrupted episode, built from the same recipe
        act = smooth_action(n=300, seed=999)
        ep = corrupt(act)
        raws["bad"] = compute_raw(ep, "bad", list(METRICS))
        results, _ = finalize(raws, list(METRICS), min_group=20)
        return results

    def test_jitter_is_flagged(self):
        rng = np.random.default_rng(7)
        res = self._batch(lambda a: make_episode(a + rng.normal(0, 0.05, a.shape)))
        self.assertGreater(res["bad"].profiles["policy"]["score"], 3.0)
        self.assertEqual(res["bad"].profiles["policy"]["driver"], "motion")

    def test_inserted_idle_is_flagged(self):
        res = self._batch(lambda a: make_episode(np.concatenate([np.repeat(a[:1], 90, 0), a])))
        # a lone member of the nonzero tail lands on the warn threshold by design
        self.assertGreaterEqual(res["bad"].metrics["idle_lead_s"]["z"], normalize.WARN_Z)
        self.assertGreater(res["bad"].metrics["idle_lead_s"]["value"], 2.5)
        self.assertEqual(res["e0"].metrics["idle_lead_s"]["z"], 0.0)

    def test_truncated_episode_moves_length_z(self):
        res = self._batch(lambda a: make_episode(a[:120]))
        self.assertLess(res["bad"].metrics["length_z"]["value"], -3.0)

    def test_duplicated_frame_index_fails_integrity(self):
        def corrupt(a):
            fi = np.arange(len(a))
            fi[50] = 49
            return make_episode(a, frame_index=fi)

        res = self._batch(corrupt)
        self.assertEqual(res["bad"].integrity_verdict, "fail")
        self.assertEqual(res["e0"].integrity_verdict, "pass")

    def test_placeholder_task_is_flagged(self):
        res = self._batch(lambda a: make_episode(a, tasks=["task desc"]))
        self.assertEqual(res["bad"].language_verdict, "warn")


class Batch1MetricTests(unittest.TestCase):
    """Slice 2A batch 1: motion and time metrics, each with the corruption it must catch."""

    def _ep(self, action, **kw):
        return make_episode(action, **kw)

    def test_ldlj_jitter_is_ranked_worse(self):
        clean = smooth_action(n=450, seed=11)
        jittery = clean + np.random.default_rng(12).normal(0, 0.03, clean.shape)
        a = motion.ldlj(self._ep(clean))["arm_all"].value
        b = motion.ldlj(self._ep(jittery))["arm_all"].value
        self.assertGreater(a, b)

    def test_jerk_and_psd_directions(self):
        clean = smooth_action(n=450, seed=13)
        jittery = clean + np.random.default_rng(14).normal(0, 0.03, clean.shape)
        self.assertLess(motion.jerk_rms(self._ep(clean))["arm_all"].value, motion.jerk_rms(self._ep(jittery))["arm_all"].value)
        self.assertGreater(motion.psd_lf_hf(self._ep(clean))["arm_all"].value, motion.psd_lf_hf(self._ep(jittery))["arm_all"].value)

    def test_idle_trail_is_measured(self):
        clean = smooth_action(n=300, seed=15)
        padded = np.concatenate([clean, np.repeat(clean[-1:], 60, axis=0)])
        a = time_metrics.idle_trail_s(self._ep(clean))[""].value
        b = time_metrics.idle_trail_s(self._ep(padded))[""].value
        self.assertAlmostEqual(b - a, 2.0, delta=0.6)

    def test_mid_episode_pause_is_the_longest_pause(self):
        clean = smooth_action(n=300, seed=16)
        paused = np.concatenate([clean[:150], np.repeat(clean[150:151], 90, axis=0), clean[150:]])
        a = time_metrics.longest_pause_s(self._ep(clean))[""].value
        b = time_metrics.longest_pause_s(self._ep(paused))[""].value
        self.assertGreater(b, 2.0)
        self.assertGreater(b, a + 1.5)

    def test_edge_idle_is_not_a_pause(self):
        clean = smooth_action(n=300, seed=17)
        edged = np.concatenate([np.repeat(clean[:1], 90, axis=0), clean, np.repeat(clean[-1:], 90, axis=0)])
        self.assertLess(time_metrics.longest_pause_s(self._ep(edged))[""].value, 1.0)

    def test_cut_off_episode_has_high_end_motion(self):
        clean = smooth_action(n=300, seed=18)
        a = time_metrics.end_motion_ratio(self._ep(clean))[""].value
        b = time_metrics.end_motion_ratio(self._ep(clean[:100]))[""].value
        rested = np.concatenate([clean, np.repeat(clean[-1:], 60, axis=0)])  # ends at rest
        c = time_metrics.end_motion_ratio(self._ep(rested))[""].value
        self.assertLess(c, 0.05)
        self.assertGreater(b, 0.6)
        self.assertGreater(b, a)

    def test_idle_frac_rises_with_idle_time(self):
        clean = smooth_action(n=300, seed=19)
        padded = np.concatenate([np.repeat(clean[:1], 150, axis=0), clean])
        self.assertGreater(
            time_metrics.idle_frac(self._ep(padded))[""].value,
            time_metrics.idle_frac(self._ep(clean))[""].value + 0.2,
        )

    def test_wandering_path_is_longer(self):
        clean = smooth_action(n=300, seed=20)
        t = np.linspace(0, 6 * np.pi, 300)[:, None]
        wander = clean + 0.4 * np.sin(t) * np.ones((1, clean.shape[1]))
        self.assertGreater(time_metrics.path_length(self._ep(wander))[""].value, time_metrics.path_length(self._ep(clean))[""].value)

    def test_sparc_phase_needs_a_named_gripper(self):
        arm = smooth_action(n=450, dims=3, seed=21)
        gripper = np.zeros((450, 1))
        gripper[150:300] = 1.0  # closes at 150, opens at 300
        action = np.hstack([arm, gripper])
        unnamed = self._ep(action)
        self.assertEqual(motion.sparc_phase(unnamed), {})
        named = self._ep(action, names=["waist", "shoulder", "elbow", "gripper"])
        named.feature_map.action_key = "action"
        out = motion.sparc_phase(named)
        self.assertEqual(set(out), {"arm_all"})
        self.assertTrue(np.isfinite(out["arm_all"].value))

    def test_sparc_phase_splits_at_gripper_transitions(self):
        from lerobot_data_curation.engine.signals import gripper_transitions

        unit = np.concatenate([np.zeros(100), np.ones(100), np.zeros(100)])
        self.assertEqual(gripper_transitions(unit), [100, 200])
        chatter = np.array([0, 0.45, 0.55, 0.45, 0.55, 0.5, 0.5] * 10, dtype=float)
        self.assertEqual(gripper_transitions(chatter), [])  # noise inside the hysteresis band


def _follower(action, lag, noise=0.0, seed=0):
    """A state that follows the action `lag` samples late."""
    state = np.vstack([np.repeat(action[:1], lag, axis=0), action[: len(action) - lag]])
    return state + np.random.default_rng(seed).normal(0, noise, state.shape)


def _busy_action(n=450, dims=4, seed=0):
    """Multi-frequency motion, so cross-correlation has something to lock on to."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n)[:, None]
    freqs = rng.uniform(2, 6, size=(1, dims))
    phase = rng.uniform(0, 6.28, size=(1, dims))
    return np.sin(2 * np.pi * freqs * t + phase) + 0.3 * np.sin(2 * np.pi * 2.3 * freqs * t)


class Batch2MetricTests(unittest.TestCase):
    """Slice 2A batch 2: tracking, contact and gripper metrics."""

    def test_lag_is_recovered(self):
        action = _busy_action(seed=1)
        ep = make_episode(action, state=_follower(action, 3, noise=0.005), fps=30.0)
        lag = tracking.track_lag_ms(ep)[""].value
        self.assertAlmostEqual(lag, 3 / 30 * 1000, delta=40)

    def test_tracking_is_off_when_action_is_not_joint_positions(self):
        action = _busy_action(seed=2)
        unrelated = np.random.default_rng(3).normal(size=action.shape)
        ep = make_episode(action, state=unrelated)
        self.assertEqual(tracking.track_lag_ms(ep), {})
        self.assertEqual(tracking.track_resid(ep), {})

    def test_tracking_is_off_without_state(self):
        self.assertEqual(tracking.track_resid(make_episode(_busy_action(seed=4))), {})

    def test_stalled_joint_raises_the_residual(self):
        action = _busy_action(seed=5)
        state = _follower(action, 2, noise=0.005)
        stalled = state.copy()
        stalled[150:350, 0] = stalled[150, 0]  # joint 0 stops following
        good = tracking.track_resid(make_episode(action, state=state))["arm_all"].value
        bad = tracking.track_resid(make_episode(action, state=stalled))["arm_all"].value
        self.assertGreater(bad, good * 1.3)

    def test_constant_offset_is_not_a_tracking_error(self):
        action = _busy_action(seed=6)
        state = _follower(action, 2, noise=0.005)
        base = tracking.track_resid(make_episode(action, state=state))["arm_all"].value
        offset = tracking.track_resid(make_episode(action, state=state + 0.5))["arm_all"].value
        self.assertAlmostEqual(base, offset, delta=0.05 * max(base, 1e-3) + 0.005)

    def test_injected_jolt_raises_spike_fraction(self):
        action = smooth_action(n=450, dims=4, seed=7)
        state = _follower(action, 2, noise=0.001)
        jolted = state.copy()
        for i in (100, 200, 300):
            jolted[i, :] += 0.6  # a one-frame jump
        clean = tracking.accel_spike_frac(make_episode(action, state=state))[""].value
        bad = tracking.accel_spike_frac(make_episode(action, state=jolted))[""].value
        self.assertLess(clean, 0.01)
        self.assertGreater(bad, clean + 0.005)

    def test_idle_joints_do_not_create_spikes(self):
        action = np.zeros((300, 4))
        state = np.zeros((300, 4)) + np.random.default_rng(8).normal(0, 1e-4, (300, 4))
        self.assertLess(tracking.accel_spike_frac(make_episode(action, state=state))[""].value, 0.01)

    def test_joint_limit_fraction(self):
        action = smooth_action(n=300, dims=2, seed=9)
        state = action.copy()
        bounds = (np.zeros(2), np.array([1.0, 1.0]) * state.max(axis=0))
        near = tracking.joint_limit_frac(make_episode(action, state=state, state_bounds=bounds))[""].value
        wide = (np.full(2, -3.0), np.full(2, 3.0))
        far = tracking.joint_limit_frac(make_episode(action, state=state, state_bounds=wide))[""].value
        self.assertGreater(near, far)
        self.assertEqual(far, 0.0)
        self.assertEqual(tracking.joint_limit_frac(make_episode(action, state=state)), {})

    def test_gripper_chatter_is_counted(self):
        arm = smooth_action(n=300, dims=2, seed=10)
        calm = np.zeros((300, 1))
        calm[100:200] = 1.0
        chatter = (np.arange(300) // 10 % 2).reshape(-1, 1).astype(float)
        names = ["waist", "elbow", "gripper"]
        a = gripper.gripper_flips_per_s(make_episode(np.hstack([arm, calm]), names=names))["gripper_all"].value
        b = gripper.gripper_flips_per_s(make_episode(np.hstack([arm, chatter]), names=names))["gripper_all"].value
        self.assertGreater(b, a * 5)
        self.assertEqual(gripper.gripper_flips_per_s(make_episode(np.hstack([arm, chatter]))), {})  # unnamed


class Batch3MetricTests(unittest.TestCase):
    """Slice 2A batch 3: integrity, language and consistency."""

    def test_length_mismatch(self):
        ep = make_episode(smooth_action(n=100))
        self.assertEqual(integrity.length_mismatch(ep)[""].value, 0.0)
        ep.expected_length = 120
        self.assertEqual(integrity.length_mismatch(ep)[""].value, 20.0)

    def test_video_window_mismatch_in_frames(self):
        ep = make_episode(smooth_action(n=300), fps=30.0)  # 10 s
        ep.videos = {"observation.images.top": VideoWindow("observation.images.top", "x", 5.0, 15.0)}
        self.assertAlmostEqual(integrity.video_window_mismatch(ep)[""].value, 0.0, places=6)
        ep.videos = {"observation.images.top": VideoWindow("observation.images.top", "x", 5.0, 14.0)}  # 1 s short
        mv = integrity.video_window_mismatch(ep)[""]
        self.assertAlmostEqual(mv.value, 30.0, places=6)
        self.assertIn("top", mv.note)

    def test_nonfinite_values(self):
        action = smooth_action(n=100)
        self.assertEqual(integrity.nonfinite_values(make_episode(action))[""].value, 0.0)
        bad = action.copy()
        bad[10, 2] = np.nan
        bad[20, 0] = np.inf
        self.assertEqual(integrity.nonfinite_values(make_episode(bad))[""].value, 2.0)

    def test_too_short(self):
        self.assertEqual(integrity.too_short(make_episode(smooth_action(n=300)))[""].value, 0.0)
        self.assertEqual(integrity.too_short(make_episode(smooth_action(n=5)))[""].value, 1.0)
        self.assertEqual(integrity.too_short(make_episode(smooth_action(n=20), fps=30.0))[""].value, 1.0)  # 0.67 s

    def test_timestamp_dev_is_zero_when_derived_from_fps(self):
        self.assertAlmostEqual(integrity.timestamp_dev(make_episode(smooth_action(n=100)))[""].value, 0.0, places=9)

    def test_task_missing(self):
        self.assertEqual(language.task_missing(make_episode(smooth_action(n=30), tasks=[""]))[""].value, 1.0)
        self.assertEqual(language.task_missing(make_episode(smooth_action(n=30), tasks=[]))[""].value, 1.0)
        self.assertEqual(language.task_missing(make_episode(smooth_action(n=30)))[""].value, 0.0)

    def _raws(self, episodes):
        raws = {}
        for i, ep in enumerate(episodes):
            raws["e%d" % i] = compute_raw(ep, "e%d" % i, [])
        return raws

    def test_action_divergence_flags_the_contrary_episode(self):
        rng = np.random.default_rng(0)
        eps = []
        for i in range(8):
            state = smooth_action(n=200, dims=4, seed=i) + rng.normal(0, 0.01, (200, 4))
            action = state * 2.0 + 0.1  # same state -> action mapping everywhere
            eps.append(make_episode(action, state=state, episode_index=i))
        state = smooth_action(n=200, dims=4, seed=99)
        eps.append(make_episode(-state * 2.0 + 3.0, state=state, episode_index=8))  # contrary mapping
        raws = self._raws(eps)
        out = consistency.action_divergence(raws, {sid: "g" for sid in raws})
        bad = out["e8"][""].value
        self.assertGreater(bad, 3 * np.median([out["e%d" % i][""].value for i in range(8)]))

    def test_action_divergence_needs_enough_peers(self):
        eps = [make_episode(smooth_action(n=100, dims=3, seed=i), state=smooth_action(n=100, dims=3, seed=i)) for i in range(3)]
        raws = self._raws(eps)
        self.assertEqual(consistency.action_divergence(raws, {sid: "g" for sid in raws}), {})

    def test_stats_leverage_counts_stretched_dimensions(self):
        eps = []
        for i in range(6):
            a = smooth_action(n=100, dims=4, seed=i)
            eps.append(make_episode(a, state=a.copy(), episode_index=i))
        a = smooth_action(n=100, dims=4, seed=50)
        a[:, 1] += 5.0  # one dimension far beyond everyone else
        eps.append(make_episode(a, state=a.copy(), episode_index=6))
        raws = self._raws(eps)
        out = consistency.stats_leverage(raws, {sid: "g" for sid in raws})
        self.assertGreaterEqual(out["e6"][""].value, 2.0)  # the action and the state copy
        self.assertEqual(out["e0"][""].value, 0.0)

    def test_stats_leverage_survives_episodes_missing_an_array(self):
        eps = []
        for i in range(6):
            a = smooth_action(n=100, dims=4, seed=i)
            eps.append(make_episode(a, state=a.copy(), episode_index=i))
        broken = smooth_action(n=100, dims=4, seed=60)
        broken[10, 0] = np.nan  # a non-finite action drops that array from the aux data
        eps.append(make_episode(broken, state=broken.copy(), episode_index=6))
        raws = self._raws(eps)
        out = consistency.stats_leverage(raws, {sid: "g" for sid in raws})
        self.assertEqual(out["e0"][""].value, 0.0)

    def test_schema_mismatch_flags_the_minority_source(self):
        eps = [make_episode(smooth_action(n=60), fps=30.0) for _ in range(5)]
        eps.append(make_episode(smooth_action(n=60), fps=50.0))
        raws = self._raws(eps)
        out = integrity.schema_mismatch(raws, {})
        self.assertEqual(out["e0"][""].value, 0.0)
        self.assertEqual(out["e5"][""].value, 1.0)
        self.assertIn("fps", out["e5"][""].note)

    def test_mixed_dataset_is_not_flagged(self):
        eps = [make_episode(smooth_action(n=60), fps=f) for f in (10.0, 20.0, 30.0, 50.0)]
        raws = self._raws(eps)
        out = integrity.schema_mismatch(raws, {})
        self.assertTrue(all(v[""].value == 0.0 for v in out.values()))
        self.assertTrue(any("mixed dataset" in (v[""].note or "") for v in out.values()))

    def test_integrity_verdict_levels(self):
        from lerobot_data_curation.engine.score import _integrity_verdict

        ok = {"frame_gaps": {"value": 0.0}, "timestamp_dev": {"value": 9.9}}  # info only never counts
        self.assertEqual(_integrity_verdict(ok), "pass")
        self.assertEqual(_integrity_verdict({"frame_gaps": {"value": 2.0}}), "fail")
        self.assertEqual(_integrity_verdict({"schema_mismatch": {"value": 1.0}}), "warn")
        self.assertEqual(_integrity_verdict({"video_window_mismatch": {"value": 1.5}}), "pass")  # within tolerance
        self.assertEqual(_integrity_verdict({"video_window_mismatch": {"value": 30.0}}), "fail")
        self.assertEqual(_integrity_verdict({"idle_lead_s": {"value": 1.0}}), "unknown")


class OutlierTests(unittest.TestCase):
    def _run(self, n_clean, corrupt):
        raws = {}
        for i in range(n_clean):
            act = smooth_action(n=300, seed=i) + np.random.default_rng(200 + i).normal(0, 0.002, (300, 6))
            raws["e%d" % i] = compute_raw(make_episode(act, episode_index=i), "e%d" % i, list(METRICS))
        raws["odd"] = compute_raw(corrupt(smooth_action(n=300, seed=777)), "odd", list(METRICS))
        return finalize(raws, list(METRICS), min_group=5)[0]

    def test_outlier_is_flagged_but_never_scored(self):
        rng = np.random.default_rng(3)
        res = self._run(30, lambda a: make_episode(a + rng.normal(0, 0.08, a.shape)))
        odd, clean = res["odd"].metrics, res["e0"].metrics
        self.assertEqual(odd["is_outlier"]["value"], 1.0)
        self.assertGreater(odd["iforest_score"]["value"], clean["iforest_score"]["value"])
        # outlier metrics never appear in any profile group
        from lerobot_data_curation.engine.profiles import GROUPS
        members = {m for g in GROUPS.values() for m in g}
        self.assertTrue({"iforest_score", "novelty_knn", "is_outlier"}.isdisjoint(members))

    def test_small_groups_are_skipped(self):
        res = self._run(3, lambda a: make_episode(a))
        self.assertNotIn("iforest_score", res["e0"].metrics)


NAMES = ["waist", "elbow", "wrist", "gripper"]


def _grasp_episode(open_is="high", stall=True, reopen=True, arm_moves=False, n=400):
    """Arm + gripper. Gripper closes at 100, (optionally) reopens at 160, closes again at 200."""
    arm = np.zeros((n, 3))
    arm[:, 0] = 0.2 * np.sin(2 * np.pi * np.arange(n) / 100)  # same position at 100 and 200
    if arm_moves:
        arm[200:, 0] += 3.0
    lo, hi = (0.0, 1.0)
    g = np.full(n, hi if open_is == "high" else lo)
    closed = lo if open_is == "high" else hi
    g[100:] = closed
    if reopen:
        g[160:200] = hi if open_is == "high" else lo
    action = np.hstack([arm, g[:, None]])
    state = action.copy()
    if stall:  # an object keeps the follower from closing all the way
        stalled = 0.35 * closed + 0.65 * (hi if open_is == "high" else lo)
        state[:, 3] = np.where(g == closed, stalled, g)
    return action, state


class DatasetCheckTests(unittest.TestCase):
    def test_camera_roles(self):
        self.assertEqual(dc.camera_role("observation.images.wrist_left"), "wrist")
        self.assertEqual(dc.camera_role("observation.images.top"), "top")
        self.assertEqual(dc.camera_role("observation.images.ego"), "ego")
        self.assertEqual(dc.camera_role("observation.images.cam_high"), "top")
        self.assertEqual(dc.camera_role("observation.images.zzz"), "unknown")

    def test_joint_units(self):
        self.assertEqual(dc.infer_joint_units(np.random.default_rng(0).uniform(-3, 3, (200, 4))), "radians")
        self.assertEqual(dc.infer_joint_units(np.random.default_rng(0).uniform(-170, 170, (200, 4))), "degrees")
        self.assertEqual(dc.infer_joint_units(np.random.default_rng(0).uniform(-100, 100, (200, 4))), "normalized")

    def test_gripper_open_direction(self):
        high = np.concatenate([np.ones(50), np.zeros(100), np.ones(50)])
        open_is, sure = dc.infer_gripper_open_is({"all": high})
        self.assertEqual((open_is, sure), ("high", True))
        low = 1 - high
        self.assertEqual(dc.infer_gripper_open_is({"all": low}), ("low", True))
        ends_apart = np.concatenate([np.ones(100), np.zeros(100)])
        self.assertFalse(dc.infer_gripper_open_is({"all": ends_apart})[1])
        self.assertEqual(dc.infer_gripper_open_is({}), (None, False))

    def test_source_checks_on_a_leader_follower_episode(self):
        action, state = _grasp_episode()
        ep = make_episode(action, state=state, names=NAMES)
        c = dc.infer_source_checks([ep])
        self.assertEqual(c.action_semantics, "joint_positions")
        self.assertEqual(c.gripper_dims, ["gripper"])
        self.assertEqual(c.gripper_open_is, "high")
        self.assertEqual(dc.assumptions_for(c), {"action_semantics": "joint_positions", "gripper_open_is": "high"})

    def test_unnamed_joints_have_no_gripper(self):
        action, state = _grasp_episode()
        c = dc.infer_source_checks([make_episode(action, state=state)])
        self.assertEqual(c.gripper_dims, [])
        self.assertIsNone(c.gripper_open_is)
        self.assertTrue(any("names missing" in n for n in c.notes))

    def test_overrides_apply_to_every_source(self):
        action, state = _grasp_episode()
        checks = {"a": dc.infer_source_checks([make_episode(action, state=state, names=NAMES)])}
        dc.apply_overrides(checks, action_semantics="other", gripper_open_is="low")
        self.assertEqual(checks["a"].action_semantics, "other")
        self.assertEqual(checks["a"].gripper_open_is, "low")
        dc.apply_overrides(checks, action_semantics="auto", gripper_open_is="auto")  # auto keeps the value
        self.assertEqual(checks["a"].action_semantics, "other")

    def test_balance(self):
        b = dc.balance(["a"] * 10 + ["b"] * 2, ["s1"] * 11 + ["s2"])
        self.assertEqual(b["under_covered"], ["b"])
        self.assertEqual(b["dominant_sources"], ["s1"])


class GripperEventMetricTests(unittest.TestCase):
    def _ep(self, action, state=None, open_is="high", **kw):
        ep = make_episode(action, state=state, names=NAMES, **kw)
        ep.assumptions = {"gripper_open_is": open_is, "action_semantics": "joint_positions"}
        return ep

    def test_regrasp_is_a_recovery(self):
        action, state = _grasp_episode(reopen=True)
        ep = self._ep(action, state)
        self.assertEqual(gripper.recovery_count(ep)[""].value, 1.0)
        self.assertEqual(len(gripper.recovery_events(ep)), 1)

    def test_single_grasp_is_not_a_recovery(self):
        action, state = _grasp_episode(reopen=False)
        self.assertEqual(gripper.recovery_count(self._ep(action, state))[""].value, 0.0)

    def test_regrasp_somewhere_else_is_not_a_recovery(self):
        action, state = _grasp_episode(reopen=True, arm_moves=True)
        self.assertEqual(gripper.recovery_count(self._ep(action, state))[""].value, 0.0)

    def test_recovery_needs_a_known_convention(self):
        action, state = _grasp_episode()
        ep = make_episode(action, state=state, names=NAMES)
        self.assertEqual(gripper.recovery_count(ep), {})
        self.assertEqual(gripper.missed_grasp_frac(ep), {})

    def test_holding_an_object_is_not_a_missed_grasp(self):
        action, state = _grasp_episode(stall=True, reopen=False)
        self.assertEqual(gripper.missed_grasp_frac(self._ep(action, state))["arm_all"].value, 0.0)

    def test_closing_on_nothing_is_a_missed_grasp(self):
        action, state = _grasp_episode(stall=False, reopen=False)
        self.assertEqual(gripper.missed_grasp_frac(self._ep(action, state))["arm_all"].value, 1.0)

    def test_leader_follower_offset_is_ignored(self):
        action, state = _grasp_episode(stall=False, reopen=False)
        state = state.copy()
        state[:, 3] += 0.3  # constant calibration offset
        self.assertEqual(gripper.missed_grasp_frac(self._ep(action, state))["arm_all"].value, 1.0)

    def test_pause_next_to_a_regrasp_is_not_a_stall(self):
        action, state = _grasp_episode(reopen=True)
        # the arm rests while the gripper reopens between 160 and 200
        action[160:200, :3] = action[160, :3]
        ep = self._ep(action, state)
        with_recovery = time_metrics.longest_pause_s(ep)[""].value
        ep_no_convention = make_episode(action, state=state, names=NAMES)
        without = time_metrics.longest_pause_s(ep_no_convention)[""].value
        self.assertLess(with_recovery, without)


class TimelineSpanTests(unittest.TestCase):
    def _run(self, corrupt):
        raws = {}
        for i in range(30):
            act = smooth_action(n=300, seed=i) + np.random.default_rng(300 + i).normal(0, 0.002, (300, 6))
            raws["e%d" % i] = compute_raw(make_episode(act, episode_index=i), "e%d" % i, list(METRICS))
        raws["bad"] = compute_raw(corrupt(smooth_action(n=300, seed=555)), "bad", list(METRICS))
        return finalize(raws, list(METRICS), min_group=5)[0]

    def test_only_flagged_spans_are_tagged(self):
        res = self._run(lambda a: make_episode(np.concatenate([np.repeat(a[:1], 90, 0), a])))
        bad_kinds = {s["kind"] for s in res["bad"].spans}
        self.assertIn("idle", bad_kinds)
        idle = next(s for s in res["bad"].spans if s["kind"] == "idle")
        self.assertIn(idle["severity"], ("warn", "fail"))
        self.assertAlmostEqual(idle["end_s"], 3.0, delta=0.6)
        # clean episodes are not all flagged: most get no idle or pause tag
        flagged = [sid for sid, r in res.items() if sid != "bad" and any(s["kind"] in ("idle", "pause") for s in r.spans)]
        self.assertLess(len(flagged), 10)
        self.assertEqual([s for s in res["e0"].spans if s["kind"] in ("idle", "pause")], [])

    def test_temporal_tags_are_anchored_and_ordered(self):
        from lerobot_data_curation import write

        tags = write.build_temporal_tags(
            "sid",
            [
                {"start_s": 1.0, "end_s": 2.5, "label": "idle at start", "kind": "idle", "severity": "warn"},
                {"start_s": 4.0, "end_s": 4.0, "label": "acceleration spike", "kind": "spike", "severity": "fail"},
            ],
        )
        self.assertEqual([t.start for t in tags], [1_000_000_000, 4_000_000_000])
        self.assertTrue(all(t.end > t.start for t in tags))  # zero-length spans are widened
        self.assertTrue(all(t.anchor == write.TEMPORAL_TAG_ANCHOR for t in tags))
        self.assertEqual(tags[0].tag, "idle at start:warn")


if __name__ == "__main__":
    unittest.main()
