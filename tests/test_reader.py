"""Reader tests on a tiny synthetic LeRobot v3 directory (no FiftyOne dataset needed).

Run from the plugin's parent folder:
    python -m unittest lerobot_data_curation.tests.test_reader -v
"""

import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot_data_curation.engine.reader import EpisodeReadError, LeRobotReader

DIM = 4


def _write_dataset(root, episodes, features=None, swap_rows=()):
    """Writes one parquet file holding several episodes. Returns the info dict."""
    os.makedirs(os.path.join(root, "meta"))
    os.makedirs(os.path.join(root, "data", "chunk-000"))
    features = features or {
        "action": {"dtype": "float32", "shape": [DIM], "names": {"motors": ["a", "b", "c", "d"]}},
        "observation.state": {"dtype": "float32", "shape": [DIM], "names": None},
        "observation.images.top": {"dtype": "video", "shape": [4, 4, 3]},
        "timestamp": {"dtype": "float32", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
    }
    info = {
        "fps": 10,
        "robot_type": "test",
        "features": features,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    }
    with open(os.path.join(root, "meta", "info.json"), "w") as f:
        json.dump(info, f)

    ep_col, fi_col, ts_col, act_col, st_col = [], [], [], [], []
    for ep, n in episodes:
        for i in range(n):
            ep_col.append(ep)
            fi_col.append(i)
            ts_col.append(i / 10)
            act_col.append([float(ep * 100 + i)] * DIM)
            st_col.append([float(-ep * 100 - i)] * DIM)
    for a, b in swap_rows:  # recorded out of order
        for col in (fi_col, ts_col, act_col, st_col):
            col[a], col[b] = col[b], col[a]
    columns = {
        "episode_index": pa.array(ep_col, pa.int64()),
        "frame_index": pa.array(fi_col, pa.int64()),
        "timestamp": pa.array(ts_col, pa.float32()),
    }
    if "action" in features:
        columns["action"] = pa.array(act_col, pa.list_(pa.float32()))
    if "observation.state" in features:
        columns["observation.state"] = pa.array(st_col, pa.list_(pa.float32()))
    pq.write_table(pa.table(columns), os.path.join(root, "data", "chunk-000", "file-000.parquet"))
    return info


def _dataset(root):
    return SimpleNamespace(
        media_sources=[{"kind": "lerobot-episode", "id": "s1", "loc": root,
                        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"}]
    )


def _sample(sid, episode, n, tasks=("pick up the cube",)):
    ref = SimpleNamespace(
        key="s1/%d" % episode, episode=episode, data=[0, 0, 0, n], videos={}, tasks=list(tasks)
    )
    return SimpleNamespace(id=sid, media_reference=ref)


class ReaderTests(unittest.TestCase):
    def test_batch_read_splits_episodes_in_one_file(self):
        with tempfile.TemporaryDirectory() as root:
            _write_dataset(root, [(0, 5), (1, 8)])
            reader = LeRobotReader(_dataset(root))
            got = {s.id: ep for s, ep in reader.read_many([_sample("a", 0, 5), _sample("b", 1, 8)])}
            self.assertEqual((got["a"].length, got["b"].length), (5, 8))
            self.assertEqual(got["b"].action.shape, (8, DIM))
            self.assertEqual(got["b"].action[0, 0], 100.0)  # episode 1, frame 0
            self.assertEqual(got["a"].state[2, 0], -2.0)
            self.assertEqual(got["a"].fps, 10.0)

    def test_raw_order_is_kept_and_metrics_see_sorted_arrays(self):
        with tempfile.TemporaryDirectory() as root:
            _write_dataset(root, [(0, 6)], swap_rows=[(2, 3)])
            ep = LeRobotReader(_dataset(root)).read(_sample("a", 0, 6))
            self.assertEqual(list(ep.raw_frame_index), [0, 1, 3, 2, 4, 5])
            self.assertEqual(list(ep.frame_index), [0, 1, 2, 3, 4, 5])
            np.testing.assert_array_equal(ep.action[:, 0], np.arange(6, dtype=float))

    def test_missing_action_feature_gives_none_not_an_error(self):
        with tempfile.TemporaryDirectory() as root:
            features = {
                "observation.state": {"dtype": "float32", "shape": [DIM], "names": None},
                "timestamp": {"dtype": "float32", "shape": [1]},
                "frame_index": {"dtype": "int64", "shape": [1]},
                "episode_index": {"dtype": "int64", "shape": [1]},
            }
            _write_dataset(root, [(0, 4)], features=features)
            ep = LeRobotReader(_dataset(root)).read(_sample("a", 0, 4))
            self.assertIsNone(ep.action)
            self.assertIsNotNone(ep.state)
            self.assertIsNone(ep.feature_map.action_key)

    def test_unreadable_file_is_an_error_value_not_an_exception(self):
        with tempfile.TemporaryDirectory() as root:
            _write_dataset(root, [(0, 4)])
            os.remove(os.path.join(root, "data", "chunk-000", "file-000.parquet"))
            reader = LeRobotReader(_dataset(root))
            (sample, result), = list(reader.read_many([_sample("a", 0, 4)]))
            self.assertIsInstance(result, EpisodeReadError)

    def test_episode_absent_from_file_is_an_error_value(self):
        with tempfile.TemporaryDirectory() as root:
            _write_dataset(root, [(0, 4)])
            reader = LeRobotReader(_dataset(root))
            results = {s.id: r for s, r in reader.read_many([_sample("a", 0, 4), _sample("ghost", 7, 4)])}
            self.assertNotIsInstance(results["a"], EpisodeReadError)
            self.assertIsInstance(results["ghost"], EpisodeReadError)

    def test_cloud_source_is_refused(self):
        ds = SimpleNamespace(media_sources=[{"kind": "lerobot-episode", "id": "s1", "loc": "gs://bucket/x"}])
        with self.assertRaises(NotImplementedError):
            LeRobotReader(ds)


if __name__ == "__main__":
    unittest.main()
