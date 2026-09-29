"""Synthetic episodes for tests."""

import numpy as np

from lerobot_data_curation.engine.reader import EpisodeData, FeatureMap


def smooth_action(n=300, dims=6, fps=30.0, seed=0):
    """Slow, smooth joint trajectories (bell-shaped speed)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n)
    base = 0.5 - 0.5 * np.cos(2 * np.pi * t)  # 0 -> 1 -> 0 over the episode
    amp = rng.uniform(0.5, 1.5, size=dims)
    return base[:, None] * amp[None, :]


def make_episode(
    action,
    fps=30.0,
    tasks=("pick up the cube and place it in the box",),
    names=None,
    frame_index=None,
    robot_type="test_robot",
    episode_index=0,
    state=None,
    state_bounds=None,
):
    n = len(action)
    fi = np.arange(n) if frame_index is None else np.asarray(frame_index)
    order = np.argsort(fi, kind="stable")
    ts = fi / fps
    action_names = names or ["action[%d]" % i for i in range(action.shape[1])]
    return EpisodeData(
        episode_index=episode_index,
        source_id="src",
        fps=fps,
        length=n,
        expected_length=n,
        tasks=list(tasks),
        timestamps=ts[order],
        frame_index=fi[order],
        raw_frame_index=fi,
        raw_timestamps=ts,
        state=None if state is None else np.asarray(state, dtype=np.float64)[order],
        action=np.asarray(action, dtype=np.float64)[order],
        state_names=list(action_names) if state is not None else [],
        action_names=action_names,
        state_range=None,
        action_range=None,
        state_bounds=state_bounds,
        videos={},
        robot_type=robot_type,
        feature_map=FeatureMap(action_key="action", state_key="observation.state" if state is not None else None),
    )
