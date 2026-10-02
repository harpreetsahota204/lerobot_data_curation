"""Reads native FiftyOne LeRobot episode samples into numpy arrays.

The LeRobot analogue of ``demo_quality_scorer.engine.decode``. A sample built
by ``fo.types.LeRobotDataset`` carries only a ``LeRobotEpisodeReference``
(``data=[chunk, file, first_row, last_row]``, ``videos={cam: [chunk, file,
from_ts, to_ts]}``); this module resolves that into arrays on demand.

Design points:

- Arrays are read through a :class:`FeatureMap`, so datasets that do not use
  ``observation.state`` / ``action`` still work (or report "missing").
- Frame order is kept as recorded in ``raw_frame_index`` / ``raw_timestamps``
  so integrity checks can see gaps and duplicates. ``state`` / ``action`` /
  ``timestamps`` are sorted by ``frame_index`` for the metrics.
- :meth:`LeRobotReader.read_many` reads each parquet file once per batch.

No FiftyOne write dependency and no scoring logic.
"""

import json
import os
import re
from dataclasses import dataclass, field

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

_FRAME_COLUMNS = ("episode_index", "frame_index", "timestamp")

# Columns that are bookkeeping, never a state or action vector.
_META_KEYS = frozenset(
    {
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
        "next.done",
        "next.reward",
        "next.success",
    }
)
_VECTOR_DTYPES = ("float16", "float32", "float64", "int8", "int16", "int32", "int64")


@dataclass
class FeatureMap:
    """Which dataset columns hold state, action and cameras.

    Attributes:
        state_key: column holding proprioceptive state, or None
        action_key: column holding actions, or None
        camera_keys: video / image feature keys
        notes: human-readable reasons behind any missing or ambiguous entry
    """

    state_key: "str | None" = None
    action_key: "str | None" = None
    camera_keys: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def as_dict(self):
        return {
            "state_key": self.state_key,
            "action_key": self.action_key,
            "camera_keys": list(self.camera_keys),
            "notes": list(self.notes),
        }


def _vector_keys(info):
    return [
        k
        for k, v in info["features"].items()
        if k not in _META_KEYS
        and v.get("dtype") in _VECTOR_DTYPES
        and len(v.get("shape") or []) == 1
        and not k.startswith("annotation")
    ]


def _pick(keys, exact, pattern, label, notes):
    """Exact match first, then a sole pattern match, else None with a note."""
    for name in exact:
        if name in keys:
            return name
    matches = [k for k in keys if re.search(pattern, k)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        notes.append("%s is ambiguous: %s" % (label, ", ".join(sorted(matches))))
    else:
        notes.append("no %s column found" % label)
    return None


def infer_feature_map(info, overrides=None):
    """Infers a :class:`FeatureMap` from a source's ``meta/info.json``.

    Args:
        info: the parsed ``info.json``
        overrides (None): ``{"state_key": ..., "action_key": ...}``. An override
            is applied only when the key exists in this source

    Returns:
        a :class:`FeatureMap`
    """
    keys = _vector_keys(info)
    notes = []
    action_key = _pick(
        keys, ("action",), r"(^|[._])actions?($|[._])", "action", notes
    )
    state_key = _pick(
        keys,
        ("observation.state",),
        r"(^|[._])(joint_)?states?($|[._])",
        "state",
        notes,
    )
    for name, value in (overrides or {}).items():
        if value and value in info["features"]:
            if name == "state_key":
                state_key = value
            elif name == "action_key":
                action_key = value

    cameras = [
        k for k, v in info["features"].items() if v.get("dtype") in ("video", "image")
    ]
    return FeatureMap(
        state_key=state_key,
        action_key=action_key,
        camera_keys=cameras,
        notes=notes,
    )


def picked_feature_map(info, picks):
    """A :class:`FeatureMap` from the user's picks alone, with no inference.

    A pick that names a column this source does not have is dropped, with a note.
    Cameras are read from the declared features (a fact, not a guess).
    """
    keys = _vector_keys(info)
    notes = []
    chosen = {}
    for name in ("state_key", "action_key"):
        value = (picks or {}).get(name)
        if not value:
            continue
        if value in keys:
            chosen[name] = value
        else:
            notes.append("%s %r is not a vector feature of this source" % (name.replace("_key", ""), value))
    cameras = [k for k, v in info["features"].items() if v.get("dtype") in ("video", "image")]
    return FeatureMap(
        state_key=chosen.get("state_key"),
        action_key=chosen.get("action_key"),
        camera_keys=cameras,
        notes=notes,
    )


@dataclass
class VideoWindow:
    """One camera's slice of a (shared) MP4 file."""

    camera: str
    path: str
    from_timestamp: float
    to_timestamp: float


@dataclass
class EpisodeData:
    """Everything the scoring engine needs about one episode."""

    episode_index: int
    source_id: str
    fps: float
    length: int  # rows actually read
    expected_length: int  # rows the reference promises
    tasks: list
    timestamps: np.ndarray  # (T,) sorted by frame_index
    frame_index: np.ndarray  # (T,) sorted
    raw_frame_index: np.ndarray  # (T,) as recorded
    raw_timestamps: np.ndarray  # (T,) as recorded
    state: "np.ndarray | None"  # (T, D) or None
    action: "np.ndarray | None"  # (T, D) or None
    state_names: list = field(default_factory=list)
    action_names: list = field(default_factory=list)
    state_range: "np.ndarray | None" = None  # (D,) max - min from stats.json, or None
    action_range: "np.ndarray | None" = None
    state_bounds: "tuple | None" = None  # (min, max) arrays from stats.json, or None
    action_bounds: "tuple | None" = None
    videos: dict = field(default_factory=dict)  # camera -> VideoWindow
    robot_type: "str | None" = None
    feature_map: "FeatureMap | None" = None
    # Confirmed dataset-level assumptions for this episode's source, e.g.
    # {"action_semantics": "joint_positions", "gripper_open_is": "high"}
    assumptions: dict = field(default_factory=dict)
    # Seams for the camera metrics (see engine/metrics/camera.py). `decoder` is any object
    # with the functions of engine/decode.py, or None for the real one; the harness swaps
    # in a corrupting wrapper. `cache` holds small per-camera results (never raw frames),
    # so the five camera metrics decode each video once between them.
    decoder: "object | None" = field(default=None, repr=False)
    cache: dict = field(default_factory=dict, repr=False)
    # The user's joint groups over the action's dimensions (see engine/picks.py): a list of
    # {"name", "role", "dims", optional "arm"}. Empty means no group was picked, and every
    # metric that needs one switches off.
    groups: list = field(default_factory=list)


class EpisodeReadError(Exception):
    """One episode could not be read. Carried as a value by ``read_many``."""


class LeRobotReader:
    """Resolves episode samples of one FiftyOne dataset into `EpisodeData`.

    Args:
        dataset: a ``media_type="multimodal"`` dataset built with
            ``fo.types.LeRobotDataset``
        overrides (None): ``{"state_key": ..., "action_key": ...}`` applied to
            every source that has the named column. Only used when `picks` is None
        picks (None): the user's picks (see ``engine/picks.py``). When given, nothing is
            inferred: the state and action arrays are exactly the picked ones, and the
            joint groups come from the picks. When None the arrays are inferred from
            the standard LeRobot names, which the harness and tests rely on
    """

    def __init__(self, dataset, overrides=None, picks=None):
        self._picks = picks
        self._sources = {}
        self._maps = {}
        self._stats = {}
        for src in dataset.media_sources or []:
            if src.get("kind") != "lerobot-episode":
                continue
            root = src["loc"]
            if "://" in root:
                raise NotImplementedError(
                    "Cloud LeRobot sources are not supported yet: %s" % root
                )
            with open(os.path.join(root, "meta", "info.json")) as f:
                info = json.load(f)
            self._sources[src["id"]] = (src, info)
            self._maps[src["id"]] = (
                picked_feature_map(info, picks) if picks is not None else infer_feature_map(info, overrides)
            )
        if not self._sources:
            raise ValueError(
                "Dataset has no LeRobot media sources; import it with "
                "fo.types.LeRobotDataset"
            )

    # -- source metadata -------------------------------------------------

    @property
    def source_ids(self):
        return list(self._sources)

    def feature_map(self, source_id):
        return self._maps[source_id]

    def info(self, source_id):
        return self._sources[source_id][1]

    def source_for(self, reference):
        source_id = reference.key.rpartition("/")[0]
        if source_id not in self._sources:
            raise EpisodeReadError("Unknown media source %r" % source_id)
        return source_id, self._sources[source_id][0], self._sources[source_id][1]

    def _groups(self, fmap, matrix_dim):
        """The picked joint groups that fit this source's action (none when nothing was picked)."""
        if self._picks is None:
            return []
        from .picks import groups_for_source

        return groups_for_source(self._picks, matrix_dim)

    def feature_names(self, source_id, feature):
        """Per-dimension names of a feature, with a positional fallback."""
        if feature is None:
            return []
        feat = self.info(source_id)["features"].get(feature)
        if feat is None:
            return []
        return _flatten_names(feat.get("names"), feature, feat["shape"])

    def feature_bounds(self, source_id, feature):
        """``(min, max)`` arrays from ``meta/stats.json``, or None."""
        if feature is None:
            return None
        key = (source_id, feature)
        if key not in self._stats:
            root = self._sources[source_id][0]["loc"]
            path = os.path.join(root, "meta", "stats.json")
            value = None
            if os.path.isfile(path):
                try:
                    with open(path) as f:
                        stats = json.load(f).get(feature)
                    if stats and "min" in stats and "max" in stats:
                        value = (
                            np.asarray(stats["min"], float).reshape(-1),
                            np.asarray(stats["max"], float).reshape(-1),
                        )
                except (OSError, ValueError):
                    value = None
            self._stats[key] = value
        return self._stats[key]

    def feature_range(self, source_id, feature):
        """Per-dimension ``max - min`` from ``meta/stats.json``, or None."""
        bounds = self.feature_bounds(source_id, feature)
        return None if bounds is None else bounds[1] - bounds[0]

    # -- reading ---------------------------------------------------------

    def read(self, sample):
        """Reads one episode. Raises :class:`EpisodeReadError` on failure."""
        for _, result in self.read_many([sample]):
            if isinstance(result, EpisodeReadError):
                raise result
            return result

    def read_many(self, samples):
        """Reads many episodes, touching each parquet file once.

        Yields:
            ``(sample, EpisodeData | EpisodeReadError)`` pairs, grouped by
            parquet file rather than in input order
        """
        groups = {}
        for sample in samples:
            ref = sample.media_reference
            try:
                source_id, src, _ = self.source_for(ref)
            except EpisodeReadError as e:
                yield sample, e
                continue
            chunk, file_, _, _ = ref.data
            path = os.path.join(
                src["loc"],
                src["data_path"].format(chunk_index=chunk, file_index=file_),
            )
            groups.setdefault((source_id, path), []).append(sample)

        for (source_id, path), group in groups.items():
            try:
                tables = self._read_file(source_id, path, group)
            except Exception as e:  # unreadable file: fail every episode in it
                err = EpisodeReadError("%s: %s" % (os.path.basename(path), e))
                for sample in group:
                    yield sample, err
                continue
            for sample in group:
                try:
                    yield sample, self._build(source_id, sample, tables)
                except Exception as e:
                    yield sample, EpisodeReadError(
                        "episode %s: %s" % (sample.media_reference.episode, e)
                    )

    def _read_file(self, source_id, path, group):
        fmap = self._maps[source_id]
        want = [k for k in (fmap.state_key, fmap.action_key) if k]
        episodes = sorted({s.media_reference.episode for s in group})
        table = pq.read_table(
            path,
            columns=list(_FRAME_COLUMNS) + want,
            filters=[("episode_index", "in", episodes)],
        )
        ep_col = table["episode_index"].to_numpy()
        out = {}
        for ep in episodes:
            rows = np.flatnonzero(ep_col == ep)
            out[ep] = (table, rows)
        return out

    def _build(self, source_id, sample, tables):
        ref = sample.media_reference
        src, info = self._sources[source_id]
        fmap = self._maps[source_id]
        table, rows = tables[ref.episode]
        if len(rows) == 0:
            raise EpisodeReadError("no rows in the data file")

        sub = table.take(pa.array(rows))
        raw_fi = sub["frame_index"].to_numpy().astype(np.int64)
        raw_ts = sub["timestamp"].to_numpy().astype(np.float64)
        order = np.argsort(raw_fi, kind="stable")

        def matrix(key):
            if key is None:
                return None
            feat = info["features"][key]
            dim = int(np.prod(feat["shape"]))
            return _to_matrix(sub[key], dim)[order]

        videos = {}
        for cam, (vc, vf, fts, tts) in (ref.videos or {}).items():
            videos[cam] = VideoWindow(
                camera=cam,
                path=os.path.join(
                    src["loc"],
                    src["video_path"].format(
                        video_key=cam, chunk_index=vc, file_index=vf
                    ),
                ),
                from_timestamp=float(fts),
                to_timestamp=float(tts),
            )

        return EpisodeData(
            episode_index=ref.episode,
            source_id=source_id,
            fps=float(info["fps"]),
            length=int(len(rows)),
            expected_length=int(ref.data[3] - ref.data[2]),
            tasks=list(ref.tasks or []),
            timestamps=raw_ts[order],
            frame_index=raw_fi[order],
            raw_frame_index=raw_fi,
            raw_timestamps=raw_ts,
            state=matrix(fmap.state_key),
            action=matrix(fmap.action_key),
            state_names=self.feature_names(source_id, fmap.state_key),
            action_names=self.feature_names(source_id, fmap.action_key),
            state_range=self.feature_range(source_id, fmap.state_key),
            action_range=self.feature_range(source_id, fmap.action_key),
            state_bounds=self.feature_bounds(source_id, fmap.state_key),
            action_bounds=self.feature_bounds(source_id, fmap.action_key),
            videos=videos,
            robot_type=info.get("robot_type"),
            feature_map=fmap,
            groups=self._groups(fmap, matrix_dim=None if fmap.action_key is None else int(np.prod(info["features"][fmap.action_key]["shape"]))),
        )


def _to_matrix(column, dim):
    """A (list or fixed-size-list) parquet column as a (T, dim) float array."""
    chunk = column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column
    try:
        flat = chunk.flatten().to_numpy(zero_copy_only=False)
        if flat.size == len(chunk) * dim:
            return flat.reshape(len(chunk), dim).astype(np.float64)
    except (pa.ArrowInvalid, AttributeError, ValueError):
        pass
    return np.stack(chunk.to_numpy(zero_copy_only=False)).astype(np.float64)


def _flatten_names(names, feature, shape):
    """`names` is a list, a ``{"motors": [...]}`` dict, or missing."""
    size = int(np.prod(shape))
    if isinstance(names, dict):
        names = next(iter(names.values()), None)
    if isinstance(names, list) and len(names) == size:
        return [str(n) for n in names]
    return ["%s[%d]" % (feature, i) for i in range(size)]


def is_named(names, feature):
    """True when names came from the dataset rather than the positional fallback."""
    return bool(names) and not names[0].startswith("%s[" % feature)


def slug(text):
    """A field-name-safe version of a signal or camera name."""
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(text)).strip("_").lower()
