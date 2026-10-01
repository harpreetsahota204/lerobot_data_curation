"""Unlisted operators backing the React panel.

The panel is a JS component (``src/``) with no Python Panel class. These
operators are its backend, called through ``useOperatorExecutor`` and hidden from
the operator browser.
"""

import base64
import logging

import fiftyone.operators as foo

from .engine import dataset_checks
from .engine.detail import episode_detail, thumbnail_times
from .engine.frames import frame_jpeg
from .engine.reader import EpisodeReadError, LeRobotReader
from .panel_data import build_panel_data
from .write import RUN_KEY

logger = logging.getLogger(__name__)

MAX_CAMERAS = 2


class GetPanelData(foo.Operator):
    """Serves the panel its whole payload in one call."""

    @property
    def config(self):
        return foo.OperatorConfig(name="lr_get_panel_data", unlisted=True)

    def execute(self, ctx):
        return build_panel_data(ctx.dataset, ctx.view)


class OpenEpisode(foo.Operator):
    """Opens one episode in the viewer and toasts why it ranks where it does."""

    @property
    def config(self):
        return foo.OperatorConfig(name="lr_open_episode", unlisted=True)

    def execute(self, ctx):
        sample_id = ctx.params.get("sample_id")
        if not sample_id:
            return {}
        profile = ctx.params.get("profile") or "policy"
        ctx.ops.open_sample(sample_id)

        sample = ctx.dataset[sample_id]
        score = sample.get_field("lr_score_%s" % profile) if sample.has_field("lr_score_%s" % profile) else None
        driver = sample.get_field("lr_driver_%s" % profile) if sample.has_field("lr_driver_%s" % profile) else None
        if score is not None and driver:
            ctx.ops.notify("Score %.2f, driven by the '%s' group." % (score, driver))
        return {}


class TagEpisodes(foo.Operator):
    """Records a reviewer's verdict as a sample tag.

    An empty `sample_ids` falls back to the current view, which is what makes
    "tag everything I am looking at" work after the user filters to a cohort.
    """

    @property
    def config(self):
        return foo.OperatorConfig(name="lr_tag_episodes", unlisted=True)

    def execute(self, ctx):
        tag = ctx.params["tag"]
        ids = ctx.params.get("sample_ids") or []
        target = ctx.dataset.select(ids) if ids else ctx.view
        target.tag_samples(tag)
        count = len(target)
        ctx.ops.notify("Tagged %d episode(s) as '%s'." % (count, tag))
        return {"tagged": count}


class ShowEpisodes(foo.Operator):
    """Filters the samples panel to episodes picked in a panel chart."""

    @property
    def config(self):
        return foo.OperatorConfig(name="lr_show_episodes", unlisted=True)

    def execute(self, ctx):
        ids = ctx.params.get("sample_ids") or []
        description = ctx.params.get("description") or "selection"
        if not ids:
            ctx.ops.notify("No episodes match %s." % description)
            return {}
        ctx.ops.set_view(view=ctx.dataset.select(ids, ordered=True))
        ctx.ops.notify(
            "Showing %d episode(s): %s. Clear the view bar to see everything again."
            % (len(ids), description)
        )
        return {}


class PromptVision(foo.Operator):
    """Opens the compute form on its Camera tab, with every camera metric ticked."""

    @property
    def config(self):
        return foo.OperatorConfig(name="lr_prompt_vision", unlisted=True)

    def execute(self, ctx):
        from .operators import camera_metric_names

        ctx.trigger(
            "lerobot-data-curation/lr_compute_quality",
            params={"tab": "CAMERA", "camera_cfg": {"metric_%s" % name: True for name in camera_metric_names()}},
        )
        return {}


class PromptCompute(foo.Operator):
    """Opens the compute form from the panel's empty state."""

    @property
    def config(self):
        return foo.OperatorConfig(name="lr_prompt_compute", unlisted=True)

    def execute(self, ctx):
        ctx.trigger("lerobot-data-curation/lr_compute_quality")
        return {}


def _assumptions(dataset, source_id):
    """The confirmed dataset-level assumptions the last run used for one source."""
    try:
        recorded = (dataset.load_run_results(RUN_KEY).dataset_checks or {}).get(source_id)
    except Exception:  # noqa: BLE001 - no run record: the inspector still works
        return {}
    if not recorded:
        return {}
    checks = dataset_checks.SourceChecks(
        action_semantics=recorded.get("action_semantics", "unknown"),
        gripper_dims=recorded.get("gripper_dims", []),
        gripper_open_is=recorded.get("gripper_open_is"),
    )
    return dataset_checks.assumptions_for(checks)


class GetEpisodeDetail(foo.Operator):
    """One episode's traces, speed profile, gripper timeline, flagged spans and thumbnails.

    Called when a row is selected, so the main payload stays small. Decoding is
    sparse: at most a few frames from the first two cameras.
    """

    @property
    def config(self):
        return foo.OperatorConfig(name="lr_get_episode_detail", unlisted=True)

    def execute(self, ctx):
        sample_id = ctx.params.get("sample_id")
        if not sample_id:
            return {"error": "no sample_id"}
        dataset = ctx.dataset
        sample = dataset[sample_id]
        overrides = {}
        try:
            overrides = dict(dataset.get_run_info(RUN_KEY).config.overrides or {})
        except Exception:  # noqa: BLE001
            pass
        reader = LeRobotReader(dataset, overrides=overrides)
        episode = reader.read(sample)
        if isinstance(episode, EpisodeReadError):
            return {"error": str(episode)}

        episode.assumptions = _assumptions(dataset, episode.source_id)
        detail = episode_detail(episode)
        detail["sample_id"] = sample_id
        detail["task"] = sample.task
        detail["episode"] = "%s / ep %s" % (episode.source_id[-6:], episode.episode_index)

        thumbs = []
        times = thumbnail_times(episode, detail["spans"])
        for camera, window in list(episode.videos.items())[:MAX_CAMERAS]:
            for t in times:
                jpeg = frame_jpeg(window.path, window.from_timestamp + t)
                if jpeg is not None:
                    thumbs.append(
                        {"camera": camera.split(".")[-1], "t": t, "jpeg": base64.b64encode(jpeg).decode("ascii")}
                    )
        detail["thumbnails"] = thumbs
        return detail
