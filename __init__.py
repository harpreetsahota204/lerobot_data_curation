"""LeRobot Data Curation: episode-level curation for LeRobot v3 datasets."""

from .operators import ComputeQuality, ExportKept
from .panel_ops import (
    GetEpisodeDetail,
    GetPanelData,
    OpenEpisode,
    PromptCompute,
    PromptExport,
    ShowEpisodes,
    TagEpisodes,
)


def register(p):
    """FiftyOne's plugin entry point.

    Only `ComputeQuality` is listed in the operator browser. The rest are the
    panel's unlisted backend and must still be registered to be callable from it.
    """
    p.register(ComputeQuality)
    p.register(ExportKept)
    p.register(GetPanelData)
    p.register(GetEpisodeDetail)
    p.register(OpenEpisode)
    p.register(TagEpisodes)
    p.register(ShowEpisodes)
    p.register(PromptCompute)
    p.register(PromptExport)
