"""GAE lesion saliency maps for BanffNET.

Public symbols load lazily so importing the package does not load model or
slide-processing dependencies.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

__version__ = "0.1.0"

# name -> submodule providing it
_EXPORTS = {
    "ExplainabilityPipeline": "pipeline",
    "ExplainableBanffModel": "model",
    "VARIANT": "model",
    "GIP_TARGETS": "model",
    "build_uni_vit_gae": "backbone",
    "get_uni_vit": "backbone",
    "instrument_attention": "backbone",
    "disable_fused_attention": "backbone",
    "Attention": "backbone",
    "compute_relevance": "gae",
    "saliency_for_patches": "gae",
    "ROIAccumulator": "aggregate",
    "read_roi_rgb": "aggregate",
    "resolve_threshold": "thresholding",
    "otsu_threshold": "thresholding",
    "default_threshold": "thresholding",
    "SaveOptions": "io",
}

__all__ = list(_EXPORTS) + ["render", "setup_logging"]

if TYPE_CHECKING:
    from .aggregate import ROIAccumulator as ROIAccumulator
    from .aggregate import read_roi_rgb as read_roi_rgb
    from .backbone import Attention as Attention
    from .backbone import build_uni_vit_gae as build_uni_vit_gae
    from .backbone import disable_fused_attention as disable_fused_attention
    from .backbone import get_uni_vit as get_uni_vit
    from .backbone import instrument_attention as instrument_attention
    from .gae import compute_relevance as compute_relevance
    from .gae import saliency_for_patches as saliency_for_patches
    from .io import SaveOptions as SaveOptions
    from .model import VARIANT as VARIANT
    from .model import ExplainableBanffModel as ExplainableBanffModel
    from .pipeline import ExplainabilityPipeline as ExplainabilityPipeline
    from .thresholding import default_threshold as default_threshold
    from .thresholding import otsu_threshold as otsu_threshold
    from .thresholding import resolve_threshold as resolve_threshold


def __getattr__(name: str):
    if name in _EXPORTS:
        mod = importlib.import_module(f".{_EXPORTS[name]}", __name__)
        return getattr(mod, name)
    if name in ("render", "setup_logging"):
        if name == "render":
            return importlib.import_module(".render", __name__)
        return importlib.import_module(".logging", __name__).setup_logging
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
