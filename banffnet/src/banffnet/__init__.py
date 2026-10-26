"""BanffNET prediction pipelines, MIL heads, defaults, and output writers.

Use ``BanffNETPipeline.from_staining`` with assets under ``BANFFNET_DATA_ROOT``,
or ``from_paths`` with explicit model and configuration paths. ``BanffPredictor``
provides slide, ROI, and patch inference from precomputed features.
"""

from __future__ import annotations

# Defaults
from .defaults import (
    DATA_ROOT,
    DEFAULT_MPP,
    DEFAULT_PATCH_SIZE,
    STAININGS,
    StainingPaths,
    staining_paths,
)
from .io import ResultsWriter

# Models
from .models import (
    MODELS,
    LinearGatedRatio,
    LinearNoisyOR,
    logistic_sum_model,
)
from .pipeline import (
    BanffNETPipeline,
    SlideResult,
    staining_cache,
)

# Predictor
from .predictor import BanffPredictor

__version__ = "0.1.0"

__all__ = [
    # Defaults
    "DATA_ROOT",
    "DEFAULT_MPP",
    "DEFAULT_PATCH_SIZE",
    "STAININGS",
    "StainingPaths",
    "staining_paths",
    # Models
    "LinearGatedRatio",
    "LinearNoisyOR",
    "MODELS",
    "logistic_sum_model",
    # Predictor
    "BanffPredictor",
    "BanffNETPipeline",
    "SlideResult",
    "staining_cache",
    "ResultsWriter",
]
