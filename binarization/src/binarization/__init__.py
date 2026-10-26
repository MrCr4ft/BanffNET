"""ROI extraction and tissue segmentation for whole-slide images.

``BinarizationPipeline`` writes ROI thumbnails, binary masks, and QC records.
ROI identifiers use the annotation Name when present, otherwise the positional
index, as resolved by ``wsipatch.resolve_roi_id``. The ``tissue-binarize`` CLI
provides equivalent single-slide and directory entry points.
"""

from __future__ import annotations

# Re-export the shared ASAP annotation API.
from wsipatch.annotations import ASAPAnnotations

from .config import (
    IMAGE_NORM_MEAN,
    IMAGE_NORM_STD,
    QC_FIELDNAMES,
    BinarizationConfig,
    LogConfig,
)
from .extraction import (
    BoundingBox,
    ClippingResult,
    QCRecord,
    ROIExtractor,
    SlideProcessor,
    SlideProcessorError,
)
from .io import (
    CSVWriter,
    FileManager,
    ensure_directories,
    get_biopsy_list,
    setup_logging,
)
from .pipeline import (
    BinarizationPipeline,
    ProcessingStats,
)
from .segmentation import (
    ModelLoadError,
    SegmentationPipeline,
    enhance_contrast_clahe,
    load_segmentation_model,
    select_device,
)

__version__ = "0.2.0"

__all__ = [
    "BinarizationConfig",
    "LogConfig",
    "IMAGE_NORM_MEAN",
    "IMAGE_NORM_STD",
    "QC_FIELDNAMES",
    "BoundingBox",
    "ClippingResult",
    "QCRecord",
    "ROIExtractor",
    "SlideProcessor",
    "SlideProcessorError",
    "ModelLoadError",
    "SegmentationPipeline",
    "enhance_contrast_clahe",
    "load_segmentation_model",
    "select_device",
    "BinarizationPipeline",
    "ProcessingStats",
    "CSVWriter",
    "FileManager",
    "ensure_directories",
    "get_biopsy_list",
    "setup_logging",
    "ASAPAnnotations",
]
