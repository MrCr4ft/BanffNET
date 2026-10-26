"""Plan, persist, and read whole-slide image patches.

``plan_slide`` and ``plan_directory`` build tissue-filtered patch grids;
their ``_multi`` variants support multiple targets. ``save_index`` and
``load_index`` handle binary ``.wsix`` and JSON plans. ``PatchReader`` reads
batches or materializes patches as HDF5. ROI identifiers use shared
``resolve_roi_id`` and ``sanitize_roi_id`` helpers.
"""

from .annotations import (
    Annotation,
    AnnotationParseError,
    ASAPAnnotations,
    asap_to_rois,
    load_asap_rois,
)
from .io import load_index, load_json, load_wsix, save_index, save_json, save_wsix
from .planner import (
    build_slide_index,
    build_slide_index_from_metadata,
    plan_directory,
    plan_directory_multi,
    plan_slide,
    plan_slide_multi,
)
from .reader import PatchLocation, PatchReader, PatchSampler
from .types import (
    ROIGrid,
    ROISpec,
    SlideIndex,
    TargetSpec,
    resolve_roi_id,
    sanitize_roi_id,
)

__all__ = [
    "TargetSpec",
    "ROISpec",
    "ROIGrid",
    "SlideIndex",
    "resolve_roi_id",
    "sanitize_roi_id",
    "plan_slide",
    "plan_slide_multi",
    "plan_directory",
    "plan_directory_multi",
    "build_slide_index",
    "build_slide_index_from_metadata",
    "save_index",
    "load_index",
    "save_wsix",
    "load_wsix",
    "save_json",
    "load_json",
    "PatchReader",
    "PatchSampler",
    "PatchLocation",
    "Annotation",
    "ASAPAnnotations",
    "AnnotationParseError",
    "asap_to_rois",
    "load_asap_rois",
]

__version__ = "0.2.0"
