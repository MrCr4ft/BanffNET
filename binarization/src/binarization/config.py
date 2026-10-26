"""Configuration and preprocessing constants for tissue binarization."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class BinarizationConfig:
    """Processing parameters for the tissue binarization pipeline.

    Parameters
    ----------
    roi_dir, mask_dir
        Output directories. Created on first :class:`FileManager` access.
    model_path
        Path to a segmentation-model checkpoint readable by
        :func:`segmentation_models_pytorch.UnetPlusPlus`.
    architecture, encoder_name
        Model topology. Defaults to UNet++ with a MobileNetV2 encoder.
    input_shape
        Side length of the square network input. ROIs are resized to
        ``(input_shape, input_shape)`` before inference.
    epsilon_overlap
        Minimum overlap (as a fraction of the requested ROI area) between
        an annotation bounding box and the slide. Annotations below this
        are skipped with reason ``tiny_overlap``.
    use_entire_wsi_if_no_annotation
        When no annotation XML is available, treat the entire slide as a
        single ROI. If ``False``, slides without annotations are skipped.
    apply_contrast_enhancement, clahe_*
        CLAHE preprocessing controls (applied per-ROI before inference).
    probability_threshold
        Sigmoid output is thresholded at this value to produce the binary
        mask.
    apply_morphology_postprocessing, closing_kernel_size
        Optional morphological closing to fill small holes in the mask.
    batch_size
        Inference batch size; ``None`` triggers auto-detection by
        doubling until the first OOM.
    biopsy_ext
        File extension for :func:`get_biopsy_list` discovery (CLI / batch
        mode only).
    """

    # Required output paths + model.
    roi_dir: str
    mask_dir: str
    model_path: str

    # Model.
    architecture: str = "UnetPlusPlus"
    encoder_name: str = "mobilenet_v2"

    # Extraction.
    input_shape: int = 1024
    epsilon_overlap: float = 0.02
    use_entire_wsi_if_no_annotation: bool = True

    # Pre-processing.
    apply_contrast_enhancement: bool = False
    clahe_clip_limit: float = 1.0
    clahe_tile_size: int = 32

    # Segmentation.
    probability_threshold: float = 0.5
    apply_morphology_postprocessing: bool = True
    closing_kernel_size: int = 7
    batch_size: Optional[int] = None  # None -> auto-detect

    # Discovery (used by CLI / batch helpers only).
    biopsy_ext: str = "tiff"

    @property
    def fixed_shape(self) -> Tuple[int, int]:
        """Square network input as a ``(w, h)`` tuple."""
        return (self.input_shape, self.input_shape)

    def __post_init__(self) -> None:
        if self.input_shape <= 0:
            raise ValueError(f"input_shape must be positive, got {self.input_shape}")
        if not 0 < self.epsilon_overlap < 1:
            raise ValueError(f"epsilon_overlap must be in (0, 1), got {self.epsilon_overlap}")
        if not 0 < self.probability_threshold < 1:
            raise ValueError(
                f"probability_threshold must be in (0, 1), got {self.probability_threshold}"
            )
        if self.closing_kernel_size < 1:
            raise ValueError(f"closing_kernel_size must be >= 1, got {self.closing_kernel_size}")
        if self.batch_size is not None and self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")


@dataclass
class LogConfig:
    level: int = logging.INFO
    format: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    datefmt: str = "%Y-%m-%d %H:%M:%S"


#: ImageNet RGB mean — applied during preprocessing.
IMAGE_NORM_MEAN: List[float] = [0.485, 0.456, 0.406]

#: ImageNet RGB std — applied during preprocessing.
IMAGE_NORM_STD: List[float] = [0.229, 0.224, 0.225]

#: Column order for the QC CSV emitted at the end of a batch run.
QC_FIELDNAMES: List[str] = [
    "slide",
    "roi_index",
    "roi_id",
    "requested_x",
    "requested_y",
    "requested_w",
    "requested_h",
    "clipped_x",
    "clipped_y",
    "clipped_w",
    "clipped_h",
    "overlap_area",
    "requested_area",
    "overlap_ratio",
    "level",
    "read_w",
    "read_h",
    "decision",
    "reason",
]
