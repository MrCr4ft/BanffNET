"""Whole-slide region reading, ROI extraction, and quality-control records."""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple, Union

import numpy as np
import openslide
from openslide import OpenSlideError
from PIL import Image

log = logging.getLogger(__name__)


class SlideProcessorError(Exception):
    """Raised when a slide cannot be opened or read."""


@dataclass(frozen=True)
class BoundingBox:
    """An axis-aligned rectangle in level-0 pixels."""

    x: int
    y: int
    width: int
    height: int

    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass(frozen=True)
class ClippingResult:
    """Outcome of clipping a :class:`BoundingBox` against slide bounds."""

    clipped_x: int
    clipped_y: int
    clipped_w: int
    clipped_h: int
    overlap_area: int
    requested_area: int
    overlap_ratio: float
    off_slide: bool

    @classmethod
    def compute(cls, bbox: BoundingBox, slide_w: int, slide_h: int) -> "ClippingResult":
        """Clip ``bbox`` to the rectangle ``[0, slide_w) × [0, slide_h)``."""
        requested_area = bbox.area
        cx = max(0, bbox.x)
        cy = max(0, bbox.y)
        x_end = min(slide_w, bbox.x + bbox.width)
        y_end = min(slide_h, bbox.y + bbox.height)
        if x_end <= cx or y_end <= cy:
            return cls(0, 0, 0, 0, 0, requested_area, 0.0, off_slide=True)
        cw = x_end - cx
        ch = y_end - cy
        area = cw * ch
        ratio = area / requested_area if requested_area > 0 else 0.0
        return cls(cx, cy, cw, ch, area, requested_area, ratio, off_slide=False)


@dataclass
class QCRecord:
    """Per-ROI QC record. ``slide`` and ``roi_index`` are always set;
    everything else is ``None`` until populated by the extractor.

    The field order matches :data:`binarization.config.QC_FIELDNAMES`, so
    ``QC_FIELDNAMES`` and :meth:`to_csv_row` stay in sync.
    """

    slide: str
    roi_index: int
    #: Human-readable ROI identifier used for the ROI/mask filenames —
    #: the annotation's Name if present, else str(roi_index).
    roi_id: str = ""
    requested_x: Optional[int] = None
    requested_y: Optional[int] = None
    requested_w: Optional[int] = None
    requested_h: Optional[int] = None
    clipped_x: Optional[int] = None
    clipped_y: Optional[int] = None
    clipped_w: Optional[int] = None
    clipped_h: Optional[int] = None
    overlap_area: Optional[int] = None
    requested_area: Optional[int] = None
    overlap_ratio: Optional[float] = None
    level: Optional[int] = None
    read_w: Optional[int] = None
    read_h: Optional[int] = None
    decision: str = ""
    reason: str = ""

    def to_csv_row(self) -> dict:
        """Convert to a CSV-friendly dict. ``None`` becomes ``""``,
        ``overlap_ratio`` is rounded to 6 decimals.
        """
        out: dict = {}
        for k, v in asdict(self).items():
            if v is None:
                out[k] = ""
            elif k == "overlap_ratio":
                out[k] = round(v, 6)
            else:
                out[k] = v
        return out


PathLike = Union[str, Path]


class SlideProcessor:
    """Context-managed openslide wrapper with pyramid-aware reading.

    Example::

        with SlideProcessor("slide.tiff") as sp:
            lvl = sp.compute_level_for_roi(2000, 1500, target_shape=(1024, 1024))
            arr = sp.read_roi_region(x=100, y=200, width=1024, height=1024, level=lvl)
    """

    def __init__(self, slide_path: PathLike):
        self.slide_path = str(slide_path)
        try:
            self.slide = openslide.OpenSlide(self.slide_path)
        except (OpenSlideError, OSError, RuntimeError) as e:
            raise SlideProcessorError(f"Failed to open slide {slide_path}: {e}") from e

    def __enter__(self) -> "SlideProcessor":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if getattr(self, "slide", None) is not None:
            self.slide.close()
            self.slide = None

    @property
    def level_count(self) -> int:
        return self.slide.level_count

    def get_level_dimensions(self, level: int) -> Tuple[int, int]:
        return self.slide.level_dimensions[level]

    def compute_level_for_roi(
        self,
        roi_width: int,
        roi_height: int,
        target_shape: Tuple[int, int],
    ) -> int:
        """Pick the coarsest level where the ROI still meets ``target_shape``.

        Iterates from coarsest level to finest; the first level whose
        downsampled ROI dimensions are at least ``target_shape`` wins.
        Falls back to level 0.
        """
        tgt_w, tgt_h = target_shape
        for level in reversed(range(self.slide.level_count)):
            ds = self.slide.level_downsamples[level]
            if roi_width / ds >= tgt_w and roi_height / ds >= tgt_h:
                log.debug(
                    "Selected level %d (ds=%.1f) for ROI %dx%d -> %.0fx%.0f",
                    level,
                    ds,
                    roi_width,
                    roi_height,
                    roi_width / ds,
                    roi_height / ds,
                )
                return level
        log.warning(
            "ROI too small (%dx%d) for target %s; falling back to level 0",
            roi_width,
            roi_height,
            target_shape,
        )
        return 0

    def read_roi_region(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        level: int,
    ) -> np.ndarray:
        """Read a region at level-0 coords ``(x, y)`` with size at ``level``.

        Returns ``(H, W, 3)`` ``uint8`` RGB.
        """
        if not 0 <= level < self.slide.level_count:
            raise ValueError(f"Invalid level {level}; slide has {self.slide.level_count} levels")
        region = self.slide.read_region((x, y), level, (width, height))
        return np.array(region)[:, :, :3]


class ROIExtractor:
    """Turn annotations into ROI thumbnails on a given slide.

    Stateless aside from ``epsilon_overlap``. One instance per pipeline.
    """

    def __init__(self, epsilon_overlap: float = 0.02):
        self.epsilon_overlap = epsilon_overlap

    def extract_rois_from_slide(
        self,
        slide_processor: SlideProcessor,
        annotations: Iterable,
        target_shape: Tuple[int, int],
        slide_basename: str,
        indices_to_process: Optional[Set[int]] = None,
        roi_ids: Optional[List[str]] = None,
    ) -> Tuple[List[Tuple[int, Image.Image]], List[QCRecord]]:
        """Run extraction for every annotation (or a subset by index).

        Parameters
        ----------
        roi_ids
            Human-readable identifier per annotation, aligned by
            position (see :func:`resolve_roi_id`). ``None`` falls back
            to the positional index for every annotation.

        Returns
        -------
        indexed_images : list of (annotation_index, PIL.Image)
            Each successfully extracted ROI with its index in
            ``annotations``.
        qc_records : list of QCRecord
            One per *processed* annotation (skipped by ``indices_to_process``
            -> no record emitted here).
        """
        rois: List[Tuple[int, Image.Image]] = []
        qc_records: List[QCRecord] = []
        slide_w, slide_h = slide_processor.get_level_dimensions(0)

        for idx, ann in enumerate(annotations):
            if indices_to_process is not None and idx not in indices_to_process:
                continue
            roi_id = roi_ids[idx] if roi_ids is not None else str(idx)
            roi, qc = self._process_single_roi(
                slide_processor,
                ann,
                idx,
                target_shape,
                slide_basename,
                slide_w,
                slide_h,
                roi_id,
            )
            qc_records.append(qc)
            if roi is not None:
                rois.append((idx, roi))
        return rois, qc_records

    def _process_single_roi(
        self,
        slide_processor: SlideProcessor,
        annotation,
        idx: int,
        target_shape: Tuple[int, int],
        slide_basename: str,
        slide_w: int,
        slide_h: int,
        roi_id: str,
    ) -> Tuple[Optional[Image.Image], QCRecord]:
        # 1. Annotation → level-0 bbox.
        x_min, x_max, y_min, y_max = annotation.get_enclosing_rectangle(1.0)
        bbox = BoundingBox(x_min, y_min, x_max - x_min, y_max - y_min)

        # 2. Clip to slide bounds.
        clipping = ClippingResult.compute(bbox, slide_w, slide_h)

        if clipping.off_slide or clipping.overlap_area == 0:
            log.warning(
                "[%s | roi %d (%s)] no overlap with slide bounds", slide_basename, idx, roi_id
            )
            return None, self._qc(
                slide_basename,
                idx,
                roi_id,
                bbox,
                clipping,
                None,
                None,
                None,
                decision="skip",
                reason="no_overlap_off_slide",
            )

        if clipping.overlap_ratio < self.epsilon_overlap:
            log.warning(
                "[%s | roi %d (%s)] tiny overlap ratio = %.5f",
                slide_basename,
                idx,
                roi_id,
                clipping.overlap_ratio,
            )
            return None, self._qc(
                slide_basename,
                idx,
                roi_id,
                bbox,
                clipping,
                None,
                None,
                None,
                decision="skip",
                reason=f"tiny_overlap(<{self.epsilon_overlap:.3f})",
            )

        # 3. Pick level + size at that level.
        level = slide_processor.compute_level_for_roi(
            clipping.clipped_w,
            clipping.clipped_h,
            target_shape,
        )
        ds = slide_processor.slide.level_downsamples[level]
        read_w = int(np.ceil(clipping.clipped_w / ds))
        read_h = int(np.ceil(clipping.clipped_h / ds))

        if read_w < target_shape[0] * 0.5 or read_h < target_shape[1] * 0.5:
            log.warning(
                "[%s | roi %d (%s)] too small (%dx%d at level %d)",
                slide_basename,
                idx,
                roi_id,
                read_w,
                read_h,
                level,
            )
            return None, self._qc(
                slide_basename,
                idx,
                roi_id,
                bbox,
                clipping,
                level,
                read_w,
                read_h,
                decision="skip",
                reason=f"roi_too_small_{read_w}x{read_h}",
            )

        # 4. Read pixels.
        try:
            arr = slide_processor.read_roi_region(
                clipping.clipped_x,
                clipping.clipped_y,
                read_w,
                read_h,
                level,
            )
            # Replace pure-black border with white (a common openslide quirk).
            black = np.all(arr == [0, 0, 0], axis=-1)
            if black.any():
                arr = arr.copy()
                arr[black] = [255, 255, 255]
            img = Image.fromarray(arr)
            return img, self._qc(
                slide_basename,
                idx,
                roi_id,
                bbox,
                clipping,
                level,
                read_w,
                read_h,
                decision="keep",
            )
        except Exception as e:
            log.error("[%s | roi %d (%s)] read failed: %s", slide_basename, idx, roi_id, e)
            return None, self._qc(
                slide_basename,
                idx,
                roi_id,
                bbox,
                clipping,
                level,
                read_w,
                read_h,
                decision="skip",
                reason=f"read_error:{str(e)[:50]}",
            )

    @staticmethod
    def _qc(
        slide: str,
        roi_index: int,
        roi_id: str,
        bbox: BoundingBox,
        clipping: ClippingResult,
        level: Optional[int],
        read_w: Optional[int],
        read_h: Optional[int],
        *,
        decision: str,
        reason: str = "",
    ) -> QCRecord:
        return QCRecord(
            slide=slide,
            roi_index=roi_index,
            roi_id=roi_id,
            requested_x=bbox.x,
            requested_y=bbox.y,
            requested_w=bbox.width,
            requested_h=bbox.height,
            clipped_x=clipping.clipped_x,
            clipped_y=clipping.clipped_y,
            clipped_w=clipping.clipped_w,
            clipped_h=clipping.clipped_h,
            overlap_area=clipping.overlap_area,
            requested_area=clipping.requested_area,
            overlap_ratio=clipping.overlap_ratio,
            level=level,
            read_w=read_w,
            read_h=read_h,
            decision=decision,
            reason=reason,
        )

    @staticmethod
    def get_expected_roi_paths(base_name: str, roi_dir: str, roi_ids: List[str]) -> List[str]:
        """Where each ROI would be saved by :meth:`FileManager.save_roi`.

        ``roi_ids`` is one identifier per annotation (see
        :func:`wsipatch.resolve_roi_id`), positionally aligned with the
        annotations list — not a count, since ROI identity may now be a
        name rather than a plain index.
        """
        return [os.path.join(roi_dir, f"{base_name}_roi_{rid}.png") for rid in roi_ids]

    @staticmethod
    def check_existing_rois(expected_paths: List[str]) -> Tuple[List[bool], Set[int]]:
        """Return ``(exists_flags, indices_still_to_process)``."""
        flags = [os.path.exists(p) for p in expected_paths]
        return flags, {i for i, e in enumerate(flags) if not e}
