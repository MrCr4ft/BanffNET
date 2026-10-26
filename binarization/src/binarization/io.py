"""ROI and mask file management, QC CSV output, slide discovery, and logging."""

from __future__ import annotations

import csv
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

PathLike = Union[str, Path]


class FileManager:
    """Filesystem layout for ROI and mask outputs.

    Conventions::

        {roi_dir}/{slide}_roi_{roi_id}.png
        {mask_dir}/{slide}_roi_{roi_id}_mask.png

    ``roi_id`` is the ROI's annotation Name when it has one (see
    :func:`wsipatch.resolve_roi_id`), otherwise its positional index —
    either an ``int`` or a ``str`` is accepted everywhere below and
    simply interpolated into the filename.

    Both output directories are created on construction.
    """

    def __init__(self, roi_dir: PathLike, mask_dir: PathLike):
        self.roi_dir = Path(roi_dir)
        self.mask_dir = Path(mask_dir)
        self.roi_dir.mkdir(parents=True, exist_ok=True)
        self.mask_dir.mkdir(parents=True, exist_ok=True)

    def roi_path(self, slide_name: str, roi_id: Union[int, str]) -> Path:
        return self.roi_dir / f"{slide_name}_roi_{roi_id}.png"

    def mask_path(self, slide_name: str, roi_id: Union[int, str]) -> Path:
        return self.mask_dir / f"{slide_name}_roi_{roi_id}_mask.png"

    def roi_exists(self, slide_name: str, roi_id: Union[int, str]) -> bool:
        return self.roi_path(slide_name, roi_id).exists()

    def mask_exists(self, slide_name: str, roi_id: Union[int, str]) -> bool:
        return self.mask_path(slide_name, roi_id).exists()

    def save_roi(self, image: Image.Image, slide_name: str, roi_id: Union[int, str]) -> str:
        """Save a PIL image as the ROI thumbnail. Skips writes if the file exists."""
        p = self.roi_path(slide_name, roi_id)
        if not p.exists():
            image.save(p)
            log.debug("Saved ROI: %s", p)
        return str(p)

    def save_mask(self, mask: np.ndarray, slide_name: str, roi_id: Union[int, str]) -> str:
        """Save a binary mask via ``cv2.imwrite``."""
        p = self.mask_path(slide_name, roi_id)
        cv2.imwrite(str(p), mask)
        log.debug("Saved mask: %s", p)
        return str(p)

    def load_roi(self, slide_name: str, roi_id: Union[int, str]) -> Optional[Image.Image]:
        p = self.roi_path(slide_name, roi_id)
        return Image.open(p) if p.exists() else None

    def list_roi_files(self) -> List[Path]:
        return sorted(self.roi_dir.glob("*_roi_*.png"))

    def list_mask_files(self) -> List[Path]:
        return sorted(self.mask_dir.glob("*_mask.png"))


class CSVWriter:
    """Minimal dict-row CSV writer."""

    def __init__(self, filepath: PathLike, fieldnames: List[str]):
        self.filepath = str(filepath)
        self.fieldnames = fieldnames

    def write_records(self, records: List[Dict]) -> None:
        """Overwrite-and-write."""
        with open(self.filepath, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.fieldnames)
            w.writeheader()
            w.writerows(records)
        log.info("Wrote %d records to %s", len(records), self.filepath)

    def append_records(self, records: List[Dict]) -> None:
        """Append; writes a header if the file did not exist."""
        existed = os.path.exists(self.filepath)
        with open(self.filepath, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.fieldnames)
            if not existed:
                w.writeheader()
            w.writerows(records)
        log.info("Appended %d records to %s", len(records), self.filepath)


def get_biopsy_list(
    biopsy_dir: PathLike,
    annotations_dir: PathLike,
    biopsy_ext: str = "tiff",
    use_entire_wsi_if_no_annotation: bool = False,
) -> Dict[str, List[str]]:
    """Pair WSIs under ``biopsy_dir`` with XML files under ``annotations_dir``.

    Returns
    -------
    dict
        ``{"biopsy": [paths], "annotation": [paths]}`` — parallel lists.
        When ``use_entire_wsi_if_no_annotation`` is ``True``, slides with
        no matching XML are still included with an empty-string sentinel
        in the ``annotation`` list (the pipeline reads that as
        "fall back to full-slide ROI").
    """
    bp = Path(biopsy_dir)
    ap = Path(annotations_dir)
    out: Dict[str, List[str]] = {"biopsy": [], "annotation": []}

    for slide in sorted(bp.glob(f"*.{biopsy_ext}")):
        ann = ap / f"{slide.stem}.xml"
        if ann.exists():
            out["biopsy"].append(str(slide))
            out["annotation"].append(str(ann))
        elif use_entire_wsi_if_no_annotation:
            log.warning(
                "No annotation for %s — adding without annotation (full WSI mode)",
                slide.name,
            )
            out["biopsy"].append(str(slide))
            out["annotation"].append("")
        else:
            log.warning("No annotation for %s — skipping", slide.name)

    log.info("Found %d biopsies to process", len(out["biopsy"]))
    return out


def ensure_directories(*dirs: PathLike) -> None:
    """``mkdir -p`` over a sequence of paths."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def setup_logging(
    level: int = logging.INFO,
    format_string: Optional[str] = None,
    log_file: Optional[str] = None,
) -> None:
    """Configure the root logger; quiets openslide/PIL chatter."""
    if format_string is None:
        format_string = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=level,
        format=format_string,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )
    logging.getLogger("openslide").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
