"""Coordinate ROI extraction, tissue segmentation, and QC reporting.

Use ``process_slide`` for one slide, ``extract_slide`` and ``segment_paths``
for separate stages, or ``run`` for a batch. Models load lazily; the context
manager releases GPU resources on exit. ASAP annotations are parsed by wsipatch.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import tqdm
from PIL import Image
from wsipatch.annotations import ASAPAnnotations
from wsipatch.types import resolve_roi_id

from .config import QC_FIELDNAMES, BinarizationConfig
from .extraction import (
    QCRecord,
    ROIExtractor,
    SlideProcessor,
    SlideProcessorError,
)
from .io import CSVWriter, FileManager
from .segmentation import SegmentationPipeline, load_segmentation_model

log = logging.getLogger(__name__)

PathLike = Union[str, Path]


@dataclass
class ProcessingStats:
    """Running counters for a batch run."""

    total_slides: int = 0
    processed_slides: int = 0
    failed_slides: int = 0
    total_rois: int = 0
    processed_rois: int = 0
    skipped_rois: int = 0
    failed_rois: int = 0
    processing_time: float = 0.0

    def add_slide(self, success: bool = True) -> None:
        self.total_slides += 1
        if success:
            self.processed_slides += 1
        else:
            self.failed_slides += 1

    def add_roi(self, status: str = "processed") -> None:
        self.total_rois += 1
        if status == "processed":
            self.processed_rois += 1
        elif status == "skipped":
            self.skipped_rois += 1
        elif status == "failed":
            self.failed_rois += 1

    def to_dict(self) -> Dict:
        return asdict(self)

    def print_summary(self) -> None:
        print("\n" + "=" * 50)
        print("PROCESSING SUMMARY")
        print("=" * 50)
        print(f"Slides: {self.processed_slides}/{self.total_slides} processed")
        if self.failed_slides:
            print(f"        {self.failed_slides} failed")
        print(f"ROIs:   {self.processed_rois}/{self.total_rois} processed")
        if self.skipped_rois:
            print(f"        {self.skipped_rois} skipped")
        if self.failed_rois:
            print(f"        {self.failed_rois} failed")
        if self.processing_time > 0:
            print(f"Time:   {self.processing_time:.2f} seconds")
        print("=" * 50)


class BinarizationPipeline:
    """ROI extraction + tissue segmentation for one or many slides.

    Lifecycle
    ---------
    The segmentation model is lazily loaded on the first call to
    :meth:`segment_paths` (directly, or via
    :meth:`process_slide`/:meth:`run`). Use the pipeline as a context
    manager so :meth:`close` releases the model and frees GPU memory
    deterministically.

    Persistent state on the instance
    --------------------------------
    * :attr:`stats` — running :class:`ProcessingStats` across all calls.
    * :attr:`qc_records` — list of :class:`QCRecord` accumulated by
      :meth:`extract_slide`. Use :meth:`write_qc_csv` to flush.

    Call :meth:`reset` to zero both back out without rebuilding the
    pipeline.
    """

    def __init__(self, config: BinarizationConfig):
        self.config = config
        self.stats = ProcessingStats()
        self.qc_records: List[QCRecord] = []
        self.file_manager = FileManager(config.roi_dir, config.mask_dir)
        self.roi_extractor = ROIExtractor(epsilon_overlap=config.epsilon_overlap)
        self._segmenter: Optional[SegmentationPipeline] = None
        self._batch_size: Optional[int] = None

    @classmethod
    def from_paths(
        cls,
        *,
        roi_dir: PathLike,
        mask_dir: PathLike,
        model_path: PathLike,
        **kwargs,
    ) -> "BinarizationPipeline":
        """Construct from the three required paths plus any
        :class:`BinarizationConfig` keyword arguments.
        """
        return cls(
            BinarizationConfig(
                roi_dir=str(roi_dir),
                mask_dir=str(mask_dir),
                model_path=str(model_path),
                **kwargs,
            )
        )

    def __enter__(self) -> "BinarizationPipeline":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self) -> None:
        """Drop the segmentation model and free GPU memory if present."""
        if self._segmenter is not None:
            self._segmenter = None
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    def reset(self) -> None:
        """Zero out stats and QC records (does not unload the model)."""
        self.stats = ProcessingStats()
        self.qc_records = []

    def extract_slide(
        self,
        slide_path: PathLike,
        annotation_path: Optional[PathLike] = None,
    ) -> List[str]:
        """Extract ROI thumbnails from one slide. Returns paths of newly
        written ROIs (existing ones are skipped silently; the
        corresponding QC records still get appended).
        """
        slide_path = str(slide_path)
        base_name = Path(slide_path).stem
        log.info("Extracting ROIs from %s", base_name)

        annotations = self._load_annotations(slide_path, annotation_path, base_name)
        if annotations is None:
            self.stats.add_slide(success=False)
            return []

        num_rois = len(annotations)
        # Match ROI identifiers used by downstream mask and feature lookups.
        roi_ids = [
            resolve_roi_id(getattr(ann, "name", None), i) for i, ann in enumerate(annotations)
        ]
        expected = ROIExtractor.get_expected_roi_paths(base_name, self.config.roi_dir, roi_ids)
        exists_flags, indices_to_process = ROIExtractor.check_existing_rois(expected)

        # QC for already-existing ROIs.
        for idx, e in enumerate(exists_flags):
            if e:
                self.qc_records.append(
                    QCRecord(
                        slide=base_name,
                        roi_index=idx,
                        roi_id=roi_ids[idx],
                        decision="skip",
                        reason="already_exists",
                    )
                )
                self.stats.add_roi("skipped")

        if not indices_to_process:
            log.info("All %d ROIs already exist for %s", num_rois, base_name)
            self.stats.add_slide(success=True)
            return []

        roi_paths: List[str] = []
        try:
            with SlideProcessor(slide_path) as sp:
                indexed, qc_records = self.roi_extractor.extract_rois_from_slide(
                    slide_processor=sp,
                    annotations=annotations,
                    target_shape=self.config.fixed_shape,
                    slide_basename=base_name,
                    indices_to_process=indices_to_process,
                    roi_ids=roi_ids,
                )
                for idx, img in indexed:
                    roi_paths.append(self.file_manager.save_roi(img, base_name, roi_ids[idx]))
                    self.stats.add_roi("processed")
                for qc in qc_records:
                    self.qc_records.append(qc)
                    if qc.decision == "skip":
                        self.stats.add_roi("skipped")
                self.stats.add_slide(success=True)
        except SlideProcessorError as e:
            log.error("Failed to process %s: %s", base_name, e)
            for idx in indices_to_process:
                self.qc_records.append(
                    QCRecord(
                        slide=base_name,
                        roi_index=idx,
                        roi_id=roi_ids[idx],
                        decision="skip",
                        reason=f"slide_error:{str(e)[:50]}",
                    )
                )
                self.stats.add_roi("failed")
            self.stats.add_slide(success=False)

        return roi_paths

    def _load_annotations(
        self,
        slide_path: str,
        annotation_path: Optional[PathLike],
        base_name: str,
    ) -> Optional[ASAPAnnotations]:
        """Load annotations, optionally falling back to the full slide."""
        anns = ASAPAnnotations.from_xml_file(annotation_path) if annotation_path else None
        if anns is None or len(anns) == 0:
            if self.config.use_entire_wsi_if_no_annotation:
                log.warning(
                    "No annotations found for %s — using entire WSI as single ROI",
                    base_name,
                )
                anns = ASAPAnnotations.from_slide(slide_path)
                if anns is None:
                    log.error("Failed to build full-slide annotation for %s", base_name)
                    return None
            else:
                log.warning("No valid annotations for %s", base_name)
                return None
        return anns

    def segment_paths(self, roi_paths: List[PathLike]) -> List[str]:
        """Segment a list of saved ROI thumbnails into binary masks.

        Returns the list of newly written mask paths. Existing masks for
        the same slide/roi are overwritten (use
        :meth:`get_pending_segmentations` to filter beforehand if you
        want resumability).
        """
        if not roi_paths:
            return []
        self._ensure_model_loaded()
        bs = self._batch_size
        log.info("Segmenting %d ROIs (batch_size=%d)", len(roi_paths), bs)

        mask_paths: List[str] = []
        for i in tqdm.trange(0, len(roi_paths), bs, desc="Segmenting ROIs"):
            batch_paths = list(roi_paths[i : i + bs])
            images = [Image.open(p).copy() for p in batch_paths]
            masks = self._segmenter.segment_batch(
                images,
                apply_contrast_enhancement=self.config.apply_contrast_enhancement,
                clahe_clip_limit=self.config.clahe_clip_limit,
                clahe_tile_size=self.config.clahe_tile_size,
            )
            for p, m in zip(batch_paths, masks):
                slide_name, roi_id = self._parse_roi_path(p)
                if slide_name is None:
                    continue
                mask_paths.append(self.file_manager.save_mask(m, slide_name, roi_id))
        return mask_paths

    def get_pending_segmentations(self) -> List[str]:
        """List ROI files on disk that don't yet have a corresponding mask."""
        pending: List[str] = []
        for roi_path in self.file_manager.list_roi_files():
            slide_name, roi_id = self._parse_roi_path(roi_path)
            if slide_name is None:
                continue
            if not self.file_manager.mask_exists(slide_name, roi_id):
                pending.append(str(roi_path))
        return pending

    def _ensure_model_loaded(self) -> None:
        if self._segmenter is not None:
            return
        log.info("Loading segmentation model from %s", self.config.model_path)
        model, device = load_segmentation_model(
            self.config.model_path,
            architecture=self.config.architecture,
            encoder_name=self.config.encoder_name,
        )
        bs = self.config.batch_size
        if bs is None:
            bs = SegmentationPipeline.detect_optimal_batch_size(
                model,
                device,
                self.config.fixed_shape,
            )
        self._batch_size = bs
        self._segmenter = SegmentationPipeline(
            model=model,
            device=device,
            fixed_shape=self.config.fixed_shape,
            probability_threshold=self.config.probability_threshold,
            apply_morphology_postprocessing=self.config.apply_morphology_postprocessing,
            closing_kernel_size=self.config.closing_kernel_size,
        )
        log.info("Segmentation model ready (batch_size=%d)", bs)

    @staticmethod
    def _parse_roi_path(path: PathLike) -> Tuple[Optional[str], Optional[str]]:
        """Split ``{slide}_roi_{roi_id}`` back into ``(slide_name, roi_id)``.

        ``roi_id`` may be a name (e.g. ``"core_01"``) or a plain index
        string — it's returned as-is, unparsed, since the filename
        already carries whatever :func:`wsipatch.resolve_roi_id` chose.
        """
        stem = Path(path).stem
        parts = stem.rsplit("_roi_", 1)
        if len(parts) != 2:
            log.warning("Cannot parse ROI id from %s", stem)
            return None, None
        return parts[0], parts[1]

    def process_slide(
        self,
        slide_path: PathLike,
        annotation_path: Optional[PathLike] = None,
        *,
        segment: bool = True,
    ) -> Tuple[List[str], List[str]]:
        """Run both phases for one slide.

        Returns ``(roi_paths, mask_paths)``. ``mask_paths`` is empty when
        ``segment=False`` or when extraction yielded no new ROIs.

        This is the canonical entry point for downstream orchestrators
        that integrate binarization with other steps (e.g. feeding the
        masks into ``wsipatch.plan_slide(...)``).
        """
        roi_paths = self.extract_slide(slide_path, annotation_path)
        mask_paths: List[str] = []
        if segment and roi_paths:
            mask_paths = self.segment_paths(roi_paths)
        return roi_paths, mask_paths

    def run(
        self,
        biopsy_list: Dict[str, List[str]],
        *,
        extract_only: bool = False,
    ) -> None:
        """Batch mode: process every slide, write QC CSV, print summary.

        Extract all slides, write QC, then segment ROIs with missing masks.
        Existing outputs are skipped so interrupted batches can resume.
        """
        t0 = time.time()
        if not biopsy_list["biopsy"]:
            log.warning("No biopsies to process")
            return

        log.info("=" * 50)
        log.info("PHASE 1: Extracting ROIs")
        log.info("=" * 50)
        for slide_path, ann in tqdm.tqdm(
            zip(biopsy_list["biopsy"], biopsy_list["annotation"]),
            total=len(biopsy_list["biopsy"]),
            desc="Processing slides",
        ):
            self.extract_slide(slide_path, ann if ann else None)

        qc_path = Path(self.config.roi_dir) / "roi_qc.csv"
        self.write_qc_csv(qc_path)

        if not extract_only:
            log.info("=" * 50)
            log.info("PHASE 2: Segmenting ROIs (incremental)")
            log.info("=" * 50)
            pending = self.get_pending_segmentations()
            total = len(self.file_manager.list_roi_files())
            log.info("Total ROIs:                 %d", total)
            log.info("Existing masks:             %d", total - len(pending))
            log.info("ROIs needing binarization:  %d", len(pending))
            if pending:
                self.segment_paths(pending)
            else:
                log.info("All ROIs already have masks — nothing to segment")

        self.stats.processing_time = time.time() - t0
        self.stats.print_summary()
        self._log_qc_summary()

    def write_qc_csv(self, path: PathLike) -> str:
        """Write accumulated QC records to a CSV. Returns the path written."""
        rows = [r.to_csv_row() for r in self.qc_records]
        CSVWriter(path, QC_FIELDNAMES).write_records(rows)
        return str(path)

    def get_qc_summary(self) -> Dict:
        """Aggregate counts of kept/skipped ROIs, broken down by skip reason."""
        kept = sum(1 for r in self.qc_records if r.decision == "keep")
        skipped = sum(1 for r in self.qc_records if r.decision == "skip")
        reasons: Dict[str, int] = {}
        for r in self.qc_records:
            if r.decision == "skip":
                reasons[r.reason] = reasons.get(r.reason, 0) + 1
        return {
            "total_records": len(self.qc_records),
            "kept": kept,
            "skipped": skipped,
            "skip_reasons": reasons,
        }

    def _log_qc_summary(self) -> None:
        summary = self.get_qc_summary()
        if summary["skipped"]:
            log.info("QC Summary — Skip reasons:")
            for reason, count in summary["skip_reasons"].items():
                log.info("  %s: %d", reason, count)
