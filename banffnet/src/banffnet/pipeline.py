"""Compose tissue segmentation, patch planning, UNI features, and BANFF inference.

Each pipeline uses one staining configuration. ``staining_cache`` switches
between stainings while keeping at most one pipeline loaded.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch

from binarization import BinarizationPipeline
from features_extraction import FeatureExtractor, FeatureH5Writer
from wsipatch import (
    ASAPAnnotations,
    PatchReader,
    TargetSpec,
    asap_to_rois,
    plan_slide_multi,
    resolve_roi_id,
    save_index,
)

from .defaults import DEFAULT_MPP, DEFAULT_PATCH_SIZE, staining_paths
from .predictor import BanffPredictor

log = logging.getLogger(__name__)

PathLike = Union[str, Path]


@dataclass
class SlideResult:
    """Per-slide pipeline output.

    Slide-level :attr:`slide_scores` is populated for ``level='slide'``;
    patch-level :attr:`patch_scores` for ``level='patch'``; per-ROI
    :attr:`roi_scores` for ``level='roi'`` — exactly one of the three is
    set. :attr:`features_path` is set when features were saved to disk.
    """

    slide_id: str
    slide_scores: Optional[Dict[str, float]] = None
    patch_scores: Optional[np.ndarray] = None
    roi_scores: Optional[Dict[str, Dict[str, float]]] = None
    features_path: Optional[Path] = None
    num_patches: int = 0


class BanffNETPipeline:
    """End-to-end orchestrator for one staining.

    Use :meth:`from_staining` for configured asset paths, :meth:`from_paths` for full control.

    Use as a context manager so the binarization and feature-extraction
    GPU state are released deterministically::

        with BanffNETPipeline.from_staining("HE", workspace_dir="./workspace") as p:
            scores = p.predict_slide("slide.tiff", "ann.xml")
    """

    def __init__(
        self,
        *,
        binarization_pipeline: BinarizationPipeline,
        feature_extractor: FeatureExtractor,
        banff_predictor: BanffPredictor,
        features_dir: PathLike,
        plans_dir: PathLike,
        target_spec: Optional[TargetSpec] = None,
        low_memory: bool = False,
        batch_size: int = 64,
        save_rois: bool = True,
        save_masks: bool = True,
        save_features: bool = True,
        save_plans: bool = True,
        tissue_threshold: float = 0.5,
    ):
        self.binarization = binarization_pipeline
        self.feature_extractor = feature_extractor
        self.banff_predictor = banff_predictor
        self.features_dir = Path(features_dir)
        self.plans_dir = Path(plans_dir)
        self.target_spec = target_spec or TargetSpec(
            name="uni",
            patch_size=DEFAULT_PATCH_SIZE,
            mpp=DEFAULT_MPP,
        )
        self.low_memory = low_memory
        self.batch_size = batch_size
        self.save_rois = save_rois
        self.save_masks = save_masks
        self.save_features = save_features
        self.save_plans = save_plans
        self.tissue_threshold = tissue_threshold
        if self.save_features:
            self.features_dir.mkdir(parents=True, exist_ok=True)
        if self.save_plans:
            self.plans_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_staining(
        cls,
        staining: str,
        *,
        workspace_dir: PathLike,
        device: Union[str, torch.device] = "cuda",
        **overrides,
    ) -> "BanffNETPipeline":
        """Build using the configured asset paths for one staining.

        ``overrides`` accepts any keyword accepted by :meth:`from_paths`
        — useful for swapping a single path (e.g. a custom Banff config)
        while keeping the rest of the staining's defaults.
        """
        d = staining_paths(staining)
        kwargs = dict(
            binarization_model=d.binarization_model,
            uni_checkpoint=d.uni_checkpoint,
            reinhard_config=d.reinhard_config,
            banff_config=d.banff_config,
            workspace_dir=workspace_dir,
            device=device,
        )
        kwargs.update(overrides)
        return cls.from_paths(**kwargs)

    @classmethod
    def from_paths(
        cls,
        *,
        binarization_model: PathLike,
        uni_checkpoint: PathLike,
        reinhard_config: PathLike,
        banff_config: PathLike,
        workspace_dir: PathLike,
        device: Union[str, torch.device] = "cuda",
        target_spec: Optional[TargetSpec] = None,
        low_memory: bool = False,
        batch_size: int = 64,
        save_rois: bool = True,
        save_masks: bool = True,
        save_features: bool = True,
        save_plans: bool = True,
        feature_extractor_kwargs: Optional[Dict[str, Any]] = None,
        binarization_kwargs: Optional[Dict[str, Any]] = None,
        tissue_threshold: float = 0.5,
    ) -> "BanffNETPipeline":
        """Construct from individually-specified paths.

        Parameters
        ----------
        binarization_model
            Path to the MobileNetV2 tissue-segmentation checkpoint.
        uni_checkpoint
            Path to the UNI ViT-L/16 checkpoint.
        reinhard_config
            Path to the staining's Reinhard normalisation YAML.
        banff_config
            Path to the staining's BANFF prediction YAML.
        workspace_dir
            Where intermediate artifacts are written, one subfolder each:
            ``rois/``, ``masks/``, ``features/`` and ``wsipatch_plans/``
            (the ``.wsix`` patch plans).
        device
            Torch device for UNI and the predictor.
        save_rois, save_masks, save_features, save_plans
            Per-artifact persistence toggles; everything is saved by default.
            ROIs/masks are produced by binarization during the run regardless
            (the planner reads the masks) and are deleted afterwards when the
            corresponding flag is ``False``.
        target_spec
            Patch-extraction target. Defaults to ``256 px @ 0.5 mpp``.
        low_memory
            Forwarded to :class:`wsipatch.PatchReader`; trades RAM for I/O.
        batch_size
            Patches per UNI forward pass.
        save_features
            Persist per-slide features under ``{workspace_dir}/features/``.
        feature_extractor_kwargs, binarization_kwargs
            Escape hatches for tuning the wrapped pipelines.
        """
        workspace = Path(workspace_dir)
        bin_pipeline = BinarizationPipeline.from_paths(
            roi_dir=workspace / "rois",
            mask_dir=workspace / "masks",
            model_path=binarization_model,
            **(binarization_kwargs or {}),
        )
        extractor = FeatureExtractor.from_files(
            checkpoint_path=uni_checkpoint,
            reinhard_config_path=reinhard_config,
            device=device,
            **(feature_extractor_kwargs or {}),
        )
        predictor = BanffPredictor.from_yaml(banff_config, device=device)
        return cls(
            binarization_pipeline=bin_pipeline,
            feature_extractor=extractor,
            banff_predictor=predictor,
            features_dir=workspace / "features",
            plans_dir=workspace / "wsipatch_plans",
            target_spec=target_spec,
            low_memory=low_memory,
            batch_size=batch_size,
            save_rois=save_rois,
            save_masks=save_masks,
            save_features=save_features,
            save_plans=save_plans,
            tissue_threshold=tissue_threshold,
        )

    def __enter__(self) -> "BanffNETPipeline":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self) -> None:
        """Release wrapped pipelines and free GPU memory."""
        self.binarization.close()
        # FeatureExtractor doesn't own a close(); drop the model ref.
        self.feature_extractor.model = None
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 — defensive against CUDA hiccups
            pass

    @property
    def attribute_names(self) -> List[str]:
        """Names of the BANFF attributes this pipeline predicts."""
        return self.banff_predictor.attribute_names

    def predict_slide(
        self,
        slide_path: PathLike,
        annotation_path: Optional[PathLike] = None,
        *,
        level: str = "slide",
    ) -> SlideResult:
        """Run the full pipeline on one slide.

        Parameters
        ----------
        slide_path
            Path to the WSI.
        annotation_path
            Path to an ASAP XML. ``None`` falls back to a single full-slide
            ROI (per :class:`BinarizationPipeline`'s config flag).
        level
            ``'slide'`` for one score per attribute (pooled across every
            ROI), ``'roi'`` for one score set per ROI (each biopsy
            core/fragment scored as its own MIL bag), or ``'patch'`` for
            per-patch probability matrices.
        """
        if level not in ("slide", "patch", "roi"):
            raise ValueError(f"level must be 'slide', 'patch', or 'roi', got {level!r}")

        slide_id = Path(slide_path).stem

        # 1. Binarization → tissue masks (saved to disk by BinarizationPipeline).
        log.info("[%s] phase 1/4: binarization", slide_id)
        self.binarization.process_slide(slide_path, annotation_path)

        try:
            # 2. Patchification → SlideIndex with masked tissue grids.
            log.info("[%s] phase 2/4: patchification", slide_id)
            rois, masks = self._collect_rois_and_masks(slide_path, annotation_path)
            if not rois:
                log.warning("[%s] no usable ROIs after binarization — skipping", slide_id)
                return SlideResult(slide_id=slide_id, num_patches=0)
            plan = plan_slide_multi(
                slide_path,
                targets=[self.target_spec],
                rois=rois,
                masks=masks,
                slide_id=slide_id,
                tissue_threshold=self.tissue_threshold,
            )
            if self.save_plans:
                self._save_plan(plan, slide_id)

            # 3. Feature extraction.
            log.info("[%s] phase 3/4: feature extraction", slide_id)
            features, roi_id = self._extract_features(slide_path, plan, slide_id=slide_id)
            n_patches = features.shape[0]
            if n_patches == 0:
                log.warning("[%s] zero patches after planning — skipping", slide_id)
                return SlideResult(slide_id=slide_id, num_patches=0)

            # 4. Prediction.
            log.info(
                "[%s] phase 4/4: prediction (%d patches, level=%s)", slide_id, n_patches, level
            )
            feats_t = torch.from_numpy(features).to(self.banff_predictor.device)
            features_path = self.features_dir / f"{slide_id}.h5" if self.save_features else None
            if level == "slide":
                return SlideResult(
                    slide_id=slide_id,
                    slide_scores=self.banff_predictor.predict_slide(feats_t),
                    features_path=features_path,
                    num_patches=n_patches,
                )
            if level == "roi":
                return SlideResult(
                    slide_id=slide_id,
                    roi_scores=self.banff_predictor.predict_slide_by_roi(feats_t, roi_id),
                    features_path=features_path,
                    num_patches=n_patches,
                )
            return SlideResult(
                slide_id=slide_id,
                patch_scores=self.banff_predictor.predict_patch(feats_t),
                features_path=features_path,
                num_patches=n_patches,
            )
        finally:
            # Drop ROI/mask intermediates the caller asked not to keep. Runs on
            # every exit path (including the early skips above).
            self._cleanup_intermediates(slide_id)

    def predict_features(
        self,
        h5_path: PathLike,
        *,
        level: str = "slide",
    ) -> SlideResult:
        """Run only the prediction stage on a precomputed features H5.

        ``level='roi'`` requires the H5 to carry a ``metadata/roi_id``
        dataset (written automatically by :meth:`predict_slide` when
        ``save_features=True``).
        """
        slide_id = Path(h5_path).stem
        scores = self.banff_predictor.predict_features_file(h5_path, level=level)
        with h5py.File(h5_path, "r") as f:
            n = f["features"].shape[0]
        if level == "slide":
            return SlideResult(slide_id=slide_id, slide_scores=scores, num_patches=n)
        if level == "roi":
            return SlideResult(slide_id=slide_id, roi_scores=scores, num_patches=n)
        return SlideResult(slide_id=slide_id, patch_scores=scores, num_patches=n)

    def _collect_rois_and_masks(
        self,
        slide_path: PathLike,
        annotation_path: Optional[PathLike],
    ) -> Tuple[List, List[str]]:
        """Pair every ROI from the XML with its tissue mask on disk.

        ROIs whose binarization mask is missing (e.g. they were skipped
        as off-slide or with tiny overlap) are dropped.
        """
        if annotation_path:
            anns = ASAPAnnotations.from_xml_file(annotation_path)
        else:
            anns = ASAPAnnotations.from_slide(slide_path)
        all_rois = asap_to_rois(anns, clip_to_origin=True, min_area=1)
        slide_id = Path(slide_path).stem

        rois, masks = [], []
        for i, roi in enumerate(all_rois):
            # Match the ROI identifier used in binarization mask filenames.
            roi_id = resolve_roi_id(roi.name, i)
            mp = self.binarization.file_manager.mask_path(slide_id, roi_id)
            if mp.exists():
                rois.append(roi)
                masks.append(str(mp))
        return rois, masks

    def _save_plan(self, plan, slide_id: str) -> Path:
        """Persist the wsipatch plan as ``{plans_dir}/{slide_id}.wsix``."""
        self.plans_dir.mkdir(parents=True, exist_ok=True)
        out = self.plans_dir / f"{slide_id}.wsix"
        save_index(plan, out)
        return out

    def _cleanup_intermediates(self, slide_id: str) -> None:
        """Delete this slide's ROI/mask files when their save flag is off."""
        if not self.save_rois:
            roi_dir = Path(self.binarization.file_manager.roi_dir)
            for p in roi_dir.glob(f"{slide_id}_roi_*.png"):
                p.unlink(missing_ok=True)
        if not self.save_masks:
            mask_dir = Path(self.binarization.file_manager.mask_dir)
            for p in mask_dir.glob(f"{slide_id}_roi_*_mask.png"):
                p.unlink(missing_ok=True)

    def _extract_features(
        self,
        slide_path: PathLike,
        plan,
        *,
        slide_id: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run :class:`PatchReader` → :class:`FeatureExtractor`, optionally streaming to disk.

        Returns ``(features, roi_id)`` — ``roi_id`` is the per-patch ROI
        identifier (name, or positional index if unnamed) aligned with
        ``features``' first dimension, needed for ``level='roi'``
        prediction.
        """
        with PatchReader(
            plan,
            slide_path=str(slide_path),
            low_memory=self.low_memory,
        ) as reader:
            if self.save_features:
                return self._extract_features_streaming(reader, slide_id=slide_id)
            features, metadata = self.feature_extractor.extract_iterable(
                reader.iter_batches(batch_size=self.batch_size),
                progress=False,
            )
            roi_id = metadata.get("roi_id", np.zeros((features.shape[0],), dtype=np.int64))
            return features, roi_id

    def _extract_features_streaming(
        self, reader, *, slide_id: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Stream features to HDF5; read back after write completes (no RAM accumulation)."""
        out_path = self.features_dir / f"{slide_id}.h5"
        with FeatureH5Writer(out_path) as writer:
            writer.set_attrs(
                model="UNI",
                target_name=self.target_spec.name,
                patch_size=self.target_spec.patch_size,
                mpp=self.target_spec.mpp,
            )
            for batch, meta in reader.iter_batches(batch_size=self.batch_size):
                feats = self.feature_extractor.extract_batch(batch)
                writer.append(feats, meta)
        # HDF5 is fully flushed and closed by FeatureH5Writer.__exit__.
        if writer.n == 0:
            return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.int64)
        with h5py.File(out_path, "r") as f:
            features = f["features"][:]
            # roi_id is stored as a variable-length utf-8 string dataset;
            # .asstr() decodes it to Python str instead of raw bytes.
            roi_id = f["metadata"]["roi_id"].asstr()[:]
        return features, roi_id


@contextmanager
def staining_cache(workspace_dir: PathLike, **pipeline_kwargs):
    """Provide a callable that returns a pipeline for a given staining.

    Holds at most one pipeline at a time — closes the previous when a
    different staining is requested. Used by the ``banffnet list`` CLI
    to process a CSV with mixed stainings without keeping three
    pipelines on the GPU.

    Usage::

        with staining_cache("./workspace", device="cuda") as get:
            for slide, ann, staining in rows:
                pipeline = get(staining)
                pipeline.predict_slide(slide, ann)
    """
    current: Dict[str, BanffNETPipeline] = {}

    def _get(staining: str) -> BanffNETPipeline:
        s = staining.upper()
        if "active" in current and current["active_staining"] != s:
            current["active"].close()
            current.pop("active")
            current.pop("active_staining")
        if "active" not in current:
            current["active"] = BanffNETPipeline.from_staining(
                s,
                workspace_dir=workspace_dir,
                **pipeline_kwargs,
            )
            current["active_staining"] = s
        return current["active"]

    try:
        yield _get
    finally:
        if "active" in current:
            current["active"].close()
