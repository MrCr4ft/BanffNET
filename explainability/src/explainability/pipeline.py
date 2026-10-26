"""Generate slide-level and patch-level BanffNET lesion saliency maps.

Plan or load patches, compute probabilities, select patches by threshold, and
propagate GAE relevance through UNI and a lesion head. Slide maps average
probability-weighted relevance per ROI. Features and the backbone are shared
across lesion heads.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import yaml
from banffnet.defaults import DEFAULT_MPP, DEFAULT_PATCH_SIZE, staining_paths

from . import io as _io
from .aggregate import ROIAccumulator, read_roi_rgb
from .gae import saliency_for_patches
from .model import ExplainableBanffModel
from .thresholding import resolve_threshold, select_mask

log = logging.getLogger(__name__)

PathLike = Union[str, Path]

#: Paper stride for whole-slide saliency sampling.
DEFAULT_STRIDE = 64


class ExplainabilityPipeline:
    """Lesion saliency orchestrator for one staining / model set."""

    def __init__(
        self,
        *,
        uni_checkpoint: PathLike,
        banff_config: PathLike,
        reinhard_config: Optional[PathLike] = None,
        binarization_model: Optional[PathLike] = None,
        workspace_dir: Optional[PathLike] = None,
        device: Union[str, torch.device] = "cuda",
        batch_size: int = 32,
        saliency_batch_size: int = 8,
        patch_size: int = DEFAULT_PATCH_SIZE,
        mpp: float = DEFAULT_MPP,
        stride: int = DEFAULT_STRIDE,
        tissue_threshold: float = 0.5,
        image_size: int = 224,
        gip_target: str = "classifier",
        gip_detach_gate: bool = False,
        progress: bool = True,
        amp: bool = False,
    ):
        self.uni_checkpoint = str(uni_checkpoint)
        self.banff_config = str(banff_config)
        self.reinhard_config = str(reinhard_config) if reinhard_config else None
        self.binarization_model = str(binarization_model) if binarization_model else None
        self.workspace_dir = Path(workspace_dir) if workspace_dir else None
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.saliency_batch_size = int(saliency_batch_size)
        self.patch_size = int(patch_size)
        self.mpp = float(mpp)
        self.stride = int(stride)
        self.tissue_threshold = float(tissue_threshold)
        self.image_size = int(image_size)
        self.gip_target = gip_target
        self.gip_detach_gate = bool(gip_detach_gate)
        self.progress = bool(progress)
        self.amp = bool(amp)

        self._config = yaml.safe_load(Path(self.banff_config).read_text())
        self._backbone = None  # lazy (needs timm)
        self._preprocess = None  # lazy (BatchPreprocess on device)
        self._models: Dict[str, ExplainableBanffModel] = {}

    @classmethod
    def from_staining(cls, staining: str, **overrides) -> "ExplainabilityPipeline":
        """Build from the configured asset paths for a staining."""
        d = staining_paths(staining)
        kwargs = dict(
            uni_checkpoint=d.uni_checkpoint,
            banff_config=d.banff_config,
            reinhard_config=d.reinhard_config,
            binarization_model=d.binarization_model,
        )
        kwargs.update(overrides)
        return cls(**kwargs)

    def __enter__(self) -> "ExplainabilityPipeline":
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def close(self) -> None:
        self._models.clear()
        self._backbone = None
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass

    @property
    def preprocess(self):
        if self._preprocess is None:
            from features_extraction import BatchPreprocess, load_reinhard_from_yaml

            reinhard = (
                load_reinhard_from_yaml(self.reinhard_config) if self.reinhard_config else None
            )
            self._preprocess = BatchPreprocess(
                image_size=self.image_size,
                reinhard=reinhard,
            ).to(self.device)
        return self._preprocess

    def _backbone_model(self):
        if self._backbone is None:
            from .backbone import build_uni_vit_gae

            self._backbone = build_uni_vit_gae(self.uni_checkpoint, device=self.device)
        return self._backbone

    def _model(self, lesion: str) -> ExplainableBanffModel:
        if lesion not in self._models:
            self._models[lesion] = ExplainableBanffModel.from_config(
                uni_checkpoint=self.uni_checkpoint,
                banff_config=self.banff_config,
                lesion=lesion,
                device=self.device,
                backbone=self._backbone_model(),
                gip_target=self.gip_target,
                gip_detach_gate=self.gip_detach_gate,
            )
        return self._models[lesion]

    def resolve_lesions(self, lesions: Optional[Sequence[str]]) -> List[str]:
        """Expand ``None`` / ``["all"]`` to every lesion in the banff config."""
        all_lesions = list(self._config.keys())
        if not lesions or list(lesions) == ["all"]:
            return all_lesions
        unknown = [l for l in lesions if l not in all_lesions]
        if unknown:
            raise KeyError(f"Unknown lesion(s) {unknown}; available: {all_lesions}")
        return list(lesions)

    @torch.inference_mode()
    def _batched_features(
        self, patches: List[np.ndarray], *, desc: str = "features"
    ) -> torch.Tensor:
        """UNI features for all patches, computed **once** (shared across lesions).

        Forward-only, so it runs with attention storage off and — when
        ``self.amp`` and on CUDA — in float16 for speed. The tiny lesion heads
        are then applied to these features per lesion (see
        :meth:`ExplainableBanffModel.proba_from_features`), avoiding a full
        backbone pass per lesion.
        """
        import contextlib

        from .backbone import set_store_attention

        bb = self._backbone_model()
        n = len(patches)
        use_amp = self.amp and self.device.type == "cuda"
        set_store_attention(bb, False)
        chunks: List[torch.Tensor] = []
        try:
            starts = range(0, n, self.batch_size)
            for start in self._pbar(
                starts, desc=desc, total=(n + self.batch_size - 1) // self.batch_size
            ):
                end = min(start + self.batch_size, n)
                x = self.preprocess(np.stack(patches[start:end], axis=0)).to(
                    self.device, dtype=torch.float32
                )
                ctx = (
                    torch.autocast(device_type=self.device.type, dtype=torch.float16)
                    if use_amp
                    else contextlib.nullcontext()
                )
                with ctx:
                    feats = bb(x)
                chunks.append(feats.float())
        finally:
            set_store_attention(bb, True)  # restore for the GAE pass
        return torch.cat(chunks, dim=0) if chunks else torch.empty((0, 0), device=self.device)

    @staticmethod
    def _proba_from_features(model: ExplainableBanffModel, feats: torch.Tensor) -> np.ndarray:
        """Per-patch probability from shared UNI features (cheap: head only)."""
        return model.proba_from_features(feats).squeeze(-1).float().cpu().numpy()

    def _pbar(self, iterable, *, desc: str, total: Optional[int] = None, unit: str = "batch"):
        """Wrap an iterable in a tqdm bar when progress is enabled."""
        if not self.progress:
            return iterable
        from tqdm.auto import tqdm

        return tqdm(iterable, desc=desc, total=total, unit=unit, leave=False)

    def _pbar(self, iterable, *, desc: str, total: Optional[int] = None, unit: str = "batch"):
        """Wrap an iterable in a tqdm bar when progress is enabled."""
        if not self.progress:
            return iterable
        from tqdm.auto import tqdm

        return tqdm(iterable, desc=desc, total=total, unit=unit, leave=False)

    def _select(self, model, probs, threshold_spec):
        thr = resolve_threshold(
            threshold_spec,
            probs,
            model_type=model.model_type,
            params=model.threshold_params,
        )
        mask, thr = select_mask(probs, thr)
        return mask, thr

    def _build_plan(self, slide_path: PathLike, annotation_path: Optional[PathLike]):
        """Plan stride-64 patches inside the slide's ROIs, masked by tissue."""
        from wsipatch import ASAPAnnotations, TargetSpec, asap_to_rois, plan_slide_multi

        slide_id = Path(slide_path).stem
        if annotation_path:
            anns = ASAPAnnotations.from_xml_file(annotation_path)
        else:
            anns = ASAPAnnotations.from_slide(slide_path)
        rois = asap_to_rois(anns, clip_to_origin=True, min_area=1)
        if not rois:
            raise RuntimeError(f"No usable ROIs for {slide_id}")

        rois, masks = self._tissue_masks(slide_path, annotation_path, rois)
        target = TargetSpec(
            name="explain", patch_size=self.patch_size, mpp=self.mpp, stride=self.stride
        )
        return plan_slide_multi(
            slide_path,
            targets=[target],
            rois=rois,
            masks=masks,
            slide_id=slide_id,
            tissue_threshold=self.tissue_threshold,
        )

    def _tissue_masks(self, slide_path, annotation_path, rois):
        """Produce per-ROI tissue masks with ``binarization`` (optional).

        Returns ``(rois, masks)`` with ROIs lacking a mask dropped. When no
        binarization model is configured, returns ``(rois, None)`` so planning
        accepts the full ROI extent.
        """
        if not self.binarization_model:
            return rois, None
        from binarization import BinarizationPipeline
        from wsipatch import resolve_roi_id

        ws = self.workspace_dir or Path.cwd() / "explainability_workspace"
        binp = BinarizationPipeline.from_paths(
            roi_dir=Path(ws) / "rois",
            mask_dir=Path(ws) / "masks",
            model_path=self.binarization_model,
        )
        try:
            binp.process_slide(slide_path, annotation_path)
            slide_id = Path(slide_path).stem
            kept_rois, masks = [], []
            for i, roi in enumerate(rois):
                # Match the ROI identifier used in binarization mask filenames.
                roi_id = resolve_roi_id(roi.name, i)
                mp = binp.file_manager.mask_path(slide_id, roi_id)
                if Path(mp).exists():
                    kept_rois.append(roi)
                    masks.append(str(mp))
                else:
                    log.warning(
                        "[%s] ROI %d (%s) has no tissue mask — dropping", slide_id, i, roi_id
                    )
            return (kept_rois, masks) if kept_rois else (rois, None)
        finally:
            binp.close()

    def explain_slide_level(
        self,
        *,
        output_dir: PathLike,
        lesions: Optional[Sequence[str]] = None,
        slide_path: Optional[PathLike] = None,
        annotation_path: Optional[PathLike] = None,
        plan: Optional[PathLike] = None,
        threshold: str = "paper",
        save_opts: Optional[_io.SaveOptions] = None,
    ) -> None:
        """Biopsy-averaged, probability-weighted overlay per ROI per lesion."""
        import openslide

        from wsipatch import SlideIndex, load_index

        save_opts = save_opts or _io.SaveOptions()
        lesions = self.resolve_lesions(lesions)

        if plan is not None:
            index = plan if isinstance(plan, SlideIndex) else load_index(plan)
            resolved_slide = str(slide_path or index.slide_path)
        else:
            if slide_path is None:
                raise ValueError("slide-level needs --slide-path or --plan")
            index = self._build_plan(slide_path, annotation_path)
            resolved_slide = str(slide_path)

        slide_id = index.slide_id
        # Own plans have a single 'explain' target; a user --plan may carry more.
        t_idx = 0
        if index.num_targets > 1:
            try:
                t_idx = index.resolve_target("explain")
            except (KeyError, ValueError):
                log.warning(
                    "Plan has %d targets and no 'explain' target — using target 0",
                    index.num_targets,
                )
        grids = index.roi_grids[t_idx]
        models = {l: self._model(l) for l in lesions}

        slide = openslide.OpenSlide(resolved_slide)
        try:
            for grid in grids:
                if grid.num_valid == 0:
                    continue
                roi_rgb = read_roi_rgb(slide, grid)
                positions, patches = self._slice_roi_patches(roi_rgb, grid)
                if not patches:
                    continue
                log.info(
                    "[%s] ROI %d (%s): %d patches",
                    slide_id,
                    grid.roi_index,
                    grid.roi_label,
                    len(patches),
                )
                roi = grid.roi_label

                # Reuse ROI features across lesion heads.
                feats = self._batched_features(patches, desc=f"ROI {roi} | features")

                for lesion, model in models.items():
                    probs = self._proba_from_features(model, feats)
                    mask, thr = self._select(model, probs, threshold)
                    sel = np.nonzero(mask)[0]
                    log.info(
                        "  %s: %d/%d patches @ threshold %s", lesion, sel.size, len(patches), thr
                    )
                    if sel.size == 0:
                        continue
                    sel_patches = [patches[i] for i in sel]
                    for tgt in model.saliency_targets():
                        model.active_target = tgt
                        saliency = saliency_for_patches(
                            model,
                            self.preprocess,
                            sel_patches,
                            device=self.device,
                            batch_size=self.saliency_batch_size,
                            out_size=self.image_size,
                            progress=self.progress,
                            desc=f"ROI {roi} | {lesion}"
                            + (f" | {tgt}" if model.is_gip else "")
                            + " | saliency",
                        )
                        acc = ROIAccumulator(grid.rescaled_h, grid.rescaled_w)
                        for j, i in enumerate(sel):
                            x, y = positions[i]
                            acc.add(x, y, grid.patch_size, float(probs[i]), saliency[j])
                        _io.save_slide_roi(
                            save_opts,
                            output_dir,
                            slide_id,
                            lesion,
                            roi,
                            roi_rgb,
                            acc.result(),
                            support=acc.support(),
                            tag=(tgt if model.is_gip else ""),
                        )
                    model.active_target = None
        finally:
            slide.close()

    def _slice_roi_patches(self, roi_rgb: np.ndarray, grid):
        """Slice valid stride patches out of a rescaled-ROI image."""
        ps = grid.patch_size
        h, w = roi_rgb.shape[:2]
        positions, patches = [], []
        for flat in grid.bitmap:
            row, col = divmod(flat, grid.grid_w)
            x, y = col * grid.stride, row * grid.stride
            if x + ps > w or y + ps > h:
                continue
            positions.append((x, y))
            patches.append(np.ascontiguousarray(roi_rgb[y : y + ps, x : x + ps]))
        return positions, patches

    def explain_patch_level(
        self,
        *,
        output_dir: PathLike,
        lesions: Optional[Sequence[str]] = None,
        slide_path: Optional[PathLike] = None,
        annotation_path: Optional[PathLike] = None,
        plan: Optional[PathLike] = None,
        patches_h5: Optional[PathLike] = None,
        features_h5: Optional[PathLike] = None,
        threshold: str = "paper",
        max_patches: Optional[int] = None,
        save_opts: Optional[_io.SaveOptions] = None,
    ) -> None:
        """Per-patch saliency for patches passing the probability threshold."""
        save_opts = save_opts or _io.SaveOptions()
        lesions = self.resolve_lesions(lesions)

        slide_id, patches, features = self._load_patch_source(
            slide_path=slide_path,
            annotation_path=annotation_path,
            plan=plan,
            patches_h5=patches_h5,
            features_h5=features_h5,
            max_patches=max_patches,
        )
        if not patches:
            log.warning("[%s] no patches to explain", slide_id)
            return
        log.info("[%s] patch-level on %d patches", slide_id, len(patches))

        # Features either come from the H5 (reused) or are computed once here
        # and shared across every lesion head.
        if features is not None:
            feats = torch.from_numpy(features).to(self.device, dtype=torch.float32)
        else:
            feats = self._batched_features(patches, desc=f"{slide_id} | features")

        for lesion in lesions:
            model = self._model(lesion)
            probs = self._proba_from_features(model, feats)

            mask, thr = self._select(model, probs, threshold)
            sel = np.nonzero(mask)[0]
            log.info("  %s: %d/%d patches @ threshold %s", lesion, sel.size, len(patches), thr)
            if sel.size == 0:
                continue
            sel_patches = [patches[i] for i in sel]
            for tgt in model.saliency_targets():
                model.active_target = tgt
                saliency = saliency_for_patches(
                    model,
                    self.preprocess,
                    sel_patches,
                    device=self.device,
                    batch_size=self.saliency_batch_size,
                    out_size=self.image_size,
                    progress=self.progress,
                    desc=f"{lesion}" + (f" | {tgt}" if model.is_gip else "") + " | saliency",
                )
                tag = tgt if model.is_gip else ""
                for j, i in enumerate(sel):
                    _io.save_patch(
                        save_opts,
                        output_dir,
                        slide_id,
                        lesion,
                        int(i),
                        float(probs[i]),
                        patches[i],
                        saliency[j],
                        tag=tag,
                    )
            model.active_target = None

    def _load_patch_source(
        self,
        *,
        slide_path,
        annotation_path,
        plan,
        patches_h5,
        features_h5,
        max_patches,
    ):
        """Resolve a patch-level input into ``(slide_id, patches, features|None)``."""
        import h5py

        features = None
        if features_h5:
            with h5py.File(features_h5, "r") as f:
                if "features" in f:
                    features = f["features"][:]

        if patches_h5:
            slide_id = Path(patches_h5).stem
            from features_extraction import detect_patches_dataset

            key = detect_patches_dataset(patches_h5)
            with h5py.File(patches_h5, "r") as f:
                imgs = f[key][:] if max_patches is None else f[key][:max_patches]
            patches = [np.ascontiguousarray(im[..., :3]) for im in imgs]
        elif plan is not None or slide_path is not None:
            from wsipatch import PatchReader

            if plan is None:
                plan = self._build_plan(slide_path, annotation_path)
            with PatchReader(plan, slide_path=slide_path, low_memory=False) as reader:
                slide_id = reader.index.slide_id
                total = max_patches or reader.num_patches
                patches = []
                reader_iter = self._pbar(
                    reader.iter_patches(),
                    desc=f"{slide_id} | reading patches",
                    total=total,
                    unit="patch",
                )
                for patch, _meta in reader_iter:
                    patches.append(np.ascontiguousarray(patch))
                    if max_patches is not None and len(patches) >= max_patches:
                        break
        else:
            raise ValueError("patch-level needs one of --patches-h5, --plan, or --slide-path")

        if features is not None and len(features) != len(patches):
            log.warning(
                "features (%d) and patches (%d) count mismatch — recomputing probabilities",
                len(features),
                len(patches),
            )
            features = None
        return slide_id, patches, features
