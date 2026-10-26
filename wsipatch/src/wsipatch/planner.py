"""Build patch plans from slide metadata, ROI geometry, and optional tissue masks.

Choose pyramid levels, filter tissue using strip-scanned integral images, and
assemble ``SlideIndex`` objects. Directory helpers also write a dataset
manifest. Planning reads slide metadata and masks without extracting patches.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import (
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import cv2
import numpy as np
import pyroaring

from .types import ROIGrid, ROISpec, SlideIndex, TargetSpec

log = logging.getLogger(__name__)

MaskLike = Union[np.ndarray, str, Path]
MissingMaskPolicy = Literal["accept_all", "reject_all", "raise"]
FormatName = Literal["wsix", "json"]
ROILoader = Callable[[Path], Optional[List[ROISpec]]]
"""``slide_path -> Optional[List[ROISpec]]``; ``None`` = no annotation found."""


def _choose_level(
    base_mpp_x: float,
    base_mpp_y: float,
    level_downsamples: Sequence[float],
    target_mpp: float,
    max_upscale: float = 1.0,
) -> int:
    """Pick the pyramid level whose mpp is closest to ``target_mpp``.

    Anisotropic mpp is collapsed to the larger axis (conservative: keeps
    the finer axis below the requested mpp). Upscale factors above
    ``max_upscale`` are rejected; if nothing qualifies, returns 0.
    """
    base_mpp = max(base_mpp_x, base_mpp_y)
    best_idx, best_diff = 0, float("inf")
    for i, ds in enumerate(level_downsamples):
        upscale = (base_mpp * ds) / target_mpp  # >1 means we'd need to upscale
        if upscale > max_upscale and upscale > 1.0:
            continue
        diff = abs(upscale - 1.0)
        if diff < best_diff:
            best_diff, best_idx = diff, i
    return best_idx


def _load_mask(path: Union[str, Path]) -> Optional[np.ndarray]:
    """Load a grayscale mask; None if missing or unreadable."""
    p = Path(path)
    if not p.exists():
        return None
    return cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)


def _resolve_mask(
    mask: Optional[MaskLike],
    on_missing: MissingMaskPolicy,
    context: str,
) -> Optional[np.ndarray]:
    """Coerce a ``MaskLike`` to ``np.ndarray`` or ``None`` (per policy)."""
    if mask is None:
        if on_missing == "raise":
            raise FileNotFoundError(f"No mask provided for {context}")
        return None
    if isinstance(mask, np.ndarray):
        return mask
    arr = _load_mask(mask)
    if arr is None:
        if on_missing == "raise":
            raise FileNotFoundError(f"Mask not found: {mask} (context: {context})")
        log.warning(
            "Mask not found at %s (context %s); applying on_missing_mask=%r",
            mask,
            context,
            on_missing,
        )
    return arr


def _scan_grid_for_tissue(
    mask: np.ndarray,
    rescaled_size: Tuple[int, int],
    patch_size: int,
    stride: int,
    tissue_threshold: float,
    grid_w: int,
    grid_h: int,
) -> List[int]:
    """Strip-scan integral over the rescaled mask, returning valid flat indices.

    Computes one ``patch_size``-tall integral strip at a time
    (``int32``), so peak memory is bounded by ``ws × patch_size × 4 B``
    rather than the full ``ws × hs × 8 B`` int64 integral image. Output
    is bit-identical to a full-integral implementation.
    """
    if grid_w <= 0 or grid_h <= 0:
        return []

    ws, hs = rescaled_size
    patch_area = patch_size * patch_size

    if mask.shape[:2] != (hs, ws):
        scaled = cv2.resize(mask, (ws, hs), interpolation=cv2.INTER_NEAREST)
    else:
        scaled = mask
    binary = (scaled > 0).astype(np.uint8)

    valid: List[int] = []
    for r in range(grid_h):
        y = r * stride
        y2 = y + patch_size
        strip = binary[y:y2, :]
        integral = cv2.integral(strip, sdepth=cv2.CV_32S)  # (patch_size+1, ws+1)
        for c in range(grid_w):
            x = c * stride
            x2 = x + patch_size
            tissue = int(
                integral[patch_size, x2]
                - integral[0, x2]
                - integral[patch_size, x]
                + integral[0, x]
            )
            if tissue / patch_area >= tissue_threshold:
                valid.append(r * grid_w + c)
    return valid


def _resolve_slide_metadata(
    slide_path: Path,
    fallback_mpp_x: float,
    fallback_mpp_y: float,
) -> dict:
    """Brief open() to pull pyramid + mpp metadata. No pixels are read."""
    import openslide  # local import so the rest of the module is openslide-free

    slide = openslide.OpenSlide(str(slide_path))
    try:
        return {
            "base_mpp_x": float(slide.properties.get("openslide.mpp-x", fallback_mpp_x)),
            "base_mpp_y": float(slide.properties.get("openslide.mpp-y", fallback_mpp_y)),
            "level_count": slide.level_count,
            "level_downsamples": [float(d) for d in slide.level_downsamples],
            "level_dims": [tuple(slide.level_dimensions[i]) for i in range(slide.level_count)],
        }
    finally:
        slide.close()


def _full_slide_roi(level0_dims: Tuple[int, int]) -> ROISpec:
    w, h = level0_dims
    return ROISpec(x0=0, y0=0, w=w, h=h, name="full_slide")


def _build_roi_grid(
    *,
    roi: ROISpec,
    roi_index: int,
    spec: TargetSpec,
    base_mpp_x: float,
    base_mpp_y: float,
    level_downsamples: Sequence[float],
    max_upscale: float,
    mask: Optional[np.ndarray],
    on_missing_mask: MissingMaskPolicy,
    tissue_threshold: float,
) -> ROIGrid:
    """Build the patch grid for one (ROI, target) pair."""
    rlev = _choose_level(base_mpp_x, base_mpp_y, level_downsamples, spec.mpp, max_upscale)
    ds = level_downsamples[rlev]
    lvl_mpp_x = base_mpp_x * ds
    lvl_mpp_y = base_mpp_y * ds
    dsc_x = lvl_mpp_x / spec.mpp
    dsc_y = lvl_mpp_y / spec.mpp
    interp = "area" if (dsc_x < 1.0 or dsc_y < 1.0) else "linear"

    # ROI dims: read level, then rescaled frame.
    wL = max(1, int(math.ceil(roi.w / ds)))
    hL = max(1, int(math.ceil(roi.h / ds)))
    ws = max(1, int(round(wL * dsc_x)))
    hs = max(1, int(round(hL * dsc_y)))

    # Grid: only full patches.
    if ws >= spec.patch_size and hs >= spec.patch_size:
        gw = (ws - spec.patch_size) // spec.stride + 1
        gh = (hs - spec.patch_size) // spec.stride + 1
    else:
        gw = gh = 0

    if mask is None:
        # accept_all => keep every cell; reject_all => empty; raise was handled upstream.
        valid: List[int] = list(range(gw * gh)) if on_missing_mask != "reject_all" else []
    else:
        valid = _scan_grid_for_tissue(
            mask=mask,
            rescaled_size=(ws, hs),
            patch_size=spec.patch_size,
            stride=spec.stride,
            tissue_threshold=tissue_threshold,
            grid_w=gw,
            grid_h=gh,
        )

    bm = pyroaring.BitMap(valid)
    bm.run_optimize()

    return ROIGrid(
        roi_index=roi_index,
        roi_name=roi.name,
        read_level=rlev,
        level_downsample=float(ds),
        read_origin=(roi.x0, roi.y0),
        read_size_at_level=(wL, hL),
        rescale_factor_x=float(dsc_x),
        rescale_factor_y=float(dsc_y),
        rescaled_w=ws,
        rescaled_h=hs,
        interpolation=interp,
        patch_size=spec.patch_size,
        stride=spec.stride,
        grid_w=gw,
        grid_h=gh,
        bitmap=bm,
    )


def build_slide_index_from_metadata(
    *,
    slide_id: str,
    slide_path: str,
    base_mpp_x: float,
    base_mpp_y: float,
    level_count: int,
    level_downsamples: Sequence[float],
    level_dims: Sequence[Tuple[int, int]],
    targets: Sequence[TargetSpec],
    rois: Optional[Sequence[ROISpec]] = None,
    masks: Optional[Sequence[Optional[MaskLike]]] = None,
    tissue_threshold: float = 0.50,
    max_upscale: float = 1.0,
    on_missing_mask: MissingMaskPolicy = "accept_all",
    roi_source: str = "",
) -> SlideIndex:
    """Build a :class:`SlideIndex` from pre-resolved slide metadata.

    The lowest-level entry point. Use it when pyramid metadata is
    already in hand (cached, or under test) to avoid touching openslide.
    """
    if not targets:
        raise ValueError("at least one TargetSpec is required")

    level_downsamples = [float(d) for d in level_downsamples]
    level_dims = [tuple(d) for d in level_dims]

    if rois is None:
        rois = [_full_slide_roi(level_dims[0])]
        if not roi_source:
            roi_source = "full_slide"
    else:
        rois = list(rois)
        if not roi_source:
            roi_source = "user_provided"
    if not rois:
        raise ValueError("At least one ROI is required.")
    n_rois = len(rois)

    if masks is None:
        resolved_masks: List[Optional[np.ndarray]] = [None] * n_rois
    else:
        if len(masks) != n_rois:
            raise ValueError(f"masks length {len(masks)} != rois length {n_rois}")
        resolved_masks = [
            _resolve_mask(m, on_missing_mask, f"{slide_id}_roi_{i}") for i, m in enumerate(masks)
        ]

    roi_grids: List[List[ROIGrid]] = []
    for spec in targets:
        per_target: List[ROIGrid] = []
        for r_idx, roi in enumerate(rois):
            g = _build_roi_grid(
                roi=roi,
                roi_index=r_idx,
                spec=spec,
                base_mpp_x=base_mpp_x,
                base_mpp_y=base_mpp_y,
                level_downsamples=level_downsamples,
                max_upscale=max_upscale,
                mask=resolved_masks[r_idx],
                on_missing_mask=on_missing_mask,
                tissue_threshold=tissue_threshold,
            )
            per_target.append(g)
            log.debug(
                "%s · target=%s · roi=%d · grid=%dx%d · valid=%d",
                slide_id,
                spec.name,
                r_idx,
                g.grid_w,
                g.grid_h,
                g.num_valid,
            )
        roi_grids.append(per_target)

    return SlideIndex(
        slide_id=slide_id,
        slide_path=slide_path,
        base_mpp_x=float(base_mpp_x),
        base_mpp_y=float(base_mpp_y),
        level_count=int(level_count),
        level_downsamples=level_downsamples,
        level_dims=level_dims,
        targets=list(targets),
        roi_grids=roi_grids,
        roi_source=roi_source,
    )


def build_slide_index(
    *,
    slide_path: Union[str, Path],
    targets: Sequence[TargetSpec],
    rois: Optional[Sequence[ROISpec]] = None,
    masks: Optional[Sequence[Optional[MaskLike]]] = None,
    tissue_threshold: float = 0.50,
    max_upscale: float = 1.0,
    on_missing_mask: MissingMaskPolicy = "accept_all",
    fallback_mpp_x: float = 0.25,
    fallback_mpp_y: float = 0.25,
    slide_id: Optional[str] = None,
    roi_source: str = "",
) -> SlideIndex:
    """Build a :class:`SlideIndex` for one slide.

    Opens the WSI briefly to read pyramid + mpp metadata, then defers
    to :func:`build_slide_index_from_metadata`. Pixels are **not** read.
    """
    slide_path = Path(slide_path)
    if slide_id is None:
        slide_id = slide_path.stem
    meta = _resolve_slide_metadata(slide_path, fallback_mpp_x, fallback_mpp_y)
    return build_slide_index_from_metadata(
        slide_id=slide_id,
        slide_path=str(slide_path),
        targets=targets,
        rois=rois,
        masks=masks,
        tissue_threshold=tissue_threshold,
        max_upscale=max_upscale,
        on_missing_mask=on_missing_mask,
        roi_source=roi_source,
        **meta,
    )


def plan_slide_multi(
    slide_path: Union[str, Path],
    targets: Sequence[TargetSpec],
    *,
    rois: Optional[Sequence[ROISpec]] = None,
    masks: Optional[Sequence[Optional[MaskLike]]] = None,
    tissue_threshold: float = 0.50,
    max_upscale: float = 1.0,
    on_missing_mask: MissingMaskPolicy = "accept_all",
    fallback_mpp_x: float = 0.25,
    fallback_mpp_y: float = 0.25,
    slide_id: Optional[str] = None,
    roi_source: str = "",
) -> SlideIndex:
    """Plan one slide at multiple resolutions.

    All targets share the same ROI list; each gets its own pyramid
    level, rescale factor, and Roaring bitmap.
    """
    return build_slide_index(
        slide_path=slide_path,
        targets=targets,
        rois=rois,
        masks=masks,
        tissue_threshold=tissue_threshold,
        max_upscale=max_upscale,
        on_missing_mask=on_missing_mask,
        fallback_mpp_x=fallback_mpp_x,
        fallback_mpp_y=fallback_mpp_y,
        slide_id=slide_id,
        roi_source=roi_source,
    )


def plan_slide(
    slide_path: Union[str, Path],
    patch_size: int,
    mpp: float,
    *,
    stride: Optional[int] = None,
    name: str = "default",
    **kwargs,
) -> SlideIndex:
    """Plan a single slide at one resolution. Convenience around
    :func:`plan_slide_multi`.
    """
    target = TargetSpec(name=name, patch_size=patch_size, mpp=mpp, stride=stride)
    return plan_slide_multi(slide_path=slide_path, targets=[target], **kwargs)


def _make_asap_loader(
    annotations_dir: Optional[Path],
    annotations_ext: str,
) -> ROILoader:
    """Build a loader that resolves ``<dir>/<slide_stem>.<ext>`` per slide."""
    from .annotations import load_asap_rois

    if annotations_dir is None:
        return lambda _: None
    ext = annotations_ext.lstrip(".")

    def _loader(slide_path: Path) -> Optional[List[ROISpec]]:
        anno_path = annotations_dir / f"{slide_path.stem}.{ext}"
        if not anno_path.exists():
            return None
        rois = load_asap_rois(anno_path, clip_to_origin=True, min_area=1)
        return rois if rois else None

    return _loader


def _resolve_roi_masks(
    slide_id: str,
    n_rois: int,
    mask_dir: Optional[Path],
    mask_template: str,
    mask_ext: str,
) -> List[Optional[Path]]:
    """Return one mask path per ROI (or ``None`` if the file is missing)."""
    if mask_dir is None:
        return [None] * n_rois
    ext = mask_ext.lstrip(".")
    out: List[Optional[Path]] = []
    for i in range(n_rois):
        p = mask_dir / mask_template.format(slide_id=slide_id, roi_idx=i, ext=ext)
        out.append(p if p.exists() else None)
    return out


def _plan_one(
    *,
    slide_path: Path,
    targets: Sequence[TargetSpec],
    roi_loader: ROILoader,
    mask_dir: Optional[Path],
    mask_template: str,
    mask_ext: str,
    tissue_threshold: float,
    max_upscale: float,
    on_missing_mask: MissingMaskPolicy,
    use_full_slide_if_no_annotation: bool,
    fallback_mpp_x: float,
    fallback_mpp_y: float,
) -> Optional[SlideIndex]:
    """Plan one slide; ``None`` means "skip this slide" (per policy)."""
    rois = roi_loader(slide_path)
    if rois is None:
        if use_full_slide_if_no_annotation:
            roi_source = "full_slide:no_annotation"
        else:
            log.warning("No annotation for %s — skipping", slide_path.stem)
            return None
    else:
        roi_source = "asap_xml"

    n_rois = len(rois) if rois is not None else 1
    mask_paths = _resolve_roi_masks(
        slide_id=slide_path.stem,
        n_rois=n_rois,
        mask_dir=mask_dir,
        mask_template=mask_template,
        mask_ext=mask_ext,
    )
    masks = mask_paths if mask_dir is not None else None

    return build_slide_index(
        slide_path=slide_path,
        targets=targets,
        rois=rois,
        masks=masks,
        tissue_threshold=tissue_threshold,
        max_upscale=max_upscale,
        on_missing_mask=on_missing_mask,
        fallback_mpp_x=fallback_mpp_x,
        fallback_mpp_y=fallback_mpp_y,
        roi_source=roi_source,
    )


def plan_directory_multi(
    biopsy_dir: Union[str, Path],
    targets: Sequence[TargetSpec],
    output_dir: Union[str, Path],
    *,
    annotations_dir: Optional[Union[str, Path]] = None,
    annotations_ext: str = "xml",
    mask_dir: Optional[Union[str, Path]] = None,
    mask_ext: str = "png",
    mask_template: str = "{slide_id}_roi_{roi_idx}_mask.{ext}",
    biopsy_ext: str = "tiff",
    tissue_threshold: float = 0.50,
    max_upscale: float = 1.0,
    on_missing_mask: MissingMaskPolicy = "accept_all",
    use_full_slide_if_no_annotation: bool = True,
    fallback_mpp_x: float = 0.25,
    fallback_mpp_y: float = 0.25,
    format: FormatName = "wsix",
    skip_existing: bool = True,
    progress: bool = True,
    roi_loader: Optional[ROILoader] = None,
    include_coords: bool = False,
) -> Path:
    """Plan every slide in a directory at multiple resolutions.

    Writes one plan file per slide plus a ``dataset_manifest.json`` summary.
    Re-running over an existing output dir skips slides whose plan already
    exists, unless ``skip_existing=False``.
    """
    from .io import load_index, save_index

    biopsy_dir = Path(biopsy_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir_p = Path(annotations_dir) if annotations_dir else None
    mask_dir_p = Path(mask_dir) if mask_dir else None

    if not targets:
        raise ValueError("at least one TargetSpec is required")
    if roi_loader is None:
        roi_loader = _make_asap_loader(annotations_dir_p, annotations_ext)

    biopsy_glob = f"*.{biopsy_ext.lstrip('.')}"
    slides = sorted(biopsy_dir.glob(biopsy_glob))
    log.info("Found %d slides under %s (%s)", len(slides), biopsy_dir, biopsy_glob)

    iterator = slides
    if progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(slides, desc="Planning", unit="slide")
        except ImportError:
            pass

    suffix = ".wsix" if format == "wsix" else ".json"
    manifest_entries: List[dict] = []
    totals: Dict[str, int] = {t.name: 0 for t in targets}
    n_ok = 0

    for slide_path in iterator:
        stem = slide_path.stem
        out_path = output_dir / f"{stem}{suffix}"

        if skip_existing and out_path.exists():
            try:
                existing = load_index(out_path)
                vpt = existing.valid_per_target()
                manifest_entries.append(
                    {
                        "stem": stem,
                        "index_file": out_path.name,
                        "skipped": True,
                        "valid_per_target": {
                            existing.targets[i].name: vpt[i] for i in range(len(existing.targets))
                        },
                        "total_valid": existing.total_valid(),
                    }
                )
                for i, t in enumerate(existing.targets):
                    totals[t.name] = totals.get(t.name, 0) + vpt[i]
                n_ok += 1
            except Exception as e:
                log.warning("Could not re-read existing %s: %s", out_path, e)
            continue

        try:
            idx = _plan_one(
                slide_path=slide_path,
                targets=targets,
                roi_loader=roi_loader,
                mask_dir=mask_dir_p,
                mask_template=mask_template,
                mask_ext=mask_ext,
                tissue_threshold=tissue_threshold,
                max_upscale=max_upscale,
                on_missing_mask=on_missing_mask,
                use_full_slide_if_no_annotation=use_full_slide_if_no_annotation,
                fallback_mpp_x=fallback_mpp_x,
                fallback_mpp_y=fallback_mpp_y,
            )
        except Exception as e:
            log.error("Failed to plan %s: %s", stem, e)
            manifest_entries.append({"stem": stem, "index_file": None, "error": str(e)})
            continue

        if idx is None:
            continue

        save_index(idx, out_path, format=format, include_coords=include_coords)
        vpt = idx.valid_per_target()
        for i, t in enumerate(targets):
            totals[t.name] += vpt[i]
        manifest_entries.append(
            {
                "stem": stem,
                "index_file": out_path.name,
                "valid_per_target": {targets[i].name: vpt[i] for i in range(len(targets))},
                "total_valid": idx.total_valid(),
            }
        )
        n_ok += 1
        log.info(
            "%s: %s",
            stem,
            ", ".join(f"{targets[i].name}={vpt[i]}" for i in range(len(targets))),
        )

    manifest = {
        "format": "wsipatch-manifest",
        "format_version": 1,
        "biopsy_dir": str(biopsy_dir),
        "annotations_dir": str(annotations_dir_p) if annotations_dir_p else None,
        "mask_dir": str(mask_dir_p) if mask_dir_p else None,
        "index_format": format,
        "targets": [
            {"name": t.name, "patch_size": t.patch_size, "mpp": t.mpp, "stride": t.stride}
            for t in targets
        ],
        "tissue_threshold": tissue_threshold,
        "max_upscale": max_upscale,
        "on_missing_mask": on_missing_mask,
        "totals_per_target": totals,
        "n_slides_indexed": n_ok,
        "n_slides_seen": len(slides),
        "slides": manifest_entries,
    }
    manifest_path = output_dir / "dataset_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path


def plan_directory(
    biopsy_dir: Union[str, Path],
    patch_size: int,
    mpp: float,
    output_dir: Union[str, Path],
    *,
    stride: Optional[int] = None,
    name: str = "default",
    **kwargs,
) -> Path:
    """Plan every slide in a directory at a single resolution. Convenience
    wrapper around :func:`plan_directory_multi`.
    """
    target = TargetSpec(name=name, patch_size=patch_size, mpp=mpp, stride=stride)
    return plan_directory_multi(
        biopsy_dir=biopsy_dir,
        targets=[target],
        output_dir=output_dir,
        **kwargs,
    )
