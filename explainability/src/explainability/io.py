"""Save saliency artifacts under ``output_dir/slide_id/lesion``.

Slide artifacts use ``roi_{roi_id}`` prefixes; patch artifacts use
``patch_{k:05d}_p{prob}``. ``SaveOptions`` controls overlay PNGs, saliency PNGs,
and raw NumPy relevance maps. ROI identifiers follow ``wsipatch.resolve_roi_id``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

from . import render

PathLike = Union[str, Path]


@dataclass
class SaveOptions:
    """What to save and how to render it."""

    overlay: bool = True
    saliency_png: bool = True
    raw: bool = True
    alpha: float = 0.3
    normalize_percentile: float = 99.0
    enhance: bool = False
    enhance_params: Dict = field(default_factory=dict)

    def any(self) -> bool:
        return self.overlay or self.saliency_png or self.raw


def lesion_dir(output_dir: PathLike, slide_id: str, lesion: str) -> Path:
    d = Path(output_dir) / slide_id / lesion
    d.mkdir(parents=True, exist_ok=True)
    return d


def _finalize(relevance01: np.ndarray, opts: SaveOptions) -> np.ndarray:
    if opts.enhance:
        return render.enhance(relevance01, **opts.enhance_params)
    return relevance01


def save_slide_roi(
    opts: SaveOptions,
    out_dir: PathLike,
    slide_id: str,
    lesion: str,
    roi_id: Union[int, str],
    roi_rgb: np.ndarray,
    relevance: np.ndarray,
    support: Optional[np.ndarray] = None,
    tag: str = "",
) -> Dict[str, Path]:
    """Save the biopsy-averaged ROI map (overlay / standalone / raw).

    ``tag`` (e.g. a GIP saliency target) is appended to the filename stem when
    non-empty, so multiple maps for one ROI don't collide.
    """
    written: Dict[str, Path] = {}
    d = lesion_dir(out_dir, slide_id, lesion)
    stem = f"roi_{roi_id}_{tag}" if tag else f"roi_{roi_id}"

    if opts.raw:
        written["raw"] = render.save_raw(relevance, d / f"{stem}_relevance.npy")

    rel01 = render.normalize(relevance, support=support, percentile=opts.normalize_percentile)
    rel01 = _finalize(rel01, opts)

    if opts.overlay:
        img = render.overlay(roi_rgb, rel01, alpha=opts.alpha)
        written["overlay"] = render.save_rgb(img, d / f"{stem}_overlay.png")
    if opts.saliency_png:
        written["saliency"] = render.save_rgb(render.colorize(rel01), d / f"{stem}_saliency.png")
    return written


def save_patch(
    opts: SaveOptions,
    out_dir: PathLike,
    slide_id: str,
    lesion: str,
    patch_idx: int,
    prob: float,
    patch_rgb: np.ndarray,
    saliency: np.ndarray,
    tag: str = "",
) -> Dict[str, Path]:
    """Save one patch's saliency (overlay / standalone / raw)."""
    written: Dict[str, Path] = {}
    d = lesion_dir(out_dir, slide_id, lesion)
    stem = f"patch_{patch_idx:05d}_p{prob:.4f}"
    if tag:
        stem = f"{stem}_{tag}"

    if opts.raw:
        written["raw"] = render.save_raw(saliency, d / f"{stem}_relevance.npy")

    rel01 = render.normalize(saliency, percentile=opts.normalize_percentile)
    rel01 = _finalize(rel01, opts)

    if opts.overlay:
        img = render.overlay(patch_rgb, rel01, alpha=opts.alpha)
        written["overlay"] = render.save_rgb(img, d / f"{stem}_overlay.png")
    if opts.saliency_png:
        written["saliency"] = render.save_rgb(render.colorize(rel01), d / f"{stem}_saliency.png")
    return written
