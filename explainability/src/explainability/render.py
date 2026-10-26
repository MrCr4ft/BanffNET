"""Render float relevance maps as RGB images using JET and alpha overlays.

Defaults use alpha 0.3 and percentile-99 normalization. Optional enhancement
applies smoothing, small-component removal, and morphological closing.
OpenCV operations convert RGB to BGR only at the library boundary.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

log = logging.getLogger(__name__)

PathLike = Union[str, Path]


def normalize(
    relevance: np.ndarray,
    *,
    support: Optional[np.ndarray] = None,
    percentile: float = 99.0,
) -> np.ndarray:
    """Scale a relevance map to [0, 1] by its ``percentile`` value.

    ``support`` (bool mask of valid pixels) restricts the percentile to covered
    pixels — important for slide-level maps that are mostly empty canvas.
    """
    rel = np.asarray(relevance, dtype=np.float32)
    vals = rel[support] if support is not None and support.any() else rel.ravel()
    if vals.size == 0:
        return np.zeros_like(rel)
    hi = float(np.percentile(vals, percentile))
    if hi <= 0:
        hi = float(vals.max())
    if hi <= 0:
        return np.zeros_like(rel)
    return np.clip(rel / hi, 0.0, 1.0)


def enhance(
    relevance01: np.ndarray,
    *,
    smooth_sigma: float = 1.5,
    min_component_frac: float = 0.0005,
    floor_percentile: float = 0.0,
    close_ksize: int = 3,
) -> np.ndarray:
    """Gentle, tunable post-processing for a [0, 1] relevance map.

    Pipeline: light Gaussian smoothing → optional soft floor → removal of tiny
    connected specks → morphological close. Crucially it **keeps the real
    relevance values** (no Otsu hard-binarization) and measures "tiny" against
    the *covered* area, not the whole (mostly-empty) canvas — so scattered small
    foci survive.

    Parameters
    ----------
    smooth_sigma
        Gaussian blur sigma. ``0`` disables smoothing.
    min_component_frac
        Drop connected components smaller than this fraction of the **covered
        (non-zero) area**. Small default removes only true speckle; ``0`` keeps
        everything.
    floor_percentile
        Optional soft floor: zero out the weakest ``floor_percentile`` % of the
        covered values. ``0`` (default) applies no floor — turn it up for more
        aggressive suppression.
    close_ksize
        Ellipse kernel size for morphological closing (fills pinholes). ``<=1``
        disables it.
    """
    rel = np.asarray(relevance01, dtype=np.float32)
    if rel.max() <= 0:
        return rel

    if smooth_sigma > 0:
        rel = cv2.GaussianBlur(rel, ksize=(0, 0), sigmaX=smooth_sigma, sigmaY=smooth_sigma)

    support = rel > 0
    n_support = int(support.sum())
    if n_support == 0:
        return rel

    # Optional soft floor among the covered values (not a hard Otsu cut).
    if floor_percentile and floor_percentile > 0:
        t = float(np.percentile(rel[support], floor_percentile))
        rel = np.where(rel >= t, rel, 0.0).astype(np.float32)

    # Remove tiny specks, sized relative to the covered area (keep values).
    if min_component_frac and min_component_frac > 0:
        mask = (rel > 0).astype(np.uint8)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        min_area = max(1, int(min_component_frac * n_support))
        keep = np.zeros_like(mask)
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                keep[labels == i] = 1
        rel = np.where(keep > 0, rel, 0.0).astype(np.float32)

    # Grayscale close: fills pinholes without discarding intensity.
    if close_ksize and close_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        rel = cv2.morphologyEx(rel, cv2.MORPH_CLOSE, k)

    mx = float(rel.max())
    return (rel / mx).astype(np.float32) if mx > 0 else rel


def colorize(relevance01: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """[0, 1] map → RGB uint8 heatmap."""
    u8 = np.clip(np.asarray(relevance01) * 255, 0, 255).astype(np.uint8)
    bgr = cv2.applyColorMap(u8, colormap)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def overlay(
    image_rgb: np.ndarray,
    relevance01: np.ndarray,
    *,
    alpha: float = 0.3,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Blend a heatmap onto ``image_rgb`` with per-pixel opacity ``relevance*alpha``.

    Matches the paper (transparency 0.3): flat regions stay fully visible, high-
    relevance regions get up to ``alpha`` of heatmap.
    """
    img = np.asarray(image_rgb)[..., :3].astype(np.float32)
    rel = np.asarray(relevance01, dtype=np.float32)
    if rel.shape[:2] != img.shape[:2]:
        rel = cv2.resize(rel, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    heat = colorize(rel, colormap).astype(np.float32)
    a = (rel * alpha)[..., None]
    return (img * (1 - a) + heat * a).clip(0, 255).astype(np.uint8)


def save_rgb(image_rgb: np.ndarray, path: PathLike) -> Path:
    """Write an RGB uint8 image (OpenCV expects BGR)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), cv2.cvtColor(np.asarray(image_rgb), cv2.COLOR_RGB2BGR))
    return p


def save_raw(relevance: np.ndarray, path: PathLike) -> Path:
    """Save the raw float relevance losslessly as ``.npy`` for later re-render/quant."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(p), np.asarray(relevance, dtype=np.float32))
    return p
