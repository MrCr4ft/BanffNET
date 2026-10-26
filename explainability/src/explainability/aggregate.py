"""Reconstruct ROI saliency maps from probability-weighted patch relevance.

Patch maps accumulate in the rescaled-ROI frame defined by ``ROIGrid``.
Overlapping contributions are averaged before rendering.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

log = logging.getLogger(__name__)


class ROIAccumulator:
    """Probability-weighted running average of patch saliency over an ROI canvas."""

    def __init__(self, height: int, width: int):
        self.h = int(height)
        self.w = int(width)
        self._num = np.zeros((self.h, self.w), np.float32)
        self._den = np.zeros((self.h, self.w), np.float32)

    def add(self, x: int, y: int, patch_size: int, prob: float, saliency: np.ndarray) -> None:
        """Accumulate one patch.

        Args:
            x, y: patch top-left in the rescaled-ROI frame.
            patch_size: patch side length in that frame.
            prob: patch probability (the weight).
            saliency: (S, S) float saliency in [0, 1]; resized to ``patch_size``.
        """
        if prob <= 0.0:
            return
        x0, y0 = int(x), int(y)
        x1, y1 = min(x0 + patch_size, self.w), min(y0 + patch_size, self.h)
        if x1 <= x0 or y1 <= y0:
            return

        sm = np.squeeze(saliency).astype(np.float32)
        if sm.shape != (patch_size, patch_size):
            sm = cv2.resize(sm, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
        sm = sm[: y1 - y0, : x1 - x0]

        self._num[y0:y1, x0:x1] += prob * sm
        self._den[y0:y1, x0:x1] += prob

    @property
    def coverage(self) -> float:
        """Fraction of the canvas touched by at least one patch."""
        return float((self._den > 0).mean())

    def result(self) -> np.ndarray:
        """Biopsy-averaged relevance (H, W) float32; zero where no patch landed."""
        return np.divide(
            self._num,
            self._den,
            out=np.zeros_like(self._num),
            where=self._den > 0,
        )

    def support(self) -> np.ndarray:
        """Boolean (H, W) mask of pixels covered by any contributing patch."""
        return self._den > 0


def read_roi_rgb(slide, grid) -> np.ndarray:
    """Read one ROI at its plan geometry and rescale to the patch frame.

    Mirrors ``wsipatch.PatchReader._read_whole_roi`` using only the public
    :class:`~wsipatch.types.ROIGrid` fields, so the overlay background is
    pixel-aligned with the coordinates patches are placed at.

    Returns:
        (rescaled_h, rescaled_w, 3) uint8 RGB.
    """
    region = slide.read_region(grid.read_origin, grid.read_level, grid.read_size_at_level)
    roi = np.asarray(region)[..., :3]
    if grid.rescale_factor_x != 1.0 or grid.rescale_factor_y != 1.0:
        interp = cv2.INTER_AREA if grid.interpolation == "area" else cv2.INTER_LINEAR
        roi = cv2.resize(roi, (grid.rescaled_w, grid.rescaled_h), interpolation=interp)
    return np.ascontiguousarray(roi)
