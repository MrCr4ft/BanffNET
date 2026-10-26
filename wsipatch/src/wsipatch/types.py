"""Patch planning inputs and serialized grid data.

``TargetSpec`` and ``ROISpec`` describe requested geometry. ``ROIGrid`` and
``SlideIndex`` hold planned grids. Grid coordinates use the rescaled-ROI frame,
with row-major flat indices: ``flat = row * grid_w + col``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pyroaring

#: Characters unsafe to embed directly in a filename, collapsed to "_".
_UNSAFE_FILENAME_CHARS = re.compile(r'[\\/:*?"<>|\s]+')


def sanitize_roi_id(name: str) -> str:
    """Make a ROI/annotation name safe to embed in a filename."""
    return _UNSAFE_FILENAME_CHARS.sub("_", name.strip())


def resolve_roi_id(name: Optional[str], fallback_index: int) -> str:
    """Human-readable, filename-safe ROI identifier.

    Returns the sanitized ``name`` if it is a non-empty string (e.g. the
    ASAP annotation's ``Name`` attribute), otherwise the positional
    ``fallback_index`` as a string.

    This is the single source of truth for "ROI identity" across the
    pipeline: binarization uses it to name ROI/mask files, wsipatch
    uses it for :attr:`ROIGrid.roi_label` and per-patch metadata, and
    banffnet uses it to key per-ROI prediction output. Callers passing
    the *same* annotation name and the *same* fallback index always
    resolve to the same id, which is what keeps ROI/mask filenames and
    downstream lookups in sync.
    """
    name = (name or "").strip()
    return sanitize_roi_id(name) if name else str(fallback_index)


@dataclass
class TargetSpec:
    """A target patch-extraction resolution.

    All spatial fields are in the **target-mpp frame** — the coordinate
    space obtained after reading at the chosen pyramid level and resizing
    to ``mpp``.

    Parameters
    ----------
    name
        Identifier used in indices, manifests, and on-disk file names.
    patch_size
        Side length of square patches, in target-mpp pixels.
    mpp
        Target microns per pixel.
    stride
        Sliding-window stride in target-mpp pixels. Defaults to
        ``patch_size`` (non-overlapping tiling).
    """

    name: str
    patch_size: int
    mpp: float
    stride: Optional[int] = None

    def __post_init__(self) -> None:
        if self.stride is None:
            self.stride = self.patch_size
        if self.patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {self.patch_size}")
        if self.stride <= 0:
            raise ValueError(f"stride must be positive, got {self.stride}")
        if self.mpp <= 0:
            raise ValueError(f"mpp must be positive, got {self.mpp}")
        if not self.name:
            raise ValueError("name must be a non-empty string")


@dataclass
class ROISpec:
    """A region of interest at level-0 (full-resolution) coordinates.

    Parameters
    ----------
    x0, y0
        Top-left corner in level-0 pixels.
    w, h
        Width and height in level-0 pixels.
    name
        Annotation Name, when available.
    """

    x0: int
    y0: int
    w: int
    h: int
    name: Optional[str] = None

    def __post_init__(self) -> None:
        pass


@dataclass
class ROIGrid:
    """Per-(target, ROI) patch grid backed by a Roaring bitmap.

    All spatial fields refer to the **rescaled-ROI frame**. The bitmap
    encodes which cells of the ``grid_h × grid_w`` lattice passed the
    tissue-coverage test, in row-major order: ``flat = row * grid_w + col``.
    For each valid flat index, the patch top-left in the rescaled frame
    is ``(col * stride, row * stride)``.
    """

    roi_index: int

    # Pyramid level selection.
    read_level: int
    level_downsample: float

    # Region to read at the chosen level (origin in level-0 coords, size at the level).
    read_origin: Tuple[int, int]
    read_size_at_level: Tuple[int, int]

    # Rescaling read-level -> target-mpp.
    rescale_factor_x: float
    rescale_factor_y: float
    rescaled_w: int
    rescaled_h: int
    interpolation: str  # "area" or "linear"

    # Patch grid in the rescaled frame.
    patch_size: int
    stride: int
    grid_w: int
    grid_h: int

    # Valid flat-grid indices.
    bitmap: pyroaring.BitMap

    #: Annotation Name, or None. ``roi_label`` supplies a filename-safe fallback.
    roi_name: Optional[str] = None

    @property
    def num_valid(self) -> int:
        return len(self.bitmap)

    @property
    def num_total(self) -> int:
        return self.grid_w * self.grid_h

    @property
    def valid_fraction(self) -> float:
        n = self.num_total
        return self.num_valid / n if n > 0 else 0.0

    @property
    def roi_label(self) -> str:
        """Human-readable, filename-safe ROI identifier (name or index)."""
        return resolve_roi_id(self.roi_name, self.roi_index)


@dataclass
class SlideIndex:
    """Complete patch index for one slide.

    ``roi_grids`` is target-major: ``roi_grids[t][r]`` is the
    :class:`ROIGrid` for target index ``t`` and ROI index ``r``. The same
    set of ROIs is used for every target; only grid geometry (level,
    scale, patch size, stride) changes between targets.
    """

    slide_id: str
    slide_path: str

    base_mpp_x: float
    base_mpp_y: float
    level_count: int
    level_downsamples: List[float]
    level_dims: List[Tuple[int, int]]

    targets: List[TargetSpec]
    roi_grids: List[List[ROIGrid]]

    roi_source: str = ""

    @property
    def num_targets(self) -> int:
        return len(self.targets)

    @property
    def num_rois(self) -> int:
        return len(self.roi_grids[0]) if self.roi_grids else 0

    def total_valid(self) -> int:
        return sum(g.num_valid for per in self.roi_grids for g in per)

    def valid_per_target(self) -> List[int]:
        return [sum(g.num_valid for g in per) for per in self.roi_grids]

    def grid(self, target_idx: int, roi_idx: int) -> ROIGrid:
        return self.roi_grids[target_idx][roi_idx]

    def target_by_name(self, name: str) -> int:
        """Index of a target by name; raises KeyError if not found."""
        for i, t in enumerate(self.targets):
            if t.name == name:
                return i
        raise KeyError(f"No target named {name!r}; have {[t.name for t in self.targets]}")

    def resolve_target(self, target: Optional[object]) -> int:
        """Resolve ``target`` (None / int / str) to a target index.

        ``None`` is allowed only when the plan has exactly one target.
        """
        if target is None:
            if self.num_targets != 1:
                raise ValueError(
                    f"Plan has {self.num_targets} targets "
                    f"({[t.name for t in self.targets]}); specify one."
                )
            return 0
        if isinstance(target, int):
            if not 0 <= target < self.num_targets:
                raise IndexError(f"target index {target} out of range")
            return target
        if isinstance(target, str):
            return self.target_by_name(target)
        raise TypeError(f"target must be None, int, or str; got {type(target).__name__}")
