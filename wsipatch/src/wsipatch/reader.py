"""Read patches from a ``SlideIndex``.

``PatchLocation`` describes read geometry, ``PatchSampler`` resolves locations,
and ``PatchReader`` owns the OpenSlide handle. Whole-ROI reads favor throughput;
per-patch reads bound memory use. HDF5 materialization streams in either mode.
"""

from __future__ import annotations

import math
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Union

import numpy as np

from .io import load_index
from .types import ROIGrid, SlideIndex, TargetSpec


@dataclass
class PatchLocation:
    """Everything needed to extract one patch from a WSI.

    Two read strategies are supported by consumers; this object carries
    parameters for both.

    Strategy A — read whole ROI then slice (best when many patches will
    be drawn from the same ROI in sequence). Uses ``roi_origin``,
    ``read_level``, ``read_size`` to read the ROI, then slices at
    ``(patch_x, patch_y)`` in the rescaled frame.

    Strategy B — read just the patch footprint (best for random
    sampling). Uses ``patch_origin_level0``, ``read_level``, and
    ``patch_read_size`` directly.
    """

    # Strategy A — ROI-level read
    roi_origin: Tuple[int, int]
    read_level: int
    read_size: Tuple[int, int]
    rescaled_w: int
    rescaled_h: int
    rescale_factor_x: float
    rescale_factor_y: float
    interpolation: str

    # Patch position in the rescaled-ROI frame
    patch_x: int
    patch_y: int
    patch_w: int
    patch_h: int

    # Strategy B — patch-only read
    patch_origin_level0: Tuple[int, int]
    patch_read_size: Tuple[int, int]

    # Metadata
    target_mpp: float
    target_name: str
    roi_index: int
    grid_row: int
    grid_col: int
    #: Human-readable ROI identifier (annotation Name, or the positional
    #: index as a string when the annotation had no Name).
    roi_id: str = ""


def _make_location(spec: TargetSpec, g: ROIGrid, row: int, col: int) -> PatchLocation:
    """Build a :class:`PatchLocation` from explicit (row, col)."""
    px, py = col * g.stride, row * g.stride
    rx0, ry0 = g.read_origin
    ds = g.level_downsample
    patch_x0_lv0 = int(round(rx0 + px / g.rescale_factor_x * ds))
    patch_y0_lv0 = int(round(ry0 + py / g.rescale_factor_y * ds))
    patch_wL = max(1, int(round(g.patch_size / g.rescale_factor_x)))
    patch_hL = max(1, int(round(g.patch_size / g.rescale_factor_y)))
    return PatchLocation(
        roi_origin=g.read_origin,
        read_level=g.read_level,
        read_size=g.read_size_at_level,
        rescaled_w=g.rescaled_w,
        rescaled_h=g.rescaled_h,
        rescale_factor_x=g.rescale_factor_x,
        rescale_factor_y=g.rescale_factor_y,
        interpolation=g.interpolation,
        patch_x=px,
        patch_y=py,
        patch_w=spec.patch_size,
        patch_h=spec.patch_size,
        patch_origin_level0=(patch_x0_lv0, patch_y0_lv0),
        patch_read_size=(patch_wL, patch_hL),
        target_mpp=spec.mpp,
        target_name=spec.name,
        roi_index=g.roi_index,
        roi_id=g.roi_label,
        grid_row=row,
        grid_col=col,
    )


class PatchSampler:
    """Random and deterministic location access for a :class:`SlideIndex`.

    O(num_targets × num_rois) construction; each lookup is
    O(log num_rois) plus a Roaring ``select``. Holds no slide handle —
    callers are responsible for reading pixels. Use :class:`PatchReader`
    if you want pixels too.
    """

    def __init__(self, index: SlideIndex):
        self.index = index
        self._target_cum: List[List[int]] = []
        self._target_totals: List[int] = []
        for per_target in index.roi_grids:
            cum, running = [], 0
            for g in per_target:
                running += g.num_valid
                cum.append(running)
            self._target_cum.append(cum)
            self._target_totals.append(running)

    @property
    def slide_id(self) -> str:
        return self.index.slide_id

    @property
    def slide_path(self) -> str:
        return self.index.slide_path

    @property
    def num_targets(self) -> int:
        return self.index.num_targets

    def num_valid(self, target_idx: int) -> int:
        return self._target_totals[target_idx]

    def total_valid(self) -> int:
        return sum(self._target_totals)

    def sample(self, target_idx: int, rng=None) -> PatchLocation:
        """Uniformly sample one valid patch from a target."""
        total = self._target_totals[target_idx]
        if total == 0:
            raise RuntimeError(f"Target {target_idx} has 0 valid patches")
        if rng is None:
            import random

            k = random.randint(0, total - 1)
        elif isinstance(rng, np.random.Generator):
            k = int(rng.integers(0, total))
        else:
            k = rng.randint(0, total - 1)
        return self._resolve(target_idx, k)

    def get(self, target_idx: int, k: int) -> PatchLocation:
        """The k-th valid patch (across all ROIs, in plan order)."""
        total = self._target_totals[target_idx]
        if not 0 <= k < total:
            raise IndexError(f"k={k} out of range [0, {total})")
        return self._resolve(target_idx, k)

    def iter_locations(
        self,
        target_idx: int,
        roi_idx: Optional[int] = None,
    ) -> Iterator[PatchLocation]:
        """Yield every valid :class:`PatchLocation` in plan order."""
        spec = self.index.targets[target_idx]
        grids = self.index.roi_grids[target_idx]
        roi_range = [roi_idx] if roi_idx is not None else range(len(grids))
        for ri in roi_range:
            g = grids[ri]
            for flat in g.bitmap:
                row, col = divmod(flat, g.grid_w)
                yield _make_location(spec, g, row, col)

    def _resolve(self, target_idx: int, k: int) -> PatchLocation:
        cum = self._target_cum[target_idx]
        grids = self.index.roi_grids[target_idx]
        spec = self.index.targets[target_idx]
        # Binary search for the ROI containing the k-th valid patch.
        lo, hi = 0, len(cum) - 1
        while lo < hi:
            mid = (lo + hi) >> 1
            if cum[mid] <= k:
                lo = mid + 1
            else:
                hi = mid
        roi_idx = lo
        local_k = k if roi_idx == 0 else k - cum[roi_idx - 1]
        g = grids[roi_idx]
        flat = g.bitmap[local_k]
        row, col = divmod(flat, g.grid_w)
        return _make_location(spec, g, row, col)


PlanLike = Union[SlideIndex, str, Path]


class PatchReader:
    """Read patches from a :class:`SlideIndex` plan.

    Two extraction strategies, selected at construction by
    ``low_memory``:

    * ``low_memory=False`` (default) — for each ROI, read it whole at
      ``read_level``, rescale once to ``(rescaled_w, rescaled_h)``,
      then slice patches. Fastest, but the rescaled ROI is held in RAM.
    * ``low_memory=True`` — one ``read_region`` per patch, then resize
      to ``(patch_size, patch_size)`` if needed. Peak RAM bounded by the
      batch buffer; slower because of many small reads.

    Use as a context manager (recommended) so the slide handle is
    released deterministically::

        with PatchReader(plan, slide_path=wsi, target="default") as r:
            for batch, meta in r.iter_batches(batch_size=64):
                feats = model(preprocess(batch))

    Parameters
    ----------
    plan
        Either a :class:`SlideIndex` or a path to a saved plan
        (``.wsix`` / ``.json``).
    slide_path
        Override the slide path stored in the plan (useful when files
        have moved). If omitted, ``plan.slide_path`` is used.
    target
        Target to read: ``None`` (only when the plan has exactly one
        target), an ``int`` index, or a ``str`` name.
    low_memory
        Which read strategy to use (see above).
    """

    def __init__(
        self,
        plan: PlanLike,
        *,
        slide_path: Optional[Union[str, Path]] = None,
        target: Optional[Union[int, str]] = None,
        low_memory: bool = False,
    ):
        self.index: SlideIndex = plan if isinstance(plan, SlideIndex) else load_index(plan)
        self.target_idx: int = self.index.resolve_target(target)
        self.spec: TargetSpec = self.index.targets[self.target_idx]
        self.low_memory: bool = bool(low_memory)

        sp = str(slide_path) if slide_path is not None else self.index.slide_path
        if not Path(sp).exists():
            raise FileNotFoundError(
                f"Slide not found: {sp}. Pass slide_path=... to override the stored path."
            )
        self.slide_path: str = sp
        self.sampler = PatchSampler(self.index)
        self._slide = None  # opened in __enter__/open()

    def open(self) -> "PatchReader":
        """Open the slide handle. Idempotent. Prefer the ``with`` form."""
        if self._slide is None:
            import openslide  # local import to keep import-time light

            self._slide = openslide.OpenSlide(self.slide_path)
        return self

    def close(self) -> None:
        """Close the slide handle. Safe to call multiple times."""
        if self._slide is not None:
            self._slide.close()
            self._slide = None

    def __enter__(self) -> "PatchReader":
        return self.open()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @property
    def num_patches(self) -> int:
        """Total valid patches for the chosen target."""
        return self.sampler.num_valid(self.target_idx)

    @property
    def patch_size(self) -> int:
        return self.spec.patch_size

    @property
    def target_name(self) -> str:
        return self.spec.name

    @property
    def num_rois(self) -> int:
        return len(self.index.roi_grids[self.target_idx])

    def _require_open(self):
        if self._slide is None:
            raise RuntimeError(
                "PatchReader's slide handle is not open. "
                "Use it as a context manager or call .open() first."
            )
        return self._slide

    @staticmethod
    def _cv_interp(name: str) -> int:
        import cv2

        return cv2.INTER_AREA if name == "area" else cv2.INTER_LINEAR

    def _read_patch_per_patch(self, slide, g: ROIGrid, row: int, col: int) -> np.ndarray:
        """Low-memory strategy: read just the patch's level-0 footprint."""
        import cv2

        ps = self.spec.patch_size
        x_rs, y_rs = col * g.stride, row * g.stride
        rx0, ry0 = g.read_origin
        ds = g.level_downsample
        patch_x0_lv0 = int(round(rx0 + x_rs / g.rescale_factor_x * ds))
        patch_y0_lv0 = int(round(ry0 + y_rs / g.rescale_factor_y * ds))
        patch_wL = max(1, int(round(ps / g.rescale_factor_x)))
        patch_hL = max(1, int(round(ps / g.rescale_factor_y)))
        region = slide.read_region(
            (patch_x0_lv0, patch_y0_lv0),
            g.read_level,
            (patch_wL, patch_hL),
        )
        patch = np.asarray(region)[..., :3]
        if patch.shape[:2] != (ps, ps):
            patch = cv2.resize(patch, (ps, ps), interpolation=self._cv_interp(g.interpolation))
        return patch

    def _read_whole_roi(self, slide, g: ROIGrid) -> np.ndarray:
        """Standard strategy: read the ROI once and rescale once."""
        import cv2

        region = slide.read_region(g.read_origin, g.read_level, g.read_size_at_level)
        roi = np.asarray(region)[..., :3]
        if g.rescale_factor_x != 1.0 or g.rescale_factor_y != 1.0:
            roi = cv2.resize(
                roi,
                (g.rescaled_w, g.rescaled_h),
                interpolation=self._cv_interp(g.interpolation),
            )
        return roi

    def iter_patches(self) -> Iterator[Tuple[np.ndarray, dict]]:
        """Yield ``(patch_hwc_uint8, meta)`` in plan order.

        Patches are owned arrays (safe to keep across iterations). The
        meta dict has ``coords`` (rescaled-ROI ``(x, y)``), ``roi_index``
        (positional), ``roi_id`` (name, or the positional index as a
        string when the ROI has no name), ``grid_row``, ``grid_col``, and
        ``index`` (global).
        """
        slide = self._require_open()
        ps = self.spec.patch_size
        global_idx = 0
        for r_idx, g in enumerate(self.index.roi_grids[self.target_idx]):
            if g.num_valid == 0:
                continue
            roi_id = g.roi_label
            if self.low_memory:
                for flat in g.bitmap:
                    row, col = divmod(flat, g.grid_w)
                    patch = self._read_patch_per_patch(slide, g, row, col)
                    yield (
                        patch,
                        {
                            "coords": (col * g.stride, row * g.stride),
                            "roi_index": r_idx,
                            "roi_id": roi_id,
                            "grid_row": row,
                            "grid_col": col,
                            "index": global_idx,
                        },
                    )
                    global_idx += 1
            else:
                roi_scaled = self._read_whole_roi(slide, g)
                for flat in g.bitmap:
                    row, col = divmod(flat, g.grid_w)
                    x, y = col * g.stride, row * g.stride
                    # Copy: roi_scaled goes out of scope on the next ROI.
                    patch = roi_scaled[y : y + ps, x : x + ps].copy()
                    yield (
                        patch,
                        {
                            "coords": (x, y),
                            "roi_index": r_idx,
                            "roi_id": roi_id,
                            "grid_row": row,
                            "grid_col": col,
                            "index": global_idx,
                        },
                    )
                    global_idx += 1
                del roi_scaled

    def iter_batches(
        self,
        batch_size: int = 64,
    ) -> Iterator[Tuple[np.ndarray, typing.Dict[str, np.ndarray]]]:
        """Yield ``(batch_hwc_uint8, meta_dict)`` batches.

        Batch shape is ``(B, patch_size, patch_size, 3)`` ``uint8``,
        where ``B <= batch_size``; the last batch may be partial. Meta
        keys: ``coords`` (B, 2) int32, ``roi_index`` (B,) int32
        (positional), ``roi_id`` (B,) str (name, or the positional index
        as a string when the ROI has no name), ``_index`` (B,) int64.

        Uses pre-allocated buffers that are copied on emit, so the
        consumer can safely retain a batch across iterations.
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        slide = self._require_open()
        ps = self.spec.patch_size

        buf_imgs = np.empty((batch_size, ps, ps, 3), dtype=np.uint8)
        buf_coords = np.empty((batch_size, 2), dtype=np.int32)
        buf_roi = np.empty((batch_size,), dtype=np.int32)
        buf_roi_id = np.empty((batch_size,), dtype=object)
        buf_gidx = np.empty((batch_size,), dtype=np.int64)
        fill = 0
        global_idx = 0

        def _emit():
            return (
                buf_imgs[:fill].copy(),
                {
                    "coords": buf_coords[:fill].copy(),
                    "roi_index": buf_roi[:fill].copy(),
                    "roi_id": buf_roi_id[:fill].copy(),
                    "_index": buf_gidx[:fill].copy(),
                },
            )

        for r_idx, g in enumerate(self.index.roi_grids[self.target_idx]):
            if g.num_valid == 0:
                continue
            roi_id = g.roi_label
            if self.low_memory:
                for flat in g.bitmap:
                    row, col = divmod(flat, g.grid_w)
                    buf_imgs[fill] = self._read_patch_per_patch(slide, g, row, col)
                    buf_coords[fill] = (col * g.stride, row * g.stride)
                    buf_roi[fill] = r_idx
                    buf_roi_id[fill] = roi_id
                    buf_gidx[fill] = global_idx
                    fill += 1
                    global_idx += 1
                    if fill == batch_size:
                        yield _emit()
                        fill = 0
            else:
                roi_scaled = self._read_whole_roi(slide, g)
                for flat in g.bitmap:
                    row, col = divmod(flat, g.grid_w)
                    x, y = col * g.stride, row * g.stride
                    buf_imgs[fill] = roi_scaled[y : y + ps, x : x + ps]
                    buf_coords[fill] = (x, y)
                    buf_roi[fill] = r_idx
                    buf_roi_id[fill] = roi_id
                    buf_gidx[fill] = global_idx
                    fill += 1
                    global_idx += 1
                    if fill == batch_size:
                        yield _emit()
                        fill = 0
                del roi_scaled

        if fill > 0:
            yield _emit()

    def read_patch(self, k: int) -> Tuple[np.ndarray, PatchLocation]:
        """Return the k-th valid patch (global order) plus its location.

        Always uses the per-patch read strategy regardless of
        ``low_memory`` — random access from a fresh ROI would otherwise
        require reading the whole ROI for a single patch.
        """
        slide = self._require_open()
        loc = self.sampler.get(self.target_idx, k)
        g = self.index.roi_grids[self.target_idx][loc.roi_index]
        patch = self._read_patch_per_patch(slide, g, loc.grid_row, loc.grid_col)
        return patch, loc

    def materialise(
        self,
        out_path: Union[str, Path],
        *,
        batch_size: int = 64,
        compression: Optional[str] = "gzip",
        compression_opts: Optional[int] = 4,
        chunks: Optional[int] = None,
        progress: bool = False,
    ) -> Path:
        """Stream all patches to a single HDF5 file.

        Output layout::

            imgs    (N, ps, ps, 3)  uint8   compressed
            coords  (N, 2)          int32   compressed   attrs:
                                                          frame="rescaled_roi"
                                                          roi_offsets=(num_rois+1,)

        ``imgs`` also carries ``slide_id``, ``slide_path``,
        ``target_name``, ``patch_size``, ``mpp``, ``stride`` attrs.

        Peak memory is bounded by the batch buffer regardless of patch count.

        Parameters
        ----------
        out_path
            Destination ``.h5`` path.
        batch_size
            Internal batch size for streaming reads / writes.
        compression, compression_opts
            ``h5py.create_dataset`` knobs. Use ``compression=None`` to
            disable.
        chunks
            HDF5 chunk size along the first axis. Defaults to
            ``min(64, 2**floor(log2(N)))`` for nonempty output.
        progress
            Show a tqdm bar (if installed).
        """
        import h5py

        n_total = self.num_patches
        if n_total == 0:
            raise ValueError(
                f"Plan has 0 valid patches for target {self.spec.name!r}; nothing to write."
            )

        if chunks is None:
            chunks_dim = min(64, int(2 ** math.floor(math.log2(n_total))))
        else:
            chunks_dim = max(1, min(chunks, n_total))

        ps = self.spec.patch_size
        out_p = Path(out_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        slide_was_open = self._slide is not None
        if not slide_was_open:
            self.open()

        try:
            with h5py.File(out_p, "w") as h5:
                dset_imgs = h5.create_dataset(
                    "imgs",
                    shape=(n_total, ps, ps, 3),
                    dtype=np.uint8,
                    compression=compression,
                    compression_opts=compression_opts,
                    chunks=(chunks_dim, ps, ps, 3),
                )
                dset_coords = h5.create_dataset(
                    "coords",
                    shape=(n_total, 2),
                    dtype=np.int32,
                    compression=compression,
                    compression_opts=compression_opts,
                    chunks=(chunks_dim, 2),
                )

                roi_offsets: List[int] = [0]
                current_roi = 0
                write_pos = 0

                iterator = self.iter_batches(batch_size=batch_size)
                if progress:
                    try:
                        from tqdm.auto import tqdm

                        iterator = tqdm(
                            iterator,
                            total=math.ceil(n_total / batch_size),
                            desc="Materialising",
                            unit="batch",
                        )
                    except ImportError:
                        pass

                for batch, meta in iterator:
                    b = batch.shape[0]
                    dset_imgs[write_pos : write_pos + b] = batch
                    dset_coords[write_pos : write_pos + b] = meta["coords"]

                    # Track ROI boundaries from roi_index. Patches arrive
                    # in target-major, roi-minor order so ROI changes are
                    # monotone within a batch.
                    rois_in_batch = meta["roi_index"]
                    for i, r in enumerate(rois_in_batch):
                        if int(r) != current_roi:
                            # Skip any empty ROIs between current and r.
                            while current_roi < int(r):
                                roi_offsets.append(write_pos + i)
                                current_roi += 1
                    write_pos += b

                # Close any ROIs we haven't yet appended an offset for.
                while len(roi_offsets) <= self.num_rois:
                    roi_offsets.append(write_pos)

                assert write_pos == n_total, f"wrote {write_pos}, expected {n_total}"

                dset_imgs.attrs["slide_id"] = self.index.slide_id
                dset_imgs.attrs["slide_path"] = self.slide_path
                dset_imgs.attrs["target_name"] = self.spec.name
                dset_imgs.attrs["patch_size"] = self.spec.patch_size
                dset_imgs.attrs["mpp"] = self.spec.mpp
                dset_imgs.attrs["stride"] = self.spec.stride
                # roi_ids[i] is the label of the ROI whose patches occupy
                # coords[roi_offsets[i]:roi_offsets[i+1]] — one entry per ROI,
                # in the same order as roi_offsets.
                dset_imgs.attrs["roi_ids"] = np.array(
                    [g.roi_label for g in self.index.roi_grids[self.target_idx]],
                    dtype=object,
                )
                dset_coords.attrs["frame"] = "rescaled_roi"
                dset_coords.attrs["roi_offsets"] = np.asarray(roi_offsets, dtype=np.int64)
        finally:
            if not slide_was_open:
                self.close()

        return out_p
