"""Persistence for :class:`SlideIndex`.

Two formats, one set of entry points:

* **wsix** — compact binary. A small JSON header (all spatial /
  geometric metadata) followed by length-prefixed Roaring bitmap blobs
  for every (target, ROI) pair. Use this in production.
* **json** — human-readable equivalent. Stores the same metadata as
  wsix, plus the bitmap content expanded as a sorted ``valid_indices``
  list. Optionally adds explicit ``valid_coords`` for grep-ability.

Use :func:`save_index` and :func:`load_index` — they auto-detect the
format from the file extension (``.wsix`` → binary, ``.json`` → json,
anything else defaults to wsix). The format-specific functions
(:func:`save_wsix` / :func:`load_wsix` / :func:`save_json` /
:func:`load_json`) are exposed too for callers that want to force one.

wsix wire layout::

    MAGIC      b"WSIX"               (4 bytes)
    JSON_LEN   uint32 little-endian  (4 bytes)
    JSON_BLOB  utf-8 metadata        (JSON_LEN bytes)
    for each grid in target-major, roi-minor order:
        BM_LEN  uint32 little-endian (4 bytes)
        BITMAP  pyroaring serialised (BM_LEN bytes)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Literal, Optional, Union

import pyroaring

from .types import ROIGrid, SlideIndex, TargetSpec

FormatName = Literal["wsix", "json"]
MAGIC = b"WSIX"


def _grid_meta_dict(g: ROIGrid) -> dict:
    """Spatial / geometric fields of a grid (no bitmap content)."""
    return {
        "roi_index": g.roi_index,
        "roi_name": g.roi_name,
        "read_level": g.read_level,
        "level_downsample": g.level_downsample,
        "read_origin": list(g.read_origin),
        "read_size_at_level": list(g.read_size_at_level),
        "rescale_factor_x": g.rescale_factor_x,
        "rescale_factor_y": g.rescale_factor_y,
        "rescaled_w": g.rescaled_w,
        "rescaled_h": g.rescaled_h,
        "interpolation": g.interpolation,
        "patch_size": g.patch_size,
        "stride": g.stride,
        "grid_w": g.grid_w,
        "grid_h": g.grid_h,
        "num_valid": g.num_valid,
    }


def _grid_from_meta(gd: dict, bitmap: pyroaring.BitMap) -> ROIGrid:
    return ROIGrid(
        roi_index=gd["roi_index"],
        roi_name=gd.get("roi_name"),  # .get(): absent in plans saved before this field existed
        read_level=gd["read_level"],
        level_downsample=gd["level_downsample"],
        read_origin=tuple(gd["read_origin"]),
        read_size_at_level=tuple(gd["read_size_at_level"]),
        rescale_factor_x=gd["rescale_factor_x"],
        rescale_factor_y=gd["rescale_factor_y"],
        rescaled_w=gd["rescaled_w"],
        rescaled_h=gd["rescaled_h"],
        interpolation=gd["interpolation"],
        patch_size=gd["patch_size"],
        stride=gd["stride"],
        grid_w=gd["grid_w"],
        grid_h=gd["grid_h"],
        bitmap=bitmap,
    )


def _slide_header(idx: SlideIndex) -> dict:
    """Slide-level metadata shared by both formats."""
    return {
        "slide_id": idx.slide_id,
        "slide_path": idx.slide_path,
        "roi_source": idx.roi_source,
        "base_mpp_x": idx.base_mpp_x,
        "base_mpp_y": idx.base_mpp_y,
        "level_count": idx.level_count,
        "level_downsamples": list(idx.level_downsamples),
        "level_dims": [list(d) for d in idx.level_dims],
        "targets": [
            {"name": t.name, "patch_size": t.patch_size, "mpp": t.mpp, "stride": t.stride}
            for t in idx.targets
        ],
    }


def _slide_from_header(header: dict, roi_grids: List[List[ROIGrid]]) -> SlideIndex:
    return SlideIndex(
        slide_id=header["slide_id"],
        slide_path=header["slide_path"],
        base_mpp_x=header["base_mpp_x"],
        base_mpp_y=header["base_mpp_y"],
        level_count=header["level_count"],
        level_downsamples=header["level_downsamples"],
        level_dims=[tuple(d) for d in header["level_dims"]],
        targets=[TargetSpec(**td) for td in header["targets"]],
        roi_grids=roi_grids,
        roi_source=header.get("roi_source", ""),
    )


def save_wsix(idx: SlideIndex, path: Union[str, Path]) -> Path:
    """Write a :class:`SlideIndex` as a single ``.wsix`` file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    meta = {
        "format": "wsix",
        **_slide_header(idx),
        "roi_grids": [[_grid_meta_dict(g) for g in per] for per in idx.roi_grids],
    }
    json_blob = json.dumps(meta, separators=(",", ":")).encode("utf-8")
    bitmap_blobs = [g.bitmap.serialize() for per in idx.roi_grids for g in per]

    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(len(json_blob).to_bytes(4, "little"))
        f.write(json_blob)
        for blob in bitmap_blobs:
            f.write(len(blob).to_bytes(4, "little"))
            f.write(blob)
    return path


def load_wsix(path: Union[str, Path]) -> SlideIndex:
    """Load a :class:`SlideIndex` from a ``.wsix`` file."""
    path = Path(path)
    with open(path, "rb") as f:
        if f.read(4) != MAGIC:
            raise ValueError(f"Not a WSIX file (bad magic): {path}")
        json_len = int.from_bytes(f.read(4), "little")
        meta = json.loads(f.read(json_len).decode("utf-8"))

        roi_grids: List[List[ROIGrid]] = []
        for per_target_meta in meta["roi_grids"]:
            grids: List[ROIGrid] = []
            for gd in per_target_meta:
                bm_len = int.from_bytes(f.read(4), "little")
                bm = pyroaring.BitMap.deserialize(f.read(bm_len))
                grids.append(_grid_from_meta(gd, bm))
            roi_grids.append(grids)

    return _slide_from_header(meta, roi_grids)


JSON_VERSION = 1


def save_json(
    idx: SlideIndex,
    path: Union[str, Path],
    *,
    include_coords: bool = False,
    indent: int = 2,
) -> Path:
    """Write a :class:`SlideIndex` as JSON.

    If ``include_coords`` is True, each grid also stores ``valid_coords``
    — the (x, y) rescaled-frame coordinates of every valid cell. These
    are redundant (derivable from ``valid_indices`` + ``grid_w`` +
    ``stride``) but useful for grepping or eyeballing.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _grid_dict(g: ROIGrid) -> dict:
        d = _grid_meta_dict(g)
        valid = list(g.bitmap)
        d["valid_indices"] = valid
        if include_coords:
            gw, s = g.grid_w, g.stride
            d["valid_coords"] = [[(k % gw) * s, (k // gw) * s] for k in valid]
        return d

    out = {
        "format": "wsipatch-json",
        "format_version": JSON_VERSION,
        **_slide_header(idx),
        "roi_grids": [[_grid_dict(g) for g in per] for per in idx.roi_grids],
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=indent)
    return path


def load_json(path: Union[str, Path]) -> SlideIndex:
    """Load a :class:`SlideIndex` from a JSON-format plan."""
    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    if data.get("format") not in ("wsipatch-json", None):
        raise ValueError(f"Unrecognised format: {data.get('format')!r}")
    version = int(data.get("format_version", 1))
    if version > JSON_VERSION:
        raise ValueError(
            f"JSON format version {version} is newer than supported ({JSON_VERSION}); "
            f"upgrade wsipatch."
        )

    roi_grids: List[List[ROIGrid]] = []
    for per_target in data["roi_grids"]:
        grids: List[ROIGrid] = []
        for gd in per_target:
            bm = pyroaring.BitMap(gd["valid_indices"])
            bm.run_optimize()
            grids.append(_grid_from_meta(gd, bm))
        roi_grids.append(grids)

    return _slide_from_header(data, roi_grids)


def _infer_format(path: Path, default: FormatName = "wsix") -> FormatName:
    suffix = path.suffix.lower().lstrip(".")
    if suffix == "wsix":
        return "wsix"
    if suffix == "json":
        return "json"
    return default


def save_index(
    idx: SlideIndex,
    path: Union[str, Path],
    *,
    format: Optional[FormatName] = None,
    include_coords: bool = False,
) -> Path:
    """Write a :class:`SlideIndex`. Format inferred from extension if omitted.

    ``include_coords`` is honoured only for JSON.
    """
    path = Path(path)
    fmt = format or _infer_format(path)
    if fmt == "wsix":
        return save_wsix(idx, path)
    if fmt == "json":
        return save_json(idx, path, include_coords=include_coords)
    raise ValueError(f"Unknown format: {fmt!r}")


def load_index(path: Union[str, Path]) -> SlideIndex:
    """Load a :class:`SlideIndex` from disk; format detected by extension."""
    path = Path(path)
    fmt = _infer_format(path)
    return load_wsix(path) if fmt == "wsix" else load_json(path)
