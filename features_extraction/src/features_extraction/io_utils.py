"""Patch source iterators and streaming feature output.

All iterators yield ``(batch, metadata)`` pairs for ``FeatureExtractor`` or
``FeatureH5Writer``.
"""

import os
import typing
from pathlib import Path

import h5py
import numpy as np

PATCHES_KEY_CANDIDATES: typing.Tuple[str, ...] = ("patches", "imgs", "images", "tiles")


def detect_patches_dataset(h5_path: typing.Union[str, Path]) -> str:
    """Locate the dataset holding the patches in an HDF5 file.

    Tries the names in :data:`PATCHES_KEY_CANDIDATES` first, then falls back to
    the largest 4-D dataset with a trailing dimension of 3.
    """
    with h5py.File(h5_path, "r") as f:
        for key in PATCHES_KEY_CANDIDATES:
            if key in f and isinstance(f[key], h5py.Dataset):
                return key
        best_name: typing.Optional[str] = None
        best_size: int = -1

        def _visit(name: str, obj: h5py.HLObject) -> None:
            nonlocal best_name, best_size
            if isinstance(obj, h5py.Dataset) and obj.ndim == 4 and obj.shape[-1] == 3:
                if obj.size > best_size:
                    best_name, best_size = name, obj.size

        f.visititems(_visit)
        if best_name is None:
            raise KeyError(f"No (N, H, W, 3) patches dataset found in {h5_path}")
        return best_name


def iter_h5_patches(
    h5_path: typing.Union[str, Path],
    batch_size: int,
    dataset_key: typing.Optional[str] = None,
    extra_keys: typing.Optional[typing.Sequence[str]] = None,
) -> typing.Iterator[typing.Tuple[np.ndarray, typing.Dict[str, np.ndarray]]]:
    """Yield ``(patches_batch, metadata)`` chunks from an HDF5 file.

    ``extra_keys`` are sliced in lockstep with the patches dataset and returned
    in the metadata dict. A synthetic ``_index`` key with the global row indices
    is always added.
    """
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as f:
        key = dataset_key or detect_patches_dataset(h5_path)
        ds = f[key]
        n = ds.shape[0]
        extras = {k: f[k] for k in (extra_keys or []) if k in f}
        for i in range(0, n, batch_size):
            j = min(i + batch_size, n)
            batch = ds[i:j]
            meta = {k: extras[k][i:j] for k in extras}
            meta["_index"] = np.arange(i, j, dtype=np.int64)
            yield batch, meta


def iter_folder_patches(
    folder: typing.Union[str, Path],
    batch_size: int,
    extensions: typing.Sequence[str] = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"),
) -> typing.Iterator[typing.Tuple[np.ndarray, typing.Dict[str, typing.Any]]]:
    """Yield ``(images_batch, metadata)`` from a folder of image files.

    PIL is used purely for decoding (not for transforms); images become uint8
    RGB ``np.ndarray`` immediately. If a chunk contains images of mismatching
    sizes the chunk is broken up into singletons so :class:`BatchPreprocess`
    can resize each one to the target size independently.
    """
    from PIL import Image  # local import so the rest of the package has no hard PIL dep

    folder = Path(folder)
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in extensions)
    if not files:
        raise FileNotFoundError(f"No image files in {folder} with extensions {extensions}")

    def _load(path: Path) -> np.ndarray:
        with Image.open(path) as im:
            return np.asarray(im.convert("RGB"), dtype=np.uint8)

    for i in range(0, len(files), batch_size):
        chunk = files[i : i + batch_size]
        imgs = [_load(p) for p in chunk]
        shapes = {img.shape for img in imgs}
        if len(shapes) == 1:
            batch = np.stack(imgs, axis=0)
            meta = {
                "paths": np.array([str(p) for p in chunk], dtype=object),
                "_index": np.arange(i, i + len(chunk), dtype=np.int64),
            }
            yield batch, meta
        else:
            # Heterogeneous: pass one at a time; preprocess will resize them.
            for k, (img, p) in enumerate(zip(imgs, chunk)):
                yield (
                    img[None, ...],
                    {
                        "paths": np.array([str(p)], dtype=object),
                        "_index": np.array([i + k], dtype=np.int64),
                    },
                )


def iter_plan_patches(
    plan_path: typing.Union[str, Path],
    wsi_path: typing.Optional[typing.Union[str, Path]] = None,
    batch_size: int = 64,
    target_name: typing.Optional[str] = None,
    low_memory: bool = False,
) -> typing.Iterator[typing.Tuple[np.ndarray, typing.Dict[str, np.ndarray]]]:
    """Yield ``(patches_batch, metadata)`` from a wsipatch patching plan.

    Thin wrapper around :class:`wsipatch.PatchReader`. All the heavy
    lifting — target resolution, slide-handle lifecycle, the two read
    strategies, batch buffering — lives in wsipatch. This function exists
    only to give the feature-extraction pipeline a uniform-shaped
    ``(batch, metadata)`` iterator next to ``iter_h5_patches`` and
    ``iter_folder_patches``.

    Parameters
    ----------
    plan_path
        Path to the ``.wsix`` / ``.json`` patching plan (or a
        :class:`wsipatch.SlideIndex`).
    wsi_path
        Optional override for the slide path stored in the plan.
    batch_size
        Patches per emitted batch.
    target_name
        Target to extract. Required when the plan has multiple targets.
    low_memory
        Per-patch ``read_region`` (bounded RAM) instead of whole-ROI reads.

    Yields
    ------
    (batch, metadata)
        ``batch`` is ``(B, ps, ps, 3) uint8``; ``metadata`` has keys
        ``coords`` (B, 2) int32 in the rescaled-ROI frame, ``roi_index``
        (B,) int32 (positional), ``roi_id`` (B,) str (the ROI's
        annotation Name, or ``str(roi_index)`` if it has none), and
        ``_index`` (B,) int64 (global patch index).
    """
    try:
        from wsipatch import PatchReader  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError("iter_plan_patches requires the wsipatch package on PYTHONPATH.") from e

    with PatchReader(
        plan_path,
        slide_path=wsi_path,
        target=target_name,
        low_memory=low_memory,
    ) as reader:
        yield from reader.iter_batches(batch_size=batch_size)


class FeatureH5Writer:
    """Streaming writer for features and per-patch metadata.

    Datasets are created on the first ``append`` and resized incrementally.
    Strings in metadata are stored as variable-length UTF-8.

    Usage::

        with FeatureH5Writer("out.h5") as w:
            w.set_attrs(model="UNI", reinhard=True)
            for batch, meta in source:
                w.append(extractor.extract_batch(batch), meta)
    """

    def __init__(
        self,
        path: typing.Union[str, os.PathLike],
        compression: typing.Optional[str] = "gzip",
        compression_opts: typing.Optional[int] = 4,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(self.path, "w")
        self._features_ds: typing.Optional[h5py.Dataset] = None
        self._meta_dsets: typing.Dict[str, h5py.Dataset] = {}
        self._compression = compression
        self._compression_opts = compression_opts
        self.n = 0

    def __enter__(self) -> "FeatureH5Writer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None

    def set_attrs(self, **attrs: typing.Any) -> None:
        for k, v in attrs.items():
            self._file.attrs[k] = v

    def _ensure_datasets(
        self, features: np.ndarray, metadata: typing.Mapping[str, np.ndarray]
    ) -> None:
        if self._features_ds is None:
            d = int(features.shape[1])
            chunk_rows = min(256, max(1, int(features.shape[0])))
            self._features_ds = self._file.create_dataset(
                "features",
                shape=(0, d),
                maxshape=(None, d),
                dtype=np.float32,
                chunks=(chunk_rows, d),
                compression=self._compression,
                compression_opts=self._compression_opts,
            )
            if metadata:
                grp = self._file.create_group("metadata")
                for k, v in metadata.items():
                    arr = np.asarray(v)
                    trailing = arr.shape[1:] if arr.ndim > 1 else ()
                    if arr.dtype.kind in ("U", "O"):
                        dtype = h5py.string_dtype(encoding="utf-8")
                    else:
                        dtype = arr.dtype
                    ds = grp.create_dataset(
                        k,
                        shape=(0,) + trailing,
                        maxshape=(None,) + trailing,
                        dtype=dtype,
                        compression=self._compression,
                        compression_opts=self._compression_opts,
                    )
                    self._meta_dsets[k] = ds

    def append(
        self,
        features: np.ndarray,
        metadata: typing.Optional[typing.Mapping[str, np.ndarray]] = None,
    ) -> None:
        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2:
            raise ValueError(f"features must be 2D (B, D); got {features.shape}")
        metadata = metadata or {}
        self._ensure_datasets(features, metadata)

        b = features.shape[0]
        self._features_ds.resize((self.n + b, features.shape[1]))
        self._features_ds[self.n : self.n + b] = features

        for k, ds in self._meta_dsets.items():
            if k not in metadata:
                continue
            arr = np.asarray(metadata[k])
            if arr.dtype.kind in ("U", "O"):
                arr = arr.astype(object)
            new_shape = (self.n + b,) + ds.shape[1:]
            ds.resize(new_shape)
            ds[self.n : self.n + b] = arr

        self.n += b


def write_features_h5(
    out_path: typing.Union[str, Path],
    features: np.ndarray,
    metadata: typing.Optional[typing.Mapping[str, np.ndarray]] = None,
    attrs: typing.Optional[typing.Mapping[str, typing.Any]] = None,
    compression: typing.Optional[str] = "gzip",
    compression_opts: typing.Optional[int] = 4,
) -> None:
    """One-shot HDF5 writer for the case where everything fits in memory."""
    with FeatureH5Writer(out_path, compression=compression, compression_opts=compression_opts) as w:
        if attrs:
            w.set_attrs(**attrs)
        if len(features) > 0:
            w.append(features, metadata)
