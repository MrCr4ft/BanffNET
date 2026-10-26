"""Write slide and ROI scores to CSV, or per-patch scores to HDF5.

``ResultsWriter`` provides a context-managed interface for all output levels.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Sequence, Union

import h5py
import numpy as np

PathLike = Union[str, Path]


class ResultsWriter:
    """Unified writer for slide-level/roi-level CSV and patch-level H5 outputs.

    Parameters
    ----------
    output_path
        For ``level='slide'`` or ``level='roi'``: the path of the CSV to
        write. For ``level='patch'``: the directory under which
        per-slide H5 files are written.
    level
        ``'slide'``, ``'roi'``, or ``'patch'``.
    attribute_names
        Column order for the slide-level/roi-level CSV; ignored at
        patch-level since the order is implicit in the H5 (and recorded
        as an attribute on the dataset).

    Usage::

        with ResultsWriter("out.csv", level="slide", attribute_names=attrs) as w:
            w.add("slide_001", {"i": 0.8, "t": 0.2, ...})

        with ResultsWriter("out.csv", level="roi", attribute_names=attrs) as w:
            w.add("slide_001", {"core_01": {"i": 0.8, ...}, "1": {"i": 0.3, ...}})

        with ResultsWriter("out_dir", level="patch", attribute_names=attrs) as w:
            w.add("slide_001", np.zeros((123, 4)))  # writes out_dir/slide_001.h5
    """

    def __init__(
        self,
        output_path: PathLike,
        *,
        level: str,
        attribute_names: Sequence[str],
    ):
        if level not in ("slide", "patch", "roi"):
            raise ValueError(f"level must be 'slide', 'patch', or 'roi', got {level!r}")
        self.level = level
        self.attribute_names = list(attribute_names)
        self.output_path = Path(output_path)
        self._fp = None
        self._csv = None

    def __enter__(self) -> "ResultsWriter":
        if self.level in ("slide", "roi"):
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self._fp = open(self.output_path, "w", newline="")
            self._csv = csv.writer(self._fp)
            header = ["slide_id"]
            if self.level == "roi":
                header.append("roi_id")
            header += self.attribute_names
            self._csv.writerow(header)
        else:
            self.output_path.mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self) -> None:
        if self._fp is not None:
            self._fp.close()
            self._fp = None
            self._csv = None

    def add(
        self,
        slide_id: str,
        scores: Union[Dict[str, float], np.ndarray, Dict[str, Dict[str, float]]],
    ) -> None:
        """Append one slide's results.

        ``scores`` shape depends on ``level``:

        * slide-level: ``Dict[attribute_name, float]``.
        * roi-level: ``Dict[roi_id, Dict[attribute_name, float]]`` — one
          CSV row is written per ROI, sorted by ``roi_id``. ``roi_id``
          is the ROI's annotation Name when it has one, else its
          positional index as a string.
        * patch-level: ``np.ndarray`` of shape ``[N_patches, N_attributes]``.
        """
        if self.level == "slide":
            if not isinstance(scores, dict):
                raise TypeError(f"slide-level scores must be a dict, got {type(scores).__name__}")
            row = [slide_id] + [round(scores[a], 3) for a in self.attribute_names]
            self._csv.writerow(row)
            self._fp.flush()
        elif self.level == "roi":
            if not isinstance(scores, dict):
                raise TypeError(f"roi-level scores must be a dict, got {type(scores).__name__}")
            for roi_id in sorted(scores):
                roi_scores = scores[roi_id]
                row = [slide_id, roi_id] + [round(roi_scores[a], 3) for a in self.attribute_names]
                self._csv.writerow(row)
            self._fp.flush()
        else:
            if not isinstance(scores, np.ndarray):
                raise TypeError(
                    f"patch-level scores must be an ndarray, got {type(scores).__name__}"
                )
            out = self.output_path / f"{slide_id}.h5"
            with h5py.File(out, "w") as f:
                ds = f.create_dataset(
                    "scores",
                    data=scores.astype(np.float32),
                    compression="gzip",
                    compression_opts=4,
                )
                ds.attrs["attributes"] = np.array(self.attribute_names, dtype="S")
