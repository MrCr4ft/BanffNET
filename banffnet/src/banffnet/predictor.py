"""Multi-attribute BANFF prediction from patch features.

YAML configuration format::

    attribute_name:
      model_type: linear_noisy_or
      model_params: {input_features_size: 1024}
      checkpoint: /path/to/state_dict.pt

``linear_gated_ratio`` is also supported. Predictions can be pooled by slide,
grouped by ROI, or returned per patch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import h5py
import numpy as np
import torch
import torch.nn as nn
import yaml

from .models import MODELS

PathLike = Union[str, Path]


class BanffPredictor(nn.Module):
    """Bundle of per-attribute MIL heads with both bag-level and patch-level inference.

    Parameters
    ----------
    banff_attributes
        Mapping ``{attribute_name: {model_type, model_params, checkpoint}}``.
        Matches the shape produced by ``yaml.safe_load`` on the existing
        per-staining config files.
    device
        Where to place the loaded heads. ``None`` keeps them on CPU.
    load_checkpoints
        Whether to load trained weights. Set to ``False`` for fresh heads.
    strict
        Passed through to :meth:`nn.Module.load_state_dict`.

    """

    def __init__(
        self,
        banff_attributes: Dict[str, Dict[str, Any]],
        *,
        device: Optional[Union[str, torch.device]] = None,
        load_checkpoints: bool = True,
        strict: bool = True,
    ):
        super().__init__()
        self.attribute_names: List[str] = list(banff_attributes.keys())

        self.predictors = nn.ModuleDict(
            {
                attr: MODELS[banff_attributes[attr]["model_type"]](
                    **banff_attributes[attr]["model_params"]
                )
                for attr in self.attribute_names
            }
        )
        self._checkpoint_paths: Dict[str, str] = {
            attr: banff_attributes[attr]["checkpoint"] for attr in self.attribute_names
        }

        if load_checkpoints:
            self._load_checkpoints(strict=strict)
        if device is not None:
            self.to(device)
        self.eval()

    @classmethod
    def from_yaml(
        cls,
        config_path: PathLike,
        *,
        device: Optional[Union[str, torch.device]] = None,
        load_checkpoints: bool = True,
        strict: bool = True,
    ) -> "BanffPredictor":
        """Construct from a YAML configuration file."""
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        return cls(cfg, device=device, load_checkpoints=load_checkpoints, strict=strict)

    def _load_checkpoints(self, *, strict: bool) -> None:
        for attr, head in self.predictors.items():
            path = self._checkpoint_paths[attr]
            state_dict = torch.load(path, map_location="cpu", weights_only=True)
            head.load_state_dict(state_dict, strict=strict)

    @property
    def device(self) -> torch.device:
        """The device the first parameter lives on."""
        return next(self.parameters()).device

    @torch.inference_mode()
    def predict_slide(self, features: torch.Tensor) -> Dict[str, float]:
        """Slide-level prediction: one float per attribute.

        Parameters
        ----------
        features
            ``[N_patches, D]`` feature tensor, already on the right device.

        Returns
        -------
        dict
            ``{attribute_name: bag_probability_float}``.
        """
        return {
            attr: float(self.predictors[attr].infer(features).item())
            for attr in self.attribute_names
        }

    @torch.inference_mode()
    def predict_patch(self, features: torch.Tensor) -> np.ndarray:
        """Patch-level prediction: one probability vector per patch.

        Parameters
        ----------
        features
            ``[N_patches, D]`` feature tensor, already on the right device.

        Returns
        -------
        np.ndarray
            ``[N_patches, N_attributes]`` float32 array on CPU.
        """
        per_attr = [
            self.predictors[attr].infer_patch_level(features) for attr in self.attribute_names
        ]
        return torch.cat(per_attr, dim=1).cpu().numpy()

    @torch.inference_mode()
    def predict_slide_by_roi(
        self,
        features: torch.Tensor,
        roi_id: Union[np.ndarray, Iterable[str]],
    ) -> Dict[str, Dict[str, float]]:
        """Per-ROI prediction: one bag-level score dict per ROI.

        Splits ``features`` into groups by ``roi_id`` and runs the same
        bag-level aggregation as :meth:`predict_slide` independently
        within each group — so a slide with three biopsy cores yields
        three sets of attribute scores instead of one pooled set.

        Parameters
        ----------
        features
            ``[N_patches, D]`` feature tensor, already on the right device.
        roi_id
            Length-``N_patches`` str array/sequence giving the ROI each
            row of ``features`` came from — the ``roi_id`` metadata
            produced by :class:`wsipatch.PatchReader` (the ROI's
            annotation Name, or its positional index as a string when
            it has none).

        Returns
        -------
        dict
            ``{roi_id: {attribute_name: bag_probability_float}}``, one
            entry per distinct value in ``roi_id``, sorted by id.
        """
        roi_arr = np.asarray(roi_id)
        if roi_arr.shape[0] != features.shape[0]:
            raise ValueError(
                f"roi_id length ({roi_arr.shape[0]}) != features length ({features.shape[0]})"
            )
        out: Dict[str, Dict[str, float]] = {}
        for rid in sorted(set(roi_arr.tolist()), key=str):
            mask = torch.from_numpy(roi_arr == rid).to(features.device)
            out[str(rid)] = self.predict_slide(features[mask])
        return out

    def predict_features_file(
        self,
        h5_path: PathLike,
        *,
        level: str = "slide",
        features_key: str = "features",
    ) -> Union[Dict[str, float], np.ndarray, Dict[str, Dict[str, float]]]:
        """Run inference on a single features H5 file."""
        with h5py.File(h5_path, "r") as f:
            arr = f[features_key][:]
            roi_id = None
            if level == "roi":
                if "metadata" not in f or "roi_id" not in f["metadata"]:
                    raise KeyError(
                        f"{h5_path} has no 'metadata/roi_id' dataset, "
                        f"required for level='roi'. It must be produced by "
                        f"a pipeline that tracks per-patch ROI provenance "
                        f"(e.g. BanffNETPipeline with save_features=True)."
                    )
                # Decode HDF5 variable-length UTF-8 strings.
                roi_id = f["metadata"]["roi_id"].asstr()[:]
        feats = torch.from_numpy(arr).to(self.device)
        if level == "slide":
            return self.predict_slide(feats)
        if level == "patch":
            return self.predict_patch(feats)
        if level == "roi":
            return self.predict_slide_by_roi(feats, roi_id)
        raise ValueError(f"level must be 'slide', 'patch', or 'roi', got {level!r}")

    def predict_features_directory(
        self,
        features_dir: PathLike,
        *,
        level: str = "slide",
        file_list: Optional[Iterable[PathLike]] = None,
        features_key: str = "features",
    ) -> Iterable:
        """Yield ``(slide_id, scores)`` for every H5 in ``features_dir``.

        Slide ID is the H5 filename stem.
        """
        if file_list is None:
            file_list = sorted(Path(features_dir).glob("*.h5"))
        else:
            file_list = [Path(features_dir) / p for p in file_list]

        for fp in file_list:
            slide_id = Path(fp).stem
            scores = self.predict_features_file(
                fp,
                level=level,
                features_key=features_key,
            )
            yield slide_id, scores
