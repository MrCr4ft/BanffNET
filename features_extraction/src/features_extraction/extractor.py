"""Reusable ViT feature extraction with batched preprocessing and optional AMP.

``extract_batch`` returns NumPy features for one batch. ``extract_iterable``
collects features from batches or ``(batch, metadata)`` pairs.
"""

import typing
from pathlib import Path

import numpy as np
import torch

from .model import load_uni_vit
from .reinhard import Reinhard, load_reinhard_from_yaml
from .transforms import IMAGENET_MEAN, IMAGENET_STD, BatchPreprocess

BatchLike = typing.Union[np.ndarray, torch.Tensor]


class FeatureExtractor:
    """Encapsulates preprocessing + backbone inference for a single device."""

    def __init__(
        self,
        checkpoint_path: typing.Union[str, Path],
        reinhard: typing.Optional[Reinhard] = None,
        device: typing.Union[str, torch.device] = "cuda",
        image_size: int = 224,
        model_name: str = "vit_large_patch16_224",
        patch_size: int = 16,
        init_values: float = 1e-5,
        mean: typing.Sequence[float] = IMAGENET_MEAN,
        std: typing.Sequence[float] = IMAGENET_STD,
        use_amp: bool = False,
    ):
        self.device = torch.device(device)
        self.model = load_uni_vit(
            checkpoint_path=checkpoint_path,
            device=self.device,
            model_name=model_name,
            img_size=image_size,
            patch_size=patch_size,
            init_values=init_values,
        )
        self.preprocess = BatchPreprocess(
            image_size=image_size,
            mean=mean,
            std=std,
            reinhard=reinhard,
        ).to(self.device)
        self.use_amp = bool(use_amp) and self.device.type == "cuda"

    @classmethod
    def from_files(
        cls,
        checkpoint_path: typing.Union[str, Path],
        reinhard_config_path: typing.Optional[typing.Union[str, Path]] = None,
        **kwargs,
    ) -> "FeatureExtractor":
        """Build an extractor, loading the Reinhard config from YAML if given."""
        reinhard = load_reinhard_from_yaml(reinhard_config_path) if reinhard_config_path else None
        return cls(checkpoint_path=checkpoint_path, reinhard=reinhard, **kwargs)

    @torch.inference_mode()
    def extract_batch(self, batch: BatchLike) -> np.ndarray:
        """Run preprocessing + backbone on a single batch.

        Parameters
        ----------
        batch
            See :class:`BatchPreprocess` for accepted shapes/dtypes.

        Returns
        -------
        np.ndarray
            (B, D) float32 array of features on CPU.
        """
        x = self.preprocess(batch)
        if self.use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                feats = self.model(x)
        else:
            feats = self.model(x)
        return feats.float().cpu().numpy()

    @torch.inference_mode()
    def extract_iterable(
        self,
        iterable: typing.Iterable[
            typing.Union[BatchLike, typing.Tuple[BatchLike, typing.Mapping[str, typing.Any]]]
        ],
        total: typing.Optional[int] = None,
        progress: bool = True,
    ) -> typing.Tuple[np.ndarray, typing.Dict[str, np.ndarray]]:
        """Drain an iterator into ``(features, metadata)``.

        Items may be either bare batches or ``(batch, metadata_dict)`` tuples.
        Metadata is concatenated along axis 0 per key. Use this for small to
        medium runs; for large runs, prefer streaming to disk via
        :class:`features_extraction.io_utils.FeatureH5Writer`.
        """
        if progress:
            try:
                from tqdm.auto import tqdm

                iterable = tqdm(iterable, total=total, desc="Extracting features")
            except ImportError:
                pass

        feat_chunks: typing.List[np.ndarray] = []
        meta_chunks: typing.Dict[str, typing.List[np.ndarray]] = {}
        for item in iterable:
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], typing.Mapping):
                batch, meta = item
            else:
                batch, meta = item, {}
            feat_chunks.append(self.extract_batch(batch))
            for k, v in meta.items():
                meta_chunks.setdefault(k, []).append(np.asarray(v))

        if not feat_chunks:
            return np.empty((0,), dtype=np.float32), {}
        features = np.concatenate(feat_chunks, axis=0)
        metadata = {k: np.concatenate(vs, axis=0) for k, vs in meta_chunks.items()}
        return features, metadata
