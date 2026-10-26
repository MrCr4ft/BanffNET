"""Batched resizing, stain normalization, and ImageNet normalization.

Accept NumPy arrays or tensors in channel-first or channel-last layout and
return a normalized ``(B, 3, H, W)`` tensor on the module's device.
"""

import typing

import numpy as np
import torch
import torch.nn.functional as F

from .reinhard import Reinhard

IMAGENET_MEAN: typing.Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: typing.Tuple[float, float, float] = (0.229, 0.224, 0.225)


class BatchPreprocess(torch.nn.Module):
    """Resize -> [Reinhard] -> Normalize, batched, on whatever device the
    module is moved to.

    Accepts as input any of:

    * ``np.ndarray`` (H, W, 3) or (B, H, W, 3), uint8 or float in [0, 1]
    * ``torch.Tensor`` (3, H, W), (B, 3, H, W), or channel-last equivalents,
      uint8 or float in [0, 1]

    Single images are auto-promoted to a batch of size 1.
    """

    def __init__(
        self,
        image_size: int = 224,
        mean: typing.Sequence[float] = IMAGENET_MEAN,
        std: typing.Sequence[float] = IMAGENET_STD,
        reinhard: typing.Optional[Reinhard] = None,
        interpolation: str = "bilinear",
        antialias: bool = True,
    ):
        super().__init__()
        self.image_size = int(image_size)
        self.interpolation = interpolation
        self.antialias = bool(antialias)
        # Reinhard registered as a submodule so it follows .to(device).
        self.reinhard = reinhard
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

    @staticmethod
    def _as_batch_chw(batch: typing.Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Coerce input into a 4-D channel-first tensor without changing dtype."""
        if isinstance(batch, np.ndarray):
            t = torch.from_numpy(np.ascontiguousarray(batch))
        elif isinstance(batch, torch.Tensor):
            t = batch
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")
        if t.dim() == 3:
            t = t.unsqueeze(0)
        if t.dim() != 4:
            raise ValueError(f"Expected 3D or 4D input, got shape {tuple(t.shape)}")
        # Channel-last -> channel-first if needed.
        if t.shape[1] != 3 and t.shape[-1] == 3:
            t = t.permute(0, 3, 1, 2).contiguous()
        if t.shape[1] != 3:
            raise ValueError(f"Expected 3 channels at dim 1, got {tuple(t.shape)}")
        return t

    def forward(self, batch: typing.Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        x = self._as_batch_chw(batch)
        # Move to device while still uint8 if applicable (smaller H2D transfer).
        target_device = self.mean.device
        x = x.to(target_device, non_blocking=True)
        # uint8 -> float32 in [0, 1]; existing floats are passed through.
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        elif x.dtype != torch.float32:
            x = x.float()
        # Resize (skip if already the target size).
        if x.shape[-2] != self.image_size or x.shape[-1] != self.image_size:
            if self.interpolation == "nearest":
                x = F.interpolate(x, size=(self.image_size, self.image_size), mode="nearest")
            else:
                x = F.interpolate(
                    x,
                    size=(self.image_size, self.image_size),
                    mode=self.interpolation,
                    align_corners=False,
                    antialias=self.antialias,
                )
        # Reinhard (batched, fully on-tensor).
        if self.reinhard is not None:
            x = self.reinhard(x)
        # ImageNet normalize.
        x = (x - self.mean) / self.std
        return x
