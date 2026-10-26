"""Segmentation model loading, device selection, CLAHE, and batched inference."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from .config import IMAGE_NORM_MEAN, IMAGE_NORM_STD

log = logging.getLogger(__name__)


class ModelLoadError(Exception):
    """Raised when a model checkpoint cannot be loaded."""


SUPPORTED_ENCODERS: List[str] = ["mobilenet_v2"]
SUPPORTED_ARCHITECTURES: List[str] = ["UnetPlusPlus"]


def enhance_contrast_clahe(
    image: Image.Image,
    clip_limit: float = 1.0,
    tile_size: int = 32,
) -> Image.Image:
    """Apply CLAHE to the L channel of an L*a*b* representation.

    Preserves color (only the lightness channel is contrast-stretched).
    """
    arr = np.array(image)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l_enhanced = clahe.apply(l)
    merged = cv2.merge((l_enhanced, a, b))
    return Image.fromarray(cv2.cvtColor(merged, cv2.COLOR_LAB2RGB))


def select_device(prefer_cuda: bool = True) -> torch.device:
    """Return ``cuda`` if available and ``prefer_cuda``, else ``cpu``."""
    if prefer_cuda and torch.cuda.is_available():
        dev = torch.device("cuda")
        log.info("Using GPU: %s", torch.cuda.get_device_name(0))
        return dev
    log.info("Using CPU")
    return torch.device("cpu")


def _extract_state_dict(checkpoint: Any) -> Dict:
    """Find the actual state-dict inside common checkpoint wrappers."""
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in checkpoint:
                return checkpoint[key]
    return checkpoint


def load_segmentation_model(
    checkpoint_path: str,
    architecture: str = "UnetPlusPlus",
    encoder_name: str = "mobilenet_v2",
    *,
    in_channels: int = 3,
    classes: int = 1,
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> Tuple[nn.Module, torch.device]:
    """Construct the network and load its weights.

    Returns ``(model, device)``. The model is moved to ``device`` and set
    to ``eval()`` before returning.
    """
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(
            f"Unsupported architecture: {architecture!r}; supported: {SUPPORTED_ARCHITECTURES}"
        )
    if encoder_name not in SUPPORTED_ENCODERS:
        log.warning(
            "Encoder %r not in tested list %s — proceeding anyway",
            encoder_name,
            SUPPORTED_ENCODERS,
        )

    device = device or select_device()
    try:
        cls = getattr(smp, architecture)
        model = cls(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=in_channels,
            classes=classes,
        )
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(_extract_state_dict(ckpt), strict=strict)
        model = model.to(device).eval()
        log.info("Loaded %s/%s from %s", architecture, encoder_name, checkpoint_path)
        return model, device
    except Exception as e:
        raise ModelLoadError(f"Failed to load model from {checkpoint_path}: {e}") from e


class SegmentationPipeline:
    """Wrap a pretrained binary-segmentation model for batched inference.

    Steps applied to a batch of PIL images:

    1. Resize each to ``fixed_shape``.
    2. Optional CLAHE in L*a*b* space (controlled per-call).
    3. ImageNet normalisation, stack into ``(B, 3, H, W)`` float tensor.
    4. Forward pass under :func:`torch.no_grad`; sigmoid output thresholded
       at ``probability_threshold`` to produce binary masks.
    5. Optional morphological closing.
    6. Resize each mask back to its original (pre-resize) image size with
       nearest-neighbour.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        fixed_shape: Tuple[int, int] = (1024, 1024),
        *,
        probability_threshold: float = 0.5,
        apply_morphology_postprocessing: bool = True,
        closing_kernel_size: int = 7,
    ):
        self.model = model
        self.device = device
        self.fixed_shape = fixed_shape
        self.probability_threshold = probability_threshold
        self.apply_morphology = apply_morphology_postprocessing
        self._morph_kernel = (
            np.ones((closing_kernel_size, closing_kernel_size), np.uint8)
            if self.apply_morphology
            else None
        )
        self._transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGE_NORM_MEAN, std=IMAGE_NORM_STD),
            ]
        )

    def segment_batch(
        self,
        images: List[Image.Image],
        *,
        apply_contrast_enhancement: bool = False,
        clahe_clip_limit: float = 1.0,
        clahe_tile_size: int = 32,
    ) -> List[np.ndarray]:
        """Run the full pre/forward/post pipeline on a batch."""
        if not images:
            return []
        original_sizes = [img.size for img in images]

        tensors = []
        for img in images:
            img = img.resize(self.fixed_shape)
            if apply_contrast_enhancement:
                img = enhance_contrast_clahe(img, clahe_clip_limit, clahe_tile_size)
            tensors.append(self._transform(img))

        with torch.no_grad():
            batch = torch.stack(tensors).to(self.device, non_blocking=True)
            probs = self.model(batch).squeeze(1).cpu().numpy()  # (B, H, W)

        masks: List[np.ndarray] = []
        for prob, (w, h) in zip(probs, original_sizes):
            mask = (prob > self.probability_threshold).astype(np.uint8) * 255
            if self._morph_kernel is not None:
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._morph_kernel)
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            masks.append(mask)
        return masks

    @staticmethod
    def detect_optimal_batch_size(
        model: nn.Module,
        device: torch.device,
        fixed_shape: Tuple[int, int],
        max_batch_size: int = 64,
    ) -> int:
        """Double the batch size until CUDA OOM, then back off by 2.

        Returns ``8`` on CPU. Returns the last successful size on GPU,
        capped at ``max_batch_size``.
        """
        if device.type != "cuda":
            return 8

        log.info("Detecting optimal batch size...")
        bs = 1
        while bs <= max_batch_size:
            try:
                dummy = torch.randn(bs, 3, fixed_shape[0], fixed_shape[1], device=device)
                with torch.no_grad():
                    _ = model(dummy)
                torch.cuda.empty_cache()
                if bs == max_batch_size:
                    return bs
                bs *= 2
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    return max(bs // 2, 1)
                raise
        return max_batch_size
