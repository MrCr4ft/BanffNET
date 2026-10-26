"""Backbone model loading."""

import os
import typing

import timm
import torch


def _unwrap_state_dict(obj: typing.Any) -> typing.Mapping[str, torch.Tensor]:
    """Unwrap state dicts saved with common wrapping keys."""
    if not isinstance(obj, dict):
        return obj  # assume it's already a state_dict-like
    for key in ("state_dict", "model_state_dict", "model"):
        if key in obj and isinstance(obj[key], dict):
            return obj[key]
    return obj


def load_uni_vit(
    checkpoint_path: typing.Union[str, os.PathLike],
    device: typing.Union[str, torch.device] = "cuda",
    model_name: str = "vit_large_patch16_224",
    img_size: int = 224,
    patch_size: int = 16,
    init_values: float = 1e-5,
    strict: bool = True,
) -> torch.nn.Module:
    """Build a timm ViT configured for UNI-style feature extraction and load
    weights from ``checkpoint_path``. Returns the model in eval mode on
    ``device``.
    """
    model = timm.create_model(
        model_name,
        img_size=img_size,
        patch_size=patch_size,
        init_values=init_values,
        num_classes=0,
        dynamic_img_size=True,
    )
    state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    state = _unwrap_state_dict(state)
    model.load_state_dict(state, strict=strict)
    model.eval()
    model.to(device)
    return model
