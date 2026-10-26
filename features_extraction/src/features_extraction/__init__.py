"""features_extraction — batched ViT feature extraction with optional Reinhard."""

from .extractor import FeatureExtractor
from .io_utils import (
    FeatureH5Writer,
    detect_patches_dataset,
    iter_folder_patches,
    iter_h5_patches,
    iter_plan_patches,
    write_features_h5,
)
from .model import load_uni_vit
from .reinhard import (
    Reinhard,
    get_reinhard_normalizer,
    lab_to_rgb,
    load_reinhard_from_yaml,
    rgb_to_lab,
)
from .transforms import IMAGENET_MEAN, IMAGENET_STD, BatchPreprocess

__all__ = [
    "rgb_to_lab",
    "lab_to_rgb",
    "Reinhard",
    "get_reinhard_normalizer",
    "load_reinhard_from_yaml",
    "BatchPreprocess",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "load_uni_vit",
    "FeatureExtractor",
    "FeatureH5Writer",
    "detect_patches_dataset",
    "iter_folder_patches",
    "iter_h5_patches",
    "iter_plan_patches",
    "write_features_h5",
]

__version__ = "0.1.0"
