"""Extract features from HDF5 patches, image folders, or wsipatch plans.

Each command streams a ``features`` dataset and associated ``metadata`` to HDF5.
"""

from __future__ import annotations

import typing

import click

from .extractor import FeatureExtractor
from .io_utils import (
    FeatureH5Writer,
    iter_folder_patches,
    iter_h5_patches,
    iter_plan_patches,
)


def _common_options(fn: typing.Callable) -> typing.Callable:
    """Decorator stack with options shared by every subcommand."""
    decorators = [
        click.option(
            "--checkpoint",
            "checkpoint_path",
            required=True,
            type=click.Path(exists=True, dir_okay=False),
            help="UNI/ViT checkpoint (.pt / .pth).",
        ),
        click.option(
            "--reinhard-config",
            "reinhard_config_path",
            default=None,
            type=click.Path(exists=True, dir_okay=False),
            help="Optional Reinhard YAML config; if omitted, no stain normalization is applied.",
        ),
        click.option(
            "--output",
            "-o",
            "output_path",
            required=True,
            type=click.Path(dir_okay=False),
            help="Output HDF5 path.",
        ),
        click.option("--batch-size", type=int, default=768, show_default=True),
        click.option("--device", type=str, default="cuda", show_default=True),
        click.option("--image-size", type=int, default=224, show_default=True),
        click.option("--model-name", type=str, default="vit_large_patch16_224", show_default=True),
        click.option(
            "--amp/--no-amp",
            default=False,
            show_default=True,
            help="Mixed-precision inference (CUDA only).",
        ),
    ]
    for dec in reversed(decorators):
        fn = dec(fn)
    return fn


def _build_extractor(checkpoint_path, reinhard_config_path, device, image_size, model_name, amp):
    return FeatureExtractor.from_files(
        checkpoint_path=checkpoint_path,
        reinhard_config_path=reinhard_config_path,
        device=device,
        image_size=image_size,
        model_name=model_name,
        use_amp=amp,
    )


def _drain(
    extractor: FeatureExtractor, iterator, output_path: str, attrs: typing.Mapping[str, typing.Any]
) -> None:
    """Run an iterator through the extractor and stream the results to disk."""
    try:
        from tqdm.auto import tqdm

        bar = tqdm(iterator, desc="Extracting features", unit="batch")
    except ImportError:
        bar = iterator

    with FeatureH5Writer(output_path) as writer:
        writer.set_attrs(**{k: v for k, v in attrs.items() if v is not None})
        for batch, meta in bar:
            feats = extractor.extract_batch(batch)
            writer.append(feats, meta)
        click.echo(f"Wrote {writer.n} feature vectors to {output_path}")


@click.group()
def cli() -> None:
    """Patch-based feature extraction with ViT backbones and optional Reinhard stain normalization."""


@cli.command("from-h5")
@click.option(
    "--input",
    "-i",
    "h5_input",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Input HDF5 file containing patches.",
)
@click.option(
    "--patches-key",
    type=str,
    default=None,
    help="Name of the patches dataset (auto-detected if omitted).",
)
@click.option(
    "--extra-keys",
    type=str,
    multiple=True,
    help="Sibling datasets to copy through as metadata (repeatable).",
)
@_common_options
def from_h5(
    h5_input,
    patches_key,
    extra_keys,
    checkpoint_path,
    reinhard_config_path,
    output_path,
    batch_size,
    device,
    image_size,
    model_name,
    amp,
):
    """Extract features from patches stored in an HDF5 file."""
    extractor = _build_extractor(
        checkpoint_path, reinhard_config_path, device, image_size, model_name, amp
    )
    iterator = iter_h5_patches(
        h5_input,
        batch_size=batch_size,
        dataset_key=patches_key,
        extra_keys=list(extra_keys) if extra_keys else None,
    )
    attrs = {
        "source": "h5",
        "input_path": str(h5_input),
        "reinhard": bool(reinhard_config_path),
        "model_name": model_name,
        "image_size": image_size,
    }
    _drain(extractor, iterator, output_path, attrs)


@cli.command("from-folder")
@click.option(
    "--folder",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Folder containing image files.",
)
@_common_options
def from_folder(
    folder,
    checkpoint_path,
    reinhard_config_path,
    output_path,
    batch_size,
    device,
    image_size,
    model_name,
    amp,
):
    """Extract features from a folder of image files (.png, .jpg, .tif, ...)."""
    extractor = _build_extractor(
        checkpoint_path, reinhard_config_path, device, image_size, model_name, amp
    )
    iterator = iter_folder_patches(folder, batch_size=batch_size)
    attrs = {
        "source": "folder",
        "input_path": str(folder),
        "reinhard": bool(reinhard_config_path),
        "model_name": model_name,
        "image_size": image_size,
    }
    _drain(extractor, iterator, output_path, attrs)


@cli.command("from-plan")
@click.option(
    "--plan",
    "plan_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="wsipatch patching plan (.wsix / .json).",
)
@click.option(
    "--slide-path",
    "wsi_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Override the slide path stored in the plan.",
)
@click.option(
    "-t",
    "--target",
    "target_name",
    default=None,
    help="Target name to extract (required if the plan has >1 target).",
)
@click.option(
    "--low-memory/--standard",
    default=False,
    show_default=True,
    help="Per-patch reads (bounded RAM) vs whole-ROI reads (faster).",
)
@_common_options
def from_plan(
    plan_path,
    wsi_path,
    target_name,
    low_memory,
    checkpoint_path,
    reinhard_config_path,
    output_path,
    batch_size,
    device,
    image_size,
    model_name,
    amp,
):
    """Extract features from a wsipatch patching plan."""
    extractor = _build_extractor(
        checkpoint_path, reinhard_config_path, device, image_size, model_name, amp
    )
    iterator = iter_plan_patches(
        plan_path,
        wsi_path=wsi_path,
        batch_size=batch_size,
        target_name=target_name,
        low_memory=low_memory,
    )
    attrs = {
        "source": "plan",
        "plan_path": str(plan_path),
        "slide_path": str(wsi_path) if wsi_path else "",
        "target_name": target_name or "",
        "low_memory": bool(low_memory),
        "reinhard": bool(reinhard_config_path),
        "model_name": model_name,
        "image_size": image_size,
    }
    _drain(extractor, iterator, output_path, attrs)


if __name__ == "__main__":
    cli()
