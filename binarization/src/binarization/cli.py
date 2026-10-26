"""Command-line entry points for single-slide and directory tissue segmentation."""

from __future__ import annotations

import logging
import sys
from typing import Tuple

import click
from PIL import Image

from .config import BinarizationConfig
from .io import get_biopsy_list, setup_logging
from .pipeline import BinarizationPipeline

log = logging.getLogger(__name__)
Image.MAX_IMAGE_PIXELS = None


def common_options(f):
    """Options shared by both subcommands."""
    decorators = [
        click.option(
            "--roi-dir", required=True, type=click.Path(), help="Directory to write ROI thumbnails"
        ),
        click.option(
            "--mask-dir", required=True, type=click.Path(), help="Directory to write binary masks"
        ),
        click.option(
            "--model-path",
            default=None,
            type=click.Path(exists=True, dir_okay=False),
            help="Segmentation checkpoint (required unless --extract-only).",
        ),
        click.option(
            "--biopsy-ext",
            default="tiff",
            show_default=True,
            help="Extension of biopsy images (used by `dir` mode)",
        ),
        click.option(
            "--input-shape",
            default=1024,
            type=int,
            show_default=True,
            help="Square input size expected by the network",
        ),
        click.option(
            "--apply-contrast-enhancement",
            is_flag=True,
            help="Apply CLAHE to ROIs before inference",
        ),
        click.option(
            "--probability-threshold",
            default=0.5,
            type=float,
            show_default=True,
            help="Probability threshold for binarization",
        ),
        click.option(
            "--apply-morphology-postprocessing/--no-apply-morphology-postprocessing",
            default=True,
            show_default=True,
            help="Fill holes in predicted masks via morphological closing",
        ),
        click.option(
            "--closing-kernel-size",
            default=7,
            type=int,
            show_default=True,
            help="Kernel size for the closing operation",
        ),
        click.option(
            "--batch-size", default=None, type=int, help="Override the auto batch-size detection"
        ),
        click.option(
            "--extract-only", is_flag=True, help="Skip phase 2 (segmentation) — only extract ROIs"
        ),
        click.option(
            "--log-level",
            default="INFO",
            show_default=True,
            type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
            help="Logging level",
        ),
        click.option(
            "--use-entire-wsi-if-no-annotation/--require-annotation",
            default=True,
            show_default=True,
            help="Fall back to the entire WSI as a single ROI when no annotation XML is found",
        ),
    ]
    for d in reversed(decorators):
        f = d(f)
    return f


def _setup(kwargs: dict) -> Tuple[BinarizationConfig, bool]:
    """Set up logging, pop the CLI-only flags, build the config."""
    setup_logging(level=getattr(logging, kwargs.pop("log_level")))
    extract_only = kwargs.pop("extract_only", False)
    if not extract_only and kwargs["model_path"] is None:
        raise click.UsageError("Provide --model-path unless using --extract-only.")
    config = BinarizationConfig(
        roi_dir=kwargs["roi_dir"],
        mask_dir=kwargs["mask_dir"],
        model_path=kwargs["model_path"] or "",
        biopsy_ext=kwargs["biopsy_ext"],
        input_shape=kwargs["input_shape"],
        apply_contrast_enhancement=kwargs["apply_contrast_enhancement"],
        probability_threshold=kwargs["probability_threshold"],
        apply_morphology_postprocessing=kwargs["apply_morphology_postprocessing"],
        closing_kernel_size=kwargs["closing_kernel_size"],
        batch_size=kwargs["batch_size"],
        use_entire_wsi_if_no_annotation=kwargs["use_entire_wsi_if_no_annotation"],
    )
    return config, extract_only


def _run(config: BinarizationConfig, biopsy_list: dict, extract_only: bool) -> None:
    """Run the pipeline with the same error handling both subcommands need."""
    try:
        with BinarizationPipeline(config) as pipeline:
            pipeline.run(biopsy_list, extract_only=extract_only)
    except KeyboardInterrupt:
        log.info("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        log.error("Pipeline failed: %s", e, exc_info=True)
        sys.exit(1)


@click.group("tissue-binarize", context_settings={"show_default": True})
def cli():
    """ROI extraction and tissue segmentation for whole-slide images."""


@cli.command("slide")
@click.option(
    "--slide-path", required=True, type=click.Path(exists=True), help="Path to a single WSI file"
)
@click.option(
    "--annotation-path",
    default=None,
    type=click.Path(),
    help="ASAP XML annotation. Omit to process the entire WSI.",
)
@common_options
def slide_cmd(slide_path, annotation_path, **kwargs):
    """Binarize tissue regions for a single whole-slide image."""
    config, extract_only = _setup(kwargs)
    biopsy_list = {"biopsy": [slide_path], "annotation": [annotation_path]}
    _run(config, biopsy_list, extract_only)


@cli.command("dir")
@click.option(
    "--biopsy-dir", required=True, type=click.Path(exists=True), help="Directory with biopsy images"
)
@click.option(
    "--annotations-dir",
    required=True,
    type=click.Path(exists=True),
    help="Directory with annotation XMLs",
)
@common_options
def dir_cmd(biopsy_dir, annotations_dir, **kwargs):
    """Binarize tissue regions for every WSI in a directory."""
    config, extract_only = _setup(kwargs)
    biopsy_list = get_biopsy_list(
        biopsy_dir,
        annotations_dir,
        config.biopsy_ext,
        use_entire_wsi_if_no_annotation=config.use_entire_wsi_if_no_annotation,
    )
    _run(config, biopsy_list, extract_only)


# Standalone entry-point aliases.

main_single = slide_cmd
main_multi = dir_cmd


def main() -> int:
    """Top-level entry: ``tissue-binarize {slide|dir} ...``."""
    cli()
    return 0


if __name__ == "__main__":
    sys.exit(main())
