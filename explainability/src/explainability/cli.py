"""Command-line interface for slide-level and patch-level saliency maps.

Model and slide-processing dependencies load inside commands so ``--help``
is available without a complete inference environment.
"""

from __future__ import annotations

import logging
import sys

import click

log = logging.getLogger(__name__)


def _model_options(f):
    """Model paths + device/batching, shared by both subcommands."""
    decorators = [
        click.option(
            "--staining",
            default=None,
            help="Use the configured asset paths for this staining "
            "(overridden per-path by the options below).",
        ),
        click.option(
            "--uni-checkpoint",
            type=click.Path(exists=True),
            default=None,
            help="UNI ViT checkpoint. Overrides the staining default.",
        ),
        click.option(
            "--banff-config",
            type=click.Path(exists=True),
            default=None,
            help="BanffNET YAML config (lesions → head + checkpoint).",
        ),
        click.option(
            "--reinhard-config",
            type=click.Path(exists=True),
            default=None,
            help="Reinhard stain-normalisation YAML (optional).",
        ),
        click.option(
            "--binarization-model",
            type=click.Path(exists=True),
            default=None,
            help="Tissue-segmentation checkpoint; enables mask-filtered "
            "planning when starting from a slide.",
        ),
        click.option(
            "--workspace-dir",
            type=click.Path(),
            default=None,
            help="Where intermediate ROIs/masks go when planning from a slide.",
        ),
        click.option("--device", default="cuda:0", show_default=True),
        click.option(
            "--batch-size",
            type=int,
            default=32,
            show_default=True,
            help="Backbone batch size for probabilities.",
        ),
        click.option(
            "--saliency-batch-size",
            type=int,
            default=8,
            show_default=True,
            help="Batch size for the (memory-heavy) GAE backward pass.",
        ),
        click.option("--patch-size", type=int, default=256, show_default=True),
        click.option("--mpp", type=float, default=0.5, show_default=True),
        click.option("--tissue-threshold", type=float, default=0.5, show_default=True),
        click.option(
            "--gip-target",
            type=click.Choice(["classifier", "gate", "contribution", "both"]),
            default="classifier",
            show_default=True,
            help="GIP-only: which branch GAE explains. 'classifier' (paper), "
            "'gate' (native attention), 'contribution' (weighted bag term), "
            "'both' (classifier + gate maps). Ignored for NIP lesions.",
        ),
        click.option(
            "--gip-detach-gate",
            is_flag=True,
            help="GIP-only: detach the gate in the 'classifier' target for a "
            "pure classifier-evidence map.",
        ),
        click.option(
            "--progress/--no-progress",
            default=True,
            show_default=True,
            help="Show tqdm progress bars for reading / features / saliency.",
        ),
        click.option(
            "--amp/--no-amp",
            default=False,
            show_default=True,
            help="Mixed-precision (fp16) for the forward-only feature/triage "
            "pass (CUDA only). Off by default: fp16 can shift the "
            "patch probabilities and change which patches are selected. "
            "GAE always runs in float32 regardless.",
        ),
    ]
    for d in reversed(decorators):
        f = d(f)
    return f


def _saliency_options(f):
    """Lesion selection, thresholding, rendering and output toggles."""
    decorators = [
        click.option(
            "--lesion",
            "lesions",
            multiple=True,
            help="Lesion to explain (repeatable). Omit or pass 'all' for every lesion.",
        ),
        click.option(
            "--output-dir",
            required=True,
            type=click.Path(),
            help="Root output directory ({output}/{slide}/{lesion}/...).",
        ),
        click.option(
            "--threshold",
            default="paper",
            show_default=True,
            help="Probability threshold: 'paper', 'auto' (Otsu), 'pNN' "
            "(percentile, e.g. p90), or a float.",
        ),
        click.option(
            "--alpha",
            type=float,
            default=0.3,
            show_default=True,
            help="Overlay transparency (paper: 0.3).",
        ),
        click.option(
            "--enhance",
            is_flag=True,
            help="Gently post-process maps (smooth, drop tiny specks, close) "
            "for cleaner-looking output. Keeps real values (no hard cut).",
        ),
        click.option(
            "--enhance-smooth-sigma",
            type=float,
            default=1.5,
            show_default=True,
            help="--enhance: Gaussian smoothing sigma (0 disables).",
        ),
        click.option(
            "--enhance-min-component-frac",
            type=float,
            default=0.0005,
            show_default=True,
            help="--enhance: drop specks smaller than this fraction of the "
            "COVERED area (0 keeps everything; raise to filter more).",
        ),
        click.option(
            "--enhance-floor-percentile",
            type=float,
            default=0.0,
            show_default=True,
            help="--enhance: soft floor — zero out the weakest N%% of covered "
            "values (0 = no floor; raise for more aggressive suppression).",
        ),
        click.option(
            "--normalize-percentile",
            type=float,
            default=99.0,
            show_default=True,
            help="Percentile used to normalise the map before rendering.",
        ),
        click.option("--save-overlay/--no-save-overlay", default=True, show_default=True),
        click.option("--save-saliency-png/--no-save-saliency-png", default=True, show_default=True),
        click.option("--save-raw/--no-save-raw", default=True, show_default=True),
        click.option(
            "--log-level",
            default="INFO",
            show_default=True,
            type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
        ),
    ]
    for d in reversed(decorators):
        f = d(f)
    return f


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def _resolve_paths(staining, uni_checkpoint, banff_config, reinhard_config, binarization_model):
    """Merge staining defaults with explicit per-path overrides (banffnet-style)."""
    defaults = {}
    if staining is not None:
        from banffnet.defaults import staining_paths

        d = staining_paths(staining)
        defaults = dict(
            uni_checkpoint=str(d.uni_checkpoint),
            banff_config=str(d.banff_config),
            reinhard_config=str(d.reinhard_config),
            binarization_model=str(d.binarization_model),
        )
    resolved = dict(
        uni_checkpoint=uni_checkpoint or defaults.get("uni_checkpoint"),
        banff_config=banff_config or defaults.get("banff_config"),
        reinhard_config=reinhard_config or defaults.get("reinhard_config"),
        binarization_model=binarization_model or defaults.get("binarization_model"),
    )
    if not resolved["uni_checkpoint"] or not resolved["banff_config"]:
        raise click.UsageError(
            "Provide --staining, or at least --uni-checkpoint and --banff-config."
        )
    return resolved


def _build_pipeline(
    *,
    staining,
    uni_checkpoint,
    banff_config,
    reinhard_config,
    binarization_model,
    workspace_dir,
    device,
    batch_size,
    saliency_batch_size,
    patch_size,
    mpp,
    tissue_threshold,
    stride,
    gip_target,
    gip_detach_gate,
    progress,
    amp,
):
    from .pipeline import ExplainabilityPipeline

    paths = _resolve_paths(
        staining, uni_checkpoint, banff_config, reinhard_config, binarization_model
    )
    return ExplainabilityPipeline(
        workspace_dir=workspace_dir,
        device=device,
        batch_size=batch_size,
        saliency_batch_size=saliency_batch_size,
        patch_size=patch_size,
        mpp=mpp,
        stride=stride,
        tissue_threshold=tissue_threshold,
        gip_target=gip_target,
        gip_detach_gate=gip_detach_gate,
        progress=progress,
        amp=amp,
        **paths,
    )


def _save_opts(
    save_overlay,
    save_saliency_png,
    save_raw,
    alpha,
    enhance,
    normalize_percentile,
    enhance_smooth_sigma=1.5,
    enhance_min_component_frac=0.0005,
    enhance_floor_percentile=0.0,
):
    from .io import SaveOptions

    return SaveOptions(
        overlay=save_overlay,
        saliency_png=save_saliency_png,
        raw=save_raw,
        alpha=alpha,
        enhance=enhance,
        normalize_percentile=normalize_percentile,
        enhance_params={
            "smooth_sigma": enhance_smooth_sigma,
            "min_component_frac": enhance_min_component_frac,
            "floor_percentile": enhance_floor_percentile,
        },
    )


@click.group("run-explainability", context_settings={"show_default": True})
def cli():
    """GAE lesion saliency maps for BanffNET whole-slide images."""


@cli.command("slide-level")
@click.option(
    "--slide-path",
    type=click.Path(exists=True),
    default=None,
    help="WSI to explain. Required unless --plan carries the slide path.",
)
@click.option(
    "--annotation-path",
    type=click.Path(exists=True),
    default=None,
    help="ASAP XML; omit to use the whole slide as one ROI.",
)
@click.option(
    "--plan",
    type=click.Path(exists=True),
    default=None,
    help="Existing .wsix plan; skips planning (stride comes from the plan).",
)
@click.option(
    "--stride",
    type=int,
    default=64,
    show_default=True,
    help="Sampling stride when planning from a slide (paper: 64).",
)
@_model_options
@_saliency_options
def slide_level_cmd(
    slide_path,
    annotation_path,
    plan,
    stride,
    lesions,
    output_dir,
    threshold,
    alpha,
    enhance,
    enhance_smooth_sigma,
    enhance_min_component_frac,
    enhance_floor_percentile,
    normalize_percentile,
    save_overlay,
    save_saliency_png,
    save_raw,
    log_level,
    **model_kwargs,
):
    """Biopsy-averaged, probability-weighted saliency overlay per ROI per lesion."""
    _setup_logging(log_level)
    try:
        with _build_pipeline(stride=stride, **model_kwargs) as pipe:
            pipe.explain_slide_level(
                output_dir=output_dir,
                lesions=list(lesions) or None,
                slide_path=slide_path,
                annotation_path=annotation_path,
                plan=plan,
                threshold=threshold,
                save_opts=_save_opts(
                    save_overlay,
                    save_saliency_png,
                    save_raw,
                    alpha,
                    enhance,
                    normalize_percentile,
                    enhance_smooth_sigma,
                    enhance_min_component_frac,
                    enhance_floor_percentile,
                ),
            )
    except Exception as e:  # noqa: BLE001
        log.error("slide-level failed: %s", e, exc_info=True)
        sys.exit(1)


@cli.command("patch-level")
@click.option("--slide-path", type=click.Path(exists=True), default=None)
@click.option("--annotation-path", type=click.Path(exists=True), default=None)
@click.option(
    "--plan",
    type=click.Path(exists=True),
    default=None,
    help="Existing .wsix plan (patches read from the slide).",
)
@click.option(
    "--patches-h5",
    type=click.Path(exists=True),
    default=None,
    help="Materialised patches H5 (imgs). Fastest patch source.",
)
@click.option(
    "--features-h5",
    type=click.Path(exists=True),
    default=None,
    help="Features H5 to reuse for probabilities (saliency still reads pixels).",
)
@click.option(
    "--max-patches",
    type=int,
    default=None,
    help="Cap the number of patches processed (debug / preview).",
)
@click.option(
    "--stride",
    type=int,
    default=64,
    show_default=True,
    help="Sampling stride if planning from a slide.",
)
@_model_options
@_saliency_options
def patch_level_cmd(
    slide_path,
    annotation_path,
    plan,
    patches_h5,
    features_h5,
    max_patches,
    stride,
    lesions,
    output_dir,
    threshold,
    alpha,
    enhance,
    enhance_smooth_sigma,
    enhance_min_component_frac,
    enhance_floor_percentile,
    normalize_percentile,
    save_overlay,
    save_saliency_png,
    save_raw,
    log_level,
    **model_kwargs,
):
    """Per-patch GAE saliency for patches passing the probability threshold."""
    _setup_logging(log_level)
    try:
        with _build_pipeline(stride=stride, **model_kwargs) as pipe:
            pipe.explain_patch_level(
                output_dir=output_dir,
                lesions=list(lesions) or None,
                slide_path=slide_path,
                annotation_path=annotation_path,
                plan=plan,
                patches_h5=patches_h5,
                features_h5=features_h5,
                threshold=threshold,
                max_patches=max_patches,
                save_opts=_save_opts(
                    save_overlay,
                    save_saliency_png,
                    save_raw,
                    alpha,
                    enhance,
                    normalize_percentile,
                    enhance_smooth_sigma,
                    enhance_min_component_frac,
                    enhance_floor_percentile,
                ),
            )
    except Exception as e:  # noqa: BLE001
        log.error("patch-level failed: %s", e, exc_info=True)
        sys.exit(1)


def main() -> int:
    cli()
    return 0


if __name__ == "__main__":
    sys.exit(main())
