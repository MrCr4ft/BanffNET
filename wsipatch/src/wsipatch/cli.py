"""Command-line interface for wsipatch.

Subcommands:
* ``plan-slide``    — one slide → one plan file.
* ``plan-dir``      — directory of slides → plans + manifest.
* ``materialise``   — plan file → HDF5 of RGB patches.
* ``inspect``       — print a summary of a plan.
* ``convert``       — convert between ``.wsix`` and ``.json``.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional, Tuple

import click

from .io import load_index, save_index
from .planner import plan_directory_multi, plan_slide_multi
from .reader import PatchReader
from .types import TargetSpec

log = logging.getLogger(__name__)


def _setup_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _parse_target(s: str) -> TargetSpec:
    """Parse ``name:size:mpp[:stride]``."""
    parts = s.split(":")
    if len(parts) < 3:
        raise click.BadParameter(f"target spec must be 'name:size:mpp[:stride]', got {s!r}")
    return TargetSpec(
        name=parts[0],
        patch_size=int(parts[1]),
        mpp=float(parts[2]),
        stride=int(parts[3]) if len(parts) > 3 else None,
    )


@click.group()
def cli() -> None:
    """wsipatch — WSI patch planning."""


@cli.command("plan-dir")
@click.argument("biopsy_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("output_dir", type=click.Path())
@click.option(
    "-t",
    "--target",
    "targets",
    multiple=True,
    required=True,
    help="Target as name:size:mpp[:stride] (repeatable).",
)
@click.option("--annotations-dir", type=click.Path(exists=True, file_okay=False))
@click.option("--annotations-ext", default="xml", show_default=True)
@click.option("--mask-dir", type=click.Path(exists=True, file_okay=False))
@click.option("--mask-ext", default="png", show_default=True)
@click.option("--mask-template", default="{slide_id}_roi_{roi_idx}_mask.{ext}", show_default=True)
@click.option("--biopsy-ext", default="tiff", show_default=True)
@click.option("--tissue-threshold", default=0.5, show_default=True, type=float)
@click.option("--max-upscale", default=1.0, show_default=True, type=float)
@click.option(
    "--on-missing-mask",
    type=click.Choice(["accept_all", "reject_all", "raise"]),
    default="accept_all",
    show_default=True,
)
@click.option(
    "--no-full-slide-fallback",
    is_flag=True,
    help="Skip slides without annotations instead of treating them as full-slide ROIs.",
)
@click.option(
    "--format", "fmt", type=click.Choice(["wsix", "json"]), default="wsix", show_default=True
)
@click.option(
    "--include-coords", is_flag=True, help="Include explicit (x,y) coords in JSON output."
)
@click.option("--no-skip-existing", is_flag=True, help="Rebuild plans even if they already exist.")
@click.option("-v", "--verbose", count=True)
def plan_dir_cmd(
    biopsy_dir: str,
    output_dir: str,
    targets: Tuple[str, ...],
    annotations_dir: Optional[str],
    annotations_ext: str,
    mask_dir: Optional[str],
    mask_ext: str,
    mask_template: str,
    biopsy_ext: str,
    tissue_threshold: float,
    max_upscale: float,
    on_missing_mask: str,
    no_full_slide_fallback: bool,
    fmt: str,
    include_coords: bool,
    no_skip_existing: bool,
    verbose: int,
):
    """Plan patches for every slide in BIOPSY_DIR; write to OUTPUT_DIR."""
    _setup_logging(verbose)
    manifest_path = plan_directory_multi(
        biopsy_dir=biopsy_dir,
        targets=[_parse_target(t) for t in targets],
        output_dir=output_dir,
        annotations_dir=annotations_dir,
        annotations_ext=annotations_ext,
        mask_dir=mask_dir,
        mask_ext=mask_ext,
        mask_template=mask_template,
        biopsy_ext=biopsy_ext,
        tissue_threshold=tissue_threshold,
        max_upscale=max_upscale,
        on_missing_mask=on_missing_mask,  # type: ignore[arg-type]
        use_full_slide_if_no_annotation=not no_full_slide_fallback,
        format=fmt,  # type: ignore[arg-type]
        include_coords=include_coords,
        skip_existing=not no_skip_existing,
    )
    click.echo(f"Manifest: {manifest_path}")


@cli.command("plan-slide")
@click.argument("slide_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_path", type=click.Path())
@click.option(
    "-t",
    "--target",
    "targets",
    multiple=True,
    required=True,
    help="Target as name:size:mpp[:stride] (repeatable).",
)
@click.option(
    "--annotations",
    "annotations_path",
    type=click.Path(exists=True, dir_okay=False),
    help="ASAP XML annotation file. Each annotation's bounding "
    "box becomes an ROI. Without this flag, the full slide "
    "is used.",
)
@click.option(
    "--annotation-group",
    "annotation_groups",
    multiple=True,
    help="Keep only annotations with this PartOfGroup. Repeatable.",
)
@click.option(
    "--mask",
    "masks",
    multiple=True,
    type=click.Path(),
    help="Per-ROI mask path; pass once per ROI in order.",
)
@click.option("--tissue-threshold", default=0.5, show_default=True, type=float)
@click.option("--max-upscale", default=1.0, show_default=True, type=float)
@click.option(
    "--on-missing-mask",
    type=click.Choice(["accept_all", "reject_all", "raise"]),
    default="accept_all",
    show_default=True,
)
@click.option(
    "--include-coords", is_flag=True, help="JSON output only: include explicit (x,y) coords."
)
@click.option("-v", "--verbose", count=True)
def plan_slide_cmd(
    slide_path: str,
    output_path: str,
    targets: Tuple[str, ...],
    annotations_path: Optional[str],
    annotation_groups: Tuple[str, ...],
    masks: Tuple[str, ...],
    tissue_threshold: float,
    max_upscale: float,
    on_missing_mask: str,
    include_coords: bool,
    verbose: int,
):
    """Plan patches for one SLIDE and save them to OUTPUT_PATH.

    Without --annotations, uses the slide's full extent as a single ROI.
    With --annotations, loads ROIs from an ASAP XML file (one per
    annotation, by bounding box). OUTPUT_PATH's extension determines
    the format (``.wsix``/``.json``).
    """
    _setup_logging(verbose)
    target_specs = [_parse_target(t) for t in targets]

    rois = None
    roi_source = ""
    if annotations_path is not None:
        from .annotations import load_asap_rois

        rois = load_asap_rois(
            annotations_path,
            groups=list(annotation_groups) if annotation_groups else None,
        )
        if not rois:
            raise click.UsageError(
                f"No usable ROIs found in {annotations_path}"
                + (f" for groups {list(annotation_groups)}" if annotation_groups else "")
                + "."
            )
        roi_source = f"asap_xml:{annotations_path}"
        if masks and len(masks) != len(rois):
            raise click.UsageError(
                f"Got {len(masks)} --mask paths but {len(rois)} ROIs were "
                f"loaded; pass one --mask per ROI in order."
            )

    idx = plan_slide_multi(
        slide_path=slide_path,
        targets=target_specs,
        rois=rois,
        masks=list(masks) if masks else None,
        tissue_threshold=tissue_threshold,
        max_upscale=max_upscale,
        on_missing_mask=on_missing_mask,  # type: ignore[arg-type]
        roi_source=roi_source,
    )
    out = save_index(idx, output_path, include_coords=include_coords)
    click.echo(
        f"Wrote {out}  ({idx.total_valid()} valid patches across "
        f"{idx.num_rois} ROI{'s' if idx.num_rois != 1 else ''})"
    )


@cli.command("inspect")
@click.argument("path", type=click.Path(exists=True, dir_okay=False))
def inspect_cmd(path: str):
    """Print a summary of a plan file."""
    idx = load_index(path)
    click.echo(f"Slide:        {idx.slide_id}")
    click.echo(f"Path:         {idx.slide_path}")
    click.echo(f"ROI source:   {idx.roi_source}")
    click.echo(
        f"Levels:       {idx.level_count}  "
        f"(downsamples: {[round(d, 2) for d in idx.level_downsamples]})"
    )
    click.echo(f"Base mpp:     {idx.base_mpp_x:.4f} x {idx.base_mpp_y:.4f}")
    click.echo(f"ROIs:         {idx.num_rois}")
    click.echo("")
    click.echo("Targets:")
    for t_idx, spec in enumerate(idx.targets):
        click.echo(
            f"  [{t_idx}] {spec.name:30s} "
            f"{spec.patch_size}x{spec.patch_size} @ {spec.mpp:.3f} µm/px "
            f"stride={spec.stride}"
        )
        for r_idx, g in enumerate(idx.roi_grids[t_idx]):
            click.echo(
                f"      ROI {r_idx}: grid {g.grid_w}x{g.grid_h} "
                f"= {g.num_total} cells, {g.num_valid} valid "
                f"({100 * g.valid_fraction:.1f}%), "
                f"level={g.read_level}, rescale={g.rescale_factor_x:.3f}"
            )
    click.echo("")
    click.echo(f"Total valid:  {idx.total_valid()}")


@cli.command("convert")
@click.argument("in_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("out_path", type=click.Path())
@click.option(
    "--include-coords", is_flag=True, help="When writing JSON: include explicit (x,y) coords."
)
def convert_cmd(in_path: str, out_path: str, include_coords: bool):
    """Convert a plan between formats (auto-detected by extension)."""
    idx = load_index(in_path)
    out = save_index(idx, out_path, include_coords=include_coords)
    click.echo(f"Wrote {out}")


@cli.command("materialise")
@click.argument("plan_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("out_path", type=click.Path())
@click.option(
    "--slide-path",
    type=click.Path(exists=True, dir_okay=False),
    help="Override the slide path stored in the plan.",
)
@click.option(
    "-t",
    "--target",
    "target_name",
    default=None,
    help="Target name to materialise (required if the plan has >1 target).",
)
@click.option(
    "--low-memory/--standard",
    default=False,
    show_default=True,
    help="Per-patch reads vs whole-ROI reads.",
)
@click.option(
    "--batch-size",
    type=int,
    default=64,
    show_default=True,
    help="Streaming batch size for read/write.",
)
@click.option(
    "--compression", default="gzip", show_default=True, type=click.Choice(["gzip", "lzf", "none"])
)
@click.option(
    "--chunks",
    type=int,
    default=None,
    help="HDF5 chunk size along axis 0. Default: min(64, 2**floor(log2(N))).",
)
@click.option("-v", "--verbose", count=True)
def materialise_cmd(
    plan_path: str,
    out_path: str,
    slide_path: Optional[str],
    target_name: Optional[str],
    low_memory: bool,
    batch_size: int,
    compression: str,
    chunks: Optional[int],
    verbose: int,
):
    """Materialise a plan into an HDF5 of RGB patches and coords.

    Produces a single ``.h5`` with two datasets::

        imgs    (N, patch_size, patch_size, 3)  uint8   compressed
        coords  (N, 2)                          int32   compressed

    ``coords`` are ``(x, y)`` in the rescaled-ROI frame, in plan order.
    Per-ROI offsets are stored in ``coords.attrs["roi_offsets"]``.
    """
    _setup_logging(verbose)
    comp = None if compression == "none" else compression

    with PatchReader(
        plan_path,
        slide_path=slide_path,
        target=target_name,
        low_memory=low_memory,
    ) as reader:
        n = reader.num_patches
        click.echo(
            f"Materialising {n} patches "
            f"({reader.patch_size}x{reader.patch_size} @ "
            f"{reader.spec.mpp:.3f} µm/px) → {out_path} "
            f"[{'low-memory' if low_memory else 'whole-ROI'}]"
        )
        out_p = reader.materialise(
            out_path,
            batch_size=batch_size,
            compression=comp,
            chunks=chunks,
            progress=True,
        )
    click.echo(f"Wrote {out_p}  ({n} patches, {comp or 'no'} compression)")


def main() -> int:
    cli()
    return 0


if __name__ == "__main__":
    sys.exit(main())
