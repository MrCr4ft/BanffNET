"""Command-line interface for slide, directory, CSV, and feature inference.

``--staining`` resolves model paths from ``banffnet.defaults``; individual path
options override those defaults. ``--level`` selects slide, ROI, or patch output.
"""

from __future__ import annotations

import csv
import logging
import sys
import traceback as tb
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import click
import tqdm

from .defaults import STAININGS, staining_paths
from .io import ResultsWriter
from .pipeline import BanffNETPipeline, SlideResult, staining_cache
from .predictor import BanffPredictor

log = logging.getLogger(__name__)


def _level_and_output_options(f):
    """``--level`` + ``--output`` accepted by every subcommand."""
    decorators = [
        click.option(
            "--level",
            type=click.Choice(["slide", "roi", "patch"]),
            default="slide",
            show_default=True,
            help="Output granularity: slide-level (one row per slide, CSV), "
            "roi-level (one row per ROI/biopsy core, CSV), or "
            "patch-level (one HDF5 file per slide).",
        ),
        click.option(
            "--output",
            required=True,
            type=click.Path(),
            help="Where to write results. For --level slide or roi, a CSV "
            "file path. For --level patch, a directory.",
        ),
        click.option(
            "--device",
            default="cuda:0",
            show_default=True,
            help="Torch device for UNI and the predictor.",
        ),
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


def _pipeline_options(f):
    """Path options shared by every subcommand that runs the *full* pipeline."""
    decorators = [
        click.option(
            "--staining",
            type=click.Choice(list(STAININGS), case_sensitive=False),
            default=None,
            help="Use the configured asset paths for this staining. Either this "
            "or all four path options below must be provided.",
        ),
        click.option(
            "--binarization-model",
            type=click.Path(exists=True),
            default=None,
            help="Override the staining default.",
        ),
        click.option(
            "--uni-checkpoint",
            type=click.Path(exists=True),
            default=None,
            help="Override the staining default.",
        ),
        click.option(
            "--reinhard-config",
            type=click.Path(exists=True),
            default=None,
            help="Override the staining default.",
        ),
        click.option(
            "--banff-config",
            type=click.Path(exists=True),
            default=None,
            help="Override the staining default.",
        ),
        click.option(
            "--workspace-dir",
            required=True,
            type=click.Path(),
            help="Where intermediate artifacts go, in subfolders: rois/, "
            "masks/, features/, wsipatch_plans/.",
        ),
        click.option(
            "--low-memory",
            is_flag=True,
            help="Forward wsipatch's low-memory mode (less RAM, more I/O).",
        ),
        click.option(
            "--batch-size",
            default=64,
            type=int,
            show_default=True,
            help="UNI inference batch size.",
        ),
        click.option(
            "--no-save-rois",
            is_flag=True,
            help="Delete ROI thumbnails after each slide instead of keeping them.",
        ),
        click.option(
            "--no-save-masks",
            is_flag=True,
            help="Delete tissue masks after each slide instead of keeping them.",
        ),
        click.option(
            "--no-save-features",
            is_flag=True,
            help="Skip writing per-slide features H5 to the workspace.",
        ),
        click.option(
            "--no-save-plans",
            is_flag=True,
            help="Skip writing wsipatch patch plans (.wsix) to the workspace.",
        ),
        click.option(
            "--tissue-threshold",
            default=0.5,
            type=float,
            show_default=True,
            help="Minimum tissue fraction for a patch to be processed.",
        ),
    ]
    for d in reversed(decorators):
        f = d(f)
    return f


def _resolve_paths(
    staining: Optional[str],
    binarization_model: Optional[str],
    uni_checkpoint: Optional[str],
    reinhard_config: Optional[str],
    banff_config: Optional[str],
) -> Dict[str, str]:
    """Merge staining defaults with explicit overrides.

    Either ``--staining`` is given (fills any unspecified path), or all
    four paths must be supplied explicitly. Mixed mode (e.g. ``--staining HE
    --banff-config /custom.yaml``) is also supported: the override wins
    per path.
    """
    if staining is None and any(
        p is None for p in (binarization_model, uni_checkpoint, reinhard_config, banff_config)
    ):
        raise click.UsageError(
            "Provide either --staining {HE,JONES,PAS} for configured asset paths, "
            "or all four of: --binarization-model, --uni-checkpoint, "
            "--reinhard-config, --banff-config."
        )
    if staining is not None:
        d = staining_paths(staining)
        defaults = dict(
            binarization_model=str(d.binarization_model),
            uni_checkpoint=str(d.uni_checkpoint),
            reinhard_config=str(d.reinhard_config),
            banff_config=str(d.banff_config),
        )
    else:
        defaults = {}

    overrides = dict(
        binarization_model=binarization_model,
        uni_checkpoint=uni_checkpoint,
        reinhard_config=reinhard_config,
        banff_config=banff_config,
    )
    return {k: (overrides[k] or defaults.get(k)) for k in overrides}


def _build_pipeline(
    *,
    staining,
    binarization_model,
    uni_checkpoint,
    reinhard_config,
    banff_config,
    workspace_dir,
    device,
    low_memory,
    batch_size,
    no_save_rois,
    no_save_masks,
    no_save_features,
    no_save_plans,
    tissue_threshold,
) -> BanffNETPipeline:
    paths = _resolve_paths(
        staining,
        binarization_model,
        uni_checkpoint,
        reinhard_config,
        banff_config,
    )
    return BanffNETPipeline.from_paths(
        workspace_dir=workspace_dir,
        device=device,
        low_memory=low_memory,
        batch_size=batch_size,
        save_rois=not no_save_rois,
        save_masks=not no_save_masks,
        save_features=not no_save_features,
        save_plans=not no_save_plans,
        tissue_threshold=tissue_threshold,
        **paths,
    )


def _build_predictor(
    *,
    staining: Optional[str],
    banff_config: Optional[str],
    device: str,
) -> BanffPredictor:
    """For ``banffnet features`` — only the predictor is needed."""
    if banff_config is None:
        if staining is None:
            raise click.UsageError("Provide either --staining or --banff-config.")
        banff_config = str(staining_paths(staining).banff_config)
    return BanffPredictor.from_yaml(banff_config, device=device)


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def _emit(writer: ResultsWriter, result: SlideResult, level: str) -> None:
    """Append one result to the writer (skipping empty slides)."""
    if result.num_patches == 0:
        log.warning("[%s] no patches produced — not written", result.slide_id)
        return
    if level == "slide":
        payload = result.slide_scores
    elif level == "roi":
        payload = result.roi_scores
    else:
        payload = result.patch_scores
    writer.add(result.slide_id, payload)


def _run_slide_safe(
    pipeline: BanffNETPipeline,
    slide_path: str,
    ann: Optional[str],
    level: str,
    writer: ResultsWriter,
    failures: List[Dict],
) -> None:
    """Process one slide; on error record the failure and continue."""
    slide_id = Path(slide_path).stem
    try:
        result = pipeline.predict_slide(slide_path, ann, level=level)
        _emit(writer, result, level)
    except Exception as e:  # noqa: BLE001
        log.error(
            "[%s] slide failed — skipping: %s: %s",
            slide_id,
            type(e).__name__,
            e,
            exc_info=True,
        )
        failures.append(
            {
                "slide_id": slide_id,
                "slide_path": slide_path,
                "annotation_path": ann or "",
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": tb.format_exc(),
            }
        )


def _write_failures(failures: List[Dict], output: str) -> None:
    """Write a CSV of failed slides next to the main output, if any failed."""
    if not failures:
        return
    out = Path(output)
    fail_path = out / "failed.csv" if not out.suffix else out.with_name(out.stem + "_failed.csv")
    fieldnames = [
        "slide_id",
        "slide_path",
        "annotation_path",
        "error_type",
        "error_message",
        "traceback",
    ]
    with open(fail_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(failures)
    log.warning("failure log written to %s", fail_path)


@click.group("banffnet", context_settings={"show_default": True})
def cli():
    """End-to-end BANFF score prediction from whole-slide images."""


@cli.command("slide")
@click.option(
    "--slide-path", required=True, type=click.Path(exists=True), help="Path to a single WSI file."
)
@click.option(
    "--annotation-path",
    default=None,
    type=click.Path(),
    help="ASAP XML annotation; omit to process the entire WSI.",
)
@_pipeline_options
@_level_and_output_options
def slide_cmd(slide_path, annotation_path, output, level, log_level, **pipeline_kwargs):
    """Run the full pipeline on a single WSI."""
    _setup_logging(log_level)
    try:
        with _build_pipeline(**pipeline_kwargs) as pipeline:
            result = pipeline.predict_slide(slide_path, annotation_path, level=level)
            with ResultsWriter(output, level=level, attribute_names=pipeline.attribute_names) as w:
                _emit(w, result, level)
    except KeyboardInterrupt:
        log.info("interrupted")
        sys.exit(1)
    except Exception as e:  # noqa: BLE001
        log.error("pipeline failed: %s", e, exc_info=True)
        sys.exit(1)


@cli.command("dir")
@click.option(
    "--biopsy-dir", required=True, type=click.Path(exists=True), help="Directory of WSIs."
)
@click.option(
    "--annotations-dir",
    default=None,
    type=click.Path(exists=True),
    help="Directory of ASAP XML files (matched by stem).",
)
@click.option(
    "--biopsy-ext", default="tiff", show_default=True, help="Extension to glob in --biopsy-dir."
)
@_pipeline_options
@_level_and_output_options
def dir_cmd(biopsy_dir, annotations_dir, biopsy_ext, output, level, log_level, **pipeline_kwargs):
    """Run the full pipeline on every WSI in a directory."""
    _setup_logging(log_level)
    slides = sorted(Path(biopsy_dir).glob(f"*.{biopsy_ext}"))
    if not slides:
        log.error("no .%s files in %s", biopsy_ext, biopsy_dir)
        sys.exit(1)

    failures: List[Dict] = []
    exit_code = 0
    try:
        with (
            _build_pipeline(**pipeline_kwargs) as pipeline,
            ResultsWriter(output, level=level, attribute_names=pipeline.attribute_names) as w,
        ):
            for slide in tqdm.tqdm(slides, desc="Processing slides"):
                ann = None
                if annotations_dir is not None:
                    candidate = Path(annotations_dir) / f"{slide.stem}.xml"
                    if candidate.exists():
                        ann = str(candidate)
                _run_slide_safe(pipeline, str(slide), ann, level, w, failures)
    except KeyboardInterrupt:
        log.info("interrupted")
        exit_code = 1
    except Exception as e:  # noqa: BLE001
        log.error("pipeline setup failed: %s", e, exc_info=True)
        exit_code = 1
    _write_failures(failures, output)
    if failures:
        log.warning("%d/%d slides failed", len(failures), len(slides))
        exit_code = 1
    sys.exit(exit_code)


REQUIRED_LIST_COLUMNS = ("biopsy_path", "annotation_path", "staining")


def _read_list_csv(path: Path) -> List[Tuple[str, Optional[str], str]]:
    """Parse the list CSV. Returns rows of (slide, annotation_or_None, staining)."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED_LIST_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise click.UsageError(
                f"CSV is missing required columns: {missing}. "
                f"Expected: {list(REQUIRED_LIST_COLUMNS)}."
            )
        rows: List[Tuple[str, Optional[str], str]] = []
        for r in reader:
            ann = r["annotation_path"].strip() or None
            rows.append((r["biopsy_path"].strip(), ann, r["staining"].strip().upper()))
    return rows


@cli.command("list")
@click.option(
    "--csv",
    "csv_path",
    required=True,
    type=click.Path(exists=True),
    help="CSV with columns: biopsy_path, annotation_path, staining.",
)
@click.option(
    "--workspace-dir",
    required=True,
    type=click.Path(),
    help="Where intermediate artifacts go (rois/, masks/, features/, wsipatch_plans/).",
)
@click.option("--low-memory", is_flag=True)
@click.option("--batch-size", default=64, type=int, show_default=True)
@click.option("--no-save-rois", is_flag=True, help="Delete ROI thumbnails after each slide.")
@click.option("--no-save-masks", is_flag=True, help="Delete tissue masks after each slide.")
@click.option("--no-save-features", is_flag=True, help="Skip writing per-slide features H5.")
@click.option("--no-save-plans", is_flag=True, help="Skip writing wsipatch patch plans (.wsix).")
@_level_and_output_options
def list_cmd(
    csv_path,
    workspace_dir,
    low_memory,
    batch_size,
    no_save_rois,
    no_save_masks,
    no_save_features,
    no_save_plans,
    output,
    level,
    log_level,
    device,
):
    """Run the full pipeline on a CSV of cases (mixed stainings allowed)."""
    _setup_logging(log_level)
    rows = _read_list_csv(Path(csv_path))
    if not rows:
        log.error("CSV is empty")
        sys.exit(1)

    # Group by staining for ordered processing — keeps one pipeline hot at a time.
    rows.sort(key=lambda r: r[2])

    failures: List[Dict] = []
    exit_code = 0
    try:
        with staining_cache(
            workspace_dir,
            device=device,
            low_memory=low_memory,
            batch_size=batch_size,
            save_rois=not no_save_rois,
            save_masks=not no_save_masks,
            save_features=not no_save_features,
            save_plans=not no_save_plans,
        ) as get:
            # Open the writer with the attribute set of the *first* staining we'll
            # encounter — all stainings are assumed to share the same attribute list.
            first_pipeline = get(rows[0][2])
            attribute_names = first_pipeline.attribute_names
            with ResultsWriter(output, level=level, attribute_names=attribute_names) as w:
                for slide, ann, staining in tqdm.tqdm(rows, desc="Processing cases"):
                    pipeline = get(staining)
                    if pipeline.attribute_names != attribute_names:
                        log.warning(
                            "%s has a different attribute set; writing what's common",
                            staining,
                        )
                    _run_slide_safe(pipeline, slide, ann, level, w, failures)
    except KeyboardInterrupt:
        log.info("interrupted")
        exit_code = 1
    except Exception as e:  # noqa: BLE001
        log.error("pipeline setup failed: %s", e, exc_info=True)
        exit_code = 1
    _write_failures(failures, output)
    if failures:
        log.warning("%d/%d cases failed", len(failures), len(rows))
        exit_code = 1
    sys.exit(exit_code)


@cli.command("features")
@click.option(
    "--features-dir",
    required=True,
    type=click.Path(exists=True),
    help="Directory containing *.h5 files with a 'features' dataset.",
)
@click.option(
    "--files-list",
    default=None,
    type=click.Path(),
    help="Optional newline-delimited file with H5 filenames to process.",
)
@click.option(
    "--staining",
    type=click.Choice(list(STAININGS), case_sensitive=False),
    default=None,
    help="Use the configured BANFF config for this staining.",
)
@click.option(
    "--banff-config",
    type=click.Path(exists=True),
    default=None,
    help="Override the staining default.",
)
@_level_and_output_options
def features_cmd(
    features_dir, files_list, staining, banff_config, output, level, log_level, device
):
    """Run only the prediction stage on an existing features H5 directory."""
    _setup_logging(log_level)
    failures: List[Dict] = []
    exit_code = 0
    try:
        predictor = _build_predictor(
            staining=staining,
            banff_config=banff_config,
            device=device,
        )
        file_list = None
        if files_list:
            file_list = [
                ln.strip() for ln in Path(files_list).read_text().splitlines() if ln.strip()
            ]

        with ResultsWriter(output, level=level, attribute_names=predictor.attribute_names) as w:
            for slide_id, scores in tqdm.tqdm(
                predictor.predict_features_directory(
                    features_dir,
                    level=level,
                    file_list=file_list,
                ),
                desc="Predicting",
            ):
                try:
                    w.add(slide_id, scores)
                except Exception as e:  # noqa: BLE001
                    log.error("[%s] failed: %s: %s", slide_id, type(e).__name__, e, exc_info=True)
                    failures.append(
                        {
                            "slide_id": slide_id,
                            "slide_path": str(Path(features_dir) / f"{slide_id}.h5"),
                            "annotation_path": "",
                            "error_type": type(e).__name__,
                            "error_message": str(e),
                            "traceback": tb.format_exc(),
                        }
                    )
    except KeyboardInterrupt:
        log.info("interrupted")
        exit_code = 1
    except Exception as e:  # noqa: BLE001
        log.error("pipeline failed: %s", e, exc_info=True)
        exit_code = 1
    _write_failures(failures, output)
    if failures:
        log.warning("%d feature file(s) failed", len(failures))
        exit_code = 1
    sys.exit(exit_code)


def main() -> int:
    cli()
    return 0


if __name__ == "__main__":
    sys.exit(main())
