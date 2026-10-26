# binarization

ROI extraction and tissue segmentation for whole-slide images. Takes
WSIs (+ optional ASAP annotations), emits cropped ROI thumbnails and
binary tissue masks plus a CSV QC manifest.

## Layout

Modules under `src/binarization/`:

| Module           | Contents                                                      |
| ---------------- | ------------------------------------------------------------- |
| `config.py`      | `BinarizationConfig`, `LogConfig`, image-norm & QC constants  |
| `extraction.py`  | `SlideProcessor`, `ROIExtractor`, `QCRecord`, bbox/clip types |
| `segmentation.py`| `load_segmentation_model`, `SegmentationPipeline`, CLAHE      |
| `pipeline.py`    | `BinarizationPipeline`, `ProcessingStats`                     |
| `io.py`          | `FileManager`, `CSVWriter`, discovery + `setup_logging`       |
| `cli.py`         | `tissue-binarize` Click group (`slide` / `dir` subcommands)   |
| `__init__.py`    | Public API re-exports                                         |

## Install

```bash
pip install -e ./wsipatch -e ./binarization
```

Run from the repository root. ASAP XML parsing is provided by `wsipatch`.
See [INSTALL.md](../INSTALL.md) for CUDA setup. Segmentation checkpoints are
supplied separately and passed with `--model-path`.

## CLI

```bash
# Process one WSI
tissue-binarize slide \
    --slide-path /data/slides/case_01.tiff \
    --annotation-path /data/annotations/case_01.xml \
    --roi-dir /out/rois \
    --mask-dir /out/masks \
    --model-path /models/mobilenet_v2.pt

# Process every WSI in a directory
tissue-binarize dir \
    --biopsy-dir /data/slides \
    --annotations-dir /data/annotations \
    --roi-dir /out/rois \
    --mask-dir /out/masks \
    --model-path /models/mobilenet_v2.pt
```

Pass `--extract-only` to skip segmentation; `--model-path` can then be omitted.
Run either subcommand with `--help` for the complete option list.

## Programmatic API

`BinarizationPipeline` is the main entry point. Designed to be embedded
in a larger orchestrator (e.g. chained with `wsipatch.plan_slide` and a
feature extractor).

### Per slide, end-to-end

```python
from binarization import BinarizationPipeline, BinarizationConfig

config = BinarizationConfig(
    roi_dir="/data/rois",
    mask_dir="/data/masks",
    model_path="/models/mobilenet_v2.pt",
)

with BinarizationPipeline(config) as pipeline:
    roi_paths, mask_paths = pipeline.process_slide(
        "/data/slides/case_01.tiff",
        "/data/annotations/case_01.xml",  # optional
    )
```

`process_slide` returns the new files written; existing outputs on disk
are skipped silently (and recorded in QC). The context manager frees
GPU memory deterministically on exit.

### Phase-by-phase across many slides

```python
with BinarizationPipeline(config) as pipeline:
    # Phase 1: extract every slide first (no model loaded yet).
    for slide, ann in zip(slides, annotations):
        pipeline.extract_slide(slide, ann)

    pipeline.write_qc_csv("/out/rois/roi_qc.csv")

    # Phase 2: model loads lazily on the first segment_paths call.
    pending = pipeline.get_pending_segmentations()
    pipeline.segment_paths(pending)
```

### Batch CLI flow from Python

```python
from binarization import BinarizationPipeline, get_biopsy_list

biopsy_list = get_biopsy_list(
    "/data/slides",
    "/data/annotations",
    biopsy_ext="tiff",
    use_entire_wsi_if_no_annotation=True,
)

pipeline = BinarizationPipeline.from_paths(
    roi_dir="/out/rois",
    mask_dir="/out/masks",
    model_path="/models/mobilenet_v2.pt",
)
pipeline.run(biopsy_list)
```

### Chaining with downstream packages

```python
from binarization import BinarizationPipeline
from wsipatch import plan_slide, PatchReader, load_asap_rois

with BinarizationPipeline(binarization_cfg) as binarizer:
    roi_paths, mask_paths = binarizer.process_slide(slide_path, ann_path)

# Keep the annotation ROI order aligned with the returned masks.
plan = plan_slide(
    slide_path,
    patch_size=224,
    mpp=0.5,
    rois=load_asap_rois(ann_path),
    masks=mask_paths,
)
with PatchReader(plan) as reader:
    for batch, meta in reader.iter_batches():
        ...  # features
```

## Public API

Everything below is importable from the top-level package
(`from binarization import X`).

```text
# Config + constants
BinarizationConfig, LogConfig
IMAGE_NORM_MEAN, IMAGE_NORM_STD, QC_FIELDNAMES

# Extraction
BoundingBox, ClippingResult, QCRecord
ROIExtractor, SlideProcessor, SlideProcessorError

# Segmentation
SegmentationPipeline, ModelLoadError
load_segmentation_model, select_device, enhance_contrast_clahe

# Pipeline
BinarizationPipeline, ProcessingStats

# I/O
FileManager, CSVWriter
get_biopsy_list, ensure_directories, setup_logging
```

## ROI identity

ROI/mask filenames are `{slide}_roi_{roi_id}.png` / `{slide}_roi_{roi_id}_mask.png`.
`roi_id` is the source annotation's `Name` attribute when present (e.g.
`<Annotation Name="core_01" ...>` → `case01_roi_core_01_mask.png`),
sanitized for filesystem safety; otherwise it's the ROI's positional
index. This resolution (`wsipatch.resolve_roi_id`) is
shared with wsipatch and banffnet, so a named ROI's mask, patch-plan
metadata, and per-ROI prediction output all use the same identifier.
