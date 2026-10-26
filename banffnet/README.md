# banffnet

End-to-end BANFF score prediction for kidney biopsies. Combines the
three processing packages — `binarization`, `wsipatch`, `features_extraction`
— with the BANFF MIL predictors on top.

```
WSI + ASAP XML
      │
      ▼  binarization        tissue masks (per ROI)
      ▼  wsipatch.plan_slide  patch grid (mask-aware)
      ▼  features_extraction  UNI ViT-L/16 + Reinhard
      ▼  banffnet.predictor   noisy-OR / gated-ratio MIL heads
      ▼
slide-level scores  (CSV row per slide)
ROI-level scores    (CSV row per slide and ROI)
patch-level scores  (H5 per slide, [N_patches, N_attributes])
```

## Layout

Modules under `src/banffnet/`:

| Module        | Contents                                                  |
| ------------- | --------------------------------------------------------- |
| `defaults.py` | `STAININGS`, `DATA_ROOT`, `staining_paths()`              |
| `models.py`   | `LinearNoisyOR`, `LinearGatedRatio`, `MODELS` (2 entries) |
| `predictor.py`| `BanffPredictor` — slide / ROI / patch inference        |
| `pipeline.py` | `BanffNETPipeline`, `SlideResult`, `staining_cache`       |
| `io.py`       | `ResultsWriter` — CSV / H5 output                         |
| `cli.py`      | `banffnet` group with `slide` / `dir` / `list` / `features` |
| `__init__.py` | Public API re-exports                                     |

## Install

```bash
pip install -e ./wsipatch -e ./binarization -e ./features_extraction -e ./banffnet
```

Run from the repository root. See [INSTALL.md](../INSTALL.md) for CUDA setup.

## Model assets

Checkpoints and staining configurations are supplied separately. Set
`BANFFNET_DATA_ROOT` to their root directory; the default is `data/` relative to
the working directory. See [INSTALL.md](../INSTALL.md) for the expected layout.
Individual path options override staining defaults.

## CLI

Each pipeline subcommand accepts **either** `--staining {HE,JONES,PAS}`
(resolves configured model and staining assets) **or** all four of `--binarization-model`,
`--uni-checkpoint`, `--reinhard-config`, `--banff-config` (full
override). Any individual flag overrides its staining default — useful
for swapping just the BANFF config while keeping the rest of the
defaults.

Every subcommand also accepts `--level {slide,roi,patch}`:

* `--level slide` → one bag per slide (every ROI pooled together);
  `--output` is a **CSV file path** with one row per slide.
* `--level roi` → one bag per ROI (each biopsy core/fragment scored
  independently); `--output` is a **CSV file path** with one row per
  `(slide_id, roi_id)`.
* `--level patch` → `--output` is a **directory** of per-slide H5 files.

### 1. Single slide

```bash
banffnet slide \
    --slide-path /data/case_01.tiff \
    --annotation-path /data/case_01.xml \
    --staining HE \
    --workspace-dir ./workspace \
    --output /out/scores.csv \
    --level slide
```

### 2. Directory

```bash
banffnet dir \
    --biopsy-dir /data/slides \
    --annotations-dir /data/annotations \
    --biopsy-ext tiff \
    --staining JONES \
    --workspace-dir ./workspace \
    --output /out/scores.csv \
    --level slide
```

### 3. CSV list (multi-staining)

CSV columns: `biopsy_path,annotation_path,staining` (annotation may be
empty).

```bash
banffnet list \
    --csv /data/cases.csv \
    --workspace-dir ./workspace \
    --output /out/patch_scores \
    --level patch
```

Multi-staining is handled by sorting the CSV by staining and holding
one pipeline on the GPU at a time.

### 4. Per-ROI scores (each biopsy core scored independently)

```bash
banffnet slide \
    --slide-path /data/case_01.tiff \
    --annotation-path /data/case_01.xml \
    --staining HE \
    --workspace-dir ./workspace \
    --output /out/roi_scores.csv \
    --level roi
```

Each row of `roi_scores.csv` is `slide_id, roi_id, <attribute columns>`
— one row per ROI that produced at least one patch, rather than one
pooled row per slide. `roi_id` is the ROI's annotation `Name` (e.g.
`core_01`) when the XML provides one, otherwise its positional index.

### 5. Features-only inference

For when binarization → patches → features has already been run.

```bash
banffnet features \
    --features-dir /data/features \
    --staining HE \
    --output /out/scores.csv \
    --level slide
```

Or with an explicit files list and a custom BANFF config:

```bash
banffnet features \
    --features-dir /data/features \
    --files-list /data/include.txt \
    --banff-config /custom/banff.yaml \
    --output /out/patch \
    --level patch
```

## Programmatic API

`BanffNETPipeline` is the main entry point for embedding the pipeline
into Python code.

### End-to-end on a single slide

```python
from banffnet import BanffNETPipeline

with BanffNETPipeline.from_staining("HE", workspace_dir="./workspace") as p:
    result = p.predict_slide("/data/case.tiff", "/data/case.xml")
    print(result.slide_scores)  # {'i': 0.83, 't': 0.21, ...}
    print(result.features_path)  # ./workspace/features/case.h5
    print(result.num_patches)
```

### Per-ROI output

```python
with BanffNETPipeline.from_staining("HE", workspace_dir="./workspace") as p:
    result = p.predict_slide("/data/case.tiff", "/data/case.xml", level="roi")
    print(result.roi_scores)  # {'core_01': {'i': 0.83, ...}, '1': {'i': 0.40, ...}}
```

### Patch-level output

```python
with BanffNETPipeline.from_staining("PAS", workspace_dir="./workspace") as p:
    result = p.predict_slide("/data/case.tiff", level="patch")
    print(result.patch_scores.shape)  # (N_patches, N_attributes)
```

### Manual path overrides

```python
pipeline = BanffNETPipeline.from_paths(
    binarization_model="/path/to/models/mobilenet.pt",
    uni_checkpoint="/path/to/models/uni.pt",
    reinhard_config="/path/to/configs/he_reinhard.yaml",
    banff_config="/path/to/configs/he_banff.yaml",
    workspace_dir="./workspace",
    device="cuda:0",
    low_memory=True,
    batch_size=128,
)
```

### Features-only (no full pipeline)

```python
from banffnet import BanffPredictor

predictor = BanffPredictor.from_yaml(
    "/path/to/configs/he_banff.yaml",
    device="cuda:0",
)
for slide_id, scores in predictor.predict_features_directory(
    "/data/features",
    level="slide",
):
    print(slide_id, scores)
```

### Mixed-staining batches in Python

```python
from banffnet import staining_cache

with staining_cache("./workspace", device="cuda:0") as get:
    for slide, ann, staining in rows:
        pipeline = get(staining)  # reuses the open pipeline if it matches
        result = pipeline.predict_slide(slide, ann)
```

## Public API

Everything below is importable from the top-level package
(`from banffnet import X`).

```text
# Defaults
STAININGS, DATA_ROOT, DEFAULT_PATCH_SIZE, DEFAULT_MPP
StainingPaths, staining_paths

# Models
LinearNoisyOR, LinearGatedRatio, MODELS, logistic_sum_model

# Predictor
BanffPredictor

# Pipeline
BanffNETPipeline, SlideResult, staining_cache

# I/O
ResultsWriter
```
