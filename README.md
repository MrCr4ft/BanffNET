# BanffNET

This repository contains the inference pipeline accompanying
[BanffNET, a Deep Learning System for Comprehensive Histological Lesion
Quantification in Kidney Transplant Biopsies](https://www.medrxiv.org/content/10.64898/2026.08.28.26360029v1).

BanffNET predicts BANFF lesion scores from kidney transplant biopsy whole-slide images.
The pipeline extracts tissue regions, plans patches, computes UNI ViT features,
and applies multiple-instance learning heads. Predictions are available per
slide, per region of interest (ROI), or per patch.

| Package | Purpose | Command |
| --- | --- | --- |
| [binarization](binarization/README.md) | ROI extraction and tissue segmentation | `tissue-binarize` |
| [wsipatch](wsipatch/README.md) | Patch planning, reading, and HDF5 export | `wsipatch` |
| [features-extraction](features_extraction/README.md) | UNI feature extraction and Reinhard stain normalization | `features-extract` |
| [banffnet](banffnet/README.md) | Slide, ROI, and patch predictions | `banffnet` |
| [explainability](explainability/README.md) | GAE lesion saliency maps | `run-explainability` |

## Installation

Python 3.10 or later is required. See [INSTALL.md](INSTALL.md) for the conda
installers, CUDA settings, and manual installation.

```bash
bash install.sh
conda activate banffnet
banffnet --help
```

On Windows, run `./install.ps1` from PowerShell.

## Model assets

Model checkpoints and staining configurations are not included in this source
release. Set `BANFFNET_DATA_ROOT` to the asset directory described in
[INSTALL.md](INSTALL.md), or pass explicit paths through the CLI or Python API.
The supported stainings are `HE`, `JONES`, and `PAS`.

```bash
banffnet slide \
    --slide-path /path/to/slide.tiff \
    --annotation-path /path/to/slide.xml \
    --staining HE \
    --workspace-dir ./workspace \
    --output ./scores.csv
```

Annotations are optional when processing an entire slide as one ROI. Package
READMEs document batch processing, Python APIs, output formats, and saliency
options.
