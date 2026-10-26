# Installation

The installers create a conda environment containing `wsipatch`, `binarization`,
`features-extraction`, `banffnet`, and `explainability`. They install the CUDA
PyTorch stack before the local packages and check that OpenSlide is available.

## Prerequisites

- Conda, such as [Miniforge](https://github.com/conda-forge/miniforge).
- A CUDA-capable GPU and compatible driver for GPU inference. CPU inference
  is also supported through the device options.
- OpenSlide's native library. The installers use an existing installation or
  install it through conda-forge or `openslide-bin`.

## Quick start

Linux:

```bash
bash install.sh
conda activate banffnet
banffnet --help
```

Windows PowerShell:

```powershell
./install.ps1
conda activate banffnet
banffnet --help
```

The installers use Python 3.10, `torch==2.11.0+cu128`,
`torchvision==0.26.0+cu128`, and `xformers==0.0.35`. The Windows installer
treats xformers as optional.

## Installer settings

Set these environment variables before running an installer to override its
defaults:

| Variable | Default | Purpose |
| --- | --- | --- |
| `BANFFNET_ENV` | `banffnet` | Conda environment name |
| `BANFFNET_PYTHON` | `3.10` | Python version |
| `BANFFNET_CUDA` | `cu128` | PyTorch CUDA build tag |
| `BANFFNET_TORCH` | `2.11.0` | PyTorch version |
| `BANFFNET_TORCHVISION` | `0.26.0` | torchvision version |
| `BANFFNET_XFORMERS` | `0.0.35` | xformers version |
| `BANFFNET_DATA_ROOT` | Prompted when interactive | Model and configuration directory |

If changing CUDA or PyTorch versions, select compatible torch, torchvision,
and xformers builds together.

```bash
BANFFNET_ENV=banffnet-release BANFFNET_DATA_ROOT=/path/to/assets bash install.sh
```

## Model and configuration assets

Checkpoints and staining configurations are not included. Provide these files
under `BANFFNET_DATA_ROOT`:

```text
assets/
├── models/
│   ├── binarization/mobilenet_v2.pt
│   └── uni/uni_vit.pt
└── stainings/
    ├── HE/{banff_config.yaml, reinhard.yaml}
    ├── JONES/{banff_config.yaml, reinhard.yaml}
    └── PAS/{banff_config.yaml, reinhard.yaml}
```

Each `banff_config.yaml` supplies the lesion heads, their parameters, and their
checkpoint paths. Reinhard configurations supply the reference stain statistics.
See the [prediction](banffnet/README.md) and
[feature extraction](features_extraction/README.md) READMEs for their formats.

The installer saves `BANFFNET_DATA_ROOT` in the conda environment when supplied.
To change it later:

```bash
conda env config vars set BANFFNET_DATA_ROOT=/path/to/assets -n banffnet
```

Reactivate the environment to apply the change. Without an environment override,
the library resolves assets under `data/` relative to the working directory.
Individual CLI path options can override the staining defaults.

## Manual installation

Run from the repository root:

```bash
conda create -y -n banffnet python=3.10 pip wheel
conda activate banffnet

pip install --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.11.0+cu128 torchvision==0.26.0+cu128
pip install xformers==0.0.35
pip install -e ./wsipatch -e ./binarization -e ./features_extraction \
    -e ./banffnet -e ./explainability
```

If `python -c "import openslide"` fails because the native library is missing,
install `openslide` through conda-forge or `openslide-bin` through pip.

## Outputs

Pipeline runs write intermediates under `--workspace-dir`: `rois/`, `masks/`,
`features/`, and `wsipatch_plans/`. Disable retention with `--no-save-rois`,
`--no-save-masks`, `--no-save-features`, or `--no-save-plans`.

The root `.gitignore` excludes checkpoints, local asset directories, generated
pipeline data, environments, and editor settings from Git.
