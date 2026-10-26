# features-extraction

Batched ViT feature extraction for computational pathology, with an optional
Reinhard stain normalization that operates **directly on tensors** (no PIL
round-trip per patch).

## Install

```bash
pip install -e ./wsipatch -e ./features_extraction
```

Run from the repository root. See [INSTALL.md](../INSTALL.md) for CUDA setup.
UNI checkpoints and Reinhard configurations are supplied separately.

## CLI

The package installs one console script with three subcommands.

### 1. From an HDF5 of patches

```bash
features-extract from-h5 \
  --input patches.h5 \
  --patches-key patches \
  --extra-keys coords --extra-keys patch_size \
  --checkpoint /path/to/uni_vit_large.pt \
  --reinhard-config reinhard_he.yaml \
  --output features.h5 \
  --batch-size 128 --device cuda --amp
```

The patches dataset is auto-detected if `--patches-key` is omitted. Any
`--extra-keys` (repeatable) are sliced in lockstep and copied into the
`metadata/` group of the output file.

### 2. From a folder of images

```bash
features-extract from-folder \
  --folder /path/to/patches/ \
  --checkpoint /path/to/uni_vit_large.pt \
  --output features.h5 \
  --batch-size 64
```

Supports `.png .jpg .jpeg .tif .tiff .bmp`. File paths are stored as metadata.

### 3. From a wsipatch patching plan (`.wsix`)

```bash
features-extract from-plan \
  --plan slide.wsix \
  --slide-path slide.tif \
  --low-memory \
  --checkpoint /path/to/uni_vit_large.pt \
  --reinhard-config reinhard_he.yaml \
  --output features.h5
```

`--low-memory` selects per-patch reads through `wsipatch.PatchReader`.
Coordinates returned by the reader are copied through as metadata.

## Python API

The CLI is a thin wrapper around the library API. In a larger pipeline you'll
usually keep one `FeatureExtractor` alive across stages and feed it batches
yourself:

```python
from features_extraction import FeatureExtractor

ext = FeatureExtractor.from_files(
    checkpoint_path="uni_vit_large.pt",
    reinhard_config_path="reinhard_he.yaml",
    device="cuda",
    use_amp=True,
)

# `batch` can be:
#   - np.ndarray (B, H, W, 3) uint8 from your binarization/patcher
#   - torch.Tensor (B, 3, H, W) float in [0, 1]
features = ext.extract_batch(batch)  # -> (B, D) float32 ndarray
```

To stream a long source straight to disk:

```python
from features_extraction import FeatureH5Writer, iter_plan_patches

with FeatureH5Writer("features.h5") as w:
    w.set_attrs(model="UNI", reinhard=True)
    for batch, meta in iter_plan_patches("slide.wsix", "slide.tif", batch_size=64):
        w.append(ext.extract_batch(batch), meta)
```

## Reinhard on batched tensors

`features_extraction.Reinhard` is an `nn.Module` that accepts both
`(3, H, W)` and `(B, 3, H, W)` tensors in `[0, 1]`. Reference Lab statistics
are registered buffers, so `FeatureExtractor` moves them onto the selected device with the preprocessing
module. The standard and
modified variants are both vectorized — no Python loop over the batch
dimension.

Reinhard YAML configuration format:

```yaml
apply_modified_reinhard: false
l_mean: 72.7   # reference Lab means
a_mean: 18.4
b_mean: -10.3
l_std: 16.5    # reference Lab stds
a_std: 8.6
b_std: 7.2
```

## Output HDF5 layout

```
features.h5
├── features              # (N, D) float32, gzip-compressed
└── metadata/
    ├── _index            # (N,) int64, global row index
    └── ...               # any extra keys from the source iterator
```

The file is built incrementally with resizable datasets so memory stays
bounded regardless of the source size.