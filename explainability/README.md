# explainability

Generic Attention Explainability (GAE, Chefer et al., ICCV 2021) lesion saliency
maps for BanffNET, integrated with the rest of the WSI pipeline.

The relevance map is produced from the **full architecture** — the frozen UNI
ViT backbone plus the BanffNET lesion head — with the head **sigmoid removed**
and everything in **float32** for a numerically stable backward pass (per the
paper). During the GAE backward, the raw self-attention matrix `A` of each
transformer block is multiplied by its class-specific gradient, the positive
part is averaged across heads, and relevance is rolled up as `R ← R + ĀR` from
the identity. The CLS→patch row is bilinearly upsampled to patch resolution.

Whole-slide (slide-level) maps sample patches at **stride 64**, weight each
patch's saliency by its BanffNET probability, average across the biopsy, and
overlay on the ROI at transparency **0.3**.

## Pipeline components

| Concern | Package |
| --- | --- |
| Patch planning / ROI geometry / reading | `wsipatch` |
| Preprocessing, UNI load, features H5 | `features_extraction` |
| MIL heads (NIP/GIP), YAML config, staining defaults | `banffnet` |
| Tissue masks | `binarization` |

The package implements attention instrumentation (`backbone.py`), GAE relevance
(`gae.py`), lesion saliency targets (`model.py`), and slide aggregation and rendering.

## Installation

Install through the root scripts or follow [INSTALL.md](../INSTALL.md). Model
weights and staining configurations are supplied separately.

## CLI

```bash
# Whole-slide overlay for one lesion, straight from a slide
run-explainability slide-level \
    --staining JONES --lesion ptc \
    --slide-path case.tiff --annotation-path case.xml \
    --workspace-dir ./ws --output-dir ./out

# From an existing wsipatch plan, every lesion, automatic per-case threshold
run-explainability slide-level \
    --uni-checkpoint uni.pt --banff-config banff.yaml --reinhard-config reinhard.yaml \
    --plan case.wsix --lesion all --threshold auto --enhance \
    --output-dir ./out

# Per-patch saliency, reusing already-extracted patches + features
run-explainability patch-level \
    --staining JONES --lesion g \
    --patches-h5 case_patches.h5 --features-h5 case_features.h5 \
    --output-dir ./out
```

`--threshold` accepts `paper` (default, per-variant fixed values), `auto`
(Otsu), `pNN` (percentile, e.g. `p90`), or a bare float. Outputs per lesion:
`*_overlay.png`, `*_saliency.png` (standalone colormap), and `*_relevance.npy`
(raw float map). `--enhance` adds smoothing / small-component removal /
morphological closing to refine coarse saliency regions.
