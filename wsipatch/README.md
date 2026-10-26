# wsipatch

Patch planning and reading for whole-slide images, backed by Roaring
bitmaps. Decides which patches to extract from a WSI (tissue-filtered
optionally), stores the plan compactly, and reads patches back on demand
with two memory-vs-throughput modes.

## Layout

```
src/wsipatch/
├── types.py         # TargetSpec, ROISpec, ROIGrid, SlideIndex
├── annotations.py   # ASAP XML + asap_to_rois adapter
├── planner.py       # level selection + tissue scan + plan_slide_* + plan_directory_*
├── io.py            # save_index / load_index (.wsix and .json)
├── reader.py        # PatchReader (the main read API) + PatchSampler + PatchLocation
└── cli.py           # CLI: plan-slide, plan-dir, materialise, inspect, convert
```

Planning reads slide metadata and optional tissue masks without extracting
patches. The plan stores the geometry needed to read them through `PatchReader`.

## Quick start

### Plan

```python
from wsipatch import plan_slide

idx = plan_slide(
    "slide.tiff",
    patch_size=224,
    mpp=0.5,
    masks=["slide_tissue.png"],
    tissue_threshold=0.25,
)
```

Or via CLI for a whole directory:

```bash
wsipatch plan-dir /data/slides /data/plans \
    -t default:224:0.5 \
    --annotations-dir /data/annotations \
    --mask-dir /data/masks
```

### Save & load

```python
from wsipatch import save_index, load_index

save_index(idx, "slide.wsix")  # binary, compact
save_index(idx, "slide.json")  # human-readable
idx = load_index("slide.wsix")  # format auto-detected
```

### Read

```python
from wsipatch import PatchReader

with PatchReader("slide.wsix", slide_path="slide.tiff", low_memory=False) as r:
    print(f"{r.num_patches} patches @ {r.patch_size}px")
    for batch, meta in r.iter_batches(batch_size=64):
        # batch: (B, ps, ps, 3) uint8 — owned arrays, safe to keep
        # meta:  {"coords": (B, 2) int32,        # (x, y) in rescaled-ROI frame
        #         "roi_index": (B,) int32,       # positional
        #         "roi_id": (B,) str,             # name, or str(roi_index) if unnamed
        #         "_index": (B,) int64}          # global patch index
        ...
```

`low_memory=False` (default) reads each ROI whole, rescales once,
slices patches: fastest, holds the rescaled ROI in RAM.
`low_memory=True` does one `read_region` per patch: peak RAM bounded by
the batch buffer, but slower because of many small reads.

Random access (always uses per-patch reads):

```python
patch, loc = reader.read_patch(k=12345)
```

Materialise everything to a single HDF5 (streamed, bounded memory in
both modes):

```python
reader.materialise("patches.h5", batch_size=64, compression="gzip")
```

Or from the CLI:

```bash
wsipatch materialise slide.wsix patches.h5 --target default --low-memory
```

### Plug into a feature extractor

```python
from wsipatch import PatchReader
from features_extraction import FeatureExtractor, FeatureH5Writer

ext = FeatureExtractor.from_files(
    checkpoint_path="uni_vit_large.pt",
    reinhard_config_path="reinhard_he.yaml",
    device="cuda",
    use_amp=True,
)

with PatchReader("slide.wsix", slide_path="slide.tiff") as r, FeatureH5Writer("features.h5") as w:
    w.set_attrs(model="UNI", slide_id=r.index.slide_id)
    for batch, meta in r.iter_batches(batch_size=64):
        w.append(ext.extract_batch(batch), meta)
```

## ROI identity

Every ROI carries an optional human-readable name — the ASAP annotation's
`Name` attribute (e.g. `<Annotation Name="core_01" ...>`). When present,
it's used as the ROI's identifier everywhere one is needed; when absent,
the ROI's positional index is used instead. `resolve_roi_id(name,
fallback_index)` is the single function that makes this decision, and
`ROIGrid.roi_label` / `PatchLocation.roi_id` / the `roi_id` key in
`PatchReader`'s per-patch metadata all resolve through it — so a named ROI
always shows up under the same identifier in ROI/mask filenames
(binarization), plan metadata (wsipatch), and per-ROI prediction output
(banffnet).

## Public API

| Group        | Names                                                       |
|--------------|-------------------------------------------------------------|
| Types        | `TargetSpec`, `ROISpec`, `ROIGrid`, `SlideIndex`            |
| ROI identity | `resolve_roi_id`, `sanitize_roi_id`                          |
| Planning     | `plan_slide`, `plan_slide_multi`, `plan_directory`, `plan_directory_multi`, `build_slide_index`, `build_slide_index_from_metadata` |
| IO           | `save_index`, `load_index` (auto), `save_wsix`/`load_wsix`/`save_json`/`load_json` (forced) |
| Reading      | `PatchReader`, `PatchSampler`, `PatchLocation`              |
| Annotations  | `Annotation`, `ASAPAnnotations`, `AnnotationParseError`, `asap_to_rois`, `load_asap_rois` |

## File formats

`.wsix` (binary, recommended):

```
MAGIC      b"WSIX"               (4 bytes)
JSON_LEN   uint32 little-endian  (4 bytes)
JSON_BLOB  utf-8 metadata        (JSON_LEN bytes)
for each grid in target-major, roi-minor order:
    BM_LEN  uint32 little-endian (4 bytes)
    BITMAP  pyroaring serialised (BM_LEN bytes)
```

`.json` (human-readable): the same metadata plus a `valid_indices`
list per grid, and optionally `valid_coords` for grep-ability.

Both formats round-trip losslessly.