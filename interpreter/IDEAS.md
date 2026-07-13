# Future Ideas — Interpreter Sub-Project

Ideas parked during design of the map comparison benchmark, deferred until the
binary-only version of that benchmark is fully operational.

## 1. Local SNIC-like patch generation

Build a processing chain upstream of the current pipeline that downloads
imagery directly from GEE and generates SNIC-like patches locally, with
patch-size parameters we control ourselves — rather than depending on
pre-generated rasters in `downloaded_images/`. Would remove the current
dependency on external SNIC patch generation and let us sweep patch size as
an experiment variable.

## 2. Spatial pattern characterization of validation landscapes

Build tools to characterize the spatial pattern statistics (e.g. patch size
distribution, shape complexity, edge density, spatial autocorrelation) of the
validation landscapes. Use this to test how patch size and spatial structure
affect the effectiveness of the sampling-based methods (A–D) relative to
pixel-level ground truth — i.e. does map comparison performance depend on the
spatial character of the landscape being mapped, not just sensor/interpreter
noise levels?

## 3. Multi-class extension

Once the binary version of the map comparison benchmark is operational,
revisit the multi-class (per-forest-type) version originally sketched:

- Perfect-interpreter reference as `multiclass_truth = disturbed * (forest_type_raster + 1)`
  → 0 (undisturbed), 1 (type1 disturbed), 2 (type2 disturbed)
- Multi-class classifier: sweep type-specific thresholds independently over
  type1 and type2 pixels (requires `forest_type_raster`)
- Multi-class pixel-level scoring: N×N confusion matrix, per-class IoU/precision/recall/F1
- Multi-class window sampling: extend `window_sample_A–D` (and
  `_extract_windows`) beyond the current binary-only tie-break/confusion
  logic; resolve whether window D reports 3 per-class proportions or
  collapses to disturbed-vs-not
