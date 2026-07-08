# Map Comparison Benchmark — Status & Recreate Guide

Binary-only (disturbed/not-disturbed) first pass of the map comparison
benchmark described in `CLAUDE.md` ("Map comparison benchmark" under
Architecture → Map Comparison, and under Pending Work). This doc captures
enough detail to recreate, continue, or hand off this work without replaying
the design conversation that produced it.

**Branch:** `map-comparison-benchmark` (not yet merged to `main` — this is a
`src/` change and needs a PR per the collaboration convention in `CLAUDE.md`).

**Status:** Working end-to-end. Ran the full notebook successfully; pixel-level
metrics degrade monotonically with sensor noise as expected (see "Verified
results" below). Still iterating — nothing here should be considered final.

## What was decided (scope)

The original CLAUDE.md sketch included a multi-class (per-forest-type)
variant. That was deliberately deferred for this first build — see
`interpreter/IDEAS.md` #3. Design decisions made for this binary-only pass:

1. Forest types (`type1`/`type2`) are kept in the landscape config, but
   disturbance scoring stays binary — no 3-class map or classifier yet.
2. Multiple sensors live in a JSON config in `interpreter/experiments/configs/`
   (not hardcoded), committed to git.
3. The "perfect interpreter" reference is built through the **standard**
   `InterpreterField` config mechanism (not a hand-rolled truth formula) —
   every noise-injecting parameter is set to a no-op value. See "Perfect
   interpreter config" below for exactly which values and why.
4. Multi-class classifier: deferred (`interpreter/IDEAS.md` #3).
5. Validation set: 20 scenes, disjoint from the 10 training scenes.
6. Pixel-level scoring: new, since nothing in `src/` did full-raster (as
   opposed to sampled) confusion/precision/recall/IoU before this.
7. Window sampling: reuses `window_sample_A/B/C/D` unmodified — no multi-class
   extension.
8. Meta-analysis: pixel-level metric deltas vs. sampling-based metric deltas
   between sensor pairs — first-pass/exploratory.

Reference convention: pixel-level scoring compares each sensor's binary map
against the **thresholded perfect-interpreter field** (`>= 50%`), not the raw
truth mask directly — this matches the convention already used everywhere
else in `src/` (`binary_stratified_sample`, `window_sample_A-D`), where
`ref_class` always comes from the interpreter field, never straight from
`base_landscape.disturbed`.

## Files

| File | Purpose |
|---|---|
| `src/map_comparison.py` | New module: full-raster pixel-level scoring (`pixel_confusion_matrix`, `pixel_level_metrics`, `pixel_level_metrics_for_stack/collection`, `pixel_agreement_between_maps`) + exploratory `map_comparison_meta_analysis()`. Free functions, not `LandscapeStackCollection` methods — kept separate since that class is already ~3600+ lines and this operates on full rasters rather than the sampling apparatus. |
| `interpreter/experiments/configs/map_comparison_baseline.json` | 4-sensor config (clean → high noise) + perfect-interpreter config + train/val/classifier/window-sample settings. |
| `interpreter/notebooks/01_map_comparison_benchmark.ipynb` | End-to-end driver notebook. No runner script exists for this sub-project yet — the notebook is the entry point. |
| `interpreter/IDEAS.md` | Deferred future work: local SNIC patch generation, spatial-pattern characterization, multi-class extension. |

## Perfect interpreter config

No exact no-op exists for the base-probability step (`detection_curve` /
`definition_curve`) — a logistic can only asymptotically approach a step
function, since ground truth (`disturbance_landscape.py`) is itself a hard
threshold. Every other step has an exact, code-verified no-op:

```json
{
  "detection_curve": { "pivot_loss": 0.0, "slope": 100000.0 },
  "definition_curve": { "slope": 100000.0, "threshold_override": null },
  "gaussian_patch_noise": { "std_dev_percent": 0.0 },
  "beta_patch_uncertainty": { "alpha": 0.0, "beta": 5.0 },
  "lowpass_filter": { "kernel_size": 1 },
  "spatial_uncertainty": { "spatial_autocorr_distance": 8.0, "max_scaler": 0.0 }
}
```

- `gaussian_patch_noise.std_dev_percent: 0.0` — code skips RNG entirely when
  all per-type sigmas are `<= 0` (`interpreter_field.py`).
- `beta_patch_uncertainty.alpha: 0.0` — code short-circuits the shrink to
  `scaler = 0.0` whenever `alpha <= 0` or `beta <= 0` (a real Beta draw with
  positive params can never be exactly 0, so this degenerate-parameter path
  is the only way to get a true no-op).
- `lowpass_filter.kernel_size: 1` — `apply_lowpass_filter` returns an
  unmodified copy whenever `kernel_size <= 1`.
- `spatial_uncertainty.max_scaler: 0.0` — code returns the field unchanged
  and skips the `gstools` call entirely when `max_scaler <= 0`.
- `detection_curve`/`definition_curve` extreme slope (`1e5`) — no exact
  no-op; this is an approximation. **Validated empirically** in the
  notebook's step 0 cell by comparing the thresholded perfect-interpreter
  field directly against `base_landscape.disturbed` — see results below.

## How to recreate / re-run from scratch

1. **Environment**: `spatial_base` conda env must actually have `gstools`
   installed (3 of the 4 sensor configs use spatial noise). If you get
   `ImportError: gstools is required...`, run:
   ```bash
   conda env update -f environment.yml -n spatial_base
   ```
   This was needed once already this session — `environment.yml` had been
   fixed to list `gstools` under the `conda-forge` channel, but the existing
   env hadn't been recreated/updated to pick it up.

2. **Checkout the branch**:
   ```bash
   git checkout map-comparison-benchmark
   ```

3. **Run the notebook** end-to-end (from `interpreter/notebooks/`):
   ```bash
   conda run -n spatial_base jupyter nbconvert --to notebook --execute \
     --output 01_map_comparison_benchmark.ipynb \
     --ExecutePreprocessor.timeout=600 \
     01_map_comparison_benchmark.ipynb
   ```
   Or open it in Jupyter and run all cells. It builds 10 training + 20
   validation stacks from `downloaded_images/` (401 rasters available at
   time of writing — plenty of headroom), so no extra data setup is needed.

## Verified results (from the run used to build this doc)

**Step 0 — perfect interpreter vs. ground truth** (first 3 validation stacks):
agreement/precision/recall/f1/iou all `1.0`. The extreme-slope approximation
is indistinguishable from ground truth in practice at this scale — float
overflow saturates the logistic to exactly 0/1 well before any real
edge case.

**Step 6 — pixel-level metrics, averaged across 20 validation stacks:**

| sensor | agreement | precision | recall | f1 | iou |
|---|---|---|---|---|---|
| Sensor_Clean | 0.9986 | 0.934 | 0.991 | 0.960 | 0.926 |
| Sensor_Low | 0.9923 | 0.807 | 0.856 | 0.827 | 0.710 |
| Sensor_Medium | 0.9890 | 0.718 | 0.826 | 0.765 | 0.623 |
| Sensor_High | 0.9863 | 0.711 | 0.651 | 0.675 | 0.511 |

Monotonic degradation from clean → high noise on agreement/f1/iou, matching
the deliberate noise gradient built into the 4 sensor configs. Good sign the
scoring machinery is behaving sensibly.

**Step 7 — window sampling (A/B/C, `window_size=3`, `n_windows=100`):**
confusion matrices per sensor showed the same pattern — off-diagonal counts
grow with sensor noise. B/C (window-level aggregation) collapse each 3×3
window to one sample, so their totals are much smaller than A (pixel-level)
but show cleaner separation.

**Step 8 — meta-analysis:** pixel-level and sampling-based metric deltas
mostly agree in sign and rough magnitude across sensor pairs, with a few
small-magnitude sign disagreements (e.g. Sensor_Low vs. Sensor_Medium
recall). This is exactly the kind of finding the meta-analysis step exists to
surface — not yet interpreted, just recorded here as the current baseline
output.

## Known open items

- `map_comparison_meta_analysis()` is explicitly a first-pass API — expect it
  to change once there's more data to look at.
- No runner script for this sub-project yet (unlike `continuous_binary`'s
  `experiment_runner.py`) — the notebook is manually driven.
- `src/map_comparison.py` and this whole branch still need review/PR before
  merging to `main`.
- Multi-class extension, local SNIC patch generation, and spatial-pattern
  characterization are deferred — tracked in `interpreter/IDEAS.md`.
