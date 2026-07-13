# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

This project develops and compares two paradigms for estimating the area of forest disturbance from remote sensing data.

**Paradigm 1 — Binary (traditional design-based approach):** Both the sensor-derived disturbance map and interpreter labels are treated as binary (disturbed / not-disturbed). Reference labels are collected via a stratified sample and used to correct for map errors via Olofsson-style area estimation. This approach is well-established but is sensitive to false positives and false negatives in both the map and the labels — errors that this simulation quantifies explicitly.

**Paradigm 2 — Continuous probability approach (under development):** Rather than thresholding to binary, the simulation generates a continuous probability-of-disturbance layer that explicitly models the sources of uncertainty that cause a map to be wrong: sensor noise, spatial autocorrelation in sensor error, interpreter perceptual noise, interpreter bias, and interpreter confidence. The goal is to use this probability layer directly to estimate disturbance area, bypassing the error propagation problems inherent in the binary approach.

The project uses Monte Carlo simulation across many synthetic landscapes (`LandscapeStackCollection`) to evaluate how each approach performs under varying levels of sensor degradation and interpreter uncertainty, and to assess whether the probability-based approach yields more accurate area estimates than the design-based binary approach.

## Repository Structure

```
disturbance_uncertainty/
├── src/                        # shared simulation code — both sub-projects use this
├── continuous_binary/          # sub-project: continuous vs binary paradigm comparison
│   ├── notebooks/              # numbered development notebooks (00–20)
│   ├── experiments/configs/    # JSON experiment configs (baseline.json etc.)
│   └── experiment_logs/        # master_experiment_logger.csv + per-run CSVs
├── interpreter/                # sub-project: interpreter behaviour studies (in development)
│   ├── notebooks/
│   ├── experiments/configs/
│   └── experiment_logs/
└── downloaded_images/          # SNIC patch rasters — gitignored, not versioned
```

Both sub-projects share `src/`. Notebooks in `continuous_binary/notebooks/` use `sys.path.insert(0, os.path.abspath('../../src'))` (kernel CWD = notebook directory, standard Jupyter behaviour). Notebooks in `interpreter/notebooks/` (e.g. `01_map_comparison_benchmark.ipynb`) instead resolve the repo root robustly at runtime, since the Jupyter kernel's CWD for these notebooks isn't consistent — it depends on the VS Code Jupyter extension's `jupyter.notebookFileRoot` setting, which varies by machine/workspace config and isn't tracked by the repo. The first cell walks upward from `os.getcwd()` looking for `.git` via a small `find_project_root()` helper, sets `PROJ_ROOT` to the result, and does `sys.path.insert(0, os.path.join(PROJ_ROOT, 'src'))` — this works regardless of whether the kernel starts at the repo root or at `interpreter/notebooks/`. Per-run output folders (`experiment_outputs/`) are gitignored in both sub-project directories via `**/experiment_outputs/`.

## Collaboration conventions

This repo is worked on by two people, each primarily in their own sub-project directory. Key rules:

- **Day-to-day work** (notebooks, configs, experiment logs) goes directly to `main`. Each person works in their own sub-project directory, so conflicts are rare.
- **Changes to `src/`** go on a short-lived feature branch and are merged via PR. This gives both collaborators a review step before shared code changes land in `main`. Bundle any `CLAUDE.md` updates that describe the change in the same branch.
- **`CLAUDE.md`** is shared project documentation — treat it like code. Pull before a session, let Claude propose updates as part of `src/` changes, and commit them intentionally.
- **`.claude/` and `memory/`** are gitignored and user-local. Each collaborator's Claude Code instance maintains its own memory.

## Environment

A conda environment file is provided at `environment.yml`. To create the environment:

```bash
conda env create -f environment.yml
conda activate spatial_base
```

Key dependencies: `numpy`, `scipy`, `matplotlib`, `rasterio`, `pandas`, `geopandas`, `gstools` (spatially correlated noise — required for spatial noise pipeline steps), `earthengine-api`, `geemap`, `dask`, `xarray`, `rioxarray`.

Scripts in `src/` assume they are run with `src/` on the Python path (e.g., via notebook `sys.path.insert` or by running from `src/`).

## Architecture

The simulation is a layered pipeline where each class wraps the output of the one before it.

### Layer 1 — Ground Truth: `DisturbanceLandscape`
(`src/disturbance_landscape.py`)

Takes a SNIC-segmented patch raster (`.tif` from `downloaded_images/`) and a config dict. Assigns patches to two forest types, draws initial cover from a Beta distribution per type, then draws proportional cover loss from a two-component Beta mixture (background + catastrophic). The `disturbed` boolean mask is the final output — pixels where `final_cover < threshold` AND `cover_loss > 0`.

Key attributes: `initial_cover`, `final_cover`, `cover_loss`, `proportional_loss`, `disturbed`, `forest_type_raster`.

### Layer 2a — Sensor Simulation: `SensorField`
(`src/sensor_field.py`)

Takes a `DisturbanceLandscape` as `truth`. Converts percent cover → sensor signal via a log-asymptotic transform (`cover_to_signal`), computes the before/after difference, then applies three noise stages in sequence:
1. Per-patch Gaussian noise (std scaled by signal range, per forest type)
2. Low-pass 2D uniform filter (spatial smoothing)
3. Spatially correlated Gaussian noise via `gstools`

Final output: `sensor_field` (a 2D signal-difference raster).

### Layer 2b — Interpreter Simulation: `InterpreterField`
(`src/interpreter_field.py`)

Takes the same `DisturbanceLandscape` as truth. Produces a probability-of-loss field (0–100%) via:
1. Base probability = logistic(cover_loss) × logistic(threshold − final_cover)
2. Per-patch Gaussian additive noise on probability
3. Per-patch Beta-based shrink toward 0.5 (uncertainty)
4. Low-pass filter
5. Spatially correlated scaler field (gstools) shrinks probabilities toward 0.5

Final output: `interpreter_field` (0–100% probability raster).

**Type-aware:** Steps 1–4 support per-forest-type parameter overrides via a `types` block in the interpreter config (same override pattern as `SensorField`). Add `"types": {"type1": {...}, "type2": {...}}` inside `interpreter.config` to give each forest type different detection curves, noise levels, and lowpass kernel sizes. Omitting the block falls back to global defaults. For step 4, if any type defines a `lowpass_filter` key, the filter is applied per-type (each type's kernel applied to the full field, result stitched by mask); otherwise a single global pass is used. Example:
```json
"types": {
    "type1": { "lowpass_filter": { "kernel_size": 5 } },
    "type2": { "lowpass_filter": { "kernel_size": 1 } }
}
```

### Layer 3 — Individual Interpreter Agent: `InterpreterAgent`
(`src/interpreter_agent.py`)

Operates on *sampled* probability values (not full rasters) from an `InterpreterField`. Each agent has three latent traits drawn once at construction: consistency noise (per-sample Gaussian), directional bias (constant additive offset), and confidence scalar (positive → push toward extremes, negative → pull toward 0.5). Note: `confidence_std` is currently used as a direct scalar, not drawn from a distribution — the stochastic version is commented out. `InterpreterGroup.from_config()` builds multiple agents at once from a list config.

### Layer 4a — Single Experiment: `LandscapeStack`
(`src/landscape_stack.py`)

A dataclass bundling one `DisturbanceLandscape`, an arbitrary number of named `SensorField`s, and one `InterpreterField`. Built via `LandscapeStack.from_config(cfg)`. Stack IDs are either deterministic SHA-1 hashes of the config dict (reproducible across machines given the same config) or random UUIDs. A module-level `LANDSCAPE_REGISTRY` dict keeps lightweight metadata for all stacks built in a session.

### Layer 4b — Experiment Collection: `LandscapeStackCollection`
(`src/landscape_stack_collection.py`, ~1700 lines)

The core experiment-running class. Builds N stacks from a template config, assigning a different patch raster to each, with per-stack random seeds. Supports excluding already-used rasters for train/test splits. All science methods live here:

**Sampling**
- `stratified_sample()` — equal-width strata on sensor values; samples (sensor_value, interpreter_prob, truth) across all stacks
- `probability_stratified_sample()` — strata on calibrated model probabilities (used internally by `buildInterpreterProbModel`)

**Continuous paradigm**
- `buildModel()` — fits a Richards (generalized logistic) curve mapping sensor signal → interpreter probability, using stratified samples across all training stacks; produces scatter + residual diagnostic plots
- `loadModel()`, `applyModel()` — transfer fitted model to a validation collection and apply to every stack pixel; stores result in `stack.calibrated_prob_by_sensor`
- `buildInterpreterProbModel()` — fits a calibration curve mapping model probability → interpreter-adjusted probability. Pipeline: (1) **RANSAC outlier removal** (`ransac_n_iter`, `ransac_min_sample`, `ransac_inlier_eps`, `ransac_min_inlier_frac` — all configurable) identifies inliers; (2) **multi-model competition** fits linear, quadratic, power, and Richards candidates on the inlier set and selects the winner by AIC; (3) **post-hoc endpoint rescaling** stretches the winner so f(0)=0 and f(1)=1 exactly. Winner's `model_type`, `params`, `scale_lo`, `scale_hi`, `aic`, and `aic_delta` are stored in `model_info`. Outliers shown as orange dots in diagnostic plot alongside all candidate curves.
- `applyInterpreterProbModel()` — applies the winning model (dispatches on `model_type`; backward-compatible with old Richards-only `model_info`) to all stacks; result stored in `stack.interpreter_calibrated_prob_by_sensor`

**Area estimation (continuous)**
- `calc_area_distribution_from_prob_image()` — sweeps probability cutoff from 0→1 and records area above each threshold; the resulting distribution is the continuous-paradigm area estimate
- `calc_area_distribution_for_stack()`, `plot_area_distribution_for_stack()` — single-stack wrapper with diagnostic 2×3 figure (sensor signal, training model prob, interpreter-adjusted prob, histograms of each)
- `calc_area_distribution_spatial_threshold_for_stack()` — Monte Carlo variant that uses spatially correlated random threshold surfaces (gstools) rather than uniform thresholds; `m`, `l`, `sigma` control the threshold surface mean, correlation length, and variance
- `plot_spatial_area_distribution_for_stack()`, `plot_spatial_threshold_realization_for_stack()` — diagnostics for the spatial Monte Carlo approach

**Binary paradigm**
- `buildBinaryClassifier()` — finds optimal sensor threshold by sweeping candidate thresholds and minimising a weighted FPR+FNR cost; `fp_to_fn_ratio` controls the tradeoff
- `plotBinaryCurves()`, `loadBinaryModel()`, `applyBinaryClassifier()` — transfer and apply; stores binary maps in `stack.binary_class_by_sensor`
- `binary_stratified_sample()` — stratified sample from the binary map with truth labels via an `InterpreterAgent`; strata are map classes (0/1)
- `binary_confusion_from_samples()` — confusion matrix (n_ij map×ref)
- `olofsson_area_estimates()` — Olofsson-style design-based area estimation with 95% CIs; returns whether true disturbed proportion falls within the CI

**Window-based sampling (interpreter sub-project)**
- `_extract_windows()` — shared backbone; places `n_windows` non-overlapping W×W windows per stack using random sequential adsorption: candidate centers drawn uniformly at random from the valid inset region, accepted only if no overlap with prior accepted centers (`|Δrow|<W` AND `|Δcol|<W`). Interpreter field thresholded at 50% (with optional `InterpreterAgent` perturbation) before patch extraction. `window_size=1` degenerates to simple random per-pixel sampling. `max_attempts` (default `n_windows×50`) caps the search; warns and returns partial results if unreachable.
- `window_sample_A()` — every pixel in every window → one `(map_class, ref_class)` row; same schema as `binary_stratified_sample`, feeds into `binary_confusion_from_samples` / `olofsson_area_estimates`
- `window_sample_B()` — dominant pixel-pair combination per window → one row; ties broken by preferring `(1,1)`, `(0,0)`, `(1,0)`, `(0,1)`
- `window_sample_C()` — independent majority label per field per window (`mean >= 0.5`) → one row; each window is one sample
- `window_sample_D()` — `(prop_map, prop_interp)` per window for scatter plot against 1:1 line; no confusion matrix
- `plot_window_D()` — scatter plot helper for approach D output

**Multi-stack evaluation**
- `evaluate_validation_collection()` — runs both continuous and binary evaluation across all stacks; returns `continuous_df` and `binary_df` DataFrames; accepts `save_dir` to write summary figures and CSVs to disk
- `save_all_stack_plots()` — loops all stacks, saves `{prefix}_area_dist.png` and `{prefix}_diagnostics.png` per stack to a directory; closes figures after each to manage memory

**Output saving** — all plot methods (`buildModel`, `plotBinaryCurves`, `evaluate_validation_collection`, `plot_area_distribution_for_stack`) accept an optional `save_dir` parameter. Pass the path returned by `create_output_dir()` to save figures automatically before display.

### Helper Functions: `disturbance_helper_functions.py`

- `cover_to_signal` — log-asymptotic cover→signal transform: `signal = max_signal * log1p(k*x) / log1p(k)`
- `add_gaussian_patch_noise_by_type` — per-patch Gaussian noise with type-specific std proportional to signal range
- `apply_lowpass_filter` — `scipy.ndimage.uniform_filter` wrapper
- `add_spatially_correlated_noise` — gstools Gaussian model → clipped/normalized noise field added to signal
- `spatial_uncertainty_scaler_field` — gstools field scaled to [0, max_scaler] for shrinking probabilities toward 0.5
- `logistic`, `move_towards_mid`, `richards` — core probability math
- `generate_run_id` — Pacific-time timestamp + short hash, e.g. `run_ID_20260515_143022_abc123`
- `create_output_dir(run_id, experiment_name, base_dir)` — creates `experiment_outputs/{name}_{run_id}/figures/per_stack/`; returns dict with keys `root`, `figures`, `per_stack`
- `flatten_config`, `export_with_metadata` — flattens nested config dict and writes per-run CSV + appends to `experiment_logs/master_experiment_logger.csv`; now also accepts `output_dir` to record the output folder path in the master log

### Experiment Runner: `src/experiment_runner.py`

Standalone script in `src/` that reads a JSON config file and runs the full train/validate pipeline, saving all outputs to disk.

```bash
python src/experiment_runner.py continuous_binary/experiments/configs/baseline.json
```

Can also be imported and called directly from a notebook (matplotlib backend is left as-is so figures display inline; `Agg` is only set when run as a script):

```python
import sys; sys.path.insert(0, '../src')
from experiment_runner import run_experiment
run_id = run_experiment('../continuous_binary/experiments/configs/baseline.json')
```

The runner: creates an output folder via `create_output_dir`, builds training and validation collections, fits continuous and binary models, evaluates, saves all figures and CSVs, and logs to the sub-project's `experiment_logs/`.

### Map Comparison: `src/map_comparison.py`

Full-raster pixel-level scoring for the interpreter sub-project's map comparison benchmark (binary-only first pass — see `interpreter/notebooks/01_map_comparison_benchmark.ipynb`). Unlike every other confusion/metric function in `src/`, which operates on stratified *samples*, these operate on entire rasters:

- `perfect_reference_mask(stack)` — thresholds `stack.interpreter_field.interpreter_field` at 50%; used as the pixel-level "ref_class" throughout, following the same map-vs-interpreter convention used everywhere else in `src/` (never the raw truth mask directly)
- `pixel_confusion_matrix(map_field, ref_field)` — full-raster 2×2 confusion (`n_ij`/`n_i`/`n_total`), same shape as `binary_confusion_from_samples` but computed from two arrays instead of a sampled DataFrame
- `pixel_level_metrics(map_field, ref_field)` — agreement, precision, recall, F1, IoU for class 1
- `pixel_level_metrics_for_stack()`, `pixel_level_metrics_for_collection()` — wrappers scoring `stack.binary_class_by_sensor[sensor_name]` against `perfect_reference_mask(stack)`
- `pixel_agreement_between_maps(map_a, map_b)` — same metrics, map-vs-map (no reference/truth involved)
- `map_comparison_meta_analysis(collection, sensor_names, window_fns, window_kwargs, stack_indices)` — for every pair of sensors, compares pixel-level metric deltas against sampling-based metric deltas. `window_fns` is a dict `{'A': fn, 'B': fn, 'C': fn, 'D': fn}`; A/B/C use agreement/precision/recall/f1/iou, D uses Pearson r between prop_map and prop_interp. Returns wide-format DataFrame with `pixel_delta` and `sampling_delta_{method}` columns; entries that don't apply to a metric/method combination are NaN

A "perfect interpreter" config (see `interpreter/experiments/configs/map_comparison_baseline.json`) drives the reference: every noise-injecting `InterpreterField` step except the base-probability detection/definition curves has an exact, code-verified no-op value (`gaussian_patch_noise.std_dev_percent: 0.0`, `beta_patch_uncertainty.alpha: 0.0`, `lowpass_filter.kernel_size: 1`, `spatial_uncertainty.max_scaler: 0.0`). The detection/definition curves have no exact no-op — a logistic can only asymptotically approach a step function — so an extreme slope (`1e5`) is used instead; this is validated empirically in the notebook (step 0) by comparing the thresholded field directly against `base_landscape.disturbed`.

## Config Structure

Experiment configs are stored as JSON in `<sub-project>/experiments/configs/`. The canonical template is `continuous_binary/experiments/configs/baseline.json`. Top-level keys and what consumes them:

| JSON key | Used by |
|---|---|
| `landscape_stack_config` | `LandscapeStack.from_config()` / `LandscapeStackCollection.from_config()` |
| `training_experiment_config` | training `LandscapeStackCollection.from_config()` |
| `training_model_config` | `buildModel()` arguments |
| `training_interp_prob_model_config` | `buildInterpreterProbModel()` arguments |
| `training_interp_config` | `InterpreterAgent(**...)` for model building |
| `binary_classifier_config` | `buildBinaryClassifier()` arguments |
| `validation_experiment_config` | validation `LandscapeStackCollection.from_config()` |
| `validation_interp_config` | `InterpreterAgent(**...)` for validation |
| `evaluate_config` | `evaluate_validation_collection()` arguments |
| `stack_plots_config` | `save_all_stack_plots()` arguments |

JSON paths (`patch_dir`) are project-root-relative. `src/experiment_runner.py` resolves them to absolute paths automatically. In notebooks, prepend `../../` (notebooks are two levels deep under the sub-project directory) or use `os.path.join(PROJ_ROOT, ...)`.

The equivalent Python dict structure for a full run:

```python
stack_cfg = {
    "id_prefix": "DL",
    "id_strategy": "hash",       # or "uuid"
    "disturbance_config": {
        "random": {"seed": 42},
        "raster": {"band": 1},
        "types": {
            "type1": {
                "proportion_of_patches": 0.6,
                "name": "Dense Forest",
                "initial_cover": {"target_mean": 0.85, "kappa": 20.0},
                "proportional_loss": {"mixture": {
                    "catastrophic_probability": 0.02,
                    "background_alpha": 0.3, "background_beta": 8.0,
                    "catastrophic_alpha": 8.0, "catastrophic_beta": 0.7,
                }},
            },
            "type2": { ... }
        },
        "thresholds": {"final_cover_percent": 15.0},
        "simulation": {"max_retries": 10},
    },
    "sensors": [
        {"name": "Sensor_A", "config": {
            "signal_transform": {"max_signal": 1000.0, "curve_k": 3.0},
            "types": {"type1": {...}, "type2": {...}},
            "gaussian_patch_noise": {"std_proportion_of_range": 0.02},
            "lowpass_filter": {"kernel_size": 3},
            "spatial_noise": {"spatial_autocorr_distance": 10.0, "amplitude_proportion_of_range": 0.1},
        }},
    ],
    "interpreter": {
        "name": "Expert",
        "config": {
            "detection_curve": {"pivot_loss": 15.0, "slope": 0.3},
            "definition_curve": {"slope": 0.4, "threshold_override": null},
            "gaussian_patch_noise": {"std_dev_percent": 8.0},
            "beta_patch_uncertainty": {"alpha": 2.0, "beta": 5.0},
            "lowpass_filter": {"kernel_size": 3},
            "spatial_uncertainty": {"spatial_autocorr_distance": 8.0, "max_scaler": 0.3},
        }
    },
}
```

Collection-level config:
```python
experiment_cfg = {
    "n_stacks": 50,
    "patch_dir": "downloaded_images/",
    "patch_selection_seed": 42,
    "disturbance_seed_base": 1000,
    "use_separate_seeds_per_stack": True,
}
```

## Pending Work (as of 2026-06-19)

### Notebook path fixes required after src/notebooks split + sub-project reorganization

Notebooks 01–14 in `continuous_binary/notebooks/` have not been updated since the directory moves. Each needs:

**1. First code cell:**
```python
import sys, os
sys.path.insert(0, os.path.abspath('../../src'))
```
(Two levels up: `continuous_binary/notebooks/` → `continuous_binary/` → repo root → `src/`)

**2. Path strings:** any bare `"downloaded_images/"` → `"../../downloaded_images/"` (covers `patch_dir` in inline configs).

| Notebook | sys.path cell | image paths |
|----------|:---:|:---:|
| 01, 04–14 | ✓ | ✓ |
| 02, 03 | — | ✓ |
| 00 | — | — |

Notebook `20_jsonexperiment_tester_1.ipynb` also needs its config path updated to `'../../continuous_binary/experiments/configs/baseline.json'`.

### Map comparison benchmark (interpreter sub-project) — binary-only first pass built

Uses window-based sampling (existing A–D, unmodified) alongside new full-raster pixel-level scoring (`src/map_comparison.py`) to test whether sampling-based metrics correctly rank maps relative to pixel-level ground-truth metrics. Scope was deliberately narrowed to **binary disturbed/not-disturbed only** for this first build — forest types are retained in the landscape but no 3-class classification/scoring exists yet.

Built: `interpreter/experiments/configs/map_comparison_baseline.json` (4 sensors spanning clean→high noise, perfect-interpreter reference config), `src/map_comparison.py` (pixel-level scoring + meta-analysis), `interpreter/notebooks/01_map_comparison_benchmark.ipynb` (drives the full pipeline: 10 training stacks, 20 disjoint validation stacks, binary classifiers per sensor, pixel-level scoring, window sampling A–D, meta-analysis).

Multi-class extension, local SNIC patch generation, and spatial-pattern characterization of validation landscapes are deferred — see `interpreter/IDEAS.md`.

### Visualizer

`experiment_visualizer.py` has not been built yet. Planned to load a run's output folder and display summary figures, per-stack browsing, and cross-run comparison. Discussed but not started.

---

## Notebooks

Notebooks live in `<sub-project>/notebooks/` and are numbered sequentially. For the `continuous_binary/` sub-project: the master notebook (`00_`) documents the overall experiment design; individual numbered notebooks correspond to development stages: landscape building (01–04), model fitting (05–07), diagnostics (08–09), and full runs/evaluation (10–14).

`20_jsonexperiment_tester_1.ipynb` — tests the JSON-driven experiment runner (`src/experiment_runner.py`) end-to-end by importing `run_experiment()` directly and running it with `baseline.json`.

For the `interpreter/` sub-project: `01_map_comparison_benchmark.ipynb` drives the map comparison benchmark (see Map Comparison section above) end-to-end from `map_comparison_baseline.json` — there is no runner script equivalent for this sub-project yet, the notebook is the entry point. Sections 3b and 3c add visual inspection panels: 3b shows a 2×2 binary map panel (Truth / Sensor_Low / Sensor_Medium / Sensor_High) for two randomly chosen validation landscapes; 3c shows a 4-panel (A–D) sampling overlay for those same landscapes (window outlines over the error map for A, coloured window blocks for B/C, proportion scatter for D).

## Experiment Logging

`export_with_metadata(config, csv_path="...", notes="...")` flattens any nested config into a per-run CSV and appends a row to the sub-project's `experiment_logs/master_experiment_logger.csv`. Run IDs are auto-generated as `run_ID_YYYYMMDD_HHMMSS_<hash>` in Pacific time. Pass the sub-project log path explicitly, e.g. `csv_path="continuous_binary/experiment_logs/"`. The `.gitignore` uses `**/experiment_logs/run_ID_*.csv` so per-run CSVs are ignored in both sub-project directories; only the master logger CSV is tracked.
