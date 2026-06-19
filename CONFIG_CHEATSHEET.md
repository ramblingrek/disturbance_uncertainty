# Config Parameter Cheat-Sheet

All simulation behaviour is driven by nested Python dicts. This sheet covers every parameter, its sensible range, its purpose, and which function consumes it.

---

## 1. Disturbance Landscape (`disturbance_config`)

Used by: `DisturbanceLandscape.__init__()`

### `random`

| Parameter | Range | Purpose |
|---|---|---|
| `seed` | any int, or `None` | Seeds `np.random.seed()`. `None` → timestamp-based (non-reproducible). Set an int for reproducible runs. |

### `raster`

| Parameter | Range | Purpose |
|---|---|---|
| `band` | 1–N | Which band of the input `.tif` to read as the patch label raster. Almost always `1`. |

### `types.type1` / `types.type2`

Each type block has the same structure. The two types share all patches between them.

| Parameter | Range | Purpose | Used in |
|---|---|---|---|
| `proportion_of_patches` | 0–1 (two types should sum to 1) | Fraction of SNIC patches randomly assigned to this type. | `_setup_types_and_assign_patches()` |
| `name` | string | Human-readable label, used in plots. | diagnostic plots |
| `initial_cover.target_mean` | 0.05–0.98 | Mean of the Beta distribution for initial canopy cover (in 0–1 units, so 0.85 = 85%). | `_calculate_initial_forest_cover()` |
| `initial_cover.kappa` | 5–50 | Concentration of the Beta distribution. Higher = tighter cluster around the mean. `kappa=20` with `mean=0.85` gives roughly ±10% spread. Internally: `alpha = mean * kappa`, `beta = (1-mean) * kappa`. | `_calculate_initial_forest_cover()` |
| `proportional_loss.mixture.catastrophic_probability` | 0.001–0.2 | Probability that a patch draws its loss from the catastrophic component rather than the background component. E.g. `0.02` = 2% of patches experience severe disturbance. | `_generate_loss_rasters()` |
| `proportional_loss.mixture.background_alpha` | 0.1–2.0 | Alpha shape for the background Beta distribution. Low alpha + high beta concentrates draws near zero (most patches lose very little). | `_generate_loss_rasters()` |
| `proportional_loss.mixture.background_beta` | 3–20 | Beta shape for background loss. Higher = more mass near zero. | `_generate_loss_rasters()` |
| `proportional_loss.mixture.catastrophic_alpha` | 3–15 | Alpha shape for catastrophic Beta. High alpha + low beta concentrates draws near 1 (near-total loss). | `_generate_loss_rasters()` |
| `proportional_loss.mixture.catastrophic_beta` | 0.3–2.0 | Beta shape for catastrophic loss. Lower = more mass near 1. | `_generate_loss_rasters()` |

### `thresholds`

| Parameter | Range | Purpose | Used in |
|---|---|---|---|
| `final_cover_percent` | 5–30 | A pixel is labelled `disturbed = True` when `final_cover < this value AND cover_loss > 0`. Also used as the default pivot for the interpreter definition curve. | `_update_disturbed_mask()`, `InterpreterField._compute_base_probability()` |

### `simulation`

| Parameter | Range | Purpose | Used in |
|---|---|---|---|
| `max_retries` | 1–20 | How many times to re-draw the loss rasters if zero disturbed pixels result. Rare with sensible mixture params. | `DisturbanceLandscape.__init__()` |

---

## 2. Sensor (`sensors[].config`)

Used by: `SensorField._run_pipeline()`

Type-specific keys under `types.type1` / `types.type2` override the top-level defaults for that type.

### `signal_transform`

| Parameter | Range | Purpose | Used in |
|---|---|---|---|
| `max_signal` | 500–5000 | Maximum sensor signal value at 100% cover. Scales the whole signal range. | `_compute_signal_difference_step()` via `cover_to_signal()` |
| `curve_k` | 0.5–10 | Curvature of the log-asymptotic cover→signal transform. Higher = more concave (sensor saturates earlier). `k=1` is nearly linear; `k=5` saturates quickly. | `_compute_signal_difference_step()` via `cover_to_signal()` |

### `gaussian_patch_noise`

| Parameter | Range | Purpose | Used in |
|---|---|---|---|
| `std_proportion_of_range` | 0–0.15 | Per-patch noise std as a fraction of the total signal range observed in the scene. `0.02` = noise std is 2% of the signal range. Set to `0` to disable. | `_add_patch_noise_step()` via `add_gaussian_patch_noise_by_type()` |

### `lowpass_filter`

| Parameter | Range | Purpose | Used in |
|---|---|---|---|
| `kernel_size` | 0–15 (odd integers typical) | Size of the 2D uniform smoothing kernel in pixels. `0` or `1` = no smoothing. Larger values blur patch boundaries and spread noise spatially. | `_apply_lowpass_step()` via `apply_lowpass_filter()` |

### `spatial_noise`

| Parameter | Range | Purpose | Used in |
|---|---|---|---|
| `spatial_autocorr_distance` | 2–50 (pixels) | Correlation length of the gstools Gaussian random field added as spatially structured noise. Larger = broader correlated patches of noise. | `_add_spatial_noise_step()` via `add_spatially_correlated_noise()` |
| `amplitude_proportion_of_range` | 0–0.3 | Amplitude of spatial noise as a fraction of the signal range. `0.1` = noise can shift signal by ±10% of its range. Set to `0` to disable. | `_add_spatial_noise_step()` via `add_spatially_correlated_noise()` |

---

## 3. Interpreter Field (`interpreter.config`)

Used by: `InterpreterField._run_pipeline()`

Per-type overrides go in `types.type1` / `types.type2` using the same key names. Any key omitted from the type block falls back to the top-level default.

### `detection_curve`

Controls how strongly the interpreter perceives disturbance as a function of **cover loss**.

| Parameter | Range | Purpose | Used in |
|---|---|---|---|
| `pivot_loss` | 5–40 (% cover loss) | Cover loss value where the interpreter has 50% probability of detecting disturbance. Lower = more sensitive interpreter. | `_compute_base_probability()` |
| `slope` | 0.05–1.0 | Steepness of the detection logistic. Higher = sharper transition from "not detected" to "detected". | `_compute_base_probability()` |

### `definition_curve`

Controls how the interpreter applies the forest/non-forest **threshold** to final cover.

| Parameter | Range | Purpose | Used in |
|---|---|---|---|
| `slope` | 0.05–1.0 | Steepness of the definition logistic around the threshold. Higher = sharper. | `_compute_base_probability()` |
| `threshold_override` | % cover (or `None`) | If set, overrides the `final_cover_percent` from the disturbance config as the interpreter's threshold. `None` = use the truth threshold. Useful to simulate interpreter/map disagreement on the threshold itself. | `_compute_base_probability()` |

### `gaussian_patch_noise`

| Parameter | Range | Purpose | Used in |
|---|---|---|---|
| `std_dev_percent` | 0–20 (percentage points) | Per-patch Gaussian noise on the probability value. `8.0` means noise std = 8 percentage points. Added as an offset to the base probability before clipping to [0,1]. Set to `0` to disable. | `_apply_patch_gaussian()` |

### `beta_patch_uncertainty`

Draws a per-patch scaler from Beta(alpha, beta) and uses it to shrink the probability toward 0.5: `p_new = p + scaler × (0.5 − p)`.

| Parameter | Range | Purpose | Used in |
|---|---|---|---|
| `alpha` | 0.1–10 | Alpha shape of the Beta distribution for the shrink scaler. Low alpha + high beta = most patches barely shrink toward 0.5 (scaler near 0). | `_apply_patch_beta()` |
| `beta` | 0.1–20 | Beta shape of the shrink scaler distribution. Higher beta relative to alpha = stronger pull toward 0.5 in most patches. | `_apply_patch_beta()` |

### `lowpass_filter`

| Parameter | Range | Purpose | Used in |
|---|---|---|---|
| `kernel_size` | 0–15 | Spatial smoothing of the probability field. Simulates edge effects and spatial coherence in interpreter perception. `0` = disabled. | `_apply_lowpass_step()` via `apply_lowpass_filter()` |

**Per-type override:** `lowpass_filter` can be set independently per forest type inside the `types` block. If any type defines it, each type gets its own kernel (falling back to the global value for types that don't). Example:
```json
"types": {
    "type1": { "lowpass_filter": { "kernel_size": 5 } },
    "type2": { "lowpass_filter": { "kernel_size": 1 } }
}
```

### `spatial_uncertainty`

Generates a spatially correlated field of scalers in [0, max_scaler] and uses them to shrink probabilities toward 0.5 across the landscape.

| Parameter | Range | Purpose | Used in |
|---|---|---|---|
| `spatial_autocorr_distance` | 2–50 (pixels) | Correlation length of the spatial uncertainty field. Larger = broader regions of high/low interpreter uncertainty. | `_apply_spatial_uncertainty()` via `spatial_uncertainty_scaler_field()` |
| `max_scaler` | 0–0.8 | Maximum shrink-toward-0.5 scaler. `0.3` means uncertainty can pull probabilities at most 30% of the way toward 0.5. `0` = disabled. | `_apply_spatial_uncertainty()` via `spatial_uncertainty_scaler_field()` |

---

## 4. Interpreter Agent (`InterpreterAgent` config)

Used by: `InterpreterAgent.__post_init__()` and `InterpreterAgent.transform_probabilities()`

This controls **individual human label noise** applied at sampling time (not the full raster).

| Parameter | Range | Purpose | Used in |
|---|---|---|---|
| `interpreter_id` | string | Unique name for this agent. | identification |
| `consistency_std_pct` | 0–20 (percentage points) | Per-sample Gaussian noise. Drawn fresh for each label. Models within-interpreter inconsistency. | `transform_probabilities()` |
| `bias_std_pct` | 0–15 (percentage points) | Directional bias drawn once per agent from N(0, bias_std_pct). Positive = systematically over-labels disturbance; negative = under-labels. | `__post_init__()` |
| `confidence_std` | −1 to +1 | Directly sets the confidence scalar (not drawn stochastically — see code note). Positive = pushes all probabilities toward 0 or 1 (overconfident). Negative = pulls all probabilities toward 0.5 (underconfident). | `transform_probabilities()` |
| `seed` | int or `None` | Seeds this agent's RNG. `None` = timestamp + hash of interpreter_id. | `__post_init__()` |

---

## 5. Collection / Experiment Config (`experiment_cfg`)

Used by: `LandscapeStackCollection.from_config()`

| Parameter | Range | Purpose |
|---|---|---|
| `n_stacks` | 1–200+ | Number of landscape stacks to build. More = better Monte Carlo coverage but slower. |
| `patch_dir` | path string | Directory containing SNIC `.tif` patch rasters. One raster is assigned to each stack. |
| `patch_selection_seed` | int or `None` | Seeds the random selection of rasters from `patch_dir`. Fix for reproducible stack assignments. |
| `disturbance_seed_base` | int or `None` | Base random seed for `DisturbanceLandscape`. Stack `i` gets `seed = base + i` (if `use_separate_seeds_per_stack=True`). |
| `use_separate_seeds_per_stack` | bool | If `True`, each stack gets a unique seed (`base + i`). If `False`, all stacks share the same seed (not recommended). |

---

## 6. Binary Classifier Parameters

Used by: `LandscapeStackCollection.buildBinaryClassifier()`

| Parameter | Range | Purpose |
|---|---|---|
| `n_strata` | 3–10 | Number of equal-width strata on sensor values for training sample stratification. |
| `samples_per_stratum` | 20–500 | Samples drawn per stratum per stack. More = more stable threshold estimate. |
| `sampling_seed` | int or `None` | Seeds the stratified sample draw. |
| `fp_to_fn_ratio` | 0.1–10 | Tradeoff weight: `cost = fp_to_fn_ratio × FPR + FNR`. `>1` penalises false positives more; `<1` penalises false negatives more. `1.0` = equal cost. |
| `interpreter_agent` | `InterpreterAgent` or `None` | If provided, perturbs interpreter probabilities before thresholding to binary labels. |

---

## 6b. Window-Based Sampling Parameters

Used by: `_extract_windows()` (backbone) and `window_sample_A/B/C/D()`

All four public methods share this parameter set. `window_size=1` makes all four degenerate to simple random per-pixel sampling, which is the special case of the general approach.

| Parameter | Range | Purpose |
|---|---|---|
| `sensor_name` | string | Which sensor's binary map to sample from. |
| `window_size` | 1, 3, 5, 7, ... (odd int) | Side length W of the square sampling window. `1` = per-pixel (special case). |
| `n_windows` | 10–1000+ | Number of non-overlapping windows to place **per stack**. |
| `sampling_seed` | int or `None` | Seeds the random center selection. Fix for reproducibility. |
| `interpreter_agent` | `InterpreterAgent` or `None` | If provided, perturbs raw interpreter probabilities before the 50% threshold is applied. |
| `stack_indices` | list[int] or `None` | Restrict sampling to a subset of stacks by index. `None` = all stacks. |
| `max_attempts` | int or `None` | Maximum candidate draws before giving up per stack. Default = `n_windows × 50`. Warns and returns partial results if exhausted. |

### Window placement (shared backbone)

Centers drawn uniformly at random from the valid inset region (inset by `W//2` on all sides). Each candidate accepted only if it does not overlap any prior accepted window: overlap defined as `|Δrow| < W` AND `|Δcol| < W`. This is random sequential adsorption — no fixed tiling, no grid artifacts.

### Approach-specific outputs

| Method | Output schema | Feeds into |
|---|---|---|
| `window_sample_A()` | `map_class`, `ref_class` per **pixel** | `binary_confusion_from_samples()`, `olofsson_area_estimates()` |
| `window_sample_B()` | `map_class`, `ref_class` per **window** (dominant cell) | same |
| `window_sample_C()` | `map_class`, `ref_class` per **window** (independent majority per field) | same |
| `window_sample_D()` | `prop_map`, `prop_interp` per **window** | `plot_window_D()` scatter plot |

---

## 7. Continuous Model Parameters

Used by: `LandscapeStackCollection.buildModel()` and `buildInterpreterProbModel()`

| Parameter | Range | Purpose |
|---|---|---|
| `n_strata` | 3–10 | Strata for training sample stratification (on sensor values for `buildModel`; on model probabilities for `buildInterpreterProbModel`). |
| `samples_per_stratum` | 20–500 | Samples per stratum per stack. |
| `sampling_seed` | int or `None` | Seeds the sample draw. |
| `interpreter_agent` | `InterpreterAgent` or `None` | If provided, transforms base interpreter probabilities before fitting the calibration curve. |

### RANSAC outlier removal (`buildInterpreterProbModel` only)

Run before model fitting to exclude outliers from the inlier set used for curve fitting.

| Parameter | Range | Purpose |
|---|---|---|
| `ransac_n_iter` | 50–500 | Number of RANSAC iterations. Each draws a random mini-batch, fits a candidate curve, and counts inliers. | 
| `ransac_min_sample` | 5–30 | Size of the random mini-batch drawn each iteration. |
| `ransac_inlier_eps` | 0.05–0.3 | Inlier tolerance: a point is an inlier if `|y − ŷ| ≤ eps`. **Primary tuning knob** — raise if valid points are being removed as outliers, lower if outliers are sneaking through. |
| `ransac_min_inlier_frac` | 0.5–0.9 | If the best inlier count is below `frac × n`, RANSAC falls back to using all data with a warning. Default 0.66. |

### Model competition (`buildInterpreterProbModel` — fitted, not configured directly)

Fits four candidate curves on the RANSAC inlier set, applies post-hoc endpoint rescaling to each (so all candidates pass through (0,0) and (1,1)), gates out non-monotone candidates, and selects the winner by lowest AIC (`n·log(RSS/n) + 2k`). Fitted results stored in `model_info`:

| `model_info` key | Meaning |
|---|---|
| `model_type` | Winning model name: `"linear"`, `"quadratic"`, `"power"`, or `"richards"` |
| `params` | Fitted parameter list (order: linear=[a,b]; quadratic=[a,b,c]; power=[a,b]; richards=[x0,b,nu]) |
| `scale_lo`, `scale_hi` | Endpoint rescaling values — applied in `applyInterpreterProbModel` |
| `aic` | AIC of the winner |
| `aic_delta` | AIC(winner) − AIC(runner-up); larger = more decisive win |

### `buildModel` — Richards curve (sensor → model probability)

`buildModel` still fits a single Richards curve: `p(x) = (1 + exp(−b × (x − x0)))^(−nu)`. Stored as `x0`, `b`, `nu` in the trained model dict.

---

## 8. Spatial Monte Carlo Area Estimation

Used by: `LandscapeStackCollection.calc_area_distribution_spatial_threshold_for_stack()`

| Parameter | Range | Purpose |
|---|---|---|
| `n_iter` | 100–5000 | Number of Monte Carlo threshold surface draws. More = smoother area distribution. |
| `m` | 0–1, or `(low, high)` tuple | Mean of the spatially correlated threshold surface. Can be a fixed float or a `(low, high)` tuple to draw uniformly each iteration. |
| `l` | 2–100 pixels, or tuple | Spatial correlation length of the threshold surface. Larger = broader patches of high/low threshold. Can be fixed or drawn per iteration. |
| `sigma` | 0.05–0.5 | Std dev of threshold surface about `m`. Larger = more spatially variable threshold, wider area distribution. |
