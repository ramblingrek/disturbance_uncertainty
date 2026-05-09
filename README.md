# disturbance_uncertainty

Simulation framework for estimating forest disturbance area from remote sensing data. Compares a traditional binary/Olofsson design-based approach against a continuous probability-based approach under varying sensor and interpreter noise conditions.

---

## Running an experiment

Experiments are defined as JSON config files in `experiments/configs/`. A template with all parameters and sensible baseline values is provided at `experiments/configs/baseline.json`.

Load the JSON and pass each block to the corresponding method:

| JSON key | Used by |
|---|---|
| `landscape_stack_config` | `LandscapeStack.from_config()` / `LandscapeStackCollection.from_config()` |
| `training_experiment_config` | `LandscapeStackCollection.from_config()` for the training collection |
| `training_model_config` | `buildModel()` arguments |
| `training_interp_prob_model_config` | `buildInterpreterProbModel()` arguments |
| `training_interp_config` | `InterpreterAgent(**...)` for model building |
| `binary_classifier_config` | `buildBinaryClassifier()` arguments |
| `validation_experiment_config` | `LandscapeStackCollection.from_config()` for the validation collection |
| `validation_interp_config` | `InterpreterAgent(**...)` for validation |
| `evaluate_config` | `evaluate_validation_collection()` arguments |
| `stack_plots_config` | `save_all_stack_plots()` arguments |

**Notes:**
- JSON `null` = Python `None`; `true`/`false` = Python booleans
- `tif_path` in `landscape_stack_config` is a fallback only — `patch_dir` in the experiment configs drives raster selection for collections
- The `interpreter.config.types` block lets type1/type2 have different detection curves and noise levels; omit it to use global defaults for both types

### Output folder structure

Each run writes outputs to `experiment_outputs/{experiment_name}_{run_id}/`:

```
experiment_outputs/
  baseline_run_ID_20260508_143022_abc123/
    continuous_metrics.csv
    binary_metrics.csv
    figures/
      model_fit_Optical_Sensor_1.png
      binary_threshold_curves_Optical_Sensor_1.png
      continuous_summary.png
      binary_summary.png
      per_stack/
        stack_000_{id}_area_dist.png
        stack_000_{id}_diagnostics.png
        ...
```

Run metadata and the output folder path are recorded in `experiment_logs/master_experiment_logger.csv`.

---

## Parameter reference

See `CONFIG_CHEATSHEET.md` for a full description of every config parameter, its sensible range, and which function consumes it.
