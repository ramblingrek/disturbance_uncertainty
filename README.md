# disturbance_uncertainty

Simulation framework for estimating forest disturbance area from remote sensing data. Compares a traditional binary/Olofsson design-based approach against a continuous probability-based approach under varying sensor and interpreter noise conditions.

---

## Repository structure

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

Both sub-projects share `src/`. Notebooks use `sys.path.insert(0, os.path.abspath('../../src'))`. Per-run output folders (`experiment_outputs/`) are gitignored in both sub-project directories.

---

## Running an experiment

Experiments are defined as JSON config files in `<sub-project>/experiments/configs/`. The canonical template is `continuous_binary/experiments/configs/baseline.json`.

Run from the command line:

```bash
python src/experiment_runner.py continuous_binary/experiments/configs/baseline.json
```

Or call directly from a notebook:

```python
import sys; sys.path.insert(0, '../../src')
from experiment_runner import run_experiment
run_id = run_experiment('../../continuous_binary/experiments/configs/baseline.json')
```

The JSON keys and what consumes them:

| JSON key | Used by |
|---|---|
| `landscape_stack_config` | `LandscapeStack.from_config()` / `LandscapeStackCollection.from_config()` |
| `training_experiment_config` | `LandscapeStackCollection.from_config()` for the training collection |
| `training_model_config` | `buildModel()` arguments |
| `training_interp_prob_model_config` | `buildInterpreterProbModel()` arguments (includes RANSAC params) |
| `training_interp_config` | `InterpreterAgent(**...)` for model building |
| `binary_classifier_config` | `buildBinaryClassifier()` arguments |
| `validation_experiment_config` | `LandscapeStackCollection.from_config()` for the validation collection |
| `validation_interp_config` | `InterpreterAgent(**...)` for validation |
| `evaluate_config` | `evaluate_validation_collection()` arguments |
| `stack_plots_config` | `save_all_stack_plots()` arguments |

**Notes:**
- JSON `null` = Python `None`; `true`/`false` = Python booleans
- `patch_dir` in `training_experiment_config` and `validation_experiment_config` drives raster selection; paths are project-root-relative when using `experiment_runner.py`
- The `interpreter.config.types` block lets type1/type2 have different detection curves and noise levels; omit it to use global defaults for both types

### Output folder structure

Each run writes outputs to `<sub-project>/experiment_outputs/{experiment_name}_{run_id}/`:

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

Run metadata and the output folder path are recorded in `<sub-project>/experiment_logs/master_experiment_logger.csv`.

---

## Parameter reference

See `CONFIG_CHEATSHEET.md` for a full description of every config parameter, its sensible range, and which function consumes it.
