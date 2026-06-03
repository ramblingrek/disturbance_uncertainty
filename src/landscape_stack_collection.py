import copy
import os
import glob
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from scipy.optimize import curve_fit
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
import time
try:
    import gstools as gs
except ImportError:
    gs = None


# project specific libraries and such
from landscape_stack import LandscapeStack
from disturbance_landscape import DisturbanceLandscape
from sensor_field import SensorField
from disturbance_helper_functions import logistic, richards
from interpreter_field import InterpreterField
from interpreter_agent import InterpreterAgent

# Helper for later logistic
def _logistic_for_fit(x, x0, k):
    return logistic(x, x0=x0, k=k)
    
    
def _resolve_sampler(val, rng):
    """
    Helper: allow val to be
      - float (constant)
      - tuple (low, high) meaning uniform draw each iteration
      - callable rng -> float
      - list/np.ndarray meaning pick random element each iteration
    """
    if callable(val):
        return float(val(rng))
    if isinstance(val, (tuple, list)) and len(val) == 2 and all(isinstance(x, (int, float)) for x in val):
        lo, hi = float(val[0]), float(val[1])
        return float(rng.uniform(lo, hi))
    if isinstance(val, (list, np.ndarray)) and not (len(val) == 2 and all(isinstance(x, (int, float)) for x in val)):
        return float(rng.choice(val))
    return float(val)
    


@dataclass
class LandscapeStackCollection:
    """
    A collection of LandscapeStacks built from a common template config.

    Now supports:
      - Assigning a different patch raster (.tif) to each stack, drawn
        randomly without replacement from a directory.
    """
    stacks: List[LandscapeStack]
    experiment_config: dict = field(repr=False)
    models: Dict[str, Dict[str, Any]] = field(default_factory=dict, repr=False)
    binary_models: Dict[str, Dict[str, Any]] = field(default_factory=dict, repr=False)  # threshold models

    # ------------------------ Construction from config ------------------------

    @classmethod
    def from_config(
            cls,
            stack_template_cfg: dict,
            experiment_cfg: dict,
            exclude_tif_paths: Optional[list[str]] = None,
        ) -> "LandscapeStackCollection":  
        
        """
        Build a collection of LandscapeStacks.

        Parameters
        ----------
        stack_template_cfg : dict
            A single-stack config (like landscape_stack_config). Its "tif_path"
            will be overridden per stack if experiment_cfg['patch_dir'] is set.

        experiment_cfg : dict
            Controls:
              - "n_stacks": int
              - "disturbance_seed_base": int or None
              - "use_separate_seeds_per_stack": bool
              - "patch_dir": str (directory with .tif files)
              - "patch_selection_seed": int or None
         exclude_tif_paths : list[str] or None
            Optional list of .tif paths that should NOT be used in this
            collection (e.g., patches already used for training).
        """
        n_stacks = int(experiment_cfg.get("n_stacks", 10))
        seed_base = experiment_cfg.get("disturbance_seed_base", int(time.time()))
        separate_seeds = experiment_cfg.get("use_separate_seeds_per_stack", True)

        patch_dir = experiment_cfg.get("patch_dir", None)
        patch_sel_seed = experiment_cfg.get("patch_selection_seed", int(time.time()))

        # Normalize exclusion list
        exclude_set = set()
        if exclude_tif_paths is not None:
            exclude_set = {os.path.abspath(p) for p in exclude_tif_paths}

        # --- Discover patch rasters if patch_dir is provided ---
        if patch_dir is not None:
            pattern1 = os.path.join(patch_dir, "*.tif")
            pattern2 = os.path.join(patch_dir, "*.tiff")
            all_paths = sorted(glob.glob(pattern1) + glob.glob(pattern2))

            if len(all_paths) == 0:
                raise RuntimeError(f"No .tif/.tiff files found in patch_dir={patch_dir!r}")

            # Apply exclusions
            tif_paths = [
                p for p in all_paths
                if os.path.abspath(p) not in exclude_set
            ]

            if len(tif_paths) < n_stacks:
                raise RuntimeError(
                    f"Not enough patch rasters in {patch_dir!r} after exclusions: "
                    f"available {len(tif_paths)}, but n_stacks={n_stacks}. "
                    f"(You may need more rasters or fewer stacks.)"
                )

            # Randomize order of rasters with optional seed
            rng_patch = np.random.default_rng(patch_sel_seed)
            perm = rng_patch.permutation(len(tif_paths))
            tif_paths = [tif_paths[i] for i in perm[:n_stacks]]
        else:
            tif_path_template = stack_template_cfg.get("tif_path", None)
            if tif_path_template is None:
                raise RuntimeError(
                    "No 'patch_dir' provided in experiment_cfg and no 'tif_path' in stack_template_cfg."
                )
            tif_paths = [tif_path_template] * n_stacks

        stacks: List[LandscapeStack] = []

        for i in range(n_stacks):
            cfg_i = copy.deepcopy(stack_template_cfg)

            # Overwrite tif_path for this stack with the i-th selected raster
            cfg_i["tif_path"] = tif_paths[i]

            # Optionally set a unique random seed for the DisturbanceLandscape
            if seed_base is not None:
                cfg_i.setdefault("disturbance_config", {})
                cfg_i["disturbance_config"].setdefault("random", {})
                if separate_seeds:
                    cfg_i["disturbance_config"]["random"]["seed"] = seed_base + i
                else:
                    cfg_i["disturbance_config"]["random"]["seed"] = seed_base

            # Build the stack
            stack_i = LandscapeStack.from_config(cfg_i)
            stacks.append(stack_i)

        return cls(stacks=stacks, experiment_config=experiment_cfg)

    # ------------------------- Sampling / stratification ----------------------

    def stratified_sample(
        self,
        sensor_name: Optional[str] = None,
        n_strata: int = 5,
        samples_per_stratum: int = 50,
        sampling_seed: Optional[int] = int(time.time()),
        include_truth: bool = True,
    ) -> pd.DataFrame:
        """
        Stratify sensor_field values for a chosen sensor and sample locations
        across strata for ALL stacks in this collection.

        NOW uses equal-width value intervals, not quantiles.

        For each stack:
          - Take its sensor_field[sensor_name].
          - Compute min/max sensor value over valid pixels.
          - Divide [min, max] into n_strata equal-width intervals.
          - Assign each pixel to a stratum.
          - Sample up to 'samples_per_stratum' pixels per stratum (no replacement).
          - Extract sensor value and interpreter probability at those pixels.
          - Optionally extract truth variables from the base DisturbanceLandscape.
        """
        rng = np.random.default_rng(sampling_seed)

        rows: List[Dict[str, Any]] = []

        for si, stack in enumerate(self.stacks):
            # Choose sensor
            if sensor_name is None:
                if not stack.sensor_fields:
                    raise RuntimeError(f"Stack {stack.stack_id} has no sensors.")
                s_name = next(iter(stack.sensor_fields.keys()))
            else:
                s_name = sensor_name

            if s_name not in stack.sensor_fields:
                raise KeyError(f"Sensor {s_name!r} not found in stack {stack.stack_id}")

            sensor: SensorField = stack.sensor_fields[s_name]
            base: DisturbanceLandscape = stack.base_landscape
            interp: InterpreterField = stack.interpreter_field

            sensor_field = np.asarray(sensor.sensor_field, dtype=float)
            interp_prob = np.asarray(interp.interpreter_field, dtype=float)  # 0–100

            # Flatten indices
            nrows, ncols = sensor_field.shape
            rr, cc = np.meshgrid(np.arange(nrows), np.arange(ncols), indexing="ij")

            values_flat = sensor_field.ravel()
            rr_flat = rr.ravel()
            cc_flat = cc.ravel()

            # Handle NaNs: only consider finite values for stratification
            valid_mask = np.isfinite(values_flat)
            if not np.any(valid_mask):
                raise RuntimeError(f"All sensor values are NaN for stack {stack.stack_id}")

            valid_values = values_flat[valid_mask]

            # ---------------------------------------------------------
            # Equal-interval strata edges (not quantiles)
            # ---------------------------------------------------------
            vmin = float(np.min(valid_values))
            vmax = float(np.max(valid_values))

            if vmin == vmax:
                # Degenerate case: all values equal → put all valid pixels in stratum 0
                strata_flat = np.full_like(values_flat, fill_value=-1, dtype=int)
                strata_flat[valid_mask] = 0
                effective_n_strata = 1
            else:
                edges = np.linspace(vmin, vmax, n_strata + 1)

                # Assign strata: 0..(n_strata-1)
                # np.digitize bins define intervals (-inf, edges[0]], (edges[0], edges[1]], ...
                # We use edges[1:-1] as internal bin edges.
                strata_flat = np.full_like(values_flat, fill_value=-1, dtype=int)
                strata_flat[valid_mask] = np.digitize(
                    valid_values,
                    bins=edges[1:-1],
                    right=False,
                )
                effective_n_strata = n_strata

            # ---------------------------------------------------------
            # Sample from each stratum
            # ---------------------------------------------------------
            for stratum_id in range(effective_n_strata):
                stratum_mask = valid_mask & (strata_flat == stratum_id)
                idx_stratum = np.where(stratum_mask)[0]

                if idx_stratum.size == 0:
                    continue  # no pixels in this stratum

                n_to_sample = min(samples_per_stratum, idx_stratum.size)
                chosen_idx = rng.choice(idx_stratum, size=n_to_sample, replace=False)

                for flat_idx in chosen_idx:
                    r = int(rr_flat[flat_idx])
                    c = int(cc_flat[flat_idx])

                    row_dict: Dict[str, Any] = {
                        "stack_id": stack.stack_id,
                        "stack_index": si,
                        "sensor_name": s_name,
                        "row": r,
                        "col": c,
                        "stratum": stratum_id,
                        "sensor_value": float(sensor_field[r, c]),
                        "interpreter_prob": float(interp_prob[r, c]),  # 0–100
                    }

                    if include_truth:
                        if getattr(base, "forest_type_raster", None) is not None:
                            row_dict["forest_type"] = int(base.forest_type_raster[r, c])

                        if getattr(base, "cover_loss", None) is not None:
                            row_dict["cover_loss"] = float(base.cover_loss[r, c])
                        if getattr(base, "final_cover", None) is not None:
                            row_dict["final_cover"] = float(base.final_cover[r, c])
                        if getattr(base, "disturbed", None) is not None:
                            row_dict["disturbed"] = bool(base.disturbed[r, c])

                    rows.append(row_dict)

        samples = pd.DataFrame(rows)
        return samples


    ###------------------------------------------------------------------------------
    #
    #               Continuous predictors
    #
    ###------------------------------------------------------------------------------
   

    def buildModel(
        self,
        sensor_names: Optional[List[str]] = None,
        n_strata: int = 5,
        samples_per_stratum: int = 200,
        kappa_for_logistic: Optional[float] = 1.0,
        sampling_seed: Optional[int] = int(time.time()),
        interpreter_agent: Optional[InterpreterAgent] = None,
        samples: Optional[pd.DataFrame] = None,
        save_dir: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fit a richards function mapping sensor values -> interpreter probability of loss
        for one or more sensors, using stratified samples across all stacks.

        For each sensor s:
          - Use stratified_sample(...) to get (sensor_value, interpreter_prob) pairs.
          - Fit p ~ richards function [fill in later]
          
          - Store model info in self.models[s].

        Returns
        -------
        self.models : dict
            Keys are sensor names, values are model dictionaries with:
              - 'sensor_name'
              - 'params': {'a', 'b'}
              - 'n_samples'
              - 'strata': {...}
              - 'experiment_config'
              - 'base_config_example', 'sensor_config_example', 'interpreter_config_example'
        """
        if not self.stacks:
            raise RuntimeError("No stacks in this collection.")

        # Default: use all sensors from the first stack
        if sensor_names is None:
            sensor_names = list(self.stacks[0].sensor_fields.keys())
        else:
            sensor_names = list(sensor_names)

        for s_name in sensor_names:
            if samples is not None:
                s = samples[samples["sensor_name"] == s_name] if "sensor_name" in samples.columns else samples
            else:
                s = self.stratified_sample(
                    sensor_name=s_name,
                    n_strata=n_strata,
                    samples_per_stratum=samples_per_stratum,
                    sampling_seed=sampling_seed,
                    include_truth=True,
                )

            if s.empty:
                raise RuntimeError(f"No samples obtained for sensor {s_name!r}.")

            x = s["sensor_value"].to_numpy(dtype=float)
            
            
            #############
            # Adding in the interpreter variability. Read the landscape
            #   but add variability by agent. 
            
            y_probs = s["interpreter_prob"].to_numpy(dtype=float)  # 0–100

            # Apply interpreter uncertainty if agent provided
            if interpreter_agent is not None:
                y_probs = interpreter_agent.transform_probabilities(y_probs)
            
            y = y_probs / 100.0  # 0..1 for logistic fit
            ####################
            
            # Drop NaNs / infinities
            mask = np.isfinite(x) & np.isfinite(y)
            x = x[mask]
            y = y[mask]

            if x.size < 5:
                raise RuntimeError(
                    f"Not enough valid samples to fit model for sensor {s_name!r}."
                )


            #  Trying the richards function
            # -----------------------------
            # Generalized logistic (Richards) fit
            # p(x) = (1 + exp(-b*(x-x0)))^(-nu)
            # -----------------------------
         
            # ---- Initial guesses ----
            # x0 guess: sensor value near y ~ 0.5 (fallback to median)
            try:
                x0_0 = float(np.interp(0.5, np.sort(y), np.sort(x)))  # rough
            except Exception:
                x0_0 = float(np.median(x))
            
            # slope guess: small positive
            b0 = 0.01
            
            # nu guess: start at 1 (standard logistic)
            nu0 = 1.0
            
            p0 = [x0_0, b0, nu0]
            
            # ---- Bounds ----
            # b should be positive if higher sensor -> higher loss prob (adjust if needed)
            # nu should be > 0
            # x0 can roam within a padded range of observed x
            xmin, xmax = float(np.min(x)), float(np.max(x))
            pad = 0.2 * (xmax - xmin + 1e-9)
            bounds = (
                [xmin - pad,  1e-6,  1e-3],   # lower bounds: x0, b, nu
                [xmax + pad,  10.0,  50.0],   # upper bounds
            )
            
            popt, pcov = curve_fit(
                richards,
                x,
                y,
                p0=p0,
                bounds=bounds,
                maxfev=30000,
            )
            x0_hat, b_hat, nu_hat = popt

           

            # -----------------------------
            # Diagnostic plot: X–Y scatter + fitted richards (colored by forest type)
            # -----------------------------
            
            # Recompute type array aligned to x/y (after NaN filtering)
            if "forest_type" in s.columns:
                type_all = s["forest_type"].to_numpy()
                type_vals = type_all[mask]
            else:
                type_vals = None

            # Smooth curve across observed x-range
            x_plot = np.linspace(np.nanmin(x), np.nanmax(x), 300)
            y_plot = richards(x_plot, x0_hat, b_hat, nu_hat)


            # Predicted values and residuals (same y-scale as fit: 0..1)
            y_hat = richards(x, x0_hat, b_hat, nu_hat)
            resid = y - y_hat

            fig, ax = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

           # ---- Panel 1: scatter + fitted curve ----
            if type_vals is not None:
                unique_types = np.unique(type_vals)
                cmap = plt.get_cmap("tab10")
                for i, t in enumerate(unique_types):
                    m = (type_vals == t)
                    ax[0].scatter(x[m], y[m], s=12, alpha=0.5, color=cmap(i), label=f"Type {int(t)}")
            else:
                ax[0].scatter(x, y, s=12, alpha=0.4, color="gray", label="Samples")
            
            ax[0].plot(x_plot, y_plot, color="black", linewidth=2, label="Richards fit")
            ax[0].set_xlabel("Sensor value")
            ax[0].set_ylabel("Interpreter probability of loss")
            ax[0].set_ylim(0, 1)
            ax[0].grid(alpha=0.3)
            
            title = (
                f"Sensor–Interpreter Richards fit ({s_name})\n"
                f"n={x.size}, x0={x0_hat:.2f}, b={b_hat:.4f}, nu={nu_hat:.3f}"
            )
            if type_vals is not None:
                title += "\nColored by forest_type"
            ax[0].set_title(title)
            ax[0].legend(frameon=True)
            
            # ---- Panel 2: residual vs sensor ----
            if type_vals is not None:
                for i, t in enumerate(unique_types):
                    m = (type_vals == t)
                    ax[1].scatter(x[m], resid[m], s=12, alpha=0.5, color=cmap(i))
            else:
                ax[1].scatter(x, resid, s=12, alpha=0.4, color="gray")
            
            ax[1].axhline(0.0, color="black", linewidth=1)
            ax[1].set_xlabel("Sensor value")
            ax[1].set_ylabel("Residual (interp − model)")
            ax[1].set_title("Residual vs sensor")
            ax[1].grid(alpha=0.3)

            plt.tight_layout()
            if save_dir is not None:
                fname = f"model_fit_{s_name.replace(' ', '_')}.png"
                fig.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches="tight")
            plt.show()


            

            # plt.figure(figsize=(7, 5))

            # if type_vals is not None:
            #     unique_types = np.unique(type_vals)
            #     cmap = plt.get_cmap("tab10")

            #     for i, t in enumerate(unique_types):
            #         m = (type_vals == t)
            #         plt.scatter(
            #             x[m],
            #             y[m],
            #             s=12,
            #             alpha=0.5,
            #             color=cmap(i),
            #             label=f"Type {int(t)}",
            #         )
            # else:
            #     plt.scatter(x, y, s=12, alpha=0.4, color="gray", label="Samples")

            # plt.plot(x_plot, y_plot, color="black", linewidth=2, label=f"Fitted logistic")
           
            # plt.xlabel("Sensor value")
            # plt.ylabel("Interpreter probability of loss")
            # plt.ylim(0, 1)

            # title = f"Sensor–Interpreter logistic fit ({s_name})\n" \
            #         f"n={x.size}, a={a_hat:.3f}, b={b_hat:.3f}"
            # if type_vals is not None:
            #     title += "\nColored by forest_type"
            # plt.title(title)

            # plt.legend(frameon=True)
            # plt.grid(alpha=0.3)
            # plt.tight_layout()
            # plt.show()




            
            # Grab example configs from the first stack (assuming template)
            example_stack = self.stacks[0]
            base_cfg = getattr(example_stack.base_landscape, "config", None)
            sensor_cfg = getattr(example_stack.sensor_fields[s_name], "config", None)
            interp_cfg = getattr(example_stack.interpreter_field, "config", None)

            model_info = {
                "sensor_name": s_name,
                "model_type": "richards",
                "params": {"x0": float(x0_hat), "b": float(b_hat), "nu": float(nu_hat)},
                "fit_covariance": pcov.tolist(),
                "n_samples": int(x.size),
                "strata": {
                    "n_strata": n_strata,
                    "samples_per_stratum": samples_per_stratum,
                },
                "experiment_config": self.experiment_config,
                "base_config_example": base_cfg,
                "sensor_config_example": sensor_cfg,
                "interpreter_config_example": interp_cfg,
            }

            self.models[s_name] = model_info

        return self.models


    def loadModel(self, model_data: Dict[str, Any]) -> None:
        """
        Load model information into this collection.

        model_data can be either:
          - a single model dict with keys 'sensor_name' and 'params', OR
          - a dict mapping sensor_name -> model_dict.

        The loaded models are stored in self.models.
        """
        if "sensor_name" in model_data and "params" in model_data:
            # Single model dict
            s_name = model_data["sensor_name"]
            self.models[s_name] = model_data
        else:
            # Assume mapping sensor_name -> model_dict
            for s_name, m in model_data.items():
                if not isinstance(m, dict):
                    continue
                self.models[s_name] = m




                
    def applyModel(
        self,
        sensor_names: Optional[List[str]] = None,
            ) -> None:
        """
        Apply the fitted logistic models to create new probability-of-loss
        surfaces for each sensor in each stack.

        For each stack and each sensor with a model:
          - Take sensor_field (x)
          - Compute p = logistic(a + b x)
          - Store as stack.calibrated_prob_by_sensor[sensor_name] (0–100).
        """
        if not self.models:
            raise RuntimeError("No models available. Call buildModel() or loadModel() first.")

        # Default: all sensors for which we have models
        if sensor_names is None:
            sensor_names = list(self.models.keys())
        else:
            sensor_names = list(sensor_names)

        for stack in self.stacks:
            if not hasattr(stack, "calibrated_prob_by_sensor"):
                stack.calibrated_prob_by_sensor = {}

            for s_name in sensor_names:
                model = self.models.get(s_name, None)
                if model is None:
                    continue

                if s_name not in stack.sensor_fields:
                    continue

                sensor = stack.sensor_fields[s_name]
                x = np.asarray(sensor.sensor_field, dtype=float)

                x0 = model["params"]["x0"]
                b  = model["params"]["b"]
                nu = model["params"]["nu"]
                p  = richards(x, x0, b, nu)  # 0..1

                stack.calibrated_prob_by_sensor[s_name] = p * 100.0

    # this is used by buildInterpreterModel, not called on its own. 
    def probability_stratified_sample(
                    self,
                    sensor_name: str,
                    n_strata: int = 5,
                    samples_per_stratum: int = 200,
                    sampling_seed: Optional[int] = int(time.time()),
                ) -> pd.DataFrame:
        """
        Draw a stratified sample from calibrated probability maps across all stacks.

        Stratification is by equal-width intervals of the calibrated probability
        (0–100%), based on global min/max across all stacks.

        Returns a DataFrame with columns:
          - 'stack_index'
          - 'stack_id'
          - 'row', 'col'
          - 'model_prob'      (0–100, from calibrated_prob_by_sensor)
          - 'interp_base_prob' (0–100, from interpreter_field.interpreter_field)
        """
        rng = np.random.default_rng(sampling_seed)

        # Collect global min/max of model probabilities for this sensor
        all_vals = []
        for stack in self.stacks:
            prob_img = stack.calibrated_prob_by_sensor.get(sensor_name, None)
            if prob_img is None:
                raise KeyError(
                    f"Stack {stack.stack_id} has no calibrated_prob_by_sensor[{sensor_name!r}]. "
                    "Did you call applyModel() on the training collection?"
                )
            all_vals.append(prob_img[np.isfinite(prob_img)])

        if not all_vals:
            raise RuntimeError("No calibrated probability values found for any stack.")

        all_vals_flat = np.concatenate(all_vals)
        global_min = float(all_vals_flat.min())
        global_max = float(all_vals_flat.max())

        if global_min == global_max:
            raise RuntimeError(
                f"All calibrated probabilities for sensor {sensor_name!r} "
                f"are the same value {global_min}."
            )

        # Equal-interval strata in [global_min, global_max]
        edges = np.linspace(global_min, global_max, n_strata + 1)

        rows_out = []
        for si, stack in enumerate(self.stacks):
            prob_img = stack.calibrated_prob_by_sensor[sensor_name]
            interp_img = stack.interpreter_field.interpreter_field  # 0–100

            nrows, ncols = prob_img.shape
            rr, cc = np.indices((nrows, ncols))

            flat_prob = prob_img.ravel()
            flat_interp = interp_img.ravel()
            flat_r = rr.ravel()
            flat_c = cc.ravel()

            for k in range(n_strata):
                lo = edges[k]
                hi = edges[k + 1]
                # include upper edge on last bin
                if k == n_strata - 1:
                    mask = (flat_prob >= lo) & (flat_prob <= hi)
                else:
                    mask = (flat_prob >= lo) & (flat_prob < hi)

                idx = np.where(mask)[0]
                if idx.size == 0:
                    continue

                n_draw = min(samples_per_stratum, idx.size)
                chosen = rng.choice(idx, size=n_draw, replace=False)

                for j in chosen:
                    rows_out.append({
                        "stack_index": si,
                        "stack_id": stack.stack_id,
                        "row": int(flat_r[j]),
                        "col": int(flat_c[j]),
                        "model_prob": float(flat_prob[j]),         # 0–100
                        "interp_base_prob": float(flat_interp[j])  # 0–100
                    })

        if not rows_out:
            raise RuntimeError("No samples drawn in any stratum; check your stratification settings.")

        return pd.DataFrame(rows_out)


    def buildInterpreterProbModel(
        self,
        sensor_name: str,
        n_strata: int = 5,
        samples_per_stratum: int = 200,
        sampling_seed: Optional[int] = int(time.time()),
        interpreter_agent: Optional[InterpreterAgent] = None,
    ) -> Dict[str, Any]:
        """
        Fit a RICHARDS function mapping MODEL probabilities (from calibrated_prob_by_sensor)
        -> INTERPRETER probabilities (after InterpreterAgent), using the logistic()
        helper from disturbance_helper_functions:

            y = logistic(x, x0, k)
              = 1 / (1 + exp(-k * (x - x0)))

        where x and y are in [0,1].

        Steps:
          1) Draw stratified samples from calibrated_prob_by_sensor via probability_stratified_sample.
          2) For each sample, take base interpreter probability from InterpreterField.
          3) If interpreter_agent is provided, transform base probabilities with it.
          4) Fit logistic(x0, k).
          5) Store model parameters in self.interpreter_prob_models[sensor_name].
        """
        # 1) Draw stratified samples (model_prob & interp_base_prob in 0–100)
        samples = self.probability_stratified_sample(
            sensor_name=sensor_name,
            n_strata=n_strata,
            samples_per_stratum=samples_per_stratum,
            sampling_seed=sampling_seed,
        )

        # X: model probabilities -> [0,1]
        x = samples["model_prob"].to_numpy(dtype=float)
        x01 = np.clip(x / 100.0, 0.0, 1.0)

        # Y: interpreter probabilities (0–100) -> [0,1]
        base_interp = samples["interp_base_prob"].to_numpy(dtype=float)

        if interpreter_agent is not None:
            y_interp = interpreter_agent.transform_probabilities(base_interp)
        else:
            y_interp = base_interp

        y01 = np.clip(y_interp / 100.0, 0.0, 1.0)

        # 3) Fit Richards via curve_fit with a data-adaptive pivot (x0 seeded from
        #    the median of x) and hard bounds.  Post-hoc endpoint rescaling then
        #    stretches/shifts the curve so that f_scaled(0)=0 and f_scaled(1)=1,
        #    correcting the OLS tendency to miss the corners.
        x0_init = float(np.clip(np.percentile(x01, 50), 0.01, 0.99))
        p0      = [x0_init, 5.0, 1.0]
        bounds  = ([0.0, 0.1, 0.01], [1.0, 50.0, 20.0])

        try:
            popt, _ = curve_fit(
                richards,
                x01, y01,
                p0=p0,
                bounds=bounds,
                maxfev=5000,
            )
            x0_hat, b_hat, nu_hat = float(popt[0]), float(popt[1]), float(popt[2])
        except Exception as e:
            print("Warning: curve_fit Richards failed; using defaults:", e)
            x0_hat, b_hat, nu_hat = 0.5, 5.0, 1.0

        # Post-hoc endpoint rescaling: stretch/shift so f_scaled(0)=0, f_scaled(1)=1.
        scale_lo   = float(richards(0.0, x0_hat, b_hat, nu_hat))
        scale_hi   = float(richards(1.0, x0_hat, b_hat, nu_hat))
        scale_span = scale_hi - scale_lo
        if scale_span < 1e-6:
            scale_lo, scale_hi, scale_span = 0.0, 1.0, 1.0

        def _scaled_richards(x: np.ndarray) -> np.ndarray:
            return (richards(x, x0_hat, b_hat, nu_hat) - scale_lo) / scale_span

        # -----------------------------
        # Diagnostic plot: fit + residuals
        # -----------------------------
        yhat  = _scaled_richards(x01)
        resid = y01 - yhat

        rmse = float(np.sqrt(np.mean(resid**2)))
        mae  = float(np.mean(np.abs(resid)))

        # Smooth fitted curve for display
        x_plot = np.linspace(float(np.min(x01)), float(np.max(x01)), 400)
        y_plot = _scaled_richards(x_plot)

        fig, ax = plt.subplots(1, 2, figsize=(14, 5))

        # ---- Panel 1: data + fit ----
        ax0 = ax[0]
        ax0.scatter(x01, y01, s=12, alpha=0.5, color="gray", label="Samples")
        ax0.plot(x_plot, y_plot, color="black", linewidth=2, label="Richards fit")

        ax0.set_title(
            f"Interpreter prob vs model prob (Richards)\n"
            f"sensor={sensor_name}, n={len(x01)}\n"
            f"x0={x0_hat:.3f}, b={b_hat:.3f}, nu={nu_hat:.3f}"
        )
        ax0.set_xlabel("Model probability x (0–1)")
        ax0.set_ylabel("Interpreter probability y (0–1)")
        ax0.set_xlim(0, 1)
        ax0.set_ylim(0, 1)
        ax0.grid(alpha=0.3)
        ax0.legend(frameon=True)

        ax0.text(
            0.02, 0.98,
            f"RMSE={rmse:.3f}\nMAE={mae:.3f}",
            transform=ax0.transAxes,
            va="top",
        )

        # ---- Panel 2: residuals vs x ----
        ax1 = ax[1]
        ax1.scatter(x01, resid, s=12, alpha=0.5, color="gray")
        ax1.axhline(0.0, color="black", linewidth=1)

        ax1.set_title("Residuals: (y − ŷ) vs model prob")
        ax1.set_xlabel("Model probability x (0–1)")
        ax1.set_ylabel("Residual (y − ŷ)")
        ax1.set_xlim(0, 1)
        ax1.grid(alpha=0.3)

        # Residual summaries by x-bin
        bins = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        bin_ids = np.digitize(x01, bins[1:-1], right=False)
        for i in range(len(bins) - 1):
            m = (bin_ids == i)
            if np.any(m):
                med = float(np.median(resid[m]))
                ax1.text(
                    0.02, 0.95 - i * 0.08,
                    f"{bins[i]:.1f}–{bins[i+1]:.1f}: median={med:+.3f} (n={m.sum()})",
                    transform=ax1.transAxes,
                    va="top",
                    fontsize=9,
                )

        plt.tight_layout()
        plt.show()
        print(f"Sample size: {len(x01)}")

        
        model_info = {
            "sensor_name": sensor_name,
            "model_type": "richards",
            "x0": float(x0_hat),
            "b": float(b_hat),
            "nu": float(nu_hat),
            "scale_lo": float(scale_lo),
            "scale_hi": float(scale_hi),
            "n_samples": int(len(x01)),
            "n_strata": int(n_strata),
            "samples_per_stratum": int(samples_per_stratum),
        }

        if not hasattr(self, "interpreter_prob_models"):
            self.interpreter_prob_models: Dict[str, Dict[str, Any]] = {}
        self.interpreter_prob_models[sensor_name] = model_info

        print(
            f"Fitted interpreter-probability model for sensor {sensor_name!r}: "
            f"x0={x0_hat:.3f}, b={b_hat:.3f}, nu={nu_hat:.3f}"
        )
        return model_info

  
    def applyInterpreterProbModel(
        self,
        sensor_name: str,
        model_info: Optional[Dict[str, Any]] = None,
        store_attr_name: str = "interpreter_calibrated_prob_by_sensor",
    ) -> None:
        """
        Apply a previously fitted interpreter-probability model to all stacks.
    
        Default (new): Richards generalized logistic
            y = richards(x, x0, b, nu) = (1 + exp(-b*(x-x0)))^(-nu)
    
        Backward-compatible: simple logistic
            y = logistic(x, x0, k) = 1 / (1 + exp(-k*(x-x0)))
    
        Inputs:
          - stack.calibrated_prob_by_sensor[sensor_name] in 0–100
        Outputs:
          - stack.<store_attr_name>[sensor_name] in 0–100
        """
        # ---- choose model_info ----
        if model_info is None:
            if (not hasattr(self, "interpreter_prob_models")) or (sensor_name not in self.interpreter_prob_models):
                raise KeyError(
                    f"No model_info provided and no stored interpreter_prob_models[{sensor_name!r}] found."
                )
            model_info = self.interpreter_prob_models[sensor_name]
    
        x0_hat     = float(model_info["x0"])
        b_hat      = float(model_info["b"])
        nu_hat     = float(model_info["nu"])
        scale_lo   = float(model_info.get("scale_lo", 0.0))
        scale_hi   = float(model_info.get("scale_hi", 1.0))
        scale_span = scale_hi - scale_lo
        if scale_span < 1e-6:
            scale_span = 1.0

        def _apply_fn(x01: np.ndarray) -> np.ndarray:
            raw = richards(x01, x0=x0_hat, b=b_hat, nu=nu_hat)
            return np.clip((raw - scale_lo) / scale_span, 0.0, 1.0)
    
        # ---- apply to each stack ----
        for stack in self.stacks:
            # get calibrated model probs (0–100)
            prob_img = None
            if hasattr(stack, "calibrated_prob_by_sensor"):
                prob_img = stack.calibrated_prob_by_sensor.get(sensor_name, None)
    
            if prob_img is None:
                raise KeyError(
                    f"Stack {stack.stack_id} has no calibrated_prob_by_sensor[{sensor_name!r}]."
                )
    
            x = np.asarray(prob_img, dtype=float)
            x01 = np.clip(x / 100.0, 0.0, 1.0)
    
            y01 = _apply_fn(x01)
            y_pct = y01 * 100.0
    
            # store result
            if not hasattr(stack, store_attr_name):
                setattr(stack, store_attr_name, {})
            getattr(stack, store_attr_name)[sensor_name] = y_pct
        
    def calc_area_distribution_from_prob_image(
        self,
        prob_image,
        pixel_area: float = 1.0,
        step: float = 0.001,
        bins: int = 100,
    ):
        """
        Compute a distribution of disturbed-area estimates induced by uncertainty
        in the probability cutoff.

        For each threshold t in [0,1) with spacing `step`:
            A(t) = (# pixels with prob > t) * pixel_area

        Then build a histogram of A(t) values.

        Parameters
        ----------
        prob_image : array-like
            2D array of probabilities. Can be 0–1 or 0–100.
        pixel_area : float, optional
            Area represented by one pixel (e.g., 900 for 30m pixels). Default 1.0.
        step : float, optional
            Threshold spacing in [0,1). Default 0.001.
        bins : int, optional
            Number of histogram bins. Default 100.

        Returns
        -------
        thresholds : np.ndarray
            Thresholds in [0,1).
        areas : np.ndarray
            Area above each threshold, in units of pixel_area.
        hist : np.ndarray
            Histogram of 'areas' (density=True).
        bin_edges : np.ndarray
            Histogram bin edges.
        """

        prob = np.asarray(prob_image, dtype=float)

        # Convert 0–100 -> 0–1 if needed
        if np.nanmax(prob) > 1.0 + 1e-6:
            prob = prob / 100.0
        prob = np.clip(prob, 0.0, 1.0)

        thresholds = np.arange(0.0, 1.0, step)
        areas = np.empty_like(thresholds, dtype=float)

        # Count pixels above threshold for each t
        for i, t in enumerate(thresholds):
            areas[i] = float((prob > t).sum()) * pixel_area

        hist, bin_edges = np.histogram(areas, bins=bins, density=True)
        return thresholds, areas, hist, bin_edges

    
    
    # trying to do this with the gaussian approach. 

    

    
    def calc_area_distribution_spatial_threshold_for_stack(
        self,
        stack_index: int,
        sensor_name: str,
        prob_source_attr: str = "interpreter_calibrated_prob_by_sensor",
        pixel_area: float = 1.0,
        n_iter: int = 1000,
        m: float | tuple[float, float] = 0.5,
        l: float | tuple[float, float] = 10.0,
        sigma: float = 0.15,
        seed: Optional[int] = int(time.time()),
        bins: int = 100,
    ):
        """
        Spatial-threshold Monte Carlo area distribution.
    
        For each iteration:
          1) Generate spatially correlated Gaussian field G(x) ~ N(0,1) with len_scale=l.
          2) Build threshold surface T(x) = clip(m + sigma * G(x), 0..1).
          3) Disturbed mask = prob01 > T(x)
          4) Area = disturbed.sum() * pixel_area
    
        Parameters
        ----------
        m : float OR (low,high) OR callable OR list
            Mean of the threshold surface (center in [0,1]).
            If tuple, draws m ~ Uniform(low,high) each iteration.
        l : float OR (low,high) OR callable OR list
            Spatial autocorrelation length in *pixels* (len_scale for gstools Gaussian model).
            If tuple, draws l ~ Uniform(low,high) each iteration.
        sigma : float
            Std dev of threshold surface about m (before clipping). Bigger => more spatial variability.
        """
        if gs is None:
            raise ImportError(
                "gstools is required for spatial-threshold Monte Carlo. "
                "Install with `conda install -c conda-forge gstools`."
            )
    
        stack = self.stacks[stack_index]
    
        # ---- get probability image (expect 0–100 stored) ----
        prob_dict = getattr(stack, prob_source_attr, None)
        if not isinstance(prob_dict, dict) or sensor_name not in prob_dict:
            raise AttributeError(
                f"Stack {stack.stack_id} missing {prob_source_attr}[{sensor_name!r}]."
            )
    
        prob_img = np.asarray(prob_dict[sensor_name], dtype=float)
        prob01 = np.clip(prob_img / 100.0, 0.0, 1.0)
    
        # ---- valid mask (ignore NaNs) ----
        valid = np.isfinite(prob01)
        if not np.any(valid):
            raise RuntimeError("Probability image has no finite values.")
    
        nrows, ncols = prob01.shape
        total_area = float(valid.sum()) * float(pixel_area)  # NOTE: total over valid pixels only
    
        rng = np.random.default_rng(seed)
    
        # Prebuild grid coords once
        x = np.arange(ncols, dtype=float)
        y = np.arange(nrows, dtype=float)
        xs, ys = np.meshgrid(x, y, indexing="xy")
        coords = (xs.ravel(), ys.ravel())
    
        areas = np.empty(n_iter, dtype=float)
        exceedance_acc = np.zeros((nrows, ncols), dtype=float)
        thr_acc = np.zeros((nrows, ncols), dtype=float)

        for i in range(n_iter):
            m_i = _resolve_sampler(m, rng)
            l_i = _resolve_sampler(l, rng)

            # keep in bounds
            m_i = float(np.clip(m_i, 0.0, 1.0))
            l_i = max(float(l_i), 1e-6)

            # Spatial Gaussian random field G ~ N(0,1) correlated
            model = gs.Gaussian(dim=2, var=1.0, len_scale=l_i)
            srf = gs.SRF(model, seed=int(rng.integers(0, 2**31 - 1)))

            g = srf(coords).reshape((nrows, ncols))  # ~N(0,1)

            # Transform N(0,1) field to uniform [0,1] via normal CDF
            thr = norm.cdf(g)

            # Apply mask (valid pixels only)
            disturbed = valid & (prob01 > thr)

            areas[i] = float(disturbed.sum()) * float(pixel_area)
            exceedance_acc += disturbed.astype(float)
            thr_acc += thr

        exceedance_freq = exceedance_acc / n_iter
        mean_thr = thr_acc / n_iter

        # Histogram of areas (density)
        hist, bin_edges = np.histogram(areas, bins=bins, density=True)

        # Also provide proportion distribution for convenience (over valid pixels)
        proportions = areas / total_area if total_area > 0 else np.full_like(areas, np.nan)

        return {
            "areas": areas,
            "proportions": proportions,
            "hist": hist,
            "bin_edges": bin_edges,
            "exceedance_freq": exceedance_freq,
            "mean_thr": mean_thr,
            "pixel_area": float(pixel_area),
            "total_area_valid": float(total_area),
            "n_iter": int(n_iter),
            "m": m,
            "l": l,
            "sigma": float(sigma),
            "seed": seed,
            "prob_source_attr": prob_source_attr,
            "sensor_name": sensor_name,
            "stack_id": stack.stack_id,
        }
    
    
    def plot_spatial_threshold_realization_for_stack(
        self,
        stack_index: int,
        sensor_name: str,
        prob_source_attr: str = "interpreter_calibrated_prob_by_sensor",
        m: float = 0.5,
        l: float = 10.0,
        sigma: float = 0.15,
        seed: int | None = None,
        show_true: bool = True,
        figsize: tuple[int, int] = (15, 5),
    ):
        """
        Diagnostic plot for the spatial-threshold Monte Carlo approach.
    
        Generates ONE spatially correlated threshold surface T(x) and plots:
          (1) prob map (0..1)
          (2) threshold field (0..1)
          (3) disturbed mask = prob > threshold
    
        Optionally overlays the true disturbed mask outline (if available).
        """
        if gs is None:
            raise ImportError(
                "gstools is required for spatial-threshold plots. "
                "Install with `conda install -c conda-forge gstools`."
            )
    
        stack = self.stacks[stack_index]
    
        # ---- get probability image (expect stored 0–100) ----
        prob_dict = getattr(stack, prob_source_attr, None)
        if not isinstance(prob_dict, dict) or sensor_name not in prob_dict:
            raise AttributeError(
                f"Stack {stack.stack_id} missing {prob_source_attr}[{sensor_name!r}]."
            )
    
        prob_img = np.asarray(prob_dict[sensor_name], dtype=float)
        prob01 = np.clip(prob_img / 100.0, 0.0, 1.0)
        valid = np.isfinite(prob01)
    
        if not np.any(valid):
            raise RuntimeError("Probability image has no finite values.")
    
        nrows, ncols = prob01.shape
    
        # ---- build spatial GRF G ~ N(0,1) with correlation length l ----
        l = max(float(l), 1e-6)
    
        rng = np.random.default_rng(seed)
        model = gs.Gaussian(dim=2, var=1.0, len_scale=l)
        srf = gs.SRF(model, seed=int(rng.integers(0, 2**31 - 1)))
    
        x = np.arange(ncols, dtype=float)
        y = np.arange(nrows, dtype=float)
        xs, ys = np.meshgrid(x, y, indexing="xy")
        g = srf((xs.ravel(), ys.ravel())).reshape((nrows, ncols))
    
        thr = np.clip(float(m) + float(sigma) * g, 0.0, 1.0)
        disturbed = valid & (prob01 > thr)
    
        # ---- optional true disturbed mask ----
        true_mask = None
        if show_true:
            try:
                true_mask = np.asarray(stack.base_landscape.disturbed, dtype=bool)
                if true_mask.shape != disturbed.shape:
                    true_mask = None
            except Exception:
                true_mask = None
    
        # ---- plot ----
        fig, ax = plt.subplots(1, 3, figsize=figsize)
    
        im0 = ax[0].imshow(prob01, origin="upper", vmin=0.0, vmax=1.0)
        ax[0].set_title(f"Probability map (source={prob_source_attr})\nstack={stack.stack_id}, sensor={sensor_name}")
        ax[0].set_xticks([]); ax[0].set_yticks([])
        plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
    
        im1 = ax[1].imshow(thr, origin="upper", vmin=0.0, vmax=1.0)
        ax[1].set_title(f"Spatial threshold field T(x)\n(m={m:.2f}, l={l:.2f}px, sigma={sigma:.2f})")
        ax[1].set_xticks([]); ax[1].set_yticks([])
        plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
    
        # disturbed mask (binary)
        im2 = ax[2].imshow(disturbed.astype(int), origin="upper", vmin=0, vmax=1)
        ax[2].set_title("Disturbed mask: prob > T(x)")
        ax[2].set_xticks([]); ax[2].set_yticks([])
        plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04, ticks=[0, 1])
    
        # Overlay true mask outline (optional)
        if true_mask is not None:
            for a in ax:
                a.contour(true_mask.astype(int), levels=[0.5], colors ="white", linewidths=1)
    
        plt.tight_layout()
        plt.show()
    
        # Return arrays for further debugging if you want
        return {
            "prob01": prob01,
            "threshold": thr,
            "disturbed": disturbed,
            "true_disturbed": true_mask,
            "m": float(m),
            "l": float(l),
            "sigma": float(sigma),
            "seed": seed,
            "stack_id": stack.stack_id,
            "sensor_name": sensor_name,
            "prob_source_attr": prob_source_attr,
        }


    #wrapper 
    def calc_spatial_area_distribution_for_stack(
        self,
        stack_index: int,
        sensor_name: str,
        prob_source_attr: str = "interpreter_calibrated_prob_by_sensor",
        pixel_area: float = 1.0,
        n_iter: int = 1000,
        m: float | tuple[float, float] = 0.5,
        l: float | tuple[float, float] = 10.0,
        sigma: float = 0.15,
        seed: int | None = None,
        bins: int = 80,
    ) -> Dict[str, Any]:
        """
        Wrapper around calc_area_distribution_spatial_threshold_for_stack(...),
        returning both absolute area and proportion-of-landscape distributions.
        """
        out = self.calc_area_distribution_spatial_threshold_for_stack(
            stack_index=stack_index,
            sensor_name=sensor_name,
            prob_source_attr=prob_source_attr,
            pixel_area=pixel_area,
            n_iter=n_iter,
            m=m,
            l=l,
            sigma=sigma,
            seed=seed,
            bins=bins,
        )
    
        areas = out["areas"]
        props = out["proportions"]
    
        # Helpful summary stats
        pct = {
            "p05": float(np.nanpercentile(props, 5)),
            "p10": float(np.nanpercentile(props, 10)),
            "p50": float(np.nanpercentile(props, 50)),
            "p90": float(np.nanpercentile(props, 90)),
            "p95": float(np.nanpercentile(props, 95)),
            "mean": float(np.nanmean(props)),
            "std": float(np.nanstd(props)),
        }
    
        out["proportion_summary"] = pct
        return out


    def plot_spatial_area_distribution_for_stack(
        self,
        stack_index: int,
        sensor_name: str,
        prob_source_attr: str = "interpreter_calibrated_prob_by_sensor",
        pixel_area: float = 1.0,
        n_iter: int = 1000,
        m: float | tuple[float, float] = 0.5,
        l: float | tuple[float, float] = 10.0,
        sigma: float = 0.15,
        seed: int | None = None,
        bins: int = 80,
        show_true: bool = True,
        show_realization_maps: bool = True,
        x_prop_range: Optional[tuple[float, float]] = None,  # NEW: enforce x-range for proportion histograms
    ) -> Dict[str, Any]:
        """
        Plot a spatial-threshold Monte Carlo area/proportion distribution.
    
        Produces:
          Figure 1: Histogram(s) of disturbed PROPORTION distribution (and key percentiles).
          Optionally Figure 2: realization map diagnostic (prob, threshold, mask).
        """
       
        out = self.calc_spatial_area_distribution_for_stack(
            stack_index=stack_index,
            sensor_name=sensor_name,
            prob_source_attr=prob_source_attr,
            pixel_area=pixel_area,
            n_iter=n_iter,
            m=m,
            l=l,
            sigma=sigma,
            seed=seed,
            bins=bins,
        )
    
        stack = self.stacks[stack_index]
    
        props = np.asarray(out["proportions"], dtype=float)
        props = props[np.isfinite(props)]
        pct = out["proportion_summary"]
    
        # ---- true disturbed proportion (optional) ----
        true_prop = None
        if show_true:
            try:
                truth = np.asarray(stack.base_landscape.disturbed, dtype=bool)
                true_prop = float(truth.mean())  # FULL denominator (consistent with Olofsson-style truth)
            except Exception:
                true_prop = None
    
        # ---- Figure 1: histograms of disturbed proportion ----
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    
        # Left: density histogram of proportion
        ax0 = ax[0]
        ax0.hist(props, bins=bins, density=True, alpha=0.7)
        ax0.set_title(
            f"Spatial-threshold disturbed proportion distribution\n"
            f"stack={stack_index} ({stack.stack_id}), sensor={sensor_name}\n"
            f"source={prob_source_attr}, n_iter={n_iter}, m={m}, l={l}, sigma={sigma}"
        )
        ax0.set_xlabel("Disturbed proportion (of full landscape)")
        ax0.set_ylabel("Density")
    
        if x_prop_range is not None:
            ax0.set_xlim(*x_prop_range)
    
        # Percentile lines (10/50/90)
        ax0.axvline(pct["p10"], linestyle="--", linewidth=1, color="gray")
        ax0.axvline(pct["p50"], linestyle="-", linewidth=2, color="black")
        ax0.axvline(pct["p90"], linestyle="--", linewidth=1, color="gray")
    
        ax0.text(
            0.02, 0.98,
            f"p10={pct['p10']:.4f}\n"
            f"p50={pct['p50']:.4f}\n"
            f"p90={pct['p90']:.4f}\n"
            f"p05={pct['p05']:.4f}\n"
            f"p95={pct['p95']:.4f}",
            transform=ax0.transAxes, va="top"
        )
    
        if true_prop is not None:
            ax0.axvline(true_prop, linestyle=":", linewidth=2, color="red")
            ax0.text(0.62, 0.98, f"True prop={true_prop:.4f}", transform=ax0.transAxes, va="top")
    
        # Right: same distribution but as COUNTS (helpful when comparing runs)
        ax1 = ax[1]
        ax1.hist(props, bins=bins, density=False, alpha=0.7)
        ax1.set_title("Same distribution (counts)")
        ax1.set_xlabel("Disturbed proportion (of full landscape)")
        ax1.set_ylabel("Count")
    
        if x_prop_range is not None:
            ax1.set_xlim(*x_prop_range)
    
        # Add same percentile + truth markers for consistency
        ax1.axvline(pct["p10"], linestyle="--", linewidth=1, color="gray")
        ax1.axvline(pct["p50"], linestyle="-", linewidth=2, color="black")
        ax1.axvline(pct["p90"], linestyle="--", linewidth=1, color="gray")
        if true_prop is not None:
            ax1.axvline(true_prop, linestyle=":", linewidth=2, color="red")
    
        plt.tight_layout()
        plt.show()
    
        # ---- Figure 2: one realization map diagnostic ----
        if show_realization_maps:
            # Keep your existing behavior for picking a single realization.
            # (Note: this uses your existing _resolve_sampler helper.)
            self.plot_spatial_threshold_realization_for_stack(
                stack_index=stack_index,
                sensor_name=sensor_name,
                prob_source_attr=prob_source_attr,
                m=float(_resolve_sampler(m, np.random.default_rng(seed))) if not isinstance(m, (int, float)) else float(m),
                l=float(_resolve_sampler(l, np.random.default_rng(seed))) if not isinstance(l, (int, float)) else float(l),
                sigma=float(sigma),
                seed=seed,
                show_true=show_true,
            )
    
        # ---- Print an “Olofsson-like” summary for proportions ----
        print("\n=== Spatial-threshold area distribution summary (proportions) ===")
        print(f"stack={stack_index} ({stack.stack_id}), sensor={sensor_name}, source={prob_source_attr}")
        print(f"Median disturbed proportion (class 1): {pct['p50']:.6f}")
        print(f"5th–95th percentile (class 1):       [{pct['p05']:.6f}, {pct['p95']:.6f}]")
        print(f"Median class 0 proportion:            {1.0 - pct['p50']:.6f}")
        print(f"5th–95th (class 0):                   [{1.0 - pct['p95']:.6f}, {1.0 - pct['p05']:.6f}]")
        if true_prop is not None:
            inside = (pct["p05"] <= true_prop <= pct["p95"])
            print(f"True disturbed proportion:            {true_prop:.6f}")
            print(f"Is true within 5th–95th?              {inside}")
        print("===============================================================\n")
    
        out["true_disturbed_proportion"] = true_prop
        return out
        


    # the simpler version where the slice is uniform. 
    def calc_area_distribution_for_stack(
        self,
        stack_index: int,
        sensor_name: str,
        prob_source_attr: str = "interpreter_calibrated_prob_by_sensor",
        pixel_area: float = 1.0,
        step: float = 0.001,
        bins: int = 100,
    ):
        """
        Convenience wrapper: fetch a probability image from a given stack and sensor,
        then compute the area distribution over thresholds.

        prob_source_attr examples:
          - "calibrated_prob_by_sensor"
          - "interpreter_calibrated_prob_by_sensor"
          - any stack attribute that is a dict keyed by sensor_name

        Returns (thresholds, areas, hist, bin_edges).
        """
        if not (0 <= stack_index < len(self.stacks)):
            raise IndexError(f"stack_index {stack_index} out of range for {len(self.stacks)} stacks.")

        stack = self.stacks[stack_index]

        if not hasattr(stack, prob_source_attr):
            raise AttributeError(f"Stack {stack.stack_id} has no attribute {prob_source_attr!r}.")

        d = getattr(stack, prob_source_attr)
        if not isinstance(d, dict):
            raise TypeError(f"stack.{prob_source_attr} is not a dict; got {type(d)}")

        if sensor_name not in d:
            raise KeyError(f"{sensor_name!r} not found in stack.{prob_source_attr} for stack {stack.stack_id}")

        prob_img = d[sensor_name]
        return self.calc_area_distribution_from_prob_image(
            prob_image=prob_img,
            pixel_area=pixel_area,
            step=step,
            bins=bins,
        )
        
    def plot_area_distribution_for_stack(
        self,
        stack_index: int,
        sensor_name: str,
        prob_source_attr: str = "interpreter_calibrated_prob_by_sensor",
        pixel_area: float = 1.0,
        step: float = 0.001,
        bins: int = 100,
        show_true_area: bool = True,
        show_diagnostics: bool = True,
        signal_attr: str = "sensor_fields_by_name",
        x_prop_range: Optional[tuple[float, float]] = None,
        save_dir: Optional[str] = None,
        filename_prefix: Optional[str] = None,
    ) -> tuple:
        """
        Figure 1 (1x2):
          (A) Area above cutoff t: A(t) = (# pixels with prob > t) * pixel_area
          (B) Histogram of disturbed AREA PROPORTIONS (= A(t) / total_area)
    
        Figure 2 (Diagnostics, 2x3) if show_diagnostics=True:
          Top row maps:
            (1) Sensor degraded signal image
            (2) Probability image after training model (stack.calibrated_prob_by_sensor)
            (3) Probability image after interpreter adjustment (stack.<prob_source_attr>)
          Bottom row histograms for the same three images.
    
        Notes
        -----
        - Probability images may be 0–1 or 0–100; displayed consistently.
        - Sensor signal is displayed as-is (continuous numeric).
        - Histogram in Fig 1 is in PROPORTION OF LANDSCAPE (0–1), for easy comparison to binary case.
        """
    
        # ---------- compute area distribution using the probability source requested ----------
        thresholds, areas, hist, bin_edges = self.calc_area_distribution_for_stack(
            stack_index=stack_index,
            sensor_name=sensor_name,
            prob_source_attr=prob_source_attr,
            pixel_area=pixel_area,
            step=step,
            bins=bins,
        )
    
        stack = self.stacks[stack_index]
    
        # ---------- total area for proportion conversion ----------
        try:
            truth_mask = np.asarray(stack.base_landscape.disturbed, dtype=bool)
            n_pixels_total = int(truth_mask.size)
        except Exception:
            # fallback: infer from probability source image
            prob_img_any = getattr(stack, prob_source_attr, None)
            if isinstance(prob_img_any, dict) and sensor_name in prob_img_any:
                n_pixels_total = int(np.asarray(prob_img_any[sensor_name]).size)
            else:
                raise RuntimeError("Could not determine total pixel count for proportion conversion.")
    
        total_area = float(n_pixels_total) * float(pixel_area)
    
        # ---------- convert areas -> proportions ----------
        areas_arr = np.asarray(areas, dtype=float)
        areas_arr = areas_arr[np.isfinite(areas_arr)]
        area_props = np.clip(areas_arr / total_area, 0.0, 1.0)
    
        # ---------- percentiles of proportion distribution ----------
        p10, p50, p90 = np.percentile(area_props, [10, 50, 90])
    
        # ---------- true disturbed area/proportion (optional) ----------
        true_area = None
        true_prop = None
        if show_true_area:
            try:
                truth = np.asarray(stack.base_landscape.disturbed, dtype=bool)
                true_area = float(truth.sum()) * pixel_area
                true_prop = float(truth.mean())
            except Exception:
                true_area = None
                true_prop = None
    
        # ---------- FIGURE 1: area curve + histogram (PROPORTIONS) ----------
        fig1, ax = plt.subplots(1, 2, figsize=(12, 5))
    
        # (A) Area vs threshold (absolute)
        ax0 = ax[0]
        ax0.plot(thresholds, areas)
        ax0.set_title(
            f"Area above cutoff t\nstack={stack_index} ({stack.stack_id}), sensor={sensor_name}\nsource={prob_source_attr}"
        )
        ax0.set_xlabel("Probability cutoff t (0–1)")
        ax0.set_ylabel(f"Area (units of pixel_area={pixel_area})")
        if true_area is not None:
            ax0.axhline(true_area, linestyle="--", linewidth=1)
            ax0.text(
                0.12, 0.98,
                f"True disturbed area: {true_area:,.2f}",
                transform=ax0.transAxes, va="top"
            )
    
        # (B) Histogram of disturbed proportion distribution
        ax1 = ax[1]
        ax1.hist(area_props, bins=bins, density=True, alpha=0.7)
        ax1.set_title("Distribution of disturbed area (proportion)\n(from varying cutoff t)")
        ax1.set_xlabel("Proportion of landscape disturbed")
        ax1.set_ylabel("Density")
    
        # Optional consistent x-range
        if x_prop_range is not None:
            ax1.set_xlim(*x_prop_range)
    
        # True reference line (proportion)
        if true_prop is not None:
            ax1.axvline(true_prop, linestyle="--", linewidth=3, color="red")
            ax1.text(
                0.50, 0.98,
                f"True disturbed proportion: {true_prop:.3f}",
                transform=ax1.transAxes, va="top"
            )
    
        # Percentile markers (proportion space)
        ax1.axvline(p10, linestyle=":", linewidth=2, color="gray")
        ax1.axvline(p50, linestyle="-", linewidth=3, color="black")
        ax1.axvline(p90, linestyle=":", linewidth=2, color="gray")
    
        ax1.text(0.50, 0.90, f"P10: {p10:.3f}", transform=ax1.transAxes, va="top")
        ax1.text(0.50, 0.84, f"P50: {p50:.3f}", transform=ax1.transAxes, va="top")
        ax1.text(0.50, 0.78, f"P90: {p90:.3f}", transform=ax1.transAxes, va="top")
    
        plt.tight_layout()
        if save_dir is not None:
            _prefix = filename_prefix or f"stack_{stack_index:03d}_{stack.stack_id}"
            fig1.savefig(
                os.path.join(save_dir, f"{_prefix}_area_dist.png"), dpi=150, bbox_inches="tight"
            )
        plt.show()

        print(f"10th, 50th, and 90th percentile DISTURBED PROPORTIONS: {p10:.6f}, {p50:.6f}, {p90:.6f}")
        if true_prop is not None:
            print(f"True disturbed proportion: {true_prop:.6f}")
        else:
            print("True disturbed proportion: (unavailable)")

        # ---------- FIGURE 2: diagnostics (maps + histograms) ----------
        if not show_diagnostics:
            return (fig1, None)
    
        # (1) Sensor degraded signal image
        if not hasattr(stack, "sensor_fields") or not isinstance(stack.sensor_fields, dict):
            raise AttributeError(
                f"Stack {stack.stack_id} has no dict attribute 'sensor_fields'. "
                "Expected stack.sensor_fields[sensor_name].sensor_field"
            )
    
        if sensor_name not in stack.sensor_fields:
            raise KeyError(
                f"Sensor {sensor_name!r} not found in stack.sensor_fields for stack {stack.stack_id}. "
                f"Available sensors: {list(stack.sensor_fields.keys())}"
            )
    
        sensor_obj = stack.sensor_fields[sensor_name]
        if not hasattr(sensor_obj, "sensor_field"):
            raise AttributeError(
                f"stack.sensor_fields[{sensor_name!r}] has no attribute 'sensor_field'. "
                f"Got type={type(sensor_obj)}"
            )
    
        signal_img = np.asarray(sensor_obj.sensor_field, dtype=float)
    
        # (2) Probability after training model
        if not hasattr(stack, "calibrated_prob_by_sensor"):
            raise AttributeError(
                f"Stack {stack.stack_id} lacks 'calibrated_prob_by_sensor'. Did you run applyModel()?"
            )
        if sensor_name not in stack.calibrated_prob_by_sensor:
            raise KeyError(f"{sensor_name!r} not found in stack.calibrated_prob_by_sensor for stack {stack.stack_id}")
        prob_train = np.asarray(stack.calibrated_prob_by_sensor[sensor_name], dtype=float)
    
        # (3) Probability after interpreter adjustment (uses prob_source_attr)
        if not hasattr(stack, prob_source_attr):
            raise AttributeError(f"Stack {stack.stack_id} has no attribute {prob_source_attr!r}.")
        d_adj = getattr(stack, prob_source_attr)
        if not isinstance(d_adj, dict):
            raise TypeError(f"stack.{prob_source_attr} is not a dict; got {type(d_adj)}")
        if sensor_name not in d_adj:
            raise KeyError(f"{sensor_name!r} not found in stack.{prob_source_attr} for stack {stack.stack_id}")
        prob_adj = np.asarray(d_adj[sensor_name], dtype=float)
    
        # Normalize probabilities to 0–1 for consistent hist/visual scaling
        def _to01(arr):
            a = np.asarray(arr, dtype=float)
            if np.nanmax(a) > 1.0 + 1e-6:
                a = a / 100.0
            return np.clip(a, 0.0, 1.0)
    
        prob_train01 = _to01(prob_train)
        prob_adj01 = _to01(prob_adj)
    
        # ---------- Make 2x3 figure ----------
        fig2, axm = plt.subplots(2, 3, figsize=(16, 9))
    
        # Top row: maps
        im0 = axm[0, 0].imshow(signal_img, origin="upper")
        axm[0, 0].set_title(f"Sensor signal\n({sensor_name})")
        fig2.colorbar(im0, ax=axm[0, 0], fraction=0.046, pad=0.04)
    
        im1 = axm[0, 1].imshow(prob_train01, origin="upper", vmin=0.0, vmax=1.0)
        axm[0, 1].set_title("Probability after training model\n(calibrated_prob_by_sensor)")
        fig2.colorbar(im1, ax=axm[0, 1], fraction=0.046, pad=0.04)
    
        im2 = axm[0, 2].imshow(prob_adj01, origin="upper", vmin=0.0, vmax=1.0)
        axm[0, 2].set_title(f"Probability after interpreter match\n({prob_source_attr})")
        fig2.colorbar(im2, ax=axm[0, 2], fraction=0.046, pad=0.04)
    
        # Remove ticks + add borders
        for j in range(3):
            axm[0, j].set_xticks([])
            axm[0, j].set_yticks([])
            for side in ["left", "right", "top", "bottom"]:
                axm[0, j].spines[side].set_visible(True)
                axm[0, j].spines[side].set_edgecolor("black")
                axm[0, j].spines[side].set_linewidth(1.2)
    
        # Bottom row: histograms
        axm[1, 0].hist(signal_img.ravel(), bins=60, density=True, alpha=0.7)
        axm[1, 0].set_title("Histogram: sensor signal")
        axm[1, 0].set_xlabel("Signal value")
        axm[1, 0].set_ylabel("Density")
    
        axm[1, 1].hist(prob_train01.ravel(), bins=60, density=True, alpha=0.7)
        axm[1, 1].set_title("Histogram: prob (training model)")
        axm[1, 1].set_xlabel("Probability (0–1)")
        axm[1, 1].set_ylabel("Density")
    
        axm[1, 2].hist(prob_adj01.ravel(), bins=60, density=True, alpha=0.7)
        axm[1, 2].set_title("Histogram: prob (after interpreter match)")
        axm[1, 2].set_xlabel("Probability (0–1)")
        axm[1, 2].set_ylabel("Density")
    
        plt.tight_layout()
        if save_dir is not None:
            _prefix = filename_prefix or f"stack_{stack_index:03d}_{stack.stack_id}"
            fig2.savefig(
                os.path.join(save_dir, f"{_prefix}_diagnostics.png"), dpi=150, bbox_inches="tight"
            )
        plt.show()

        # -- Proportion-of-area reporting (analogous to Olofsson-style summaries) --
        p1_dist = area_props
        p0_dist = 1.0 - p1_dist
    
        p1_05, p1_50, p1_95 = np.percentile(p1_dist, [5, 50, 95])
        p0_05, p0_50, p0_95 = np.percentile(p0_dist, [5, 50, 95])
    
        print("\n================ AREA PROPORTION SUMMARY (Probability-based) ================")
        print(f"Stack: {stack_index} ({stack.stack_id})   Sensor: {sensor_name}   Source: {prob_source_attr}")
        print(f"Total pixels: {n_pixels_total:,}   pixel_area: {pixel_area}   total_area: {total_area:,.3f}\n")
    
        print("Estimated disturbed proportion P(class 1) from cutoff-sweep distribution:")
        print(f"  P5 :  {p1_05:.6f}")
        print(f"  P50:  {p1_50:.6f}   (median)")
        print(f"  P95:  {p1_95:.6f}")
    
        print("\nEstimated undisturbed proportion P(class 0) = 1 - P(class 1):")
        print(f"  P5 :  {p0_05:.6f}")
        print(f"  P50:  {p0_50:.6f}   (median)")
        print(f"  P95:  {p0_95:.6f}")
    
        if true_prop is not None:
            within = (true_prop >= p1_05) and (true_prop <= p1_95)
            print("\nTruth (from base_landscape.disturbed):")
            print(f"  True disturbed proportion: {true_prop:.6f}")
            print(f"  Is truth within [P5, P95] of estimated P(class 1)? {within}")
        else:
            print("\nTruth disturbed proportion not available (could not read base_landscape.disturbed).")
            print("=============================================================================\n")

        return (fig1, fig2)









        
            
    def save_all_stack_plots(
        self,
        save_dir: str,
        sensor_name: str,
        prob_source_attr: str = "interpreter_calibrated_prob_by_sensor",
        pixel_area: float = 1.0,
        step: float = 0.001,
        bins: int = 100,
        x_prop_range: Optional[tuple] = None,
        show_diagnostics: bool = True,
    ) -> None:
        """
        Generate and save area-distribution figures for every stack in the collection.

        Files are written to save_dir as:
            stack_000_{stack_id}_area_dist.png
            stack_000_{stack_id}_diagnostics.png
            ...

        Figures are closed after saving to keep memory use low.
        """
        import matplotlib
        for si, stack in enumerate(self.stacks):
            prefix = f"stack_{si:03d}_{stack.stack_id}"
            self.plot_area_distribution_for_stack(
                stack_index=si,
                sensor_name=sensor_name,
                prob_source_attr=prob_source_attr,
                pixel_area=pixel_area,
                step=step,
                bins=bins,
                show_true_area=True,
                show_diagnostics=show_diagnostics,
                x_prop_range=x_prop_range,
                save_dir=save_dir,
                filename_prefix=prefix,
            )
            plt.close("all")

    ###------------------------------------------------------------------------------
    #
    #               Binary classifiers
    #
    ###------------------------------------------------------------------------------
    

    def buildBinaryClassifier(
        self,
        sensor_names: Optional[List[str]] = None,
        n_strata: int = 5,
        samples_per_stratum: int = 200,
        sampling_seed: Optional[int] = int(time.time()),
        fp_to_fn_ratio: float = 1.0,
        interpreter_agent: Optional[InterpreterAgent] = None,
        samples: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Build a threshold-based binary classifier for one or more sensors.

        Steps for each sensor:
          1) Convert interpreter probabilities to binary labels:
                y = 1 if interpreter_prob >= 50%, else 0
          2) For many candidate thresholds t (sensor values), classify:
                y_hat = 1 if sensor_value >= t else 0
             and compute false positive rate (FPR) and false negative rate (FNR).
          3) Using a cost ratio between false positives and false negatives,
             choose the threshold t that minimizes:
                cost(t) = fp_to_fn_ratio * FPR(t) + 1 * FNR(t)
          4) Store the chosen threshold and full curves in self.binary_models[sensor].

        Returns
        -------
        self.binary_models : dict
        """
        if not self.stacks:
            raise RuntimeError("No stacks in this collection.")

        # Default: all sensors from first stack
        if sensor_names is None:
            sensor_names = list(self.stacks[0].sensor_fields.keys())
        else:
            sensor_names = list(sensor_names)

        for s_name in sensor_names:
            if samples is not None:
                s = samples[samples["sensor_name"] == s_name] if "sensor_name" in samples.columns else samples
            else:
                s = self.stratified_sample(
                    sensor_name=s_name,
                    n_strata=n_strata,
                    samples_per_stratum=samples_per_stratum,
                    sampling_seed=sampling_seed,
                    include_truth=True,
                )

            if s.empty:
                raise RuntimeError(f"No samples obtained for sensor {s_name!r}.")

            x = s["sensor_value"].to_numpy(dtype=float)

            y_prob_pct = s["interpreter_prob"].to_numpy(dtype=float)

            #Update for this interpreter
            if interpreter_agent is not None:
                y_prob_pct = interpreter_agent.transform_probabilities(y_prob_pct)

            
            y = (y_prob_pct >= 50.0).astype(int)

            # Drop NaNs / infinities
            mask = np.isfinite(x) & np.isfinite(y_prob_pct)
            x = x[mask]
            y = y[mask]

            if x.size < 5:
                raise RuntimeError(
                    f"Not enough valid samples to fit binary classifier for sensor {s_name!r}."
                )

            # Sort by sensor value
            order = np.argsort(x)
            x_sorted = x[order]
            y_sorted = y[order]

            n = x_sorted.size
            n_pos = int(y_sorted.sum())
            n_neg = int(n - n_pos)

            if n_pos == 0 or n_neg == 0:
                # Degenerate case: all labels same
                # Just store a trivial classifier at mid-range
                t_trivial = float(0.5 * (x_sorted.min() + x_sorted.max()))
                self.binary_models[s_name] = {
                    "sensor_name": s_name,
                    "threshold": t_trivial,
                    "fp_to_fn_ratio": fp_to_fn_ratio,
                    "curve": {
                        "thresholds": [t_trivial],
                        "fpr": [0.0],
                        "fnr": [0.0],
                    },
                    "n_samples": int(n),
                    "experiment_config": self.experiment_config,
                }
                continue

            # Unique candidate thresholds: we use each unique sensor value
            uniq_vals, first_indices = np.unique(x_sorted, return_index=True)
            # For classification rule: predict 1 if x >= t,
            # we will consider thresholds at these uniq_vals.

            # Precompute suffix sums of positives to get TP quickly
            # Suffix TP at index i = sum(y_sorted[i:])
            tp_suffix = np.cumsum(y_sorted[::-1])[::-1]
            # Suffix length
            idx_all = np.arange(n)
            suffix_len = n - first_indices  # not used directly; we compute per threshold

            thresholds = []
            fpr_list = []
            fnr_list = []

            for t_val, idx_start in zip(uniq_vals, first_indices):
                # All samples with x >= t_val are predicted positive
                # TP = positives in suffix from idx_start
                tp = int(tp_suffix[idx_start])
                # Number of predicted positives
                n_pred_pos = n - idx_start
                fp = n_pred_pos - tp

                fn = n_pos - tp
                tn = n_neg - fp

                # Rates (guard against zero denom)
                fpr = fp / n_neg if n_neg > 0 else np.nan  # P(pred=1 | y=0)
                fnr = fn / n_pos if n_pos > 0 else np.nan  # P(pred=0 | y=1)

                thresholds.append(float(t_val))
                fpr_list.append(float(fpr))
                fnr_list.append(float(fnr))

            thresholds = np.array(thresholds)
            fpr_arr = np.array(fpr_list)
            fnr_arr = np.array(fnr_list)

            # Cost function: weighted combination of error rates
            # fp_to_fn_ratio = w_fp / w_fn  -> cost = w_fp*FPR + w_fn*FNR
            w_fp = fp_to_fn_ratio
            w_fn = 1.0
            cost = w_fp * fpr_arr + w_fn * fnr_arr

            # Ignore NaN cost points
            valid_cost_mask = np.isfinite(cost)
            if not np.any(valid_cost_mask):
                raise RuntimeError(f"All cost values NaN for sensor {s_name!r}.")

            idx_best = np.argmin(cost[valid_cost_mask])
            # Map back to full arrays
            best_indices = np.where(valid_cost_mask)[0]
            best_idx = best_indices[idx_best]

            best_t = thresholds[best_idx]
            best_fpr = fpr_arr[best_idx]
            best_fnr = fnr_arr[best_idx]
            best_cost = cost[best_idx]

            # Store model info
            model_info = {
                "sensor_name": s_name,
                "threshold": float(best_t),
                "fp_to_fn_ratio": float(fp_to_fn_ratio),
                "n_samples": int(n),
                "best": {
                    "fpr": float(best_fpr),
                    "fnr": float(best_fnr),
                    "cost": float(best_cost),
                },
                "curve": {
                    "thresholds": thresholds.tolist(),
                    "fpr": fpr_arr.tolist(),
                    "fnr": fnr_arr.tolist(),
                    "cost": cost.tolist(),
                },
                "experiment_config": self.experiment_config,
            }

            self.binary_models[s_name] = model_info

        return self.binary_models


    def plotBinaryCurves(
        self,
        sensor_name: str,
        show_cost: bool = True,
        save_dir: Optional[str] = None,
    ) -> "plt.Figure":
        """
        Plot FPR and FNR (and optionally cost) as a function of threshold
        for a previously built binary classifier.

        Returns the matplotlib Figure.
        """
        if sensor_name not in self.binary_models:
            raise KeyError(f"No binary model found for sensor {sensor_name!r}. Call buildBinaryClassifier() first.")

        model = self.binary_models[sensor_name]
        curve = model["curve"]

        thresholds = np.array(curve["thresholds"])
        fpr = np.array(curve["fpr"])
        fnr = np.array(curve["fnr"])
        cost = np.array(curve["cost"])

        fig = plt.figure(figsize=(7, 5))
        plt.plot(thresholds, fpr, label="FPR", linewidth=2)
        plt.plot(thresholds, fnr, label="FNR", linewidth=2)

        if show_cost:
            plt.plot(thresholds, cost, label="Cost", linestyle="--", alpha=0.7)

        best_t = model["threshold"]
        plt.axvline(best_t, color="k", linestyle=":", label=f"Best t = {best_t:.2f}")

        plt.xlabel("Threshold (sensor value)")
        plt.ylabel("Rate / Cost")
        plt.title(f"Binary classifier tradeoff – {sensor_name}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_dir is not None:
            fname = f"binary_threshold_curves_{sensor_name.replace(' ', '_')}.png"
            fig.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches="tight")
        plt.show()
        return fig

    def loadBinaryModel(self, model_data: Dict[str, Any]) -> None:
        """
        Load threshold-based binary classifier models into this collection.

        model_data can be either:
          - a single model dict with keys 'sensor_name' and 'threshold', OR
          - a dict mapping sensor_name -> model_dict.

        Loaded models are stored in self.binary_models.
        """
        if "sensor_name" in model_data and "threshold" in model_data:
            # Single binary model
            s_name = model_data["sensor_name"]
            self.binary_models[s_name] = model_data
        else:
            # Assume mapping sensor_name -> model_dict
            for s_name, model in model_data.items():
                if not isinstance(model, dict):
                    continue
                if "threshold" not in model:
                    raise ValueError(
                        f"Binary model for sensor {s_name!r} does not contain a 'threshold' key."
                    )
                self.binary_models[s_name] = model

    

    def applyBinaryClassifier(
        self,
        sensor_names: Optional[List[str]] = None,
    ) -> None:
        """
        Apply the threshold-based binary classifiers to all stacks.

        For each stack and each sensor with a learned threshold:
          - Predict loss (1) where sensor_field >= threshold, else 0.
          - Store as stack.binary_class_by_sensor[sensor_name] (0/1 array).
        """
        if not self.binary_models:
            raise RuntimeError("No binary_models available. Call buildBinaryClassifier() or load them first.")

        # Default: all sensors for which we have binary models
        if sensor_names is None:
            sensor_names = list(self.binary_models.keys())
        else:
            sensor_names = list(sensor_names)

        for stack in self.stacks:
            # Create container on each stack if not present
            if not hasattr(stack, "binary_class_by_sensor"):
                stack.binary_class_by_sensor = {}

            for s_name in sensor_names:
                model = self.binary_models.get(s_name, None)
                if model is None:
                    continue
                if s_name not in stack.sensor_fields:
                    continue

                threshold = model["threshold"]
                sensor = stack.sensor_fields[s_name]
                x = np.asarray(sensor.sensor_field, dtype=float)

                binary_map = (x >= threshold).astype(int)
                stack.binary_class_by_sensor[s_name] = binary_map




    def binary_stratified_sample(
        self,
        sensor_name: str,
        n_samples_per_class: Optional[Dict[int, int]] = None,
        sampling_seed: Optional[int] = None,
        include_truth: bool = True,
        interpreter_agent: Optional["InterpreterAgent"] = None,
        stack_indices: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Draw a stratified sample from the binary classifier output for a
        given sensor, sampling from map classes 0 and 1.
    
        Parameters
        ----------
        sensor_name : str
            Which sensor's binary classifier to use.
        n_samples_per_class : dict or None
            Mapping from map class -> number of samples, e.g. {0: 100, 1: 200}.
            If None, defaults to {0: 100, 1: 100}.
        sampling_seed : int or None
            Random seed for reproducible sampling. If None, uses timestamp.
        include_truth : bool
            If True, also attach 'disturbed' and 'final_cover' from the truth landscape.
        interpreter_agent : InterpreterAgent or None
            If provided, applies interpreter variability to each sampled interpreter_prob.
        stack_indices : list[int] or None
            If provided, restrict sampling to these stacks (indices into self.stacks).
    
        Returns
        -------
        samples : pd.DataFrame
            Columns include:
              - stack_id
              - stack_index
              - sensor_name
              - row, col
              - map_class      (0/1, classifier output)
              - interpreter_prob
              - ref_class      (0/1, interpreter >= 50%?)
              - (optional) disturbed, final_cover
        """
        import time
        import numpy as np
        import pandas as pd
        from typing import Any, Dict, List, Optional
    
        if not self.stacks:
            raise RuntimeError("This collection has no stacks.")
    
        if not hasattr(self.stacks[0], "binary_class_by_sensor"):
            raise RuntimeError(
                "No binary classifier outputs found. "
                "Call applyBinaryClassifier() on this collection first."
            )
    
        if n_samples_per_class is None:
            n_samples_per_class = {0: 100, 1: 100}
    
        if sampling_seed is None:
            sampling_seed = int(time.time())
    
        rng = np.random.default_rng(sampling_seed)
    
        # -----------------------------
        # Choose which stacks to sample from
        # -----------------------------
        if stack_indices is None:
            idx_used = list(range(len(self.stacks)))
        else:
            idx_used = [int(i) for i in stack_indices]
            n = len(self.stacks)
            bad = [i for i in idx_used if i < 0 or i >= n]
            if bad:
                raise IndexError(f"stack_indices out of range: {bad} (n_stacks={n})")
    
        # Collect candidates for each class across chosen stacks
        # Each entry: (stack_index, row, col) where stack_index is index into self.stacks
        candidates_by_class: Dict[int, List[tuple[int, int, int]]] = {0: [], 1: []}
    
        for si in idx_used:
            stack = self.stacks[si]
    
            if sensor_name not in stack.binary_class_by_sensor:
                raise KeyError(
                    f"Sensor {sensor_name!r} not found in stack.binary_class_by_sensor "
                    f"for stack {stack.stack_id}"
                )
    
            bin_map = np.asarray(stack.binary_class_by_sensor[sensor_name], dtype=int)
    
            # (Optional) sanity: ensure only 0/1
            vals = set(np.unique(bin_map).tolist())
            if not vals.issubset({0, 1}):
                raise ValueError(
                    f"binary_class_by_sensor[{sensor_name!r}] contains values other than 0/1 "
                    f"for stack {stack.stack_id}: {sorted(vals)}"
                )
    
            nrows, ncols = bin_map.shape
            rr, cc = np.meshgrid(np.arange(nrows), np.arange(ncols), indexing="ij")
    
            for cls in (0, 1):
                mask = (bin_map == cls)
                idx_r = rr[mask]
                idx_c = cc[mask]
                for r, c in zip(idx_r, idx_c):
                    candidates_by_class[cls].append((si, int(r), int(c)))
    
        # -----------------------------
        # Sample requested numbers from each class
        # -----------------------------
        rows: List[Dict[str, Any]] = []
    
        for cls in (0, 1):
            req_n = int(n_samples_per_class.get(cls, 0))
            pool = candidates_by_class[cls]
    
            if req_n <= 0 or len(pool) == 0:
                continue
    
            if len(pool) <= req_n:
                chosen = pool  # take all
            else:
                picked = rng.choice(len(pool), size=req_n, replace=False)
                chosen = [pool[i] for i in picked]
    
            for si, r, c in chosen:
                stack = self.stacks[si]
                base = stack.base_landscape
                interp = stack.interpreter_field
    
                # Base interpreter probability at this pixel (0–100)
                interp_prob_base = float(np.asarray(interp.interpreter_field, dtype=float)[r, c])
    
                # Apply interpreter variability if agent provided
                if interpreter_agent is not None:
                    interp_prob = float(interpreter_agent.transform_probabilities([interp_prob_base])[0])
                else:
                    interp_prob = interp_prob_base
    
                ref_class = int(interp_prob >= 50.0)
    
                row_dict: Dict[str, Any] = {
                    "stack_id": stack.stack_id,
                    "stack_index": si,               # NOTE: original index in self.stacks
                    "sensor_name": sensor_name,
                    "row": r,
                    "col": c,
                    "map_class": int(cls),
                    "interpreter_prob": float(interp_prob),
                    "ref_class": int(ref_class),
                }
    
                if include_truth:
                    if getattr(base, "disturbed", None) is not None:
                        row_dict["disturbed"] = bool(base.disturbed[r, c])
                    if getattr(base, "final_cover", None) is not None:
                        row_dict["final_cover"] = float(base.final_cover[r, c])
    
                rows.append(row_dict)
    
        return pd.DataFrame(rows)

    def binary_confusion_from_samples(self, samples: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute a 2x2 confusion matrix from sampled points, where:
          - rows = map_class (i = 0,1)
          - cols = ref_class (j = 0,1)
    
        n_ij[i, j] = count(map=i, ref=j)
        """
        if "map_class" not in samples or "ref_class" not in samples:
            raise ValueError("samples must contain 'map_class' and 'ref_class' columns.")
    
        for col in ["map_class", "ref_class"]:
            vals = set(samples[col].unique())
            if not vals.issubset({0, 1}):
                raise ValueError(f"Column {col!r} contains values other than 0/1: {vals}")
    
        n_00 = int(((samples["map_class"] == 0) & (samples["ref_class"] == 0)).sum())
        n_01 = int(((samples["map_class"] == 0) & (samples["ref_class"] == 1)).sum())
        n_10 = int(((samples["map_class"] == 1) & (samples["ref_class"] == 0)).sum())
        n_11 = int(((samples["map_class"] == 1) & (samples["ref_class"] == 1)).sum())
    
        # rows = map_class (0,1); cols = ref_class (0,1)
        #          ref 0   ref 1
        # map 0     n00     n01
        # map 1     n10     n11
        n_ij = np.array([[n_00, n_01],
                         [n_10, n_11]], dtype=int)
    
        n_i = n_ij.sum(axis=1)   # totals per map class (row totals)
        n_total = int(n_ij.sum())
    
        return {"n_ij": n_ij, "n_i": n_i, "n_total": n_total}

    def olofsson_area_estimates(
        self,
        samples: pd.DataFrame,
        sensor_name: str,
        disturbed_class: int = 1,
        alpha: float = 0.05,
        stack_indices: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Apply Olofsson's error-adjusted area estimation to binary classifier outputs,
        optionally restricted to a subset of stacks (by index).
    
        Parameters
        ----------
        samples : DataFrame
            Output from binary_stratified_sample, with map_class, ref_class.
        sensor_name : str
            Which sensor's binary map to use when computing mapped area proportions W_i.
        disturbed_class : int
            Which reference class is 'disturbed' (1 by default).
        alpha : float
            Significance level for confidence intervals (default 0.05 → 95% CI).
        stack_indices : list[int] or None
            If provided, restrict all map-area and truth-area calculations to only
            these stacks (by index into self.stacks). If None, uses all stacks.
    
        Returns
        -------
        results : dict
            {
              "W": [W_0, W_1],
              "p_ij": 2x2 array,
              "P_hat": [P_hat_0, P_hat_1],
              "SE": [SE_0, SE_1],
              "CI": {"lower": [l0, l1], "upper": [u0, u1]},
              "true_disturbed_prop": float,
              "disturbed_in_CI": bool,
              "stack_indices_used": list[int],
            }
        """
        # -------------------------
        # Choose which stacks apply
        # -------------------------
        if stack_indices is None:
            stacks_used = list(self.stacks)
            idx_used = list(range(len(self.stacks)))
        else:
            idx_used = [int(i) for i in stack_indices]
            # validate indices
            n = len(self.stacks)
            bad = [i for i in idx_used if i < 0 or i >= n]
            if bad:
                raise IndexError(f"stack_indices contains out-of-range indices: {bad} (n_stacks={n})")
            stacks_used = [self.stacks[i] for i in idx_used]
    
        # ------------------------------------
        # 1) Confusion matrix from *samples*
        # ------------------------------------
        cm = self.binary_confusion_from_samples(samples)
        n_ij = cm["n_ij"]   # shape (2,2) with rows = reference, cols = predicted (per your printout)
        n_i = cm["n_i"]     # length 2; SHOULD correspond to "map class i" counts in the sample design
    
        # IMPORTANT:
        # Your downstream math assumes p_ij = P(ref=j | map=i), i.e. rows = map class.
        # So binary_confusion_from_samples MUST be producing n_ij as [map_i, ref_j] OR
        # you must transpose here. In your earlier fix, the direction was corrected.
        #
        # If your binary_confusion_from_samples returns n_ij with rows=map and cols=ref,
        # keep as-is. If it returns rows=ref and cols=map, transpose it.
        #
        # We'll detect using column names if present; otherwise assume your current
        # implementation already matches p_ij = P(ref | map).
        #
        # If you are 100% sure it's already [map, ref], you can delete the next block.
    
        if "matrix_orientation" in cm:
            # optional: if you stored this metadata in cm earlier
            orientation = cm["matrix_orientation"]
            if orientation == "ref_vs_map":  # rows=ref, cols=map
                n_ij_map_ref = n_ij.T
            else:  # "map_vs_ref"
                n_ij_map_ref = n_ij
        else:
            # Heuristic: if samples has map_class/ref_class, we can build quick check.
            # We'll assume your current cm already outputs map_vs_ref (since you fixed earlier).
            n_ij_map_ref = n_ij
    
        # map sample totals by map class
        n_i = np.asarray(n_i, dtype=float)
    
        # --------------------------------------------
        # 2) Conditional probabilities p_ij = P(ref=j | map=i)
        # --------------------------------------------
        p_ij = np.zeros((2, 2), dtype=float)
        for i in (0, 1):
            if n_i[i] > 0:
                p_ij[i, :] = n_ij_map_ref[i, :] / n_i[i]
            else:
                p_ij[i, :] = np.nan
    
        # ----------------------------------------------------
        # 3) Map area proportions W_i from full maps (restricted to stacks_used)
        # ----------------------------------------------------
        N_i = np.zeros(2, dtype=float)  # pixel counts by map class across chosen stacks
        for stack in stacks_used:
            if not hasattr(stack, "binary_class_by_sensor"):
                raise RuntimeError(
                    "Stacks lack binary_class_by_sensor. Call applyBinaryClassifier() before olofsson_area_estimates()."
                )
            if sensor_name not in stack.binary_class_by_sensor:
                raise KeyError(
                    f"Sensor {sensor_name!r} not found in stack.binary_class_by_sensor for stack {stack.stack_id}"
                )
    
            bin_map = np.asarray(stack.binary_class_by_sensor[sensor_name], dtype=int)
            N_i[0] += np.sum(bin_map == 0)
            N_i[1] += np.sum(bin_map == 1)
    
        N_total = float(N_i.sum())
        if N_total == 0:
            raise RuntimeError("No pixels found in binary maps across selected stacks.")
    
        W = N_i / N_total  # proportions of map area
    
        # -----------------------------------------
        # 4) Olofsson area proportions P_hat_j = sum_i W_i * p_ij[i,j]
        # -----------------------------------------
        P_hat = np.zeros(2, dtype=float)
        for j in (0, 1):
            P_hat[j] = np.nansum(W * p_ij[:, j])
    
        # -------------------------------------------------
        # 5) Standard error SE(P_hat_j)
        # SE^2 = sum_i W_i^2 * [ p_ij(1 - p_ij) / (n_i - 1) ]
        # -------------------------------------------------
        SE = np.zeros(2, dtype=float)
        for j in (0, 1):
            var_j = 0.0
            for i in (0, 1):
                if n_i[i] > 1 and np.isfinite(p_ij[i, j]):
                    var_j += (W[i] ** 2) * (p_ij[i, j] * (1.0 - p_ij[i, j]) / (n_i[i] - 1.0))
            SE[j] = np.sqrt(var_j)
    
        # -------------------------
        # 6) Confidence intervals
        # -------------------------
        from scipy.stats import norm
        z = norm.ppf(1.0 - alpha / 2.0)
    
        lower = np.clip(P_hat - z * SE, 0.0, 1.0)
        upper = np.clip(P_hat + z * SE, 0.0, 1.0)
    
        # ----------------------------------------------------
        # 7) True disturbed proportion from underlying truth (restricted to stacks_used)
        # ----------------------------------------------------
        total_disturbed = 0.0
        total_pixels = 0.0
        for stack in stacks_used:
            base = stack.base_landscape
            disturbed = np.asarray(base.disturbed, dtype=bool)
            total_disturbed += float(disturbed.sum())
            total_pixels += float(disturbed.size)
    
        true_disturbed_prop = float(total_disturbed / total_pixels) if total_pixels > 0 else np.nan
    
        j = int(disturbed_class)
        disturbed_in_CI = bool((true_disturbed_prop >= lower[j]) and (true_disturbed_prop <= upper[j]))
    
        return {
            "W": W,
            "p_ij": p_ij,
            "P_hat": P_hat,
            "SE": SE,
            "CI": {"lower": lower, "upper": upper},
            "true_disturbed_prop": true_disturbed_prop,
            "disturbed_in_CI": disturbed_in_CI,
            "stack_indices_used": idx_used,
            "N_i": N_i,
            "N_total": N_total,
            "n_ij_samples": n_ij.astype(int),
            "n_i_samples": n_i.astype(int),
        }

   
   
    def debug_plot_truth_vs_binary(
        self,
        stack_index: int,
        sensor_name: str,
        samples: "pd.DataFrame",
    ) -> None:
        """
        Visualize, for a single LandscapeStack in this collection:
          1) The true disturbance mask from DisturbanceLandscape.
          2) The binary classifier output for a chosen sensor, recoded into:
                - TN (truth=0, map=0) -> white
                - FP (truth=0, map=1) -> red
                - FN (truth=1, map=0) -> blue
                - TP (truth=1, map=1) -> black
          3) Sample points used for validation, with symbology:
                - tiny black dots where map_class == ref_class (agreement)
                - red dots where map=1, ref=0 (FP vs interpreter)
                - blue dots where map=0, ref=1 (FN vs interpreter)
        """
        # ------------- Grab the stack and core fields -------------
        if not (0 <= stack_index < len(self.stacks)):
            raise IndexError(f"stack_index {stack_index} is out of range for {len(self.stacks)} stacks.")

        stack = self.stacks[stack_index]
        base = stack.base_landscape

        # True disturbance landscape (simulation truth)
        truth = np.asarray(base.disturbed, dtype=bool)

        # Binary classifier mask for this sensor
        if not hasattr(stack, "binary_class_by_sensor"):
            raise RuntimeError(
                "Stack has no 'binary_class_by_sensor'. Did you call applyBinaryClassifier()?"
            )

        if sensor_name not in stack.binary_class_by_sensor:
            raise KeyError(
                f"Sensor {sensor_name!r} not found in stack.binary_class_by_sensor "
                f"for stack {stack.stack_id}"
            )

        bin_map = np.asarray(stack.binary_class_by_sensor[sensor_name], dtype=int)

        if truth.shape != bin_map.shape:
            raise ValueError(
                f"Shape mismatch: truth {truth.shape} vs bin_map {bin_map.shape} "
                f"for stack {stack.stack_id}"
            )

        # ------------- Subset samples to this stack -------------
        sub = samples[samples["stack_index"] == stack_index].copy()

        if sub.empty:
            print(samples["stack_index"])
            print(stack_index)
            print(f"No samples found for stack_index={stack_index} in provided samples.")

        r = sub["row"].to_numpy(dtype=int)
        c = sub["col"].to_numpy(dtype=int)
        map_class = sub["map_class"].to_numpy(dtype=int)
        ref_class = sub["ref_class"].to_numpy(dtype=int)

        # ------------- Diagnostics on proportions -------------
        true_prop = float(truth.mean())
        map_prop = float((bin_map == 1).mean())

        print(f"\n=== Debug plot for stack_index={stack_index}, id={stack.stack_id} ===")
        print(f"  True disturbed proportion (DisturbanceLandscape): {true_prop:0.6f}")
        print(f"  Mapped disturbed proportion (binary classifier): {map_prop:0.6f}")
        print(f"  Difference (map - truth):                       {map_prop - true_prop:0.6f}")

        # ------------- Build 4-class map for FP/FN/TP/TN (wrt truth) -------------
        # Codes:
        #   0 = TN (truth=0, map=0)
        #   1 = FP (truth=0, map=1)
        #   2 = FN (truth=1, map=0)
        #   3 = TP (truth=1, map=1)
        cls_map = np.zeros_like(bin_map, dtype=int)  # default TN
        cls_map[(truth == False) & (bin_map == 1)] = 1  # FP
        cls_map[(truth == True) & (bin_map == 0)] = 2   # FN
        cls_map[(truth == True) & (bin_map == 1)] = 3   # TP

        # Colormap: TN white, FP red, FN blue, TP black
        cls_cmap = ListedColormap(["white", "red", "blue", "gray"])

        # ------------- Plotting -------------

        def _sync_axes_to_raster(ax, raster_shape):
            nrows, ncols = raster_shape
            ax.set_xlim(-0.5, ncols - 0.5)
            ax.set_ylim(nrows - 0.5, -0.5)  # origin="upper"
            ax.set_aspect("equal")          # no squish

        # nrows, ncols = cls_map.shape
        # ratio = ncols / nrows
        # fig, axes = plt.subplots(1, 3, figsize=(15 * ratio, 6))

        
        fig, axes = plt.subplots(1, 3, figsize=(15, 10))

        # Panel 1: True disturbance landscape (binary legend)
        ax0 = axes[0]
        truth_int = truth.astype(int)
        truth_cmap = ListedColormap(["white", "gray"])  # 0=no disturbance, 1=disturbance

        

        ax0.imshow(truth_int, cmap=truth_cmap, origin="upper")
        _sync_axes_to_raster(ax0, truth_int.shape)

        
        ax0.set_title("True disturbance (DisturbanceLandscape)\n0 = no, 1 = yes")
        
        
        # Disable ticks but keep axis frame
        ax=ax0
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Make sure the frame is visible
        for side in ["left", "right", "top", "bottom"]:
            ax.spines[side].set_visible(True)
            ax.spines[side].set_edgecolor("black")
            ax.spines[side].set_linewidth(1.2)

            
        # Legend for 0/1
        legend_truth = [
            Line2D([0], [0], marker="s", color="none",
                   label="0: no disturbance",
                   markerfacecolor="white", markeredgecolor="black", markersize=8),
            Line2D([0], [0], marker="s", color="none",
                   label="1: disturbed",
                   markerfacecolor="gray", markeredgecolor="black", markersize=8),
        ]
        ax0.legend(handles=legend_truth, loc="upper right", fontsize=8)
        
        
        # Panel 2: Binary classifier output recoded as TN/FP/FN/TP wrt truth
        ax1 = axes[1]

        im1 = ax1.imshow(cls_map, cmap=cls_cmap, origin="upper")
        _sync_axes_to_raster(ax1, cls_map.shape)
        
        ax1.set_title(
            f"Binary map vs truth ({sensor_name})\n"
            "white=TN, red=FP, blue=FN, gray=TP"
        )
       # Disable ticks but keep axis frame
        ax=ax1
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Make sure the frame is visible
        for side in ["left", "right", "top", "bottom"]:
            ax.spines[side].set_visible(True)
            ax.spines[side].set_edgecolor("black")
            ax.spines[side].set_linewidth(1.2)

        legend_cls = [
            Line2D([0], [0], marker="s", color="none",
                   label="TN (truth=0, map=0)",
                   markerfacecolor="white", markeredgecolor="black", markersize=8),
            Line2D([0], [0], marker="s", color="none",
                   label="FP (truth=0, map=1)",
                   markerfacecolor="red", markeredgecolor="black", markersize=8),
            Line2D([0], [0], marker="s", color="none",
                   label="FN (truth=1, map=0)",
                   markerfacecolor="blue", markeredgecolor="black", markersize=8),
            Line2D([0], [0], marker="s", color="none",
                   label="TP (truth=1, map=1)",
                   markerfacecolor="gray", markeredgecolor="black", markersize=8),
        ]
        ax1.legend(handles=legend_cls, loc="upper right", fontsize=8)
      

        
        # Panel 3: Sample points on the same 4-class background
        ax2 = axes[2]
       

        ax2.imshow(cls_map, cmap=cls_cmap, origin="upper")
        _sync_axes_to_raster(ax2, cls_map.shape)
        ax2.set_title(
            "Sample points on map\n"
            "background: vs truth; points: vs interpreter"
        )
        # Disable ticks but keep axis frame
        ax=ax2
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Make sure the frame is visible
        for side in ["left", "right", "top", "bottom"]:
            ax.spines[side].set_visible(True)
            ax.spines[side].set_edgecolor("black")
            ax.spines[side].set_linewidth(1.2)

        if not sub.empty:
            # Disagreement vs interpreter (ref_class):
            # FP vs interpreter: map=1, ref=0
            # FN vs interpreter: map=0, ref=1
            fp_int = (map_class == 1) & (ref_class == 0)
            fn_int = (map_class == 0) & (ref_class == 1)
            agree = (map_class == ref_class)

            # Tiny black dots for agreements
            if np.any(agree):
                ax2.scatter(
                    c[agree], r[agree],
                    s=5,
                    c="black",
                    marker="o",
                    linewidths=0,
                    zorder=3,
                )

            # Red dots for FP vs interpreter
            if np.any(fp_int):
                ax2.scatter(
                    c[fp_int], r[fp_int],
                    s=30,
                    c="red",
                    marker="o",
                    edgecolors="black",
                    linewidths=0.5,
                    zorder=4,
                    label="FP vs interpreter (map=1, ref=0)",
                )

            # Blue dots for FN vs interpreter
            if np.any(fn_int):
                ax2.scatter(
                    c[fn_int], r[fn_int],
                    s=30,
                    c="blue",
                    marker="o",
                    edgecolors="black",
                    linewidths=0.5,
                    zorder=4,
                    label="FN vs interpreter (map=0, ref=1)",
                )

            # Legend for sample points (only disagreements need labels)
            legend_pts = []
            if np.any(fp_int):
                legend_pts.append(
                    Line2D(
                        [0], [0],
                        marker="o", color="none",
                        label="Sample FP (map=1, ref=0)",
                        markerfacecolor="red",
                        markeredgecolor="black",
                        markersize=6,
                    )
                )
            if np.any(fn_int):
                legend_pts.append(
                    Line2D(
                        [0], [0],
                        marker="o", color="none",
                        label="Sample FN (map=0, ref=1)",
                        markerfacecolor="blue",
                        markeredgecolor="black",
                        markersize=6,
                    )
                )
            if legend_pts:
                ax2.legend(handles=legend_pts, loc="upper right", fontsize=7)
        


        
        plt.tight_layout()
        plt.show()

        
    def binary_metrics_for_collection(
        self,
        sensor_name: str,
        stack_indices: Optional[List[int]] = None,
        # sampling
        n_samples_map0: int = 200,
        n_samples_map1: int = 200,
        sampling_seed: Optional[int] = None,
        # Olofsson
        disturbed_class: int = 1,
        alpha: float = 0.05,
        # output controls
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Compute binary classifier validation metrics for this collection, optionally
        restricted to stack_indices.
    
        Uses:
          - binary_stratified_sample(...)  -> samples with map_class, ref_class
          - binary_confusion_from_samples(samples)  (rows=map, cols=ref)
          - olofsson_area_estimates(samples, ..., stack_indices=stack_indices)
    
        Returns a dict with:
          - 'binary_summary': mean/-2SE/+2SE/CI for class 1 and truth
          - 'confusion': n_ij (map x ref), plus TN/FP/FN/TP in *Ref vs Pred* language
          - 'olofsson': the full olofsson result payload
        """
    
        # ------------------
        # 0) choose stacks subset (for sampling)
        # ------------------
        if stack_indices is None:
            idx_used = list(range(len(self.stacks)))
        else:
            idx_used = [int(i) for i in stack_indices]
            n = len(self.stacks)
            bad = [i for i in idx_used if i < 0 or i >= n]
            if bad:
                raise IndexError(f"stack_indices out of range: {bad} (n_stacks={n})")
    
        # ------------------
        # 1) draw binary stratified sample
        #    (assumes your existing method supports stack_indices)
        # ------------------
        # Expected output columns: map_class, ref_class (0/1)
        samples = self.binary_stratified_sample(
            sensor_name=sensor_name,
            n_samples_map0=n_samples_map0,
            n_samples_map1=n_samples_map1,
            sampling_seed=sampling_seed,
            stack_indices=idx_used,   # <- IMPORTANT: sample only within the requested subset
        )
        if samples is None or len(samples) == 0:
            raise RuntimeError("binary_stratified_sample produced no samples.")
    
        # Ensure integer 0/1
        samples = samples.copy()
        samples["map_class"] = samples["map_class"].astype(int)
        samples["ref_class"] = samples["ref_class"].astype(int)


        # Expected output columns: map_class, ref_class (0/1)
        samples = self.binary_stratified_sample(
            sensor_name=sensor_name,
            n_samples_per_class={0: int(n_samples_map0), 1: int(n_samples_map1)},
            sampling_seed=sampling_seed,
            stack_indices=idx_used,   # sample only within requested subset
            include_truth=True,
            interpreter_agent=interpreter_agent,  # if you have one in scope
        )
        
        if samples is None or len(samples) == 0:
            raise RuntimeError("binary_stratified_sample produced no samples.")
        
        samples = samples.copy()
        samples["map_class"] = samples["map_class"].astype(int)
        samples["ref_class"] = samples["ref_class"].astype(int)
                
        # ------------------
        # 2) confusion matrix from samples (map x ref)
        # ------------------
        cm = self.binary_confusion_from_samples(samples)
        n_ij = np.asarray(cm["n_ij"], dtype=int)   # rows=map, cols=ref
        n_i = np.asarray(cm["n_i"], dtype=int)
    
        # Convert to the "intuitive" TN/FP/FN/TP terms (Ref vs Pred)
        # With our n_ij definition:
        #   n_ij[map, ref]
        # TN = ref0 pred0 = map0 ref0 = n_ij[0,0]
        # FN = ref1 pred0 = map0 ref1 = n_ij[0,1]
        # FP = ref0 pred1 = map1 ref0 = n_ij[1,0]
        # TP = ref1 pred1 = map1 ref1 = n_ij[1,1]
        TN = int(n_ij[0, 0])
        FN = int(n_ij[0, 1])
        FP = int(n_ij[1, 0])
        TP = int(n_ij[1, 1])
        n_total = TN + FP + FN + TP
    
        # Derived accuracy metrics (optional, but handy)
        overall_acc = (TP + TN) / n_total if n_total else np.nan
        tpr = TP / (TP + FN) if (TP + FN) else np.nan  # sensitivity / recall
        tnr = TN / (TN + FP) if (TN + FP) else np.nan  # specificity
        ppv = TP / (TP + FP) if (TP + FP) else np.nan  # precision
        npv = TN / (TN + FN) if (TN + FN) else np.nan  # negative predictive value
    
        if verbose:
            print("\n================ CONFUSION MATRIX (Map vs Reference) ================")
            print("              Reference")
            print("            |   0     |   1")
            print("  ----------+---------+--------")
            print(f"Map 0       | {n_ij[0,0]:5d}  | {n_ij[0,1]:5d}")
            print(f"Map 1       | {n_ij[1,0]:5d}  | {n_ij[1,1]:5d}")
            print("\nInterpretation (Ref vs Pred terms):")
            print(f"  TN (ref0,pred0): {TN}")
            print(f"  FP (ref0,pred1): {FP}")
            print(f"  FN (ref1,pred0): {FN}")
            print(f"  TP (ref1,pred1): {TP}")
            print("\nDerived metrics:")
            print(f"  Overall Accuracy:     {overall_acc:.3f}")
            print(f"  Sensitivity (TPR):    {tpr:.3f}")
            print(f"  Specificity (TNR):    {tnr:.3f}")
            print(f"  Precision (PPV):      {ppv:.3f}")
            print(f"  Neg. Predictive Val.: {npv:.3f}")
            print("=====================================================================\n")
    
        # ------------------
        # 3) Olofsson area estimates (restricted to same stacks)
        # ------------------
        olo = self.olofsson_area_estimates(
            samples=samples,
            sensor_name=sensor_name,
            disturbed_class=disturbed_class,
            alpha=alpha,
            stack_indices=idx_used,
        )
    
        P_hat = np.asarray(olo["P_hat"], dtype=float)   # ref class proportions
        SE = np.asarray(olo["SE"], dtype=float)
        CI_lower = np.asarray(olo["CI"]["lower"], dtype=float)
        CI_upper = np.asarray(olo["CI"]["upper"], dtype=float)
        true_p1 = float(olo["true_disturbed_prop"])
    
        j = int(disturbed_class)  # class 1 by default
        mean = float(P_hat[j])
        se = float(SE[j])
    
        # “-2SE, mean, +2SE” (as you requested for binary plot)
        minus_2se = float(np.clip(mean - 2.0 * se, 0.0, 1.0))
        plus_2se = float(np.clip(mean + 2.0 * se, 0.0, 1.0))
    
        if verbose:
            print("=============== OLOFSSON AREA-ADJUSTED PROPORTIONS =================")
            print(f"Stacks used: {idx_used}")
            print(f"Map area proportions W (map class 0,1): {olo['W'][0]:.6f}, {olo['W'][1]:.6f}")
            print(f"Adjusted area proportions P_hat (ref class 0,1): {P_hat[0]:.6f}, {P_hat[1]:.6f}")
            print(f"SE(P_hat) (class 0,1): {SE[0]:.6f}, {SE[1]:.6f}")
            print("95% CI (class 0,1):")
            print(f"  class 0: [{CI_lower[0]:.6f}, {CI_upper[0]:.6f}]")
            print(f"  class 1: [{CI_lower[1]:.6f}, {CI_upper[1]:.6f}]")
            print(f"True disturbed proportion: {true_p1:.6f}")
            print(f"Is true disturbed within CI for class 1? {olo['disturbed_in_CI']}")
            print("====================================================================\n")
    
        # ------------------
        # 4) package results
        # ------------------
        out: Dict[str, Any] = {
            "sensor_name": sensor_name,
            "stack_indices_used": idx_used,
            "samples": samples,  # keep so you can debug/plot later
            "confusion": {
                "n_ij_map_by_ref": n_ij,
                "TN": TN, "FP": FP, "FN": FN, "TP": TP,
                "overall_accuracy": overall_acc,
                "tpr": tpr, "tnr": tnr, "ppv": ppv, "npv": npv,
            },
            "olofsson": olo,
            "binary_summary": {
                "mean": mean,
                "minus_2se": minus_2se,
                "plus_2se": plus_2se,
                "ci_lower": float(CI_lower[j]),
                "ci_upper": float(CI_upper[j]),
                "true": true_p1,
                "inside_ci": bool(olo["disturbed_in_CI"]),
            },
        }
        return out


    def evaluate_validation_collection(
        self,
        sensor_name: str,
        # continuous
        prob_attr: str = "interpreter_calibrated_prob_by_sensor",
        step: float = 0.001,
        # binary
        n_samples_map0: int = 100,
        n_samples_map1: int = 100,
        interpreter_agent=None,  # InterpreterAgent or None
        alpha: float = 0.05,     # 95% CI by default
        # plotting
        xlim_prop: tuple[float, float] | None = None,
        title_prefix: str = "Validation",
        save_dir: Optional[str] = None,
    ) -> dict:
        """
        Evaluate per-stack performance for BOTH:
          - continuous probability area distribution (P5/P50/P95 disturbed proportion + truth)
          - binary Olofsson-style estimate (mean +/- 2SE disturbed proportion + truth)
    
        Returns
        -------
        dict with:
          - continuous_df : pd.DataFrame
          - binary_df     : pd.DataFrame
          - fig_continuous, fig_binary (matplotlib figures)
        """
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from scipy.stats import norm
    
        # ----------------------------
        # helpers
        # ----------------------------
        def _to01(arr):
            a = np.asarray(arr, dtype=float)
            if np.nanmax(a) > 1.0 + 1e-6:
                a = a / 100.0
            return np.clip(a, 0.0, 1.0)
    
        def _continuous_metrics_for_stack(si: int):
            stack = self.stacks[si]
    
            if not hasattr(stack, prob_attr):
                raise AttributeError(f"Stack {stack.stack_id} missing attr {prob_attr!r}")
            d = getattr(stack, prob_attr)
            if not isinstance(d, dict) or sensor_name not in d:
                raise KeyError(f"Stack {stack.stack_id} missing {prob_attr}[{sensor_name!r}]")
    
            p_img01 = _to01(d[sensor_name])
            n_pix = int(p_img01.size)
    
            # sweep thresholds and compute disturbed proportion distribution
            thresholds = np.arange(0.0, 1.0 + 1e-12, step, dtype=float)
            props = np.empty_like(thresholds)
            flat = p_img01.ravel()
    
            # vectorized-ish: still loops thresholds but fast enough for ~1000 steps
            for i, t in enumerate(thresholds):
                props[i] = float(np.mean(flat > t))
    
            # percentiles
            p10, p50, p90 = np.percentile(props, [10, 50, 90])
    
            # truth disturbed proportion (full landscape)
            truth = np.asarray(stack.base_landscape.disturbed, dtype=bool)
            true_prop = float(truth.mean())
    
            return {
                "stack_index": si,
                "stack_id": stack.stack_id,
                "n_pixels": n_pix,
                "cont_p10": float(p10),
                "cont_p50": float(p50),
                "cont_p90": float(p90),
                "true_prop": float(true_prop),
                "abs_err": float(abs(p50 - true_prop)),
            }
    
        def _binary_olofsson_for_stack(si: int):
            """
            Per-stack Olofsson-like estimate.
            We compute:
              - W_i from *this stack's* binary map pixel proportions
              - p_ij from sampled points restricted to this stack
              - P_hat_j = sum_i W_i * p_ij(i,j)
              - SE via Olofsson variance expression
            """
            stack = self.stacks[si]
    
            if not hasattr(stack, "binary_class_by_sensor") or sensor_name not in stack.binary_class_by_sensor:
                raise AttributeError(
                    f"Stack {stack.stack_id} missing binary_class_by_sensor[{sensor_name!r}]"
                )
    
            # draw per-stack stratified sample (map 0 and map 1)
            samples = self.binary_stratified_sample(
                sensor_name=sensor_name,
                n_samples_per_class={0: int(n_samples_map0), 1: int(n_samples_map1)},
                sampling_seed=None,
                stack_indices=[si],
                include_truth=False,
                interpreter_agent=interpreter_agent,
            )
            if samples is None or len(samples) == 0:
                raise RuntimeError(f"No binary samples for stack {stack.stack_id}")
    
            # IMPORTANT: here we treat ROWS=map_class, COLS=ref_class
            # n_ij[i,j] = count(map=i, ref=j)
            n_ij = np.zeros((2, 2), dtype=float)
            for i in (0, 1):
                for j in (0, 1):
                    n_ij[i, j] = float(((samples["map_class"] == i) & (samples["ref_class"] == j)).sum())
            n_i = n_ij.sum(axis=1)  # by map class
    
            # conditional probs p_ij = P(ref=j | map=i)
            p_ij = np.zeros((2, 2), dtype=float)
            for i in (0, 1):
                if n_i[i] > 0:
                    p_ij[i, :] = n_ij[i, :] / n_i[i]
                else:
                    p_ij[i, :] = np.nan
    
            # map area proportions W_i from THIS stack's full map
            bin_map = np.asarray(stack.binary_class_by_sensor[sensor_name], dtype=int)
            N0 = float(np.sum(bin_map == 0))
            N1 = float(np.sum(bin_map == 1))
            N = N0 + N1
            if N <= 0:
                raise RuntimeError("Empty binary map?")
            W = np.array([N0 / N, N1 / N], dtype=float)
    
            # Olofsson adjusted area proportions P_hat_j = sum_i W_i * p_ij(i,j)
            P_hat = np.zeros(2, dtype=float)
            for j in (0, 1):
                P_hat[j] = np.nansum(W * p_ij[:, j])
    
            # SE^2(P_hat_j) = sum_i W_i^2 * [ p_ij(1-p_ij) / (n_i - 1) ]
            SE = np.zeros(2, dtype=float)
            for j in (0, 1):
                var_j = 0.0
                for i in (0, 1):
                    if n_i[i] > 1 and np.isfinite(p_ij[i, j]):
                        var_j += (W[i] ** 2) * (p_ij[i, j] * (1.0 - p_ij[i, j]) / (n_i[i] - 1.0))
                SE[j] = np.sqrt(var_j)
    
            z = norm.ppf(1.0 - alpha / 2.0)
            lower = np.clip(P_hat - z * SE, 0.0, 1.0)
            upper = np.clip(P_hat + z * SE, 0.0, 1.0)
    
            truth = np.asarray(stack.base_landscape.disturbed, dtype=bool)
            true_prop = float(truth.mean())
    
            # We care about class 1 (“disturbed”)
            j = 1
            mean1 = float(P_hat[j])
            se1 = float(SE[j])
            lo2 = float(np.clip(mean1 - 2.0 * se1, 0.0, 1.0))
            hi2 = float(np.clip(mean1 + 2.0 * se1, 0.0, 1.0))
    
            return {
                "stack_index": si,
                "stack_id": stack.stack_id,
                "W0": float(W[0]),
                "W1": float(W[1]),
                "bin_mean": mean1,
                "bin_se": se1,
                "bin_m2se": lo2,
                "bin_p2se": hi2,
                "bin_ci_lo": float(lower[j]),
                "bin_ci_hi": float(upper[j]),
                "true_prop": true_prop,
                "abs_err": float(abs(mean1 - true_prop)),
                "n_map0": int(n_i[0]),
                "n_map1": int(n_i[1]),
                "samples": samples
            }
    
        # ----------------------------
        # compute per-stack metrics
        # ----------------------------
        cont_rows = []
        bin_rows = []
    
        for si in range(len(self.stacks)):
            cont_rows.append(_continuous_metrics_for_stack(si))
            bin_rows.append(_binary_olofsson_for_stack(si))
    
        continuous_df = pd.DataFrame(cont_rows)
        binary_df = pd.DataFrame(bin_rows)
    
        # sort by mismatch (largest error last for plot readability)
        # Use continuous truth as the canonical ordering
        order = np.argsort(continuous_df["true_prop"].to_numpy())
        
        continuous_df = continuous_df.iloc[order].reset_index(drop=True)
        binary_df     = binary_df.iloc[order].reset_index(drop=True)
            
        # ----------------------------
        # Plot 1: continuous
        # ----------------------------
        x = np.arange(len(continuous_df))
    
        fig_cont, ax = plt.subplots(1, 1, figsize=(12, 5))
        ax.plot(x, continuous_df["cont_p10"].to_numpy(), label="P10 (disturbed prop)")
        ax.plot(x, continuous_df["cont_p50"].to_numpy(), label="P50 (median)")
        ax.plot(x, continuous_df["cont_p90"].to_numpy(), label="P90")
        ax.plot(x, continuous_df["true_prop"].to_numpy(), label="Truth", linewidth=2)
    
        ax.set_title(f"{title_prefix} — Continuous probability area metrics\nsensor={sensor_name}, source={prob_attr}")
        ax.set_xlabel("Validation stacks (sorted by TRUE disturbed proportion))")
        ax.set_ylabel("Disturbed proportion")
        if xlim_prop is not None:
            ax.set_ylim(xlim_prop)
        ax.grid(alpha=0.3)
        ax.legend(frameon=True)
        plt.tight_layout()
    
        # ----------------------------
        # Plot 2: binary
        # ----------------------------
        xb = np.arange(len(binary_df))
    
        fig_bin, axb = plt.subplots(1, 1, figsize=(12, 5))
        axb.plot(xb, binary_df["bin_m2se"].to_numpy(), label="Mean − 2SE")
        axb.plot(xb, binary_df["bin_mean"].to_numpy(), label="Mean (P̂ disturbed)")
        axb.plot(xb, binary_df["bin_p2se"].to_numpy(), label="Mean + 2SE")
        axb.plot(xb, binary_df["true_prop"].to_numpy(), label="Truth", linewidth=2)
    
        axb.set_title(f"{title_prefix} — Binary Olofsson-like metrics\nsensor={sensor_name}")
        axb.set_xlabel("Validation stacks (sorted by TRUE disturbed proportion)")
        axb.set_ylabel("Disturbed proportion")
        if xlim_prop is not None:
            axb.set_ylim(xlim_prop)
        axb.grid(alpha=0.3)
        axb.legend(frameon=True)
        plt.tight_layout()
    
        if save_dir is not None:
            fig_cont.savefig(
                os.path.join(save_dir, "continuous_summary.png"), dpi=150, bbox_inches="tight"
            )
            fig_bin.savefig(
                os.path.join(save_dir, "binary_summary.png"), dpi=150, bbox_inches="tight"
            )
            continuous_df.to_csv(os.path.join(save_dir, "continuous_metrics.csv"), index=False)
            binary_df.drop(columns=["samples"], errors="ignore").to_csv(
                os.path.join(save_dir, "binary_metrics.csv"), index=False
            )

        return {
            "continuous_df": continuous_df,
            "binary_df": binary_df,
            "fig_continuous": fig_cont,
            "fig_binary": fig_bin,
        }





        