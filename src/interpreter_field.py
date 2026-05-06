# interpreter_field.py

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist

from disturbance_helper_functions import (
    logistic,
    move_towards_mid,
    apply_lowpass_filter,
    spatial_uncertainty_scaler_field,
)


class InterpreterField:
    """
    Simulate an interpreter's probability-of-loss field (0–100%) from the
    truth in a DisturbanceLandscape.

    Pipeline (run during __init__):

        1. Base probability of loss (0..1):
            - p_detect = logistic(cover_loss)
            - p_def    = logistic(final_cover around threshold)
            - p_base   = p_detect * p_def

        2. Patch-level degradations:
            a) Gaussian probability adjustment (additive, truncated).
            b) Beta-based shrink towards 0.5 (more uncertainty).

        3. Spatial low-pass filter (edge effects).

        4. Spatial uncertainty field (gstools): scalers in [0,max_scaler],
           used to shrink probabilities towards 0.5.

    All probabilities are stored internally as 0..1; the exposed
    interpreter_field is 0..100%.
    """

    def __init__(self, name: str, truth, config: dict) -> None:
        """
        Parameters
        ----------
        name : str
            Name of the interpreter model (e.g., 'Expert', 'Novice').
        truth : DisturbanceLandscape
            Underlying truth landscape, providing:
                - cover_loss (0–100%)
                - final_cover (0–100%)
                - image (patch IDs)
                - config['thresholds']['final_cover_percent']
        config : dict
            Interpreter configuration dictionary. Suggested structure:

                {
                  "detection_curve": {
                      "pivot_loss": 15.0,
                      "slope": 0.3
                  },
                  "definition_curve": {
                      "slope": 0.4,
                      "threshold_override": null
                  },
                  "gaussian_patch_noise": {
                      "std_dev_percent": 8.0
                  },
                  "beta_patch_uncertainty": {
                      "alpha": 2.0,
                      "beta": 5.0
                  },
                  "lowpass_filter": {
                      "kernel_size": 3
                  },
                  "spatial_uncertainty": {
                      "spatial_autocorr_distance": 8.0,
                      "max_scaler": 0.3
                  }
                }

        Notes
        -----
        - Random seeds are derived from the timestamp; no seed in config.
        - All intermediate prob fields are in [0,1].
        """
        self.name = name
        self.truth = truth
        self.config = config

        # Timestamp-based master seed
        self.seed = int(time.time())

        # Core prob fields (0..1)
        self.prob_base: Optional[np.ndarray] = None
        self.prob_after_gaussian: Optional[np.ndarray] = None
        self.prob_after_beta: Optional[np.ndarray] = None
        self.prob_after_smoothing: Optional[np.ndarray] = None
        self.prob_final: Optional[np.ndarray] = None  # final 0..1

        # Patch-level metadata
        self.patch_gaussian_field: Optional[np.ndarray] = None
        self.patch_gaussian_offsets: dict[int, float] = {}
        self.patch_beta_field: Optional[np.ndarray] = None
        self.patch_beta_scaler: dict[int, float] = {}

        # Spatial uncertainty field
        self.spatial_uncertainty_field: Optional[np.ndarray] = None

        # Run full pipeline
        self._run_pipeline()

    # ------------------------------------------------------------------
    # Exposed probability (0–100%)
    # ------------------------------------------------------------------

    @property
    def interpreter_field(self) -> np.ndarray:
        """
        Final interpreter probability-of-loss field in [0,100].
        """
        if self.prob_final is None:
            raise RuntimeError("Interpreter pipeline has not been run successfully.")
        return self.prob_final * 100.0

    # ------------------------------------------------------------------
    # Config helper
    # ------------------------------------------------------------------

    def _type_cfg(self, step: str, type_key: str) -> dict:
        """Return config for `step`, with type-specific keys overriding global defaults."""
        global_cfg = self.config.get(step, {})
        type_cfg   = self.config.get("types", {}).get(type_key, {})
        return {**global_cfg, **type_cfg.get(step, {})}

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def _run_pipeline(self) -> None:
        if getattr(self.truth, "cover_loss", None) is None:
            # cover_loss = initial_cover - final_cover * (loss proportion)
            # but by your design, DisturbanceLandscape already has cover_loss
            raise RuntimeError("Truth object must provide 'cover_loss' array.")

        if getattr(self.truth, "final_cover", None) is None:
            raise RuntimeError("Truth object must provide 'final_cover' array.")

        if getattr(self.truth, "image", None) is None:
            raise RuntimeError("Truth object must provide 'image' (patch IDs).")

        # 1) Base probability of loss
        self._compute_base_probability()

        # 2) Patch-level degradations
        self._apply_patch_gaussian()
        self._apply_patch_beta()

        # 3) Spatial smoothing
        self._apply_lowpass_step()

        # 4) Spatial uncertainty shrink towards 0.5
        self._apply_spatial_uncertainty()

    # ------------------------------------------------------------------
    # Step 1: base probability of loss
    # ------------------------------------------------------------------

    def _compute_base_probability(self) -> None:
        """
        Compute base probability per forest type:

            p_detect = logistic(cover_loss; pivot_loss, slope)
            p_def    = logistic(final_cover around forest threshold)
            prob_base = p_detect * p_def

        Detection and definition curve params may differ by type via config['types'].
        """
        cover_loss  = np.asarray(self.truth.cover_loss,  dtype=float)  # 0..100
        final_cover = np.asarray(self.truth.final_cover, dtype=float)  # 0..100
        type_raster = self.truth.forest_type_raster

        truth_thresh = float(
            self.truth.config.get("thresholds", {}).get("final_cover_percent", 15.0)
        )

        p_base = np.zeros_like(cover_loss)

        for t_idx, type_key in enumerate(self.truth.type_keys):
            mask = (type_raster == t_idx)
            if not np.any(mask):
                continue

            det_cfg = self._type_cfg("detection_curve", type_key)
            def_cfg = self._type_cfg("definition_curve", type_key)

            pivot_loss   = det_cfg.get("pivot_loss", 15.0)
            slope_detect = det_cfg.get("slope", 0.3)

            threshold_override = def_cfg.get("threshold_override", None)
            threshold = float(threshold_override) if threshold_override is not None else truth_thresh
            slope_def = def_cfg.get("slope", 0.4)

            p_detect = logistic(cover_loss[mask], x0=pivot_loss, k=slope_detect)
            p_def    = logistic(threshold - final_cover[mask], x0=0.0, k=slope_def)
            p_base[mask] = np.clip(p_detect * p_def, 0.0, 1.0)

        self.prob_base = p_base

    # ------------------------------------------------------------------
    # Step 2a: Gaussian patch-level probability adjustment
    # ------------------------------------------------------------------

    def _apply_patch_gaussian(self) -> None:
        """
        Apply per-patch Gaussian probability offsets to prob_base.

        For each patch:
            offset ~ Normal(0, sigma), where sigma is given in percentage points
            in config['gaussian_patch_noise']['std_dev_percent'].

        p_new = clip(p_base + offset, 0, 1)

        Stores:
            prob_after_gaussian,
            patch_gaussian_field (raster of offsets),
            patch_gaussian_offsets (dict patch -> offset in 0..1).
        """
        if self.prob_base is None:
            raise RuntimeError("prob_base is not set.")

        prob         = self.prob_base.copy()
        patch_raster = self.truth.image
        type_raster  = self.truth.forest_type_raster

        # Build per-type sigma (0..1 scale)
        sigma_by_type = {
            t_idx: max(
                self._type_cfg("gaussian_patch_noise", type_key).get("std_dev_percent", 0.0),
                0.0,
            ) / 100.0
            for t_idx, type_key in enumerate(self.truth.type_keys)
        }

        nrows, ncols = prob.shape
        noise_field   = np.zeros((nrows, ncols), dtype=float)
        patch_offsets: dict[int, float] = {}

        # Skip entirely if all sigmas are zero
        if all(s <= 0 for s in sigma_by_type.values()):
            self.prob_after_gaussian    = prob
            self.patch_gaussian_field   = noise_field
            self.patch_gaussian_offsets = patch_offsets
            return

        rng = np.random.default_rng(self.seed)

        for pid in np.unique(patch_raster):
            mask = (patch_raster == pid)
            if not np.any(mask):
                continue

            t_idx = int(type_raster[mask][0])
            sigma = sigma_by_type.get(t_idx, 0.0)
            offset = rng.normal(loc=0.0, scale=sigma) if sigma > 0 else 0.0
            patch_offsets[int(pid)] = offset
            prob[mask]       += offset
            noise_field[mask] = offset

        prob = np.clip(prob, 0.0, 1.0)

        self.prob_after_gaussian    = prob
        self.patch_gaussian_field   = noise_field
        self.patch_gaussian_offsets = patch_offsets

    # ------------------------------------------------------------------
    # Step 2b: Beta-based “uncertainty shrink” toward 0.5
    # ------------------------------------------------------------------

    def _apply_patch_beta(self) -> None:
        """
        For each patch, draw a scaler ~ Beta(alpha, beta) in [0,1] and shrink
        the probability towards 0.5:

            p_new = p + scaler * (0.5 - p)

        Stores:
            prob_after_beta,
            patch_beta_field (raster of scalers),
            patch_beta_scaler (dict patch -> scaler).
        """
        if self.prob_after_gaussian is None:
            raise RuntimeError("prob_after_gaussian is not set.")

        prob         = self.prob_after_gaussian.copy()
        patch_raster = self.truth.image
        type_raster  = self.truth.forest_type_raster

        # Build per-type (alpha, beta) pairs
        beta_by_type = {
            t_idx: (
                self._type_cfg("beta_patch_uncertainty", type_key).get("alpha", 2.0),
                self._type_cfg("beta_patch_uncertainty", type_key).get("beta",  5.0),
            )
            for t_idx, type_key in enumerate(self.truth.type_keys)
        }

        nrows, ncols = prob.shape
        scaler_field  = np.zeros((nrows, ncols), dtype=float)
        patch_scalers: dict[int, float] = {}

        # Skip entirely if all types have degenerate params
        if all(a <= 0 or b <= 0 for a, b in beta_by_type.values()):
            self.prob_after_beta  = prob
            self.patch_beta_field = scaler_field
            self.patch_beta_scaler = patch_scalers
            return

        rng = np.random.default_rng(self.seed + 1)

        for pid in np.unique(patch_raster):
            mask = (patch_raster == pid)
            if not np.any(mask):
                continue

            t_idx = int(type_raster[mask][0])
            alpha, beta_param = beta_by_type.get(t_idx, (2.0, 5.0))

            if alpha > 0 and beta_param > 0:
                scaler = rng.beta(alpha, beta_param)
            else:
                scaler = 0.0

            patch_scalers[int(pid)] = scaler
            prob[mask]        = move_towards_mid(prob[mask], scaler, mid=0.5)
            scaler_field[mask] = scaler

        prob = np.clip(prob, 0.0, 1.0)

        self.prob_after_beta   = prob
        self.patch_beta_field  = scaler_field
        self.patch_beta_scaler = patch_scalers

    # ------------------------------------------------------------------
    # Step 3: low-pass filter (edge effects)
    # ------------------------------------------------------------------

    def _apply_lowpass_step(self) -> None:
        """
        Apply spatial smoothing (low-pass filter) to simulate edge effects.
        """
        if self.prob_after_beta is None:
            raise RuntimeError("prob_after_beta is not set.")

        lp_cfg = self.config.get("lowpass_filter", {})
        ksize = int(lp_cfg.get("kernel_size", 3))

        smoothed = apply_lowpass_filter(self.prob_after_beta, kernel_size=ksize)
        smoothed = np.clip(smoothed, 0.0, 1.0)

        self.prob_after_smoothing = smoothed

    # ------------------------------------------------------------------
    # Step 4: spatial uncertainty shrink towards 0.5
    # ------------------------------------------------------------------

    def _apply_spatial_uncertainty(self) -> None:
        """
        Create a spatial uncertainty field (0..max_scaler) and use it to
        shrink probabilities towards 0.5:

            p_final = p_smooth + scaler * (0.5 - p_smooth)
        """
        if self.prob_after_smoothing is None:
            raise RuntimeError("prob_after_smoothing is not set.")

        su_cfg = self.config.get("spatial_uncertainty", {})
        corr_len = su_cfg.get("spatial_autocorr_distance", 8.0)
        max_scaler = su_cfg.get("max_scaler", 0.3)

        prob = self.prob_after_smoothing.copy()
        nrows, ncols = prob.shape

        if max_scaler <= 0:
            self.prob_final = prob
            self.spatial_uncertainty_field = np.zeros((nrows, ncols), dtype=float)
            return

        scaler_field = spatial_uncertainty_scaler_field(
            shape=(nrows, ncols),
            spatial_autocorr_distance=corr_len,
            max_scaler=max_scaler,
            random_state=self.seed + 2,
        )

        prob = move_towards_mid(prob, scaler_field, mid=0.5)
        prob = np.clip(prob, 0.0, 1.0)

        self.prob_final = prob
        self.spatial_uncertainty_field = scaler_field

    # ------------------------------------------------------------------
    # Plotting helper
    # ------------------------------------------------------------------

    def plot_pipeline(self, figsize: tuple[int, int] = (14, 18), bins: int = 30) -> None:
        """
        Plot probability stages (as %) in a 5x2 grid:
          Left column: maps
          Right column: histograms for the same stage
        """
        if self.prob_final is None:
            raise RuntimeError("Interpreter pipeline has not been run successfully.")
    
        import numpy as np
        import matplotlib.pyplot as plt
    
        stages = [
            ("Base prob (%)", self.prob_base),
            ("After Gaussian patch noise (%)", self.prob_after_gaussian),
            ("After Beta shrink (%)", self.prob_after_beta),
            ("After smoothing (%)", self.prob_after_smoothing),
            (f"Final interpreter field ({self.name}) (%)", self.prob_final),
        ]
    
        fig, ax = plt.subplots(nrows=5, ncols=2, figsize=figsize)
    
        for i, (title, p01) in enumerate(stages):
            if p01 is None:
                raise RuntimeError(f"Missing stage '{title}' (value is None).")
    
            p = np.asarray(p01, dtype=float) * 100.0  # convert 0..1 -> 0..100
    
            # --- Map panel (left) ---
            im = ax[i, 0].imshow(p, vmin=0, vmax=100, cmap="viridis")
            ax[i, 0].set_title(title)
            ax[i, 0].axis("off")
            fig.colorbar(im, ax=ax[i, 0], fraction=0.046, pad=0.04)
    
            # --- Histogram panel (right) ---
            ax[i, 1].hist(p.ravel(), bins=bins, density=True, alpha=0.7)
            ax[i, 1].set_xlim(0, 100)
            ax[i, 1].set_title("Histogram")
            ax[i, 1].set_xlabel("Probability of loss (%)")
            ax[i, 1].set_ylabel("Density")
    
        plt.tight_layout()
        plt.show()