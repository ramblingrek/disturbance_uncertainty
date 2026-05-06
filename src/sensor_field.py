# sensor_field.py

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from disturbance_helper_functions import (
    cover_to_signal,
    add_gaussian_patch_noise_by_type,
    apply_lowpass_filter,
    add_spatially_correlated_noise,
)


class SensorField:
    """
    Simulate a sensor signal field from a DisturbanceLandscape 'truth' object,
    with type-specific behavior for signal transform and patch-level noise.

    Pipeline (run during __init__):

        1. For each forest type, convert initial and final percent cover to
           sensor signals (asymptotic transform) with type-specific params;
           compute difference:
               signal_diff = signal_before - signal_after

        2. Add per-patch Gaussian noise with std depending on forest type.

        3. Apply a low-pass 2D filter (uniform kernel).

        4. Add a spatially correlated Gaussian noise field (gstools).
    """

    def __init__(self, name: str, truth, config: dict) -> None:
        """
        Parameters
        ----------
        name : str
            Name of the sensor (e.g., 'SensorA', 'LandsatLike').
        truth : DisturbanceLandscape
            Underlying truth, providing:
                - truth.initial_cover
                - truth.final_cover
                - truth.image           (relabeled patches)
                - truth.forest_type_raster
                - truth.type_keys
        config : dict
            Sensor configuration dictionary.

            Expected structure:

                {
                  "signal_transform": {
                      "max_signal": 1000.0,   # defaults
                      "curve_k": 3.0
                  },
                  "gaussian_patch_noise": {
                      "std_proportion_of_range": 0.0  # default if not overridden
                  },
                  "types": {
                      "type1": {
                          "signal_transform": {
                              "max_signal": 1000.0,
                              "curve_k": 3.0
                          },
                          "gaussian_patch_noise": {
                              "std_proportion_of_range": 0.02
                          }
                      },
                      "type2": {
                          "signal_transform": { ... },
                          "gaussian_patch_noise": { ... }
                      }
                  },
                  "lowpass_filter": {
                      "kernel_size": 3
                  },
                  "spatial_noise": {
                      "spatial_autocorr_distance": 10.0,
                      "amplitude_proportion_of_range": 0.1
                  }
                }

            - type-specific entries override the top-level defaults.
            - random seeds are NOT read from config; we seed from timestamp.
        """
        self.name = name
        self.truth = truth
        self.config = config

        # Seed from timestamp (used for both patch noise and spatial noise)
        self.seed = int(time.time())

        # Fields at each pipeline step
        self.signal_before: Optional[np.ndarray] = None
        self.signal_after: Optional[np.ndarray] = None
        self.signal_diff_raw: Optional[np.ndarray] = None

        self.patch_noise_field: Optional[np.ndarray] = None
        self.patch_noise_per_patch: dict[int, float] = {}

        self.signal_after_patch_noise: Optional[np.ndarray] = None
        self.signal_after_smoothing: Optional[np.ndarray] = None

        self.spatial_noise_field: Optional[np.ndarray] = None
        self.sensor_field: Optional[np.ndarray] = None  # final output

        # Run the pipeline
        self._run_pipeline()

    # ------------------------------------------------------------------
    # Internal: main pipeline
    # ------------------------------------------------------------------

    def _run_pipeline(self) -> None:
        """Run the full sensor simulation pipeline."""
        if self.truth.initial_cover is None or self.truth.final_cover is None:
            raise RuntimeError("Truth object lacks initial_cover or final_cover.")

        if getattr(self.truth, "forest_type_raster", None) is None:
            raise RuntimeError("Truth object lacks forest_type_raster for type-specific behavior.")

        # 1) Cover → signal → difference (per type)
        self._compute_signal_difference_step()

        # 2) Per-patch Gaussian noise (per type)
        self._add_patch_noise_step()

        # 3) Low-pass filter
        self._apply_lowpass_step()

        # 4) Spatially correlated noise
        self._add_spatial_noise_step()

    # ------------------------------------------------------------------
    # Step 1: cover → signal difference (per type)
    # ------------------------------------------------------------------

    def _compute_signal_difference_step(self) -> None:
        init = self.truth.initial_cover
        final = self.truth.final_cover
        type_raster = self.truth.forest_type_raster

        global_st_cfg = self.config.get("signal_transform", {})
        types_cfg = self.config.get("types", {})

        signal_before = np.zeros_like(init, dtype=float)
        signal_after = np.zeros_like(init, dtype=float)

        for t_idx, type_key in enumerate(self.truth.type_keys):
            mask = (type_raster == t_idx)
            if not np.any(mask):
                continue

            type_cfg = types_cfg.get(type_key, {})
            st_cfg = type_cfg.get("signal_transform", {})

            max_signal = st_cfg.get(
                "max_signal",
                global_st_cfg.get("max_signal", 1000.0),
            )
            curve_k = st_cfg.get(
                "curve_k",
                global_st_cfg.get("curve_k", 3.0),
            )

            # Apply transform only to this type's pixels
            sb = cover_to_signal(init[mask], max_signal=max_signal, curve_k=curve_k)
            sa = cover_to_signal(final[mask], max_signal=max_signal, curve_k=curve_k)

            signal_before[mask] = sb
            signal_after[mask] = sa

        self.signal_before = signal_before
        self.signal_after = signal_after
        self.signal_diff_raw = signal_before - signal_after

    # ------------------------------------------------------------------
    # Step 2: per-patch Gaussian noise (per type)
    # ------------------------------------------------------------------

    def _add_patch_noise_step(self) -> None:
        type_raster = self.truth.forest_type_raster
        patch_raster = self.truth.image

        global_g_cfg = self.config.get("gaussian_patch_noise", {})
        types_cfg = self.config.get("types", {})

        # Build type -> std_prop mapping
        std_prop_by_type: dict[int, float] = {}
        for t_idx, type_key in enumerate(self.truth.type_keys):
            type_cfg = types_cfg.get(type_key, {})
            g_cfg = type_cfg.get("gaussian_patch_noise", {})

            std_prop = g_cfg.get(
                "std_proportion_of_range",
                global_g_cfg.get("std_proportion_of_range", 0.0),
            )
            std_prop_by_type[t_idx] = std_prop

        # If all std props are zero, skip noise
        if all(v <= 0 for v in std_prop_by_type.values()):
            self.signal_after_patch_noise = self.signal_diff_raw.copy()
            self.patch_noise_field = np.zeros_like(self.signal_diff_raw, dtype=float)
            self.patch_noise_per_patch = {}
            return

        noisy, noise_field, patch_offsets = add_gaussian_patch_noise_by_type(
            self.signal_diff_raw,
            patch_raster=patch_raster,
            type_raster=type_raster,
            std_prop_by_type=std_prop_by_type,
            random_state=self.seed,  # timestamp-based seed
        )

        self.signal_after_patch_noise = noisy
        self.patch_noise_field = noise_field
        self.patch_noise_per_patch = patch_offsets

    # ------------------------------------------------------------------
    # Step 3: low-pass filter
    # ------------------------------------------------------------------

    def _apply_lowpass_step(self) -> None:
        lp_cfg = self.config.get("lowpass_filter", {})
        ksize = int(lp_cfg.get("kernel_size", 3))

        self.signal_after_smoothing = apply_lowpass_filter(
            self.signal_after_patch_noise,
            kernel_size=ksize,
        )

    # ------------------------------------------------------------------
    # Step 4: spatially correlated noise (gstools)
    # ------------------------------------------------------------------

    def _add_spatial_noise_step(self) -> None:
        sn_cfg = self.config.get("spatial_noise", {})
        corr_len = sn_cfg.get("spatial_autocorr_distance", 10.0)
        amp_prop = sn_cfg.get("amplitude_proportion_of_range", 0.1)

        if amp_prop <= 0:
            self.sensor_field = self.signal_after_smoothing.copy()
            self.spatial_noise_field = np.zeros_like(self.sensor_field, dtype=float)
            return

        noisy, noise_field = add_spatially_correlated_noise(
            self.signal_after_smoothing,
            spatial_autocorr_distance=corr_len,
            amplitude_proportion_of_range=amp_prop,
            random_state=self.seed + 1,  # different but related seed
        )
        self.sensor_field = noisy
        self.spatial_noise_field = noise_field

    # ------------------------------------------------------------------
    # Convenience: quick plotting
    # ------------------------------------------------------------------

    def plot_pipeline(self, figsize: tuple[int, int] = (16, 8)) -> None:
        """
        Plot key steps of the sensor pipeline:

            (0,0): raw signal difference (before noise)
            (0,1): after patch noise
            (0,2): after smoothing
            (1,0): final sensor field
            (1,1): patch-level noise
            (1,2): spatial noise
        """
        if self.sensor_field is None:
            raise RuntimeError("Sensor pipeline has not been run successfully.")

        fig, ax = plt.subplots(2, 3, figsize=figsize)

        im00 = ax[0, 0].imshow(self.signal_diff_raw, cmap="viridis")
        ax[0, 0].set_title("Signal diff (raw)")
        ax[0, 0].axis("off")
        fig.colorbar(im00, ax=ax[0, 0], fraction=0.046, pad=0.04)

        im01 = ax[0, 1].imshow(self.signal_after_patch_noise, cmap="viridis")
        ax[0, 1].set_title("After patch noise")
        ax[0, 1].axis("off")
        fig.colorbar(im01, ax=ax[0, 1], fraction=0.046, pad=0.04)

        im02 = ax[0, 2].imshow(self.signal_after_smoothing, cmap="viridis")
        ax[0, 2].set_title("After smoothing")
        ax[0, 2].axis("off")
        fig.colorbar(im02, ax=ax[0, 2], fraction=0.046, pad=0.04)

        im10 = ax[1, 0].imshow(self.sensor_field, cmap="viridis")
        ax[1, 0].set_title(f"Final sensor field ({self.name})")
        ax[1, 0].axis("off")
        fig.colorbar(im10, ax=ax[1, 0], fraction=0.046, pad=0.04)

        im11 = ax[1, 1].imshow(self.patch_noise_field, cmap="bwr")
        ax[1, 1].set_title("Patch noise")
        ax[1, 1].axis("off")
        fig.colorbar(im11, ax=ax[1, 1], fraction=0.046, pad=0.04)

        im12 = ax[1, 2].imshow(self.spatial_noise_field, cmap="bwr")
        ax[1, 2].set_title("Spatial noise")
        ax[1, 2].axis("off")
        fig.colorbar(im12, ax=ax[1, 2], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()