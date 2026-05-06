# code
import time
from typing import Optional, Tuple

import numpy as np
import rasterio
from rasterio.io import DatasetReader
from scipy.stats import beta
import matplotlib.pyplot as plt


class DisturbanceLandscape:
    """
    Generate a synthetic disturbance landscape with TWO forest types.

    Workflow (all done in __init__):

        1. Load patch raster, relabel patches to 0..N-1.
        2. Assign each patch to one of two types based on config proportions.
        3. For each type:
            a. Compute (alpha, beta) for initial cover from target_mean, kappa.
            b. Draw initial cover per patch, build initial_cover raster (0–100%).
            c. Draw proportional loss per patch from a Beta mixture:
                - background: small loss
                - catastrophic: large loss
        4. Compute cover_loss and final_cover rasters.
        5. Build disturbed mask:
              disturbed = (final_cover < final_cover_percent) & (cover_loss > 0).
        6. Optionally retry if no disturbed pixels are found.
    """

    # ----------------------------- PUBLIC API -----------------------------

    def __init__(self, tif_path: str, config: dict) -> None:
        """
        Parameters
        ----------
        tif_path : str
            Path to the input TIFF file containing patch labels.
        config : dict
            Configuration dictionary (disturbance_config_two_types).
        """
        self.tif_path = tif_path
        self.config = config

        # --- Random seed ---
        seed = self.config.get("random", {}).get("seed", None)
        if seed is None:
            seed = int(time.time())
        self.seed = seed
        np.random.seed(self.seed)

        # --- Load raster ---
        band = self.config.get("raster", {}).get("band", 1)
        self.band: int = band
        self.src: DatasetReader = rasterio.open(self.tif_path)
        self.image: np.ndarray = self.src.read(self.band)
        self.original_image: np.ndarray = self.image.copy()

        # Patches & forest types
        self.patches: Optional[list[int]] = None
        self.forest_type_ids: Optional[np.ndarray] = None   # per-patch type index
        self.forest_type_raster: Optional[np.ndarray] = None  # same shape as image

        # Type metadata
        self.type_keys: list[str] = []          # e.g. ["type1", "type2"]
        self.type_names: dict[int, str] = {}    # idx -> human-readable name

        # Per-type initial cover params
        self.initial_params: dict[int, Tuple[float, float]] = {}  # type_idx -> (alpha, beta)

        # Mixture params for loss, per type
        self.mixture_params: dict[int, dict[str, float]] = {}

        # Rasters
        self.initial_cover: Optional[np.ndarray] = None       # 0–100
        self.proportional_loss: Optional[np.ndarray] = None   # 0–100 (% of initial)
        self.cover_loss: Optional[np.ndarray] = None          # 0–100 (% lost)
        self.final_cover: Optional[np.ndarray] = None         # 0–100 (% remaining)
        self.disturbed: Optional[np.ndarray] = None           # bool

        # Stats
        self.n_retries: int = 0

        # --- Full pipeline ---
        self._relabel_patches()
        self._setup_types_and_assign_patches()
        self._calculate_initial_forest_cover()

        sim_cfg = self.config.get("simulation", {})
        max_retries = sim_cfg.get("max_retries", 10)

        # Retry-loop in case we get zero disturbed pixels
        for attempt in range(max_retries):
            self._set_mixture_parameters()
            self._generate_loss_rasters()
            self._update_disturbed_mask()

            if self.disturbed_fraction is not None and self.disturbed_fraction > 0:
                self.n_retries = attempt
                break
        else:
            print("Warning: no disturbed pixels generated after max_retries attempts.")

    # ----------------------------- PROPERTIES -----------------------------

    @property
    def disturbed_fraction(self) -> Optional[float]:
        """Fraction of pixels classified as disturbed."""
        if self.disturbed is None:
            return None
        return float(np.mean(self.disturbed))

    @property
    def mean_initial_cover(self) -> Optional[float]:
        """Mean initial cover (%) over the landscape."""
        if self.initial_cover is None:
            return None
        return float(np.mean(self.initial_cover))

    @property
    def mean_final_cover(self) -> Optional[float]:
        """Mean final cover (%) over the landscape."""
        if self.final_cover is None:
            return None
        return float(np.mean(self.final_cover))

    @property
    def mean_cover_loss(self) -> Optional[float]:
        """Mean cover loss (%) over the landscape."""
        if self.cover_loss is None:
            return None
        return float(np.mean(self.cover_loss))

    def disturbed_fraction_by_type(self, type_index: int) -> Optional[float]:
        """Fraction of disturbed pixels within a given forest type."""
        if self.disturbed is None or self.forest_type_raster is None:
            return None
        mask_type = (self.forest_type_raster == type_index)
        if mask_type.sum() == 0:
            return 0.0
        return float(np.mean(self.disturbed[mask_type]))

    # -------------------------- INTERNAL METHODS --------------------------

    def _relabel_patches(self) -> None:
        """
        Relabel patches in the image to consecutive int IDs starting at 0.
        Operates on self.image in-place and populates self.patches.
        """
        img = self.image
        uniques = np.unique(img)
        patches: list[int] = []

        for l, z in enumerate(uniques):
            mask = (img == z)
            img[mask] = l
            patches.append(l)

        self.image = img
        self.patches = patches

    def _setup_types_and_assign_patches(self) -> None:
        """
        Read type configs and randomly assign patches to two types based on
        'proportion_of_patches' for each type.
        """
        if self.patches is None:
            raise RuntimeError("Patches have not been relabeled yet.")

        types_cfg = self.config.get("types", {})
        if len(types_cfg) != 2:
            raise ValueError("This implementation expects exactly two types.")

        # Stable order of type keys
        self.type_keys = sorted(types_cfg.keys())  # e.g., ["type1", "type2"]

        # Extract proportions (normalize if needed)
        proportions = []
        for key in self.type_keys:
            tcfg = types_cfg[key]
            p = tcfg.get("proportion_of_patches", 0.5)
            proportions.append(p)

        proportions = np.array(proportions, dtype=float)
        proportions = proportions / proportions.sum()  # normalize to sum to 1

        # Human-readable names
        for idx, key in enumerate(self.type_keys):
            name = types_cfg[key].get("name", key)
            self.type_names[idx] = name

        # Assign each patch a type index via multinomial sampling
        n_patches = len(self.patches)
        type_ids = np.random.choice(len(self.type_keys), size=n_patches, p=proportions)
        self.forest_type_ids = type_ids  # per patch

        # Build forest_type_raster (same shape as image)
        type_raster = np.zeros_like(self.image, dtype=int)
        for patch_id, t_idx in zip(self.patches, type_ids):
            mask = (self.image == patch_id)
            type_raster[mask] = t_idx

        self.forest_type_raster = type_raster

    @staticmethod
    def _beta_from_mean_and_kappa(mean: float, kappa: float) -> Tuple[float, float]:
        """
        Given desired mean and kappa=alpha+beta for a Beta distribution on [0,1],
        return (alpha, beta_param).
        """
        if not (0.0 < mean < 1.0):
            raise ValueError("mean must be in (0,1)")
        if kappa <= 0:
            raise ValueError("kappa must be > 0")

        alpha = mean * kappa
        beta_param = (1.0 - mean) * kappa
        return alpha, beta_param

    def _calculate_initial_forest_cover(self) -> None:
        """
        For each type, solve for (alpha, beta) of the initial cover distribution
        using target_mean and kappa, then draw an initial cover value per patch
        and assign to a 0–100% raster.
        """
        if self.patches is None or self.forest_type_ids is None:
            raise RuntimeError("Patches or type assignments are not set.")

        types_cfg = self.config.get("types", {})
        initial_cover = np.zeros_like(self.image, dtype=float)

        # Clear any previous records
        self.initial_params.clear()

        # For each type index, compute its (alpha, beta) and assign covers
        for t_idx, type_key in enumerate(self.type_keys):
            tcfg = types_cfg[type_key]
            init_cfg = tcfg.get("initial_cover", {})

            target_mean = init_cfg.get("target_mean", 0.9)  # in (0,1)
            kappa = init_cfg.get("kappa", 20.0)

            alpha0, beta0 = self._beta_from_mean_and_kappa(target_mean, kappa)
            self.initial_params[t_idx] = (alpha0, beta0)

            # Which patches belong to this type?
            patch_ids = [p for p, t in zip(self.patches, self.forest_type_ids) if t == t_idx]
            n_type_patches = len(patch_ids)
            if n_type_patches == 0:
                continue

            # Draw covers for these patches
            draws = beta.rvs(alpha0, beta0, size=n_type_patches)
            init_vals = np.trunc(draws * 1000.0) / 10.0  # 0–100, one decimal

            for patch_id, c0 in zip(patch_ids, init_vals):
                mask = (self.image == patch_id)
                initial_cover[mask] = c0

        self.initial_cover = initial_cover

    def _set_mixture_parameters(self) -> None:
        """
        For each type, read mixture parameters for proportional loss from config
        and store them.
        """
        types_cfg = self.config.get("types", {})
        self.mixture_params.clear()

        for t_idx, type_key in enumerate(self.type_keys):
            tcfg = types_cfg[type_key]
            prop_cfg = tcfg.get("proportional_loss", {})
            mix_cfg = prop_cfg.get("mixture", {})

            params = {
                "p_cat": mix_cfg.get("catastrophic_probability", 0.01),
                "bg_alpha": mix_cfg.get("background_alpha", 0.3),
                "bg_beta": mix_cfg.get("background_beta", 8.0),
                "cat_alpha": mix_cfg.get("catastrophic_alpha", 8.0),
                "cat_beta": mix_cfg.get("catastrophic_beta", 0.7),
            }
            self.mixture_params[t_idx] = params

    @staticmethod
    def _beta_mixture_rvs(
        n: int,
        alpha1: float,
        beta1: float,
        alpha2: float,
        beta2: float,
        p_catastrophic: float,
    ) -> np.ndarray:
        """
        Draw samples from a two-component Beta mixture distribution.

        With probability p_catastrophic:  Beta(alpha2, beta2)  (catastrophic)
        With probability 1 - p_catastrophic: Beta(alpha1, beta1) (background)
        """
        u = np.random.rand(n)
        r = np.empty(n, dtype=float)

        mask_cat = (u < p_catastrophic)
        n_cat = mask_cat.sum()
        n_bg = n - n_cat

        if n_cat > 0:
            r[mask_cat] = beta.rvs(alpha2, beta2, size=n_cat)
        if n_bg > 0:
            r[~mask_cat] = beta.rvs(alpha1, beta1, size=n_bg)

        return r

    def _generate_loss_rasters(self) -> None:
        """
        For each type, use its Beta mixture parameters to draw proportional loss
        per patch, then compute cover_loss and final_cover rasters.
        """
        if (
            self.patches is None
            or self.forest_type_ids is None
            or self.initial_cover is None
        ):
            raise RuntimeError(
                "Need patches, type assignments, and initial cover before generating loss."
            )

        proportional_loss = np.zeros_like(self.image, dtype=float)

        # For each type, draw losses for its patches
        for t_idx, params in self.mixture_params.items():
            p_cat = params["p_cat"]
            bg_alpha = params["bg_alpha"]
            bg_beta = params["bg_beta"]
            cat_alpha = params["cat_alpha"]
            cat_beta = params["cat_beta"]

            # Which patches belong to this type?
            patch_ids = [p for p, t in zip(self.patches, self.forest_type_ids) if t == t_idx]
            n_type_patches = len(patch_ids)
            if n_type_patches == 0:
                continue

            draws = self._beta_mixture_rvs(
                n=n_type_patches,
                alpha1=bg_alpha,
                beta1=bg_beta,
                alpha2=cat_alpha,
                beta2=cat_beta,
                p_catastrophic=p_cat,
            )
            loss_vals = np.trunc(draws * 1000.0) / 10.0  # 0–100

            for patch_id, L in zip(patch_ids, loss_vals):
                mask = (self.image == patch_id)
                proportional_loss[mask] = L

        cover_loss = self.initial_cover * (proportional_loss / 100.0)
        final_cover = self.initial_cover - cover_loss

        self.proportional_loss = proportional_loss
        self.cover_loss = cover_loss
        self.final_cover = final_cover

    def _update_disturbed_mask(self) -> None:
        """
        Build disturbed mask based on final cover and cover loss:

            disturbed = (final_cover < final_cover_percent) & (cover_loss > 0)
        """
        if self.final_cover is None or self.cover_loss is None:
            raise RuntimeError("Need final_cover and cover_loss to define disturbance.")

        final_thresh = self.config.get("thresholds", {}).get(
            "final_cover_percent", 15.0
        )

        disturbed = (self.final_cover < final_thresh) & (self.cover_loss > 0)
        self.disturbed = disturbed

    # ------------------------------ PLOTTING ------------------------------

    def plot_results(
        self,
        show_types: bool = True,
        figsize: tuple[int, int] = (16, 8),
    ) -> None:
        """
        Plot a 2x3 summary:

            (0,0): Forest type map (if show_types) else Initial forest cover (%)
            (0,1): Initial forest cover (%) or Proportional loss (if types shown)
            (0,2): Proportional loss (% of initial)
            (1,0): Cover loss (%)
            (1,1): Final forest cover (%)
            (1,2): Disturbed mask + final cover histogram inset (conceptually)

        For simplicity, we’ll stick with:

            (0,0): Forest type map (if show_types) else Initial cover
            (0,1): Initial cover
            (0,2): Proportional loss
            (1,0): Cover loss
            (1,1): Final cover
            (1,2): Disturbed mask + final cover histogram
        """
        if (
            self.initial_cover is None
            or self.proportional_loss is None
            or self.cover_loss is None
            or self.final_cover is None
            or self.disturbed is None
        ):
            raise RuntimeError(
                "Simulation not fully initialized; missing one or more rasters."
            )

        plotting_cfg = self.config.get("plotting", {})
        type_cmap = plotting_cfg.get("type_cmap", "tab10")
        init_cmap = plotting_cfg.get("initial_cover_cmap", "Greens")
        loss_prop_cmap = plotting_cfg.get("proportional_loss_cmap", "Oranges")
        loss_abs_cmap = plotting_cfg.get("loss_abs_cmap", "Reds")
        final_cmap = plotting_cfg.get("final_cover_cmap", "Blues")
        disturbed_cmap = plotting_cfg.get("disturbed_cmap", "Greys_r")

        final_thresh = self.config.get("thresholds", {}).get(
            "final_cover_percent", 15.0
        )

        fig, ax = plt.subplots(2, 3, figsize=figsize)

        # (0,0): Forest type map or Initial cover
        if show_types and self.forest_type_raster is not None:
            im00 = ax[0, 0].imshow(self.forest_type_raster, cmap=type_cmap)
            ax[0, 0].set_title("Forest type (0/1)")
            ax[0, 0].axis("off")
            fig.colorbar(im00, ax=ax[0, 0], fraction=0.046, pad=0.04)
        else:
            im00 = ax[0, 0].imshow(self.initial_cover, cmap=init_cmap, vmin=0, vmax=100)
            ax[0, 0].set_title("Initial Cover (%)")
            ax[0, 0].axis("off")
            fig.colorbar(im00, ax=ax[0, 0], fraction=0.046, pad=0.04)

        # (0,1): Initial cover
        im01 = ax[0, 1].imshow(self.initial_cover, cmap=init_cmap, vmin=0, vmax=100)
        ax[0, 1].set_title("Initial Cover (%)")
        ax[0, 1].axis("off")
        fig.colorbar(im01, ax=ax[0, 1], fraction=0.046, pad=0.04)

        # (0,2): Proportional loss (% of initial)
        im02 = ax[0, 2].imshow(self.proportional_loss, cmap=loss_prop_cmap, vmin=0, vmax=100)
        ax[0, 2].set_title("Proportional Loss (% of initial)")
        ax[0, 2].axis("off")
        fig.colorbar(im02, ax=ax[0, 2], fraction=0.046, pad=0.04)

        # (1,0): Cover loss (%)
        im10 = ax[1, 0].imshow(self.cover_loss, cmap=loss_abs_cmap, vmin=0, vmax=100)
        ax[1, 0].set_title("Cover Loss (%)")
        ax[1, 0].axis("off")
        fig.colorbar(im10, ax=ax[1, 0], fraction=0.046, pad=0.04)

        # (1,1): Final cover (%)
        im11 = ax[1, 1].imshow(self.final_cover, cmap=final_cmap, vmin=0, vmax=100)
        ax[1, 1].set_title("Final Cover (%)")
        ax[1, 1].axis("off")
        fig.colorbar(im11, ax=ax[1, 1], fraction=0.046, pad=0.04)

        # (1,2): Disturbed mask + histogram
        im12 = ax[1, 2].imshow(self.disturbed, cmap=disturbed_cmap)
        frac = self.disturbed_fraction if self.disturbed_fraction is not None else float("nan")
        ax[1, 2].set_title(
            f"Disturbed (final < {final_thresh:.1f}%, loss > 0)\n"
            f"Fraction disturbed: {frac:.3f}"
        )
        ax[1, 2].axis("off")
        fig.colorbar(im12, ax=ax[1, 2], fraction=0.046, pad=0.04)

        # Small inset histogram of final cover in the bottom-right panel
        inset_ax = ax[1, 2].inset_axes([0.05, 0.05, 0.5, 0.4])
        inset_ax.hist(self.final_cover.ravel(), bins=30, density=True, alpha=0.7)
        inset_ax.axvline(final_thresh, linestyle="--", linewidth=1)
        inset_ax.set_title("Final cover hist.", fontsize=8)
        inset_ax.tick_params(labelsize=6)

        plt.tight_layout()
        plt.show()

    # ------------------------------- CLEANUP ------------------------------

    def close(self) -> None:
        """Close the underlying rasterio dataset."""
        if self.src is not None and not self.src.closed:
            self.src.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
