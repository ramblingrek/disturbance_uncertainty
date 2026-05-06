# disturbance_helper_functions.py

import numpy as np
from scipy.ndimage import uniform_filter
import os
import csv
import hashlib
import datetime
from typing import Any, Dict, Optional


try:
    import gstools as gs
except ImportError:
    gs = None

# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

def logistic(x: np.ndarray, x0: float, k: float) -> np.ndarray:
    """
    Logistic function:

        p = 1 / (1 + exp(-k * (x - x0)))

    x0 : pivot (where p = 0.5 if k > 0)
    k  : slope (steepness). Larger |k| -> steeper transition.
    """
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))


def move_towards_mid(prob: np.ndarray, scaler: np.ndarray, mid: float = 0.5) -> np.ndarray:
    """
    Move probability values towards 'mid' (default 0.5) by a fraction given
    in 'scaler' (0..1):

        p_new = p + scaler * (mid - p)
    """
    prob = np.asarray(prob, dtype=float)
    scaler = np.asarray(scaler, dtype=float)
    return prob + scaler * (mid - prob)

# ---------------------------------------------------------------------
# tools to export dictionaries.
# ---------------------------------------------------------------------



def flatten_config(obj: Any, parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    items: Dict[str, Any] = {}

    if isinstance(obj, dict):
        for k, v in obj.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            items.update(flatten_config(v, new_key, sep))

    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            new_key = f"{parent_key}[{i}]"
            items.update(flatten_config(v, new_key, sep))

    else:
        items[parent_key] = obj

    return items


def pacific_timestamp() -> str:
    """
    Return current Pacific time as ISO string (UTC-8).
    """
    now_utc = datetime.datetime.utcnow()
    pacific = now_utc - datetime.timedelta(hours=8)
    return pacific.isoformat(timespec="seconds")


def generate_run_id(prefix: str = "run_ID") -> str:
    now = datetime.datetime.utcnow()
    pacific = now - datetime.timedelta(hours=8)
    timestamp = pacific.strftime("%Y%m%d_%H%M%S")
    hash_input = f"{timestamp}_{pacific.microsecond}".encode("utf-8")
    short_hash = hashlib.sha1(hash_input).hexdigest()[:6]
    return f"{prefix}_{timestamp}_{short_hash}"


def _append_to_master_log(
    csv_dir: str,
    created_pacific_time: str,
    run_id: str,
    notes: Optional[str],
    master_filename: str = "master_experiment_logger.csv",
) -> str:
    """
    Ensure master log exists, and append one row:
      created_utc, run_id, notes
    """
    master_path = os.path.join(csv_dir, master_filename)
    exists = os.path.exists(master_path)

    with open(master_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["created_pacific_time", "run_id", "notes"])
        writer.writerow([created_pacific_time, run_id, "" if notes is None else notes])

    return master_path

    
def export_with_metadata(
    config: Dict[str, Any],
    csv_path: str = "experiment_logs/",
    run_id: Optional[str] = None,
    notes: Optional[str] = None,
) -> str:
    """
    Export a nested experiment config dictionary to:
      1) A per-run CSV (flattened key,value) named <run_id>.csv
      2) A master ledger CSV named master_experiment_logger.csv (appended)

    Required:
      - config

    Optional:
      - csv_path (directory; default "experiment_logs/")
      - run_id (auto-generated if None)
      - notes (stored in both files)
    """
    os.makedirs(csv_path, exist_ok=True)

    if run_id is None:
        run_id = generate_run_id()

    created_pacific_time = pacific_timestamp()

    # 1) Write per-run flattened config
    wrapped = {
        "run_metadata": {
            "run_id": run_id,
            "created_pacific_time": created_pacific_time,
            "notes": notes,
        },
        **config,
    }

    flat = flatten_config(wrapped)
    run_csv_file = os.path.join(csv_path, f"{run_id}.csv")

    with open(run_csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "value"])
        for key in sorted(flat.keys()):
            val = flat[key]
            writer.writerow([key, "None" if val is None else val])

    # 2) Append to master log
    _append_to_master_log(
        csv_dir=csv_path,
        created_pacific_time=created_pacific_time,
        run_id=run_id,
        notes=notes,
    )

    return run_csv_file

    
#USAGE NOTES
# experiment_config = {
#     "disturbance_config": {...},
#     "sensors": [...],
#     "interpreter": {...},
# }

# export_with_metadata(
#     experiment_config,
#     notes="Higher interpreter uncertainty, lower sensor noise"
# )


            
# ---------------------------------------------------------------------
# 1) Cover → signal transform (asymptotic / log-like)
# ---------------------------------------------------------------------

def cover_to_signal(
    cover_percent: np.ndarray,
    max_signal: float = 1000.0,
    curve_k: float = 3.0,
) -> np.ndarray:
    """
    Map percent cover (0–100) to a sensor signal (0–max_signal) using a
    concave, asymptotic function (log-like-ish).

    cover_percent : np.ndarray
        Percent cover in [0,100].
    max_signal : float
        Maximum signal value when cover ~100%.
    curve_k : float
        Controls curvature. Higher k -> more concave (diminishing returns).

    Formula (x in [0,1]):
        signal = max_signal * log1p(k * x) / log1p(k)
    """
    cover_fraction = np.clip(cover_percent / 100.0, 0.0, 1.0)
    k = max(curve_k, 1e-6)
    signal = max_signal * np.log1p(k * cover_fraction) / np.log1p(k)
    return signal


def compute_signal_difference(
    initial_cover: np.ndarray,
    final_cover: np.ndarray,
    max_signal: float = 1000.0,
    curve_k: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert initial and final cover maps to sensor signals, then compute
    the difference (initial - final).

    Returns:
        signal_before, signal_after, signal_diff
    """
    signal_before = cover_to_signal(initial_cover, max_signal=max_signal, curve_k=curve_k)
    signal_after = cover_to_signal(final_cover, max_signal=max_signal, curve_k=curve_k)
    signal_diff = signal_before - signal_after
    return signal_before, signal_after, signal_diff


# ---------------------------------------------------------------------
# 2) Per-patch Gaussian noise, with per-type std settings
# ---------------------------------------------------------------------

def add_gaussian_patch_noise_by_type(
    signal: np.ndarray,
    patch_raster: np.ndarray,
    type_raster: np.ndarray,
    std_prop_by_type: dict[int, float],
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[int, float]]:
    """
    Add Gaussian noise per patch as a constant offset, with standard
    deviation depending on the forest type of each patch.

    signal : np.ndarray
        Input 2D signal field.
    patch_raster : np.ndarray
        2D array of patch IDs (same shape as signal).
    type_raster : np.ndarray
        2D array of forest type indices (same shape).
    std_prop_by_type : dict[int, float]
        Mapping from type index -> std_proportion_of_range for that type.
        For a given type t, sigma_t = std_prop_by_type[t] * (global signal range).
        Types not in the dict default to 0 (no patch noise).
    random_state : int or None
        Seed for reproducibility.

    Returns:
        noisy_signal, patch_noise_field, patch_offsets
        where patch_offsets is a dict: patch_id -> offset used.
    """
    rng = np.random.default_rng(random_state)
    signal = signal.astype(float)

    s_min = float(np.nanmin(signal))
    s_max = float(np.nanmax(signal))
    signal_range = s_max - s_min
    if signal_range <= 0:
        # Degenerate case: no variation
        return signal.copy(), np.zeros_like(signal, dtype=float), {}

    noisy = signal.copy()
    noise_field = np.zeros_like(signal, dtype=float)
    patch_offsets: dict[int, float] = {}

    unique_patches = np.unique(patch_raster)

    for pid in unique_patches:
        mask = (patch_raster == pid)
        if not np.any(mask):
            continue

        # Assume type is constant within patch; grab first
        type_val = int(type_raster[mask][0])
        std_prop = std_prop_by_type.get(type_val, 0.0)

        if std_prop <= 0:
            offset = 0.0
        else:
            sigma = std_prop * signal_range
            offset = rng.normal(loc=0.0, scale=sigma)

        noisy[mask] += offset
        noise_field[mask] = offset
        patch_offsets[int(pid)] = offset

    return noisy, noise_field, patch_offsets


# ---------------------------------------------------------------------
# 3) Low-pass 2D filter (uniform kernel)
# ---------------------------------------------------------------------

def apply_lowpass_filter(
    signal: np.ndarray,
    kernel_size: int = 3,
) -> np.ndarray:
    """
    Apply a simple 2D low-pass filter (uniform kernel) over the signal.
    """
    if kernel_size <= 1:
        return signal.copy()
    return uniform_filter(signal.astype(float), size=kernel_size, mode="nearest")


# ---------------------------------------------------------------------
# 4) Spatially correlated noise using gstools
# ---------------------------------------------------------------------

def add_spatially_correlated_noise(
    signal: np.ndarray,
    spatial_autocorr_distance: float,
    amplitude_proportion_of_range: float,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Add a spatially correlated Gaussian noise field using gstools.

    Steps:
        1. Generate a Gaussian random field with a given correlation length.
        2. Clip its tails at 1st and 99th percentiles.
        3. Rescale to [0,1].
        4. Shift to mean ~0 by subtracting 0.5 and scaling by the global
           signal range and amplitude_proportion_of_range.

    Returns:
        noisy_signal, spatial_noise_field
    """
    if gs is None:
        raise ImportError(
            "gstools is required for spatially correlated noise. "
            "Install with `conda install -c conda-forge gstools` or `pip install gstools`."
        )

    signal = signal.astype(float)
    s_min = float(np.nanmin(signal))
    s_max = float(np.nanmax(signal))
    signal_range = s_max - s_min
    if signal_range <= 0 or amplitude_proportion_of_range <= 0:
        return signal.copy(), np.zeros_like(signal, dtype=float)

    nrows, ncols = signal.shape

    model = gs.Gaussian(
        dim=2,
        var=1.0,
        len_scale=spatial_autocorr_distance,
    )
    srf = gs.SRF(model, seed=random_state)

    # grid: x ~ columns, y ~ rows
    x = np.arange(ncols)
    y = np.arange(nrows)
    xs, ys = np.meshgrid(x, y, indexing="xy")

    # gstools often returns flat; reshape to 2D
    field_flat = srf((xs.ravel(), ys.ravel()))
    field = field_flat.reshape(signal.shape)

    # Clip tails at 1st and 99th percentiles
    p1, p99 = np.percentile(field, [1, 99])
    clipped = np.clip(field, p1, p99)
    denom = (p99 - p1) if (p99 - p1) > 0 else 1.0
    normalized = (clipped - p1) / denom  # ~ [0,1]

    # Center around 0 and scale: ~ [-amp * range, +amp * range]
    noise = (normalized - 0.5) * 2.0 * amplitude_proportion_of_range * signal_range

    noisy_signal = signal + noise
    return noisy_signal, noise


# ---------------------------------------------------------------------
# Spatial uncertainty scaler for InterpreterField
# ---------------------------------------------------------------------

def spatial_uncertainty_scaler_field(
    shape: tuple[int, int],
    spatial_autocorr_distance: float,
    max_scaler: float,
    random_state: int | None = None,
) -> np.ndarray:
    """
    Generate a spatially correlated field of scalers in [0, max_scaler].

    Steps:
        1. Generate a Gaussian random field with given correlation length.
        2. Clip tails to [1%, 99%].
        3. Normalize to [0,1].
        4. Multiply by max_scaler.
    """
    if gs is None:
        raise ImportError(
            "gstools is required for spatial uncertainty field. "
            "Install with `conda install -c conda-forge gstools` or `pip install gstools`."
        )

    nrows, ncols = shape

    model = gs.Gaussian(dim=2, var=1.0, len_scale=spatial_autocorr_distance)
    srf = gs.SRF(model, seed=random_state)

    x = np.arange(ncols)
    y = np.arange(nrows)
    xs, ys = np.meshgrid(x, y, indexing="xy")

    field_flat = srf((xs.ravel(), ys.ravel()))
    field = field_flat.reshape(shape)

    p1, p99 = np.percentile(field, [1, 99])
    clipped = np.clip(field, p1, p99)
    denom = (p99 - p1) if (p99 - p1) > 0 else 1.0
    normalized = (clipped - p1) / denom  # [0,1]

    return normalized * max_scaler

# ---------------------------------------------------------------------
# Richards function for fitting probabilities
# ---------------------------------------------------------------------

def richards(
    x: np.ndarray,
    x0: float,
    b: float,
    nu: float,
) -> np.ndarray:
    """
    Generalized logistic (Richards) function.

    p(x) = (1 + exp(-b * (x - x0)))^(-nu)

    Parameters
    ----------
    x : np.ndarray
        Input values (typically in [0,1] or sensor units).
    x0 : float
        Location parameter (inflection-like point).
    b : float
        Slope / steepness (> 0).
    nu : float
        Shape / asymmetry parameter (> 0).
        nu = 1 reduces to standard logistic.

    Returns
    -------
    p : np.ndarray
        Output probabilities in (0,1).
    """
    x = np.asarray(x, dtype=float)

    # Clip exponent to avoid overflow
    z = np.clip(-b * (x - x0), -60.0, 60.0)

    # Richards curve
    return (1.0 + np.exp(z)) ** (-nu)


# ---------------------------------------------------------------------
# get metrics for binary classifications. 
# ---------------------------------------------------------------------


from typing import Any, Dict, Optional, List
import numpy as np
import pandas as pd

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



# ---------------------------------------------------------------------
#  
# ---------------------------------------------------------------------

    