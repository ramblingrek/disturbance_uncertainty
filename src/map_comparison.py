"""
Map comparison benchmark (binary-only first pass).

Full-raster pixel-level scoring for the interpreter sub-project's map
comparison benchmark, plus a meta-analysis comparing pixel-level metric
deltas between sensor maps against sampling-based (window_sample_A/B/C)
metric deltas.

Unlike the stratified-sampling apparatus in landscape_stack_collection.py,
every function here operates on full rasters (or pairs of them), never on
sampled subsets. Reference/"ref_class" follows the same convention used
throughout landscape_stack_collection.py: derived from the interpreter field
thresholded at 50%, not the raw truth mask directly.
"""

from itertools import combinations
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def perfect_reference_mask(stack) -> np.ndarray:
    """
    Thresholded perfect-interpreter field for a stack, used as the pixel-level
    reference ("ref_class") throughout this module.
    """
    return (stack.interpreter_field.interpreter_field >= 50.0).astype(int)


def pixel_confusion_matrix(map_field: np.ndarray, ref_field: np.ndarray) -> Dict[str, Any]:
    """
    Full-raster 2x2 confusion matrix between a binary map and a binary
    reference, where:
      - rows = map_class (i = 0,1)
      - cols = ref_class (j = 0,1)

    Same n_ij/n_i/n_total shape as
    LandscapeStackCollection.binary_confusion_from_samples, computed directly
    from two full arrays rather than a sampled DataFrame.
    """
    map_field = np.asarray(map_field, dtype=int)
    ref_field = np.asarray(ref_field, dtype=int)
    if map_field.shape != ref_field.shape:
        raise ValueError(
            f"map_field shape {map_field.shape} != ref_field shape {ref_field.shape}"
        )

    n_00 = int(np.sum((map_field == 0) & (ref_field == 0)))
    n_01 = int(np.sum((map_field == 0) & (ref_field == 1)))
    n_10 = int(np.sum((map_field == 1) & (ref_field == 0)))
    n_11 = int(np.sum((map_field == 1) & (ref_field == 1)))

    n_ij = np.array([[n_00, n_01],
                      [n_10, n_11]], dtype=int)
    n_i = n_ij.sum(axis=1)
    n_total = int(n_ij.sum())

    return {"n_ij": n_ij, "n_i": n_i, "n_total": n_total}


def pixel_level_metrics(map_field: np.ndarray, ref_field: np.ndarray) -> Dict[str, float]:
    """
    Overall agreement, precision, recall, F1, and IoU for class 1
    ("disturbed") between a binary map and a binary reference, computed over
    the full raster.
    """
    confusion = pixel_confusion_matrix(map_field, ref_field)
    n_00, n_01 = confusion["n_ij"][0]
    n_10, n_11 = confusion["n_ij"][1]
    n_total = confusion["n_total"]

    agreement = (n_00 + n_11) / n_total if n_total > 0 else np.nan
    precision = n_11 / (n_10 + n_11) if (n_10 + n_11) > 0 else np.nan
    recall = n_11 / (n_01 + n_11) if (n_01 + n_11) > 0 else np.nan
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision and recall and (precision + recall) > 0
        else np.nan
    )
    iou = n_11 / (n_11 + n_10 + n_01) if (n_11 + n_10 + n_01) > 0 else np.nan

    return {
        "agreement": agreement,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "n_total": n_total,
    }


def pixel_level_metrics_for_stack(stack, sensor_name: str) -> Dict[str, float]:
    """
    Convenience wrapper: scores stack.binary_class_by_sensor[sensor_name]
    against perfect_reference_mask(stack).
    """
    if sensor_name not in stack.binary_class_by_sensor:
        raise KeyError(
            f"Sensor {sensor_name!r} not in binary_class_by_sensor for stack "
            f"{stack.stack_id}. Call applyBinaryClassifier() first."
        )
    map_field = stack.binary_class_by_sensor[sensor_name]
    ref_field = perfect_reference_mask(stack)
    return pixel_level_metrics(map_field, ref_field)


def pixel_level_metrics_for_collection(
    collection,
    sensor_names: Optional[List[str]] = None,
    stack_indices: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    One row per (stack, sensor) with all pixel_level_metrics fields.
    """
    if sensor_names is None:
        sensor_names = list(collection.stacks[0].binary_class_by_sensor.keys())

    idx_used = range(len(collection.stacks)) if stack_indices is None else stack_indices

    rows = []
    for si in idx_used:
        stack = collection.stacks[si]
        for sensor_name in sensor_names:
            if sensor_name not in stack.binary_class_by_sensor:
                continue
            metrics = pixel_level_metrics_for_stack(stack, sensor_name)
            rows.append({
                "stack_index": si,
                "stack_id": stack.stack_id,
                "sensor_name": sensor_name,
                **metrics,
            })
    return pd.DataFrame(rows)


def pixel_agreement_between_maps(map_a: np.ndarray, map_b: np.ndarray) -> Dict[str, float]:
    """
    Same confusion/metrics machinery as pixel_level_metrics, but map-vs-map
    (no truth/reference involved) — used for step 8's map-pair comparisons.
    """
    return pixel_level_metrics(map_a, map_b)


def map_comparison_meta_analysis(
    collection,
    sensor_names: List[str],
    window_fns: Dict[str, Any],
    window_kwargs: Optional[Dict[str, Any]] = None,
    stack_indices: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    For every pair of sensors, compare pixel-level metric deltas against
    sampling-based metric deltas for each provided window sampling function.

    window_fns : dict mapping method name to callable, e.g.
        {'A': collection.window_sample_A,
         'B': collection.window_sample_B,
         'C': collection.window_sample_C,
         'D': collection.window_sample_D}

    For A/B/C (output columns 'map_class'/'ref_class'): computes deltas for
    agreement/precision/recall/f1/iou — the same metrics as pixel_delta.
    For D (output columns 'prop_map'/'prop_interp'): computes Pearson r
    between prop_map and prop_interp and reports its delta across sensor pairs;
    pixel_delta is NaN for these rows.

    Returns wide-format DataFrame: one row per (sensor_pair, metric) with
    columns pixel_delta and sampling_delta_{method} for each method.
    Entries that don't apply to a given metric/method combination are NaN.
    """
    from scipy.stats import pearsonr as _pearsonr

    window_kwargs = dict(window_kwargs or {})

    pixel_df = pixel_level_metrics_for_collection(
        collection, sensor_names=sensor_names, stack_indices=stack_indices
    )
    pixel_mean = (
        pixel_df.groupby("sensor_name")[["agreement", "precision", "recall", "f1", "iou"]]
        .mean()
    )

    # Compute per-method, per-sensor sampling metrics
    abc_results: Dict[str, Dict[str, Dict[str, float]]] = {}  # method -> sensor -> metrics
    d_results: Dict[str, Dict[str, float]] = {}               # method -> sensor -> pearson_r

    for method_name, fn in window_fns.items():
        per_sensor: Dict[str, Any] = {}
        for sensor_name in sensor_names:
            samples = fn(
                sensor_name=sensor_name,
                stack_indices=stack_indices,
                **window_kwargs,
            )
            if "map_class" in samples.columns:
                per_sensor[sensor_name] = pixel_level_metrics(
                    samples["map_class"].to_numpy(),
                    samples["ref_class"].to_numpy(),
                )
            else:
                r, _ = _pearsonr(samples["prop_map"], samples["prop_interp"])
                per_sensor[sensor_name] = {"pearson_r": float(r)}

        if all("pearson_r" in v for v in per_sensor.values()):
            d_results[method_name] = {s: v["pearson_r"] for s, v in per_sensor.items()}
        else:
            abc_results[method_name] = per_sensor

    all_methods = list(window_fns.keys())
    abc_metrics = ["agreement", "precision", "recall", "f1", "iou"]

    rows = []
    for sensor_a, sensor_b in combinations(sensor_names, 2):
        # Rows for A/B/C-style metrics
        for metric in abc_metrics:
            pixel_delta = pixel_mean.loc[sensor_a, metric] - pixel_mean.loc[sensor_b, metric]
            row: Dict[str, Any] = {
                "sensor_a": sensor_a,
                "sensor_b": sensor_b,
                "metric": metric,
                "pixel_delta": pixel_delta,
            }
            for method_name in all_methods:
                if method_name in abc_results:
                    row[f"sampling_delta_{method_name}"] = (
                        abc_results[method_name][sensor_a][metric]
                        - abc_results[method_name][sensor_b][metric]
                    )
                else:
                    row[f"sampling_delta_{method_name}"] = np.nan
            rows.append(row)

        # Rows for D-style metric (Pearson r)
        for method_name, sensor_r in d_results.items():
            row = {
                "sensor_a": sensor_a,
                "sensor_b": sensor_b,
                "metric": "pearson_r",
                "pixel_delta": np.nan,
            }
            for m in all_methods:
                if m == method_name:
                    row[f"sampling_delta_{m}"] = sensor_r[sensor_a] - sensor_r[sensor_b]
                else:
                    row[f"sampling_delta_{m}"] = np.nan
            rows.append(row)

    return pd.DataFrame(rows)
