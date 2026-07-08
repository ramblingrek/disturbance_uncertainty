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
    window_sample_fn,
    window_kwargs: Optional[Dict[str, Any]] = None,
    stack_indices: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    For every pair of sensors, compare pixel-level metric deltas against
    sampling-based metric deltas produced by a window_sample_A/B/C function.

    This is exploratory: a starting point for the meta-analysis in step 8,
    to be iterated on once real numbers are in hand rather than a finished
    API.

    Returns a long-format DataFrame: one row per (sensor_pair, metric) with
    pixel_delta and sampling_delta columns.
    """
    window_kwargs = dict(window_kwargs or {})

    pixel_df = pixel_level_metrics_for_collection(
        collection, sensor_names=sensor_names, stack_indices=stack_indices
    )
    pixel_by_sensor = (
        pixel_df.groupby("sensor_name")[["agreement", "precision", "recall", "f1", "iou"]]
        .mean()
    )

    sampling_by_sensor = {}
    for sensor_name in sensor_names:
        samples = window_sample_fn(
            sensor_name=sensor_name,
            stack_indices=stack_indices,
            **window_kwargs,
        )
        sampling_by_sensor[sensor_name] = pixel_level_metrics(
            samples["map_class"].to_numpy(), samples["ref_class"].to_numpy()
        )

    rows = []
    metric_names = ["agreement", "precision", "recall", "f1", "iou"]
    for sensor_a, sensor_b in combinations(sensor_names, 2):
        for metric in metric_names:
            pixel_delta = (
                pixel_by_sensor.loc[sensor_a, metric] - pixel_by_sensor.loc[sensor_b, metric]
            )
            sampling_delta = (
                sampling_by_sensor[sensor_a][metric] - sampling_by_sensor[sensor_b][metric]
            )
            rows.append({
                "sensor_a": sensor_a,
                "sensor_b": sensor_b,
                "metric": metric,
                "pixel_delta": pixel_delta,
                "sampling_delta": sampling_delta,
            })

    return pd.DataFrame(rows)
