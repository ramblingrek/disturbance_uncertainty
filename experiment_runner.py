#!/usr/bin/env python3
"""
experiment_runner.py

Run a full train/validate experiment from a JSON config file and save
all outputs (figures, CSVs, experiment log) to an organized folder.

Usage:
    python experiment_runner.py experiments/configs/baseline.json
    python experiment_runner.py experiments/configs/baseline.json --log-dir my_logs/
"""

import argparse
import copy
import json
import os
import sys

# Non-interactive backend — must be set before any other matplotlib import
# so plt.show() calls inside library methods are no-ops and don't open windows.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Resolve src/ relative to this script's location
PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJ_ROOT, "src"))

from landscape_stack_collection import LandscapeStackCollection
from interpreter_agent import InterpreterAgent
from disturbance_helper_functions import (
    generate_run_id,
    create_output_dir,
    export_with_metadata,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _abs(path: str) -> str:
    """Make a project-relative path absolute."""
    return os.path.join(PROJ_ROOT, path)


def _fix_path(block: dict, key: str) -> dict:
    """Return a deep copy of block with block[key] made absolute."""
    b = copy.deepcopy(block)
    if key in b and b[key]:
        b[key] = _abs(b[key])
    return b


def _print_section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_experiment(json_path: str, log_dir: str = "experiment_logs/") -> str:
    """
    Execute a full experiment from a JSON config file.

    Returns the run_id so callers can locate the output folder.
    """

    # ── Load config ──────────────────────────────────────────────────────
    _print_section(f"Loading config: {json_path}")
    with open(json_path) as f:
        cfg = json.load(f)

    # ── Output directory ─────────────────────────────────────────────────
    run_id = generate_run_id()
    dirs = create_output_dir(run_id, experiment_name=cfg["experiment_name"])
    print(f"run_id      : {run_id}")
    print(f"output root : {dirs['root']}")

    # ── Fix paths (JSON uses project-root-relative paths) ────────────────
    stack_cfg     = _fix_path(cfg["landscape_stack_config"],        "tif_path")
    train_exp_cfg = _fix_path(cfg["training_experiment_config"],    "patch_dir")
    val_exp_cfg   = _fix_path(cfg["validation_experiment_config"],  "patch_dir")

    # ── Interpreter agents ───────────────────────────────────────────────
    model_interp = InterpreterAgent(**cfg["training_interp_config"])
    val_interp   = InterpreterAgent(**cfg["validation_interp_config"])

    # ── Training collection ──────────────────────────────────────────────
    _print_section("Building training collection")
    train_collection = LandscapeStackCollection.from_config(
        stack_template_cfg=stack_cfg,
        experiment_cfg=train_exp_cfg,
    )
    print(f"  {len(train_collection.stacks)} training stacks built")

    mc = cfg["training_model_config"]
    _print_section("Fitting continuous model (Richards)")
    trained_models = train_collection.buildModel(
        n_strata=mc["n_strata"],
        samples_per_stratum=mc["samples_per_stratum"],
        sampling_seed=mc["sampling_seed"],
        interpreter_agent=model_interp,
        save_dir=dirs["figures"],
    )
    plt.close("all")
    print(f"  Saved model fit figure → {dirs['figures']}/")

    bc = cfg["binary_classifier_config"]
    _print_section("Fitting binary classifier")
    train_collection.buildBinaryClassifier(
        n_strata=bc["n_strata"],
        samples_per_stratum=bc["samples_per_stratum"],
        sampling_seed=bc["sampling_seed"],
        fp_to_fn_ratio=bc["fp_to_fn_ratio"],
        interpreter_agent=model_interp,
    )
    train_collection.plotBinaryCurves(bc["sensor_name"], save_dir=dirs["figures"])
    plt.close("all")
    print(f"  threshold = {train_collection.binary_models[bc['sensor_name']]['threshold']:.4f}")
    print(f"  Saved threshold curves figure → {dirs['figures']}/")

    # ── Validation collection ────────────────────────────────────────────
    used_tifs = [s.base_landscape.tif_path for s in train_collection.stacks]

    _print_section("Building validation collection")
    val_collection = LandscapeStackCollection.from_config(
        stack_template_cfg=stack_cfg,
        experiment_cfg=val_exp_cfg,
        exclude_tif_paths=used_tifs,
    )
    print(f"  {len(val_collection.stacks)} validation stacks built")

    _print_section("Applying models to validation collection")
    val_collection.loadModel(trained_models)
    val_collection.applyModel()
    print("  Continuous model applied")

    ipm = cfg["training_interp_prob_model_config"]
    val_collection.buildInterpreterProbModel(
        sensor_name=ipm["sensor_name"],
        n_strata=ipm["n_strata"],
        samples_per_stratum=ipm["samples_per_stratum"],
        sampling_seed=ipm["sampling_seed"],
        interpreter_agent=val_interp,
    )
    val_collection.applyInterpreterProbModel(
        sensor_name=ipm["sensor_name"],
        store_attr_name="interpreter_calibrated_prob_by_sensor",
    )
    plt.close("all")
    print("  Interpreter probability model applied")

    val_collection.loadBinaryModel(train_collection.binary_models)
    val_collection.applyBinaryClassifier()
    print("  Binary classifier applied")

    # ── Evaluate ─────────────────────────────────────────────────────────
    ec = cfg["evaluate_config"]
    _print_section("Evaluating validation collection")
    out = val_collection.evaluate_validation_collection(
        sensor_name=ec["sensor_name"],
        prob_attr=ec["prob_attr"],
        step=ec["step"],
        n_samples_map0=ec["n_samples_map0"],
        n_samples_map1=ec["n_samples_map1"],
        interpreter_agent=val_interp,
        xlim_prop=tuple(ec["xlim_prop"]) if ec.get("xlim_prop") else None,
        title_prefix=ec.get("title_prefix", cfg["experiment_name"]),
        save_dir=dirs["root"],
    )
    plt.close("all")

    cont_df = out["continuous_df"]
    bin_df  = out["binary_df"]

    print(f"\n  Continuous — median abs error : {cont_df['abs_err'].median():.4f}")
    print(f"  Continuous — mean abs error   : {cont_df['abs_err'].mean():.4f}")
    print(f"  Binary     — median abs error : {bin_df['abs_err'].median():.4f}")
    print(f"  Binary     — mean abs error   : {bin_df['abs_err'].mean():.4f}")
    print(f"  Saved summary figures + CSVs → {dirs['root']}/")

    # ── Per-stack figures ─────────────────────────────────────────────────
    spc = cfg["stack_plots_config"]
    n_stacks = len(val_collection.stacks)
    _print_section(f"Saving per-stack figures ({n_stacks} stacks)")
    val_collection.save_all_stack_plots(
        save_dir=dirs["per_stack"],
        sensor_name=ec["sensor_name"],
        prob_source_attr=ec["prob_attr"],
        pixel_area=spc["pixel_area"],
        step=spc["step"],
        bins=spc["bins"],
        x_prop_range=tuple(spc["x_prop_range"]) if spc.get("x_prop_range") else None,
        show_diagnostics=spc["show_diagnostics"],
    )
    n_saved = len(os.listdir(dirs["per_stack"]))
    print(f"  {n_saved} files saved → {dirs['per_stack']}/")

    # ── Log the run ───────────────────────────────────────────────────────
    _print_section("Logging run metadata")
    experiment_config_to_log = {
        "landscape_stack_config":       stack_cfg,
        "training_experiment_config":   train_exp_cfg,
        "training_interp_config":       cfg["training_interp_config"],
        "validation_experiment_config": val_exp_cfg,
        "validation_interp_config":     cfg["validation_interp_config"],
    }
    export_with_metadata(
        experiment_config_to_log,
        csv_path=log_dir,
        run_id=run_id,
        output_dir=dirs["root"],
        notes=cfg.get("notes", ""),
    )
    print(f"  Logged to {log_dir}master_experiment_logger.csv")

    _print_section("Complete")
    print(f"  run_id  : {run_id}")
    print(f"  outputs : {dirs['root']}\n")

    return run_id


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a disturbance uncertainty experiment from a JSON config file."
    )
    parser.add_argument(
        "config",
        help="Path to the experiment JSON config file (e.g. experiments/configs/baseline.json)",
    )
    parser.add_argument(
        "--log-dir",
        default="experiment_logs/",
        help="Directory for the experiment log CSVs (default: experiment_logs/)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"ERROR: config file not found: {args.config}")
        sys.exit(1)

    run_experiment(args.config, log_dir=args.log_dir)
