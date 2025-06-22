#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIMS Framework - Artifact Saving Utilities for ML Experiments
============================================================

This module provides the `save_model_results` utility, responsible for serializing
 essential artifacts generated during a machine learning experiment within the AIMS (Adaptive and
Intelligent Management of Slicing) framework. The ResultsManager script saves final results.

Objective:
    - Ensure **reproducibility**, **comparability**, and **traceability** across all experiments.
    - Standardize the directory and file structure for easy downstream analysis and future reuse.

Output Directory Structure:
    /results/
    └── {model_name}/
        ├── X_{model_name}.csv           # Feature matrix (CSV)
        ├── y_{model_name}.csv           # Target labels (CSV)
        ├── y_{model_name}.npy           # Target labels (binary, NumPy format)
        ├── groups_{model_name}.csv      # Group IDs for CV splits (CSV)
        ├── groups_{model_name}.npy      # Group IDs (binary, NumPy format)
        └── class_weight_{model_name}.json # Class weights (JSON for imbalance handling)

Author: Tiago do Vale Saraiva
License: MIT
"""

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

def save_model_results(
    model_name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    class_weight: Dict[int, float],
    base_path: str = "../results",
) -> Path:
    """
    Save relevant machine learning artifacts for a given model training run.

    Artifacts include:
        - Features (X)
        - Labels (y)
        - Group IDs for cross-validation
        - Class weights for imbalance correction

    Parameters:
        model_name (str): Short identifier for the model (e.g., 'rf', 'cb').
        X (pd.DataFrame): Feature set used for training and evaluation.
        y (np.ndarray): Target labels.
        groups (np.ndarray): Group identifiers for grouped CV.
        class_weight (Dict[int, float]): Mapping of class labels to weights.
        base_path (str): Root directory for storing results (default: '../results').

    Returns:
        Path: Absolute path to the directory containing saved artifacts.
    """
    output_dir = (Path(base_path) / model_name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Results will be saved to: {output_dir}")
    print(f"✓ Saving artifact files (features, labels, groups, and class weights)...")

    # Save features (X) as CSV
    X.to_csv(output_dir / f"X_{model_name}.csv", index=False)

    # Save labels (y) as CSV and NPY
    np.savetxt(output_dir / f"y_{model_name}.csv", y, delimiter=",", fmt="%d")
    np.save(output_dir / f"y_{model_name}.npy", y)

    # Save groups as CSV and NPY
    np.savetxt(output_dir / f"groups_{model_name}.csv", groups, delimiter=",", fmt="%s")
    np.save(output_dir / f"groups_{model_name}.npy", groups)

    # Save class weights as JSON
    with open(output_dir / f"class_weight_{model_name}.json", "w") as f:
        json.dump(class_weight, f, indent=4)

    print("✓ Features, labels, groups, and class weights saved.")
    return output_dir

if __name__ == "__main__":
    """
    Simple self-test block to verify artifact saving functionality.
    Creates synthetic sample data and saves it under the 'demo_model' folder.
    """
    print("\n--- Running save_utils.py self-test ---")

    X_sample = pd.DataFrame({
        "lat_ms": [10.5, 20.2, 15.3],
        "pdr": [0.99, 0.95, 0.97],
        "throughput_kbps": [500, 600, 700],
    })
    y_sample = np.array([0, 1, 2])
    groups_sample = np.array(["g1", "g1", "g2"])
    class_weight_sample = {0: 1.2, 1: 0.8, 2: 1.5}

    save_model_results("demo_model", X_sample, y_sample, groups_sample, class_weight_sample)
    print("\nSelf-test completed successfully.")