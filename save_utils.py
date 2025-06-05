#!/usr/bin/env python3
"""
AIMS Framework - Model Results Saving Utilities
==============================================

This module provides utilities for saving machine learning model artifacts
in a consistent and organized manner. It handles the serialization of
features, labels, groups, and class weights in multiple formats for
maximum compatibility and reproducibility.

The module ensures that all model experiments have their results saved
in a standardized directory structure, making it easy to compare models
and reproduce results.

Functions:
    save_model_results: Main function to save all model artifacts

Output Structure:
    artifacts/
    └── {model_name}/
        ├── X_{model_name}.csv           # Feature matrix
        ├── y_{model_name}.csv           # Labels (text format)
        ├── y_{model_name}.npy           # Labels (binary format)
        ├── groups_{model_name}.csv      # Group IDs (text format)
        ├── groups_{model_name}.npy      # Group IDs (binary format)
        └── class_weight_{model_name}.json # Class weights

Author: Tiago do Vale Saraiva
License: MIT
"""

import numpy as np
import json
from pathlib import Path
import pandas as pd
from typing import Dict, Union, Optional


def save_model_results(
        model_name: str,
        X: pd.DataFrame,
        y: np.ndarray,
        groups: np.ndarray,
        class_weight: Dict[int, float],
        base_path: str = "drive/MyDrive/Colab Notebooks/vtm2025/artifacts"
) -> Path:
    """
    Save machine learning model training artifacts in multiple formats.

    This function creates a standardized directory structure for each model
    and saves all relevant data needed for model evaluation and reproduction.
    Data is saved in both human-readable (CSV, JSON) and efficient binary
    (NPY) formats.

    Parameters
    ----------
    model_name : str
        Identifier for the model (e.g., 'cb' for CatBoost, 'rf' for Random Forest).
        This will be used to create the output directory and file prefixes.

    X : pd.DataFrame
        Feature matrix containing all input features for the model.
        Should include both numerical and categorical features.

    y : np.ndarray
        Target labels array. For impact classification, these are integers 0-3
        representing: 0=Low, 1=Minor, 2=Major, 3=Critical impact.

    groups : np.ndarray
        Group identifiers for GroupKFold cross-validation. Typically contains
        scenario+timeblock combinations to prevent temporal leakage.

    class_weight : Dict[int, float]
        Dictionary mapping class labels to their weights. Used to handle
        class imbalance during training.
        Example: {0: 1.75, 1: 1.33, 2: 0.49, 3: 1.49}

    base_path : str, optional
        Root directory for saving artifacts. Defaults to the standard
        Google Colab path structure.

    Returns
    -------
    output_dir : Path
        Absolute path to the directory where all files were saved.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>>
    >>> # Prepare sample data
    >>> X = pd.DataFrame({
    ...     'lat_ms': [10.5, 20.3, 15.7],
    ...     'pdr': [0.98, 0.95, 0.99],
    ...     'throughput_kbps': [450, 380, 520]
    ... })
    >>> y = np.array([0, 1, 0])  # Impact labels
    >>> groups = np.array(['scenario1_0', 'scenario1_0', 'scenario1_1'])
    >>> class_weight = {0: 1.2, 1: 0.8}
    >>>
    >>> # Save results
    >>> output_path = save_model_results('rf', X, y, groups, class_weight)
    >>> print(f"Results saved to: {output_path}")

    Notes
    -----
    - CSV files are human-readable but may lose precision for floating-point numbers
    - NPY files preserve exact numerical precision and data types
    - The function creates directories as needed if they don't exist
    - All paths are resolved to absolute paths for clarity
    """

    # Create output directory with resolved absolute path
    output_dir = Path(base_path) / model_name
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save feature matrix as CSV
    # CSV format allows easy inspection and loading in any tool
    X_filename = f'X_{model_name}.csv'
    X.to_csv(output_dir / X_filename, index=False)
    print(f"✓ Features saved: {output_dir / X_filename}")

    # Save labels in both CSV and NPY formats
    # CSV for human readability, NPY for exact numerical preservation
    y_csv_filename = f'y_{model_name}.csv'
    y_npy_filename = f'y_{model_name}.npy'
    np.savetxt(output_dir / y_csv_filename, y, delimiter=',', fmt='%d')  # Integer format
    np.save(output_dir / y_npy_filename, y)
    print(f"✓ Labels saved: {output_dir / y_csv_filename} (text) and {y_npy_filename} (binary)")

    # Save group identifiers for cross-validation
    # Groups are typically strings (scenario_timeblock), so we use '%s' format
    groups_csv_filename = f'groups_{model_name}.csv'
    groups_npy_filename = f'groups_{model_name}.npy'
    np.savetxt(output_dir / groups_csv_filename, groups, delimiter=',', fmt='%s')
    np.save(output_dir / groups_npy_filename, groups)
    print(f"✓ Groups saved: {output_dir / groups_csv_filename} (text) and {groups_npy_filename} (binary)")

    # Save class weights as JSON for easy reading and modification
    # JSON format preserves the dictionary structure and is human-editable
    class_weight_filename = f'class_weight_{model_name}.json'
    with open(output_dir / class_weight_filename, 'w') as f:
        json.dump(class_weight, f, indent=4)  # Pretty-print with indentation
    print(f"✓ Class weights saved: {output_dir / class_weight_filename}")

    print(f"\n📁 All artifacts saved to: {output_dir}")

    return output_dir


# Demonstration and testing
if __name__ == "__main__":
    """
    Demonstration of the save_model_results function with sample data.
    This can be used to verify the function works correctly in your environment.
    """
    print("AIMS Framework - Save Utils Demo")
    print("=" * 50)

    # Create realistic sample data
    X_example = pd.DataFrame({
        'lat_ms': [12.5, 25.3, 18.7, 45.2],
        'pdr': [0.98, 0.95, 0.99, 0.92],
        'throughput_kbps': [450, 380, 520, 250],
        'n_cars': [10, 15, 12, 20],
        'categoria': ['s', 'e', 's', 'g']  # Safety, Efficiency, Safety, Generic
    })

    y_example = np.array([0, 1, 0, 2])  # Low, Minor, Low, Major impact

    groups_example = np.array([
        'highway_scenario_0',
        'highway_scenario_0',
        'urban_scenario_1',
        'urban_scenario_1'
    ])

    class_weight_example = {
        0: 1.2,  # Low impact (slightly overweighted)
        1: 1.5,  # Minor impact (overweighted)
        2: 0.8,  # Major impact (underweighted due to frequency)
        3: 1.3  # Critical impact (overweighted)
    }

    # Test saving for different models
    print("\nTesting CatBoost artifact saving...")
    cb_path = save_model_results('cb_demo', X_example, y_example,
                                 groups_example, class_weight_example)

    print("\nTesting Random Forest artifact saving...")
    rf_path = save_model_results('rf_demo', X_example, y_example,
                                 groups_example, class_weight_example)

    print("\n✅ Demo completed successfully!")
    print(f"Check the output directories:")
    print(f"  - {cb_path}")
    print(f"  - {rf_path}")