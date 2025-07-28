#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vehicular Network QoS Impact Labeling Module for AIMS Framework
===============================================================

This module implements the impact labeling logic used in the AIMS (Adaptive and Intelligent Management of
Slicing) framework. It assigns discrete impact levels to vehicular network QoS data, enabling machine learning
models to classify and optimize network slicing policies for Intelligent Transportation Systems (ITS).

Key Functionalities:
    - Maps continuous QoS metrics (latency, packet loss, throughput) to discrete impact levels (0-3).
    - Uses application-specific thresholds inspired by standards (3GPP, ETSI, 5GAA), but empirically calibrated on
    the AIMS dataset.
    - Supports weighted averaging of metric scores for realistic impact assessment.
    - Provides ML-ready outputs including features, target labels, groups, and class weights.

Author: Tiago do Vale Saraiva
License: MIT
Version: 1.0.0
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

# Define which functions are exported when using 'from impact_labeling import *'
__all__ = [
    "HARD_THRESH",
    "WEIGHTS",
    "score_metric",
    "label_weighted_average",
]

# --- Impact Classification Thresholds ---

# HARD_THRESH defines performance thresholds for each QoS metric by application type.
# These threshold ranges were inspired by values from well-known standards for vehicular communication,
# such as 3GPP TS 22.185 Rel-17, ETSI TR 102 962, and 5GAA guidelines.
# However, the exact values have been adapted and calibrated based on the characteristics
# and distribution of the AIMS dataset, applying application-specific weights
# to ensure a more realistic and representative labeling for impact classification.
#
# Threshold Format: [Adequate, Warning, Severe]
# Interpretation:
# - For latency and loss, lower values are better.
# - For throughput, higher values are better.

HARD_THRESH: Dict[str, Dict[str, List[float]]] = {
    # Safety applications
    "s": {
        "lat_ms": [100, 200, 500],       # Latency in milliseconds
        "loss": [0.01, 0.05, 0.10],      # Packet Loss Rate (1 - PDR)
        "thru_kbps": [450, 400, 300],    # Throughput in kbps
    },
    # Traffic efficiency applications
    "e": {
        "lat_ms": [300, 700, 1500],
        "loss": [0.05, 0.10, 0.20],
        "thru_kbps": [800, 600, 400],
    },
    # Entertainment/infotainment
    "e2": {
        "lat_ms": [100, 250, 1000],
        "loss": [0.02, 0.05, 0.10],
        "thru_kbps": [8000, 3000, 1000],
    },
    # Generic applications
    "g": {
        "lat_ms": [1000, 3000, 5000],
        "loss": [0.10, 0.30, 0.50],
        "thru_kbps": [5000, 2000, 500],
    },
}

# --- Metric Weights per Application Type ---

# WEIGHTS defines the relative importance of each QoS metric per application type.
# These weights were empirically fine-tuned based on the AIMS dataset distribution
# to ensure that the impact labeling better reflects realistic network performance perceptions.
# Although inspired by application priorities from the literature, the final weights were adjusted
# to improve balance and consistency across the dataset.

# For each application type, the weights sum to 1.0.
WEIGHTS: Dict[str, Dict[str, float]] = {
    "s": dict(lat=0.5, loss=0.3, thr=0.2),  # Safety: Latency is prioritized.
    "e": dict(lat=0.3, loss=0.4, thr=0.3),  # Efficiency: Balanced between latency, loss, and throughput.
    "e2": dict(lat=0.2, loss=0.3, thr=0.5),  # Entertainment: Throughput has higher relevance.
    "g": dict(lat=0.3, loss=0.3, thr=0.4),  # Generic: Slightly higher weight on throughput.
}

def score_metric(value: float, limits: List[float], *, lower_is_better: bool = True) -> int:
    """
    Map a QoS metric value to a discrete quality score (0=best, 3=worst).

    Parameters:
        value (float): Observed QoS value (e.g., latency in ms).
        limits (List[float]): Thresholds: [Adequate, Warning, Severe].
        lower_is_better (bool): True for latency/loss, False for throughput.

    Returns:
        int: Discrete score (0 to 3). NaN values return 3 (Critical).
    """
    if np.isnan(value):
        return 3  # Treat missing values as the worst-case scenario.

    # Logic for metrics where lower values are better (e.g., latency, loss).
    if lower_is_better:
        if value <= limits[0]: return 0  # Adequate
        if value <= limits[1]: return 1  # Warning
        if value <= limits[2]: return 2  # Severe
    # Logic for metrics where higher values are better (e.g., throughput).
    else:
        if value >= limits[0]: return 0  # Adequate
        if value >= limits[1]: return 1  # Warning
        if value >= limits[2]: return 2  # Severe

    return 3  # Critical (value is outside all defined thresholds).


def label_weighted_average(
    df: pd.DataFrame,
    *,
    hard: Dict[str, Dict[str, List[float]]] = HARD_THRESH,
    weights: Dict[str, Dict[str, float]] = WEIGHTS,
    out_col: str = "impact_label",
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, Dict[int, float]]:
    """
       Generate discrete impact labels for vehicular QoS data using a weighted scoring approach.

       For each sample, this function:
           1. Maps each QoS metric (latency, loss, throughput) to a discrete score (0-3).
           2. Applies application-specific weights to calculate a weighted average impact score.
           3. Rounds the final score to obtain an integer label from 0 to 3.

       Parameters:
           df (pd.DataFrame): Input DataFrame with columns: ['app_id', 'lat_ms', 'pdr', 'throughput_kbps'].
           hard (Dict): Threshold definitions (default: HARD_THRESH).
           weights (Dict): Metric weighting per app class (default: WEIGHTS).
           out_col (str): Name of the output label column (default: "impact_label").

       Returns:
           Tuple:
               - Labeled DataFrame (with impact labels)
               - Features DataFrame (X)
               - Target array (y)
               - Group array (groups)
               - Class weights array (for imbalanced learning)
               - Class weight dictionary (for external use)
       """
    labels: List[int] = []
    # Iterate over each row to calculate its specific impact label.
    for _, row in df.iterrows():
        # Identify the application class, defaulting to 'g' (generic) if not found.
        cls = str(row.get("app_id", "g")).lower()
        ref = hard.get(cls, hard["g"])
        w = weights.get(cls, weights["g"])

        # Validate that weights for the current class sum to 1.0.
        if not np.isclose(sum(w.values()), 1.0):
            raise ValueError(f"Weights for app_id '{cls}' must sum to 1.0. Got: {w}")

        # Score each individual metric based on its thresholds.
        lat_s = score_metric(row["lat_ms"], ref["lat_ms"], lower_is_better=True)
        # Packet loss = (1 - Packet Delivery Ratio).
        loss_s = score_metric(1.0 - row["pdr"], ref["loss"], lower_is_better=True)
        thr_s = score_metric(row["throughput_kbps"], ref["thru_kbps"], lower_is_better=False)

        # Calculate the weighted sum of the individual scores.
        impact_score = w["lat"] * lat_s + w["loss"] * loss_s + w["thr"] * thr_s

        # Round the score to the nearest integer to get the final discrete label.
        label = int(round(impact_score))
        labels.append(label)

    # Create a copy of the DataFrame to avoid modifying the original.
    out_df = df.copy()
    out_df[out_col] = labels

    # Prepare data structures for direct use in machine learning pipelines.
    X = out_df.drop(columns=[out_col])
    y = out_df[out_col].values
    groups = out_df.get("group_id", pd.Series([0] * len(out_df))).values

    # Calculate class weights to help models handle imbalanced datasets.
    unique_labels = np.unique(y)
    class_weights_array = compute_class_weight(class_weight="balanced", classes=unique_labels, y=y)
    class_weight_dict = {int(k): float(v) for k, v in zip(unique_labels, class_weights_array)}

    return out_df, X, y, groups, class_weights_array, class_weight_dict


if __name__ == "__main__":
    """
    Self-test block executed when the script is run directly.
    Demonstrates the usage of the label_weighted_average function with a minimal
    example DataFrame and prints the output.
    """
    print("--- Running impact_labeling.py self-test ---")

    # Create a minimal example DataFrame for demonstration.
    example_df = pd.DataFrame({
        "app_id": ["s", "e", "e2", "g"],
        "lat_ms": [80, 600, 200, 2000],          # Latency values
        "pdr": [0.99, 0.98, 0.94, 0.93],         # Packet Delivery Ratio
        "throughput_kbps": [600, 850, 9000, 5100], # Throughput
    })

    # Apply the labeling function.
    labeled_df, *_ = label_weighted_average(example_df)

    print("\nExample DataFrame with Impact Labels:")
    print(labeled_df)
    print("\n--- Self-test complete ---")

