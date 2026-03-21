#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIMS Framework - Federated Learning Data Module
================================================

Handles data loading, preprocessing (reusing AIMS pipeline), feature preparation,
StandardScaler normalization, and IID/Non-IID partitioning for FL clients.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Add parent directory to path so we can import AIMS modules
_CODE_DIR = Path(__file__).resolve().parent.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import preprocess_dataset as pp
from impact_labeling import label_weighted_average

from .fl_config import NON_IID_ALLOCATION


def load_and_prepare(
    csv_path: Path,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, float], List[str]]:
    """
    Load the AIMS dataset, apply preprocessing and impact labeling,
    then prepare numeric features for neural network training.

    Uses the same preprocessing pipeline as the traditional AIMS models
    (prepare_dataset + label_weighted_average) to ensure consistency.

    Parameters
    ----------
    csv_path : Path
        Path to the AIMS CSV dataset.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    X : np.ndarray
        Scaled feature matrix (float32).
    y : np.ndarray
        Impact labels (0-3).
    class_weight_dict : dict
        Class weights for imbalanced learning.
    feature_names : list[str]
        Names of the features in X.
    """
    # Step 1: Load raw data
    raw_df = pd.read_csv(csv_path)
    print(f"  Loaded dataset: {len(raw_df)} samples, {len(raw_df.columns)} columns")

    # Step 2: Apply AIMS preprocessing (same as RF/CatBoost/TabNet)
    processed_df = pp.prepare_dataset(raw_df)

    # Step 3: Apply impact labeling (4 levels: 0-3)
    labeled_df, X_df, y, groups, class_weights_array, class_weight_dict = (
        label_weighted_average(processed_df)
    )
    print(f"  Impact label distribution: {pd.Series(y).value_counts().sort_index().to_dict()}")

    # Step 4: Select and prepare features for neural networks
    # Drop non-predictive columns that are identifiers or metadata
    drop_cols = [
        "group_id", "time_block", "source_file", "svc_profile",
        "throughput_bps", "impact_label",
    ]
    X_df = X_df.drop(columns=[c for c in drop_cols if c in X_df.columns], errors="ignore")

    # One-hot encode categorical columns
    cat_cols = [c for c in ["app_id", "approach", "category"] if c in X_df.columns]
    if cat_cols:
        X_df = pd.get_dummies(X_df, columns=cat_cols, drop_first=False, dtype=float)

    # Ensure all columns are numeric
    X_df = X_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    feature_names = list(X_df.columns)

    # Step 5: Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X_df.values).astype("float32")
    y = y.astype(int)

    print(f"  Final feature matrix: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y, class_weight_dict, feature_names


def partition_iid(
    X: np.ndarray,
    y: np.ndarray,
    num_clients: int,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Partition data into IID (random disjoint) subsets for each client.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Labels.
    num_clients : int
        Number of FL clients.
    seed : int
        Random seed.

    Returns
    -------
    list of (X_client, y_client) tuples.
    """
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(X))
    chunks = np.array_split(indices, num_clients)

    partitions = []
    for i, idx in enumerate(chunks):
        partitions.append((X[idx], y[idx]))
        dist = pd.Series(y[idx]).value_counts().sort_index().to_dict()
        print(f"    Client {i} (IID): {len(idx)} samples, dist={dist}")

    return partitions


def partition_non_iid(
    X: np.ndarray,
    y: np.ndarray,
    num_clients: int,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Partition data into Non-IID subsets based on class distribution.

    Uses the NON_IID_ALLOCATION matrix to assign different proportions
    of each class to each client, reflecting realistic RSU deployments
    where different road segments observe different traffic impact levels.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Labels.
    num_clients : int
        Number of FL clients.
    seed : int
        Random seed.

    Returns
    -------
    list of (X_client, y_client) tuples.
    """
    rng = np.random.RandomState(seed)
    allocation = NON_IID_ALLOCATION[:num_clients]
    unique_classes = sorted(np.unique(y))

    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    for cls_idx, cls in enumerate(unique_classes):
        cls_mask = np.where(y == cls)[0]
        cls_indices = rng.permutation(cls_mask)
        n = len(cls_indices)

        if cls_idx >= allocation.shape[1]:
            # If more classes than allocation columns, distribute equally
            chunks = np.array_split(cls_indices, num_clients)
            for c in range(num_clients):
                client_indices[c].extend(chunks[c].tolist())
            continue

        # Assign proportionally based on allocation matrix
        start = 0
        for c in range(num_clients):
            frac = allocation[c, cls_idx]
            count = int(round(frac * n))
            if c == num_clients - 1:
                # Last client gets the remainder
                end = n
            else:
                end = min(start + count, n)
            client_indices[c].extend(cls_indices[start:end].tolist())
            start = end

    partitions = []
    for i in range(num_clients):
        idx = np.array(client_indices[i])
        rng.shuffle(idx)
        partitions.append((X[idx], y[idx]))
        dist = pd.Series(y[idx]).value_counts().sort_index().to_dict()
        print(f"    Client {i} (NonIID): {len(idx)} samples, dist={dist}")

    return partitions


def split_client_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split a client's data into train and validation sets (stratified).

    Returns (X_train, X_val, y_train, y_val).
    """
    min_count = pd.Series(y).value_counts().min() if len(y) > 0 else 0
    stratify = y if min_count >= 2 else None

    return train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=stratify
    )
