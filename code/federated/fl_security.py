#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIMS Framework - Federated Learning Security Module
=====================================================

Implements adversarial attacks and Byzantine-robust defenses for
FL security experiments in the AIMS vehicular network context.

Attacks
-------
- **Label-flip (Critical→Adequate)**: Malicious client flips class 3
  labels to class 0, simulating a safety-critical misclassification
  attack on vehicular network slicing.
- **Gradient scaling**: Malicious client multiplies its model update
  (gradient) by a scale factor before sending to the server,
  amplifying the poisoned contribution.

Defenses
--------
- **Krum**: Selects the single client update closest to the majority,
  effectively filtering outliers (Byzantine-robust for f < n/2 - 1).
- **Trimmed Mean**: Per-parameter aggregation that discards the
  highest and lowest values before averaging, reducing the influence
  of extreme (potentially malicious) updates.

Metrics
-------
- **C2A rate (Critical-to-Adequate)**: Percentage of true Critical
  samples misclassified as Adequate — the primary security metric.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Attacks
# ---------------------------------------------------------------------------

def apply_label_flip(
    y: np.ndarray,
    source_class: int = 3,
    target_class: int = 0,
) -> np.ndarray:
    """
    Label-flip attack: remap *source_class* → *target_class*.

    In the AIMS context this flips Critical (3) labels to Adequate (0),
    causing the model to learn that dangerous network conditions are safe.

    Parameters
    ----------
    y : np.ndarray
        Original labels (modified in-place copy returned).
    source_class : int
        Class to flip FROM (default 3 = Critical).
    target_class : int
        Class to flip TO   (default 0 = Adequate).

    Returns
    -------
    np.ndarray
        Label array with flipped values.
    """
    y_flipped = y.copy()
    mask = y_flipped == source_class
    n_flipped = int(mask.sum())
    y_flipped[mask] = target_class
    print(f"      Label-flip: {n_flipped} samples flipped "
          f"(class {source_class} → {target_class})")
    return y_flipped


def scale_gradients(
    client_weights: List[np.ndarray],
    global_weights: List[np.ndarray],
    scale_factor: float,
) -> List[np.ndarray]:
    """
    Gradient-scaling attack: amplify the malicious client's update.

    Computes ``delta = client - global`` and returns
    ``global + scale_factor * delta``, making the poisoned update
    *scale_factor* times more influential during aggregation.

    Parameters
    ----------
    client_weights : list of np.ndarray
        Weights after local training on the malicious client.
    global_weights : list of np.ndarray
        Global model weights before the round started.
    scale_factor : float
        Multiplicative factor (e.g. 2, 5, 10).

    Returns
    -------
    list of np.ndarray
        Scaled weights to be sent to the server.
    """
    scaled = []
    for cw, gw in zip(client_weights, global_weights):
        delta = cw - gw
        scaled.append(gw + scale_factor * delta)
    return scaled


# ---------------------------------------------------------------------------
# Defenses (Byzantine-robust aggregation)
# ---------------------------------------------------------------------------

def krum_aggregate(
    client_weights: List[List[np.ndarray]],
    client_sizes: List[int],
    num_byzantine: int = 1,
) -> List[np.ndarray]:
    """
    Krum aggregation: select the update closest to the majority.

    For *n* clients and *f* suspected Byzantine clients, Krum computes
    pairwise distances between all client updates, then selects the
    client whose sum of distances to the nearest ``n - f - 2`` clients
    is minimal.

    Parameters
    ----------
    client_weights : list of list of np.ndarray
        Model weights from each client.
    client_sizes : list of int
        Number of training samples per client (unused by Krum but kept
        for API consistency).
    num_byzantine : int
        Maximum expected number of Byzantine (malicious) clients.

    Returns
    -------
    list of np.ndarray
        Weights of the selected (most representative) client.
    """
    n = len(client_weights)
    # Flatten each client's weights into a single vector
    flat = [np.concatenate([w.ravel() for w in cw]) for cw in client_weights]

    # Pairwise squared Euclidean distances
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sum((flat[i] - flat[j]) ** 2)
            dists[i, j] = d
            dists[j, i] = d

    # For each client, sum distances to the (n - f - 2) closest others
    k = max(n - num_byzantine - 2, 1)
    scores = []
    for i in range(n):
        sorted_dists = np.sort(dists[i])  # first element is 0 (self)
        scores.append(np.sum(sorted_dists[1: k + 1]))

    selected = int(np.argmin(scores))
    print(f"      Krum selected client {selected} "
          f"(scores: {[f'{s:.2f}' for s in scores]})")
    return client_weights[selected]


def trimmed_mean_aggregate(
    client_weights: List[List[np.ndarray]],
    client_sizes: List[int],
    trim_fraction: float = 0.1,
) -> List[np.ndarray]:
    """
    Coordinate-wise Trimmed Mean aggregation.

    For each model parameter, stacks values from all clients, trims
    the top and bottom *trim_fraction* of values, and averages the
    remainder.  With 3 clients and ``trim_fraction >= 1/3`` the result
    equals the median; with ``trim_fraction=0`` it reduces to FedAvg.

    Parameters
    ----------
    client_weights : list of list of np.ndarray
        Model weights from each client.
    client_sizes : list of int
        Number of training samples per client (unused but kept for
        API consistency).
    trim_fraction : float
        Fraction of values to remove from EACH tail (0 to 0.5).

    Returns
    -------
    list of np.ndarray
        Aggregated weights after trimming.
    """
    n = len(client_weights)
    num_layers = len(client_weights[0])
    trim_count = max(1, int(n * trim_fraction))

    aggregated = []
    for layer_idx in range(num_layers):
        # Stack all clients' values for this layer
        stacked = np.stack([cw[layer_idx] for cw in client_weights], axis=0)
        # Sort along client axis (axis=0)
        sorted_vals = np.sort(stacked, axis=0)
        # Trim top and bottom
        trimmed = sorted_vals[trim_count: n - trim_count]
        if trimmed.shape[0] == 0:
            # Fallback: use median if all trimmed
            aggregated.append(np.median(stacked, axis=0))
        else:
            aggregated.append(np.mean(trimmed, axis=0))

    print(f"      TrimmedMean: trimmed {trim_count} from each tail "
          f"({n} clients)")
    return aggregated


# ---------------------------------------------------------------------------
# Security metrics
# ---------------------------------------------------------------------------

def compute_c2a_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    critical_class: int = 3,
    adequate_class: int = 0,
) -> float:
    """
    Compute the Critical-to-Adequate (C2A) misclassification rate.

    This is the primary security metric: the fraction of samples that
    are truly Critical but classified as Adequate.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    critical_class : int
        Label for the Critical class (default 3).
    adequate_class : int
        Label for the Adequate class (default 0).

    Returns
    -------
    float
        C2A rate in [0, 1].  Returns 0.0 if there are no Critical
        samples in y_true.
    """
    critical_mask = y_true == critical_class
    n_critical = int(critical_mask.sum())
    if n_critical == 0:
        return 0.0
    n_misclassified = int(np.sum(y_pred[critical_mask] == adequate_class))
    return n_misclassified / n_critical
