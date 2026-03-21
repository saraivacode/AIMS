#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIMS Framework - Federated Learning Simulation Orchestrator
============================================================

Implements FedAvg and FedProx aggregation strategies using a manual
simulation loop (no Ray dependency), giving full control over the
training process and avoiding serialization issues.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
)

from .fl_data import split_client_data
from .fl_models import create_model
from .fl_security import (
    apply_label_flip,
    compute_c2a_rate,
    krum_aggregate,
    scale_gradients,
    trimmed_mean_aggregate,
)


def _fedavg_aggregate(
    client_weights: List[List[np.ndarray]],
    client_sizes: List[int],
) -> List[np.ndarray]:
    """
    FedAvg: weighted average of model parameters proportional to dataset size.

    Parameters
    ----------
    client_weights : list of list of np.ndarray
        Model weights from each client after local training.
    client_sizes : list of int
        Number of training samples per client.

    Returns
    -------
    list of np.ndarray
        Aggregated global model weights.
    """
    total = sum(client_sizes)
    avg_weights = []
    for layer_idx in range(len(client_weights[0])):
        layer_sum = np.zeros_like(client_weights[0][layer_idx])
        for c_idx, weights in enumerate(client_weights):
            layer_sum += weights[layer_idx] * (client_sizes[c_idx] / total)
        avg_weights.append(layer_sum)
    return avg_weights


def run_fl_simulation(
    model_type: str,
    strategy_name: str,
    distribution: str,
    client_partitions: List[Tuple[np.ndarray, np.ndarray]],
    X_test: np.ndarray,
    y_test: np.ndarray,
    num_rounds: int,
    local_epochs: int,
    batch_size: int,
    class_weights: Dict[int, float],
    seed: int = 42,
    fedprox_mu: float = 0.1,
) -> Dict:
    """
    Run a single federated learning experiment using manual simulation.

    Implements FedAvg and FedProx without external simulation frameworks,
    avoiding Ray serialization issues while maintaining full FL semantics.

    Parameters
    ----------
    model_type : str
        Neural network type ("DNN", "LSTM", "GRU").
    strategy_name : str
        Aggregation strategy ("FedAvg" or "FedProx").
    distribution : str
        Data distribution mode ("IID" or "NonIID").
    client_partitions : list
        List of (X, y) tuples, one per client.
    X_test : np.ndarray
        Global test features for final evaluation.
    y_test : np.ndarray
        Global test labels.
    num_rounds : int
        Number of FL communication rounds.
    local_epochs : int
        Local training epochs per round.
    batch_size : int
        Training batch size.
    class_weights : dict
        Class weights for imbalanced learning.
    seed : int
        Random seed.
    fedprox_mu : float
        Proximal term coefficient for FedProx.

    Returns
    -------
    dict
        Experiment results including per-round metrics and final evaluation.
    """
    config_name = f"{model_type}_{strategy_name}_{distribution}"
    print(f"\n  Running FL simulation: {config_name}")
    print(f"    Rounds={num_rounds}, LocalEpochs={local_epochs}, Clients={len(client_partitions)}")

    num_clients = len(client_partitions)
    input_dim = client_partitions[0][0].shape[1]
    use_proximal = strategy_name.lower() == "fedprox"

    # Pre-split each client's data into train/val
    client_data = []
    for i, (X_c, y_c) in enumerate(client_partitions):
        X_train, X_val, y_train, y_val = split_client_data(X_c, y_c, test_size=0.2, seed=seed + i)
        client_data.append((X_train, y_train, X_val, y_val))

    # Initialize global model
    global_model = create_model(model_type, input_dim)
    global_weights = global_model.get_weights()

    # Prepare test input (reshape for LSTM/GRU)
    X_test_input = X_test
    if model_type.upper() in ("LSTM", "GRU"):
        X_test_input = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    # Per-round metrics storage
    round_metrics: List[Dict] = []

    start_time = time.time()

    for rnd in range(1, num_rounds + 1):
        client_updated_weights = []
        client_sizes = []

        # --- Local training on each client ---
        for c_idx in range(num_clients):
            X_train, y_train, X_val, y_val = client_data[c_idx]

            # Create fresh client model and load global weights
            client_model = create_model(model_type, input_dim)
            client_model.set_weights(global_weights)

            # Prepare input shapes
            X_tr = X_train
            if model_type.upper() in ("LSTM", "GRU"):
                X_tr = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))

            if use_proximal:
                # FedProx: train epoch-by-epoch with proximal correction
                saved_global = [w.copy() for w in global_weights]
                for _ in range(local_epochs):
                    client_model.fit(
                        X_tr, y_train,
                        epochs=1,
                        batch_size=batch_size,
                        class_weight=class_weights,
                        verbose=0,
                    )
                    # Proximal update: w <- w - mu * (w - w_global)
                    new_w = client_model.get_weights()
                    prox_w = [
                        w - fedprox_mu * (w - gw)
                        for w, gw in zip(new_w, saved_global)
                    ]
                    client_model.set_weights(prox_w)
            else:
                # FedAvg: standard local training
                client_model.fit(
                    X_tr, y_train,
                    epochs=local_epochs,
                    batch_size=batch_size,
                    class_weight=class_weights,
                    verbose=0,
                )

            client_updated_weights.append(client_model.get_weights())
            client_sizes.append(len(X_train))

            # Clean up to free memory
            del client_model

        # --- Server aggregation (FedAvg weighted average) ---
        global_weights = _fedavg_aggregate(client_updated_weights, client_sizes)
        global_model.set_weights(global_weights)

        # --- Server-side evaluation on global test set ---
        loss, accuracy = global_model.evaluate(X_test_input, y_test, verbose=0)
        preds = np.argmax(global_model.predict(X_test_input, verbose=0), axis=1)

        f1 = f1_score(y_test, preds, average="macro", zero_division=0)
        prec = precision_score(y_test, preds, average="macro", zero_division=0)
        rec = recall_score(y_test, preds, average="macro", zero_division=0)
        f1_per_class = f1_score(y_test, preds, average=None, zero_division=0)

        round_metrics.append({
            "round": rnd,
            "loss": float(loss),
            "accuracy": float(accuracy),
            "f1_macro": float(f1),
            "precision": float(prec),
            "recall": float(rec),
            "f1_per_class": [float(v) for v in f1_per_class],
        })

        if rnd % 5 == 0 or rnd == num_rounds or rnd == 1:
            print(f"    Round {rnd:3d}/{num_rounds}: "
                  f"Acc={accuracy:.4f}, F1={f1:.4f}, Loss={loss:.4f}")

    total_time = time.time() - start_time
    print(f"    Simulation completed in {total_time:.1f}s")

    # Build per-round arrays
    round_accuracies = [{"round": m["round"], "accuracy": m["accuracy"]} for m in round_metrics]
    round_f1s = [{"round": m["round"], "f1_macro": m["f1_macro"]} for m in round_metrics]
    round_losses = [{"round": m["round"], "loss": m["loss"]} for m in round_metrics]

    # Final metrics from last round
    last = round_metrics[-1] if round_metrics else {}
    final_metrics = {
        "accuracy": last.get("accuracy", 0),
        "precision": last.get("precision", 0),
        "recall": last.get("recall", 0),
        "f1_macro": last.get("f1_macro", 0),
        "f1_per_class": last.get("f1_per_class", []),
    }
    print(f"    Final test: Acc={final_metrics['accuracy']:.4f}, "
          f"F1={final_metrics['f1_macro']:.4f}, "
          f"Prec={final_metrics['precision']:.4f}, "
          f"Rec={final_metrics['recall']:.4f}")

    # Convergence metrics
    acc_values = [m["accuracy"] for m in round_metrics]
    conv_85 = _find_convergence_round(acc_values, 0.85)
    conv_90 = _find_convergence_round(acc_values, 0.90)
    stability = float(np.std(acc_values[-5:])) if len(acc_values) >= 5 else (
        float(np.std(acc_values)) if acc_values else 0.0
    )

    return {
        "config_name": config_name,
        "model_type": model_type,
        "strategy": strategy_name,
        "distribution": distribution,
        "num_clients": num_clients,
        "num_rounds": num_rounds,
        "local_epochs": local_epochs,
        "batch_size": batch_size,
        "per_round": {
            "accuracy": round_accuracies,
            "f1_macro": round_f1s,
            "loss": round_losses,
        },
        "final_metrics": final_metrics,
        "convergence": {
            "round_85": conv_85,
            "round_90": conv_90,
            "stability_std": stability,
        },
        "total_time": total_time,
    }


def _find_convergence_round(
    values: List[float], threshold: float
) -> Optional[int]:
    """Find the first round where metric >= threshold."""
    for i, v in enumerate(values):
        if v >= threshold:
            return i + 1  # rounds are 1-indexed
    return None


# ---------------------------------------------------------------------------
# Security-aware FL simulation
# ---------------------------------------------------------------------------

def _get_aggregator(defense: str):
    """Return the aggregation function for the given defense name."""
    if defense.lower() == "krum":
        return krum_aggregate
    elif defense.lower() in ("trimmedmean", "trimmed_mean"):
        return trimmed_mean_aggregate
    else:
        return _fedavg_aggregate


def run_fl_security_simulation(
    model_type: str,
    strategy_name: str,
    distribution: str,
    client_partitions: List[Tuple[np.ndarray, np.ndarray]],
    X_test: np.ndarray,
    y_test: np.ndarray,
    num_rounds: int,
    local_epochs: int,
    batch_size: int,
    class_weights: Dict[int, float],
    seed: int = 42,
    fedprox_mu: float = 0.1,
    # Security parameters
    attack_type: str = "label_flip",
    defense: str = "FedAvg",
    malicious_clients: Optional[List[int]] = None,
    gradient_scale: float = 1.0,
    source_class: int = 3,
    target_class: int = 0,
) -> Dict:
    """
    Run a federated learning experiment with adversarial attacks and defenses.

    Extends ``run_fl_simulation`` with:
    - Label-flip attack on designated malicious clients
    - Optional gradient scaling on malicious updates
    - Byzantine-robust aggregation (Krum or Trimmed Mean)
    - C2A rate tracking per round

    Parameters
    ----------
    model_type, strategy_name, distribution, client_partitions,
    X_test, y_test, num_rounds, local_epochs, batch_size,
    class_weights, seed, fedprox_mu :
        Same as ``run_fl_simulation``.
    attack_type : str
        Attack type: "label_flip" or "label_flip+gradient_scaling".
    defense : str
        Aggregation defense: "FedAvg", "Krum", or "TrimmedMean".
    malicious_clients : list of int, optional
        Indices of malicious clients (default: [0]).
    gradient_scale : float
        Scale factor for gradient scaling attack (default 1.0 = no scaling).
    source_class : int
        Source class for label-flip (default 3 = Critical).
    target_class : int
        Target class for label-flip (default 0 = Adequate).

    Returns
    -------
    dict
        Experiment results including per-round metrics, C2A rates,
        and security configuration.
    """
    if malicious_clients is None:
        malicious_clients = [0]

    scale_str = f"_x{gradient_scale:.0f}" if gradient_scale > 1.0 else ""
    config_name = (f"{model_type}_{strategy_name}_{distribution}"
                   f"_attack-{attack_type}{scale_str}_def-{defense}")

    print(f"\n  Security FL simulation: {config_name}")
    print(f"    Rounds={num_rounds}, LocalEpochs={local_epochs}, "
          f"Clients={len(client_partitions)}")
    print(f"    Attack={attack_type}, Scale={gradient_scale}, "
          f"Defense={defense}, Malicious={malicious_clients}")

    num_clients = len(client_partitions)
    input_dim = client_partitions[0][0].shape[1]
    use_proximal = strategy_name.lower() == "fedprox"
    aggregator = _get_aggregator(defense)

    # Pre-split each client's data; apply label-flip to malicious clients
    client_data = []
    for i, (X_c, y_c) in enumerate(client_partitions):
        y_local = y_c.copy()
        if i in malicious_clients:
            y_local = apply_label_flip(y_local, source_class, target_class)
        X_train, X_val, y_train, y_val = split_client_data(
            X_c, y_local, test_size=0.2, seed=seed + i
        )
        client_data.append((X_train, y_train, X_val, y_val))

    # Initialize global model
    global_model = create_model(model_type, input_dim)
    global_weights = global_model.get_weights()

    # Prepare test input
    X_test_input = X_test
    if model_type.upper() in ("LSTM", "GRU"):
        X_test_input = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    round_metrics: List[Dict] = []
    start_time = time.time()

    for rnd in range(1, num_rounds + 1):
        client_updated_weights = []
        client_sizes = []

        for c_idx in range(num_clients):
            X_train, y_train, X_val, y_val = client_data[c_idx]

            client_model = create_model(model_type, input_dim)
            client_model.set_weights(global_weights)

            X_tr = X_train
            if model_type.upper() in ("LSTM", "GRU"):
                X_tr = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))

            if use_proximal:
                saved_global = [w.copy() for w in global_weights]
                for _ in range(local_epochs):
                    client_model.fit(
                        X_tr, y_train, epochs=1,
                        batch_size=batch_size,
                        class_weight=class_weights, verbose=0,
                    )
                    new_w = client_model.get_weights()
                    prox_w = [w - fedprox_mu * (w - gw)
                              for w, gw in zip(new_w, saved_global)]
                    client_model.set_weights(prox_w)
            else:
                client_model.fit(
                    X_tr, y_train, epochs=local_epochs,
                    batch_size=batch_size,
                    class_weight=class_weights, verbose=0,
                )

            updated_w = client_model.get_weights()

            # Apply gradient scaling for malicious clients
            if c_idx in malicious_clients and gradient_scale > 1.0:
                updated_w = scale_gradients(
                    updated_w, global_weights, gradient_scale
                )

            client_updated_weights.append(updated_w)
            client_sizes.append(len(X_train))
            del client_model

        # Server aggregation with selected defense
        global_weights = aggregator(client_updated_weights, client_sizes)
        global_model.set_weights(global_weights)

        # Evaluation
        loss, accuracy = global_model.evaluate(X_test_input, y_test, verbose=0)
        preds = np.argmax(global_model.predict(X_test_input, verbose=0), axis=1)

        f1 = f1_score(y_test, preds, average="macro", zero_division=0)
        prec = precision_score(y_test, preds, average="macro", zero_division=0)
        rec = recall_score(y_test, preds, average="macro", zero_division=0)
        f1_per_class = f1_score(y_test, preds, average=None, zero_division=0)
        c2a = compute_c2a_rate(y_test, preds, source_class, target_class)

        round_metrics.append({
            "round": rnd,
            "loss": float(loss),
            "accuracy": float(accuracy),
            "f1_macro": float(f1),
            "precision": float(prec),
            "recall": float(rec),
            "f1_per_class": [float(v) for v in f1_per_class],
            "c2a_rate": float(c2a),
        })

        if rnd % 5 == 0 or rnd == num_rounds or rnd == 1:
            print(f"    Round {rnd:3d}/{num_rounds}: "
                  f"Acc={accuracy:.4f}, F1={f1:.4f}, C2A={c2a:.4f}")

    total_time = time.time() - start_time
    print(f"    Completed in {total_time:.1f}s")

    # Build result
    last = round_metrics[-1] if round_metrics else {}
    acc_values = [m["accuracy"] for m in round_metrics]
    conv_85 = _find_convergence_round(acc_values, 0.85)
    conv_90 = _find_convergence_round(acc_values, 0.90)
    stability = (float(np.std(acc_values[-5:])) if len(acc_values) >= 5
                 else float(np.std(acc_values)) if acc_values else 0.0)

    return {
        "config_name": config_name,
        "model_type": model_type,
        "strategy": strategy_name,
        "distribution": distribution,
        "num_clients": num_clients,
        "num_rounds": num_rounds,
        "local_epochs": local_epochs,
        "batch_size": batch_size,
        # Security fields
        "attack_type": attack_type,
        "defense": defense,
        "gradient_scale": gradient_scale,
        "malicious_clients": malicious_clients,
        "per_round": {
            "accuracy": [{"round": m["round"], "accuracy": m["accuracy"]}
                         for m in round_metrics],
            "f1_macro": [{"round": m["round"], "f1_macro": m["f1_macro"]}
                         for m in round_metrics],
            "loss": [{"round": m["round"], "loss": m["loss"]}
                     for m in round_metrics],
            "c2a_rate": [{"round": m["round"], "c2a_rate": m["c2a_rate"]}
                         for m in round_metrics],
        },
        "final_metrics": {
            "accuracy": last.get("accuracy", 0),
            "precision": last.get("precision", 0),
            "recall": last.get("recall", 0),
            "f1_macro": last.get("f1_macro", 0),
            "f1_per_class": last.get("f1_per_class", []),
            "c2a_rate": last.get("c2a_rate", 0),
        },
        "convergence": {
            "round_85": conv_85,
            "round_90": conv_90,
            "stability_std": stability,
        },
        "total_time": total_time,
    }
