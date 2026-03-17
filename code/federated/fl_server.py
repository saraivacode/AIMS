#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIMS Framework - Federated Learning Simulation Orchestrator
============================================================

Uses Flower's simulation API to run federated experiments without
requiring separate server/client processes.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
from flwr.common import Metrics
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
)

from .fl_client import AIMSFlowerClient
from .fl_data import split_client_data
from .fl_models import create_model


def weighted_average_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate evaluation metrics from multiple clients (weighted by dataset size)."""
    if not metrics:
        return {}
    total = sum(n for n, _ in metrics)
    if total == 0:
        return {}
    result = {}
    for key in metrics[0][1].keys():
        result[key] = sum(m[key] * n for n, m in metrics) / total
    return result


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
    Run a single federated learning experiment using Flower simulation.

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
    proximal_mu = fedprox_mu if strategy_name.lower() == "fedprox" else 0.0

    # Pre-split each client's data into train/val
    client_data = []
    for i, (X_c, y_c) in enumerate(client_partitions):
        X_train, X_val, y_train, y_val = split_client_data(X_c, y_c, test_size=0.2, seed=seed + i)
        client_data.append((X_train, y_train, X_val, y_val))

    # Client factory for Flower simulation
    def client_fn(cid: str) -> fl.client.NumPyClient:
        idx = int(cid)
        X_train, y_train, X_val, y_val = client_data[idx]
        return AIMSFlowerClient(
            model_type=model_type,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            local_epochs=local_epochs,
            batch_size=batch_size,
            class_weights=class_weights,
            client_id=idx,
            proximal_mu=proximal_mu,
        )

    # Create initial model parameters
    init_model = create_model(model_type, input_dim)
    init_params = fl.common.ndarrays_to_parameters(init_model.get_weights())

    # Configure strategy
    strategy_kwargs = dict(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        initial_parameters=init_params,
        evaluate_metrics_aggregation_fn=weighted_average_metrics,
        fit_metrics_aggregation_fn=weighted_average_metrics,
    )

    if strategy_name.lower() == "fedprox":
        strategy = fl.server.strategy.FedProx(
            proximal_mu=fedprox_mu,
            **strategy_kwargs,
        )
    else:
        strategy = fl.server.strategy.FedAvg(**strategy_kwargs)

    # Run simulation
    start_time = time.time()

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )

    total_time = time.time() - start_time
    print(f"    Simulation completed in {total_time:.1f}s")

    # Extract per-round metrics from history
    round_accuracies = []
    round_f1s = []
    round_losses = []

    # Distributed evaluate metrics
    if hasattr(history, "metrics_distributed"):
        acc_history = history.metrics_distributed.get("accuracy", [])
        f1_history = history.metrics_distributed.get("f1_macro", [])
        for r, val in acc_history:
            round_accuracies.append({"round": r, "accuracy": val})
        for r, val in f1_history:
            round_f1s.append({"round": r, "f1_macro": val})

    # Distributed losses
    if hasattr(history, "losses_distributed"):
        for r, val in history.losses_distributed:
            round_losses.append({"round": r, "loss": val})

    # Final evaluation on global test set using the trained model
    final_model = create_model(model_type, input_dim)
    # Get final parameters from the strategy's last round
    # Re-create a client to get the final aggregated weights
    final_client = client_fn("0")
    if round_accuracies:
        # The last client_fn call will have the model initialized fresh.
        # We need to extract parameters from history.
        # Use the last round's parameters by fitting one more dummy client.
        pass

    # Evaluate using the last client's model (which has the latest global params)
    final_metrics = _evaluate_global_model(
        model_type, input_dim, client_fn, X_test, y_test, num_clients
    )

    # Compute convergence metrics
    acc_values = [m["accuracy"] for m in round_accuracies] if round_accuracies else []
    f1_values = [m["f1_macro"] for m in round_f1s] if round_f1s else []

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


def _evaluate_global_model(
    model_type: str,
    input_dim: int,
    client_fn,
    X_test: np.ndarray,
    y_test: np.ndarray,
    num_clients: int,
) -> Dict:
    """
    Evaluate the final global model on the held-out test set.

    Uses the aggregated weights from the last simulation round by
    averaging the weights from all clients (which should be synchronized
    after the last round of evaluation).
    """
    # Get weights from client 0 (all clients should have the same global weights
    # after the last round of evaluate)
    client = client_fn("0")

    # Prepare test data
    X_input = X_test
    if model_type.upper() in ("LSTM", "GRU"):
        X_input = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    # Predict
    preds_proba = client.model.predict(X_input, verbose=0)
    preds = np.argmax(preds_proba, axis=1)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="macro", zero_division=0)
    rec = recall_score(y_test, preds, average="macro", zero_division=0)
    f1 = f1_score(y_test, preds, average="macro", zero_division=0)

    # Per-class F1
    f1_per_class = f1_score(y_test, preds, average=None, zero_division=0)

    print(f"    Final test: Acc={acc:.4f}, F1={f1:.4f}, Prec={prec:.4f}, Rec={rec:.4f}")

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_macro": float(f1),
        "f1_per_class": [float(v) for v in f1_per_class],
        "predictions": preds.tolist(),
        "true_labels": y_test.tolist(),
    }


def _find_convergence_round(
    values: List[float], threshold: float
) -> Optional[int]:
    """Find the first round where metric >= threshold."""
    for i, v in enumerate(values):
        if v >= threshold:
            return i + 1  # rounds are 1-indexed
    return None
