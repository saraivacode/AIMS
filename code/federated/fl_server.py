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
from flwr.common import Metrics, NDArrays, Scalar
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

    # Create initial model and use it for server-side evaluation
    eval_model = create_model(model_type, input_dim)
    init_params = fl.common.ndarrays_to_parameters(eval_model.get_weights())

    # Server-side evaluation results storage
    server_eval_results: List[Dict] = []

    def get_evaluate_fn(model, x_test, y_test_arr, mtype):
        """Return a server-side evaluation function using the global test set."""
        def evaluate(
            server_round: int,
            parameters: NDArrays,
            config: Dict[str, Scalar],
        ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
            model.set_weights(parameters)

            X_input = x_test
            if mtype.upper() in ("LSTM", "GRU"):
                X_input = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

            loss, accuracy = model.evaluate(X_input, y_test_arr, verbose=0)
            preds = np.argmax(model.predict(X_input, verbose=0), axis=1)
            f1 = f1_score(y_test_arr, preds, average="macro", zero_division=0)
            prec = precision_score(y_test_arr, preds, average="macro", zero_division=0)
            rec = recall_score(y_test_arr, preds, average="macro", zero_division=0)
            f1_per_class = f1_score(y_test_arr, preds, average=None, zero_division=0)

            server_eval_results.append({
                "round": server_round,
                "loss": float(loss),
                "accuracy": float(accuracy),
                "f1_macro": float(f1),
                "precision": float(prec),
                "recall": float(rec),
                "f1_per_class": [float(v) for v in f1_per_class],
                "predictions": preds.tolist(),
            })

            return float(loss), {
                "accuracy": float(accuracy),
                "f1_macro": float(f1),
            }
        return evaluate

    evaluate_fn = get_evaluate_fn(eval_model, X_test, y_test, model_type)

    # Configure strategy
    strategy_kwargs = dict(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        initial_parameters=init_params,
        evaluate_fn=evaluate_fn,
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

    # Extract per-round metrics from server-side centralized evaluation
    round_accuracies = []
    round_f1s = []
    round_losses = []

    for res in server_eval_results:
        r = res["round"]
        if r == 0:
            continue  # Skip initial evaluation (round 0 = before any training)
        round_accuracies.append({"round": r, "accuracy": res["accuracy"]})
        round_f1s.append({"round": r, "f1_macro": res["f1_macro"]})
        round_losses.append({"round": r, "loss": res["loss"]})

    # Final metrics from the last round's server-side evaluation
    if server_eval_results:
        last_eval = server_eval_results[-1]
        final_metrics = {
            "accuracy": last_eval["accuracy"],
            "precision": last_eval["precision"],
            "recall": last_eval["recall"],
            "f1_macro": last_eval["f1_macro"],
            "f1_per_class": last_eval["f1_per_class"],
        }
        print(f"    Final test: Acc={final_metrics['accuracy']:.4f}, "
              f"F1={final_metrics['f1_macro']:.4f}, "
              f"Prec={final_metrics['precision']:.4f}, "
              f"Rec={final_metrics['recall']:.4f}")
    else:
        final_metrics = {"accuracy": 0, "precision": 0, "recall": 0, "f1_macro": 0, "f1_per_class": []}
        print("    WARNING: No server evaluation results collected.")

    # Compute convergence metrics
    acc_values = [m["accuracy"] for m in round_accuracies] if round_accuracies else []

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
