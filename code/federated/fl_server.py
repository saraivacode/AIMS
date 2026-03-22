#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIMS Framework - Federated Learning Simulation Orchestrator
============================================================

Implements federated learning experiments using the Flower framework
with Ray-based parallel simulation, supporting FedAvg and FedProx
strategies with Byzantine-robust aggregation (Krum, TrimmedMean)
for security experiments.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
import numpy as np
from flwr.common import (
    Context,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    FitRes,
    Parameters,
    Scalar,
)
from flwr.server import ServerConfig
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
)

from .fl_client import AIMSFlowerClient
from .fl_config import FLDefaults
from .fl_data import split_client_data
from .fl_models import create_model
from .fl_security import (
    apply_label_flip,
    compute_c2a_rate,
    krum_aggregate,
    scale_gradients,
    trimmed_mean_aggregate,
)

_DEFAULTS = FLDefaults()


# ---------------------------------------------------------------------------
# Custom Flower strategies for Byzantine-robust aggregation
# ---------------------------------------------------------------------------

class _KrumStrategy(FedAvg):
    """FedAvg variant using Krum Byzantine-robust aggregation."""

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        client_weights = []
        client_sizes = []
        for _, fit_res in results:
            weights = parameters_to_ndarrays(fit_res.parameters)
            client_weights.append(weights)
            client_sizes.append(fit_res.num_examples)

        aggregated = krum_aggregate(client_weights, client_sizes)
        return ndarrays_to_parameters(aggregated), {}


class _TrimmedMeanStrategy(FedAvg):
    """FedAvg variant using coordinate-wise Trimmed Mean aggregation."""

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        client_weights = []
        client_sizes = []
        for _, fit_res in results:
            weights = parameters_to_ndarrays(fit_res.parameters)
            client_weights.append(weights)
            client_sizes.append(fit_res.num_examples)

        aggregated = trimmed_mean_aggregate(client_weights, client_sizes)
        return ndarrays_to_parameters(aggregated), {}


# ---------------------------------------------------------------------------
# Malicious Flower client for gradient scaling attacks
# ---------------------------------------------------------------------------

class _MaliciousFlowerClient(AIMSFlowerClient):
    """
    Flower client that applies gradient scaling attack.

    Extends AIMSFlowerClient to scale the model update (gradient)
    by a given factor after local training, amplifying the poisoned
    contribution during aggregation.
    """

    def __init__(self, *args, gradient_scale: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.gradient_scale = gradient_scale

    def fit(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[List[np.ndarray], int, Dict]:
        # parameters holds the global weights (not modified by super().fit())
        global_weights = parameters
        weights, num_samples, metrics = super().fit(parameters, config)
        if self.gradient_scale > 1.0:
            weights = scale_gradients(weights, global_weights, self.gradient_scale)
        return weights, num_samples, metrics


# ---------------------------------------------------------------------------
# Resource detection
# ---------------------------------------------------------------------------

def _get_client_resources(num_clients: int) -> Dict[str, float]:
    """Determine Ray client resources based on available GPUs."""
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        return {"num_cpus": 1, "num_gpus": len(gpus) / max(num_clients, 1)}
    return {"num_cpus": 1, "num_gpus": 0.0}


# ---------------------------------------------------------------------------
# Standard FL simulation (Flower + Ray)
# ---------------------------------------------------------------------------

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
    Run a single federated learning experiment using Flower simulation with Ray.

    Uses the Flower framework's simulation API for parallel client training,
    with FedAvg aggregation on the server side. FedProx proximal term is
    applied client-side in AIMSFlowerClient when proximal_mu > 0.

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
    print(f"\n  Running FL simulation (Flower): {config_name}")
    print(f"    Rounds={num_rounds}, LocalEpochs={local_epochs}, "
          f"Clients={len(client_partitions)}")

    num_clients = len(client_partitions)
    input_dim = client_partitions[0][0].shape[1]
    use_proximal = strategy_name.lower() == "fedprox"

    # Pre-split each client's data into train/val
    client_data = []
    for i, (X_c, y_c) in enumerate(client_partitions):
        X_train, X_val, y_train, y_val = split_client_data(
            X_c, y_c, test_size=0.2, seed=seed + i
        )
        client_data.append((X_train, y_train, X_val, y_val))

    # Initial global model parameters
    initial_model = create_model(model_type, input_dim)
    initial_parameters = ndarrays_to_parameters(initial_model.get_weights())
    del initial_model

    # Prepare test input (reshape for LSTM/GRU)
    X_test_input = X_test
    if model_type.upper() in ("LSTM", "GRU"):
        X_test_input = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    # Per-round metrics (captured by evaluate_fn closure)
    round_metrics: List[Dict] = []

    def evaluate_fn(server_round, parameters_ndarrays, config):
        """Server-side evaluation on global test set after each round."""
        model = create_model(model_type, input_dim)
        model.set_weights(parameters_ndarrays)

        loss, accuracy = model.evaluate(X_test_input, y_test, verbose=0)
        preds = np.argmax(model.predict(X_test_input, verbose=0), axis=1)

        f1 = f1_score(y_test, preds, average="macro", zero_division=0)
        prec = precision_score(y_test, preds, average="macro", zero_division=0)
        rec = recall_score(y_test, preds, average="macro", zero_division=0)
        f1_per_class = f1_score(y_test, preds, average=None, zero_division=0)

        round_metrics.append({
            "round": server_round,
            "loss": float(loss),
            "accuracy": float(accuracy),
            "f1_macro": float(f1),
            "precision": float(prec),
            "recall": float(rec),
            "f1_per_class": [float(v) for v in f1_per_class],
        })

        if server_round % 5 == 0 or server_round == num_rounds or server_round == 1:
            print(f"    Round {server_round:3d}/{num_rounds}: "
                  f"Acc={accuracy:.4f}, F1={f1:.4f}, Loss={loss:.4f}")

        del model
        return float(loss), {"accuracy": float(accuracy), "f1_macro": float(f1)}

    # Flower strategy: FedAvg aggregation for both FedAvg and FedProx
    # (the proximal term is applied client-side in AIMSFlowerClient)
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,  # Server-side evaluation only
        min_fit_clients=num_clients,
        min_available_clients=num_clients,
        evaluate_fn=evaluate_fn,
        initial_parameters=initial_parameters,
    )

    # Client factory for Flower simulation
    def client_fn(context: Context):
        idx = int(context.node_config["partition-id"])
        X_train, y_train, X_val, y_val = client_data[idx]
        return AIMSFlowerClient(
            model_type=model_type,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            local_epochs=local_epochs,
            batch_size=batch_size,
            class_weights=class_weights,
            client_id=idx,
            proximal_mu=fedprox_mu if use_proximal else 0.0,
        ).to_client()

    start_time = time.time()
    client_resources = _get_client_resources(num_clients)

    # Run Flower simulation with Ray
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        ray_init_args={"include_dashboard": False, "ignore_reinit_error": True},
        client_resources=client_resources,
        keep_initialised=True,
    )

    total_time = time.time() - start_time
    print(f"    Simulation completed in {total_time:.1f}s")

    # Build result dict (same format as before for compatibility)
    round_accuracies = [{"round": m["round"], "accuracy": m["accuracy"]}
                        for m in round_metrics]
    round_f1s = [{"round": m["round"], "f1_macro": m["f1_macro"]}
                 for m in round_metrics]
    round_losses = [{"round": m["round"], "loss": m["loss"]}
                    for m in round_metrics]

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
# Security-aware FL simulation (Flower + Ray)
# ---------------------------------------------------------------------------

def _get_security_strategy(
    defense: str,
    num_clients: int,
    evaluate_fn,
    initial_parameters,
) -> FedAvg:
    """Create the appropriate Flower strategy for the given defense."""
    common_kwargs = dict(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=num_clients,
        min_available_clients=num_clients,
        evaluate_fn=evaluate_fn,
        initial_parameters=initial_parameters,
    )
    defense_lower = defense.lower()
    if defense_lower == "krum":
        return _KrumStrategy(**common_kwargs)
    elif defense_lower in ("trimmedmean", "trimmed_mean"):
        return _TrimmedMeanStrategy(**common_kwargs)
    else:
        return FedAvg(**common_kwargs)


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
    Run a federated learning experiment with adversarial attacks and defenses
    using Flower simulation with Ray.

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
    epochs_str = f"_ep{local_epochs}" if local_epochs != _DEFAULTS.LOCAL_EPOCHS else ""
    config_name = (f"{model_type}_{strategy_name}_{distribution}"
                   f"_attack-{attack_type}{scale_str}_def-{defense}{epochs_str}")

    print(f"\n  Security FL simulation (Flower): {config_name}")
    print(f"    Rounds={num_rounds}, LocalEpochs={local_epochs}, "
          f"Clients={len(client_partitions)}")
    print(f"    Attack={attack_type}, Scale={gradient_scale}, "
          f"Defense={defense}, Malicious={malicious_clients}")

    num_clients = len(client_partitions)
    input_dim = client_partitions[0][0].shape[1]
    use_proximal = strategy_name.lower() == "fedprox"

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

    # Initial global model parameters
    initial_model = create_model(model_type, input_dim)
    initial_parameters = ndarrays_to_parameters(initial_model.get_weights())
    del initial_model

    # Prepare test input
    X_test_input = X_test
    if model_type.upper() in ("LSTM", "GRU"):
        X_test_input = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    # Per-round metrics
    round_metrics: List[Dict] = []

    def evaluate_fn(server_round, parameters_ndarrays, config):
        """Server-side evaluation with security metrics (C2A rate)."""
        model = create_model(model_type, input_dim)
        model.set_weights(parameters_ndarrays)

        loss, accuracy = model.evaluate(X_test_input, y_test, verbose=0)
        preds = np.argmax(model.predict(X_test_input, verbose=0), axis=1)

        f1 = f1_score(y_test, preds, average="macro", zero_division=0)
        prec = precision_score(y_test, preds, average="macro", zero_division=0)
        rec = recall_score(y_test, preds, average="macro", zero_division=0)
        f1_per_class = f1_score(y_test, preds, average=None, zero_division=0)
        c2a = compute_c2a_rate(y_test, preds, source_class, target_class)

        round_metrics.append({
            "round": server_round,
            "loss": float(loss),
            "accuracy": float(accuracy),
            "f1_macro": float(f1),
            "precision": float(prec),
            "recall": float(rec),
            "f1_per_class": [float(v) for v in f1_per_class],
            "c2a_rate": float(c2a),
        })

        if server_round % 5 == 0 or server_round == num_rounds or server_round == 1:
            print(f"    Round {server_round:3d}/{num_rounds}: "
                  f"Acc={accuracy:.4f}, F1={f1:.4f}, C2A={c2a:.4f}")

        del model
        return float(loss), {
            "accuracy": float(accuracy),
            "f1_macro": float(f1),
            "c2a_rate": float(c2a),
        }

    # Create strategy based on defense type
    strategy = _get_security_strategy(
        defense, num_clients, evaluate_fn, initial_parameters
    )

    # Client factory (malicious clients get gradient scaling)
    def client_fn(context: Context):
        idx = int(context.node_config["partition-id"])
        X_train, y_train, X_val, y_val = client_data[idx]

        client_kwargs = dict(
            model_type=model_type,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            local_epochs=local_epochs,
            batch_size=batch_size,
            class_weights=class_weights,
            client_id=idx,
            proximal_mu=fedprox_mu if use_proximal else 0.0,
        )

        if idx in malicious_clients and gradient_scale > 1.0:
            return _MaliciousFlowerClient(
                gradient_scale=gradient_scale, **client_kwargs
            ).to_client()
        else:
            return AIMSFlowerClient(**client_kwargs).to_client()

    start_time = time.time()
    client_resources = _get_client_resources(num_clients)

    # Run Flower simulation with Ray
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        ray_init_args={"include_dashboard": False, "ignore_reinit_error": True},
        client_resources=client_resources,
        keep_initialised=True,
    )

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
