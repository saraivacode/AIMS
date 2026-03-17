#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIMS Framework - Federated Learning Main Orchestrator
=====================================================

Top-level entry point for running the complete FL experiment suite.
Coordinates data loading, partitioning, FL simulation, centralized
baselines, results saving, and visualization generation.

Experiment Matrix (full suite):
    3 models (DNN, LSTM, GRU) x 2 distributions (IID, NonIID) x 2 strategies (FedAvg, FedProx)
    = 12 federated experiments + 3 centralized baselines = 15 total

Usage from main.py:
    python main.py --federated
    python main.py --federated --fl-models DNN LSTM --fl-strategies FedAvg
    python main.py --federated --skip-centralized
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List

import numpy as np
import tensorflow as tf

from .fl_centralized import train_centralized
from .fl_config import FLDefaults
from .fl_data import (
    load_and_prepare,
    partition_iid,
    partition_non_iid,
)
from .fl_models import configure_tf
from .fl_results import FLResultsManager
from .fl_server import run_fl_simulation
from .fl_visualizations import (
    plot_client_distribution,
    plot_convergence,
    plot_fl_vs_centralized,
    plot_strategy_comparison,
)

_DEFAULTS = FLDefaults()


def set_seeds(seed: int = 42):
    """Set all random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def run_federated(
    csv_path: Path,
    results_dir: Path,
    num_rounds: int = None,
    local_epochs: int = None,
    batch_size: int = None,
    num_clients: int = None,
    strategies: List[str] = None,
    distributions: List[str] = None,
    model_types: List[str] = None,
    skip_centralized: bool = False,
    seed: int = 42,
) -> None:
    """
    Run the complete federated learning experiment suite.

    Parameters
    ----------
    csv_path : Path
        Path to the AIMS dataset CSV.
    results_dir : Path
        Output directory for FL results.
    num_rounds : int
        Number of FL communication rounds.
    local_epochs : int
        Local training epochs per FL round.
    batch_size : int
        Training batch size.
    num_clients : int
        Number of simulated FL clients (RSUs).
    strategies : list[str]
        Aggregation strategies to evaluate (e.g., ["FedAvg", "FedProx"]).
    distributions : list[str]
        Data distribution modes (e.g., ["IID", "NonIID"]).
    model_types : list[str]
        Neural network architectures (e.g., ["DNN", "LSTM", "GRU"]).
    skip_centralized : bool
        If True, skip centralized baseline training.
    seed : int
        Random seed for reproducibility.
    """
    # Apply defaults
    num_rounds = num_rounds or _DEFAULTS.NUM_ROUNDS
    local_epochs = local_epochs or _DEFAULTS.LOCAL_EPOCHS
    batch_size = batch_size or _DEFAULTS.BATCH_SIZE
    num_clients = num_clients or _DEFAULTS.NUM_CLIENTS
    strategies = strategies or _DEFAULTS.STRATEGIES
    distributions = distributions or _DEFAULTS.DISTRIBUTIONS
    model_types = model_types or _DEFAULTS.MODEL_TYPES

    # Setup
    set_seeds(seed)
    configure_tf()
    tf.get_logger().setLevel("ERROR")

    output_dir = Path(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_mgr = FLResultsManager(output_dir)

    start_time = time.time()

    print("=" * 70)
    print("AIMS FRAMEWORK - FEDERATED LEARNING EXPERIMENTS")
    print("=" * 70)
    print(f"  Dataset: {csv_path}")
    print(f"  Models: {model_types}")
    print(f"  Strategies: {strategies}")
    print(f"  Distributions: {distributions}")
    print(f"  Clients: {num_clients}, Rounds: {num_rounds}, LocalEpochs: {local_epochs}")
    print(f"  Output: {output_dir}")
    print("-" * 70)

    # Step 1: Load and prepare data
    print("\n[Step 1/5] Loading and preparing data...")
    X, y, class_weight_dict, feature_names = load_and_prepare(csv_path, random_state=seed)

    # Global train/test split (80/20 stratified)
    from sklearn.model_selection import train_test_split
    X_train_global, X_test, y_train_global, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y,
    )
    print(f"  Global split: {len(X_train_global)} train, {len(X_test)} test")

    # Step 2: Create client partitions for each distribution
    print("\n[Step 2/5] Partitioning data for FL clients...")
    partitions = {}
    for dist in distributions:
        print(f"\n  {dist} partitioning:")
        if dist.lower() == "iid":
            partitions[dist] = partition_iid(X_train_global, y_train_global, num_clients, seed)
        else:
            partitions[dist] = partition_non_iid(X_train_global, y_train_global, num_clients, seed)
        plot_client_distribution(partitions[dist], dist, output_dir)

    # Step 3: Run federated experiments
    print("\n[Step 3/5] Running federated experiments...")
    total_experiments = len(model_types) * len(distributions) * len(strategies)
    fl_results = []
    exp_num = 0

    for dist in distributions:
        for strategy in strategies:
            for model_type in model_types:
                exp_num += 1
                print(f"\n{'='*60}")
                print(f"  Experiment {exp_num}/{total_experiments}: "
                      f"{model_type} / {dist} / {strategy}")
                print(f"{'='*60}")

                try:
                    result = run_fl_simulation(
                        model_type=model_type,
                        strategy_name=strategy,
                        distribution=dist,
                        client_partitions=partitions[dist],
                        X_test=X_test,
                        y_test=y_test,
                        num_rounds=num_rounds,
                        local_epochs=local_epochs,
                        batch_size=batch_size,
                        class_weights=class_weight_dict,
                        seed=seed,
                        fedprox_mu=_DEFAULTS.FEDPROX_MU,
                    )
                    fl_results.append(result)
                    results_mgr.save_experiment(result)
                except Exception as e:
                    print(f"    FAILED: {e}")
                    import traceback
                    traceback.print_exc()

    # Step 4: Centralized baselines
    centralized_results = []
    if not skip_centralized:
        print(f"\n{'='*60}")
        print("[Step 4/5] Training centralized baselines...")
        print(f"{'='*60}")

        for model_type in model_types:
            try:
                result = train_centralized(
                    model_type=model_type,
                    X_train=X_train_global,
                    y_train=y_train_global,
                    X_test=X_test,
                    y_test=y_test,
                    class_weights=class_weight_dict,
                    epochs=_DEFAULTS.CENTRALIZED_EPOCHS,
                    batch_size=batch_size,
                    seed=seed,
                )
                centralized_results.append(result)
                results_mgr.save_experiment(result)
            except Exception as e:
                print(f"    Centralized {model_type} FAILED: {e}")
    else:
        print("\n[Step 4/5] Skipping centralized baselines (--skip-centralized).")

    # Step 5: Generate summary and visualizations
    print(f"\n{'='*60}")
    print("[Step 5/5] Generating summary and visualizations...")
    print(f"{'='*60}")

    if fl_results:
        results_mgr.save_summary_table(fl_results, centralized_results)
        plot_convergence(fl_results, output_dir)
        plot_strategy_comparison(fl_results, output_dir)

        if centralized_results:
            results_mgr.save_centralized_results(centralized_results)
            plot_fl_vs_centralized(fl_results, centralized_results, output_dir)

    # Print final summary
    total_time = time.time() - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)

    print("\n" + "=" * 70)
    print("FEDERATED LEARNING EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"  Experiments completed: {len(fl_results)}/{total_experiments}")
    print(f"  Centralized baselines: {len(centralized_results)}/{len(model_types)}")
    print(f"  Total time: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")
    print(f"  Results saved in: {output_dir}")

    if fl_results:
        print("\n  Best FL results:")
        best = max(fl_results, key=lambda r: r.get("final_metrics", {}).get("f1_macro", 0))
        fm = best.get("final_metrics", {})
        print(f"    {best['config_name']}: "
              f"Acc={fm.get('accuracy', 0):.4f}, F1={fm.get('f1_macro', 0):.4f}")

    print("=" * 70)
