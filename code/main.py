#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AIMS Framework - Main Training Orchestrator
===========================================

This script is the central entry point of the AIMS framework, orchestrating the end-to-end training,
hyperparameter optimization, and evaluation of multiple machine learning models for network slicing
impact classification in vehicular networks.

The pipeline supports three complementary approaches:
1. Random Forest (RF) – Tree-based ensemble with bagging for robust baseline performance
2. TabNet – Deep learning with attention mechanisms optimized for tabular data
3. CatBoost – Gradient boosting with native categorical feature support and advanced regularization

All models are trained and evaluated using:
- GroupKFold cross-validation (5 splits) to prevent temporal leakage in time-series data
- Hyperparameter optimization with Optuna for automated parameter tuning
- Class-balanced weights to address imbalanced impact level distribution
- Consistent train/test splits for fair model comparison
- Comprehensive artifact generation including confusion matrices and feature importance

Key Components
--------------
1. **Model Execution**: Orchestrates training pipelines for all three classifiers with error handling
   and progress tracking for each model type

2. **Flexible Configuration**: Command-line interface supporting:
   - Custom dataset paths and output directories
   - Configurable cross-validation and optimization parameters
   - Selective model training (skip specific models as needed)
   - Test modes for development and debugging

3. **Reproducibility Management**: Ensures consistent results through:
   - Fixed random seeds across all models and experiments
   - Standardized data preprocessing and feature engineering
   - Unified evaluation metrics and reporting formats

4. **Results Analysis**: Integrated comparison system that:
   - Generates consolidated performance reports across all trained models
   - Produces comparative visualizations and exportable metrics
   - Supports both automated and on-demand analysis workflows

Dataset Information
------------------
The framework processes vehicular network QoS data derived from the experimental setup described in:
T. do Vale Saraiva et al., "An Application-Driven Framework for Intelligent Transportation Systems Using 5G Network Slicing,"
IEEE Transactions on Intelligent Transportation Systems, vol. 22, no. 8, pp. 5247–5260, Aug. 2021.
DOI: 10.1109/TITS.2021.3086064.

The data has been preprocessed and feature-engineered specifically for impact classification tasks.

Usage Examples
--------------
```bash
# Train all three models with 15 trials:
python main.py --compare --csv ../data/aims_dataset.csv --n-trials 15 --n-trials-tabnet 15

# Custom dataset with increased optimization trials
python main.py --csv ./data/custom_dataset.csv --n-trials 100

# Skip computationally expensive model and generate comparison
python main.py --skip-tabnet --compare

# Development mode - test comparison logic only
python main.py --test-comparison-only

# Targeted training with custom output directory
python main.py --skip-rf --results-dir ./experiments/run_001

# Federated Learning only (skips RF, TabNet, CatBoost)
python main.py --federated-only --csv ../data/aims_dataset.csv

# Random Forest
python main.py --compare --csv ../data/aims_dataset.csv --n-trials 15 --skip-catboost --skip-tabnet

# CatBoost
python main.py --compare --csv ../data/aims_dataset.csv --n-trials 15 --skip-rf --skip-tabnet

# TabNet
python main.py --compare --csv ../data/aims_dataset.csv --n-trials-tabnet 15 --skip-catboost --skip-rf
```
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

# --- Local imports ---
import train_model_rf
import train_model_catboost
from train_model_tabnet import run_tabnet

# Imports for results comparison.
from compare_results import generate_report
COMPARE_AVAILABLE = True

def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments for the main training pipeline.

    Returns:
        argparse.Namespace: An object containing the parsed arguments.
    """

    parser = argparse.ArgumentParser(
        description="AIMS Framework - Train and evaluate classifiers for vehicular impact prediction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--csv", type=Path, default=Path("../data/aims_dataset.csv"),
        help="Path to the vehicular QoS dataset CSV file.")

    parser.add_argument(
        "--n-splits", type=int, default=5, help="Number of folds for GroupKFold cross-validation.")

    parser.add_argument(
        "--n-trials", type=int, default=40, help="Number of Optuna trials for RandomForest and CatBoost.")

    parser.add_argument(
        "--n-trials-tabnet", type=int, default=40,
        help="Number of Optuna trials for TabNet (computationally expensive).")

    parser.add_argument(
        "--random-state", type=int, default=42, help="Random seed for ensuring reproducibility.")

    parser.add_argument("--skip-rf", action="store_true", help="Skip Random Forest training.")

    parser.add_argument("--skip-tabnet", action="store_true", help="Skip TabNet training.")

    parser.add_argument("--skip-catboost", action="store_true", help="Skip CatBoost training.")

    parser.add_argument(
        "--compare", action="store_true",
        help="Generate a comparison report after all training is complete.")

    parser.add_argument(
        "--results-dir", type=Path, default=Path("../results"),
        help="Base directory containing model results for comparison.")

    parser.add_argument(
        "--experiment-id", type=str, default=None,
        help="Experiment identifier used as a subdirectory under --results-dir. "
             "All outputs are saved under results-dir/<experiment-id>/. "
             "Defaults to a UTC timestamp, e.g. '20260321T143052Z'.")

    parser.add_argument(
        "--test-comparison-only", action="store_true",
        help="Skip all training and only run the comparison logic.")

    # --- Federated Learning arguments ---
    parser.add_argument(
        "--federated", action="store_true",
        help="Run Federated Learning experiments (DNN, LSTM, GRU with FedAvg/FedProx).")

    parser.add_argument(
        "--federated-only", action="store_true",
        help="Run ONLY Federated Learning experiments, skipping all centralized models (RF, TabNet, CatBoost).")

    parser.add_argument(
        "--fl-rounds", type=int, default=30,
        help="Number of FL communication rounds.")

    parser.add_argument(
        "--fl-local-epochs", type=int, default=3,
        help="Number of local training epochs per FL round.")

    parser.add_argument(
        "--fl-clients", type=int, default=3,
        help="Number of simulated FL clients (RSUs).")

    parser.add_argument(
        "--fl-strategies", nargs="+", default=["FedAvg", "FedProx"],
        help="FL aggregation strategies to evaluate.")

    parser.add_argument(
        "--fl-distributions", nargs="+", default=["IID", "NonIID"],
        help="Data distribution modes for FL clients.")

    parser.add_argument(
        "--fl-models", nargs="+", default=["DNN", "LSTM", "GRU"],
        help="Neural network architectures for FL experiments.")

    parser.add_argument(
        "--skip-centralized", action="store_true",
        help="Skip centralized baseline training in FL experiments.")

    # --- FL Security arguments ---
    parser.add_argument(
        "--security", action="store_true",
        help="Run all security experiments: Phase 1a (label-flip) + Phase 1b "
             "(label-flip + gradient scaling 2x/5x/10x) with FedAvg/Krum/TrimmedMean "
             "for DNN/LSTM/GRU on NonIID. Use --skip-phase1a or --skip-phase1b to "
             "skip individual phases.")

    parser.add_argument(
        "--skip-phase1a", action="store_true",
        help="Skip Phase 1a (label-flip only) when running --security.")

    parser.add_argument(
        "--skip-phase1b", action="store_true",
        help="Skip Phase 1b (label-flip + gradient scaling) when running --security.")

    parser.add_argument(
        "--security-sensitivity-epochs", action="store_true",
        help="Sensitivity analysis: vary local epochs (1, 3, 5, 10) under label-flip attack.")

    return parser.parse_args()


def validate_dataset(csv_path: Path) -> bool:
    """
    Validates the existence and basic format of the dataset file.

    Args:
        csv_path (Path): The path to the dataset CSV file.

    Returns:
        bool: True if the dataset is valid, False otherwise.
    """

    if not csv_path.exists():
        print(f"❌ Error: Dataset not found at {csv_path}")
        print("   Please check the path and try again.")
        return False
    if not csv_path.is_file():
        print(f"❌ Error: The provided path {csv_path} is not a file.")
        return False
    if csv_path.suffix.lower() != '.csv':
        print(f"⚠️  Warning: Expected a .csv file, but got '{csv_path.suffix}'.")
    return True


def train_random_forest(args: argparse.Namespace) -> bool:
    """
    Wrapper to run the Random Forest training pipeline and handle exceptions.

    Args:
        args (argparse.Namespace): Arguments to pass to the training script.

    Returns:
        bool: True on success, False on failure.
    """
    try:
        train_model_rf.main(args)
        return True
    except Exception as e:
        print(f"❌ Error during Random Forest training: {e}")
        return False


def train_tabnet(csv_path: Path, n_splits: int, n_trials: int, random_state: int,
                 results_dir: Path = None) -> bool:
    """
    Wrapper to run the TabNet training pipeline and handle exceptions.

    Args:
        csv_path (Path): Path to the dataset.
        n_splits (int): Number of CV splits.
        n_trials (int): Number of Optuna trials.
        random_state (int): The random seed.

    Returns:
        bool: True on success, False on failure.
    """
    try:
        run_tabnet(
            csv_path=str(csv_path),
            n_splits=n_splits,
            n_trials=n_trials,
            random_state=random_state,
            results_dir=str(results_dir) if results_dir else None,
        )
        return True
    except Exception as e:
        print(f"❌ Error during TabNet training: {e}")
        return False


def train_catboost(args: argparse.Namespace) -> bool:
    """
    Wrapper to run the CatBoost training pipeline and handle exceptions.

    Args:
        args (argparse.Namespace): Arguments to pass to the training script.

    Returns:
        bool: True on success, False on failure.
    """
    try:
        train_model_catboost.main(args)
        return True
    except Exception as e:
        print(f"❌ Error during CatBoost training: {e}")
        return False


def train_federated_learning(args: argparse.Namespace) -> bool:
    """
    Wrapper to run the Federated Learning experiment suite and handle exceptions.

    Args:
        args (argparse.Namespace): Arguments including FL-specific parameters.

    Returns:
        bool: True on success, False on failure.
    """
    try:
        from federated import run_federated
        run_federated(
            csv_path=args.csv,
            results_dir=args.results_dir / "federated",
            num_rounds=args.fl_rounds,
            local_epochs=args.fl_local_epochs,
            batch_size=32,
            num_clients=args.fl_clients,
            strategies=args.fl_strategies,
            distributions=args.fl_distributions,
            model_types=args.fl_models,
            skip_centralized=args.skip_centralized,
            seed=args.random_state,
        )
        return True
    except Exception as e:
        print(f"Error during Federated Learning: {e}")
        import traceback
        traceback.print_exc()
        return False


def train_security_phase1a(args: argparse.Namespace) -> bool:
    """Run FL Security Phase 1a experiments."""
    try:
        from federated import run_security_phase1a
        run_security_phase1a(
            csv_path=args.csv,
            results_dir=args.results_dir / "federated",
            num_rounds=args.fl_rounds,
            local_epochs=args.fl_local_epochs,
            batch_size=32,
            model_types=args.fl_models,
            seed=args.random_state,
        )
        return True
    except Exception as e:
        print(f"Error during Security Phase 1a: {e}")
        import traceback
        traceback.print_exc()
        return False


def train_security_phase1b(args: argparse.Namespace) -> bool:
    """Run FL Security Phase 1b experiments."""
    try:
        from federated import run_security_phase1b
        run_security_phase1b(
            csv_path=args.csv,
            results_dir=args.results_dir / "federated",
            num_rounds=args.fl_rounds,
            local_epochs=args.fl_local_epochs,
            batch_size=32,
            model_types=args.fl_models,
            seed=args.random_state,
        )
        return True
    except Exception as e:
        print(f"Error during Security Phase 1b: {e}")
        import traceback
        traceback.print_exc()
        return False


def train_security_sensitivity_epochs(args: argparse.Namespace) -> bool:
    """Run FL Security Sensitivity Analysis experiments."""
    try:
        from federated import run_security_sensitivity_epochs
        run_security_sensitivity_epochs(
            csv_path=args.csv,
            results_dir=args.results_dir / "federated",
            num_rounds=args.fl_rounds,
            batch_size=32,
            model_types=args.fl_models,
            seed=args.random_state,
        )
        return True
    except Exception as e:
        print(f"Error during Security Sensitivity Analysis: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_comparison(results_dir: Path, models_trained: list):
    """
    Generates a comparison report for all successfully trained models.

    Args:
        results_dir (Path): The base directory where results are stored.
        models_trained (list): A list of names of the models that were trained.
    """
    if not COMPARE_AVAILABLE:
        print("\n⚠️  Comparison module not found. Skipping comparison step.")
        return

    if len(models_trained) < 2:
        print("\n⚠️  At least two models must be trained to generate a comparison. Skipping.")
        return

    print("\n" + "=" * 80)
    print("MODEL COMPARISON ANALYSIS")
    print("=" * 80)

    file_map = {
        "RandomForest": "random_forest/training_results_random_forest.json",
        "TabNet": "tabnet/training_results_tabnet.json",
        "CatBoost": "catboost/training_results_catboost.json",
    }
    result_files = [results_dir / file_map[model] for model in models_trained if model in file_map]

    if len(result_files) < 2:
        print(f"⚠️  Could not find at least two result files to compare. Found: {len(result_files)}")
        return

    try:
        # Calls the report generation function from the imported module.
        generate_report(
            results_dir=results_dir,
            files=result_files,
            output_plot=results_dir / "model_comparison.png",
            export_csv=results_dir / "model_comparison.csv"
        )
        print("✓ Comparison report generated successfully.")
    except Exception as e:
        print(f"❌ Error during comparison: {e}")


def main():
    """
    Main execution function to orchestrate the model training pipelines.
    """

    args = parse_arguments()
    start_time = time.time()

    # Resolve experiment ID and build the final results directory
    if args.experiment_id is None:
        args.experiment_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    args.results_dir = args.results_dir / args.experiment_id

    print("=" * 80)
    print("AIMS FRAMEWORK - MODEL TRAINING ORCHESTRATOR")
    print("=" * 80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Experiment ID: {args.experiment_id}")
    print(f"Results Dir: {args.results_dir}")
    print(f"Dataset: {args.csv}")
    print(f"Random State: {args.random_state}")
    print("-" * 80)

    if not validate_dataset(args.csv):
        sys.exit(1)

    # --federated-only implies --federated and skips all centralized models
    if args.federated_only:
        args.federated = True
        args.skip_rf = True
        args.skip_tabnet = True
        args.skip_catboost = True

    models_trained = []

    if args.test_comparison_only:
        print("🧪 TEST MODE: Skipping all model training to test comparison logic.")
        models_trained = ["RandomForest", "TabNet", "CatBoost"]  # Simulate successful training
    else:
        # Create common arguments object for RF and CatBoost trainers.
        rf_cb_args = argparse.Namespace(
            csv=args.csv,
            n_splits=args.n_splits,
            n_jobs=-1,  # Use all available CPU cores
            random_state=args.random_state,
            n_trials=args.n_trials,
            results_dir=args.results_dir,
        )

        # --- RandomForest Training ---
        if not args.skip_rf:
            print(f"\n[1/3] Training Random Forest Classifier...")
            print("-" * 60)
            if train_random_forest(rf_cb_args):
                print("✓ Random Forest training completed.")
                models_trained.append("RandomForest")
            else:
                print("⚠️  Random Forest training failed.")
        else:
            print("\n[1/3] Skipping Random Forest (--skip-rf flag).")

        # --- TabNet Training ---
        if not args.skip_tabnet:
            print(f"\n[2/3] Training TabNet Classifier...")
            print("-" * 60)
            if train_tabnet(args.csv, args.n_splits, args.n_trials_tabnet, args.random_state,
                           results_dir=args.results_dir):
                print("✓ TabNet training completed.")
                models_trained.append("TabNet")
            else:
                print("⚠️  TabNet training failed.")
        else:
            print("\n[2/3] Skipping TabNet (--skip-tabnet flag).")

        # --- CatBoost Training ---
        if not args.skip_catboost:
            print(f"\n[3/3] Training CatBoost Classifier...")
            print("-" * 60)
            if train_catboost(rf_cb_args):
                print("✓ CatBoost training completed.")
                models_trained.append("CatBoost")
            else:
                print("⚠️  CatBoost training failed.")
        else:
            print("\n[3/3] Skipping CatBoost (--skip-catboost flag).")

    # --- Federated Learning ---
    if args.federated:
        print(f"\n[FL] Training Federated Learning models...")
        print("-" * 60)
        if train_federated_learning(args):
            print("Federated Learning experiments completed.")
        else:
            print("Federated Learning experiments failed.")

    # --- FL Security Experiments ---
    if args.security:
        if not args.skip_phase1a:
            print(f"\n[Security Phase 1a] Running label-flip attack experiments...")
            print("-" * 60)
            if train_security_phase1a(args):
                print("Security Phase 1a experiments completed.")
            else:
                print("Security Phase 1a experiments failed.")
        else:
            print("\n[Security Phase 1a] Skipping (--skip-phase1a flag).")

        if not args.skip_phase1b:
            print(f"\n[Security Phase 1b] Running label-flip + gradient scaling experiments...")
            print("-" * 60)
            if train_security_phase1b(args):
                print("Security Phase 1b experiments completed.")
            else:
                print("Security Phase 1b experiments failed.")
        else:
            print("\n[Security Phase 1b] Skipping (--skip-phase1b flag).")

    if args.security_sensitivity_epochs:
        print(f"\n[Security Sensitivity] Running epoch sensitivity analysis...")
        print("-" * 60)
        if train_security_sensitivity_epochs(args):
            print("Security Sensitivity Analysis completed.")
        else:
            print("Security Sensitivity Analysis failed.")

    # --- Final Summary ---
    total_time = time.time() - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)

    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 80)
    print(
        f"Models successfully trained: {len(models_trained)}/3 ({', '.join(models_trained) if models_trained else 'None'})")
    print(f"Total execution time: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")
    print(f"Results have been saved in their respective directories under '{args.results_dir}/'")

    # --- Optional Comparison Step ---
    if (args.compare or args.test_comparison_only) and models_trained:
        run_comparison(args.results_dir, models_trained)

    print("=" * 80)
    print("Orchestration complete.")

    if not models_trained and not args.test_comparison_only:
        sys.exit(1)


if __name__ == '__main__':
    main()
