#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AIMS Framework - Main Training Orchestrator
===========================================

This script is the central entry point of the AIMS framework, orchestrating the end-to-end training,
hyperparameter optimization, and evaluation of multiple machine learning models for network slicing
impact classification in vehicular networks.

The pipeline supports:
1. Random Forest (RF) – Tree-based ensemble with bagging
2. TabNet – Deep learning with attention mechanisms for tabular data
3. CatBoost – Gradient boosting with native categorical feature support

All models are trained and evaluated using:
- GroupKFold cross-validation (5 splits) to prevent temporal leakage
- Hyperparameter optimization with Optuna
- Class-balanced weights to address imbalanced data
- Consistent train/test split for fair comparison
- Standardized results management via ResultsManager

Key Components
--------------
1. **Model Execution**: Runs the training pipelines for:
    - RandomForestClassifier
    - CatBoostClassifier
    - TabNetClassifier
2. **Argument Parsing**: Flexible command-line interface to:
    - Specify dataset path
    - Configure cross-validation and optimization trials
    - Selectively skip training of specific models
3. **Workflow Management**: Ensures reproducibility by using a consistent random state and dataset across all models.
    - *Dataset note*:  
      The dataset used for the AIMS framework was generated from the experimental setup described in:  
      T. do Vale Saraiva et al., "An Application-Driven Framework for Intelligent Transportation Systems Using 5G Network Slicing,"
      IEEE Transactions on Intelligent Transportation Systems, vol. 22, no. 8, pp. 5247–5260, Aug. 2021.  
      DOI: 10.1109/TITS.2021.3086064.  
      It was further pre-processed and engineered specifically for the experiments reported in AIMS.
4. **Results Comparison**: Optionally invokes the comparison module to generate a consolidated report of model results.

Usage Example
-------------
```bash
# Train all models with default settings
python main.py

# Specify dataset and number of trials
python main.py --csv ./data/my_dataset.csv --n-trials 50

# Skip a specific model and run comparison at the end
python main.py --skip-tabnet --compare
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
        "--test-comparison-only", action="store_true",
        help="Skip all training and only run the comparison logic.")

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


def train_tabnet(csv_path: Path, n_splits: int, n_trials: int, random_state: int) -> bool:
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
            random_state=random_state
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

    print("=" * 80)
    print("AIMS FRAMEWORK - MODEL TRAINING ORCHESTRATOR")
    print("=" * 80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dataset: {args.csv}")
    print(f"Random State: {args.random_state}")
    print("-" * 80)

    if not validate_dataset(args.csv):
        sys.exit(1)

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
            n_trials=args.n_trials
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
            if train_tabnet(args.csv, args.n_splits, args.n_trials_tabnet, args.random_state):
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
