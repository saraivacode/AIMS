#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AIMS Framework – CatBoost Training & Optimization Module
========================================================

End-to-end pipeline for **training, hyper-parameter tuning, and evaluating** a CatBoost classifier
inside the *AIMS* (Adaptive and Intelligent Management of Slicing) framework. The model predicts
the **impact level** of network-slicing policies on vehicular Intelligent Transportation Systems (ITS).

Key components
--------------
1. **Data ingestion & preparation** – domain-specific cleaning and feature engineering.
2. **Custom impact labelling** – weighted aggregation of latency, loss and throughput (`label_weighted_average`).
3. **Device-aware training** – automatic CPU / GPU selection (`get_device_params`).
4. **Group-aware validation & HPO** – `GroupShuffleSplit` hold-out + `GroupKFold` inside
Optuna (**TPE** sampler; no external pruner).
5. **Final training & hold-out evaluation** – macro F1, confusion matrix, feature importance.
6. **Artifact management** – model (`joblib`), metrics JSON, PNG/HTML plots, CSV importance
(handled by `ResultsManager` & `save_model_results`).

Usage example
-------------
```bash
python train_catboost.py \
    --csv ./data/aims_dataset.csv \
    --n-trials 40 \
    --n-splits 5 \
    --random-state 42
```

Author: Tiago do Vale Saraiva
License: MIT
Version: 1.0.0
"""

# =============================================================================
# AIMS Framework - CatBoost Training Module
# -----------------------------------------------------------------------------
# • Handles all necessary imports: standard, third-party, and project-specific.
# • Configures global logging, warning suppression, and plotting style.
# • Defines global default values for CatBoost-specific training parameters.
# =============================================================================

from __future__ import annotations

# -------------------------------------------------
# Standard-library imports
# -------------------------------------------------
import argparse
import sys
import logging
import time
from pathlib import Path
from typing import Any, Dict

# -------------------------------------------------
# Third-party imports
# -------------------------------------------------
import joblib
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import torch
import catboost
from catboost import CatBoostClassifier
from typing import cast
from matplotlib import pyplot as plt
from optuna.visualization import plot_optimization_history
from optuna.exceptions import ExperimentalWarning
import sklearn
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, train_test_split

# -------------------------------------------------
# Local application imports
# -------------------------------------------------
import preprocess_dataset as pp
from impact_labeling import WEIGHTS, label_weighted_average
from results_manager import ResultsManager
from save_utils import save_model_results

import warnings

# -------------------------------------------------
# Global Configuration
# -------------------------------------------------
# • Suppresses Optuna Experimental Warnings for a cleaner output.
# • Configures logging for progress tracking and debugging.
# • Sets up consistent matplotlib/seaborn style for all visualizations.
# -------------------------------------------------

# Suppress Optuna Experimental Warnings for a cleaner output.
warnings.filterwarnings("ignore", category=ExperimentalWarning)

# Configure logging to display progress and informational messages.
logging.basicConfig(
    level=logging.INFO,
    format='\n%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CatBoostTrainer")

# Consistent seaborn/matplotlib style across all scripts
plt.rcParams.update({
    "figure.figsize": (10, 6),             # Default figure size
    "axes.titlesize": 18,                  # Title font size
    "axes.labelsize": 14,                  # Axis label font size
    "xtick.labelsize": 12,                 # Tick font size
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "axes.grid": True,                     # Enable grid by default
    "grid.alpha": 0.5,                     # Grid transparency
    "axes.titlepad": 16,                   # Padding for axes title
    "axes.edgecolor": "#444444",           # Subtle axes edge
    "axes.linewidth": 1.2,
    "font.family": "DejaVu Sans",          # Font family
    "savefig.dpi": 300,                    # Figure export DPI
    "savefig.bbox": "tight",               # Tight bounding box when saving
    "figure.autolayout": True              # Adjust layout automatically
})
sns.set_theme(context="notebook", style="whitegrid", palette="viridis")

# -------------------------------------------------
# Global Defaults
# -------------------------------------------------
class GlobalDefaults:
    """Container for global default values."""
    # Paths
    DEFAULT_CSV_PATH = "../data/aims_dataset.csv"

    # Training parameters
    DEFAULT_RANDOM_STATE = 42
    DEFAULT_N_SPLITS = 5
    DEFAULT_N_TRIALS = 40
    DEFAULT_N_JOBS = -1

    # Model-specific defaults
    CATBOOST_ITERATIONS = 2000
    CATBOOST_EARLY_STOPPING = 50

    # Impact classification
    IMPACT_CLASSES = ["Adequate", "Warning", "Severe", "Critical"]

def build_arg_parser() -> argparse.ArgumentParser:
    """Constructs the command-line argument parser for the script.

    Defines arguments for the input dataset path, cross-validation folds,
    parallel processing jobs, random state for reproducibility, and the number
    of Optuna optimization trials.

    Returns:
        argparse.ArgumentParser: The configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Train and optimize a CatBoost classifier for the AIMS Framework.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--csv", type=Path, default=Path(GlobalDefaults.DEFAULT_CSV_PATH),
        help="Path to the CSV file with the vehicular QoS dataset.")

    parser.add_argument(
        "--n-splits", type=int, default=GlobalDefaults.DEFAULT_N_SPLITS,
        help="Number of folds for GroupKFold cross-validation.")

    parser.add_argument(
        "--n-jobs", type=int, default=GlobalDefaults.DEFAULT_N_JOBS,
        help="Number of parallel jobs (-1 = all cores).")

    parser.add_argument(
        "--random-state", type=int, default=GlobalDefaults.DEFAULT_RANDOM_STATE,
        help="Random seed for ensuring reproducibility.")
    parser.add_argument(
        "--n-trials", type=int, default=GlobalDefaults.DEFAULT_N_TRIALS,
        help="Number of hyperparameter optimization trials to run with Optuna.")
    return parser

def get_device_params() -> Dict[str, Any]:
    """Detects the available compute device and returns CatBoost parameters.

    Checks for a CUDA-enabled GPU. If available, it configures CatBoost to use it for accelerated training.
    Otherwise, it defaults to CPU.

    Returns:
        Dict[str, Any]: A dictionary containing 'task_type' and 'devices' parameters for the CatBoostClassifier.
    """
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        logger.info(f"CUDA-enabled GPU detected: {gpu_name} ({gpu_memory_gb:.1f} GB). Setting task_type to 'GPU'.")
        print(f"CUDA-enabled GPU detected: {gpu_name} ({gpu_memory_gb:.1f} GB). Setting task_type to 'GPU'.")
        return {"task_type": "GPU", "devices": "0"}

    logger.info("No GPU detected. Using CPU for training.")
    return {"task_type": "CPU"}

def objective_cb(trial: optuna.Trial, args: argparse.Namespace, X_train: pd.DataFrame,
                 y_train: pd.Series, groups_train: np.ndarray, base_cb_clf: CatBoostClassifier) -> float:
    """
    Optuna objective function for CatBoost hyperparameter optimization.

    This function explores the CatBoost hyperparameter space using group-aware cross-validation
    (GroupKFold) to find the configuration that maximizes the macro F1-score.

    Args:
        trial (optuna.Trial): The Optuna Trial instance, used to suggest parameters.
        args (argparse.Namespace): Namespace object with script arguments (e.g., n_splits).
        X_train (pd.DataFrame): DataFrame containing the training features.
        y_train (pd.Series): Series containing the training labels.
        groups_train (np.ndarray): Array with group identifiers for cross-validation.
        base_cb_clf (CatBoostClassifier): The base, untrained CatBoost model to be cloned.

    Returns:
        float: The mean cross-validation macro F1-score.
    """
    # Cross-validation strategy that respects groups
    gkf = GroupKFold(n_splits=args.n_splits)

    # 1. Hyperparameter search space
    params = {
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
        "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli"]),
    }
    if params["bootstrap_type"] == "Bayesian":
        params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.0, 1.0)
    else:
        params["subsample"] = trial.suggest_float("subsample", 0.6, 1.0, step=0.1)

    scores = []
    # 2. Group-aware cross-validation
    for tr_idx, va_idx in gkf.split(X_train, y_train, groups_train):
        X_tr, y_tr = X_train.iloc[tr_idx], y_train[tr_idx]
        X_va, y_va = X_train.iloc[va_idx], y_train[va_idx]

        # 3. Create the model for this trial by cloning the base model
        clf = cast(CatBoostClassifier, clone(base_cb_clf)).set_params(
            **params, verbose=False)  # Silence CatBoost console output

        # 4. Train the model with an evaluation set for early stopping
        clf.fit( X_tr, y_tr, eval_set=(X_va, y_va),
                 verbose=0) # Set to 0 to avoid polluting the log during HPO

        # 5. Evaluate and store the score
        preds = clf.predict(X_va)
        scores.append(f1_score(y_va, preds, average="macro"))

    # Store the individual scores in the trial for later analysis
    trial.set_user_attr("cv_scores", scores)

    # Return the mean of the scores, which Optuna will try to maximize
    return np.mean(scores)

def main(args: argparse.Namespace) -> None:
    """Executes the main CatBoost model training and evaluation workflow.

    This function orchestrates the entire process: data loading, preprocessing, impact labeling,
    data splitting, hyperparameter optimization, final model training, evaluation, and artifact saving.

    Args:
        args (argparse.Namespace): Command-line arguments provided to the script.
    """

    total_start_time = time.time()
    # ==================== Initialization ====================
    logger.info("Starting Catboost Training Pipeline...")
    print("\n" + "=" * 80)
    print("CATBOOST TRAINING PIPELINE - AIMS FRAMEWORK")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  - Dataset: {args.csv}")
    print(f"  - CV Folds: {args.n_splits}")
    print(f"  - Optuna Trials: {args.n_trials}")
    print(f"  - Random State: {args.random_state}")
    print(f"  - Parallel Jobs: {args.n_jobs}")
    print("=" * 80)

    # =============================================================================
    # Step 1 ─ Data Loading & Preprocessing
    # -----------------------------------------------------------------------------
    # • Validates that the CSV file exists.
    # • Loads raw data with pandas, then applies domain-specific cleaning via
    #   pp.prepare_dataset().
    # • Logs and prints the shapes of raw and processed data for traceability.
    # =============================================================================
    logger.info("Loading and preprocessing dataset from %s ...", args.csv)
    print("-" * 60)
    print("[Step 1/11] Loading and preprocessing dataset...")
    print("-" * 60)

    # 1) Sanity-check: file must exist
    if not args.csv.exists():
        logger.error(f"Dataset not found: {args.csv}")
        raise FileNotFoundError(f"Dataset not found: {args.csv}")

    # 2) Load and preprocess
    df_raw = pd.read_csv(args.csv)
    df = pp.prepare_dataset(df_raw)

    # 3) Quick sanity printouts
    print(f"✓ Raw dataset shape: {df_raw.shape}")
    print(f"✓ Preprocessed dataset shape: {df.shape}")

    # =============================================================================
    # Step 2 ─ Impact Labeling
    # -----------------------------------------------------------------------------
    # • Computes a weighted-average impact score from latency, loss, throughput.
    # • Maps each sample to one of four labels: Adequate | Warning | Severe | Critical.
    # • Returns the updated DataFrame plus train-ready artifacts.
    # =============================================================================
    logger.info("Applying Custom Weighted-Impact Labeling...")
    print("-" * 60)
    print("[Step 2/11] Applying Custom Weighted-Impact Labeling...")
    print("-" * 60)

    # 1) Copy default weights and (optionally) tweak per slice/app type
    # Examples shown but kept commented out for clarity.
    weights = WEIGHTS.copy()
    # weights.update({
    #     "s": dict(lat=0.5, loss=0.3, thr=0.2),
    #     "e": dict(lat=0.3, loss=0.4, thr=0.3),
    #     "e2": dict(lat=0.2, loss=0.3, thr=0.5),
    #     "g": dict(lat=0.3, loss=0.3, thr=0.4),
    # })

    # 2) Apply labeling function
    (
        df,  # updated DataFrame with new “impact” column
        X, y,  # features and target labels
        groups,  # grouping IDs for CV
        _,  # list with pre-calculated class weights to handle imbalance (not used by Catboost)
        class_weights_dict  # dict with pre-calculated class weights to handle imbalance.
    ) = label_weighted_average(df, weights=weights)

    # 3) Impact level distribution
    impact_stats = pd.concat([pd.Series(y).value_counts().sort_index(),
                              pd.Series(y).value_counts(normalize=True).sort_index()], axis=1)
    impact_stats.columns = ["count", "percent (%)"]
    impact_stats["percent (%)"] = (impact_stats["percent (%)"] * 100).round(2)
    print(f"✓ Feature matrix shape : {X.shape}")
    print(f"✓ Temporal groups      : {len(np.unique(groups))}")
    print("✓ Impact distribution :")
    print(impact_stats)

    # =============================================================================
    # Step 3 ─ Data Splitting (Temporal-Aware)
    # -----------------------------------------------------------------------------
    # • Uses GroupShuffleSplit to ensure that all samples from the same group
    #   (e.g., time_slot + approach) stay together in either the training or holdout set.
    # • Prevents data leakage across time-dependent groups.
    # • Identifies and processes categorical/numerical features for consistent handling.
    # =============================================================================
    logger.info("Splitting data into temporal-aware training and holdout sets...")
    print("-" * 60)
    print("[Step 3/11] Splitting data into temporal-aware training and holdout sets...")
    print("-" * 60)

    # 1) Identify feature types and enforce string type for categoricals
    num_cols = X.select_dtypes(include=[np.number, bool]).columns.tolist()
    cat_cols = [col for col in X.columns if col not in num_cols]
    for col in cat_cols:
        X[col] = X[col].astype(str)
    print(f"✓ Numerical features: {len(num_cols)}")
    print(f"✓ Categorical features: {len(cat_cols)}")

    # 2) Group-aware train/holdout split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.random_state)
    train_idx, hold_idx = next(gss.split(X, y, groups))
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_hold, y_hold = X.iloc[hold_idx], y[hold_idx]
    groups_train = groups[train_idx]
    print(f"✓ Training set : {len(train_idx)} samples ({len(train_idx) / len(X) * 100:.1f}%)")
    print(f"✓ Holdout set  : {len(hold_idx)} samples ({len(hold_idx) / len(X) * 100:.1f}%)")

    # =============================================================================
    # Step 4 ─ Setup Output Directory
    # -----------------------------------------------------------------------------
    # • Prepares the output directory for experiment artifacts and results.
    # • Ensures all outputs (e.g., model, logs, metrics, plots) are saved in a
    #   structured location for reproducibility.
    # • Instantiates the ResultsManager to manage current model's outputs.
    # =============================================================================
    logger.info("Setting up output directory...")
    print("-" * 60)
    print("[Step 4/11] Setting up output directory...")
    print("-" * 60)

    # 1) Save artifacts and get the output directory for results
    output_dir = save_model_results('catboost', X, y, groups, class_weights_dict)

    # 2) Instantiate ResultsManager for handling results and artifacts
    rm = ResultsManager("CatBoost", output_dir)

    # =============================================================================
    # Step 5 ─ Base Model Creation (CatBoost)
    # -----------------------------------------------------------------------------
    # • Detects and logs the device type (GPU/CPU) for CatBoost training.
    # • Instantiates a base CatBoostClassifier with class weights for imbalance.
    # • Passes explicit categorical feature indices for optimized handling.
    # • Sets high iterations and early stopping for robust hyperparameter tuning.
    # • Optimizes for Macro F1-score, preserving the best model per trial.
    # =============================================================================
    logger.info("Creating base model...")
    print("-" * 60)
    print("[Step 5/11] Creating base model...")
    print("-" * 60)

    # 1) Check device type (GPU/CPU) and log
    device_params = get_device_params()
    if device_params["task_type"] == "GPU":
        print(f"✓ Using GPU acceleration")
    else:
        print(f"✓ Using CPU (no GPU detected)")

    # 2) Base CatBoost classifier configuration.
    #    This object serves as a template for each Optuna trial.
    base_cb_clf = CatBoostClassifier(
        loss_function="MultiClass",
        random_seed=args.random_state,
        verbose=False,
        class_weights=class_weights_dict,   # Use pre-calculated class weights to handle imbalance.
        cat_features=cat_cols,              # Explicitly pass categorical features.
        iterations=GlobalDefaults.CATBOOST_ITERATIONS,  # High iteration count, controlled by early stopping.
        early_stopping_rounds=GlobalDefaults.CATBOOST_EARLY_STOPPING,  # Stop if no improvement after N rounds.
        use_best_model=True,                # Keep the model from the best iteration.
        eval_metric="TotalF1:average=Macro", # Optimize for Macro F1-score.
        thread_count=args.n_jobs,
        **device_params
    )

    # =============================================================================
    # Step 6 ─ Optuna Hyperparameter Optimization (HPO) Configuration and Execution (CatBoost)
    # -----------------------------------------------------------------------------
    # • Sets up and runs Optuna for CatBoost hyperparameter optimization.
    # • Uses a TPE sampler for efficient and robust search.
    # • Passes a custom objective function, leveraging grouped cross-validation.
    # • Logs and prints progress, best parameters, and cross-validation performance.
    # =============================================================================
    logger.info("Configure and Run Hyperparameter Optimization (HPO) with Optuna ...")
    print("-" * 60)
    print("[Step 6/11] Configure and Run Hyperparameter Optimization (HPO) with Optuna ...")
    print("-" * 60)

    # 1) Configure Optuna sampler and study.
    sampler = optuna.samplers.TPESampler(seed=args.random_state, n_startup_trials=10,
                                         n_ei_candidates=24, multivariate=True)
    study_cb = optuna.create_study(direction="maximize", sampler=sampler, pruner=None)

    # 2) Define objective function and run Optuna optimization.
    logger.info(f"Starting Optuna optimization with {args.n_trials} trials...")
    print(f"Starting Optuna optimization with {args.n_trials} trials...")
    opt_start_time = time.time()
    study_cb.optimize(
        lambda trial: objective_cb(trial, args, X_train, y_train, groups_train, base_cb_clf),
        n_trials=args.n_trials, show_progress_bar=True, gc_after_trial=True)
    opt_time = time.time() - opt_start_time  # Track optimization time
    logger.info(f"✓ Optimization completed in {opt_time:.1f} seconds!")
    print(f"✓ Optimization completed in {opt_time:.1f} seconds!")

    # 3) Log and display best hyperparameters and performance.
    best_params = study_cb.best_params
    print(f"Best trial: {study_cb.best_trial.number}")
    print(f"Best parameters found:")
    for param, value in best_params.items():
        print(f"  - {param}: {value}")
    best_cv_scores = study_cb.best_trial.user_attrs.get("cv_scores", [])
    logger.info(f"Best F1-macro (CV): {study_cb.best_value:.4f} (±{np.std(best_cv_scores):.4f})")
    print(f"\nBest F1-macro (CV): {study_cb.best_value:.4f} (±{np.std(best_cv_scores):.4f})")

    # =============================================================================
    # Step 7 ─ Final Model Training (CatBoost)
    # -----------------------------------------------------------------------------
    # • Splits the training data to create a small internal validation set (10%).
    # • Trains the final CatBoost model with the best hyperparameters, using early stopping.
    # • Saves the trained model artifact and prints/logs key configuration details.
    # =============================================================================
    logger.info("Final model training with best hyperparameters...")
    print("-" * 60)
    print("[Step 7/11] Final model training with best hyperparameters...")
    print("-" * 60)

    # 1) Data split and final classifier creation with best params.
    #    Split 10% of the *training* data for internal validation.
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train,test_size=0.10,
                                                random_state=args.random_state, stratify=y_train)
    final_cb = cast(CatBoostClassifier, clone(base_cb_clf)).set_params(**best_params, verbose=False,
                                                                       metric_period=2) # Log every 2 iterations

    # 2) Final fitting.
    fit_start = time.time()
    final_cb.fit(X_tr, y_tr, eval_set=(X_val, y_val), # Enables early stopping & best-iteration selection
                 verbose=2)
    fit_time = time.time() - fit_start
    logger.info(f"✓ Final model fitting completed in {fit_time:.2f} seconds!")
    print(f"✓ Final model fitting completed in {fit_time:.2f} seconds!")

    # 3) Save the trained model object and print training results.
    model_path = output_dir / "catboost_model.pkl"
    joblib.dump(final_cb, model_path)
    logger.info(f"✓ Final model saved to: {model_path}")
    print(f"✓ Final model saved to: {model_path}")
    print(f"✓ Best iteration: {final_cb.get_best_iteration()}")
    print(f"✓ Number of trees: {final_cb.tree_count_}")
    print(f"✓ Max depth: {final_cb.get_param('depth')}")
    print(f"✓ Best epoch: {getattr(final_cb, 'best_epoch', -1)}")

    # =============================================================================
    # Step 8 ─ Evaluating Final Model on the Holdout Set (Catboost)
    # -----------------------------------------------------------------------------
    # • Generates predictions for the holdout set using the final trained TabNet model.
    # • Computes and prints detailed classification metrics and key performance indicators.
    # =============================================================================
    logger.info("Evaluating final model on the holdout set")
    print("-" * 60)
    print("[Step 8/11] Evaluating final model on the holdout set")
    print("-" * 60)

    class_names = GlobalDefaults.IMPACT_CLASSES

    # 1) Generate predictions for the holdout set.
    y_pred = final_cb.predict(X_hold)

    # 2) Print classification report.
    print("\nClassification Report:")
    print(classification_report(y_hold, y_pred, target_names=class_names))

    # 3) Print key performance indicators.
    report_dict = classification_report(y_hold, y_pred, target_names=class_names, output_dict=True)
    print("\nKey Performance Indicators:")
    print(f"  • Overall Accuracy: {report_dict['accuracy']:.1%}")
    print(f"  • Macro F1-Score: {report_dict['macro avg']['f1-score']:.1%}")
    print(f"  • Weighted F1-Score: {report_dict['weighted avg']['f1-score']:.1%}\n")

    # =============================================================================
    # Step 9 ─ Save Results: CSV and JSON Files (CatBoost)
    # -----------------------------------------------------------------------------
    # • Exports CatBoost feature importance to CSV for interpretability.
    # • Saves all experiment data, metrics, and configurations using ResultsManager to a JSON file.
    # =============================================================================
    logger.info("Save Results: CSV and JSON Files...")
    print("-" * 60)
    print("[Step 9/11] Save Results: CSV and JSON Files...")
    print("-" * 60)

    # 1) Calculate and Save Raw Feature Importance to CSV.
    importance_df = pd.DataFrame({"feature": X.columns, "importance": final_cb.get_feature_importance()
                                  }).sort_values(by="importance", ascending=False)
    feature_importance_path = output_dir / "feature_importance_catboost.csv"
    importance_df.to_csv(feature_importance_path, index=False)
    print(f"✓ Feature importance saved to: {feature_importance_path}")

    # 2) Collect all experiment results and configuration using ResultsManager.
    print("Saving comprehensive results...")
    rm.set_data_info(X_train, y_train, X_hold, y_hold, groups_train, "GroupShuffleSplit 80/20", args.random_state)
    rm.set_best_params(best_params)
    rm.set_cv_metrics(study_cb.best_trial.user_attrs.get("cv_scores", study_cb.best_value), n_folds=args.n_splits)
    rm.set_holdout_metrics(y_hold, y_pred, class_names=class_names)
    rm.set_hpo_stats(study_cb, optimization_time=opt_time, sampler="TPESampler", pruner="None")
    rm.set_training_stats(fit_time, device=device_params["task_type"], best_epoch=final_cb.get_best_iteration())
    rm.add_custom_metrics(
        catboost_info={"model_info": {"early_stopping_rounds": GlobalDefaults.CATBOOST_EARLY_STOPPING}})

    # 3) Save all collected results to a JSON file.
    rm.save("training_results_catboost.json")

    # =============================================================================
    # Step 10 ─ Visualizations (CatBoost)
    # -----------------------------------------------------------------------------
    # • Generates and saves confusion matrix plots (raw and normalized) for model diagnostics.
    # • Exports Optuna optimization history, feature importance (barplot/cumulative), and training curves.
    # • Ensures all artifacts are stored in the output directory for reproducibility and reporting.
    # =============================================================================
    logger.info("Generating and saving visualizations...")
    print("-" * 60)
    print("[Step 10/11] Generating and saving visualizations...")
    print("-" * 60)

    cm_labels = GlobalDefaults.IMPACT_CLASSES

    # 1) Generate and save both raw and normalized confusion matrix figures
    cm = confusion_matrix(y_hold, y_pred)  # Raw confusion matrix
    cm_norm = confusion_matrix(y_hold, y_pred, normalize="true") # Normalized confusion matrix
    for matrix, suffix in zip([cm, cm_norm], ["raw", "normalized"]):
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(matrix, annot=True, fmt=".2f" if "norm" in suffix else "d", cmap="Blues",
                    xticklabels=cm_labels, yticklabels=cm_labels, ax=ax)
        ax.set_title(f"CatBoost Confusion Matrix ({suffix.capitalize()})")
        ax.set_xlabel("Predicted Impact Level"), ax.set_ylabel("True Impact Level"), plt.tight_layout()
        plt.savefig(output_dir / f"confusion_matrix_cb_{suffix}.png"), plt.close(fig)
        print(f"✓ Confusion matrix plot saved: {output_dir / f'confusion_matrix_cb_{suffix}.png'}")

    # 2) Save Optuna optimization history as HTML
    fig_optuna = plot_optimization_history(study_cb)
    fig_optuna.write_html(output_dir / "optuna_history_catboost.html")
    print(f"✓ Optuna optimization history saved: {output_dir / 'optuna_history_catboost.html'}")

    # 3) Feature importance barplot and cumulative figures
    print("Generating feature importance barplot...")
    top_n = 20  # Number of top features to plot
    plot_df = importance_df.head(top_n)
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = sns.color_palette("viridis", len(plot_df))
    try:
        sns.barplot(x="importance", y="feature", data=plot_df, palette=colors, hue="feature", legend=False, ax=ax)
    except TypeError:
        sns.barplot(x="importance", y="feature", data=plot_df, palette=colors, ax=ax)  # Fallback for seaborn < 0.14
    ax.set_title(f"Top {top_n} Feature Importance (CatBoost)"), ax.set_xlabel("Importance Score")
    ax.set_ylabel("Feature"), plt.tight_layout()
    plt.savefig(output_dir / "feature_importance_catboost.png", dpi=300), plt.close(fig)
    print(f"✓ Feature importance barplot saved: {output_dir / 'feature_importance_catboost.png'}")

    # --- Cumulative Feature Importance Plot (PNG) ---
    print("Generating cumulative feature importance plot...")
    importance_df['cum_importance'] = (importance_df['importance'].cumsum() / importance_df['importance'].sum()) * 100
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(importance_df) + 1), importance_df['cum_importance'], marker='o', markersize=4)
    ax.set_title('Cumulative Feature Importance (CatBoost)'), ax.set_xlabel('Number of Features')
    ax.set_ylabel('Cumulative Importance (%)'), ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    for perc in [90, 95, 99]:
        ax.axhline(y=perc, color='r', linestyle='--', linewidth=0.8, label=f'{perc}% Threshold')
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys()), plt.tight_layout()
    plt.savefig(output_dir / "feature_importance_cumulative_catboost.png", dpi=300), plt.close(fig)
    print(f"✓ Cumulative feature importance plot saved: {output_dir / 'feature_importance_cumulative_catboost.png'}")

    # 4) Final CatBoost Training Curves (Loss/Accuracy).
    print("Generating Final CatBoost Training Curves (Loss/Accuracy)...")
    evals_result = final_cb.get_evals_result()
    learn_loss = evals_result['learn'].get('MultiClass', [])
    valid_loss = evals_result['validation'].get('MultiClass', [])
    learn_f1 = evals_result['learn'].get('TotalF1:average=Macro', [])
    valid_f1 = evals_result['validation'].get('TotalF1:average=Macro', [])
    epochs_learn_loss = range(1, len(learn_loss) + 1)
    epochs_valid_loss = range(1, len(valid_loss) + 1)
    epochs_learn_f1 = range(1, len(learn_f1) + 1)
    epochs_valid_f1 = range(1, len(valid_f1) + 1)

    # Create the figure and axes at once. 1 row, 2 columns. Do not share the Y-axis between plots.
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), sharey=False)
    # Plot the Loss curve on the first axis (ax1)
    ax1.set_title("CatBoost Training & Validation Loss"), ax1.set_xlabel("Iteration"), ax1.set_ylabel("Loss")
    if len(learn_loss) > 0:
        ax1.plot(epochs_learn_loss, learn_loss, label="Train Loss")
    if len(valid_loss) > 0:
        ax1.plot(epochs_valid_loss, valid_loss, label="Valid Loss")
    ax1.legend(), ax1.grid(True, linestyle='--', alpha=0.6)
    # Plot the F1-macro curve on the second axis (ax2)
    ax2.set_title("CatBoost Training & Validation F1-macro"), ax2.set_xlabel("Iteration"), ax2.set_ylabel("F1-macro")
    if len(learn_f1) > 0 and len(valid_f1) > 0:
        ax2.plot(epochs_learn_f1, learn_f1, label="Train F1-macro")
        ax2.plot(epochs_valid_f1, valid_f1, label="Valid F1-macro")
        ax2.legend(), ax2.grid(True, linestyle='--', alpha=0.6)
    else:
        ax2.text(0.5, 0.5, "No F1-macro history available.", horizontalalignment='center', verticalalignment='center',
            transform=ax2.transAxes)
        ax2.set_xticks([]), ax2.set_yticks([]) # Remove axis ticks
    plt.tight_layout(), plt.savefig(output_dir / "catboost_training_curves.png", dpi=300), plt.close(fig)
    print(f"✓ Final CatBoost Training & Validation Curves Plots Saved: {output_dir / 'catboost_training_curves.png'}")

    print("\n")
    logger.info(f"All CATBOOST visualizations and results saved to: {output_dir}")
    print(f"✓ All CATBOOST visualizations and results saved to: {output_dir}")

    # =============================================================================
    # Step 11 ─ Final Summary (CatBoost)
    # -----------------------------------------------------------------------------
    # • Logs and prints a summary of the entire CatBoost experiment in AIMS.
    # • Reports total execution time, save location, and model performance metrics.
    # • Calls ResultsManager.get_summary() to generate a standardized human-readable summary.
    # =============================================================================
    total_time = time.time() - total_start_time
    logger.info(f"CATBOOST COMPLETED! Total execution time: {total_time / 60:.1f} minutes")
    logger.info(f"All results saved to: {output_dir}. Generating summary...")
    print("-" * 60)
    print(f"[Step 11/11] CATBOOST COMPLETED! Total execution time: {total_time / 60:.1f} minutes")
    print(f"All results saved to: {output_dir}. Generating summary...")
    print("-" * 60)
    print(rm.get_summary())

    # Example output:
    # =============================================================================
    # CATBOOST TRAINING SUMMARY
    # =============================================================================
    # Model Library: catboost v1.2.8
    # Data: 4576 train / 824 holdout
    # CV Score (Macro F1): 0.9736
    # Holdout Accuracy: 0.9405
    # Holdout Macro F1: 0.9409
    # Total Run Time: 57.6s
    # =============================================================================

if __name__ == "__main__":
    # Entry point of the script.
    arg_parser = build_arg_parser()
    script_args = arg_parser.parse_args()

    # Log system information
    logger.info(f"Python version: {sys.version}")
    logger.info(f"NumPy version: {np.__version__}")
    logger.info(f"Pandas version: {pd.__version__}")
    logger.info(f"Scikit-learn version: {sklearn.__version__}")
    logger.info(f"Optuna version: {optuna.__version__}")
    logger.info(f"CatBoost version: {catboost.__version__}")
    logger.info(f"PyTorch version: {torch.__version__}")

    try:
        main(script_args)
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise
