#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AIMS Framework – Random Forest Training & Optimization Module
=============================================================

End-to-end pipeline for **training, hyperparameter tuning, and evaluating** a Random Forest classifier
inside the *AIMS* (Adaptive and Intelligent Management of Slicing) framework. The model predicts the
**impact level** of network-slicing policies on vehicular Intelligent Transportation Systems (ITS).

Key components
--------------
1. **Data ingestion & preparation** – domain-specific cleaning and feature engineering.
2. **Custom impact labelling** – weighted aggregation of latency, loss and throughput (`label_weighted_average`).
3. **Pre-processing pipeline** – `StandardScaler` for numerical features, `OneHotEncoder`
for categorical ones (via `ColumnTransformer`).
4. **Group-aware validation & HPO** – `GroupShuffleSplit` hold-out + `GroupKFold` inside
Optuna (**TPE** sampler; no external pruner).
5. **Final training & hold-out evaluation** – macro F1, confusion matrix, feature importance.
6. **Artifact management** – model (`joblib`), metrics JSON, PNG/HTML plots, CSV importance (handled by
`ResultsManager` & `save_model_results`).

Usage example
-------------
```bash
python train_random_forest.py \
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
# AIMS Framework - RandomForest Training Module
# -----------------------------------------------------------------------------
# • Imports standard libraries, third-party packages, and project modules.
# • Configures global logging, plotting, and warning behaviors for reproducibility.
# • Defines global defaults for data paths, training parameters, and model settings.
# =============================================================================

from __future__ import annotations

# -------------------------------------------------
# Standard-library imports
# -------------------------------------------------
import argparse
import logging
import sys
import time
import warnings
from pathlib import Path

# -------------------------------------------------
# Third-party imports
# -------------------------------------------------
import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from optuna.visualization import plot_optimization_history
from optuna.exceptions import ExperimentalWarning
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from typing import cast
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# -------------------------------------------------
# Local application imports
# -------------------------------------------------
import preprocess_dataset as pp
from impact_labeling import WEIGHTS, label_weighted_average
from results_manager import ResultsManager
from save_utils import save_model_results

# -------------------------------------------------
# Global Configuration
# -------------------------------------------------
# • Suppresses specific warnings (e.g., Optuna ExperimentalWarning).
# • Sets up logging for progress tracking and debugging.
# • Applies a unified matplotlib/seaborn style for all plots.

# Suppress Optuna Experimental Warnings
warnings.filterwarnings("ignore", category=ExperimentalWarning)

# Configure logging to display progress and informational messages.
logging.basicConfig(level=logging.INFO, format='\n%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RandomForestTrainer")

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
# Set a consistent seaborn theme and palette
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
    RF_N_ESTIMATORS_RANGE = (100, 800)
    RF_MAX_DEPTH_OPTIONS = [None, 10, 20, 30]

    # Impact classification
    IMPACT_CLASSES = ["Adequate", "Warning", "Severe", "Critical"]

def build_arg_parser() -> argparse.ArgumentParser:
    """Constructs the command-line argument parser with global defaults.

    This version uses global defaults but allows full override via CLI args.
    """
    parser = argparse.ArgumentParser(
        description="Train and optimize a Random Forest classifier for AIMS impact prediction.",
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
        help="Random seed for ensuring reproducibility across runs.")

    parser.add_argument(
        "--n-trials", type=int, default=GlobalDefaults.DEFAULT_N_TRIALS,
        help="Number of hyperparameter optimization trials to run with Optuna.")

    return parser


def objective_rf(trial: optuna.Trial, args: argparse.Namespace, X_train: pd.DataFrame,
                 y_train: pd.Series, groups_train: np.ndarray, base_pipeline: Pipeline) -> float:
    """
    Optuna objective function for Random Forest hyperparameter tuning.

    This function explores the Random Forest hyperparameter space using group-aware
    cross-validation to find the configuration that maximizes the macro F1-score.

    Args:
        trial (optuna.Trial): The Optuna Trial instance, used to suggest parameters.
        args (argparse.Namespace): Namespace object with script arguments (e.g., n_splits).
        X_train (pd.DataFrame): DataFrame containing the training features.
        y_train (pd.Series): Series containing the training labels.
        groups_train (np.ndarray): Array with group identifiers for cross-validation.
        base_pipeline (Pipeline): The base, untrained scikit-learn Pipeline to be cloned.

    Returns:
        float: The mean cross-validation macro F1-score.
    """
    # GroupKFold is used during HPO to ensure groups are not split across folds.
    gkf = GroupKFold(n_splits=args.n_splits)

    # Define the hyperparameter search space.
    params = {
        'classifier__n_estimators': trial.suggest_int("n_estimators", *GlobalDefaults.RF_N_ESTIMATORS_RANGE, step=100),
        'classifier__max_depth': trial.suggest_categorical("max_depth", GlobalDefaults.RF_MAX_DEPTH_OPTIONS),
        'classifier__min_samples_leaf': trial.suggest_int("min_samples_leaf", 1, 5),
        'classifier__max_features': trial.suggest_categorical("max_features", ["sqrt", 0.5, 0.8])}

    # Perform cross-validation to get a robust estimate of the parameters' performance.
    scores = []
    for tr_idx, va_idx in gkf.split(X_train, y_train, groups_train):
        X_tr, y_tr = X_train.iloc[tr_idx], y_train[tr_idx]
        X_va, y_va = X_train.iloc[va_idx], y_train[va_idx]
        # Clone the base pipeline and set the new hyperparameters for this trial
        pipe = cast(RandomForestClassifier, clone(base_pipeline)).set_params(**params)
        pipe.fit(X_tr, y_tr)
        preds = pipe.predict(X_va)
        scores.append(f1_score(y_va, preds, average="macro"))

    # Store the individual fold scores in the trial for later analysis
    trial.set_user_attr("cv_scores", scores)

    # Return the mean of the scores for Optuna to optimize
    return np.mean(scores)


def main(args: argparse.Namespace) -> None:
    """Executes the complete Random Forest training and evaluation pipeline.

    This function orchestrates all stages of the workflow, from initial data loading to the
    final saving of results and visualizations.

    Args:
        args (argparse.Namespace): An object containing the parsed command-line arguments.
    """

    total_start_time = time.time()
    # ==================== Initialization ====================
    logger.info("Starting Random Forest Training Pipeline...")
    print("\n" + "=" * 80)
    print("RANDOM FOREST TRAINING PIPELINE - AIMS FRAMEWORK")
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
        _,  # list with pre-calculated class weights to handle imbalance (not used by RF)
        class_weights_dict  # dict with pre-calculated class weights to handle imbalance.
    )  = label_weighted_average(df, weights=weights)

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
    base_path = str(getattr(args, 'results_dir', '../results'))
    output_dir = save_model_results('random_forest', X, y, groups, class_weights_dict,
                                    base_path=base_path)

    # 2) Instantiate ResultsManager for handling results and artifacts
    rm = ResultsManager("RandomForest", output_dir)

    # =============================================================================
    # Step 5 ─ Pipeline for Base Model Creation (Random Forest)
    # -----------------------------------------------------------------------------
    # • Constructs a pipeline that combines preprocessing and model training steps.
    # • Ensures standardized, reproducible transformations (no data leakage).
    # • Applies class weights to address class imbalance.
    # • Outputs a scikit-learn Pipeline object ready for training and hyperparameter tuning.
    # =============================================================================
    logger.info("Creating base model pipeline...")
    print("-" * 60)
    print("[Step 5/11] Creating base model pipeline...")
    print("-" * 60)

    # 1) Create a ColumnTransformer to apply different transformations to different columns:
    #    - StandardScaler: Standardizes numerical features (mean=0, variance=1)
    #    - OneHotEncoder: Converts categorical features into a numerical format
    preprocessor = ColumnTransformer(transformers=[("num", StandardScaler(), num_cols),
                                    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),],
                                     # Keep any other columns if present. Output is a DataFrame preserving column names.
                                     remainder="passthrough").set_output(transform="pandas")

    # 2) Combine preprocessor and classifier into a single pipeline
    pipe_rf = Pipeline([("preprocessor", preprocessor), ("classifier", RandomForestClassifier(
        # Apply weights to counter class imbalance.
        class_weight=class_weights_dict, random_state=args.random_state, n_jobs=args.n_jobs,))])
    print("✓ Base model pipeline created: Preprocessor → RandomForestClassifier")

    # =============================================================================
    # Step 6 ─ Optuna Hyperparameter Optimization (HPO) Configuration and Execution (Random Forest)
    # -----------------------------------------------------------------------------
    # • Sets up and runs Optuna for Random Forest hyperparameter optimization.
    # • Uses a TPE sampler for efficient hyperparameter search.
    # • Applies a custom objective function using grouped cross-validation.
    # • Logs and prints progress, best parameters, and cross-validation performance.
    # =============================================================================
    logger.info("Configure and Run Hyperparameter Optimization (HPO) with Optuna ...")
    print("-" * 60)
    print("[Step 6/11] Configure and Run Hyperparameter Optimization (HPO) with Optuna ...")
    print("-" * 60)

    # 1) Configure Optuna sampler and study.
    sampler = optuna.samplers.TPESampler(seed=args.random_state, n_startup_trials=10,
                                         n_ei_candidates=24, multivariate=True)
    study_rf = optuna.create_study(direction="maximize", sampler=sampler, pruner=None)

    # 2) Define objective function and run Optuna optimization.
    logger.info(f"Starting Optuna optimization with {args.n_trials} trials...")
    print(f"Starting Optuna optimization with {args.n_trials} trials...")
    opt_start_time = time.time()
    study_rf.optimize(lambda trial: objective_rf(trial, args, X_train, y_train, groups_train, pipe_rf),
                      n_trials=args.n_trials, show_progress_bar=True, gc_after_trial=True)
    opt_time = time.time() - opt_start_time # Track optimization time
    logger.info(f"✓ Optimization completed in {opt_time:.1f} seconds!")
    print(f"✓ Optimization completed in {opt_time:.1f} seconds!")

    # 3) Log and display best hyperparameters and performance.
    print(f"Best trial: {study_rf.best_trial.number}")
    print(f"Best parameters found:")
    for param, value in study_rf.best_params.items():
        print(f"  - {param}: {value}")
    best_cv_scores = study_rf.best_trial.user_attrs.get("cv_scores", [])
    logger.info(f"Best F1-macro (CV): {study_rf.best_value:.4f} (±{np.std(best_cv_scores):.4f})")
    print(f"\nBest F1-macro (CV): {study_rf.best_value:.4f} (±{np.std(best_cv_scores):.4f})")

    # =============================================================================
    # Step 7 ─ Final Model Training (Random Forest)
    # -----------------------------------------------------------------------------
    # • Applies the best hyperparameters from Optuna to the pipeline.
    # • Trains the final Random Forest model on the entire training set (no validation split needed).
    # • Saves the trained model artifact and logs key configuration details.
    # =============================================================================
    logger.info("Final model training with best hyperparameters...")
    print("-" * 60)
    print("[Step 7/11] Final model training with best hyperparameters...")
    print("-" * 60)

    # 1) Apply best hyperparameters and instantiate final pipeline.
    best_params_rf = {f"classifier__{k}": v for k, v in study_rf.best_params.items()}
    final_pipe_rf = clone(pipe_rf).set_params(**best_params_rf)  # type: ignore[assignment]

    # Note: Unlike CatBoost and TabNet, Random Forest does not require a validation set
    # for early stopping because it's not an iterative algorithm. Each tree in the forest
    # is built independently to its full depth (or until other stopping criteria are met).
    # Therefore, we use 100% of the training data to maximize the model's learning capacity.
    # This is the standard and optimal approach for Random Forest models.

    # 2) Fit the final model on the entire training data.
    logger.info(f"Starting final fitting...")
    print(f"Starting final fitting...")
    fit_start_time = time.time()
    final_pipe_rf.fit(X_train, y_train)  # Uses full training set (correct!)
    fit_time = time.time() - fit_start_time
    logger.info(f"✓ Final model fitting completed in {fit_time:.2f} seconds!")
    print(f"✓ Final model fitting completed in {fit_time:.2f} seconds!")

    # 3) Save the trained model and print key training configuration.
    model_path = output_dir / "random_forest_model.pkl"
    joblib.dump(final_pipe_rf, model_path)
    logger.info(f"✓ Final model saved to: {model_path}")
    print(f"✓ Final model saved to: {model_path}")
    print(f"✓ Number of trees: {final_pipe_rf.named_steps['classifier'].n_estimators}")
    print(f"✓ Max depth: {final_pipe_rf.named_steps['classifier'].max_depth}")

    # =============================================================================
    # Step 8 ─ Evaluating Final Model on the Holdout Set (Random Forest)
    # -----------------------------------------------------------------------------
    # • Generates predictions for the holdout set using the final trained TabNet model.
    # • Computes and prints detailed classification metrics and key performance indicators.
    # =============================================================================
    logger.info("Evaluating final model on the holdout set and saving results...")
    print("-" * 60)
    print("[Step 8/11] Evaluating final model on the holdout set and saving results...")
    print("-" * 60)

    class_names = GlobalDefaults.IMPACT_CLASSES

    # 1) Generate predictions for the holdout set.
    y_pred_rf = final_pipe_rf.predict(X_hold)

    # 2) Print classification report.
    print("\nClassification Report:")
    print(classification_report(y_hold, y_pred_rf, target_names=class_names))

    # 3) Print key performance indicators.
    report_dict = classification_report(y_hold, y_pred_rf, target_names=class_names, output_dict=True)
    print("\nKey Performance Indicators:")
    print(f"  • Overall Accuracy: {report_dict['accuracy']:.1%}")
    print(f"  • Macro F1-Score: {report_dict['macro avg']['f1-score']:.1%}")
    print(f"  • Weighted F1-Score: {report_dict['weighted avg']['f1-score']:.1%}\n")

    # =============================================================================
    # Step 9 ─ Save Results: CSV and JSON Files (Random Forest)
    # -----------------------------------------------------------------------------
    # • Exports Random Forest feature importance to a CSV file.
    # • Saves all experiment data, metrics, and configurations using ResultsManager to a JSON file.
    # =============================================================================
    logger.info("Save Results: CSV and JSON Files...")
    print("-" * 60)
    print("[Step 9/11] Save Results: CSV and JSON Files...")
    print("-" * 60)

    # 1) Calculate and Save Raw Feature Importance to CSV.
    print("\nExtracting feature importance...")
    feature_names = final_pipe_rf.named_steps['preprocessor'].get_feature_names_out()
    importances = final_pipe_rf.named_steps['classifier'].feature_importances_
    importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances
                                  }).sort_values('importance', ascending=False)
    feature_importance_path = output_dir / "feature_importances_rf.csv"
    importance_df.to_csv(feature_importance_path, index=False)
    print(f"✓ Feature importance saved to: {feature_importance_path}")

    # 2) Collect all experiment results and configuration using ResultsManager.
    print("Saving comprehensive results...")
    rm.set_data_info(X_train, y_train, X_hold, y_hold, groups_train, "GroupShuffleSplit 80/20", args.random_state)
    rm.set_best_params(study_rf.best_params)
    rm.set_cv_metrics(study_rf.best_trial.user_attrs["cv_scores"], args.n_splits, "macro_f1")
    rm.set_holdout_metrics(y_hold, y_pred_rf, class_names)
    rm.set_hpo_stats(study_rf, opt_time, "TPESampler", "None")
    rm.set_training_stats(fit_time, device="CPU")  # RF don't use GPU
    rm.add_custom_metrics(random_forest_info={ "model_info": {
        "estimators_range": GlobalDefaults.RF_N_ESTIMATORS_RANGE,
        "max_depth_options": GlobalDefaults.RF_MAX_DEPTH_OPTIONS}})

    # 3) Save all collected results to a JSON file.
    rm.save("training_results_random_forest.json")

    # =============================================================================
    # Step 10 ─ Visualizations (Random Forest)
    # -----------------------------------------------------------------------------
    # • Generates and saves confusion matrix plots (raw and normalized) for model diagnostics.
    # • Exports Optuna optimization history and feature importance (barplot/cumulative) plots.
    # • Notes that learning curves are not produced for Random Forest (non-iterative model).
    # • Ensures all artifacts are stored in the output directory for reproducibility and reporting.
    # =============================================================================
    logger.info("Generating and saving visualizations...")
    print("-" * 60)
    print("[Step 10/11] Generating and saving visualizations...")
    print("-" * 60)

    cm_labels = GlobalDefaults.IMPACT_CLASSES

    # 1) Generate and save both raw and normalized confusion matrices figures
    cm = confusion_matrix(y_hold, y_pred_rf) # Raw confusion matrix
    cm_norm = confusion_matrix(y_hold, y_pred_rf, normalize="true") # Normalized confusion matrix
    for matrix, suffix in zip([cm, cm_norm], ["raw", "normalized"]):
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(matrix, annot=True, fmt=".2f" if "norm" in suffix else "d", cmap="Blues",
                    xticklabels=cm_labels, yticklabels=cm_labels, ax=ax)
        ax.set_title(f"Random Forest Confusion Matrix ({suffix.capitalize()})")
        ax.set_xlabel("Predicted Impact Level"), ax.set_ylabel("True Impact Level"), plt.tight_layout()
        plt.savefig(output_dir / f"confusion_matrix_rf_{suffix}.png"), plt.close(fig)
        print(f"✓ Confusion matrix plot saved: {output_dir / f'confusion_matrix_rf_{suffix}.png'}")

    # 2) Save Optuna optimization history as HTML
    fig_optuna = plot_optimization_history(study_rf)
    fig_optuna.write_html(output_dir / "optuna_history_rf.html")
    print(f"✓ Optuna optimization history saved: {output_dir / 'optuna_history_rf.html'}")

    # 3) Feature importance barplot and cumulative figures
    print("Generating feature importance barplot...")
    top_n = 20  # Number of top features to plot (adjust as needed)
    plot_df = importance_df.head(top_n)
    plt.figure(figsize=(10, 8))
    colors = sns.color_palette("viridis", len(plot_df))
    sns.barplot(x="importance", y="feature", data=plot_df, palette=colors, hue="feature")
    plt.legend([], [], frameon=False)
    plt.title(f"Top {top_n} Feature Importance (Random Forest)")
    plt.xlabel("Importance"), plt.ylabel("Feature"), plt.tight_layout()
    plt.savefig(output_dir / "feature_importance_rf.png", dpi=300), plt.close()
    print(f"✓ Feature importance barplot saved: {output_dir / 'feature_importance_rf.png'}")

    # --- Cumulative Feature Importance Plot (PNG) ---
    print("Generating cumulative feature importance plot...")
    importance_df['cum_importance'] = (importance_df['importance'].cumsum() / importance_df['importance'].sum()) * 100
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(importance_df) + 1), importance_df['cum_importance'], marker='o', markersize=4)
    ax.set_title('Cumulative Feature Importance (Random Forest)'), ax.set_xlabel('Number of Features')
    ax.set_ylabel('Cumulative Importance (%)'), ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    for perc in [90, 95, 99]:
        ax.axhline(y=perc, color='r', linestyle='--', linewidth=0.8, label=f'{perc}% Threshold')
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys()), plt.tight_layout()
    plt.savefig(output_dir / "feature_importance_cumulative_rf.png", dpi=300), plt.close(fig)
    print(f"✓ Cumulative feature importance plot saved: {output_dir / 'feature_importance_cumulative_rf.png'}")

    # Note on Learning Curves for Random Forest:
    # We don't generate epoch-based learning curves here because RandomForest is not an iterative model.
    # Unlike Gradient Boosting or Neural Networks, it builds independent trees in parallel, so there are no
    # "training steps" or "epochs" to plot performance against.

    print("\n")
    logger.info(f"All RANDOM FOREST visualizations and results saved to: {output_dir}")
    print(f"✓ All RANDOM FOREST visualizations and results saved to: {output_dir}")

    # =============================================================================
    # Step 11 ─ Final Summary (Random Forest)
    # -----------------------------------------------------------------------------
    # • Logs and prints a summary of the entire Random Forest experiment in AIMS.
    # • Reports total execution time, save location, and model performance metrics.
    # • Calls ResultsManager.get_summary() to generate a standardized human-readable summary.
    # =============================================================================
    total_time = time.time() - total_start_time
    logger.info(f"RANDOM FOREST COMPLETED! Total execution time: {total_time / 60:.1f} minutes")
    logger.info(f"All results saved to: {output_dir}. Generating summary...")
    print("-" * 60)
    print(f"[Step 11/11] RANDOM FOREST COMPLETED! Total execution time: {total_time / 60:.1f} minutes")
    print(f"All results saved to: {output_dir}. Generating summary...")
    print("-" * 60)
    print(rm.get_summary())

    # Example output:
    # =============================================================================
    # RANDOMFOREST TRAINING SUMMARY
    # =============================================================================
    # Model Library: scikit-learn v1.6.1
    # Data: 4576 train / 824 holdout
    # CV Score (Macro F1): 0.9820
    # Holdout Accuracy: 0.9454
    # Holdout Macro F1: 0.9462
    # Total Run Time: 36.4s
    # =============================================================================


    print("=" * 80)
    print(f"All results saved to: {output_dir}")
    print("=" * 80)


    print(f"\nTotal execution time: {total_time / 60:.1f} minutes")

if __name__ == "__main__":
    """Entry point for executing the script directly from the command line."""
    parser = build_arg_parser()
    args = parser.parse_args()

    # Log system information
    logger.info(f"Python version: {sys.version}")
    logger.info(f"NumPy version: {np.__version__}")
    logger.info(f"Pandas version: {pd.__version__}")
    logger.info(f"Scikit-learn version: {sklearn.__version__}")
    logger.info(f"Optuna version: {optuna.__version__}")

    try:
        main(args)
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise
