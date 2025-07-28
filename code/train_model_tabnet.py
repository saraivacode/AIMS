#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIMS Framework – TabNet Training & Optimization Module
======================================================

End-to-end pipeline for **training, hyperparameter tuning, and evaluating** a TabNet classifier inside
the *AIMS* (Adaptive and Intelligent Management of Slicing) framework. The model predicts the **impact
level** of network-slicing policies on vehicular Intelligent Transportation Systems (ITS).

Key components
--------------
1.  **Data ingestion & preparation** – domain-specific cleaning and feature engineering.
2.  **Custom impact labelling** – weighted aggregation of latency, loss and throughput (`label_weighted_average`).
3.  **TabNet-specific preprocessing** – `OrdinalEncoder` for categorical features, as required by the model architecture.
4.  **Device-aware training** – automatic CPU / GPU selection for acceleration.
5.  **Group-aware validation & HPO** – `GroupShuffleSplit` hold-out + `GroupKFold` inside
Optuna (**TPE** sampler; no external pruner).
6.  **Final training & hold-out evaluation** – macro F1, confusion matrix, and attention-based feature importance.
7.  **Artifact management** – model (`joblib`), metrics JSON, PNG/HTML plots, CSV importance (handled by
`ResultsManager` & `save_model_results`).

Usage example
-------------
```bash
python train_tabnet.py \
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
# AIMS Framework - TabNet Training Module
# -----------------------------------------------------------------------------
# • Handles imports: standard, third-party, and project-specific modules.
# • Configures global logging, warning filters, and consistent plotting styles.
# • Defines global default values for data paths and TabNet training parameters.
# =============================================================================

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# -------------------------------------------------
# Third-party imports
# -------------------------------------------------
import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import sklearn
import torch
from optuna.exceptions import ExperimentalWarning
from optuna.visualization import plot_optimization_history
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

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
# • Suppresses unwanted warnings (pytorch_tabnet, Optuna) for cleaner logs.
# • Configures logging for progress tracking and debugging.
# • Applies a unified matplotlib/seaborn style for all plots.
# -------------------------------------------------

# Suppress pytorch_tabnet and Optuna Experimental Warnings for a cleaner output.
warnings.filterwarnings("ignore", category=UserWarning, module=r"pytorch_tabnet\..*")
warnings.filterwarnings("ignore", category=ExperimentalWarning)

# Configure logging to display progress and informational messages.
logging.basicConfig(level=logging.INFO, format='\n%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TabNetTrainer")

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

    # Model-specific defaults
    TABNET_PATIENCE = 30
    TABNET_MAX_EPOCHS = 200
    TABNET_BATCH_SIZE = 1024

    # Impact classification
    IMPACT_CLASSES = ["Adequate", "Warning", "Severe", "Critical"]

def build_arg_parser() -> argparse.ArgumentParser:
    """Constructs the command-line argument parser for the script.

    Returns:
        argparse.ArgumentParser: The configured argument parser instance.
    """
    parser = argparse.ArgumentParser(
        description="Train and optimize a TabNet classifier for AIMS impact prediction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--csv", type=Path, default=Path(GlobalDefaults.DEFAULT_CSV_PATH),
        help="Path to the CSV file with the vehicular QoS dataset."
    )

    parser.add_argument(
        "--n-splits", type=int, default=GlobalDefaults.DEFAULT_N_SPLITS,
        help="Number of folds for GroupKFold cross-validation."
    )

    parser.add_argument(
        "--n-trials", type=int, default=GlobalDefaults.DEFAULT_N_TRIALS,
        help="Number of hyperparameter optimization trials to run with Optuna."
    )

    parser.add_argument(
        "--random-state", type=int, default=GlobalDefaults.DEFAULT_RANDOM_STATE,
        help="Random seed for ensuring reproducibility across runs."
    )

    return parser

def configure_device() -> Tuple[str, Optional[str], Optional[float]]:
    """Detects and configures the computing device (GPU or CPU) for training.

    GPU acceleration significantly improves TabNet training speed. This function
    provides detailed information about the detected hardware.

    Returns:
        A tuple containing:
            - device (str): The device string ("cuda" or "cpu").
            - gpu_name (Optional[str]): The name of the GPU, if available.
            - gpu_memory_gb (Optional[float]): Total GPU memory in GB, if available.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = None
    gpu_memory_gb = None

    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        logger.info(f"CUDA-enabled GPU detected: {gpu_name} ({gpu_memory_gb:.1f} GB). Setting task_type to 'GPU'.")
        print(f"CUDA-enabled GPU detected: {gpu_name} ({gpu_memory_gb:.1f} GB). Setting task_type to 'GPU'.")
    else:
        logger.warning("No GPU detected. Using CPU, which will be significantly slower.")
    return device, gpu_name, gpu_memory_gb

def prepare_features_for_tabnet(X: pd.DataFrame,
                                preprocessor: Optional[ColumnTransformer] = None) -> Tuple[pd.DataFrame,
                                ColumnTransformer, List[int], List[int], List[str], List[str]]:
    """Prepares features specifically for TabNet's requirements.

    TabNet needs categorical features to be integer-encoded (not one-hot) and
    requires metadata about their indices and cardinality.

    Args:
        X (pd.DataFrame): The input feature dataframe.
        preprocessor (Optional[ColumnTransformer]): A pre-fitted preprocessor.
            If None, a new one is created and fitted.

    Returns:
        A tuple containing:
            - X_processed (pd.DataFrame): The preprocessed features.
            - preprocessor (ColumnTransformer): The fitted preprocessing pipeline.
            - cat_idxs (List[int]): Indices of the categorical columns.
            - cat_dims (List[int]): Cardinality of each categorical feature.
            - num_cols (List[str]): Names of the numerical columns.
            - cat_cols (List[str]): Names of the categorical columns.
    """

    # Identify numerical and categorical features.
    num_cols = X.select_dtypes(include=[np.number, bool]).columns.tolist()
    cat_cols = [col for col in X.columns if col not in num_cols]
    #  Ensure all categorical features are of string type for robust handling.
    for col in cat_cols:
        X[col] = X[col].astype(str)
    print(f"✓ Numerical features: {len(num_cols)}")
    print(f"✓ Categorical features: {len(cat_cols)}")

    if preprocessor is None:
        # Create a preprocessing pipeline.
        # TabNet requires OrdinalEncoder for categorical features.
        preprocessor = ColumnTransformer([
            ("num", StandardScaler(), num_cols),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
        ], remainder='passthrough')
        preprocessor.fit(X)

    # Transform the features using the pipeline.
    X_processed = pd.DataFrame(preprocessor.transform(X), columns=num_cols + cat_cols, index=X.index)

    # Extract metadata required by the TabNet model.
    cat_idxs = [X_processed.columns.get_loc(col) for col in cat_cols]
    cat_dims = [int(X[col].nunique()) for col in cat_cols]

    return X_processed, preprocessor, cat_idxs, cat_dims, num_cols, cat_cols


def create_tabnet_classifier(params: Dict[str, Any], cat_idxs: List[int], cat_dims: List[int],
                             device: str, random_state: int, verbose: int = 0) -> TabNetClassifier:
    """Creates a TabNetClassifier instance with a given configuration.

    This helper function ensures that the model is created consistently across different stages of
    the pipeline (e.g., HPO and final training).

    Args:
        params (Dict[str, Any]): A dictionary of hyperparameters for the model.
        cat_idxs (List[int]): List of indices for categorical features.
        cat_dims (List[int]): List of cardinalities for categorical features.
        device (str): The computing device ("cuda" or "cpu").
        random_state (int): The random seed for reproducibility.
        verbose (int): The verbosity level for training.

    Returns:
        TabNetClassifier: A configured, un-trained TabNet model.
    """

    # Default optimizer and scheduler (original behavior)
    optimizer_fn = torch.optim.Adam
    optimizer_params = dict(
        lr=params.get("lr", 2e-2),
        weight_decay=params.get("weight_decay", 1e-5)
    )
    scheduler_fn = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = dict(mode="min", patience=10, factor=0.5)

    # If the trial requests OneCycleLR, override the scheduler and optimizer
    if params.get("use_one_cycle_lr", False):
        optimizer_fn = torch.optim.AdamW
        scheduler_fn = lambda opt, **kwargs: torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=params.get("lr", 2e-2),
            pct_start=0.1,
            epochs=GlobalDefaults.TABNET_MAX_EPOCHS,
            steps_per_epoch=1,
            anneal_strategy="cos",
        )
        scheduler_params = dict(step_every_batch=False)

    clf = TabNetClassifier(
        n_d=params.get("n_d", 8),
        n_a=params.get("n_a", 8),
        n_steps=params.get("n_steps", 3),
        gamma=params.get("gamma", 1.3),
        lambda_sparse=params.get("lambda_sparse", 1e-3),
        mask_type=params.get("mask_type", "sparsemax"),
        optimizer_fn=optimizer_fn,
        optimizer_params=optimizer_params,
        scheduler_fn=scheduler_fn,
        scheduler_params=scheduler_params,
        seed=random_state,
        verbose=verbose,
        device_name=device,
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        cat_emb_dim=params.get("cat_emb_dim", 1),
        clip_value=params.get("clip_value", 2.0)
    )
    return clf

def create_optuna_objective_tabnet(X_train: pd.DataFrame, y_train: np.ndarray, groups_train: np.ndarray,
                                   cat_idxs: List[int], cat_dims: List[int], device: str, random_state: int,
                                   class_weights_tensor: torch.Tensor, cv_splitter: GroupKFold) -> callable:
    """
    Creates an Optuna objective function for TabNet HPO.

    This factory function encapsulates the training data and configuration, returning a callable objective
    that Optuna can use to find the best hyperparameters via group-aware cross-validation.

    Args:
        :param cv_splitter: The configured GroupKFold splitter.
        :param class_weights_tensor: Class weights as a PyTorch tensor.
        :param groups_train: Group identifiers for temporal cross-validation.
        :param y_train: Training labels.
        :param X_train: Training features.
        :param random_state: Training configuration.
        :param device: Training configuration.
        :param cat_dims: Metadata for categorical features.
        :param cat_idxs: Metadata for categorical features.

    Returns:
        A callable objective function for Optuna.
    """
    def objective(trial: optuna.Trial) -> float:
        """The Optuna objective function to be maximized."""
        n_d_value = trial.suggest_int("n_d", 16, 64, step=16)
        batch_size_value = trial.suggest_categorical("batch_size", [512, 1024, 2048])
        virtual_bs_ratio_value = trial.suggest_categorical("virtual_bs_ratio", [0.25, 0.5])

        params = {
            "n_d": n_d_value,
            "n_a": n_d_value, # n_a is kept symmetric to n_d
            "n_steps": trial.suggest_int("n_steps", 3, 5),
            "gamma": trial.suggest_float("gamma", 1.0, 1.5, step=0.1),
            "lambda_sparse": trial.suggest_float("lambda_sparse", 1e-6, 1e-3, log=True),
            "lr": trial.suggest_float("lr", 1e-3, 3e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
            "mask_type": trial.suggest_categorical("mask_type", ["sparsemax", "entmax"]),
            "cat_emb_dim": trial.suggest_int('cat_emb_dim', 1, 3),
            "batch_size": batch_size_value,
            "virtual_batch_size": int(batch_size_value * virtual_bs_ratio_value),
            "use_one_cycle_lr": True
        }

        cv_scores = []
        for fold, (train_idx, val_idx) in enumerate(cv_splitter.split(X_train, y_train, groups_train)):
            X_tr, y_tr = X_train.iloc[train_idx].values, y_train[train_idx]
            X_va, y_va = X_train.iloc[val_idx].values, y_train[val_idx]

            # Use the existing helper function to create the model
            clf = create_tabnet_classifier(params, cat_idxs, cat_dims, device,
                                           random_state, verbose=0)
            try:
                clf.fit(
                    X_tr, y_tr,
                    eval_set=[(X_va, y_va)],
                    eval_metric=["accuracy", "logloss"],
                    loss_fn=torch.nn.CrossEntropyLoss(weight=class_weights_tensor),
                    patience=GlobalDefaults.TABNET_PATIENCE,
                    max_epochs=GlobalDefaults.TABNET_MAX_EPOCHS,
                    batch_size=params["batch_size"],
                    virtual_batch_size=params["virtual_batch_size"],
                    drop_last=False
                )
                preds = clf.predict(X_va)
                cv_scores.append(f1_score(y_va, preds, average="macro"))

                # Report intermediate results for pruning
                trial.report(np.mean(cv_scores), fold)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            except Exception as e:
                logger.warning(f"Trial {trial.number} fold {fold} failed: {e}")
                cv_scores.append(0.0)  # Penalize failed trials.

        trial.set_user_attr("cv_scores", cv_scores)
        return np.mean(cv_scores) if cv_scores else 0.0

    return objective

def train_final_tabnet_model( best_params: Dict[str, Any], X_train: np.ndarray, y_train: np.ndarray,
                              X_val: np.ndarray, y_val: np.ndarray, cat_idxs: List[int], cat_dims: List[int],
                              device: str, random_state: int, class_weights_tensor: torch.Tensor
                              ) -> Tuple[TabNetClassifier, float, Dict[str, Any]]:
    """Trains the final TabNet model on the full training data with the best parameters.

    Args:
        :param best_params: The best hyperparameters found by Optuna.
        :param class_weights_tensor: Class weights as a PyTorch tensor.
        :param random_state: Training configuration.
        :param device: Training configuration.
        :param cat_dims: Metadata for categorical features.
        :param cat_idxs: Metadata for categorical features.
        :param y_val: The internal validation data for early stopping.
        :param X_val: The internal validation data for early stopping.
        :param y_train: The training data.
        :param X_train: The training data.
    Returns:
        A tuple containing:
            - clf: The trained TabNet model.
            - fit_time: The time taken for training in seconds.
            - training_history: A dictionary of metrics from the training process.
    """

    logger.info("Training final TabNet model with optimized parameters...")
    clf = create_tabnet_classifier(best_params, cat_idxs, cat_dims, device, random_state, verbose=1)

    fit_start = time.time()
    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_name=["valid"],
        eval_metric=["accuracy", "logloss"],
        loss_fn=torch.nn.CrossEntropyLoss(weight=class_weights_tensor),
        patience=GlobalDefaults.TABNET_PATIENCE, max_epochs=GlobalDefaults.TABNET_MAX_EPOCHS,
        batch_size=GlobalDefaults.TABNET_BATCH_SIZE, virtual_batch_size=128,
        drop_last=False,
        weights=1
    )
    fit_time = time.time() - fit_start
    logger.info(f"Final model training completed in {fit_time:.2f} seconds.")

    # Extract training history for logging and visualization.
    history_dict = clf.history.history
    training_history = {
        "train_loss": history_dict.get("loss", []),
        "valid_accuracy": history_dict.get("valid_accuracy", []),
        "valid_logloss": history_dict.get("valid_logloss", history_dict.get("valid_loss", [])),
        "best_epoch": getattr(clf, "best_epoch", -1),
        "total_epochs": len(history_dict.get("loss", [])),
    }

    return clf, fit_time, training_history

def run_tabnet(csv_path: str, n_splits: int = 5, n_trials: int = 40, random_state: int = 42,
               return_study: bool = False) -> Optional[Tuple[optuna.Study, TabNetClassifier, ColumnTransformer]]:
    """Executes the complete TabNet training and evaluation pipeline.

    Args:
        csv_path: Path to the input dataset.
        n_splits: Number of folds for cross-validation.
        n_trials: Number of trials for hyperparameter optimization.
        random_state: The random seed for reproducibility.
        return_study: If True, returns the Optuna study and trained model artifacts.

    Returns:
        If return_study is True, returns a tuple with the study, classifier,
        and preprocessor. Otherwise, returns None.
    """
    total_start_time = time.time()
    # ==================== Initialization ====================
    logger.info("Starting TabNet Training Pipeline...")
    print("\n" + "=" * 80)
    print("TABNET TRAINING PIPELINE - AIMS FRAMEWORK")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  - Dataset: {csv_path}")
    print(f"  - CV Folds: {n_splits}")
    print(f"  - Optuna Trials: {n_trials}")
    print(f"  - Random State: {random_state}")
    print("=" * 80)

    # =============================================================================
    # Step 1 ─ Data Loading & Preprocessing
    # -----------------------------------------------------------------------------
    # • Validates that the CSV file exists.
    # • Loads raw data with pandas, then applies domain-specific cleaning via
    #   pp.prepare_dataset().
    # • Logs and prints the shapes of raw and processed data for traceability.
    # =============================================================================
    logger.info("Loading and preprocessing dataset from %s ...", csv_path)
    print("-" * 60)
    print("[Step 1/11] Loading and preprocessing dataset...")
    print("-" * 60)

    # 1) Sanity-check: file must exist
    try:
        # 2) Load and preprocess
        df_raw = pd.read_csv(csv_path)
        df = pp.prepare_dataset(df_raw)
        # 3) Quick sanity printouts
        print(f"✓ Raw dataset shape: {df_raw.shape}")
        print(f"✓ Preprocessed dataset shape: {df.shape}")
    except Exception as e:
        logger.error(f"CSV process failed: {e}", exc_info=True)
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

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
        class_weights_list,  # list with pre-calculated class weights to handle imbalance.
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

    # 1) Identify categorical and numerical features and prepare fo Tabnet
    X_processed, preprocessor, cat_idxs, cat_dims, num_cols, cat_cols = prepare_features_for_tabnet(X)
    print(f"✓ Processed feature shape: {X_processed.shape}")

    # 2) Group-aware train/holdout split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    train_idx, hold_idx = next(gss.split(X_processed, y, groups))
    X_train, y_train, groups_train = X_processed.iloc[train_idx], y[train_idx], groups[train_idx]
    X_hold, y_hold = X_processed.iloc[hold_idx], y[hold_idx]
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
    output_dir = save_model_results('tabnet', X, y, groups, class_weights_dict)

    # 2) Instantiate ResultsManager for handling results and artifacts
    rm = ResultsManager("TabNet", output_dir)

    # =============================================================================
    # Steps 5 & 6 ─ Optuna Hyperparameter Optimization (HPO) Configuration and Execution (TabNet)
    # -----------------------------------------------------------------------------
    # • Sets up and runs Optuna for TabNet hyperparameter optimization.
    # • Configures a TPE sampler for efficient search.
    # • Defines and passes a custom objective function using GroupKFold CV.
    # • Supports GPU acceleration if available.
    # • Logs and print best hyperparameters and CV results.
    # =============================================================================
    logger.info("Configure and Run Hyperparameter Optimization (HPO) with Optuna ...")
    print("-" * 60)
    print("[Steps 5 & 6 /11] Configure and Run Tabnet Hyperparameter Optimization (HPO) with Optuna ...")
    print("-" * 60)

    # 1) Configure Optuna sampler and study.
    sampler = optuna.samplers.TPESampler(seed=random_state, n_startup_trials=15)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=3) # Add pruner
    study_tb = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner) # Pass pruner

    # 2) Prepare TabNet objective function and run Optuna.
    device, gpu_name, gpu_memory_gb = configure_device()
    gkf = GroupKFold(n_splits=n_splits)
    class_weights_tensor = torch.tensor(class_weights_list, dtype=torch.float32, device=device)

    # --- Model Instantiation Strategy ---
    # Inside the objective function (`create_optuna_objective_tabnet`), we use the `create_tabnet_classifier`
    # factory function to instantiate the model, instead of cloning a base model. This approach guarantees that
    # a fresh TabNetClassifier instance is created with a consistent configuration model for every Optuna trial.
    objective_func = create_optuna_objective_tabnet( X_train, y_train, groups_train, cat_idxs, cat_dims,
                                                     device, random_state, class_weights_tensor, gkf)

    logger.info(f"Starting Optuna optimization with {n_trials} trials...")
    print(f"Starting Optuna optimization with {n_trials} trials...")
    opt_start_time = time.time()
    study_tb.optimize(objective_func, n_trials=n_trials, show_progress_bar=True, gc_after_trial=True)
    opt_time = time.time() - opt_start_time  # Track optimization time
    logger.info(f"✓ Optimization completed in {opt_time:.1f} seconds!")
    print(f"✓ Optimization completed in {opt_time:.1f} seconds!")

    # 3) Log and display best hyperparameters and performance.
    best_params = study_tb.best_params
    print(f"Best trial: {study_tb.best_trial.number}")
    print(f"Best parameters found:")
    for param, value in best_params.items():
        print(f"  - {param}: {value}")
    best_cv_scores = study_tb.best_trial.user_attrs.get("cv_scores", [])
    logger.info(f"Best F1-macro (CV): {study_tb.best_value:.4f} (±{np.std(best_cv_scores):.4f})")
    print(f"\nBest F1-macro (CV): {study_tb.best_value:.4f} (±{np.std(best_cv_scores):.4f})")

    # =============================================================================
    # Step 7 ─ Final Model Training (TabNet)
    # -----------------------------------------------------------------------------
    # • Splits the training data to create a small internal validation set (10%).
    # • Trains the final TabNet model with the best hyperparameters, supporting early stopping.
    # • Saves the trained model artifact and prints/logs key configuration details.
    # =============================================================================
    logger.info("Final model training with best hyperparameters...")
    print("-" * 60)
    print("[Step 7/11] Final model training with best hyperparameters...")
    print("-" * 60)

    # 1) Data split: 10% of training set reserved for validation.
    X_tr_final, X_val_final, y_tr_final, y_val_final = train_test_split(
        X_train.values, y_train, test_size=0.10, random_state=random_state, stratify=y_train)

    # 2) Final classifier creation with best params and fitting.
    clf, fit_time, training_history = train_final_tabnet_model(best_params, X_tr_final, y_tr_final, X_val_final,
                                                               y_val_final, cat_idxs, cat_dims, device, random_state,
                                                               class_weights_tensor)
    logger.info(f"✓ Final model fitting completed in {fit_time:.2f} seconds!")
    print(f"✓ Final model fitting completed in {fit_time:.2f} seconds!")

    # 3) Save the trained model object and print training results.
    model_path = output_dir / "tabnet_model.pkl"
    joblib.dump((preprocessor, clf), model_path)
    logger.info(f"✓ Final model saved to: {model_path}")
    print(f"✓ Final model saved to: {model_path}")
    best_epoch = getattr(clf, "best_epoch", -1)
    total_epochs = training_history["total_epochs"]
    print(f"✓ Best epoch: {best_epoch}")
    print(f"✓ Number of epochs trained: {total_epochs}")

    # =============================================================================
    # Step 8 ─ Evaluating Final Model on the Holdout Set (TabNet)
    # -----------------------------------------------------------------------------
    # • Generates predictions for the holdout set using the final trained TabNet model.
    # • Computes and prints detailed classification metrics and key performance indicators.
    # =============================================================================
    logger.info("Evaluating final model on the holdout set...")
    print("-" * 60)
    print("[Step 8/11] Evaluating final model on the holdout set...")
    print("-" * 60)

    class_names = GlobalDefaults.IMPACT_CLASSES

    # 1) Generate predictions for the holdout set.
    y_pred = clf.predict(X_hold.values)

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
    # Step 9 ─ Save Results: CSV and JSON Files (TabNet)
    # -----------------------------------------------------------------------------
    # • Exports TabNet feature importance to CSV for interpretability.
    # • Saves all experiment data, metrics, and configurations using ResultsManager to a JSON file.
    # =============================================================================
    logger.info("Save Results: CSV and JSON Files...")
    print("-" * 60)
    print("[Step 9/11] Save Results: CSV and JSON Files...")
    print("-" * 60)

    # 1) Calculate and Save Raw Feature Importance to CSV
    # TabNet provides feature_importances_ attribute, which aggregates the attention mask magnitudes across all
    # decision steps. Higher value indicates that the model paid more attention to that feature when making predictions.
    importance_df = pd.DataFrame({'feature': num_cols + cat_cols,'importance': clf.feature_importances_
                                  }).sort_values('importance', ascending=False)
    feature_importance_path = output_dir / "feature_importances_tabnet.csv"
    importance_df.to_csv(feature_importance_path, index=False)
    print(f"✓ Feature importance saved to: {feature_importance_path}")

    # 2) Collect all experiment results and configuration using ResultsManager.
    print("Saving comprehensive results...")
    rm.set_data_info(X_train, y_train, X_hold, y_hold, groups_train, "GroupShuffleSplit 80/20", random_state)
    rm.set_best_params(best_params)
    rm.set_cv_metrics(best_cv_scores, n_splits, "macro_f1")
    rm.set_holdout_metrics(y_hold, y_pred, class_names)
    rm.set_hpo_stats(study_tb, opt_time, "TPESampler", "None")
    rm.set_training_stats(fit_time, device.upper(), best_epoch, sum(p.numel() for p in clf.network.parameters()))
    rm.add_custom_metrics(tabnet_info={"model_info": {"patience": GlobalDefaults.TABNET_PATIENCE,
                                                      "max_epochs": GlobalDefaults.TABNET_MAX_EPOCHS,
                                                      "batch_size": GlobalDefaults.TABNET_BATCH_SIZE}})

    # 3) Save all collected results to a JSON file.
    rm.save("training_results_tabnet.json")

    # =============================================================================
    # Step 10 ─ Visualizations (TabNet)
    # -----------------------------------------------------------------------------
    # • Generates and saves confusion matrix plots (raw and normalized) for model diagnostics.
    # • Exports Optuna optimization history, feature importance (barplot/cumulative), and training curves.
    # • Ensures all artifacts are stored in the output directory for reproducibility and reporting.
    # =============================================================================
    logger.info("Generating and saving visualizations...")
    print("-" * 60)
    print("[Step 10/11] Generating and saving visualizations...")
    print("-" * 60)

    # 1) Generate and save both raw and normalized confusion matrix figures
    cm = confusion_matrix(y_hold, y_pred)  # Raw confusion matrix
    cm_norm = confusion_matrix(y_hold, y_pred, normalize="true")  # Normalized confusion matrix
    for matrix, suffix in zip([cm, cm_norm], ["raw", "normalized"]):
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(matrix, annot=True, fmt=".2f" if "norm" in suffix else "d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title(f"TabNet Confusion Matrix ({suffix.capitalize()})")
        ax.set_xlabel("Predicted Impact Level"), ax.set_ylabel("True Impact Level"), plt.tight_layout()
        plt.savefig(output_dir / f"confusion_matrix_tb_{suffix}.png"), plt.close(fig)
        print(f"✓ Confusion matrix plot saved: {output_dir / f'confusion_matrix_tb_{suffix}.png'}")

    # 2) Save Optuna optimization history as HTML
    fig_optuna = plot_optimization_history(study_tb)
    fig_optuna.write_html(output_dir / "optuna_history_tb.html")
    print(f"✓ Optuna optimization history saved: {output_dir / 'optuna_history_tb.html'}")

    # 3) Feature importance barplot and cumulative figures
    print("Generating feature importance barplot...")
    top_n = 20  # Number of top features to plot
    plot_df = importance_df.head(top_n)
    plt.figure(figsize=(10, 8))
    colors = sns.color_palette("viridis", len(plot_df))
    try:
        sns.barplot(x="importance", y="feature", data=plot_df, palette=colors, hue="feature", legend=False)
    except TypeError:
        sns.barplot(x="importance", y="feature", data=plot_df, palette=colors)  # Fallback for seaborn <0.14
    plt.title(f"Top {top_n} Feature Importance (TabNet)")
    plt.xlabel("Importance"), plt.ylabel("Feature"), plt.tight_layout()
    plt.savefig(output_dir / "feature_importance_tabnet.png", dpi=300), plt.close()
    print(f"✓ Feature importance barplot saved: {output_dir / 'feature_importance_tabnet.png'}")

    # --- Cumulative Feature Importance Plot (PNG) ---
    importance_df['cum_importance'] = (importance_df['importance'].cumsum() / importance_df['importance'].sum()) * 100
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(importance_df) + 1), importance_df['cum_importance'], marker='o', markersize=4)
    ax.set_title('Cumulative Feature Importance'), ax.set_xlabel('Number of Features')
    ax.set_ylabel('Cumulative Importance (%)'), ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    for perc in [90, 95, 99]:
        ax.axhline(y=perc, color='r', linestyle='--', linewidth=0.8, label=f'{perc}% Threshold')
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance_cumulative_tabnet.png", dpi=300), plt.close(fig)
    print(f"✓ Cumulative feature importance plot saved: {output_dir / 'feature_importance_cumulative_tabnet.png'}")

    # 4) Final TabNet Training Curves (Loss/Accuracy).
    print("Generating Final TabNet Training Curves (Loss/Accuracy)...")
    if training_history["train_loss"]:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        epochs = range(1, training_history["total_epochs"] + 1)
        ax1.plot(epochs, training_history["train_loss"], label="Train Loss")
        if training_history["valid_logloss"]:
            ax1.plot(epochs, training_history["valid_logloss"], label="Valid Loss")
        ax1.set_xlabel("Epoch"), ax1.set_ylabel("Loss"), ax1.set_title("Training & Validation Loss"), ax1.legend()
        if training_history["valid_accuracy"]:
            ax2.plot(epochs, training_history["valid_accuracy"], label="Valid Accuracy", color="green")
            ax2.set_xlabel("Epoch"), ax2.set_ylabel("Accuracy"), ax2.set_title("Validation Accuracy"), ax2.legend()
        plt.tight_layout(), plt.savefig(output_dir / "tabnet_training_curves.png", dpi=300), plt.close(fig)
        print(f"✓ Final TabNet Training & Validation Curves Plots Saved: {output_dir / 'tabnet_training_curves.png'}")

    print("\n")
    logger.info(f"All TABNET visualizations and results saved to: {output_dir}")
    print(f"✓ All TABNET visualizations and results saved to: {output_dir}")

    # =============================================================================
    # Step 11 ─ Final Summary (TabNet)
    # -----------------------------------------------------------------------------
    # • Logs and prints a summary of the entire TabNet experiment in AIMS.
    # • Reports total execution time, save location, and model performance metrics.
    # • Calls ResultsManager.get_summary() to generate a standardized human-readable summary.
    # =============================================================================
    total_time = time.time() - total_start_time
    logger.info(f"TABNET COMPLETED! Total execution time: {total_time / 60:.1f} minutes")
    logger.info(f"All results saved to: {output_dir}. Generating summary...")
    print("-" * 60)
    print(f"[Step 11/11] TABNET COMPLETED! Total execution time: {total_time / 60:.1f} minutes")
    print(f"All results saved to: {output_dir}. Generating summary...")
    print("-" * 60)
    print(rm.get_summary())

    # Example output:
    # =============================================================================
    # TABNET TRAINING SUMMARY
    # =============================================================================
    # Model Library: pytorch-tabnet v4.1.0
    # Data: 4576 train / 824 holdout
    # CV Score (Macro F1): 0.8984
    # Holdout Accuracy: 0.8228
    # Holdout Macro F1: 0.8108
    # Total Run Time: 584.8s
    # =============================================================================

    if return_study:
        return study_tb, clf, preprocessor
    return None

if __name__ == "__main__":
    """Entry point for direct script execution."""
    parser = build_arg_parser()
    args = parser.parse_args()

    if not args.csv.is_file():
        logger.error(f"Dataset not found: {args.csv}")
        sys.exit(1)

    # Log system and library versions for reproducibility.
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"Scikit-learn version: {sklearn.__version__}")
    logger.info(f"Optuna version: {optuna.__version__}")

    try:
        run_tabnet(
            str(args.csv),
            n_splits=args.n_splits,
            n_trials=args.n_trials,
            random_state=args.random_state
        )
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed with an unhandled error: {e}", exc_info=True)
        raise
