#!/usr/bin/env python3
"""
Random Forest Training Module for AIMS Framework
===============================================

This module implements the Random Forest classifier training pipeline for the
AIMS (Adaptive and Intelligent Management of Slicing) framework. It trains
a Random Forest model to classify the impact of network slicing policies on
vehicular QoS metrics in Intelligent Transportation Systems (ITS).

Key Features:
    - Random Forest with class-balanced weights for imbalanced data
    - GroupKFold cross-validation to prevent temporal data leakage
    - Hyperparameter optimization using Optuna with TPE sampler
    - Comprehensive feature preprocessing (scaling + one-hot encoding)
    - Automated artifact saving and performance visualization

Model Characteristics:
    Random Forest is an ensemble method that:
    - Builds multiple decision trees using bootstrap samples
    - Combines predictions through majority voting
    - Provides natural feature importance rankings
    - Handles mixed numerical/categorical features well
    - Offers good interpretability compared to deep learning

Performance:
    - Typical accuracy: 94-95% on vehicular QoS impact classification
    - Training time: 5-10 minutes with 40 Optuna trials
    - Inference time: ~5ms per sample (CPU)

Usage:
    Direct execution:
        python train_model_rf.py --csv data.csv --n-splits 5 --n-trials 40

    As module:
        from train_model_rf import main
        main(args)

Requirements:
    - scikit-learn >= 1.0.0
    - optuna >= 3.0.0
    - pandas, numpy, matplotlib, seaborn
    - See requirements.txt for complete list

Author: AIMS Framework Team
License: MIT
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import make_scorer, f1_score, classification_report, confusion_matrix
from sklearn.base import clone
import preprocess_dataset as pp
from impact_labeling import label_weighted_average, WEIGHTS
import optuna
import optuna.visualization as vis
from save_utils import save_model_results
import seaborn as sns
import matplotlib.pyplot as plt

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RF")

# Set plotly renderer for compatibility
import plotly.io as pio
pio.renderers.default = "notebook"


# ==============================================================================
# Command Line Interface
# ==============================================================================

def _build_arg_parser() -> argparse.ArgumentParser:
    """
    Build command-line argument parser for the Random Forest training script.

    This parser defines all configurable parameters for the training pipeline,
    including data paths, cross-validation settings, and optimization parameters.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser with the following arguments:
        - csv: Path to input dataset
        - n_splits: Number of cross-validation folds
        - n_jobs: CPU cores for parallel processing
        - random_state: Seed for reproducibility
        - n_trials: Optuna optimization iterations
    """
    p = argparse.ArgumentParser(
        description="Train Random Forest classifier for vehicular impact prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    p.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Path to raw CSV dataset containing vehicular QoS metrics"
    )

    p.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of GroupKFold splits for temporal cross-validation"
    )

    p.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs (-1 uses all CPU cores)"
    )

    p.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    p.add_argument(
        "--n-trials",
        type=int,
        default=40,
        help="Number of Optuna hyperparameter optimization trials"
    )

    return p


# ==============================================================================
# Main Training Pipeline
# ==============================================================================

def main(args: argparse.Namespace) -> None:
    """
    Execute the complete Random Forest training pipeline.

    This function orchestrates the entire training process:
    1. Data loading and preprocessing
    2. Feature engineering and impact labeling
    3. Train/test split with temporal grouping
    4. Hyperparameter optimization with Optuna
    5. Final model training with best parameters
    6. Performance evaluation and artifact saving

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing:
        - csv: Dataset path
        - n_splits: Cross-validation folds
        - n_jobs: Parallel processing cores
        - random_state: Random seed
        - n_trials: Optimization iterations

    Returns
    -------
    None
        Results are saved to disk in the artifacts directory
    """

    print("\n" + "="*80)
    print("RANDOM FOREST TRAINING PIPELINE")
    print("="*80)

    # ========== Step 1: Data Loading and Preprocessing ==========
    logger.info("Loading dataset...")
    df_raw = pd.read_csv(args.csv)
    df = pp.prepare_dataset(df_raw)
    logger.info(f"Dataset loaded: {len(df)} samples")

    # Define custom weights for impact labeling
    # These weights reflect the relative importance of each metric per application
    weights = WEIGHTS | {
        "s": dict(lat=0.5, loss=0.3, thr=0.2),   # Safety: latency-critical
        "e": dict(lat=0.3, loss=0.4, thr=0.3),   # Efficiency: balanced
        "e2": dict(lat=0.2, loss=0.3, thr=0.5),  # Entertainment: throughput-focused
        "g": dict(lat=0.3, loss=0.3, thr=0.4)    # Generic: best-effort
    }

    # Apply impact labeling to create target variable
    df, X, y, groups, cw, class_weight = label_weighted_average(df, weights=weights)

    logger.info(f"Dataset shape: {X.shape}")
    logger.info(f"Class distribution:\n{pd.Series(y).value_counts().sort_index()}")
    logger.info(f"Class weights: {class_weight}")

    # Save preprocessing artifacts for reproducibility
    print(f"\nSaving preprocessing artifacts...")
    output_dir = save_model_results('rf', X, y, groups, class_weight)

    # ========== Step 2: Feature Preprocessing Pipeline ==========
    # Identify numerical and categorical columns
    num_cols = X.select_dtypes(include=[np.number, bool]).columns.tolist()
    cat_in = [col for col in X.columns if col not in num_cols]

    logger.info(f"Numerical features ({len(num_cols)}): {num_cols[:5]}...")
    logger.info(f"Categorical features ({len(cat_in)}): {cat_in}")

    # Create preprocessing pipeline
    # - StandardScaler for numerical features (zero mean, unit variance)
    # - OneHotEncoder for categorical features (creates binary columns)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_in),
        ]
    ).set_output(transform="pandas")  # Keep feature names for interpretability

    # ========== Step 3: Cross-Validation Setup ==========
    # GroupKFold ensures that data from the same time block stays together
    # This prevents temporal leakage in time-series vehicular data
    gkf = GroupKFold(n_splits=args.n_splits)

    # F1-macro scorer for imbalanced classification
    f1_macro = make_scorer(f1_score, average="macro")

    # Create train/holdout split preserving groups
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, hold_idx = next(gss.split(X, y, groups))

    logger.info(f"Train set: {len(train_idx)} samples")
    logger.info(f"Holdout set: {len(hold_idx)} samples")

    # ========== Step 4: Random Forest Pipeline Setup ==========
    # Create the complete ML pipeline: preprocessing + classifier
    pipe_rf = Pipeline([
        ("prep", preprocessor),
        ("rf", RandomForestClassifier(
            class_weight=class_weight,      # Handle class imbalance
            random_state=args.random_state, # Reproducibility
            n_jobs=-1,                      # Use all CPU cores
        ))
    ])

    # ========== Step 5: Hyperparameter Optimization with Optuna ==========
    print("\n" + "-"*60)
    print("HYPERPARAMETER OPTIMIZATION")
    print("-"*60)

    def objective_rf(trial):
        """
        Optuna objective function for Random Forest hyperparameter optimization.

        This function defines the search space and evaluation strategy for
        finding optimal Random Forest parameters. It uses cross-validation
        F1-macro score as the optimization metric.

        Parameters
        ----------
        trial : optuna.Trial
            Optuna trial object for suggesting hyperparameters

        Returns
        -------
        float
            Mean F1-macro score across all CV folds
        """
        # Define hyperparameter search space
        n_estimators = trial.suggest_int("n_estimators", 200, 800, step=100)
        max_depth = trial.suggest_categorical("max_depth", [None, 10, 20, 30])
        min_samples = trial.suggest_int("min_samples_leaf", 1, 4)
        max_features = trial.suggest_categorical("max_features", ["sqrt", 0.5, 0.8])

        # Create parameter dictionary for sklearn pipeline
        params = dict(
            rf__n_estimators=n_estimators,
            rf__max_depth=max_depth,
            rf__min_samples_leaf=min_samples,
            rf__max_features=max_features
        )

        # Perform cross-validation with GroupKFold
        scores = []
        for fold, (tr, va) in enumerate(gkf.split(X.iloc[train_idx],
                                                   y[train_idx],
                                                   groups[train_idx])):
            # Clone pipeline to ensure clean state
            pipe = clone(pipe_rf).set_params(**params)

            # Train on fold
            pipe.fit(X.iloc[train_idx].iloc[tr], y[train_idx][tr])

            # Predict on validation
            preds = pipe.predict(X.iloc[train_idx].iloc[va])

            # Calculate F1-macro score
            score = f1_score(y[train_idx][va], preds, average="macro")
            scores.append(score)

        return np.mean(scores)

    # Configure Optuna study
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # TPE (Tree-structured Parzen Estimator) sampler for efficient search
    sampler = optuna.samplers.TPESampler(seed=42)

    # Median pruner stops unpromising trials early
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)

    # Create and run optimization study
    study_rf = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner
    )

    n_trials = getattr(args, "n_trials", 40)
    study_rf.optimize(objective_rf, n_trials=n_trials, show_progress_bar=True)

    print(f"\nOptimization completed!")
    print(f"Best parameters: {study_rf.best_params}")
    print(f"Best F1-macro (CV): {study_rf.best_value:.3f}")

    # ========== Step 6: Train Final Model with Best Parameters ==========
    print("\n" + "-"*60)
    print("FINAL MODEL TRAINING")
    print("-"*60)

    best_params_rf = study_rf.best_params
    best_pipe_rf = clone(pipe_rf).set_params(
        rf__n_estimators=best_params_rf["n_estimators"],
        rf__max_depth=best_params_rf["max_depth"],
        rf__min_samples_leaf=best_params_rf["min_samples_leaf"],
        rf__max_features=best_params_rf["max_features"],
    )

    # Train on full training set
    logger.info("Training final model with best parameters...")
    best_pipe_rf.fit(X.iloc[train_idx], y[train_idx])

    # Save trained model
    model_path = output_dir / "rf_best.pkl"
    joblib.dump(best_pipe_rf, model_path)
    print(f"✓ Model saved: {model_path}")

    # ========== Step 7: Holdout Set Evaluation ==========
    print("\n" + "-"*60)
    print("HOLDOUT SET EVALUATION")
    print("-"*60)

    y_pred_rf = best_pipe_rf.predict(X.iloc[hold_idx])

    print("\nClassification Report:")
    print(classification_report(y[hold_idx], y_pred_rf, digits=3))

    # Save comprehensive results
    results_rf = {
        "best_params": best_params_rf,
        "cv_score": study_rf.best_value,
        "holdout_metrics": classification_report(y[hold_idx], y_pred_rf, output_dict=True),
        "study_stats": {
            "n_trials": len(study_rf.trials),
            "n_pruned": len([t for t in study_rf.trials
                           if t.state == optuna.trial.TrialState.PRUNED])
        }
    }

    with open(output_dir / "training_results_rf.json", "w") as f:
        json.dump(results_rf, f, indent=2)
    print(f"✓ Results saved: {output_dir / 'training_results_rf.json'}")

    # ========== Step 8: Visualization ==========
    print("\n" + "-"*60)
    print("GENERATING VISUALIZATIONS")
    print("-"*60)

    # Confusion Matrix (absolute values)
    cm = confusion_matrix(y[hold_idx], y_pred_rf)
    cm_norm = confusion_matrix(y[hold_idx], y_pred_rf, normalize="true")

    # Plot absolute confusion matrix
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted Impact Level")
    ax.set_ylabel("True Impact Level")
    ax.set_title("Random Forest - Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix_rf.png", dpi=300)
    plt.close(fig)

    # Plot normalized confusion matrix
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted Impact Level")
    ax.set_ylabel("True Impact Level")
    ax.set_title("Random Forest - Normalized Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix_norm_rf.png", dpi=300)
    plt.close(fig)
    print("✓ Confusion matrices saved")

    # Optuna optimization history
    fig = vis.plot_optimization_history(study_rf)
    fig.write_html(output_dir / "rf_optuna_history.html")
    print(f"✓ Optimization history saved: {output_dir / 'rf_optuna_history.html'}")

    # Try to display in notebook environment
    try:
        fig.show()
    except:
        pass

    # ========== Step 9: Feature Importance Analysis ==========
    print("\n" + "-"*60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("-"*60)

    # Extract feature names after preprocessing
    feature_names = best_pipe_rf.named_steps['prep'].get_feature_names_out()

    # Get feature importances from Random Forest
    importances = best_pipe_rf.named_steps['rf'].feature_importances_

    # Create importance dataframe and sort
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    # Display top 10 features
    print("\nTop 10 Most Important Features:")
    for idx, row in importance_df.head(10).iterrows():
        print(f"{row['feature']:30s}: {row['importance']:.4f}")

    # Save feature importances
    importance_df.to_csv(output_dir / "feature_importances_rf.csv", index=False)
    print(f"\n✓ Feature importances saved: {output_dir / 'feature_importances_rf.csv'}")

    print("\n" + "="*80)
    print("RANDOM FOREST TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)


# ==============================================================================
# Script Entry Point
# ==============================================================================

if __name__ == "__main__":
    """
    Entry point for direct script execution.
    Parses command-line arguments and runs the main training pipeline.
    """
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Validate input file exists
    if not args.csv.exists():
        raise FileNotFoundError(f"Dataset not found: {args.csv}")

    # Run training pipeline
    main(args)