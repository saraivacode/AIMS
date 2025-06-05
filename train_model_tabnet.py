#!/usr/bin/env python3
"""
TabNet Training Module for AIMS Framework
=========================================

This module implements the TabNet (Tabular Attention Network) deep learning
classifier for the AIMS framework. TabNet is a novel deep learning architecture
specifically designed for tabular data, offering interpretability through
attention mechanisms while achieving state-of-the-art performance.

Key Features:
    - Attention-based feature selection (learns which features to use)
    - End-to-end learning without extensive preprocessing
    - Built-in feature importance through attention masks
    - Sparse feature selection for interpretability
    - Sequential attention for decision-making transparency
    - GPU acceleration through PyTorch backend

Model Characteristics:
    TabNet advantages over traditional methods:
    - No need for feature engineering (learns representations)
    - Handles high-cardinality categorical features
    - Provides instance-wise feature selection
    - Mimics decision tree behavior with neural network power
    - Built-in regularization through sparsity

Architecture Details:
    - Sequential multi-step architecture
    - Soft feature selection using Sparsemax/Entmax
    - Virtual batch normalization for stability
    - Shared and independent GLU blocks

Performance Metrics:
    - Typical accuracy: 95%+ on vehicular QoS classification
    - Training time: 10-20 minutes (GPU recommended)
    - Inference time: ~10ms per sample
    - Best performance on ambiguous impact cases

Usage:
    Direct execution:
        python train_model_tabnet.py --csv data.csv --n-splits 5 --n-trials 20

    As module:
        from train_model_tabnet import run_tabnet
        run_tabnet(csv_path, n_splits=5, n_trials=20)

GPU Support:
    TabNet requires PyTorch and automatically uses GPU if available.
    CPU fallback is supported but significantly slower.

Requirements:
    - pytorch-tabnet >= 4.0.0
    - torch >= 1.10.0
    - scikit-learn >= 1.0.0
    - optuna >= 3.0.0
    - pandas, numpy, matplotlib
    - See requirements.txt for complete list

Author: Tiago do Vale Saraiva
License: MIT
"""

import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from pytorch_tabnet.tab_model import TabNetClassifier
import preprocess_dataset as pp
from impact_labeling import label_weighted_average, WEIGHTS
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import optuna.visualization as vis
import warnings
from save_utils import save_model_results

# Suppress TabNet UserWarnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# Configure matplotlib for better rendering
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ==============================================================================
# Main Training Function
# ==============================================================================

def run_tabnet(
        csv_path: str,
        n_splits: int = 5,
        n_trials: int = 20,
        random_state: int = 42,
        return_study: bool = False
) -> Optional[Tuple[optuna.Study, TabNetClassifier, ColumnTransformer]]:
    """
    Execute the complete TabNet training pipeline.

    This function implements an end-to-end training workflow for TabNet,
    specifically optimized for vehicular network impact classification.
    The pipeline includes:

    1. Data loading and preprocessing
    2. Feature encoding (numerical scaling, categorical encoding)
    3. Temporal-aware train/test splitting
    4. Hyperparameter optimization with Optuna
    5. Final model training with early stopping
    6. Comprehensive evaluation and visualization

    TabNet's attention mechanism makes it particularly suitable for
    understanding which network metrics drive impact classifications,
    providing interpretability crucial for network operators.

    Parameters
    ----------
    csv_path : str
        Path to the CSV dataset containing vehicular network metrics.
        Expected to contain RTT, PDR, throughput, and categorical features.

    n_splits : int, default=5
        Number of folds for GroupKFold cross-validation.
        Groups prevent temporal leakage in time-series data.

    n_trials : int, default=20
        Number of Optuna trials for hyperparameter optimization.
        Fewer trials than tree-based methods due to computational cost.

    random_state : int, default=42
        Random seed for reproducibility across all operations.

    return_study : bool, default=False
        If True, returns the Optuna study, trained model, and preprocessor.
        Useful for further analysis or model inspection.

    Returns
    -------
    Optional[Tuple[optuna.Study, TabNetClassifier, ColumnTransformer]]
        If return_study=True, returns:
        - study: Optuna optimization study with trial history
        - clf: Trained TabNet classifier
        - preprocessor: Fitted preprocessing pipeline
        Otherwise returns None

    Notes
    -----
    TabNet requires categorical features to be encoded as integers.
    This implementation uses OrdinalEncoder with special handling
    for unknown categories during inference.
    """

    print("\n" + "=" * 80)
    print("TABNET TRAINING PIPELINE")
    print("=" * 80)

    # ========== Step 1: Data Loading and Preprocessing ==========
    print("\n[Step 1/9] Loading and preprocessing dataset...")
    df_raw = pd.read_csv(csv_path)
    df = pp.prepare_dataset(df_raw)

    # Apply custom weights for impact labeling
    # These weights are calibrated for vehicular applications
    weights = WEIGHTS | {
        "s": dict(lat=0.5, loss=0.3, thr=0.2),  # Safety-critical
        "e": dict(lat=0.3, loss=0.4, thr=0.3),  # Efficiency
        "e2": dict(lat=0.2, loss=0.3, thr=0.5),  # Entertainment
        "g": dict(lat=0.3, loss=0.3, thr=0.4)  # Generic
    }

    df, X, y, groups, cw, class_weight = label_weighted_average(df, weights=weights)

    print(f"✓ Dataset shape: {X.shape}")
    print(f"✓ Impact distribution:\n{pd.Series(y).value_counts().sort_index()}")
    print(f"✓ Class weights: {class_weight}")

    # Save artifacts for reproducibility
    print(f"\nSaving preprocessing artifacts for TabNet...")
    output_dir = save_model_results('tabnet', X, y, groups, class_weight)

    # ========== Step 2: Feature Engineering for TabNet ==========
    print("\n[Step 2/9] Preparing features for TabNet...")

    # Identify feature types
    num_cols = X.select_dtypes(include=[np.number, bool]).columns.tolist()
    cat_cols = [col for col in X.columns if col not in num_cols]

    print(f"✓ Numerical features ({len(num_cols)}): {num_cols[:5]}...")
    print(f"✓ Categorical features ({len(cat_cols)}): {cat_cols}")

    # TabNet requires categorical features to be encoded as integers
    # We use OrdinalEncoder instead of OneHotEncoder
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OrdinalEncoder(
            handle_unknown="use_encoded_value",  # Handle unseen categories
            unknown_value=-1  # Encode unknown as -1
        ), cat_cols),
    ])

    # Transform features
    X_proc = pd.DataFrame(
        preprocessor.fit_transform(X),
        columns=num_cols + cat_cols,
        index=X.index
    )

    # TabNet needs to know which columns are categorical and their dimensions
    cat_idxs = [X_proc.columns.get_loc(col) for col in cat_cols]
    cat_dims = [int(X[col].nunique()) for col in cat_cols]

    print(f"✓ Categorical indices: {cat_idxs}")
    print(f"✓ Categorical dimensions: {cat_dims}")

    # ========== Step 3: Train/Test Split with Temporal Grouping ==========
    print("\n[Step 3/9] Creating temporal-aware data splits...")

    # GroupKFold for cross-validation
    gkf = GroupKFold(n_splits=n_splits)

    # Holdout split preserving groups
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    train_idx, hold_idx = next(gss.split(X_proc, y, groups))

    # Extract train/holdout sets
    X_train, y_train, groups_train = X_proc.iloc[train_idx], y[train_idx], groups[train_idx]
    X_hold, y_hold = X_proc.iloc[hold_idx], y[hold_idx]

    print(f"✓ Training set: {len(train_idx)} samples ({len(train_idx) / len(X) * 100:.1f}%)")
    print(f"✓ Holdout set: {len(hold_idx)} samples ({len(hold_idx) / len(X) * 100:.1f}%)")

    # ========== Step 4: Device Configuration ==========
    print("\n[Step 4/9] Configuring computing device...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✓ GPU detected: {gpu_name}")
    else:
        print("⚠ No GPU detected, using CPU (training will be slower)")

    # Convert class weights to PyTorch tensor
    class_weights_tensor = torch.tensor(cw, dtype=torch.float32, device=device)

    # ========== Step 5: Hyperparameter Optimization ==========
    print("\n[Step 5/9] Starting hyperparameter optimization with Optuna...")
    print("-" * 60)

    def objective(trial):
        """
        Optuna objective function for TabNet hyperparameter optimization.

        Explores key TabNet parameters:
        - n_d, n_a: Width of decision prediction and attention embedding
        - n_steps: Number of sequential attention steps
        - gamma: Relaxation factor for feature reuse
        - lambda_sparse: Sparsity regularization strength
        - learning_rate: Optimization step size
        - mask_type: Attention activation function

        Parameters
        ----------
        trial : optuna.Trial
            Optuna trial for hyperparameter suggestions

        Returns
        -------
        float
            Mean F1-macro score across CV folds
        """
        # Network architecture parameters
        # n_d and n_a control the network width
        n_d = trial.suggest_categorical("n_d", [8, 16, 32])
        n_a = trial.suggest_categorical("n_a", [8, 16, 32])

        # Number of decision steps (depth)
        n_steps = trial.suggest_int("n_steps", 3, 5)

        # Feature reuse coefficient
        gamma = trial.suggest_float("gamma", 1.0, 1.8, step=0.2)

        # Sparsity regularization (encourages focused attention)
        lambda_sparse = trial.suggest_float("lambda_sparse", 1e-4, 1e-3, log=True)

        # Learning rate
        lr = trial.suggest_float("lr", 1e-3, 3e-2, log=True)

        # Attention activation function
        mask_type = trial.suggest_categorical("mask_type", ["sparsemax", "entmax"])

        # Create TabNet classifier with trial parameters
        clf = TabNetClassifier(
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            lambda_sparse=lambda_sparse,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=lr),
            mask_type=mask_type,
            seed=random_state,
            verbose=0,  # Suppress training output
            device_name=device,
            cat_idxs=cat_idxs,
            cat_dims=cat_dims,
            cat_emb_dim=1,  # Embedding dimension for categoricals
        )

        # Cross-validation with temporal groups
        fold_scores = []
        for fold_idx, (tr, va) in enumerate(gkf.split(X_train, y_train, groups_train)):
            # Extract fold data
            X_tr, y_tr = X_train.iloc[tr].values, y_train[tr]
            X_va, y_va = X_train.iloc[va].values, y_train[va]

            # Train with early stopping
            clf.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                eval_metric=["accuracy"],  # TabNet's built-in metrics
                loss_fn=torch.nn.CrossEntropyLoss(weight=class_weights_tensor),
                patience=20,  # Early stopping patience
                max_epochs=50  # Maximum training epochs
            )

            # Evaluate on validation fold
            preds = clf.predict(X_va)
            fold_score = f1_score(y_va, preds, average="macro")
            fold_scores.append(fold_score)

        return np.mean(fold_scores)

    # Configure and run Optuna study
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state)
    )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\n✓ Optimization completed!")
    print(f"Best parameters found:")
    for param, value in study.best_params.items():
        print(f"  {param}: {value}")
    print(f"Best F1-macro (CV): {study.best_value:.3f}")

    # ========== Step 6: Train Final Model ==========
    print("\n[Step 6/9] Training final TabNet model with best parameters...")

    best = study.best_params
    clf = TabNetClassifier(
        # Architecture parameters
        n_d=best["n_d"],
        n_a=best["n_a"],
        n_steps=best["n_steps"],
        gamma=best["gamma"],
        lambda_sparse=best["lambda_sparse"],

        # Optimization parameters
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=best["lr"]),
        mask_type=best["mask_type"],

        # Other parameters
        seed=random_state,
        verbose=1,  # Show training progress
        device_name=device,
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        cat_emb_dim=1,
    )

    # Train with early stopping
    print("\nTraining with early stopping (patience=30)...")
    clf.fit(
        X_train.values, y_train,
        eval_set=[(X_hold.values, y_hold)],
        eval_metric=["accuracy"],
        loss_fn=torch.nn.CrossEntropyLoss(weight=class_weights_tensor),
        patience=30,  # More patience for final model
        max_epochs=200  # Allow longer training
    )

    # Save model and preprocessor
    model_path = output_dir / "tabnet_best.pkl"
    joblib.dump((preprocessor, clf), model_path)
    print(f"\n✓ Model saved: {model_path}")

    # ========== Step 7: Evaluation ==========
    print("\n[Step 7/9] Evaluating on holdout set...")
    print("-" * 60)

    # Predict holdout set
    y_pred = clf.predict(X_hold.values)

    # Generate classification report
    print("\nClassification Report:")
    print(classification_report(y_hold, y_pred, digits=3))

    # Get detailed metrics
    report_dict = classification_report(y_hold, y_pred, output_dict=True)

    # Highlight key metrics
    print("\nKey Performance Indicators:")
    print(f"  • Overall Accuracy: {report_dict['accuracy']:.1%}")
    print(f"  • Macro F1-Score: {report_dict['macro avg']['f1-score']:.1%}")
    print(f"  • Weighted F1-Score: {report_dict['weighted avg']['f1-score']:.1%}")

    # Per-class analysis
    print("\nPer-Class F1-Scores:")
    for class_idx in ['0', '1', '2', '3']:
        if class_idx in report_dict:
            f1 = report_dict[class_idx]['f1-score']
            support = report_dict[class_idx]['support']
            print(f"  • Class {class_idx}: {f1:.3f} (n={int(support)})")

    # Save results
    results_tabnet = {
        "best_params": best,
        "cv_score": study.best_value,
        "holdout_metrics": report_dict,
        "study_stats": {
            "n_trials": len(study.trials),
            "n_pruned": len([t for t in study.trials
                             if t.state == optuna.trial.TrialState.PRUNED])
        },
        "device": device,
        "model_stats": {
            "n_d": best["n_d"],
            "n_a": best["n_a"],
            "n_steps": best["n_steps"],
            "total_params": sum(p.numel() for p in clf.network.parameters())
        }
    }

    with open(output_dir / "training_results_tabnet.json", "w") as f:
        json.dump(results_tabnet, f, indent=2)
    print(f"\n✓ Results saved: {output_dir / 'training_results_tabnet.json'}")

    # ========== Step 8: Visualizations ==========
    print("\n[Step 8/9] Generating visualizations...")

    # Confusion matrices
    cm = confusion_matrix(y_hold, y_pred)
    cm_norm = confusion_matrix(y_hold, y_pred, normalize="true")

    # Plot absolute confusion matrix
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=['Low', 'Minor', 'Major', 'Critical'],
                yticklabels=['Low', 'Minor', 'Major', 'Critical'])
    ax.set_xlabel("Predicted Impact Level")
    ax.set_ylabel("True Impact Level")
    ax.set_title("TabNet - Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix_tabnet.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Plot normalized confusion matrix
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", ax=ax,
                xticklabels=['Low', 'Minor', 'Major', 'Critical'],
                yticklabels=['Low', 'Minor', 'Major', 'Critical'])
    ax.set_xlabel("Predicted Impact Level")
    ax.set_ylabel("True Impact Level")
    ax.set_title("TabNet - Normalized Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix_norm_tabnet.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("✓ Confusion matrices saved")

    # Optuna optimization history
    fig = vis.plot_optimization_history(study)
    fig.update_layout(
        title="TabNet Hyperparameter Optimization History",
        xaxis_title="Trial Number",
        yaxis_title="F1-macro Score",
        showlegend=True
    )
    fig.write_html(output_dir / "tabnet_optuna_history.html")
    print(f"✓ Optimization history saved: {output_dir / 'tabnet_optuna_history.html'}")

    # Try to display in notebook
    try:
        fig.show()
    except:
        pass

    # ========== Step 9: Feature Importance via Attention ==========
    print("\n[Step 9/9] Analyzing feature importance through attention masks...")

    # Get global feature importance from TabNet
    # This represents how often each feature is selected by attention
    feature_importances = clf.feature_importances_

    # Create importance dataframe
    feature_names = num_cols + cat_cols
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importances
    }).sort_values('importance', ascending=False)

    # Display top features
    print("\nTop 10 Most Important Features (by attention selection):")
    print("-" * 60)
    for idx, row in importance_df.head(10).iterrows():
        print(f"{row['feature']:30s}: {row['importance']:8.4f}")

    # Save feature importances
    importance_df.to_csv(output_dir / "feature_importances_tabnet.csv", index=False)
    print(f"\n✓ Feature importances saved: {output_dir / 'feature_importances_tabnet.csv'}")

    # Plot feature importance
    fig, ax = plt.subplots(figsize=(10, 6))
    top_features = importance_df.head(15)
    ax.barh(top_features['feature'], top_features['importance'])
    ax.set_xlabel('Importance Score')
    ax.set_title('TabNet Feature Importance (Top 15)')
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(output_dir / "feature_importance_tabnet.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("✓ Feature importance plot saved")

    # ========== Summary ==========
    print("\n" + "=" * 80)
    print("TABNET TRAINING SUMMARY")
    print("=" * 80)
    print(f"✓ Model Architecture: TabNet (n_d={best['n_d']}, n_a={best['n_a']}, steps={best['n_steps']})")
    print(f"✓ Total Parameters: {results_tabnet['model_stats']['total_params']:,}")
    print(f"✓ Device: {device.upper()}")
    print(f"✓ Best CV Score: {study.best_value:.3f}")
    print(f"✓ Holdout Accuracy: {report_dict['accuracy']:.1%}")
    print(f"✓ Training completed in {len(study.trials)} trials")
    print(f"✓ All artifacts saved to: {output_dir}")
    print("=" * 80)

    # Return study and model if requested
    if return_study:
        return study, clf, preprocessor
    else:
        return None


# ==============================================================================
# Command Line Interface
# ==============================================================================

if __name__ == "__main__":
    """
    Entry point for direct script execution.
    Provides command-line interface for TabNet training.
    """
    parser = argparse.ArgumentParser(
        description="Train TabNet deep learning model for vehicular impact classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Path to CSV dataset with vehicular QoS metrics"
    )

    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of GroupKFold cross-validation splits"
    )

    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Number of Optuna optimization trials"
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Validate dataset exists
    if not args.csv.exists():
        raise FileNotFoundError(f"Dataset not found: {args.csv}")

    # Print configuration
    print("TabNet Training Configuration:")
    print(f"  Dataset: {args.csv}")
    print(f"  CV Splits: {args.n_splits}")
    print(f"  Optuna Trials: {args.n_trials}")
    print(f"  Random State: {args.random_state}")

    # Run training
    run_tabnet(
        str(args.csv),
        n_splits=args.n_splits,
        n_trials=args.n_trials,
        random_state=args.random_state,
        return_study=False
    )