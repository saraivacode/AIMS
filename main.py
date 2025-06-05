#!/usr/bin/env python3
"""
AIMS Framework - Main Training Pipeline
======================================

This script orchestrates the training of three machine learning models for 
network slicing impact classification in Intelligent Transportation Systems (ITS).

The pipeline trains and evaluates:
1. Random Forest (RF) - Tree-based ensemble with bagging
2. TabNet - Deep learning with attention mechanisms for tabular data
3. CatBoost - Gradient boosting with native categorical feature support

Each model is trained with:
- GroupKFold cross-validation (5 splits) to prevent temporal leakage
- Hyperparameter optimization using Optuna
- Class-balanced weights to handle imbalanced data
- Consistent train/test split for fair comparison

Usage:
    python main.py

Input:
    - Dataset: d3/df_completo_com_interpolacao.csv
    - Contains vehicular network QoS metrics (RTT, PDR, throughput)

Output:
    - Trained models saved as .pkl files
    - Performance metrics in JSON format
    - Confusion matrices as PNG images
    - Optuna optimization history plots

Requirements:
    - Python 3.8+
    - scikit-learn, catboost, pytorch-tabnet
    - See requirements.txt for complete list

Author: Tiago do Vale Saraiva
License: MIT
"""

import argparse
import train_model_rf
import train_model_catboost
from train_model_tabnet import run_tabnet


def main():
    """
    Main execution function that trains all three models sequentially.

    The models are trained in the following order:
    1. Random Forest - Fast training, good baseline
    2. TabNet - Moderate training time, attention insights
    3. CatBoost - Fast with GPU, handles categoricals well

    All models use the same dataset and evaluation strategy for 
    fair comparison.
    """
    # Path to the preprocessed vehicular dataset
    # This dataset contains 158 vehicles' network metrics over 450 seconds
    dataset = "d3/df_completo_com_interpolacao.csv"

    # Common arguments for all models
    # These ensure consistent experimental setup
    args = argparse.Namespace(
        csv=dataset,
        n_splits=5,  # GroupKFold splits for temporal validation
        n_jobs=-1,  # Use all CPU cores for parallel processing
        random_state=42,  # Reproducibility seed
        n_trials=40  # Optuna hyperparameter search iterations
    )

    print("=" * 80)
    print("AIMS Framework - Training Pipeline")
    print("=" * 80)

    # ==================== Random Forest Training ====================
    print("\n[1/3] Training Random Forest Classifier...")
    print("-" * 60)
    train_model_rf.main(args)
    print("✓ Random Forest training completed")

    # ==================== TabNet Training ====================
    print("\n[2/3] Training TabNet Classifier...")
    print("-" * 60)
    # TabNet has its own interface, so we call it differently
    run_tabnet(
        csv_path=dataset,
        n_splits=5,
        n_trials=20,  # Fewer trials for TabNet (computationally expensive)
        random_state=42
    )
    print("✓ TabNet training completed")

    # ==================== CatBoost Training ====================
    print("\n[3/3] Training CatBoost Classifier...")
    print("-" * 60)
    train_model_catboost.main(args)
    print("✓ CatBoost training completed")

    print("\n" + "=" * 80)
    print("All models trained successfully!")
    print("Results saved in: drive/MyDrive/Colab Notebooks/vtm2025/artifacts/")
    print("=" * 80)


if __name__ == '__main__':
    main()