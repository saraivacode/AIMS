#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIMS Framework - Centralized Training Baselines for FL Comparison
=================================================================

Trains the same DNN/LSTM/GRU models in centralized mode (no federation)
to serve as baselines for evaluating federated learning performance.
"""

from __future__ import annotations

import time
from typing import Dict

import numpy as np
from keras.callbacks import EarlyStopping
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
)

from .fl_config import FLDefaults
from .fl_models import create_model

_DEFAULTS = FLDefaults()


def train_centralized(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_weights: Dict[int, float],
    epochs: int = None,
    batch_size: int = 32,
    seed: int = 42,
) -> Dict:
    """
    Train a neural network model in centralized (non-federated) mode.

    Uses the same model architecture as the FL experiments for fair comparison.
    Total epochs defaults to FL_ROUNDS * LOCAL_EPOCHS to match total compute.

    Parameters
    ----------
    model_type : str
        Neural network type ("DNN", "LSTM", "GRU").
    X_train, y_train : np.ndarray
        Training data.
    X_test, y_test : np.ndarray
        Test data.
    class_weights : dict
        Class weights for imbalanced learning.
    epochs : int, optional
        Training epochs. Defaults to CENTRALIZED_EPOCHS.
    batch_size : int
        Training batch size.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Training results with metrics and history.
    """
    if epochs is None:
        epochs = _DEFAULTS.CENTRALIZED_EPOCHS

    print(f"\n  Centralized training: {model_type.upper()} ({epochs} epochs)")

    input_dim = X_train.shape[1]
    model = create_model(model_type, input_dim)

    # Reshape for sequential models
    X_tr = X_train
    X_te = X_test
    if model_type.upper() in ("LSTM", "GRU"):
        X_tr = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_te = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    # Early stopping callback
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=_DEFAULTS.EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )

    start_time = time.time()

    history = model.fit(
        X_tr, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_te, y_test),
        class_weight=class_weights,
        callbacks=[early_stop],
        verbose=1,
    )

    total_time = time.time() - start_time

    # Evaluate
    preds = np.argmax(model.predict(X_te, verbose=0), axis=1)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="macro", zero_division=0)
    rec = recall_score(y_test, preds, average="macro", zero_division=0)
    f1 = f1_score(y_test, preds, average="macro", zero_division=0)
    f1_per_class = f1_score(y_test, preds, average=None, zero_division=0)

    print(f"    Result: Acc={acc:.4f}, F1={f1:.4f}, Time={total_time:.1f}s")

    return {
        "config_name": f"{model_type}_centralized",
        "model_type": model_type,
        "mode": "centralized",
        "epochs_trained": len(history.history["loss"]),
        "final_metrics": {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_macro": float(f1),
            "f1_per_class": [float(v) for v in f1_per_class],
        },
        "history": {
            "loss": [float(v) for v in history.history["loss"]],
            "val_loss": [float(v) for v in history.history.get("val_loss", [])],
            "accuracy": [float(v) for v in history.history.get("accuracy", [])],
            "val_accuracy": [float(v) for v in history.history.get("val_accuracy", [])],
        },
        "total_time": total_time,
    }
