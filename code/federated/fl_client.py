#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIMS Framework - Flower Client for Federated Learning
======================================================

Implements the Flower NumPyClient that wraps Keras models for federated
training, supporting both FedAvg and FedProx (with proximal term).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
from sklearn.metrics import f1_score

from .fl_models import create_model


class AIMSFlowerClient(fl.client.NumPyClient):
    """
    Flower client wrapping a Keras model for the AIMS FL experiments.

    Each client holds a local partition of the dataset and trains
    the model locally for a configurable number of epochs per round.

    Parameters
    ----------
    model_type : str
        Neural network architecture ("DNN", "LSTM", "GRU").
    X_train : np.ndarray
        Local training features.
    y_train : np.ndarray
        Local training labels.
    X_val : np.ndarray
        Local validation features.
    y_val : np.ndarray
        Local validation labels.
    local_epochs : int
        Number of training epochs per FL round.
    batch_size : int
        Training batch size.
    class_weights : dict
        Class weight dictionary for imbalanced learning.
    client_id : int
        Identifier for this client.
    proximal_mu : float
        FedProx proximal term coefficient. 0 = standard FedAvg.
    """

    def __init__(
        self,
        model_type: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        local_epochs: int,
        batch_size: int,
        class_weights: Dict[int, float],
        client_id: int = 0,
        proximal_mu: float = 0.0,
    ):
        self.model_type = model_type
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.class_weights = class_weights
        self.client_id = client_id
        self.proximal_mu = proximal_mu

        input_dim = X_train.shape[1]
        self.model = create_model(model_type, input_dim)

    def _prepare_input(self, X: np.ndarray) -> np.ndarray:
        """Reshape input for LSTM/GRU models (add timestep dimension)."""
        if self.model_type.upper() in ("LSTM", "GRU"):
            return X.reshape((X.shape[0], 1, X.shape[1]))
        return X

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Return model weights as a list of numpy arrays."""
        return self.model.get_weights()

    def fit(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Train the local model for local_epochs.

        If proximal_mu > 0, applies FedProx proximal updates after each epoch.
        """
        self.model.set_weights(parameters)

        X_train = self._prepare_input(self.X_train)

        if self.proximal_mu > 0:
            # FedProx: train epoch-by-epoch with proximal correction
            global_weights = [w.copy() for w in parameters]
            for _ in range(self.local_epochs):
                self.model.fit(
                    X_train, self.y_train,
                    epochs=1,
                    batch_size=self.batch_size,
                    class_weight=self.class_weights,
                    verbose=0,
                )
                # Proximal update: w <- w - mu * (w - w_global)
                new_weights = self.model.get_weights()
                prox_weights = [
                    w - self.proximal_mu * (w - gw)
                    for w, gw in zip(new_weights, global_weights)
                ]
                self.model.set_weights(prox_weights)
        else:
            # Standard FedAvg training
            self.model.fit(
                X_train, self.y_train,
                epochs=self.local_epochs,
                batch_size=self.batch_size,
                class_weight=self.class_weights,
                verbose=0,
            )

        return self.model.get_weights(), len(self.X_train), {}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[float, int, Dict]:
        """Evaluate the model on the local validation set."""
        self.model.set_weights(parameters)

        X_val = self._prepare_input(self.X_val)

        loss, accuracy = self.model.evaluate(X_val, self.y_val, verbose=0)
        preds = np.argmax(self.model.predict(X_val, verbose=0), axis=1)
        f1 = f1_score(self.y_val, preds, average="macro", zero_division=0)

        return loss, len(self.X_val), {
            "accuracy": float(accuracy),
            "f1_macro": float(f1),
        }
