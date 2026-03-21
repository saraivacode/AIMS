#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIMS Framework - Federated Learning Neural Network Models
==========================================================

Defines DNN, LSTM, and GRU model builders using Keras/TensorFlow,
adapted from the reference FL script for the AIMS 4-class impact
classification task.
"""

from __future__ import annotations

import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from keras.layers import Dense, Dropout, GRU, Input, LSTM
from keras.models import Sequential
from keras.regularizers import l2

from .fl_config import FLDefaults

_DEFAULTS = FLDefaults()


def configure_tf():
    """Configure TensorFlow for safe GPU usage with memory growth."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical = tf.config.list_logical_devices("GPU")
            print(f"  TF GPU detected: {len(gpus)} physical, {len(logical)} logical")
        except RuntimeError as e:
            print(f"  TF GPU config warning: {e}")
    else:
        print("  TF running on CPU.")


def build_dnn(input_dim: int, num_classes: int = 4) -> tf.keras.Model:
    """
    Build a Dense feedforward network for tabular classification.

    Architecture: Input -> Dense(128) -> Dropout -> Dense(64) -> Dropout
                  -> Dense(32) -> Dense(num_classes, softmax)
    """
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation="relu", kernel_regularizer=l2(_DEFAULTS.L2_REG)),
        Dropout(_DEFAULTS.DROPOUT_RATE),
        Dense(64, activation="relu", kernel_regularizer=l2(_DEFAULTS.L2_REG)),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_lstm(input_dim: int, num_classes: int = 4) -> tf.keras.Model:
    """
    Build an LSTM network for sequential feature processing.

    Input is reshaped to (1, input_dim) treating features as a single timestep.
    Architecture: LSTM(64) -> Dropout -> LSTM(32) -> Dropout -> Dense(softmax)
    """
    model = Sequential([
        Input(shape=(1, input_dim)),
        LSTM(_DEFAULTS.RNN_UNITS_1, return_sequences=True),
        Dropout(_DEFAULTS.DROPOUT_RATE),
        LSTM(_DEFAULTS.RNN_UNITS_2),
        Dropout(0.2),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_gru(input_dim: int, num_classes: int = 4) -> tf.keras.Model:
    """
    Build a GRU network as an efficient LSTM alternative.

    Architecture: GRU(64) -> Dropout -> GRU(32) -> Dropout -> Dense(softmax)
    """
    model = Sequential([
        Input(shape=(1, input_dim)),
        GRU(_DEFAULTS.RNN_UNITS_1, return_sequences=True),
        Dropout(_DEFAULTS.DROPOUT_RATE),
        GRU(_DEFAULTS.RNN_UNITS_2),
        Dropout(0.2),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def create_model(model_type: str, input_dim: int, num_classes: int = 4) -> tf.keras.Model:
    """
    Factory function to create a neural network model.

    Parameters
    ----------
    model_type : str
        One of "DNN", "LSTM", "GRU" (case-insensitive).
    input_dim : int
        Number of input features.
    num_classes : int
        Number of output classes (default 4 for AIMS).

    Returns
    -------
    tf.keras.Model
        Compiled Keras model ready for training.
    """
    builders = {
        "dnn": build_dnn,
        "lstm": build_lstm,
        "gru": build_gru,
    }
    key = model_type.lower()
    if key not in builders:
        raise ValueError(f"Unknown model type '{model_type}'. Choose from: {list(builders.keys())}")
    return builders[key](input_dim, num_classes)
