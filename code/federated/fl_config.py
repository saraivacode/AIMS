#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIMS Framework - Federated Learning Configuration
==================================================

Centralized configuration constants for all FL experiments.
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class FLDefaults:
    """Default configuration values for Federated Learning experiments."""

    # FL topology
    NUM_CLIENTS: int = 3
    NUM_ROUNDS: int = 10
    LOCAL_EPOCHS: int = 5
    BATCH_SIZE: int = 32

    # Aggregation strategies
    STRATEGIES: List[str] = field(default_factory=lambda: ["FedAvg", "FedProx"])
    FEDPROX_MU: float = 0.1

    # Data distribution modes
    DISTRIBUTIONS: List[str] = field(default_factory=lambda: ["IID", "NonIID"])

    # Neural network models
    MODEL_TYPES: List[str] = field(default_factory=lambda: ["DNN", "LSTM", "GRU"])

    # DNN architecture
    DNN_HIDDEN_LAYERS: List[int] = field(default_factory=lambda: [128, 64, 32])
    DROPOUT_RATE: float = 0.3
    L2_REG: float = 0.001

    # RNN units (LSTM / GRU)
    RNN_UNITS_1: int = 64
    RNN_UNITS_2: int = 32

    # Classification
    NUM_CLASSES: int = 4
    CLASS_NAMES: List[str] = field(
        default_factory=lambda: ["Adequate", "Warning", "Severe", "Critical"]
    )

    # Centralized baseline
    CENTRALIZED_EPOCHS: int = 30
    EARLY_STOPPING_PATIENCE: int = 5

    # Reproducibility
    RANDOM_STATE: int = 42

    # Results sub-directory
    RESULTS_SUBDIR: str = "federated"


# Non-IID allocation matrix: rows = clients, cols = classes (0,1,2,3)
# Each column sums to 1.0 -> all data used, no loss.
NON_IID_ALLOCATION = np.array([
    [0.55, 0.30, 0.10, 0.05],   # Client 0: primarily Adequate + Warning
    [0.15, 0.40, 0.40, 0.15],   # Client 1: primarily Warning + Severe
    [0.30, 0.30, 0.50, 0.80],   # Client 2: primarily Severe + Critical
])
# Note: columns sum to [1.0, 1.0, 1.0, 1.0]
