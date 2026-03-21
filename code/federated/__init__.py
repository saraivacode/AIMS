#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIMS Framework - Federated Learning Package
============================================

Provides federated learning capabilities for the AIMS framework,
supporting DNN, LSTM, and GRU models with FedAvg and FedProx
aggregation strategies over IID and Non-IID data distributions,
including security experiments with adversarial attacks and defenses.
"""

from .fl_main import (
    run_federated,
    run_security_phase1a,
    run_security_phase1b,
    run_security_sensitivity_epochs,
)

__all__ = [
    "run_federated",
    "run_security_phase1a",
    "run_security_phase1b",
    "run_security_sensitivity_epochs",
]
