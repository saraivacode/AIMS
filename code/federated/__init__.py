#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIMS Framework - Federated Learning Package
============================================

Provides federated learning capabilities for the AIMS framework,
supporting DNN, LSTM, and GRU models with FedAvg and FedProx
aggregation strategies over IID and Non-IID data distributions.
"""

from .fl_main import run_federated

__all__ = ["run_federated"]
