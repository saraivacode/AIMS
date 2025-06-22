#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIMS Framework - Standardized Experiment Results Manager
========================================================

This module defines the `ResultsManager` class, a centralized tool for
collecting, organizing, and serializing all artifacts and metadata generated
during machine learning experiments within the AIMS (Adaptive and Intelligent
Management of Slicing) framework.

Key Features:
    - Unified JSON output format across all ML models (RandomForest, CatBoost, TabNet)
    - Automatic capture of system information, timestamps, and library versions
    - Logging of cross-validation metrics, holdout performance, and hyperparameter optimization (HPO) statistics
    - Tracking of saved artifact paths (models, plots, feature importance, etc.)
    - Fluent API for progressive result composition

Usage Example:
    $ rm = ResultsManager("RandomForest", output_dir)
    $ rm.set_data_info(...)
    $ rm.set_best_params(...)
    $ rm.set_holdout_metrics(...)
    $ rm.save("training_results_rf.json")
    $ print(rm.get_summary())

Author: Tiago do Vale Saraiva
License: MIT
"""

from __future__ import annotations

import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from importlib import metadata
from importlib.metadata import PackageNotFoundError
#import importlib_metadata as metadata # Python < 3.8

import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.metrics import classification_report


class ResultsManager:
    """Centralized manager for collecting and saving ML experiment results."""

    def __init__(self, model_name: str, output_dir: Union[str, Path]):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.start_time = time.time()

        self.results: Dict[str, Any] = {
            "model_info": {
                "model_name": model_name,
                "library": self._get_library_info(model_name),
            },
            "run_info": {
                "framework_version": "AIMS 1.0.0",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "system": self._get_system_info(),
            },
            "data_info": {},
            "hyperparameters": {"best_params": {}},
            "metrics": {"cross_validation": {}, "holdout": {}},
            "performance": {"hpo_stats": {}, "training_stats": {}},
            "artifacts": {},
        }

    _MODEL2PKG: Dict[str, str] = {
        "RandomForest": "scikit-learn",
        "CatBoost": "catboost",
        "TabNet": "pytorch-tabnet",
    }

    @staticmethod
    def _get_library_info(model_name: str) -> str:
        """
        Return "library vX.Y.Z" for a supported model.

        If the package is missing, returns "<library> (not installed)".
        For unknown models, returns "unknown".
        """
        pkg = ResultsManager._MODEL2PKG.get(model_name)
        if not pkg:
            return "unknown"

        try:
            ver = metadata.version(pkg)
            return f"{pkg} v{ver}"
        except PackageNotFoundError:
            return f"{pkg} (not installed)"

    @staticmethod
    def _get_system_info() -> Dict[str, str]:
        # Gathers key information about the execution environment.
        info = {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor() or "unknown",
        }
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = str(torch.version.cuda)
        return info

    def set_data_info(self, x_train: pd.DataFrame, y_train: np.ndarray,
                      x_hold: pd.DataFrame, y_hold: np.ndarray,
                      groups_train: Optional[np.ndarray] = None,
                      split_strategy: str = "GroupShuffleSplit 80/20",
                      random_seed: int = 42):
        """Logs detailed information about the dataset and splitting strategy.

        Args:
            x_train (pd.DataFrame): Training feature matrix.
            y_train (np.ndarray): Training labels.
            x_hold (pd.DataFrame): Holdout feature matrix.
            y_hold (np.ndarray): Holdout labels.
            groups_train (Optional[np.ndarray]): Group identifiers for CV.
            split_strategy (str): Description of how the train/holdout split was made.
            random_seed (int): The random seed used for reproducibility.
        """
        train_dist = pd.Series(y_train).value_counts().sort_index().to_dict()
        hold_dist = pd.Series(y_hold).value_counts().sort_index().to_dict()

        self.results["data_info"] = {
            "n_features": x_train.shape[1],
            "feature_names": list(x_train.columns) if hasattr(x_train, 'columns') else None,
            "split_info": {
                "strategy": split_strategy,
                "random_seed": random_seed,
                "train_size": len(x_train),
                "holdout_size": len(x_hold),
                "n_groups": len(np.unique(groups_train)) if groups_train is not None else None,
            },
            "class_distribution": {
                "train": {str(k): int(v) for k, v in train_dist.items()},
                "holdout": {str(k): int(v) for k, v in hold_dist.items()},
            },
        }

    def set_best_params(self, params: Dict[str, Any]):
        self.results["hyperparameters"]["best_params"] = params

    def set_cv_metrics(self, cv_scores: Union[List[float], np.ndarray, float],
                       n_folds: int = 5, scorer: str = "macro_f1"):
        """Logs the results from cross-validation.

        Args:
            cv_scores: A list of scores from each fold, or a single mean score.
            n_folds (int): The number of CV folds used.
            scorer (str): The name of the performance metric used.
        """
        if isinstance(cv_scores, (list, np.ndarray)):
            scores = np.array(cv_scores)
            mean_score, std_score = float(np.mean(scores)), float(np.std(scores))
            fold_scores: Optional[List[float]] = [float(s) for s in scores]
        else:
            mean_score, std_score, fold_scores = float(cv_scores), None, None

        self.results["metrics"]["cross_validation"] = {
            "n_folds": n_folds,
            "scorer": scorer,
            "mean_score": round(mean_score, 4),
            "std_dev": round(std_score, 4) if std_score is not None else None,
            "fold_scores": fold_scores,
        }

    def set_holdout_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                            class_names: Optional[List[str]] = None):
        """Logs performance metrics on the holdout set.

        Args:
            y_true (np.ndarray): The true labels of the holdout set.
            y_pred (np.ndarray): The model's predictions on the holdout set.
            class_names (Optional[List[str]]): A list of names corresponding to the class labels.
        """
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        per_class = {}
        for label_str, metrics in report.items():
            if label_str.isdigit():
                class_idx = int(label_str)
                class_name = class_names[class_idx] if class_names and class_idx < len(class_names) else f"Class {label_str}"
                per_class[class_name] = {
                    "precision": round(metrics["precision"], 4),
                    "recall": round(metrics["recall"], 4),
                    "f1_score": round(metrics["f1-score"], 4),
                    "support": int(metrics["support"]),
                }

        self.results["metrics"]["holdout"] = {
            "overall": {
                "accuracy": round(float(report["accuracy"]), 4),
                "macro_avg_f1": round(float(report["macro avg"]["f1-score"]), 4),
                "weighted_avg_f1": round(float(report["weighted avg"]["f1-score"]), 4),
            },
            "per_class": per_class,
        }

    def set_hpo_stats(self, study: optuna.Study, optimization_time: float,
                      sampler: str = "TPESampler", pruner: Optional[str] = "MedianPruner"):
        """Logs statistics from the hyperparameter optimization process.

        Args:
            study (optuna.Study): The completed Optuna study object.
            optimization_time (float): Total HPO time in seconds.
            sampler (str): The name of the Optuna sampler used.
            pruner (Optional[str]): The name of the Optuna pruner used.
        """
        n_trials = len(study.trials)
        n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])

        self.results["performance"]["hpo_stats"] = {
            "sampler": sampler,
            "pruner": pruner,
            "n_trials": n_trials,
            "n_pruned_trials": n_pruned,
            "optimization_time_seconds": round(optimization_time, 1),
        }

    def set_training_stats(self, fit_time: float, device: str = "CPU",
                           best_epoch: Optional[int] = None, total_params: Optional[int] = None):
        """Logs statistics from the final model training phase.

        Args:
            fit_time (float): Time in seconds to fit the final model.
            device (str): The compute device used ("CPU" or "GPU").
            best_epoch (Optional[int]): The best epoch if using early stopping.
            total_params (Optional[int]): The total number of model parameters.
        """
        self.results["performance"]["training_stats"] = {
            "fit_time_seconds": round(fit_time, 2),
            "inference_device": device.upper(),
            "best_epoch": best_epoch,
            "total_parameters": total_params,
            "total_run_time_seconds": round(time.time() - self.start_time, 1),
        }

    def set_artifacts(self, **paths: Union[str, Path]):
        """Records the relative paths to all saved artifacts.

        Args:
            **paths: Keyword arguments where the key is the artifact type
                     (e.g., 'model_file') and the value is its path.
        """
        artifacts = {key: str(Path(path).relative_to(self.output_dir.parent))
                     for key, path in paths.items() if path}
        self.results["artifacts"] = artifacts

    def add_custom_metrics(self, **kwargs: Any):
        """Adds a custom section or key-value pairs to the results dictionary."""
        self.results.update(kwargs)

    def save(self, filename: str = "training_summary.json") -> Path:
        """Saves the complete results dictionary to a JSON file.

        Args:
            filename (str): The name for the output JSON file.

        Returns:
            Path: The full path to the saved JSON file.
        """
        filepath = self.output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"✓ Results successfully saved to: {filepath}")
        return filepath

    def get_summary(self) -> str:
        """Generates a concise, formatted string summary of the key results."""
        summary = [
            f"\n{'='*80}",
            f"{self.model_name.upper()} TRAINING SUMMARY",
            f"{'='*80}",
            f"Model Library: {self.results.get('model_info', {}).get('library', 'N/A')}",
            f"Data: {self.results.get('data_info', {}).get('split_info', {}).get('train_size', 'N/A')} train / "
            f"{self.results.get('data_info', {}).get('split_info', {}).get('holdout_size', 'N/A')} holdout",
            f"CV Score (Macro F1): {self.results.get('metrics', {}).get('cross_validation', {}).get('mean_score', 'N/A'):.4f}",
            f"Holdout Accuracy: {self.results.get('metrics', {}).get('holdout', {}).get('overall', {}).get('accuracy', 'N/A'):.4f}",
            f"Holdout Macro F1: {self.results.get('metrics', {}).get('holdout', {}).get('overall', {}).get('macro_avg_f1', 'N/A'):.4f}",
            f"Total Run Time: {self.results.get('performance', {}).get('training_stats', {}).get('total_run_time_seconds', 0):.1f}s",
            f"{'='*80}",
        ]
        return '\n'.join(s for s in summary if 'N/A' not in str(s))
