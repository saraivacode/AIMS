#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIMS Framework - Federated Learning Results Manager
====================================================

Manages saving, aggregating, and formatting FL experiment results,
following the AIMS results convention with JSON outputs, summary
tables (CSV + LaTeX), and metadata.
"""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class FLResultsManager:
    """
    Manages FL experiment results, compatible with AIMS's JSON format.

    Parameters
    ----------
    output_dir : Path
        Directory where results will be saved (e.g., results/federated/).
    """

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_experiment(self, result: Dict, filename: Optional[str] = None) -> Path:
        """
        Save a single experiment result as JSON.

        Parameters
        ----------
        result : dict
            Experiment result dict from run_fl_simulation() or train_centralized().
        filename : str, optional
            Custom filename. Defaults to {config_name}.json.

        Returns
        -------
        Path to saved file.
        """
        if filename is None:
            filename = f"{result['config_name']}.json"

        filepath = self.output_dir / filename

        # Add metadata
        output = {
            "model_info": {
                "model_name": result.get("config_name", "Unknown"),
                "model_type": result.get("model_type", "Unknown"),
                "library": _get_tf_version(),
            },
            "run_info": {
                "framework_version": "AIMS 1.0.0 + FL",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "system": _get_system_info(),
            },
            "fl_config": {
                "mode": result.get("mode", "federated"),
                "strategy": result.get("strategy", "-"),
                "distribution": result.get("distribution", "-"),
                "num_clients": result.get("num_clients", "-"),
                "num_rounds": result.get("num_rounds", "-"),
                "local_epochs": result.get("local_epochs", "-"),
                "batch_size": result.get("batch_size", "-"),
            },
            "metrics": {
                "final": result.get("final_metrics", {}),
                "per_round": result.get("per_round", {}),
                "convergence": result.get("convergence", {}),
                "history": result.get("history", {}),
            },
            "performance": {
                "total_time": result.get("total_time", 0),
                "epochs_trained": result.get("epochs_trained"),
            },
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, default=str)

        print(f"    Saved: {filepath}")
        return filepath

    def save_summary_table(
        self,
        fl_results: List[Dict],
        centralized_results: List[Dict],
    ) -> tuple[Path, Path]:
        """
        Create and save a summary comparison table as CSV and LaTeX.

        Returns (csv_path, tex_path).
        """
        rows = []

        for r in fl_results:
            fm = r.get("final_metrics", {})
            conv = r.get("convergence", {})
            rows.append({
                "Model": r.get("model_type", "").upper(),
                "Distribution": r.get("distribution", "").upper(),
                "Strategy": r.get("strategy", "").upper(),
                "Accuracy": f"{fm.get('accuracy', 0):.4f}",
                "Precision": f"{fm.get('precision', 0):.4f}",
                "Recall": f"{fm.get('recall', 0):.4f}",
                "F1-Macro": f"{fm.get('f1_macro', 0):.4f}",
                "@85%": conv.get("round_85", "-") or "-",
                "@90%": conv.get("round_90", "-") or "-",
                "Stability": f"{conv.get('stability_std', 0):.4f}",
                "Time (s)": f"{r.get('total_time', 0):.1f}",
            })

        for r in centralized_results:
            fm = r.get("final_metrics", {})
            rows.append({
                "Model": r.get("model_type", "").upper(),
                "Distribution": "CENTRAL",
                "Strategy": "-",
                "Accuracy": f"{fm.get('accuracy', 0):.4f}",
                "Precision": f"{fm.get('precision', 0):.4f}",
                "Recall": f"{fm.get('recall', 0):.4f}",
                "F1-Macro": f"{fm.get('f1_macro', 0):.4f}",
                "@85%": "-",
                "@90%": "-",
                "Stability": "-",
                "Time (s)": f"{r.get('total_time', 0):.1f}",
            })

        df = pd.DataFrame(rows)

        csv_path = self.output_dir / "summary_table.csv"
        df.to_csv(csv_path, index=False)
        print(f"    Saved summary CSV: {csv_path}")

        tex_path = self.output_dir / "summary_table.tex"
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(df.to_latex(index=False, escape=False))
        print(f"    Saved summary LaTeX: {tex_path}")

        return csv_path, tex_path

    def save_security_experiment(self, result: Dict, filename: Optional[str] = None) -> Path:
        """
        Save a security FL experiment result as JSON with attack/defense metadata.

        Parameters
        ----------
        result : dict
            Result dict from ``run_fl_security_simulation()``.
        filename : str, optional
            Custom filename. Defaults to ``{config_name}.json``.

        Returns
        -------
        Path to saved file.
        """
        if filename is None:
            filename = f"{result['config_name']}.json"

        filepath = self.output_dir / filename

        output = {
            "model_info": {
                "model_name": result.get("config_name", "Unknown"),
                "model_type": result.get("model_type", "Unknown"),
                "library": _get_tf_version(),
            },
            "run_info": {
                "framework_version": "AIMS 1.0.0 + FL Security",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "system": _get_system_info(),
            },
            "fl_config": {
                "mode": "federated_security",
                "strategy": result.get("strategy", "-"),
                "distribution": result.get("distribution", "-"),
                "num_clients": result.get("num_clients", "-"),
                "num_rounds": result.get("num_rounds", "-"),
                "local_epochs": result.get("local_epochs", "-"),
                "batch_size": result.get("batch_size", "-"),
            },
            "security_config": {
                "attack_type": result.get("attack_type", "-"),
                "defense": result.get("defense", "-"),
                "gradient_scale": result.get("gradient_scale", 1.0),
                "malicious_clients": result.get("malicious_clients", []),
            },
            "metrics": {
                "final": result.get("final_metrics", {}),
                "per_round": result.get("per_round", {}),
                "convergence": result.get("convergence", {}),
            },
            "performance": {
                "total_time": result.get("total_time", 0),
            },
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, default=str)

        print(f"    Saved: {filepath}")
        return filepath

    def save_security_summary_table(
        self,
        results: List[Dict],
        filename: str = "security_summary_table",
    ) -> Tuple[Path, Path]:
        """
        Create and save a security experiment summary table as CSV and LaTeX.

        Includes attack/defense columns and C2A rate metric.

        Parameters
        ----------
        results : list of dict
            Security experiment results.
        filename : str
            Base filename (without extension).

        Returns
        -------
        (csv_path, tex_path)
        """
        rows = []
        for r in results:
            fm = r.get("final_metrics", {})
            conv = r.get("convergence", {})
            rows.append({
                "Model": r.get("model_type", "").upper(),
                "Defense": r.get("defense", "-"),
                "Attack": r.get("attack_type", "-"),
                "Scale": r.get("gradient_scale", 1.0),
                "Accuracy": f"{fm.get('accuracy', 0):.4f}",
                "F1-Macro": f"{fm.get('f1_macro', 0):.4f}",
                "C2A Rate": f"{fm.get('c2a_rate', 0):.4f}",
                "Precision": f"{fm.get('precision', 0):.4f}",
                "Recall": f"{fm.get('recall', 0):.4f}",
                "@85%": conv.get("round_85", "-") or "-",
                "@90%": conv.get("round_90", "-") or "-",
                "Time (s)": f"{r.get('total_time', 0):.1f}",
            })

        df = pd.DataFrame(rows)

        csv_path = self.output_dir / f"{filename}.csv"
        df.to_csv(csv_path, index=False)
        print(f"    Saved security CSV: {csv_path}")

        tex_path = self.output_dir / f"{filename}.tex"
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(df.to_latex(index=False, escape=False))
        print(f"    Saved security LaTeX: {tex_path}")

        return csv_path, tex_path

    def save_centralized_results(self, centralized_results: List[Dict]) -> Path:
        """Save all centralized baseline results in a single JSON."""
        filepath = self.output_dir / "centralized_results.json"
        data = {r["model_type"]: r for r in centralized_results}
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"    Saved centralized results: {filepath}")
        return filepath


def _get_tf_version() -> str:
    """Get TensorFlow version string."""
    try:
        import tensorflow as tf
        return f"tensorflow v{tf.__version__}"
    except ImportError:
        return "tensorflow (not installed)"


def _get_system_info() -> Dict[str, Any]:
    """Gather system information."""
    info = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
    }
    try:
        import tensorflow as tf
        if tf.config.list_physical_devices("GPU"):
            info["gpu"] = tf.config.list_physical_devices("GPU")[0].name
    except Exception:
        pass
    return info
