#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIMS Framework - Federated Learning Visualizations
===================================================

Generates convergence plots, strategy comparisons, FL vs centralized
charts, and client data distribution visualizations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .fl_config import FLDefaults

_DEFAULTS = FLDefaults()


def plot_convergence(results: List[Dict], output_dir: Path) -> Path:
    """
    Plot accuracy convergence curves per model, grouped by strategy+distribution.

    One subplot per model type, lines for each strategy/distribution combo.
    Horizontal reference lines at 85% and 90% accuracy.
    """
    valid = [r for r in results if r.get("per_round", {}).get("accuracy")]
    if not valid:
        print("    No valid results to plot convergence.")
        return output_dir

    models = sorted(set(r["model_type"] for r in valid))
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5), squeeze=False)

    for idx, model in enumerate(models):
        ax = axes[0, idx]
        model_results = [r for r in valid if r["model_type"] == model]

        for r in model_results:
            acc_data = r["per_round"]["accuracy"]
            rounds = [d["round"] for d in acc_data]
            accs = [d["accuracy"] for d in acc_data]
            label = f"{r['distribution']}-{r['strategy']}"
            ax.plot(rounds, accs, marker="o", label=label, markersize=4)

        ax.set_title(f"{model.upper()}")
        ax.set_xlabel("Round")
        ax.set_ylabel("Accuracy")
        ax.axhline(y=0.90, color="r", linestyle="--", alpha=0.5, label="90%")
        ax.axhline(y=0.85, color="orange", linestyle="--", alpha=0.5, label="85%")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.3, 1.0])

    plt.tight_layout()
    filepath = Path(output_dir) / "convergence_comparison.png"
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {filepath}")
    return filepath


def plot_strategy_comparison(results: List[Dict], output_dir: Path) -> Path:
    """
    Compare FedAvg vs FedProx on Non-IID data for each model.

    Shows accuracy convergence curves for Non-IID experiments only.
    """
    noniid = [
        r for r in results
        if r.get("distribution", "").lower() == "noniid"
        and r.get("per_round", {}).get("accuracy")
    ]
    if len(noniid) < 2:
        print("    Not enough Non-IID results for strategy comparison.")
        return output_dir

    fig, ax = plt.subplots(figsize=(8, 5))

    for r in noniid:
        acc_data = r["per_round"]["accuracy"]
        rounds = [d["round"] for d in acc_data]
        accs = [d["accuracy"] for d in acc_data]
        label = f"{r['model_type'].upper()}-{r['strategy'].upper()}"
        ax.plot(rounds, accs, marker="o", label=label, markersize=4)

    ax.set_title("FedAvg vs FedProx on Non-IID Data")
    ax.set_xlabel("Round")
    ax.set_ylabel("Accuracy")
    ax.axhline(y=0.90, color="r", linestyle="--", alpha=0.5, label="90%")
    ax.axhline(y=0.85, color="orange", linestyle="--", alpha=0.5, label="85%")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.3, 1.0])

    plt.tight_layout()
    filepath = Path(output_dir) / "strategy_comparison.png"
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {filepath}")
    return filepath


def plot_fl_vs_centralized(
    fl_results: List[Dict],
    centralized_results: List[Dict],
    output_dir: Path,
) -> Path:
    """
    Bar chart comparing FL (best per model) vs centralized baseline.

    Groups by model type, showing F1-Macro for centralized and best FL config.
    """
    if not fl_results or not centralized_results:
        print("    Insufficient results for FL vs Centralized comparison.")
        return output_dir

    models = sorted(set(r["model_type"] for r in fl_results))
    cent_f1 = {}
    for r in centralized_results:
        mt = r["model_type"]
        cent_f1[mt] = r.get("final_metrics", {}).get("f1_macro", 0)

    # Best FL F1 per model
    best_fl_f1 = {}
    best_fl_label = {}
    for mt in models:
        model_fl = [r for r in fl_results if r["model_type"] == mt]
        if model_fl:
            best = max(model_fl, key=lambda r: r.get("final_metrics", {}).get("f1_macro", 0))
            best_fl_f1[mt] = best.get("final_metrics", {}).get("f1_macro", 0)
            best_fl_label[mt] = f"{best['strategy']}/{best['distribution']}"

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(models))
    width = 0.35

    cent_vals = [cent_f1.get(m, 0) for m in models]
    fl_vals = [best_fl_f1.get(m, 0) for m in models]

    bars1 = ax.bar(x - width / 2, cent_vals, width, label="Centralized", alpha=0.85)
    bars2 = ax.bar(x + width / 2, fl_vals, width, label="Best FL", alpha=0.85)

    ax.set_ylabel("F1-Macro")
    ax.set_title("Centralized vs Best Federated")
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in models])
    ax.legend()
    ax.set_ylim([0, 1.05])

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for i, bar in enumerate(bars2):
        model = models[i]
        label_text = f"{bar.get_height():.3f}"
        if model in best_fl_label:
            label_text += f"\n({best_fl_label[model]})"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                label_text, ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    filepath = Path(output_dir) / "fl_vs_centralized.png"
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {filepath}")
    return filepath


def plot_client_distribution(
    partitions: List[tuple],
    distribution: str,
    output_dir: Path,
    num_classes: int = 4,
) -> Path:
    """
    Stacked bar chart showing class distribution across FL clients.

    Useful for visualizing the difference between IID and Non-IID partitions.
    """
    class_names = _DEFAULTS.CLASS_NAMES[:num_classes]

    fig, ax = plt.subplots(figsize=(6, 4))
    num_clients = len(partitions)
    x = np.arange(num_clients)
    bottoms = np.zeros(num_clients)

    for cls_idx in range(num_classes):
        counts = []
        for _, y in partitions:
            counts.append(np.sum(y == cls_idx))
        bars = ax.bar(x, counts, bottom=bottoms, label=class_names[cls_idx], alpha=0.85)
        bottoms += np.array(counts)

    ax.set_xlabel("Client (RSU)")
    ax.set_ylabel("Number of Samples")
    ax.set_title(f"Client Data Distribution ({distribution})")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Client {i}" for i in range(num_clients)])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    filepath = Path(output_dir) / f"client_distribution_{distribution}.png"
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {filepath}")
    return filepath
