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

# Consistent visual style for security strategy+defense combinations
_SECURITY_STYLES = {
    "FedAvg":       {"color": "#1f77b4", "marker": "o",  "linestyle": "-"},
    "FedProx":      {"color": "#ff7f0e", "marker": "s",  "linestyle": "-"},
    "Krum":         {"color": "#2ca02c", "marker": "^",  "linestyle": "--"},
    "TrimmedMean":  {"color": "#d62728", "marker": "D",  "linestyle": "--"},
}

def _security_label(result: Dict) -> str:
    """Build a label that distinguishes strategy + defense."""
    strategy = result.get("strategy", "FedAvg")
    defense = result.get("defense", strategy)
    scale = result.get("gradient_scale", 1.0)
    scale_str = f" x{scale:.0f}" if scale > 1 else ""
    if defense in ("FedAvg", "FedProx"):
        return f"{strategy}{scale_str}"
    return f"{strategy}+{defense}{scale_str}"

def _security_style(result: Dict) -> Dict:
    """Return color/marker/linestyle for a strategy+defense combination."""
    strategy = result.get("strategy", "FedAvg")
    defense = result.get("defense", strategy)
    base = _SECURITY_STYLES.get(defense, _SECURITY_STYLES["FedAvg"])
    style = dict(base)
    # Differentiate FedProx variants with dashed lines when no byzantine defense
    if strategy == "FedProx" and defense in ("FedAvg", "FedProx"):
        style["linestyle"] = "-"
    # FedProx + byzantine defense: use defense color but dashed-dot
    if strategy == "FedProx" and defense not in ("FedAvg", "FedProx"):
        style["linestyle"] = "-."
    return style


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


def plot_security_c2a_comparison(
    results: List[Dict],
    output_dir: Path,
    filename: str = "security_c2a_comparison.png",
) -> Path:
    """
    Bar chart comparing C2A rates across defenses for each model.

    Groups by model type, bars for each defense strategy.
    """
    if not results:
        print("    No security results to plot C2A comparison.")
        return Path(output_dir)

    models = sorted(set(r["model_type"] for r in results))
    # Build unique (strategy, defense) combos preserving order
    seen = {}
    for r in results:
        key = _security_label(r)
        if key not in seen:
            seen[key] = r
    combo_labels = list(seen.keys())
    combo_results = list(seen.values())

    fig, ax = plt.subplots(figsize=(max(8, 2 * len(models) * len(combo_labels)), 5))
    x = np.arange(len(models))
    width = 0.8 / max(len(combo_labels), 1)

    for c_idx, (label, ref_r) in enumerate(zip(combo_labels, combo_results)):
        style = _security_style(ref_r)
        c2a_vals = []
        for model in models:
            matching = [r for r in results
                        if r["model_type"] == model
                        and _security_label(r) == label]
            if matching:
                c2a_vals.append(matching[0].get("final_metrics", {}).get("c2a_rate", 0))
            else:
                c2a_vals.append(0)
        offset = (c_idx - len(combo_labels) / 2 + 0.5) * width
        bars = ax.bar(x + offset, c2a_vals, width, label=label,
                      color=style["color"], alpha=0.85)
        for bar in bars:
            if bar.get_height() > 0.001:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_ylabel("C2A Rate (Critical → Adequate)")
    ax.set_title("Security: C2A Rate by Model and Strategy")
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in models])
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    filepath = Path(output_dir) / filename
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {filepath}")
    return filepath


def plot_security_accuracy_vs_c2a(
    results: List[Dict],
    output_dir: Path,
    filename: str = "security_accuracy_vs_c2a.png",
) -> Path:
    """
    Scatter plot: accuracy vs C2A rate for each security experiment.

    Each point represents one experiment, colored by defense strategy.
    """
    if not results:
        print("    No security results for accuracy vs C2A plot.")
        return Path(output_dir)

    fig, ax = plt.subplots(figsize=(9, 5))
    # Build unique combo labels
    seen = {}
    for r in results:
        key = _security_label(r)
        if key not in seen:
            seen[key] = r
    combo_labels = list(seen.keys())
    combo_results = list(seen.values())

    for label, ref_r in zip(combo_labels, combo_results):
        style = _security_style(ref_r)
        subset = [r for r in results if _security_label(r) == label]
        accs = [r.get("final_metrics", {}).get("accuracy", 0) for r in subset]
        c2as = [r.get("final_metrics", {}).get("c2a_rate", 0) for r in subset]
        model_labels = [r.get("model_type", "?") for r in subset]
        ax.scatter(accs, c2as, label=label, marker=style["marker"],
                   color=style["color"], s=80, alpha=0.85)
        for i, lbl in enumerate(model_labels):
            ax.annotate(lbl, (accs[i], c2as[i]), fontsize=7,
                        textcoords="offset points", xytext=(5, 5))

    ax.set_xlabel("Accuracy")
    ax.set_ylabel("C2A Rate")
    ax.set_title("Security Trade-off: Accuracy vs C2A Rate")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = Path(output_dir) / filename
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {filepath}")
    return filepath


def plot_security_convergence(
    results: List[Dict],
    output_dir: Path,
    filename: str = "security_convergence.png",
) -> Path:
    """
    Accuracy convergence curves for security experiments.

    One subplot per model, lines for each defense.
    """
    valid = [r for r in results if r.get("per_round", {}).get("accuracy")]
    if not valid:
        print("    No valid security results for convergence plot.")
        return Path(output_dir)

    models = sorted(set(r["model_type"] for r in valid))
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5), squeeze=False)

    for idx, model in enumerate(models):
        ax = axes[0, idx]
        model_results = [r for r in valid if r["model_type"] == model]

        for r in model_results:
            acc_data = r["per_round"]["accuracy"]
            rounds = [d["round"] for d in acc_data]
            accs = [d["accuracy"] for d in acc_data]
            label = _security_label(r)
            style = _security_style(r)
            ax.plot(rounds, accs, marker=style["marker"], color=style["color"],
                    linestyle=style["linestyle"], label=label, markersize=3)

        ax.set_title(f"{model.upper()} (Under Attack)")
        ax.set_xlabel("Round")
        ax.set_ylabel("Accuracy")
        ax.axhline(y=0.90, color="r", linestyle="--", alpha=0.5, label="90%")
        ax.axhline(y=0.85, color="orange", linestyle="--", alpha=0.5, label="85%")
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.0, 1.0])

    plt.tight_layout()
    filepath = Path(output_dir) / filename
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {filepath}")
    return filepath


def plot_security_c2a_convergence(
    results: List[Dict],
    output_dir: Path,
    filename: str = "security_c2a_convergence.png",
) -> Path:
    """
    C2A rate convergence over rounds for security experiments.

    One subplot per model, lines for each defense.
    """
    valid = [r for r in results if r.get("per_round", {}).get("c2a_rate")]
    if not valid:
        print("    No valid security results for C2A convergence plot.")
        return Path(output_dir)

    models = sorted(set(r["model_type"] for r in valid))
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5), squeeze=False)

    for idx, model in enumerate(models):
        ax = axes[0, idx]
        model_results = [r for r in valid if r["model_type"] == model]

        for r in model_results:
            c2a_data = r["per_round"]["c2a_rate"]
            rounds = [d["round"] for d in c2a_data]
            c2as = [d["c2a_rate"] for d in c2a_data]
            label = _security_label(r)
            style = _security_style(r)
            ax.plot(rounds, c2as, marker=style["marker"], color=style["color"],
                    linestyle=style["linestyle"], label=label, markersize=3)

        ax.set_title(f"{model.upper()} - C2A Rate")
        ax.set_xlabel("Round")
        ax.set_ylabel("C2A Rate")
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.0, 1.0])

    plt.tight_layout()
    filepath = Path(output_dir) / filename
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {filepath}")
    return filepath


def plot_sensitivity_epochs(
    results: List[Dict],
    output_dir: Path,
    filename: str = "security_sensitivity_epochs.png",
) -> Path:
    """
    Sensitivity analysis: accuracy and C2A vs local epochs under attack.

    Two subplots per model: accuracy and C2A rate as a function of
    local_epochs, with one line per defense.
    """
    if not results:
        print("    No results for epoch sensitivity plot.")
        return Path(output_dir)

    models = sorted(set(r["model_type"] for r in results))
    # Build unique combo labels preserving order
    seen = {}
    for r in results:
        key = _security_label(r)
        if key not in seen:
            seen[key] = r
    combo_labels = list(seen.keys())
    combo_refs = list(seen.values())

    fig, axes = plt.subplots(2, len(models),
                             figsize=(5 * len(models), 8), squeeze=False)

    for m_idx, model in enumerate(models):
        ax_acc = axes[0, m_idx]
        ax_c2a = axes[1, m_idx]
        model_results = [r for r in results if r["model_type"] == model]

        for label, ref_r in zip(combo_labels, combo_refs):
            style = _security_style(ref_r)
            subset = sorted(
                [r for r in model_results if _security_label(r) == label],
                key=lambda r: r.get("local_epochs", 0),
            )
            if not subset:
                continue
            epochs = [r.get("local_epochs", 0) for r in subset]
            accs = [r.get("final_metrics", {}).get("accuracy", 0) for r in subset]
            c2as = [r.get("final_metrics", {}).get("c2a_rate", 0) for r in subset]

            ax_acc.plot(epochs, accs, marker=style["marker"], color=style["color"],
                        linestyle=style["linestyle"], label=label, markersize=5)
            ax_c2a.plot(epochs, c2as, marker=style["marker"], color=style["color"],
                        linestyle=style["linestyle"], label=label, markersize=5)

        ax_acc.set_title(f"{model.upper()} - Accuracy")
        ax_acc.set_xlabel("Local Epochs")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.legend(fontsize=7)
        ax_acc.grid(True, alpha=0.3)

        ax_c2a.set_title(f"{model.upper()} - C2A Rate")
        ax_c2a.set_xlabel("Local Epochs")
        ax_c2a.set_ylabel("C2A Rate")
        ax_c2a.legend(fontsize=7)
        ax_c2a.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = Path(output_dir) / filename
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
