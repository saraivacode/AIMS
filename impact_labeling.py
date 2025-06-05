"""
impact_labeling.py
=====================
Vehicular Network Impact Labeler (Documented Version)
-----------------------------------------------------
Assigns a discrete **impact level** (0 = Adequate · 1 = Minor · 2 = Major · 3 = Critical) to each
row of a pandas DataFrame containing quality-of-service (QoS) metrics from vehicular network
experiments or simulations.

Implements the weighted-average thresholding approach from Saraiva et al. (2025):
    1. Applies standard or custom threshold tables for latency, loss, and throughput for each app class.
    2. Maps each metric to a quality score 0–3 via these thresholds.
    3. Combines scores using an application-specific weight vector.

Designed for modular use in ML pipelines and reproducible ITS experiments.

--------
Quick Start
-----------
>>> import pandas as pd
>>> from impact_labeling import label_weighted_average, WEIGHTS
>>> df = pd.read_csv("experiment.csv")
>>> result = label_weighted_average(df, weights=WEIGHTS)
>>> result["impact_label"].value_counts().sort_index()

--------
Expected DataFrame Schema
------------------------
    app            : str   (s, e, e2, g)
    lat_ms         : float (latency, ms)
    pdr            : float (packet delivery ratio, 0–1)
    throughput_kbps: float (throughput, kbps)

--------
Author: AIMS Framework Team
MIT License. See LICENSE.txt for details.
"""
from __future__ import annotations

from typing import Dict, List, Any
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

__all__ = [
    "HARD_THRESH",
    "WEIGHTS",
    "score_metric",
    "label_weighted_average",
]

# ---------------------------------------------------------------------------
# HARD_THRESH – Per-application QoS thresholds for each metric
# ---------------------------------------------------------------------------
# List format: [adequate, warning, severe]. Lower is better for latency/loss; higher is better for throughput.
HARD_THRESH = {
    "s": {  # Safety – 3GPP TS 22.185 Rel‑17
        "lat_ms": [100, 200, 500],
        "loss": [0.01, 0.05, 0.10],
        "thru_kbps": [450, 400, 300],
    },
    "e": {  # Efficiency – ETSI TR 102 962
        "lat_ms": [300, 700, 1500],
        "loss": [0.05, 0.10, 0.20],
        "thru_kbps": [800, 600, 400],
    },
    "e2": {  # Entertainment – 5GAA, ISO/IEC 23009-5
        "lat_ms": [100, 250, 1000],
        "loss": [0.02, 0.05, 0.10],
        "thru_kbps": [8000, 3000, 1000],
    },
    "g": {  # Generic – DASH-IF / Netflix QoE
        "lat_ms": [1000, 3000, 5000],
        "loss": [0.10, 0.30, 0.50],
        "thru_kbps": [5000, 2000, 500],
    },
}

# ---------------------------------------------------------------------------
# WEIGHTS – Default per-application weights (must sum to 1.0 each)
# ---------------------------------------------------------------------------
WEIGHTS = {
    "s": dict(lat=0.5, loss=0.3, thr=0.2),   # Safety
    "e": dict(lat=0.4, loss=0.3, thr=0.3),   # Efficiency
    "e2": dict(lat=0.4, loss=0.2, thr=0.4),  # Entertainment
    "g": dict(lat=0.3, loss=0.3, thr=0.4),   # Generic
}

# ---------------------------------------------------------------------------
# Helper function: metric value to quality score (0 best → 3 worst)
# ---------------------------------------------------------------------------
def score_metric(value: float, limits: List[float], *, lower_is_better: bool = True) -> int:
    """Map a numerical metric value to a discrete quality score (0–3).

    Parameters
    ----------
    value : float
        Observed value for the metric.
    limits : list of float
        Thresholds: [adequate, warning, severe].
    lower_is_better : bool, default True
        If False, higher values are better (e.g., throughput).

    Returns
    -------
    int
        0 (best) … 3 (worst). NaN is treated as 3 (worst).
    """
    if np.isnan(value):
        return 3
    if lower_is_better:
        if value <= limits[0]:
            return 0
        if value <= limits[1]:
            return 1
        if value <= limits[2]:
            return 2
    else:
        if value >= limits[0]:
            return 0
        if value >= limits[1]:
            return 1
        if value >= limits[2]:
            return 2
    return 3  # Out of spec/extreme case

# ---------------------------------------------------------------------------
# Core function: Compute and append impact label
# ---------------------------------------------------------------------------
def label_weighted_average(
    df: pd.DataFrame,
    *,
    hard: Dict[str, Dict[str, List[float]]] = HARD_THRESH,
    weights: Dict[str, Dict[str, float]] = WEIGHTS,
    out_col: str = "impact_label",
) -> tuple[pd.DataFrame, Any, Any, Any, Any, dict[int, float]]:
    """Assigns an impact label to each row using per-application weights.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data containing required metrics.
    hard : dict, optional
        Threshold mapping (see HARD_THRESH).
    weights : dict, optional
        Weighting scheme per application (see WEIGHTS).
    out_col : str, default 'impact_label'
        Name of the output column.

    Returns
    -------
    tuple
        (labeled DataFrame, X, y, groups, class_weights_array, class_weight_dict)
    """
    labels: List[int] = []
    for _, row in df.iterrows():
        cls = str(row.get("app", "g")).lower()
        ref = hard.get(cls, hard["g"])
        w = weights.get(cls, weights["g"])
        if not np.isclose(sum(w.values()), 1.0):
            raise ValueError(f"Weights for app '{cls}' must sum to 1.0 – got {w}")
        # Score each metric
        lat_s = score_metric(row["lat_ms"], ref["lat_ms"], lower_is_better=True)
        loss_s = score_metric(1.0 - row["pdr"], ref["loss"], lower_is_better=True)
        thr_s = score_metric(row["throughput_kbps"], ref["thru_kbps"], lower_is_better=False)
        # Weighted sum and rounding
        impact = w["lat"] * lat_s + w["loss"] * loss_s + w["thr"] * thr_s
        label = int(round(impact))
        labels.append(label)
    out_df = df.copy()
    out_df[out_col] = labels
    # Prepare for ML (X, y, groups, class weights)
    X = out_df.drop(columns=[out_col])
    y = out_df[out_col].values
    groups = out_df.get("group_id", pd.Series([0]*len(out_df))).values
    class_weights_array = compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)
    class_weight_dict = {int(k): float(v) for k, v in zip(np.unique(y), class_weights_array)}
    return out_df, X, y, groups, class_weights_array, class_weight_dict

# ---------------------------------------------------------------------------
# Self-test (if run as script)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import pandas as pd
    # Minimal example
    df = pd.DataFrame({
        "app": ["s", "e", "e2", "g"],
        "lat_ms": [80, 600, 200, 2000],
        "pdr": [0.99, 0.98, 0.94, 0.93],
        "throughput_kbps": [600, 850, 9000, 5100],
    })
    out, *_ = label_weighted_average(df)
    print(out)
