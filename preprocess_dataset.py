"""
preprocess_dataset.py
======================
Vehicular-QoS Dataset Pre-Processor (Documented Version)
--------------------------------------------------------
Compact helper functions to clean, rename, and engineer features for raw vehicular
communication datasets before feeding them into the labeling or ML pipeline.

Implements the transformation logic from Saraiva et al. (2025),
decoupling data wrangling from impact labeling and enabling reproducible experiments.

--------
Quick Start
-----------
>>> import pandas as pd
>>> import preprocess_dataset as pp
>>> df_raw = pd.read_csv("raw_metrics.csv")
>>> df_pre = pp.prepare_dataset(df_raw)

--------
Key Steps
---------
1. Column renaming (harmonizing metric names)
2. Unit conversion (throughput: bps → kbps)
3. Timestamp parsing & alias assignment (e.g. 'carros_ativos' → 'n_carros')
4. Rolling-window features (mean, std, delta) on 3-sample windows
5. Derived ratios (loss_ratio, throughput utilization)
6. NaN handling in rolling features
7. Outlier clipping (p1–p99 for latency/throughput; upper cap for ratios)
8. 60-second block assignment for grouped analysis

All functions use pandas only—no external dependencies beyond core Python data stack.

--------
Author: AIMS Framework Team
MIT License. See LICENSE.txt for details.
"""
from __future__ import annotations

from typing import List
import pandas as pd
import numpy as np

__all__ = [
    "COLUMN_MAP",
    "prepare_dataset",
]

# ---------------------------------------------------------------------------
# Canonical column mapping (raw → standard names)
# ---------------------------------------------------------------------------
COLUMN_MAP = {
    "rtt_medio_ms_interpolated": "lat_ms",
    "pdr_inst_pacotes": "pdr",
    "vazao_rec_servidor_total_bps": "throughput_bps",
    # Extend with additional aliases if needed
}

# ---------------------------------------------------------------------------
# Main preprocessing function
# ---------------------------------------------------------------------------
def prepare_dataset(
    df: pd.DataFrame,
    *,
    window: int = 3,
    throughput_ref_kbps: float = 10_000.0,
    drop_duplicates: bool = True,
    outlier_clip: bool = True,
) -> pd.DataFrame:
    """Return a cleaned & feature-rich DataFrame for labeling or modeling.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset with required columns (see COLUMN_MAP + timestamp_sec_str, carros_ativos).
    window : int, default 3
        Rolling window size for mean/std computation.
    throughput_ref_kbps : float, default 10,000
        Reference value for normalizing throughput utilization.
    drop_duplicates : bool, default True
        Remove redundant columns if present.
    outlier_clip : bool, default True
        Apply p1-p99 clipping (latency & throughput), cap loss_ratio.

    Returns
    -------
    pd.DataFrame
        DataFrame with cleaned, engineered features ready for labeling/ML.
    """
    df = df.copy()

    # Column harmonization & unit conversion
    if "rtt_medio_ms" in df.columns:
        df = df.drop(columns=["rtt_medio_ms"])
    df = df.rename(columns=COLUMN_MAP)
    df["throughput_kbps"] = df["throughput_bps"] / 1_000.0

    # Timestamp parsing and aliases
    df["timestamp"] = pd.to_datetime(df["timestamp_sec_str"], utc=True)
    df = df.rename(columns={"carros_ativos": "n_carros"})

    # Rolling-window stats & deltas
    for col in ["lat_ms", "pdr", "throughput_kbps"]:
        df[f"{col}_mean{window}"] = df[col].rolling(window, min_periods=1).mean()
        df[f"{col}_std{window}"] = df[col].rolling(window, min_periods=1).std().fillna(0.0)
        df[f"{col}_delta"] = df[col].diff().fillna(0)

    # Derived ratios
    eps = 1e-9
    df["loss_ratio"] = (1.0 - df["pdr"] + eps) / (df["pdr"] + eps)
    df["thr_util"] = df["throughput_kbps"] / throughput_ref_kbps

    # Optional duplicate column pruning
    if drop_duplicates:
        dup_cols = [
            "conf_inst_bits",
            "pdr_cum_pacotes",
            "conf_cum_bits",
            "bc_rtt",
        ]
        df = df.drop(columns=[c for c in dup_cols if c in df.columns])

    # Outlier handling
    if outlier_clip:
        clip_cols = [
            "lat_ms",
            f"lat_ms_mean{window}",
            f"lat_ms_std{window}",
            f"lat_ms_delta",
            "throughput_kbps",
            f"throughput_kbps_mean{window}",
            f"throughput_kbps_std{window}",
            f"throughput_kbps_delta",
        ]
        for c in clip_cols:
            if c in df.columns:
                low, high = np.percentile(df[c], 1), np.percentile(df[c], 99)
                df[c] = df[c].clip(lower=low, upper=high)
        # Cap loss_ratio at 2.0
        if "loss_ratio" in df.columns:
            df["loss_ratio"] = df["loss_ratio"].clip(upper=2.0)

    # Block/group assignment (e.g., 60s time blocks)
    if "timestamp" in df.columns:
        df["time_block"] = ((df["timestamp"] - df["timestamp"].min()) / pd.Timedelta("60s")).astype(int)
        df["group_id"] = df["app"].astype(str) + "_" + df["time_block"].astype(str)

    return df

# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Minimal test case
    df = pd.DataFrame({
        "rtt_medio_ms_interpolated": [50, 100, 120],
        "pdr_inst_pacotes": [0.99, 0.95, 0.97],
        "vazao_rec_servidor_total_bps": [400000, 350000, 500000],
        "timestamp_sec_str": ["2023-01-01 00:00:00", "2023-01-01 00:00:01", "2023-01-01 00:00:02"],
        "carros_ativos": [5, 6, 7],
        "app": ["s", "e", "g"],
    })
    out = prepare_dataset(df)
    print(out.head())
