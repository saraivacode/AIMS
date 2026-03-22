#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIMS Framework - Vehicular QoS Dataset Preprocessing Module
===========================================================

This module provides the `prepare_dataset` function for standardized cleaning, transformation, and feature
engineering of raw vehicular network QoS datasets. It serves as the initial step in the AIMS (Adaptive and
Intelligent Management of Slicing) machine learning pipeline.

Key Functionalities:
    - Consistent column renaming and unit normalization
    - Rolling-window statistics and delta feature calculation
    - Derived ratio features (e.g., loss ratio, throughput utilization)
    - Outlier handling and percentile-based clipping
    - Group ID generation for time-aware cross-validation

Author: Tiago do Vale Saraiva
License: MIT
"""

from __future__ import annotations
import pandas as pd

__all__ = ["INTERNAL_COLUMN_MAP", "prepare_dataset"]

# Mapping from raw column names to internal feature names
INTERNAL_COLUMN_MAP: dict[str, str] = {
    "rtt_avg_ms_interp": "lat_ms",
    "pdr_inst": "pdr",
    "srv_rx_thr_total_bps": "throughput_bps",
}

def prepare_dataset(
    df: pd.DataFrame,
    *,
    window: int = 3,
    throughput_ref_kbps: float = 10_000.0,
    drop_duplicates: bool = True,
    outlier_clip: bool = True,
) -> pd.DataFrame:
    """
    Standardize and engineer features from raw vehicular QoS datasets.

    This function applies unit conversions, rolling statistics, ratio derivations,
    outlier clipping, and group ID creation to prepare data for ML tasks.

    Parameters:
        df (pd.DataFrame): Input dataset with raw vehicular QoS metrics.
        window (int): Rolling window size for statistical feature calculation.
        throughput_ref_kbps (float): Reference throughput for utilization calculation.
        drop_duplicates (bool): Whether to remove intermediate/non-essential columns.
        outlier_clip (bool): Whether to clip outliers based on percentiles.

    Returns:
        pd.DataFrame: Cleaned and feature-rich dataset ready for modeling.
    """
    df = df.copy()

    # Step 1: Standardize column names and units
    # Drop the raw (non-interpolated) RTT column if present, as we use the interpolated one.
    if "rtt_avg_ms" in df.columns:
        df = df.drop(columns=["rtt_avg_ms"])
    # Rename standard input columns to the short internal names for easier processing.
    df = df.rename(columns=INTERNAL_COLUMN_MAP)
    # Change throughput unit to kbps
    if "throughput_bps" in df.columns:
        df["throughput_kbps"] = df["throughput_bps"] / 1_000.0

    # Step 2: Timestamp parsing and column aliasing
    if "timestamp_str" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp_str"], utc=True)
    if "active_vehicles" in df.columns:
        df = df.rename(columns={"active_vehicles": "n_vehicles"})

    # Step 3: Rolling statistics and delta features
    rolling_cols = ["lat_ms", "pdr", "throughput_kbps"]
    for col in rolling_cols:
        if col in df.columns:
            df[f"{col}_mean{window}"] = df[col].rolling(window, min_periods=1).mean()
            df[f"{col}_std{window}"] = df[col].rolling(window, min_periods=1).std().fillna(0.0)
            df[f"{col}_delta"] = df[col].diff().fillna(0.0)

    # Step 4: Derived ratio features
    # A small epsilon to avoid division by zero.
    eps = 1e-9
    if "pdr" in df.columns:
        df["loss_ratio"] = (1.0 - df["pdr"] + eps) / (df["pdr"] + eps)
    if "throughput_kbps" in df.columns:
        df["thr_util"] = df["throughput_kbps"] / throughput_ref_kbps

    # Step 5: Drop non-essential intermediate columns
    if drop_duplicates:
        drop_cols = ["rel_inst", "pdr_cum", "rel_cum", "rtt_bc", "rtt_bc_lambda"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Step 6: Outlier clipping on selected features
    if outlier_clip:
        clip_cols = [
            "lat_ms", f"lat_ms_mean{window}", f"lat_ms_std{window}", f"lat_ms_delta",
            "throughput_kbps", f"throughput_kbps_mean{window}",
            f"throughput_kbps_std{window}", f"throughput_kbps_delta",
        ]
        for col in clip_cols:
            if col in df.columns:
                p01, p99 = df[col].quantile([0.01, 0.99])
                if pd.notna(p01) and pd.notna(p99):
                    df[col] = df[col].clip(p01, p99)
        if "loss_ratio" in df.columns:
            df["loss_ratio"] = df["loss_ratio"].clip(upper=10)

    # Step 7: Time-based group ID generation
    # Create group_id BEFORE dropping timestamp columns
    if "timestamp" in df.columns:
        df["time_block"] = df["timestamp"].astype("datetime64[ns, UTC]").astype("int64") // int(6e10)
        if "approach" in df.columns:
            df["group_id"] = df["approach"] + "_" + df["time_block"].astype(str)
        # Drop non-predictive timestamp columns AFTER creating group_id
        if drop_duplicates:
            for col in ['timestamp_sec_str', 'timestamp_str', 'timestamp']:
                if col in df.columns:
                    df = df.drop(columns=[col])
    return df

if __name__ == "__main__":
    """
    Simple self-test: Apply preprocessing on a small synthetic DataFrame
    and display the resulting columns.
    """
    print("--- Running preprocess_dataset.py self-test ---")

    test_df = pd.DataFrame({
        "timestamp_str": ["2023-01-01 00:00:00", "2023-01-01 00:00:01", "2023-01-01 00:00:02"],
        "app_id": ["s", "e", "g"],
        "approach": ["fq", "fq", "fq"],
        "active_vehicles": [5, 6, 7],
        "srv_rx_thr_total_bps": [400000, 350000, 500000],
        "rtt_avg_ms_interp": [50.0, 100.0, 120.0],
        "pdr_inst": [0.99, 0.95, 0.97],
    })

    processed_df = prepare_dataset(test_df)
    print("\nProcessed columns:")
    print(processed_df.columns.tolist())
    print("\nSelf-test completed successfully.")