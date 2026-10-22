"""
Step 2: Temporal & Spatial Alignment Pipeline (Monthly Time Step & Zone Tagging)
================================================================================
This script completes Step 2 of the AI-Based Urban Climate Resilience workflow:
1. Loads the model-ready air quality observations for PCMC Nigdi.
2. Injects spatial zone metadata (zone_id, latitude, longitude) for multi-hazard integration.
3. Regularizes missing sensor values using forward-fill for short-duration gaps.
4. Resamples observations to a uniform Monthly Time Step ('ME').
5. Calculates multi-parameter statistics (mean, peak max, std, NAAQS exceedance days).
6. Exports 'Nigdi_OpenAQ_Monthly_Model_Ready.csv'.
"""

import os
import pandas as pd
import numpy as np

def run_step2_alignment():
    print("=" * 60)
    print("STEP 2: TEMPORAL & SPATIAL ALIGNMENT PIPELINE")
    print("=" * 60)

    # 1. Locate dataset (Daily or Hourly)
    daily_file = "Nigdi_OpenAQ_Daily_Model_Ready.csv"
    hourly_file = "Nigdi_OpenAQ_Hourly_Model_Ready.csv"

    if os.path.exists(daily_file):
        print(f"Loading daily data from: {daily_file}")
        df = pd.read_csv(daily_file)
        date_col = "date"
    elif os.path.exists(hourly_file):
        print(f"Loading hourly data from: {hourly_file}")
        df = pd.read_csv(hourly_file)
        date_col = "timestamp"
    else:
        raise FileNotFoundError("Neither daily nor hourly model-ready CSV found in workspace.")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    # 2. Tag spatial zone metadata
    df["zone_id"] = "PCMC_Nigdi"
    df["latitude"] = 18.65
    df["longitude"] = 73.76

    # 3. Regularize short sensor downtime (forward-fill up to 2 steps)
    core_vars = ["pm25", "pm10", "no2", "o3", "wind_speed", "wind_direction"]
    existing_vars = [c for c in core_vars if c in df.columns]
    df[existing_vars] = df[existing_vars].ffill(limit=2)

    # 4. Set temporal index for resampling
    df_ts = df.set_index(date_col)

    # 5. Resample to Monthly Frequency (ME - Month End)
    agg_rules = {}
    if "pm25" in df_ts.columns:
        agg_rules["pm25"] = ["mean", "max", "min", "std"]
    if "pm10" in df_ts.columns:
        agg_rules["pm10"] = ["mean", "max"]
    if "no2" in df_ts.columns:
        agg_rules["no2"] = ["mean"]
    if "o3" in df_ts.columns:
        agg_rules["o3"] = ["mean"]
    if "wind_speed" in df_ts.columns:
        agg_rules["wind_speed"] = ["mean"]
    if "wind_direction" in df_ts.columns:
        agg_rules["wind_direction"] = ["mean"]

    monthly = df_ts.resample("ME").agg(agg_rules)
    monthly.columns = ["_".join(col).strip() for col in monthly.columns.values]

    # Calculate days exceeding National Ambient Air Quality Standards (PM2.5 > 60 ug/m3)
    if "pm25" in df_ts.columns:
        # If hourly, calculate daily means first
        if date_col == "timestamp":
            daily_means = df_ts["pm25"].resample("D").mean()
        else:
            daily_means = df_ts["pm25"]
        monthly["pm25_exceedance_days"] = daily_means.gt(60).resample("ME").sum()
        monthly["pm25_severe_days"] = daily_means.gt(120).resample("ME").sum()

    # Reattach zone metadata
    monthly["zone_id"] = "PCMC_Nigdi"
    monthly["latitude"] = 18.65
    monthly["longitude"] = 73.76

    # Round numeric values for clean reporting
    numeric_cols = monthly.select_dtypes(include=[np.number]).columns
    monthly[numeric_cols] = monthly[numeric_cols].round(2)

    # Reset index so date is a column
    monthly = monthly.reset_index()
    monthly = monthly.rename(columns={date_col: "month"})

    # 6. Export Monthly Model-Ready CSV
    output_filename = "Nigdi_OpenAQ_Monthly_Model_Ready.csv"
    monthly.to_csv(output_filename, index=False)

    print(f"\nSUCCESS: Monthly dataset exported to '{output_filename}'")
    print(f"Shape: {monthly.shape[0]} months x {monthly.shape[1]} features")
    print("\nPreview of Monthly Aggregation:")
    print(monthly[["month", "pm25_mean", "pm25_max", "pm25_exceedance_days", "zone_id"]].head())

    return monthly

if __name__ == "__main__":
    run_step2_alignment()
