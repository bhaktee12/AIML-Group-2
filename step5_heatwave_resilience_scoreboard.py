"""
Step 5: Heatwave Resilience Scoreboard & Urban Adaptation Engine
================================================================
Mirrors step5_aq_resilience_scoreboard.py for the Heatwave Pillar:
1. Loads 'Nigdi_CDS_Heatwave_Hourly.csv' and the trained model 'nigdi_next_hour_heat_gradient_boosting.pkl'.
2. Translates hourly temperature / heat index forecasts into a normalized 0-100 Urban Heat Resilience Score.
3. Classifies risks according to Indian Meteorological Department (IMD) / NDMA Heat Action Plan (HAP) color alerts.
4. Generates automated, explainable municipal adaptation interventions.
5. Exports 'nigdi_heatwave_resilience_scoreboard.csv'.
"""

import os
import joblib
import numpy as np
import pandas as pd

def compute_heat_resilience_score(temp_val):
    """
    Computes a 0-100 Heat Resilience Score from Air Temperature (deg C)
    aligned with India Meteorological Department (IMD) & NDMA Heat Action thresholds:
      - T <= 35 deg C : High Resilience (Score 75 - 100)
      - 35 < T <= 39 deg C : Moderate Resilience / Yellow Watch (Score 50 - 74)
      - 39 < T <= 42 deg C : Vulnerable / Orange Alert (Score 25 - 49)
      - T > 42 deg C : Critical Vulnerability / Red Heatwave Alert (Score 0 - 24)
    """
    if pd.isna(temp_val):
        return np.nan
    # Linear inverse mapping from 30 deg C (pure comfort = 100) to 45 deg C (extreme hazard = 0)
    score = 100.0 - ((temp_val - 30.0) / 15.0) * 100.0
    return float(np.clip(score, 0.0, 100.0))

def get_heat_category(score):
    if pd.isna(score):
        return "Unknown"
    elif score >= 75:
        return "High Resilience (Green - Normal)"
    elif score >= 50:
        return "Moderate Resilience (Yellow - Heat Watch)"
    elif score >= 25:
        return "Vulnerable (Orange - Heat Alert)"
    else:
        return "Critical Vulnerability (Red - Severe Heatwave)"

def get_heat_adaptation_action(score):
    if pd.isna(score):
        return "Insufficient data for recommendation."
    elif score >= 75:
        return "Normal operations; routine maintenance of urban parks, reflective pavements, and tree canopies."
    elif score >= 50:
        return "Heat Watch: Activate public drinking water kiosks; ensure shaded transit stops; issue advisory to hydrate."
    elif score >= 25:
        return "Heat Alert Intervention: Adjust outdoor labor shifts (prohibit 12-3 PM unshaded work); deploy street misting fans; prepare ORS cooling corners at primary health clinics."
    else:
        return "Emergency Heatwave Response: Halt unshaded construction; dispatch emergency water tankers to dense informal settlements; open air-conditioned public cooling shelters; broadcast red alert health warnings."

def run_heatwave_scoreboard(input_csv="Nigdi_CDS_Heatwave_Hourly.csv", model_file="nigdi_next_hour_heat_gradient_boosting.pkl"):
    print("=" * 70)
    print("STEP 5: HEATWAVE RESILIENCE SCOREBOARD & URBAN ADAPTATION ENGINE")
    print("=" * 70)

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Missing '{input_csv}'. Please run extract_cds_heatwave.py first.")

    df = pd.read_csv(input_csv)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    temp_col = "temp_c" if "temp_c" in df.columns else "temperature"

    # Recreate feature engineering
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    for lag in [1, 2, 3, 6, 12, 24]:
        df[f"temp_lag_{lag}"] = df[temp_col].shift(lag)

    df["temp_mean_3"] = df[temp_col].shift(1).rolling(3).mean()
    df["temp_mean_6"] = df[temp_col].shift(1).rolling(6).mean()
    df["temp_mean_24"] = df[temp_col].shift(1).rolling(24).mean()

    if "relative_humidity" in df.columns:
        df["rh_lag_1"] = df["relative_humidity"].shift(1)
        df["rh_lag_3"] = df["relative_humidity"].shift(3)

    # Target: next hour temp
    df["temp_next_hour"] = df[temp_col].shift(-1)
    df["next_timestamp"] = df["timestamp"].shift(-1)
    is_next_hour = (df["next_timestamp"] - df["timestamp"]).dt.total_seconds().eq(3600)
    df.loc[~is_next_hour, "temp_next_hour"] = np.nan

    feature_cols = [
        temp_col, "hour_sin", "hour_cos", "day_of_week", "month",
        "temp_lag_1", "temp_lag_2", "temp_lag_3", "temp_lag_6", "temp_lag_12", "temp_lag_24",
        "temp_mean_3", "temp_mean_6", "temp_mean_24"
    ]
    if "relative_humidity" in df.columns:
        feature_cols.extend(["relative_humidity", "rh_lag_1", "rh_lag_3"])
    if "solar_radiation_mj" in df.columns:
        feature_cols.append("solar_radiation_mj")

    model_df = df[feature_cols + ["temp_next_hour", "timestamp"]].dropna(subset=["temp_next_hour"]).copy()
    split_index = int(len(model_df) * 0.80)
    test_df = model_df.iloc[split_index:].copy()

    # Load Model
    if os.path.exists(model_file):
        print(f"Loading trained Gradient Boosting model from '{model_file}'...")
        pipeline = joblib.load(model_file)
        predictions = pipeline.predict(test_df[feature_cols])
    else:
        print(f"Model '{model_file}' not found. Training model now...")
        from step3_heatwave_modeling import run_heatwave_modeling
        pipeline, _, predictions = run_heatwave_modeling(input_csv, model_file)

    # Build Scoreboard
    scoreboard = pd.DataFrame({
        "timestamp": test_df["timestamp"].values,
        "zone_id": "PCMC_Nigdi",
        "actual_temp_c": test_df["temp_next_hour"].values.round(2),
        "predicted_temp_c": np.round(predictions, 2)
    })

    if "heat_index" in test_df.columns:
        scoreboard["heat_index"] = test_df["heat_index"].values.round(2)

    # Compute 0-100 Resilience Scores
    scoreboard["observed_heat_resilience_score"] = scoreboard["actual_temp_c"].apply(compute_heat_resilience_score).round(1)
    scoreboard["predicted_heat_resilience_score"] = scoreboard["predicted_temp_c"].apply(compute_heat_resilience_score).round(1)

    # Categories & Recommendations
    scoreboard["heat_resilience_category"] = scoreboard["predicted_heat_resilience_score"].apply(get_heat_category)
    scoreboard["recommended_adaptation_action"] = scoreboard["predicted_heat_resilience_score"].apply(get_heat_adaptation_action)

    # Export
    output_csv = "nigdi_heatwave_resilience_scoreboard.csv"
    scoreboard.to_csv(output_csv, index=False)

    print(f"\nSUCCESS: Heatwave Scoreboard exported to '{output_csv}'")
    print(f"Total Test Observations: {len(scoreboard)}")
    print(f"Average Predicted Temp: {scoreboard['predicted_temp_c'].mean():.2f} deg C")
    print(f"Average Heat Resilience Score: {scoreboard['predicted_heat_resilience_score'].mean():.1f} / 100")
    print("\nHeat Resilience Risk Breakdown:")
    print(scoreboard["heat_resilience_category"].value_counts(normalize=True).mul(100).round(1).astype(str) + "%")

    print("\nScoreboard Sample (First 5 Hours):")
    print(scoreboard[["timestamp", "actual_temp_c", "predicted_temp_c", "predicted_heat_resilience_score", "heat_resilience_category"]].head())

    return scoreboard

if __name__ == "__main__":
    run_heatwave_scoreboard()
