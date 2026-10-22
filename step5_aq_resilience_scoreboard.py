"""
Step 5: Air Quality Resilience Scoreboard & Intervention Engine
==============================================================
This script converts the hourly PM2.5 machine learning predictions into:
1. A normalized Urban Climate Resilience Air Quality Sub-Score (0 to 100).
2. Risk classification aligned with Indian National Ambient Air Quality Standards (CPCB).
3. Actionable, explainable climate adaptation recommendations triggered by vulnerability levels.
4. Exports the complete evaluation and scoreboard dataset to 'nigdi_aq_resilience_scoreboard.csv'.
"""

import os
import joblib
import pandas as pd
import numpy as np

def compute_aq_resilience_score(pm25_val):
    """
    Computes a 0-100 Resilience Score from PM2.5 concentration (ug/m3)
    Based on India CPCB / NAAQS thresholds:
      - NAAQS 24-hr Standard = 60 ug/m3 (Yields Score 50: Threshold of Acceptable Resilience)
      - Good Air Quality <= 30 ug/m3 (Yields Score 75 - 100: High Resilience)
      - Poor Air Quality 60 - 90 ug/m3 (Yields Score 25 - 50: Vulnerable)
      - Severe Air Quality > 90 ug/m3 (Yields Score 0 - 25: Critical Vulnerability)
    """
    if pd.isna(pm25_val):
        return np.nan
    # Linear piece-wise resilience formula
    score = 100.0 - (pm25_val / 60.0) * 50.0
    return float(np.clip(score, 0.0, 100.0))

def get_resilience_category(score):
    if pd.isna(score):
        return "Unknown"
    elif score >= 75:
        return "High Resilience (Clean Air)"
    elif score >= 50:
        return "Moderate Resilience (Satisfactory)"
    elif score >= 25:
        return "Vulnerable (Warning Level)"
    else:
        return "Critical Vulnerability (Emergency Action)"

def get_adaptation_recommendation(score):
    if pd.isna(score):
        return "Insufficient data for recommendation."
    elif score >= 75:
        return "Normal operations; standard green-space & urban canopy preservation."
    elif score >= 50:
        return "Traffic advisory: Optimize signal timing; incentivize public transit usage."
    elif score >= 25:
        return "Intervention: Deploy mechanical sweepers & water misting on arterial roads; restrict heavy diesel idling."
    else:
        return "Emergency Intervention: Halt active construction dust emissions; enforce low-emission zone restrictions; issue sensitive-group health advisory."

def run_aq_scoreboard():
    print("=" * 65)
    print("STEP 5: AIR QUALITY RESILIENCE SCOREBOARD & INTERVENTION ENGINE")
    print("=" * 65)

    hourly_file = "Nigdi_OpenAQ_Hourly_Model_Ready.csv"
    model_file = "nigdi_next_hour_pm25_gradient_boosting.pkl"

    if not os.path.exists(hourly_file):
        raise FileNotFoundError(f"Missing {hourly_file}. Please run data prep first.")
    
    print(f"Loading hourly observations from: {hourly_file}")
    df = pd.read_csv(hourly_file)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Recreate the exact feature engineering used in AQ_score_hourly.ipynb
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["wind_direction_sin"] = np.sin(np.radians(df["wind_direction"]))
    df["wind_direction_cos"] = np.cos(np.radians(df["wind_direction"]))

    for lag in [1, 2, 3, 6, 12, 24]:
        df[f"pm25_lag_{lag}"] = df["pm25"].shift(lag)

    df["pm25_mean_3"] = df["pm25"].shift(1).rolling(3).mean()
    df["pm25_mean_6"] = df["pm25"].shift(1).rolling(6).mean()
    df["pm25_mean_24"] = df["pm25"].shift(1).rolling(24).mean()

    # Target: next hour PM2.5
    df["pm25_next_hour"] = df["pm25"].shift(-1)
    df["next_timestamp"] = df["timestamp"].shift(-1)
    is_next_hour = (df["next_timestamp"] - df["timestamp"]).dt.total_seconds().eq(3600)
    df.loc[~is_next_hour, "pm25_next_hour"] = np.nan

    features = [
        "pm25", "pm10", "no2", "o3",
        "wind_speed",
        "wind_direction_sin", "wind_direction_cos",
        "hour_sin", "hour_cos",
        "day_of_week", "month",
        "pm25_lag_1", "pm25_lag_2", "pm25_lag_3",
        "pm25_lag_6", "pm25_lag_12", "pm25_lag_24",
        "pm25_mean_3", "pm25_mean_6", "pm25_mean_24"
    ]

    model_df = df[features + ["pm25_next_hour", "timestamp"]].dropna(subset=["pm25_next_hour"]).copy()
    split_index = int(len(model_df) * 0.80)
    test_df = model_df.iloc[split_index:].copy()

    # Load trained Gradient Boosting pipeline
    if os.path.exists(model_file):
        print(f"Loading trained Gradient Boosting model from: {model_file}")
        model = joblib.load(model_file)
        predictions = model.predict(test_df[features])
    else:
        print("Model file not found; fitting pipeline directly on train set...")
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.ensemble import GradientBoostingRegressor

        train_df = model_df.iloc[:split_index]
        model = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("gradient_boosting", GradientBoostingRegressor(
                n_estimators=300, learning_rate=0.03, max_depth=3, loss="huber", random_state=42
            ))
        ])
        model.fit(train_df[features], train_df["pm25_next_hour"])
        joblib.dump(model, model_file)
        predictions = model.predict(test_df[features])

    # Build Scoreboard DataFrame
    scoreboard = pd.DataFrame({
        "timestamp": test_df["timestamp"].values,
        "zone_id": "PCMC_Nigdi",
        "actual_pm25": test_df["pm25_next_hour"].values.round(2),
        "predicted_pm25": np.round(predictions, 2)
    })

    # Compute 0-100 Resilience Scores
    scoreboard["observed_resilience_score"] = scoreboard["actual_pm25"].apply(compute_aq_resilience_score).round(1)
    scoreboard["predicted_resilience_score"] = scoreboard["predicted_pm25"].apply(compute_aq_resilience_score).round(1)

    # Classify Risk Category & Trigger Adaptation Action
    scoreboard["resilience_category"] = scoreboard["predicted_resilience_score"].apply(get_resilience_category)
    scoreboard["recommended_action"] = scoreboard["predicted_resilience_score"].apply(get_adaptation_recommendation)

    # Export Scoreboard CSV
    output_csv = "nigdi_aq_resilience_scoreboard.csv"
    scoreboard.to_csv(output_csv, index=False)

    print(f"\nSUCCESS: Air Quality Scoreboard exported to '{output_csv}'")
    print(f"Total Test Observations: {len(scoreboard)}")
    print(f"Average Predicted PM2.5: {scoreboard['predicted_pm25'].mean():.2f} ug/m3")
    print(f"Average AQ Resilience Score: {scoreboard['predicted_resilience_score'].mean():.1f} / 100")
    print("\nResilience Category Breakdown:")
    print(scoreboard["resilience_category"].value_counts(normalize=True).mul(100).round(1).astype(str) + "%")

    print("\nScoreboard Sample (First 5 hours):")
    print(scoreboard[["timestamp", "actual_pm25", "predicted_pm25", "predicted_resilience_score", "resilience_category"]].head())

    return scoreboard

if __name__ == "__main__":
    run_aq_scoreboard()
