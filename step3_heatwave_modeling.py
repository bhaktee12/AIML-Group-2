"""
Step 3 & 4: Heatwave Predictive Modeling Pipeline (Hourly Resolution)
====================================================================
Mirrors AQ_score_hourly.ipynb for the Heatwave Pillar:
1. Loads 'Nigdi_CDS_Heatwave_Hourly.csv' (from extract_cds_heatwave.py).
2. Features: Cyclical temporal transforms, multi-step lags (1h-24h), rolling averages.
3. Target: Next-Hour Temperature ('temp_next_hour') and Heat Index.
4. Chronological 80/20 Train-Test split (no temporal leakage).
5. Trains Gradient Boosting Regressor with Huber loss for extreme temperature resilience.
6. Evaluates against a strong Persistence Baseline (MAE, RMSE, R2).
7. Serializes the model to 'nigdi_next_hour_heat_gradient_boosting.pkl'.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def run_heatwave_modeling(input_csv="Nigdi_CDS_Heatwave_Hourly.csv", model_output="nigdi_next_hour_heat_gradient_boosting.pkl"):
    print("=" * 70)
    print("HEATWAVE PREDICTIVE MODELING PIPELINE (GRADIENT BOOSTING)")
    print("=" * 70)

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Missing '{input_csv}'. Please run extract_cds_heatwave.py first.")

    print(f"Loading hourly heat observations from '{input_csv}'...")
    df = pd.read_csv(input_csv)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # 1. Temporal & Cyclical Features
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # 2. Temperature & Heat Index Lags (shift(1) prevents lookahead bias)
    temp_col = "temp_c" if "temp_c" in df.columns else "temperature"
    if temp_col not in df.columns:
        raise KeyError(f"Temperature column not found in {df.columns.tolist()}")

    for lag in [1, 2, 3, 6, 12, 24]:
        df[f"temp_lag_{lag}"] = df[temp_col].shift(lag)

    df["temp_mean_3"] = df[temp_col].shift(1).rolling(3).mean()
    df["temp_mean_6"] = df[temp_col].shift(1).rolling(6).mean()
    df["temp_mean_24"] = df[temp_col].shift(1).rolling(24).mean()

    # Relative humidity lags if available
    if "relative_humidity" in df.columns:
        df["rh_lag_1"] = df["relative_humidity"].shift(1)
        df["rh_lag_3"] = df["relative_humidity"].shift(3)

    # 3. Target: Next-Hour Temperature
    df["temp_next_hour"] = df[temp_col].shift(-1)
    df["next_timestamp"] = df["timestamp"].shift(-1)
    
    # Check timestamp continuity (3600 seconds)
    is_next_hour = (df["next_timestamp"] - df["timestamp"]).dt.total_seconds().eq(3600)
    df.loc[~is_next_hour, "temp_next_hour"] = np.nan

    # Feature List
    feature_cols = [
        temp_col, "hour_sin", "hour_cos", "day_of_week", "month",
        "temp_lag_1", "temp_lag_2", "temp_lag_3", "temp_lag_6", "temp_lag_12", "temp_lag_24",
        "temp_mean_3", "temp_mean_6", "temp_mean_24"
    ]
    if "relative_humidity" in df.columns:
        feature_cols.extend(["relative_humidity", "rh_lag_1", "rh_lag_3"])
    if "solar_radiation_mj" in df.columns:
        feature_cols.append("solar_radiation_mj")

    # Filter out missing targets
    model_df = df[feature_cols + ["temp_next_hour", "timestamp"]].dropna(subset=["temp_next_hour"]).copy()

    # 4. Chronological 80/20 Train-Test Split
    split_index = int(len(model_df) * 0.80)
    train_df = model_df.iloc[:split_index]
    test_df = model_df.iloc[split_index:]

    X_train = train_df[feature_cols]
    y_train = train_df["temp_next_hour"]
    X_test = test_df[feature_cols]
    y_test = test_df["temp_next_hour"]

    print(f"Training Observations: {len(train_df)}")
    print(f"Testing Observations:  {len(test_df)}")

    # 5. Train Gradient Boosting Model
    print("\nTraining Gradient Boosting Regressor (Huber Loss, 300 estimators)...")
    pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("model", GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3,
            loss="huber",
            random_state=42
        ))
    ])
    pipeline.fit(X_train, y_train)

    # 6. Evaluation
    predictions = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    # Baseline: Persistence (tomorrow/next-hour = current reading)
    baseline_pred = X_test[temp_col]
    b_mae = mean_absolute_error(y_test, baseline_pred)
    b_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
    b_r2 = r2_score(y_test, baseline_pred)

    print("\n" + "-" * 45)
    print("MODEL PERFORMANCE RESULTS")
    print("-" * 45)
    print(f"Gradient Boosting MAE:  {mae:.2f} deg C")
    print(f"Gradient Boosting RMSE: {rmse:.2f} deg C")
    print(f"Gradient Boosting R2:   {r2:.3f}")
    print("\nPersistence Baseline:")
    print(f"Baseline MAE:           {b_mae:.2f} deg C")
    print(f"Baseline RMSE:          {b_rmse:.2f} deg C")
    print(f"Baseline R2:            {b_r2:.3f}")
    print("-" * 45)

    # 7. Serialize Model
    joblib.dump(pipeline, model_output)
    print(f"\nSUCCESS: Serialized model saved to '{model_output}'")

    return pipeline, test_df, predictions

if __name__ == "__main__":
    run_heatwave_modeling()
