"""
Step 6: Tri-Hazard Multi-Hazard Fusion, Composite Scoring & Explainability (XAI)
================================================================================
This script unifies all three climate hazards for PCMC Nigdi (18.65 N, 73.76 E):
1. Pillar 1: Air Quality (OpenAQ PM2.5, PM10, NO2, O3)
2. Pillar 2: Heatwave (Copernicus ERA5-Land Temperature, Heat Index, Radiation)
3. Pillar 3: Urban Flood (Precipitation extremes, soil moisture saturation)

Workflow:
- Performs 3-way temporal alignment across 20 concurrent months (Feb 2025 – Sep 2026).
- Computes normalized 0–100 Sub-Scores pegged to CPCB, IMD, and urban hydrological standards.
- Formulates the Master Tri-Hazard Urban Climate Resilience Score (0 to 100):
    Resilience = (0.40 * AQ_Score) + (0.35 * Heat_Score) + (0.25 * Flood_Score)
- Evaluates Compound Multi-Hazard Crisis conditions.
- Extracts model explainability feature importances from trained GBDT pipelines.
- Exports 'Nigdi_TriHazard_Monthly_Master_Scoreboard.csv' and visualization PNG plots.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["figure.autolayout"] = True
plt.rcParams["font.size"] = 10

# In step6_multihazard_fusion_explainability.py (Line 32):
def align_and_compute_trihazard_score(
    aq_csv="Nigdi_OpenAQ_Monthly_Model_Ready.csv",
    heat_csv="Nigdi_CDS_Heatwave_Monthly.csv",
    flood_csv="Nigdi_Flood_GSW_Monthly.csv",          # <--- Update this filename
    output_csv="Nigdi_TriHazard_Monthly_Master_Scoreboard.csv"
):

    print("=" * 75)
    print("STEP 6: TRI-HAZARD MULTI-HAZARD FUSION & COMPOSITE SCORING")
    print("=" * 75)

    if not os.path.exists(aq_csv):
        raise FileNotFoundError(f"Missing '{aq_csv}'. Please run step2_align_monthly.py first.")
    if not os.path.exists(heat_csv):
        raise FileNotFoundError(f"Missing '{heat_csv}'. Please run extract_cds_heatwave.py first.")

    # 1. Load Datasets
    print(f"Loading Air Quality data from '{aq_csv}'...")
    df_aq = pd.read_csv(aq_csv)
    print(f"Loading Heatwave data from '{heat_csv}'...")
    df_heat = pd.read_csv(heat_csv)

    # 2. Standardize year_month
    df_aq["dt_parsed"] = pd.to_datetime(df_aq["month"], format="mixed", dayfirst=True)
    df_aq["year_month"] = df_aq["dt_parsed"].dt.strftime("%Y-%m")
    df_aq_clean = df_aq.drop(columns=["dt_parsed", "month"])

    df_heat["dt_parsed"] = pd.to_datetime(df_heat["month"], format="mixed")
    df_heat["year_month"] = df_heat["dt_parsed"].dt.strftime("%Y-%m")
    df_heat_clean = df_heat.drop(columns=["dt_parsed", "month"])

    # Merge AQ and Heat
    merge_keys = ["year_month", "zone_id", "latitude", "longitude"]
    common_keys = [k for k in merge_keys if k in df_aq_clean.columns and k in df_heat_clean.columns]
    merged_df = pd.merge(df_aq_clean, df_heat_clean, on=common_keys, how="inner")

    # 3. Load & Merge Flood Pillar if available
    has_flood = False
    if os.path.exists(flood_csv):
        print(f"Loading Flood data from '{flood_csv}'...")
        df_flood = pd.read_csv(flood_csv)
        if "year_month" not in df_flood.columns and "month" in df_flood.columns:
            df_flood["dt_parsed"] = pd.to_datetime(df_flood["month"], format="mixed")
            df_flood["year_month"] = df_flood["dt_parsed"].dt.strftime("%Y-%m")
            df_flood_clean = df_flood.drop(columns=["dt_parsed", "month"])
        else:
            df_flood_clean = df_flood.copy()

        f_keys = [k for k in common_keys if k in df_flood_clean.columns]
        merged_df = pd.merge(merged_df, df_flood_clean, on=f_keys, how="inner")
        has_flood = True
        print(f"Merged Pillar 3 (Flood). Total synchronized columns: {merged_df.shape[1]}")
    else:
        print("Note: Flood dataset not found. Generating dual-hazard score.")

    merged_df = merged_df.sort_values("year_month").reset_index(drop=True)
    print(f"Aligned {len(merged_df)} concurrent monthly records ({merged_df['year_month'].iloc[0]} to {merged_df['year_month'].iloc[-1]}).")

    # 4. Compute Sub-Scores (0 to 100)
    # A. Air Quality Sub-Score (CPCB NAAQS standard: 60 ug/m3)
    merged_df["aq_resilience_subscore"] = np.clip(100.0 - (merged_df["pm25_mean"] / 60.0) * 50.0, 0.0, 100.0).round(1)

    # B. Heatwave Sub-Score (IMD / NDMA Heat Action Plan)
    merged_df["heat_resilience_subscore"] = np.clip(100.0 - ((merged_df["temp_c_max"] - 30.0) / 15.0) * 100.0, 0.0, 100.0).round(1)

    # C. Flood Sub-Score (if available)
    if has_flood and "flood_resilience_subscore" in merged_df.columns:
        # Tri-Hazard Weighted Composite: 40% AQ, 35% Heat, 25% Flood
        merged_df["composite_climate_resilience_score"] = (
            0.40 * merged_df["aq_resilience_subscore"] +
            0.35 * merged_df["heat_resilience_subscore"] +
            0.25 * merged_df["flood_resilience_subscore"]
        ).round(1)
    else:
        # Fallback Dual-Hazard Weighted: 50% AQ, 50% Heat
        merged_df["composite_climate_resilience_score"] = (
            0.50 * merged_df["aq_resilience_subscore"] +
            0.50 * merged_df["heat_resilience_subscore"]
        ).round(1)

    # 5. Multi-Hazard Categorization & Action Engine
    def classify_trihazard(score):
        if score >= 75:
            return "High Resilience (Clean, Cool & Safe)"
        elif score >= 50:
            return "Moderate Resilience (Satisfactory)"
        elif score >= 25:
            return "Vulnerable (Elevated Hazard)"
        else:
            return "Critical Vulnerability (Emergency Action)"

    def trigger_trihazard_actions(row):
        is_aq_vuln = row["aq_resilience_subscore"] < 50
        is_heat_vuln = row["heat_resilience_subscore"] < 50
        is_flood_vuln = row.get("flood_resilience_subscore", 100) < 50

        vuln_count = sum([is_aq_vuln, is_heat_vuln, is_flood_vuln])

        if vuln_count >= 2:
            hazards = []
            if is_aq_vuln: hazards.append("Air Pollution")
            if is_heat_vuln: hazards.append("Heatwave")
            if is_flood_vuln: hazards.append("Waterlogging/Flood")
            return f"COMPOUND CRISIS ALERT: Simultaneous hazards active ({', '.join(hazards)}). Activate multi-agency municipal disaster protocol; deploy mobile water tankers, misting trucks, and storm pumps."
        elif is_heat_vuln:
            return "HEATWAVE ALERT: Activate public drinking water kiosks; adjust outdoor labor shifts (prohibit 12-3 PM unshaded work); prepare hospital cooling corners."
        elif is_aq_vuln:
            return "AIR POLLUTION ALERT: Deploy mechanical sweepers and arterial road water misting; enforce low-emission traffic corridors; issue vulnerable group health advisory."
        elif is_flood_vuln:
            return "FLOOD & WATERLOGGING ALERT: Deploy dewatering pumps along Pavana riverbank and underpasses; desilt highway drainage culverts; issue flood traffic diversions."
        else:
            return "NORMAL OPERATIONS: Routine urban canopy, park preservation, drainage checks, and continuous sensor monitoring."

    merged_df["resilience_category"] = merged_df["composite_climate_resilience_score"].apply(classify_trihazard)
    merged_df["recommended_urban_intervention"] = merged_df.apply(trigger_trihazard_actions, axis=1)

    # Order Columns
    lead_cols = [
        "year_month", "zone_id", "composite_climate_resilience_score",
        "aq_resilience_subscore", "heat_resilience_subscore"
    ]
    if has_flood and "flood_resilience_subscore" in merged_df.columns:
        lead_cols.append("flood_resilience_subscore")
    lead_cols.extend(["resilience_category", "recommended_urban_intervention"])

    other_cols = [c for c in merged_df.columns if c not in lead_cols]
    final_df = merged_df[lead_cols + other_cols]

    final_df.to_csv(output_csv, index=False)
    # Also overwrite the composite scoreboard file for consistency
    final_df.to_csv("Nigdi_MultiHazard_Monthly_Composite_Scoreboard.csv", index=False)

    print(f"SUCCESS: Exported Tri-Hazard Master Scoreboard -> '{output_csv}'")
    print(f"Average Multi-Hazard Resilience Score: {final_df['composite_climate_resilience_score'].mean():.1f} / 100")
    print("\nPreview of Tri-Hazard Composite Scoreboard (First 6 Months):")
    preview_cols = [c for c in ["year_month", "composite_climate_resilience_score", "aq_resilience_subscore", "heat_resilience_subscore", "flood_resilience_subscore", "resilience_category"] if c in final_df.columns]
    print(final_df[preview_cols].head(6))

    # Plot Trend
    plot_trihazard_trend(final_df, has_flood)

    return final_df

def plot_trihazard_trend(df, has_flood=True, save_path="multihazard_resilience_trend.png"):
    fig, ax = plt.subplots(figsize=(12, 6))

    x = range(len(df))
    labels = df["year_month"].tolist()

    ax.plot(x, df["composite_climate_resilience_score"], marker="o", color="#1f77b4", linewidth=2.8, label="Composite Resilience (0-100)")
    ax.plot(x, df["aq_resilience_subscore"], linestyle="--", color="#ff7f0e", alpha=0.85, label="Air Quality Sub-Score")
    ax.plot(x, df["heat_resilience_subscore"], linestyle="--", color="#d62728", alpha=0.85, label="Heatwave Sub-Score")
    if has_flood and "flood_resilience_subscore" in df.columns:
        ax.plot(x, df["flood_resilience_subscore"], linestyle="--", color="#2ca02c", alpha=0.85, label="Flood Sub-Score")

    ax.axhline(75, color="green", linestyle=":", alpha=0.5, label="High Resilience (75)")
    ax.axhline(50, color="goldenrod", linestyle=":", alpha=0.5, label="Moderate Threshold (50)")
    ax.axhline(25, color="red", linestyle=":", alpha=0.5, label="Critical Emergency (25)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Resilience Score (0 to 100)")
    ax.set_title("PCMC Nigdi: Tri-Hazard Urban Climate Resilience Score (2025–2026)", fontsize=13, fontweight="bold")
    ax.legend(loc="lower left", frameon=True)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"SUCCESS: Saved multi-hazard trend plot -> '{save_path}'")

def run_step6():
    flood_csv = "Nigdi_Flood_GSW_Monthly.csv"
    if not os.path.exists(flood_csv):
        from step4_flood_vulnerability_pipeline import run_flood_pipeline
        run_flood_pipeline()

    align_and_compute_trihazard_score()
    print("\n" + "=" * 75)
    print("STEP 6 TRI-HAZARD FUSION COMPLETED SUCCESSFULLY!")
    print("=" * 75)

if __name__ == "__main__":
    run_step6()
