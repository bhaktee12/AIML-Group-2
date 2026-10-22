"""
Copernicus Climate Data Store (CDS) Heatwave Dataset Extraction & Preprocessing
================================================================================
This script extracts and processes weather/climate datasets downloaded as a ZIP 
from the Copernicus Climate Data Store (CDS, e.g., ERA5-Land or UTCI indices)
and prepares model-ready tabular data for PCMC Nigdi (Lat: 18.65, Lon: 73.76).

Workflow:
1. Detects and extracts the CDS ZIP archive.
2. Identifies downloaded formats (.nc / NetCDF, .grib, or .csv).
3. Reads NetCDF datasets using xarray/netCDF4.
4. Locates the nearest grid cell to PCMC Nigdi (Lat: 18.65, Lon: 73.76).
5. Converts units (Kelvin -> Celsius, m/s wind, relative humidity, heat index).
6. Identifies heatwave conditions (e.g., T_max >= 40 deg C or high heat index).
7. Exports two model-ready files:
   - 'Nigdi_CDS_Heatwave_Hourly.csv'
   - 'Nigdi_CDS_Heatwave_Monthly.csv' (aligned with Nigdi_OpenAQ_Monthly_Model_Ready.csv)
"""

import os
import glob
import zipfile
import numpy as np
import pandas as pd

# ==============================================================================
# CONFIGURATION
# ==============================================================================
DEFAULT_ZIP_NAME = "4575cd428167c5ca8fe76a2f378bba61.zip"
EXTRACT_DIR = "cds_heatwave_extracted"
TARGET_LAT = 18.65   # PCMC Nigdi / Pune
TARGET_LON = 73.76   # PCMC Nigdi / Pune
ZONE_ID = "PCMC_Nigdi"


def check_dependencies():
    """Verifies that required libraries are installed."""
    missing = []
    try:
        import xarray
    except ImportError:
        missing.append("xarray")
    try:
        import netCDF4
    except ImportError:
        missing.append("netCDF4")
    try:
        import scipy
    except ImportError:
        missing.append("scipy")

    if missing:
        print("=" * 70)
        print("WARNING: Missing required scientific packages for NetCDF processing:")
        print(f"  Missing: {', '.join(missing)}")
        print("\nPlease run:")
        print(f"  pip install {' '.join(missing)}")
        print("=" * 70)
        return False
    return True


def extract_zip(zip_path, output_dir):
    """Safely extracts all files from the CDS zip archive."""
    if not os.path.exists(zip_path):
        # If default name not found, search for any .zip in current directory
        zip_candidates = [f for f in glob.glob("*.zip") if not f.startswith(".")]
        if zip_candidates:
            zip_path = zip_candidates[0]
            print(f"Using found ZIP file: {zip_path}")
        else:
            raise FileNotFoundError(f"ZIP file '{zip_path}' not found in directory.")

    print(f"Extracting '{zip_path}' to '{output_dir}/'...")
    os.makedirs(output_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        file_list = zip_ref.namelist()
        print(f"Archive contains {len(file_list)} file(s):")
        for name in file_list:
            print(f"  - {name}")
        zip_ref.extractall(output_dir)

    print("Extraction complete.\n")
    return output_dir


def calculate_relative_humidity(temp_c, dewpoint_c):
    """
    Computes Relative Humidity (%) from Air Temperature and Dewpoint Temperature
    using the August-Roche-Magnus approximation.
    """
    a = 17.625
    b = 243.04
    alpha = (a * temp_c) / (b + temp_c)
    beta = (a * dewpoint_c) / (b + dewpoint_c)
    rh = 100.0 * np.exp(beta - alpha)
    return np.clip(rh, 0.0, 100.0)


def calculate_heat_index(temp_c, rh):
    """
    Calculates the Rothfusz Heat Index / Apparent Temperature (deg C).
    Formula based on NOAA National Weather Service equation.
    """
    # Convert to Fahrenheit for standard NOAA formula
    T = (temp_c * 9.0 / 5.0) + 32.0
    R = rh

    # Simple formula for mild conditions
    hi = 0.5 * (T + 61.0 + ((T - 68.0) * 1.2) + (R * 0.094))

    # Full regression for warm conditions (T >= 80 deg F)
    if hi >= 80.0:
        hi = (-42.379 + 2.04901523 * T + 10.14333127 * R
              - 0.22475541 * T * R - 0.00683783 * T * T
              - 0.05481717 * R * R + 0.00122874 * T * T * R
              + 0.00085282 * T * R * R - 0.00000199 * T * T * R * R)

    # Convert back to Celsius
    hi_c = (hi - 32.0) * 5.0 / 9.0
    return np.round(hi_c, 2)


def process_netcdf_files(extracted_dir):
    """Processes NetCDF (.nc) files extracted from the CDS zip."""
    import xarray as xr

    nc_files = glob.glob(os.path.join(extracted_dir, "**/*.nc"), recursive=True)
    if not nc_files:
        return None

    print(f"Found {len(nc_files)} NetCDF (.nc) file(s).")
    datasets = []

    for f in nc_files:
        print(f"Reading: {os.path.basename(f)}...")
        try:
            ds = xr.open_dataset(f)
            datasets.append(ds)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not datasets:
        return None

    # Merge or use first dataset
    if len(datasets) == 1:
        ds = datasets[0]
    else:
        try:
            ds = xr.merge(datasets)
        except Exception:
            ds = datasets[0]

    print("\nDataset Summary:")
    print(ds)

    # Detect coordinate names (time, lat, lon)
    coord_names = list(ds.coords)
    lat_col = next((c for c in ["latitude", "lat", "LATITUDE"] if c in coord_names), None)
    lon_col = next((c for c in ["longitude", "lon", "LONGITUDE"] if c in coord_names), None)
    time_col = next((c for c in ["valid_time", "time", "date"] if c in coord_names), None)

    if not lat_col or not lon_col:
        print("Warning: Spatial coordinates not found directly in coords. Attempting to locate in data_vars...")
        # Fallback to direct dataframe conversion
        df = ds.to_dataframe().reset_index()
        return df

    # Normalize longitude to 0..360 or -180..180 if needed
    sample_lon = float(ds[lon_col].values.flat[0])
    target_lon = TARGET_LON
    if sample_lon > 180 and target_lon < 180:
        target_lon = target_lon % 360

    print(f"\nExtracting point nearest to PCMC Nigdi (Lat: {TARGET_LAT}, Lon: {target_lon})...")
    ds_point = ds.sel({lat_col: TARGET_LAT, lon_col: target_lon}, method="nearest")

    nearest_lat = float(ds_point[lat_col].values)
    nearest_lon = float(ds_point[lon_col].values)
    print(f"Nearest grid point selected: Lat {nearest_lat:.3f}, Lon {nearest_lon:.3f}")

    # Convert point slice to pandas DataFrame
    df = ds_point.to_dataframe().reset_index()

    # Rename time column to 'timestamp'
    if time_col and time_col in df.columns:
        df = df.rename(columns={time_col: "timestamp"})
    elif "time" in df.columns:
        df = df.rename(columns={"time": "timestamp"})

    return df


def standardize_heatwave_features(df):
    """
    Standardizes variable names and calculates heatwave indices:
    - Temperature (Kelvin to Celsius)
    - Dewpoint Temperature
    - Relative Humidity
    - Heat Index / Apparent Temperature
    - Daily & Monthly Heatwave threshold flags (IMD criteria)
    """
    df = df.copy()

    # Ensure timestamp is datetime and sort
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

    # 1. Temperature (t2m)
    t2m_candidates = ["t2m", "2m_temperature", "temperature_2m", "T2M"]
    for c in t2m_candidates:
        if c in df.columns:
            # If values > 200, assume Kelvin and convert to Celsius
            if df[c].dropna().mean() > 200:
                df["temp_c"] = (df[c] - 273.15).round(2)
            else:
                df["temp_c"] = df[c].round(2)
            break

    # 2. Dewpoint Temperature (d2m)
    d2m_candidates = ["d2m", "2m_dewpoint_temperature", "dewpoint_2m", "D2M"]
    for c in d2m_candidates:
        if c in df.columns:
            if df[c].dropna().mean() > 200:
                df["dewpoint_c"] = (df[c] - 273.15).round(2)
            else:
                df["dewpoint_c"] = df[c].round(2)
            break

    # 3. Relative Humidity (derive from temp and dewpoint if not directly present)
    if "temp_c" in df.columns and "dewpoint_c" in df.columns:
        df["relative_humidity"] = calculate_relative_humidity(df["temp_c"], df["dewpoint_c"]).round(1)

    # 4. Heat Index / Apparent Temperature
    if "temp_c" in df.columns and "relative_humidity" in df.columns:
        df["heat_index"] = [calculate_heat_index(t, r) for t, r in zip(df["temp_c"], df["relative_humidity"])]

    # 5. Surface Solar Radiation (ssrd)
    ssrd_candidates = ["ssrd", "surface_solar_radiation_downwards"]
    for c in ssrd_candidates:
        if c in df.columns:
            # Convert Joules/m2 to MJ/m2 if needed
            if df[c].dropna().mean() > 10000:
                df["solar_radiation_mj"] = (df[c] / 1e6).round(2)
            else:
                df["solar_radiation_mj"] = df[c].round(2)
            break

    # Add Zone Identifier
    df["zone_id"] = ZONE_ID
    df["latitude"] = TARGET_LAT
    df["longitude"] = TARGET_LON

    return df


def aggregate_monthly(df):
    """
    Resamples hourly or daily data to a uniform Monthly Time Step ('ME')
    to merge with 'Nigdi_OpenAQ_Monthly_Model_Ready.csv'.
    """
    if "timestamp" not in df.columns:
        return None

    df_ts = df.set_index("timestamp")

    agg_rules = {}
    if "temp_c" in df_ts.columns:
        agg_rules["temp_c"] = ["mean", "max", "min"]
    if "dewpoint_c" in df_ts.columns:
        agg_rules["dewpoint_c"] = ["mean"]
    if "relative_humidity" in df_ts.columns:
        agg_rules["relative_humidity"] = ["mean", "min"]
    if "heat_index" in df_ts.columns:
        agg_rules["heat_index"] = ["mean", "max"]
    if "solar_radiation_mj" in df_ts.columns:
        agg_rules["solar_radiation_mj"] = ["mean", "sum"]

    monthly = df_ts.resample("ME").agg(agg_rules)
    monthly.columns = ["_".join(col).strip() for col in monthly.columns.values]

    # IMD Heatwave Days Criteria:
    # A heatwave day in Indian plateau/plains is defined when Daily Max Temp >= 40 deg C
    # or Apparent Heat Index >= 41 deg C (extreme caution/danger)
    if "temp_c" in df_ts.columns:
        daily_max_temp = df_ts["temp_c"].resample("D").max()
        monthly["heatwave_days_temp40"] = daily_max_temp.ge(40.0).resample("ME").sum()
        monthly["warm_days_temp37"] = daily_max_temp.ge(37.0).resample("ME").sum()

    if "heat_index" in df_ts.columns:
        daily_max_hi = df_ts["heat_index"].resample("D").max()
        monthly["severe_heat_stress_days"] = daily_max_hi.ge(41.0).resample("ME").sum()

    # Reattach Zone Metadata
    monthly["zone_id"] = ZONE_ID
    monthly["latitude"] = TARGET_LAT
    monthly["longitude"] = TARGET_LON

    # Reset Index
    monthly = monthly.reset_index().rename(columns={"timestamp": "month"})
    numeric_cols = monthly.select_dtypes(include=[np.number]).columns
    monthly[numeric_cols] = monthly[numeric_cols].round(2)

    return monthly


def run_pipeline(zip_filename=DEFAULT_ZIP_NAME):
    print("=" * 70)
    print("COPERNICUS CLIMATE DATA STORE (CDS) HEATWAVE EXTRACTION PIPELINE")
    print("=" * 70)

    # 1. Dependency check
    if not check_dependencies():
        return

    # 2. Extract ZIP
    extracted_folder = extract_zip(zip_filename, EXTRACT_DIR)

    # 3. Read and process NetCDF (.nc)
    raw_df = process_netcdf_files(extracted_folder)

    if raw_df is None or raw_df.empty:
        # Check for CSV fallback
        csv_files = glob.glob(os.path.join(extracted_folder, "**/*.csv"), recursive=True)
        if csv_files:
            print(f"Found CSV file in archive: {csv_files[0]}")
            raw_df = pd.read_csv(csv_files[0])
        else:
            print("Error: No NetCDF (.nc) or CSV files could be parsed from the archive.")
            return

    # 4. Standardize and engineer Heatwave features
    print("\nEngineering heatwave features & converting units...")
    hourly_df = standardize_heatwave_features(raw_df)

    # 5. Export Hourly Dataset
    hourly_output = "Nigdi_CDS_Heatwave_Hourly.csv"
    hourly_df.to_csv(hourly_output, index=False)
    print(f"\nSUCCESS: Exported hourly heatwave data -> '{hourly_output}'")
    print(f"Shape: {hourly_df.shape[0]} rows x {hourly_df.shape[1]} columns")
    print("Preview of Hourly Features:")
    print(hourly_df.head(3))

    # 6. Monthly Aggregation
    print("\nResampling to Monthly Frequency for Multi-Hazard Alignment...")
    monthly_df = aggregate_monthly(hourly_df)
    if monthly_df is not None:
        monthly_output = "Nigdi_CDS_Heatwave_Monthly.csv"
        monthly_df.to_csv(monthly_output, index=False)
        print(f"SUCCESS: Exported monthly heatwave data -> '{monthly_output}'")
        print(f"Shape: {monthly_df.shape[0]} months x {monthly_df.shape[1]} features")
        print("\nMonthly Heatwave Statistics (First 5 Months):")
        display_cols = [c for c in ["month", "temp_c_mean", "temp_c_max", "heatwave_days_temp40", "zone_id"] if c in monthly_df.columns]
        print(monthly_df[display_cols].head())

    print("\n" + "=" * 70)
    print("ALL STEPS COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()
