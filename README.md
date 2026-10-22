# AI-Based Urban Climate Resilience Score Using Multimodal Environmental Intelligence

[![Repository](https://img.shields.io/badge/GitHub-bhaktee12%2FAIML--Group--2-blue.svg?logo=github)](https://github.com/bhaktee12/AIML-Group-2)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Location: Nigdi, Pune](https://img.shields.io/badge/Pilot%20Zone-Nigdi%2C%20PCMC%20(18.65%C2%B0N%2C%2073.76%C2%B0E)-orange.svg)](https://maps.google.com/?q=18.65,73.76)

An end-to-end multimodal Artificial Intelligence system designed to assess, predict, and explain multi-hazard urban climate vulnerability across the **PCMC Nigdi Corridor, Pune, India**. 

The system integrates ground sensor feeds (OpenAQ), satellite surface water dynamics (EC JRC Global Surface Water), and Copernicus ERA5-Land climate reanalysis into an explainable **Tri-Hazard Urban Climate Resilience Score (0–100)** and delivers actionable adaptation recommendations through an interactive Streamlit intelligence dashboard.

---

## 📚 Dedicated Technical Documentation

For deep technical specifications, mathematical derivations, and architecture blueprints, refer to:
- 🏗️ **[System Architecture (`architecture.md`)](architecture.md)**: 6-tier modular architecture, dataflow diagrams, pipeline synchronization, and Leaflet/Streamlit UI design.
- 🔬 **[Scientific Methodology (`methodology.md`)](methodology.md)**: In-depth mathematical formulations, CPCB NAAQS / IMD Heat Action Plan benchmarks, GeoAI flood class imbalance resolution, GBDT Huber-loss training, and XAI feature importance.
- 🎤 **[Presentation & Defense Guide (`presentation_guide.md`)](presentation_guide.md)**: Comprehensive presentation slide deck formulas, examiner Q&A prep, and key talking points.

---

## 🏛️ Academic Context & Credits

- **Course:** Artificial Intelligence & Machine Learning (Formative Assessment I)
- **Institution:** [Pimpri Chinchwad College of Engineering (PCCoE)](http://www.pccoepune.com/)
- **Department:** Department of Information Technology (Third Year, Sem V)
- **Project Guide:** Dr. Roshani Raut
- **Core Team (Group 2):**
  - **Bhaktee Patil** (PRN: 124B1F122)
  - **Shraddha Khalikar** (PRN: 124B1F113)
  - **Reeya Patidar** (PRN: 124B1F132)

---

## 📌 Problem Statement & Core Innovations

Climate change is accelerating the frequency and severity of extreme heatwaves, cloudburst-induced urban waterlogging, and toxic winter smog events in rapidly urbanizing Indian metropolitan corridors. While individual municipal departments independently monitor isolated parameters (weather, air quality, or drainage), these disparate data streams are rarely unified into a single intelligent resilience scoring engine.

### Key Contributions:
1. **Multimodal Ingestion**: Automated pipelines for OpenAQ sensor observations, Copernicus CDS ERA5-Land reanalysis, and JRC satellite water recurrence.
2. **Predictive AI Modeling**: Supervised Gradient Boosting Regressors for next-hour air quality ($\text{PM}_{2.5}$) and heat index predictions with leak-free temporal splitting and cyclical harmonic transforms ($\text{hour}_{\sin/\cos}$, $\text{wind}_{\sin/\cos}$).
3. **Hydrological GeoAI Model**: Multi-criteria flood vulnerability model combining 38-year satellite recurrence, Deccan basalt field capacity soil saturation, and IMD extreme precipitation thresholds ($65\,\text{mm/day}$).
4. **Tri-Hazard Composite Fusion**: A scientifically calibrated composite resilience score ($0–100$) anchored in official Indian standards (CPCB NAAQS, IMD/NDMA Heat Action Plans).
5. **Interactive Intelligence Dashboard**: Decision-support dashboard featuring spatial Leaflet mapping, policy scenario simulators, and model explainability.

---

## 🔬 Scientific Formulations Summary

### 1. Cyclical Harmonic Feature Engineering
$$\text{hour}_{\sin} = \sin\left(\frac{2\pi \times \text{hour}}{24}\right), \quad \text{hour}_{\cos} = \cos\left(\frac{2\pi \times \text{hour}}{24}\right)$$

### 2. Air Quality Resilience Sub-Score (CPCB NAAQS Standard)
Anchored to the Indian 24-hour National Ambient Air Quality Standard ($60\,\mu\text{g/m}^3$):
$$\text{Score}_{\text{AQ}} = \max\left(0, \min\left(100, 100 - \left(\frac{\text{PM}_{2.5}}{60}\right) \times 50\right)\right)$$

### 3. Heatwave Resilience Sub-Score (IMD / NDMA Heat Action Plan)
Calibrated between the $30^\circ\text{C}$ comfortable baseline and $45^\circ\text{C}$ extreme red-alert ceiling:
$$\text{Score}_{\text{Heat}} = \max\left(0, \min\left(100, 100 - \left(\frac{T_{\max} - 30}{15}\right) \times 100\right)\right)$$

### 4. Flood Resilience Sub-Score (Hydrological & Geotechnical Multi-Criteria)
Penalizes IMD heavy rainfall ($>65\,\text{mm/day}$), JRC 38-year water recurrence, and Deccan basalt field capacity saturation ($>0.42\,\text{m}^3/\text{m}^3$):
$$\text{Score}_{\text{Flood}} = 100 - (\text{Rainfall Penalty} + \text{Satellite Recurrence Penalty} + \text{Soil Saturation Penalty})$$

### 5. Composite Tri-Hazard Master Resilience
$$\text{Resilience}_{\text{Composite}} = (0.40 \times \text{Score}_{\text{AQ}}) + (0.35 \times \text{Score}_{\text{Heat}}) + (0.25 \times \text{Score}_{\text{Flood}})$$

---

## 📂 Project Architecture & Repository Structure

```text
├── app_urban_resilience_dashboard.py      # Main Streamlit web application & UI
├── nigdi_climate_vulnerability_map.html   # Leaflet.js interactive spatial map
├── requirements.txt                       # Core Python dependencies
├── architecture.md                        # Complete 6-tier system architecture docs
├── methodology.md                         # Full mathematical & scientific methodology
├── presentation_guide.md                  # Comprehensive defense Q&A & presentation guide
├── urban_climate_dashboard_frontend_redesign.md # UX/UI architectural redesign docs
├── project_abstract.pdf                   # Formal academic abstract
├── project_presentation.pdf               # Presentation deck slides
│
├── ML & Data Processing Pipelines
│   ├── extract_cds_heatwave.py            # Copernicus CDS ERA5-Land ingestion pipeline
│   ├── extract_gsw_flood.py               # EC JRC Global Surface Water pipeline
│   ├── step2_align_monthly.py             # Temporal alignment to monthly resolution
│   ├── step3_heatwave_modeling.py         # Heatwave Gradient Boosting training script
│   ├── step4_flood_vulnerability_pipeline.py # Hydrological flood resilience scoring
│   ├── step5_aq_resilience_scoreboard.py  # Air Quality resilience evaluation
│   ├── step5_heatwave_resilience_scoreboard.py # Heatwave resilience scoreboard
│   ├── step6_multihazard_fusion_explainability.py # Fusion & feature importance export
│   └── step7_nigdi_spatial_map.py         # Geospatial Leaflet HTML generator
│
├── Research & Experimentation Notebooks
│   ├── AQ_score.ipynb                     # AQ feature engineering & baseline models
│   ├── AQ_score_hourly.ipynb              # Hourly lag autoregression analysis
│   ├── Heatwave_dataset.ipynb             # Heat index & temperature EDA
│   ├── openAQ_dataset.ipynb               # OpenAQ API ingestion & station calibration
│   └── Verifications_from_workspace.ipynb # Formula & dataset consistency tests
│
├── Trained ML Models (.pkl)
│   ├── nigdi_next_hour_heat_gradient_boosting.pkl
│   └── nigdi_next_hour_pm25_gradient_boosting.pkl
│
├── Model-Ready Datasets & Scoreboards (.csv)
│   ├── Nigdi_TriHazard_Monthly_Master_Scoreboard.csv # Core dashboard scoreboard
│   ├── Nigdi_MultiHazard_Monthly_Composite_Scoreboard.csv
│   ├── Nigdi_Future_Risk_Baseline_Forecast.csv      # 6-Month baseline risk forecast
│   ├── Nigdi_OpenAQ_Monthly_Model_Ready.csv
│   ├── Nigdi_CDS_Heatwave_Monthly.csv
│   ├── Nigdi_Flood_GSW_Monthly.csv
│   ├── Nigdi_Flood_Vulnerability_Monthly.csv
│   └── Nigdi_OpenAQ_Hourly_Model_Ready.csv
│
├── Visual Assets & Diagrams
│   ├── Report Images/                     # System architecture & infographic charts
│   ├── climate_resilience_mark.svg        # Dashboard brand SVG mark
│   ├── favicon_climate_community.svg      # Favicon SVG
│   ├── sidebar_building_mark.svg          # Sidebar icon SVG
│   └── feature_importance_*.png           # Exported SHAP/feature importance plots
```

---

## 🚀 Quickstart & Local Setup

### 1. Clone the Repository
```bash
git clone https://github.com/bhaktee12/AIML-Group-2.git
cd AIML-Group-2
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Resilience Dashboard
```bash
streamlit run app_urban_resilience_dashboard.py
```
Open your browser at `http://localhost:8501` to interact with the dashboard.

---

## 📊 Key Highlights of the Dashboard

- **Executive KPI Cards**: Real-time display of Composite Resilience (e.g., $78.2/100$), Air Quality ($86.5$), Heat Resilience ($100.0$), and Flood Vulnerability ($34.4$).
- **Interactive Geospatial Map**: Leaflet map embedded with ward-level vulnerability markers, risk buffers, and municipal waterlogged hotspots along the Old Mumbai–Pune Highway.
- **Explainable AI (XAI)**: Feature importance analysis demonstrating how 24h rolling $\text{PM}_{2.5}$ lags, cyclical wind vectors, and surface soil moisture drive vulnerability predictions.
- **Interactive Policy Simulator**: Real-time "What-If" slider allowing urban planners to simulate interventions (e.g., $+20\%$ urban tree canopy, stormwater drain desilting, water misting deployment).
- **Seasonal Risk Baseline**: 6-month seasonal planning forecast for local municipal budget allocation.

---

## 📜 License
This project is open-source under the [MIT License](LICENSE).
