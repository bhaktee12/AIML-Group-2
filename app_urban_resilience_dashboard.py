"""
Decision-Support Dashboard for Urban Climate Resilience (PCMC Nigdi)
====================================================================
Course: Artificial Intelligence & Machine Learning (Formative Assessment I)
Institution: Pimpri Chinchwad College of Engineering (PCCoE)
Department: Department of Information Technology (Third Year, Sem V)
Target Pilot Zone: PCMC Nigdi Corridor (18.65°N, 73.76°E), Pune, Maharashtra

Fulfills Five Core Project Objectives:
- Objective 1: Urban Climate Resilience Score for Nigdi (CPCB, IMD, GSW benchmarks)
- Objective 2: Climate vulnerability maps (Interactive Leaflet Geospatial Engine)
- Objective 3: AI-based adaptation recommendations (Multi-Hazard Municipal Directives)
- Objective 4: Decision-support dashboard for urban planners (Data Tables, Exports & XAI)
- Objective 5: Prediction of future climate risks (Seasonal 6-Month Baseline Forecast)

To run this dashboard:
  streamlit run app_urban_resilience_dashboard.py
"""

import os
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

PROJECT_ROOT = Path(__file__).resolve().parent
SIDEBAR_MARK = PROJECT_ROOT / "sidebar_building_mark.svg"

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Urban Climate Resilience AI | PCMC Nigdi",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# THEME STYLING: EDITORIAL, WARM, MODERN CIVIC CLIMATE INTELLIGENCE
# Palette: Warm Ivory (#F7F3EE), Soft Sand (#EEE3D8), Forest Green (#25483F),
# Pale Clay (#DCC2AD), Muted Terracotta (#A96F67), Charcoal (#292D2A)
# ==============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,500;0,600;0,700;1,400;1,600&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=DM+Sans:wght@400;500;700&display=swap');

    :root {
        --fleur-bg: #F7F3EE;                /* Warm ivory background */
        --fleur-surface: #EEE3D8;           /* Soft sand surface */
        --fleur-clay: #DCC2AD;              /* Pale clay accent */
        --fleur-terracotta: #A96F67;        /* Muted terracotta accent */
        --fleur-forest: #25483F;            /* Deep forest green */
        --fleur-forest-dark: #1A342D;       /* Deepest forest green for sidebar */
        --fleur-sage: #8DA394;              /* Sage green */
        --fleur-charcoal: #292D2A;          /* Charcoal primary text */
        --fleur-muted: #74736D;             /* Muted secondary text */
        --fleur-white: #FFFEFC;             /* Pure white card surface */
        --fleur-border: #D9D0C7;            /* Subtle border */
        --fleur-border-subtle: #E8E1D8;     /* Light card border */
        --fleur-shadow: rgba(41, 45, 42, 0.05);
        --fleur-shadow-hover: rgba(41, 45, 42, 0.09);
    }

    * {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Overall Application Canvas */
    .stApp {
        background-color: var(--fleur-bg) !important;
        color: var(--fleur-charcoal) !important;
    }

    section.main > div.block-container {
        padding-top: 1.8rem;
        padding-bottom: 3.5rem;
        max-width: 1260px;
    }

    /* Editorial Serif Headings */
    h1, h2, h3, .editorial-title {
        font-family: 'Cormorant Garamond', Georgia, serif !important;
        color: var(--fleur-charcoal) !important;
        font-weight: 600 !important;
        letter-spacing: 0.3px !important;
    }

    .hero-eyebrow {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        color: var(--fleur-forest) !important;
        font-size: 0.76rem !important;
        font-weight: 800 !important;
        letter-spacing: 2.2px !important;
        text-transform: uppercase !important;
        margin-bottom: 6px !important;
        display: inline-block;
        background: #E8EFEA;
        padding: 4px 12px;
        border-radius: 30px;
        border: 1px solid #D1DFD5;
    }

    .hero-title {
        font-family: 'Cormorant Garamond', Georgia, serif !important;
        font-size: 2.75rem !important;
        line-height: 1.15 !important;
        margin: 8px 0 6px 0 !important;
        font-weight: 700 !important;
        color: var(--fleur-charcoal) !important;
    }

    .hero-subtitle {
        font-size: 0.98rem !important;
        color: var(--fleur-muted) !important;
        line-height: 1.55 !important;
        margin: 0 0 1.4rem 0 !important;
        max-width: 900px;
    }

    .metadata-strip {
        display: flex;
        flex-wrap: wrap;
        gap: 16px;
        align-items: center;
        font-size: 0.8rem;
        color: var(--fleur-muted);
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid var(--fleur-border);
    }

    .metadata-item {
        display: flex;
        align-items: center;
        gap: 6px;
    }

    .metadata-item strong {
        color: var(--fleur-charcoal);
        font-weight: 600;
    }

    /* Top App Header / Streamlit Bar */
    [data-testid="stHeader"] {
        background-color: var(--fleur-bg) !important;
        border-bottom: 1px solid var(--fleur-border) !important;
    }

    /* =========================================================================
       SIDEBAR: REFINED DEEP FOREST GREEN PALETTE
       ========================================================================= */
    [data-testid="stSidebar"] {
        background-color: var(--fleur-forest-dark) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
        color: #F7F3EE !important;
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div {
        color: #F7F3EE;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        font-family: 'Cormorant Garamond', Georgia, serif !important;
        color: #FFFEFC !important;
        letter-spacing: 0.5px !important;
    }

    [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.12) !important;
        margin: 1.2rem 0 !important;
    }

    .sidebar-brand-box {
        padding: 4px 0 12px 0;
    }

    .sidebar-brand-title {
        font-family: 'Cormorant Garamond', Georgia, serif !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        color: #FFFEFC !important;
        line-height: 1.15;
    }

    .sidebar-brand-sub {
        font-size: 0.72rem !important;
        font-weight: 700 !important;
        letter-spacing: 1.5px !important;
        text-transform: uppercase !important;
        color: var(--fleur-sage) !important;
        margin-top: 3px;
    }

    .location-chip {
        display: inline-block;
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.14);
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.74rem;
        color: #F7F3EE;
        margin-top: 8px;
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    .sidebar-nav-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 12px 14px;
        margin-bottom: 14px;
    }

    .sidebar-nav-item {
        font-size: 0.8rem;
        color: #D3DCD5;
        padding: 4px 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .sidebar-footer-note {
        font-size: 0.68rem;
        letter-spacing: 1.6px;
        text-transform: uppercase;
        color: var(--fleur-sage);
        text-align: center;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        font-weight: 700;
    }

    /* Month Selector Control in Sidebar */
    [data-testid="stSidebar"] [data-baseweb="select"] {
        background-color: var(--fleur-white) !important;
        border-radius: 10px !important;
    }

    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background-color: var(--fleur-white) !important;
        border: 1.5px solid #D9D0C7 !important;
        border-radius: 10px !important;
    }

    [data-testid="stSidebar"] [data-baseweb="select"] input,
    [data-testid="stSidebar"] [data-baseweb="select"] span,
    [data-testid="stSidebar"] [data-baseweb="select"] div {
        color: var(--fleur-charcoal) !important;
        font-weight: 700 !important;
        font-size: 14px !important;
    }

    [data-testid="stSidebar"] [data-baseweb="select"] svg {
        fill: var(--fleur-charcoal) !important;
        color: var(--fleur-charcoal) !important;
    }

    /* Dropdown Options Popover */
    div[data-baseweb="popover"],
    div[data-baseweb="popover"] > div,
    ul[role="listbox"],
    [data-baseweb="menu"] {
        background-color: var(--fleur-white) !important;
        color: var(--fleur-charcoal) !important;
        border: 1px solid #D9D0C7 !important;
        border-radius: 12px !important;
        box-shadow: 0 10px 30px rgba(41, 45, 42, 0.15) !important;
    }

    li[role="option"],
    div[role="option"],
    [data-baseweb="menu"] li {
        background-color: var(--fleur-white) !important;
        color: var(--fleur-charcoal) !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        padding: 9px 15px !important;
    }

    li[role="option"]:hover,
    div[role="option"]:hover,
    li[role="option"][aria-selected="true"],
    div[role="option"][aria-selected="true"] {
        background-color: var(--fleur-surface) !important;
        color: var(--fleur-forest) !important;
    }

    li[role="option"]:hover span,
    div[role="option"]:hover span,
    li[role="option"][aria-selected="true"] span,
    div[role="option"][aria-selected="true"] span {
        color: var(--fleur-forest) !important;
        font-weight: 700 !important;
    }

    /* =========================================================================
       CIVIC ADVISORY DIRECTIVE PANEL
       ========================================================================= */
    .advisory-panel {
        padding: 22px 26px;
        border-radius: 16px;
        margin-bottom: 24px;
        box-shadow: 0 4px 18px var(--fleur-shadow);
        border: 1px solid var(--fleur-border);
        transition: transform 0.2s ease;
    }

    .advisory-panel-alert {
        background-color: #F8ECE9;
        border-left: 6px solid var(--fleur-terracotta);
    }

    .advisory-panel-watch {
        background-color: #F7EFE4;
        border-left: 6px solid #C28B38;
    }

    .advisory-panel-normal {
        background-color: #EFF5F1;
        border-left: 6px solid var(--fleur-forest);
    }

    .advisory-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }

    .advisory-eyebrow {
        font-size: 0.74rem;
        font-weight: 800;
        letter-spacing: 1.8px;
        text-transform: uppercase;
        color: var(--fleur-muted);
    }

    .advisory-priority {
        font-size: 0.72rem;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 20px;
        letter-spacing: 0.5px;
    }

    .priority-high {
        background: #E8C9C4;
        color: #7D3E35;
    }

    .priority-medium {
        background: #EEDBBB;
        color: #755219;
    }

    .priority-normal {
        background: #CEE0D5;
        color: #1A3E34;
    }

    .advisory-focus {
        font-size: 0.88rem;
        font-weight: 700;
        color: var(--fleur-charcoal);
        margin-bottom: 8px;
    }

    .advisory-focus span {
        color: var(--fleur-muted);
        font-weight: 600;
        font-size: 0.78rem;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    .advisory-directive {
        font-size: 1.02rem;
        font-weight: 600;
        line-height: 1.55;
        color: var(--fleur-charcoal);
        margin-bottom: 12px;
    }

    .advisory-footer {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.75rem;
        color: var(--fleur-muted);
        padding-top: 8px;
        border-top: 1px solid rgba(41, 45, 42, 0.08);
    }

    /* =========================================================================
       ALIGNED KPI SCORE CARDS (Warm White, Soft Sand, Arch Details)
       ========================================================================= */
    .kpi-card {
        background-color: var(--fleur-white);
        border: 1px solid var(--fleur-border);
        border-radius: 16px;
        padding: 20px 22px;
        box-shadow: 0 4px 18px var(--fleur-shadow);
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px var(--fleur-shadow-hover);
        border-color: var(--fleur-clay);
    }

    .kpi-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }

    .kpi-label {
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 1.1px;
        text-transform: uppercase;
        color: var(--fleur-muted);
    }

    .kpi-standard {
        font-size: 0.68rem;
        font-weight: 600;
        color: var(--fleur-forest);
        background: #E8EFEA;
        padding: 2px 7px;
        border-radius: 6px;
        border: 1px solid #D1DFD5;
    }

    .kpi-value-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
        gap: 6px;
    }

    .kpi-value {
        font-family: 'Cormorant Garamond', Georgia, serif;
        font-size: 2.05rem;
        font-weight: 700;
        color: var(--fleur-charcoal);
        line-height: 1;
        white-space: nowrap;
    }

    .kpi-scale {
        font-size: 0.88rem;
        font-weight: 500;
        color: var(--fleur-muted);
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .kpi-badge {
        font-size: 0.68rem;
        font-weight: 700;
        padding: 3px 8px;
        border-radius: 10px;
        letter-spacing: 0.2px;
        white-space: nowrap;
        flex-shrink: 0;
    }

    .kpi-progress-track {
        width: 100%;
        height: 5px;
        background-color: var(--fleur-surface);
        border-radius: 4px;
        overflow: hidden;
        margin-bottom: 10px;
    }

    .kpi-progress-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.4s ease;
    }

    .kpi-details {
        font-size: 0.78rem;
        color: var(--fleur-muted);
        line-height: 1.45;
        border-top: 1px solid #F2ECE4;
        padding-top: 8px;
    }

    /* =========================================================================
       EDITORIAL STREAMLIT TABS
       ========================================================================= */
    [data-baseweb="tab-list"] {
        background-color: transparent !important;
        gap: 10px !important;
        border-bottom: 1.5px solid var(--fleur-border) !important;
        padding-bottom: 2px !important;
        margin-bottom: 24px !important;
    }

    [data-baseweb="tab"] {
        background-color: transparent !important;
        border-radius: 8px 8px 0 0 !important;
        padding: 10px 18px !important;
        color: var(--fleur-muted) !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.94rem !important;
        letter-spacing: 0.3px !important;
        border: none !important;
        transition: all 0.2s ease !important;
    }

    [data-baseweb="tab"]:hover {
        color: var(--fleur-forest) !important;
    }

    [data-baseweb="tab"][aria-selected="true"] {
        color: var(--fleur-forest) !important;
        border-bottom: 3px solid var(--fleur-forest) !important;
        font-weight: 700 !important;
    }

    /* Content Cards & Recommendations */
    .content-card {
        background: var(--fleur-white);
        border: 1px solid var(--fleur-border);
        border-radius: 16px;
        padding: 22px 24px;
        margin-bottom: 18px;
        box-shadow: 0 3px 14px var(--fleur-shadow);
    }

    .action-card {
        background: var(--fleur-white);
        border: 1px solid var(--fleur-border);
        border-left: 5px solid var(--fleur-forest);
        border-radius: 14px;
        padding: 18px 22px;
        margin-bottom: 16px;
        box-shadow: 0 3px 12px var(--fleur-shadow);
        transition: transform 0.15s ease;
    }

    .action-card:hover {
        transform: translateY(-2px);
    }

    .action-card.terracotta-edge {
        border-left-color: var(--fleur-terracotta);
    }

    .action-card.amber-edge {
        border-left-color: #C28B38;
    }

    .action-card.slate-edge {
        border-left-color: #4D7A94;
    }

    .action-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }

    .action-tag {
        font-size: 0.7rem;
        font-weight: 800;
        letter-spacing: 1.4px;
        text-transform: uppercase;
        color: var(--fleur-muted);
    }

    .action-title {
        font-family: 'Cormorant Garamond', Georgia, serif;
        font-size: 1.4rem;
        font-weight: 700;
        color: var(--fleur-charcoal);
        margin-bottom: 6px;
    }

    .action-text {
        font-size: 0.9rem;
        color: var(--fleur-charcoal);
        line-height: 1.55;
        margin-bottom: 10px;
    }

    .action-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 14px;
        font-size: 0.76rem;
        color: var(--fleur-muted);
        border-top: 1px solid #F2ECE4;
        padding-top: 8px;
    }

    /* Dataframe table border styling */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--fleur-border) !important;
        border-radius: 12px !important;
        overflow: hidden !important;
        background-color: var(--fleur-white) !important;
    }

    /* Buttons */
    .stDownloadButton button {
        background-color: var(--fleur-forest) !important;
        color: #F7F3EE !important;
        border: 1px solid var(--fleur-forest) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 8px 18px !important;
        font-size: 0.88rem !important;
        transition: all 0.2s ease !important;
    }

    .stDownloadButton button:hover {
        background-color: #1A342D !important;
        border-color: #1A342D !important;
        color: #FFFFFF !important;
        transform: translateY(-1px) !important;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. LOAD DATASETS & REPRODUCIBLE PIPELINE ARTIFACTS
# ==============================================================================
SCOREBOARD_FILE = PROJECT_ROOT / "Nigdi_TriHazard_Monthly_Master_Scoreboard.csv"
FALLBACK_SCOREBOARD = PROJECT_ROOT / "Nigdi_MultiHazard_Monthly_Composite_Scoreboard.csv"
FORECAST_FILE = PROJECT_ROOT / "Nigdi_Future_Risk_Baseline_Forecast.csv"
MAP_FILE = PROJECT_ROOT / "nigdi_climate_vulnerability_map.html"

@st.cache_data
def load_data():
    if SCOREBOARD_FILE.exists():
        return pd.read_csv(SCOREBOARD_FILE)
    elif FALLBACK_SCOREBOARD.exists():
        return pd.read_csv(FALLBACK_SCOREBOARD)
    return None

@st.cache_data
def load_forecast():
    if FORECAST_FILE.exists():
        return pd.read_csv(FORECAST_FILE)
    return None

df = load_data()
forecast_df = load_forecast()

# ==============================================================================
# 2. SIDEBAR NAVIGATION & CIVIC CONTROLS
# ==============================================================================
st.sidebar.markdown("""
<div class="sidebar-brand-box">
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 6px;">
        <span style="font-size: 1.7rem;">🌿</span>
        <div>
            <div class="sidebar-brand-title">Urban Resilience</div>
            <div class="sidebar-brand-sub">Decision-Support AI</div>
        </div>
    </div>
    <div class="location-chip">📍 PCMC Nigdi · Pune (18.65°N, 73.76°E)</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("<p style='font-size:0.75rem; font-weight:800; letter-spacing:1.5px; text-transform:uppercase; color:#8DA394; margin-bottom:6px;'>TEMPORAL EVALUATION WINDOW</p>", unsafe_allow_html=True)

if df is not None:
    month_options = df["year_month"].tolist()
    # Default to latest completed evaluation month
    selected_month = st.sidebar.selectbox("Select Evaluation Month", month_options, index=len(month_options)-1)
    month_row = df[df["year_month"] == selected_month].iloc[0]
else:
    selected_month = "2026-09"
    month_row = {}

st.sidebar.markdown("---")

st.sidebar.markdown("""
<p style='font-size:0.75rem; font-weight:800; letter-spacing:1.5px; text-transform:uppercase; color:#8DA394; margin-bottom:10px;'>CORE SCIENTIFIC CAPABILITIES</p>
<div class="sidebar-nav-card">
    <div class="sidebar-nav-item"><span>🟢</span> Objective 1: Urban Climate Score</div>
    <div class="sidebar-nav-item"><span>🗺️</span> Objective 2: GIS Vulnerability Map</div>
    <div class="sidebar-nav-item"><span>🛡️</span> Objective 3: Adaptation Actions</div>
    <div class="sidebar-nav-item"><span>🏛️</span> Objective 4: Planner Decision Hub</div>
    <div class="sidebar-nav-item"><span>📈</span> Objective 5: 6-Month Risk Forecast</div>
</div>
""", unsafe_allow_html=True)

# Pilot zone quick summary
if df is not None and not month_row.empty:
    m_score = float(month_row.get("composite_climate_resilience_score", 75.0))
    st.sidebar.markdown(f"""
    <div style="background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); border-radius: 10px; padding: 12px; margin-top: 8px;">
        <div style="font-size: 0.72rem; color: #8DA394; font-weight: 700; text-transform: uppercase;">Selected Window Status</div>
        <div style="display: flex; justify-content: space-between; align-items: baseline; margin-top: 4px;">
            <span style="font-family: 'Cormorant Garamond', serif; font-size: 1.6rem; font-weight: 700; color: #FFFEFC;">{m_score:.1f} / 100</span>
            <span style="font-size: 0.74rem; color: #D3DCD5;">{selected_month}</span>
        </div>
        <div style="font-size: 0.76rem; color: #A8BDB0; margin-top: 2px;">{month_row.get('resilience_category', 'Satisfactory')}</div>
    </div>
    """, unsafe_allow_html=True)

st.sidebar.markdown("""
<div class="sidebar-footer-note">
    CLIMATE INTELLIGENCE · PCMC PILOT<br>
    <span style="font-size: 0.62rem; color: #6D8275;">OpenAQ · Copernicus CDS · JRC GSW</span>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# 3. HERO INTRODUCTION & CIVIC CONTEXT
# ==============================================================================
st.markdown("""
<span class="hero-eyebrow">URBAN CLIMATE INTELLIGENCE · PCMC NIGDI</span>
<div class="hero-title">Understanding Nigdi's Climate Resilience</div>
<div class="hero-subtitle">
    Synthesizing multimodal environmental intelligence across OpenAQ ground stations, Copernicus ERA5-Land reanalysis, 
    and JRC Global Surface Water recurrence to quantify urban resilience and guide localized municipal adaptation.
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="metadata-strip">
    <div class="metadata-item">📍 <strong>Zone:</strong> PCMC Nigdi Corridor, Pune</div>
    <div class="metadata-item">🌐 <strong>Coordinates:</strong> 18.65°N, 73.76°E</div>
    <div class="metadata-item">📅 <strong>Active Window:</strong> {selected_month}</div>
    <div class="metadata-item">⏱️ <strong>Data Horizon:</strong> Feb 2025 – Sep 2026 (Historical) · Oct 2026 – Mar 2027 (Forecast)</div>
</div>
""", unsafe_allow_html=True)

if df is None:
    st.error("Scoreboard data not found. Please verify that data files are available in the workspace.")
    st.stop()

# ==============================================================================
# 4. ACTIVE MUNICIPAL ADVISORY DIRECTIVE
# ==============================================================================
composite_score = float(month_row.get("composite_climate_resilience_score", 74.0))
aq_subscore = float(month_row.get("aq_resilience_subscore", 75.0))
heat_subscore = float(month_row.get("heat_resilience_subscore", 75.0))
flood_subscore = float(month_row.get("flood_resilience_subscore", 85.0))

is_aq = aq_subscore < 50
is_heat = heat_subscore < 50
is_flood = flood_subscore < 50

# Priority & Advisory Logic
if is_aq and is_heat and is_flood:
    focus_area = "Compound Multi-Hazard Emergency"
    intervention_text = "COMPOUND CRISIS: Synchronize cooling shelters, low-emission freight enforcement along NH-48, arterial culvert clearing, mobile dewatering pumps, and flood traffic diversions."
    alert_level = "alert"
    priority_label = "HIGH PRIORITY · COMPOUND THREAT"
elif is_aq and is_heat:
    focus_area = "Compound Atmospheric Stress (Air & Heat)"
    intervention_text = "DUAL HAZARD: Activate public hydration kiosks, enforce 12–3 PM unshaded labor restrictions, suppress construction dust along NH-48, and restrict heavy diesel transit."
    alert_level = "alert"
    priority_label = "HIGH PRIORITY · DUAL STRESS"
elif is_heat and is_flood:
    focus_area = "Compound Hydro-Thermal Stress"
    intervention_text = "DUAL HAZARD: Adjust outdoor municipal work hours, open cooling shelters, deploy emergency dewatering pumps to Pavana riverbanks, and monitor localized waterborne vectors."
    alert_level = "alert"
    priority_label = "HIGH PRIORITY · DUAL STRESS"
elif is_aq and is_flood:
    focus_area = "Compound Waterlogging & Particulate Burden"
    intervention_text = "DUAL HAZARD: Suppress arterial NH-48 road silt, clear stormwater culverts, deploy dewatering pumps, and barricade submerged low-lying underpasses."
    alert_level = "alert"
    priority_label = "HIGH PRIORITY · DUAL STRESS"
elif is_heat:
    focus_area = "Heatwave & Thermal Island Exposure"
    intervention_text = "HEAT ACTION PLAN (IMD): Restrict unshaded manual labor between 12:00 PM and 3:00 PM; activate public drinking water kiosks; equip primary clinics with cooling corners."
    alert_level = "watch"
    priority_label = "ADAPTATION ACTION REQUIRED"
elif is_aq:
    focus_area = "Vehicular Emissions & Particulate Inversion"
    intervention_text = "AIR QUALITY PROTOCOL (CPCB): Deploy nocturnal mechanical sweepers and water misting cannons along NH-48; enforce low-emission zones; issue public health sensitive-group alerts."
    alert_level = "watch"
    priority_label = "ADAPTATION ACTION REQUIRED"
elif is_flood:
    flood_action = str(month_row.get("flood_adaptation_action", "Deploy mobile dewatering pumps to low-lying Pavana riverbank settlements."))
    focus_area = "Monsoon Hydrology & Waterlogging Defense"
    intervention_text = f"FLOOD MITIGATION (GSW): {flood_action}"
    alert_level = "watch"
    priority_label = "ADAPTATION ACTION REQUIRED"
else:
    focus_area = "Routine Urban Resilience Operations"
    intervention_text = "NORMAL OPERATIONS: Maintain Pradhikaran urban tree canopy, conduct routine stormwater drain and bioswale desilting, and sustain real-time sensor network calibration."
    alert_level = "normal"
    priority_label = "ROUTINE MONITORING ACTIVE"

panel_class = f"advisory-panel-{alert_level}"
priority_badge_class = "priority-high" if alert_level == "alert" else ("priority-medium" if alert_level == "watch" else "priority-normal")

st.markdown(f"""
<div class="advisory-panel {panel_class}">
    <div class="advisory-top">
        <div class="advisory-eyebrow">MUNICIPAL ADVISORY DIRECTIVE · {selected_month}</div>
        <div class="advisory-priority {priority_badge_class}">{priority_label}</div>
    </div>
    <div class="advisory-focus">
        <span>PRIMARY FOCUS:</span> {focus_area}
    </div>
    <div class="advisory-directive">
        {intervention_text}
    </div>
    <div class="advisory-footer">
        <span>Grounded in IMD Heat Action, CPCB NAAQS & JRC Hydrological Thresholds</span>
        <span>PCMC Municipal Planning Corridor</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# 5. FOUR ALIGNED KPI METRIC CARDS
# ==============================================================================
def render_kpi_card(title, score, standard_label, details_line):
    if score >= 75:
        badge_bg = "#E5ECE7"
        badge_text = "#25483F"
        bar_color = "#25483F"
        status_text = "High Resilience"
    elif score >= 50:
        badge_bg = "#F5EDE3"
        badge_text = "#8E6422"
        bar_color = "#C28B38"
        status_text = "Moderate Resilience"
    else:
        badge_bg = "#F8ECE9"
        badge_text = "#A96F67"
        bar_color = "#A96F67"
        status_text = "Vulnerable / Alert"

    fill_pct = max(0.0, min(100.0, score))

    return f"""
    <div class="kpi-card">
        <div>
            <div class="kpi-header">
                <span class="kpi-label">{title}</span>
                <span class="kpi-standard">{standard_label}</span>
            </div>
            <div class="kpi-value-row">
                <div class="kpi-value">{score:.1f}<span class="kpi-scale"> / 100</span></div>
                <span class="kpi-badge" style="background-color: {badge_bg}; color: {badge_text};">{status_text}</span>
            </div>
            <div class="kpi-progress-track">
                <div class="kpi-progress-fill" style="width: {fill_pct}%; background-color: {bar_color};"></div>
            </div>
        </div>
        <div class="kpi-details">{details_line}</div>
    </div>
    """

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        render_kpi_card(
            title="Composite Resilience",
            score=composite_score,
            standard_label="Tri-Hazard Fusion",
            details_line="Weighted Index: Air Quality (40%), Heatwave (35%), Flood (25%)"
        ),
        unsafe_allow_html=True
    )

with c2:
    pm25_val = float(month_row.get("pm25_mean", 0.0))
    exc_days = int(month_row.get("pm25_exceedance_days", 0))
    st.markdown(
        render_kpi_card(
            title="Air Quality Sub-Score",
            score=aq_subscore,
            standard_label="CPCB NAAQS 24-hr",
            details_line=f"PM2.5 Mean: {pm25_val:.1f} µg/m³ · {exc_days} days exceeding 60 µg/m³"
        ),
        unsafe_allow_html=True
    )

with c3:
    max_t = float(month_row.get("temp_c_max", 0.0))
    hot_days = int(month_row.get("heatwave_days_temp40", 0))
    st.markdown(
        render_kpi_card(
            title="Heatwave Resilience",
            score=heat_subscore,
            standard_label="IMD / NDMA HAP",
            details_line=f"Peak Max: {max_t:.1f} °C · {hot_days} days exceeding 40.0 °C threshold"
        ),
        unsafe_allow_html=True
    )

with c4:
    precip_max = float(month_row.get("precip_max_daily_mm", 0.0))
    soil_sat = float(month_row.get("soil_saturation_pct", 0.0))
    st.markdown(
        render_kpi_card(
            title="Flood Resilience",
            score=flood_subscore,
            standard_label="JRC 38-yr & Basalt",
            details_line=f"Max 24h Rain: {precip_max:.1f} mm · Soil Saturation: {soil_sat:.1f}%"
        ),
        unsafe_allow_html=True
    )

st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

# ==============================================================================
# 6. FIVE CORE SECTIONS VIA EDITORIAL TABS
# ==============================================================================
tab_map, tab_forecast, tab_actions, tab_xai, tab_data = st.tabs([
    "🗺️ Vulnerability Map",
    "📈 Seasonal Risk Outlook & Forecast",
    "🛡️ Recommended Actions for Nigdi",
    "🧠 AI Model Explainability (XAI)",
    "📊 Planner Data & Exports"
])

# ------------------------------------------------------------------------------
# TAB 1: GEOSPATIAL CLIMATE VULNERABILITY MAP
# ------------------------------------------------------------------------------
with tab_map:
    st.markdown("""
    <div style="margin-bottom: 12px;">
        <h2 style="font-size: 1.85rem; margin-bottom: 4px;">Where climate risks concentrate</h2>
        <div style="font-size: 0.92rem; color: #74736D; line-height: 1.5;">
            Geospatial Multi-Hazard Vulnerability Engine for PCMC Nigdi and Ravet riparian zones. 
            Overlays the central sensor node, flood inundation buffers, vehicular particulate corridors, and tree canopy cooling oases.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Dynamic Map Generator that reflects the selected evaluation month
    def generate_dynamic_map_html(selected_m, comp_s, aq_s, heat_s, flood_s, cat_s, act_s):
        if comp_s >= 75:
            pin_color = "#25483F"
        elif comp_s >= 50:
            pin_color = "#C28B38"
        else:
            pin_color = "#A96F67"

        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>PCMC Nigdi - Urban Climate Vulnerability Map</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body {{ margin:0; padding:0; font-family: 'Plus Jakarta Sans', sans-serif; background:#F7F3EE; color:#292D2A; }}
        #map {{ width: 100vw; height: 100vh; }}
        .info-box {{
            position: absolute; top: 16px; right: 16px; z-index: 1000;
            background: #FFFEFC; color: #292D2A; padding: 18px 22px; border-radius: 14px;
            border: 1px solid #D9D0C7;
            box-shadow: 0 4px 20px rgba(41,45,42,0.12); max-width: 340px;
        }}
        .info-box h3 {{
            font-family: 'Cormorant Garamond', Georgia, serif;
            font-size: 21px; font-weight: 700; margin: 0 0 2px 0; color: #292D2A;
        }}
        .score-pill {{
            font-family: 'Cormorant Garamond', Georgia, serif;
            font-size: 22px; font-weight: 700; padding: 4px 14px;
            border-radius: 20px; display: inline-block; color: #FFFEFC;
            background-color: {pin_color}; margin: 8px 0;
        }}
        .gauge-row {{ display: flex; justify-content: space-between; margin: 5px 0; font-size: 12.5px; color: #74736D; }}
        .gauge-row strong {{ color: #292D2A; }}
        .legend {{
            position: absolute; bottom: 25px; left: 16px; z-index: 1000;
            background: #FFFEFC; color: #292D2A; padding: 14px 18px; border-radius: 12px;
            border: 1px solid #D9D0C7;
            box-shadow: 0 4px 14px rgba(41,45,42,0.1); font-size: 12px;
        }}
        .legend strong {{ font-family: 'Cormorant Garamond', Georgia, serif; font-size: 15px; color: #292D2A; }}
        .legend-item {{ display: flex; align-items: center; margin-top: 6px; font-size: 12px; }}
        .color-dot {{ width: 11px; height: 11px; border-radius: 50%; display: inline-block; margin-right: 8px; flex-shrink: 0; }}
        .color-line {{ width: 18px; height: 4px; display: inline-block; margin-right: 8px; border-radius: 2px; flex-shrink: 0; }}
    </style>
</head>
<body>
    <div id="map"></div>

    <div class="info-box">
        <h3>PCMC Nigdi · Central Station</h3>
        <span style="font-size:11.5px; color:#74736D;">Selected Window: <strong>{selected_m}</strong> · (18.65°N, 73.76°E)</span>
        <div>
            <span class="score-pill">{comp_s:.1f} / 100</span>
            <div style="font-weight:600; font-size:12.5px; color:#292D2A;">{cat_s}</div>
        </div>
        <hr style="border:0; border-top:1px solid #E8E1D8; margin:9px 0;">
        <div class="gauge-row"><span>Air Quality (CPCB):</span><strong>{aq_s:.1f} / 100</strong></div>
        <div class="gauge-row"><span>Heatwave (IMD HAP):</span><strong>{heat_s:.1f} / 100</strong></div>
        <div class="gauge-row"><span>Flood (Hydrology):</span><strong>{flood_s:.1f} / 100</strong></div>
        <hr style="border:0; border-top:1px solid #E8E1D8; margin:9px 0;">
        <div style="font-size:11px; color:#74736D; line-height:1.4;">
            <strong style="color:#292D2A;">Active Adaptation Action:</strong><br>{act_s}
        </div>
    </div>

    <div class="legend">
        <strong>Vulnerability Zones</strong>
        <div class="legend-item"><span class="color-dot" style="background:#3388ff;"></span> Pavana River Flood Buffer (Ravet)</div>
        <div class="legend-item"><span class="color-line" style="background:#ff7f0e;"></span> NH-48 Emission / Traffic Hotspot</div>
        <div class="legend-item"><span class="color-dot" style="background:#28a745;"></span> Pradhikaran Green Canopy Cooling</div>
        <div class="legend-item"><span class="color-dot" style="background:{pin_color};"></span> Active Station ({selected_m}: {comp_s:.1f})</div>
    </div>

    <script>
        var map = L.map('map').setView([18.65, 73.76], 14);

        L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
            maxZoom: 18,
            attribution: 'Tiles &copy; Esri &mdash; PCMC Urban Climate Resilience'
        }}).addTo(map);

        // 1. Pavana River Riparian Flood Inundation Zone (Ravet boundary)
        var floodBuffer = L.polygon([
            [18.642, 73.738],
            [18.648, 73.745],
            [18.653, 73.753],
            [18.650, 73.762],
            [18.644, 73.758],
            [18.638, 73.745]
        ], {{
            color: '#0077be', fillColor: '#3388ff', fillOpacity: 0.32, weight: 2
        }}).addTo(map).bindPopup("<b>Pavana River Riparian Corridor</b><br>Low-lying riverfront with elevated monsoon flood and waterlogging susceptibility.");

        // 2. NH-48 Traffic & Particulate Emission Corridor
        var trafficCorridor = L.polyline([
            [18.665, 73.750],
            [18.658, 73.758],
            [18.650, 73.766],
            [18.640, 73.775]
        ], {{
            color: '#ff7f0e', weight: 7, opacity: 0.7
        }}).addTo(map).bindPopup("<b>Old Pune-Mumbai Highway (NH-48 Corridor)</b><br>Heavy vehicular and diesel particulate emission hotspot.");

        // 3. Nigdi Pradhikaran Urban Tree Canopy (Green Cooling Oasis)
        var greenZone = L.circle([18.656, 73.775], {{
            color: '#28a745', fillColor: '#28a745', fillOpacity: 0.22, radius: 900
        }}).addTo(map).bindPopup("<b>Nigdi Pradhikaran Green Oasis</b><br>Dense urban tree canopy buffering the urban heat island effect.");

        // 4. Central Monitoring Station Marker
        var stationMarker = L.circleMarker([18.65, 73.76], {{
            radius: 11, fillColor: '{pin_color}', color: '#ffffff', weight: 3, opacity: 1, fillOpacity: 0.95
        }}).addTo(map).bindPopup("<b>PCMC Nigdi Central Monitoring Hub</b><br>OpenAQ Sensor & Copernicus ERA5-Land Reanalysis Grid.<br><b>Evaluation Window ({selected_m}): {comp_s:.1f} / 100</b>").openPopup();
    </script>
</body>
</html>
"""

    current_cat = str(month_row.get("resilience_category", "Satisfactory"))
    map_code = generate_dynamic_map_html(
        selected_month, composite_score, aq_subscore, heat_subscore, flood_subscore, current_cat, intervention_text
    )

    # Also persist to MAP_FILE for reproducible standalone usage
    try:
        with open(MAP_FILE, "w", encoding="utf-8") as f:
            f.write(map_code)
    except Exception:
        pass

    components.html(map_code, height=540)

    st.markdown("""
    <div style="display:flex; justify-content:space-between; align-items:center; font-size:0.78rem; color:#74736D; padding: 6px 4px;">
        <span>Interactive GIS: Click markers and shaded zones to inspect localized hydrological and particulate buffer zones.</span>
        <span>Tile Layer: Esri World Street Map (Zero API Key Rate Limits)</span>
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# TAB 2: SEASONAL RISK OUTLOOK & 6-MONTH BASELINE FORECAST
# ------------------------------------------------------------------------------
with tab_forecast:
    st.markdown("""
    <div style="margin-bottom: 12px;">
        <h2 style="font-size: 1.85rem; margin-bottom: 4px;">Seasonal Risk Outlook & Future Risk Baseline</h2>
        <div style="font-size: 0.92rem; color: #74736D; line-height: 1.5;">
            Chronological multi-hazard progression across observed monthly milestones and the 6-month seasonal planning forecast.
            Preserves chronological order without forward data leakage.
        </div>
    </div>
    """, unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(12, 4.6), dpi=160)
    fig.patch.set_facecolor("#F7F3EE")
    ax.set_facecolor("#FFFEFC")

    x_labels = df["year_month"].tolist()
    x = list(range(len(x_labels)))

    # Historical trajectories
    ax.plot(x, df["composite_climate_resilience_score"], marker="o", markersize=4.5, color="#25483F", linewidth=2.7, label="Composite Resilience (Historical)")
    ax.plot(x, df["aq_resilience_subscore"], linestyle="--", color="#A96F67", alpha=0.9, linewidth=1.6, label="Air Quality Sub-Score (CPCB)")
    ax.plot(x, df["heat_resilience_subscore"], linestyle="--", color="#C28B38", alpha=0.9, linewidth=1.6, label="Heat Sub-Score (IMD)")
    if "flood_resilience_subscore" in df.columns:
        ax.plot(x, df["flood_resilience_subscore"], linestyle="--", color="#4D7A94", alpha=0.9, linewidth=1.6, label="Flood Sub-Score (GSW)")

    # Forecast trajectory
    if forecast_df is not None and not forecast_df.empty:
        forecast_labels = forecast_df["year_month"].tolist()
        forecast_x = list(range(len(df), len(df) + len(forecast_df)))
        
        # Bridge historical to forecast
        bridge_x = [x[-1], forecast_x[0]]
        bridge_y = [df["composite_climate_resilience_score"].iloc[-1], forecast_df["composite_climate_resilience_score"].iloc[0]]
        ax.plot(bridge_x, bridge_y, linestyle=":", color="#3D4A41", linewidth=1.8, alpha=0.6)

        ax.plot(forecast_x, forecast_df["composite_climate_resilience_score"], marker="s", markersize=4.8, markerfacecolor="#FFFEFC", markeredgewidth=1.8, linestyle=":", color="#3D4A41", linewidth=2.2, label="6-Month Baseline Forecast")
        all_x = x + forecast_x
        all_labels = x_labels + forecast_labels
    else:
        all_x = x
        all_labels = x_labels

    # Threshold benchmarks
    ax.axhline(75, color="#25483F", linestyle=":", linewidth=1.2, alpha=0.65, label="High Resilience Threshold (75)")
    ax.axhline(50, color="#A96F67", linestyle=":", linewidth=1.2, alpha=0.65, label="Moderate Resilience Threshold (50)")

    ax.set_xticks(all_x)
    ax.set_xticklabels(all_labels, rotation=45, ha="right", fontsize=8.5, color="#292D2A")
    ax.tick_params(colors="#74736D", labelsize=8.5)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Resilience Score (0–100)", color="#292D2A", fontsize=9.5, fontweight="600")

    # Spines and grid styling
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D9D0C7")
    ax.spines["bottom"].set_color("#D9D0C7")
    ax.grid(True, linestyle="--", alpha=0.45, color="#D9D0C7")

    legend = ax.legend(loc="lower left", fontsize=8.2, framealpha=0.95, facecolor="#FFFEFC", edgecolor="#D9D0C7", ncol=2)
    for text in legend.get_texts():
        text.set_color("#292D2A")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Compact forecast metric summary cards
    if forecast_df is not None and not forecast_df.empty:
        fc_mean = forecast_df["composite_climate_resilience_score"].mean()
        min_idx = forecast_df["composite_climate_resilience_score"].idxmin()
        max_idx = forecast_df["composite_climate_resilience_score"].idxmax()
        min_month = forecast_df.loc[min_idx, "year_month"]
        min_score = forecast_df.loc[min_idx, "composite_climate_resilience_score"]
        max_month = forecast_df.loc[max_idx, "year_month"]
        max_score = forecast_df.loc[max_idx, "composite_climate_resilience_score"]

        f1, f2, f3 = st.columns(3)
        with f1:
            st.markdown(f"""
            <div class="content-card" style="padding:14px 18px; margin-bottom:12px;">
                <div style="font-size:0.72rem; font-weight:700; color:#74736D; text-transform:uppercase;">Forecast 6-Mo Mean</div>
                <div style="font-family:'Cormorant Garamond',serif; font-size:1.8rem; font-weight:700; color:#25483F;">{fc_mean:.1f} / 100</div>
                <div style="font-size:0.76rem; color:#74736D;">Average municipal resilience outlook</div>
            </div>
            """, unsafe_allow_html=True)
        with f2:
            st.markdown(f"""
            <div class="content-card" style="padding:14px 18px; margin-bottom:12px;">
                <div style="font-size:0.72rem; font-weight:700; color:#A96F67; text-transform:uppercase;">Lowest Projected Month</div>
                <div style="font-family:'Cormorant Garamond',serif; font-size:1.8rem; font-weight:700; color:#A96F67;">{min_month} ({min_score:.1f})</div>
                <div style="font-size:0.76rem; color:#74736D;">Winter smog inversion & vehicular particulate peak</div>
            </div>
            """, unsafe_allow_html=True)
        with f3:
            st.markdown(f"""
            <div class="content-card" style="padding:14px 18px; margin-bottom:12px;">
                <div style="font-size:0.72rem; font-weight:700; color:#25483F; text-transform:uppercase;">Highest Projected Month</div>
                <div style="font-family:'Cormorant Garamond',serif; font-size:1.8rem; font-weight:700; color:#25483F;">{max_month} ({max_score:.1f})</div>
                <div style="font-size:0.76rem; color:#74736D;">Post-monsoon dry equilibrium</div>
            </div>
            """, unsafe_allow_html=True)

        st.caption("ℹ️ Scientific Note: The 6-month forecast utilizes a seasonal calendar-month baseline shifted chronologically. It serves municipal budget and resource allocation rather than long-term IPCC decadal projections.")

# ------------------------------------------------------------------------------
# TAB 3: RECOMMENDED ADAPTATION ACTIONS FOR NIGDI
# ------------------------------------------------------------------------------
with tab_actions:
    st.markdown("""
    <div style="margin-bottom: 14px;">
        <h2 style="font-size: 1.85rem; margin-bottom: 4px;">Recommended actions for Nigdi</h2>
        <div style="font-size: 0.92rem; color: #74736D; line-height: 1.5;">
            Actionable urban interventions synthesized from real-time environmental sub-scores and national hazard standards. 
            Prioritized by active threshold breaches for the selected period.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Active Monthly Priority Card
    st.markdown(f"""
    <div class="action-card {'terracotta-edge' if alert_level == 'alert' else ('amber-edge' if alert_level == 'watch' else '')}" style="background: #FFFDF9;">
        <div class="action-header">
            <span class="action-tag">ACTIVE EVALUATION DIRECTIVE · {selected_month}</span>
            <span class="advisory-priority {priority_badge_class}">{priority_label}</span>
        </div>
        <div class="action-title">{focus_area}</div>
        <div class="action-text"><strong>Immediate Directive:</strong> {intervention_text}</div>
        <div class="action-meta">
            <span><strong>Target Zone:</strong> PCMC Nigdi Corridor</span>
            <span><strong>Lead Authority:</strong> PCMC Disaster Management & Municipal Health</span>
            <span><strong>Trigger Metric:</strong> Current Sub-Score Analysis</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_act1, col_act2 = st.columns(2)

    with col_act1:
        # Air Quality Card
        aq_prio = "HIGH PRIORITY" if is_aq else "SATISFACTORY / MONITOR"
        aq_edge = "terracotta-edge" if is_aq else ""
        st.markdown(f"""
        <div class="action-card {aq_edge}">
            <div class="action-header">
                <span class="action-tag">AIR QUALITY · CPCB NAAQS (60 µg/m³)</span>
                <span style="font-size:0.72rem; font-weight:700; color:{'#A96F67' if is_aq else '#25483F'};">{aq_prio}</span>
            </div>
            <div class="action-title">Particulate & Vehicular Emission Control</div>
            <div class="action-text">
                <strong>Scientific Rationale:</strong> Winter thermal inversions and NH-48 heavy diesel transit elevate particulate exposure beyond the CPCB 24-hr threshold.<br>
                <strong>Proposed Action:</strong> Deploy nocturnal mechanical road vacuum sweepers and high-pressure water mist cannons along NH-48; enforce anti-idling and strict heavy-freight bypass routing during peak inversion hours.
            </div>
            <div class="action-meta">
                <span><strong>Corridor:</strong> Old Pune-Mumbai Highway (NH-48)</span>
                <span><strong>Dept:</strong> PCMC Environment & Traffic Branch</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Hydrological Flood Card
        fl_prio = "HIGH PRIORITY" if is_flood else "ROUTINE / PREPARED"
        fl_edge = "slate-edge" if is_flood else ""
        st.markdown(f"""
        <div class="action-card {fl_edge}">
            <div class="action-header">
                <span class="action-tag">FLOOD & WATERLOGGING · JRC 38-YR & BASALT</span>
                <span style="font-size:0.72rem; font-weight:700; color:{'#4D7A94' if is_flood else '#25483F'};">{fl_prio}</span>
            </div>
            <div class="action-title">Pavana Riparian Drainage & Dewatering Defense</div>
            <div class="action-text">
                <strong>Scientific Rationale:</strong> Cloudbursts exceeding IMD's 65 mm/day threshold rapidly saturate Deccan basalt soils (field capacity 0.42), precipitating surface runoff into low-lying Ravet riverbanks.<br>
                <strong>Proposed Action:</strong> Execute pre-monsoon culvert desilting along arterial roadways; position mobile diesel dewatering pumps at low-lying riverbank choke points; maintain riparian vegetative bioswales to retard surface runoff.
            </div>
            <div class="action-meta">
                <span><strong>Corridor:</strong> Ravet / Pavana Riverbank Settlements</span>
                <span><strong>Dept:</strong> PCMC Stormwater & Civil Engineering</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_act2:
        # Heatwave Card
        heat_prio = "HIGH PRIORITY" if is_heat else "SATISFACTORY / MONITOR"
        heat_edge = "amber-edge" if is_heat else ""
        st.markdown(f"""
        <div class="action-card {heat_edge}">
            <div class="action-header">
                <span class="action-tag">HEATWAVE · IMD / NDMA HEAT ACTION PLAN</span>
                <span style="font-size:0.72rem; font-weight:700; color:{'#C28B38' if is_heat else '#25483F'};">{heat_prio}</span>
            </div>
            <div class="action-title">Urban Heat Island & Thermal Stress Suppression</div>
            <div class="action-text">
                <strong>Scientific Rationale:</strong> Pre-monsoon solar radiation and asphalt thermal mass elevate daytime ambient temperatures above 40.0 °C, triggering severe heat-stroke vulnerability.<br>
                <strong>Proposed Action:</strong> Enforce strict prohibition of unshaded manual construction labor between 12:00 PM and 3:00 PM; activate public drinking water kiosks; equip municipal primary health centers with oral rehydration salts and cooling corners.
            </div>
            <div class="action-meta">
                <span><strong>Corridor:</strong> Commercial Centers & Industrial Belt</span>
                <span><strong>Dept:</strong> PCMC Public Health & Labor Welfare</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Cross-Hazard Ecological Card
        st.markdown("""
        <div class="action-card">
            <div class="action-header">
                <span class="action-tag">CROSS-HAZARD · ECOLOGICAL RESILIENCE</span>
                <span style="font-size:0.72rem; font-weight:700; color:#25483F;">STRATEGIC HORIZON</span>
            </div>
            <div class="action-title">Pradhikaran Urban Tree Canopy & Green Buffers</div>
            <div class="action-text">
                <strong>Scientific Rationale:</strong> Mature tree canopies reduce localized land surface temperature by 2–4 °C and intercept up to 15% of high-intensity monsoon precipitation.<br>
                <strong>Proposed Action:</strong> Preserve existing native tree density across Nigdi Pradhikaran; mandate cool-roof coatings and permeable interlocking pavers in all new civic parking infrastructures.
            </div>
            <div class="action-meta">
                <span><strong>Corridor:</strong> Nigdi Pradhikaran Residential Sector</span>
                <span><strong>Dept:</strong> PCMC Town Planning & Garden Dept</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# TAB 4: AI MODEL EXPLAINABILITY (XAI)
# ------------------------------------------------------------------------------
with tab_xai:
    st.markdown("""
    <div style="margin-bottom: 14px;">
        <h2 style="font-size: 1.85rem; margin-bottom: 4px;">Why the model predicts this · Atmospheric Physics Drivers</h2>
        <div style="font-size: 0.92rem; color: #74736D; line-height: 1.5;">
            Explainable AI (XAI) feature importances derived from trained Gradient Boosting Decision Tree Regressors.
            Demonstrates which atmospheric drivers most strongly govern next-hour forecasting.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.info("ℹ️ **Scientific Interpretation Note:** Tree-based feature importance quantifies relative predictive contribution during gradient boosting optimization; it demonstrates statistical association within the fitted environmental model rather than formal physical causation.")

    col_xai1, col_xai2 = st.columns(2)

    with col_xai1:
        st.markdown("""
        <div class="content-card">
            <h3 style="font-size: 1.35rem; margin-bottom: 4px;">Air Quality Model Drivers (PM2.5 GBDT)</h3>
            <p style="font-size: 0.82rem; color: #74736D; margin-bottom: 12px;">Target: Next-Hour PM2.5 (µg/m³) · OpenAQ Ground Observation Network</p>
        </div>
        """, unsafe_allow_html=True)
        aq_plot = PROJECT_ROOT / "feature_importance_air_quality.png"
        if aq_plot.exists():
            st.image(str(aq_plot), use_container_width=True)
            st.markdown("""
            <div style="font-size: 0.8rem; color: #74736D; line-height: 1.5; margin-top: 8px;">
                <strong>Key Physical Drivers:</strong>
                <ul>
                    <li><strong>Autoregressive Lag (PM2.5 t-1):</strong> Atmospheric persistence accounts for the majority of short-term variance.</li>
                    <li><strong>Cyclical Wind Direction (sin/cos):</strong> Determines whether emissions from industrial MIDC zones or clean rural breezes enter Nigdi.</li>
                    <li><strong>Boundary Layer Height & Dewpoint:</strong> Traps particulates near ground level during cold nocturnal inversions.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Air Quality feature importance plot not found.")

    with col_xai2:
        st.markdown("""
        <div class="content-card">
            <h3 style="font-size: 1.35rem; margin-bottom: 4px;">Heatwave Model Drivers (Temperature GBDT)</h3>
            <p style="font-size: 0.82rem; color: #74736D; margin-bottom: 12px;">Target: Next-Hour Ambient Temperature (°C) · Copernicus ERA5-Land Grid</p>
        </div>
        """, unsafe_allow_html=True)
        heat_plot = PROJECT_ROOT / "feature_importance_heatwave.png"
        if heat_plot.exists():
            st.image(str(heat_plot), use_container_width=True)
            st.markdown("""
            <div style="font-size: 0.8rem; color: #74736D; line-height: 1.5; margin-top: 8px;">
                <strong>Key Physical Drivers:</strong>
                <ul>
                    <li><strong>Autoregressive Lag (Temp t-1):</strong> Primary thermodynamic thermal inertia of the urban built fabric.</li>
                    <li><strong>Cyclical Hour Transforms (sin/cos of 24h):</strong> Captures the solar diurnal insolation cycle without artificial discontinuities.</li>
                    <li><strong>Solar Radiation Flux & Humidity:</strong> Governs evaporative cooling rates and peak sensible heat accumulation.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Heatwave feature importance plot not found.")

    with st.expander("🔍 Scientific Architecture Deep-Dive (Feature Engineering & Hydrological Mechanics)"):
        st.markdown("""
        ### Multi-Hazard Data Engineering & Methodological Validation
        Examiners and civic planners can inspect the underlying data engineering mechanisms:
        """)
        
        rep_img_dir = PROJECT_ROOT / "Report Images"
        flowchart = rep_img_dir / "Multimodal Urban Resilience Framework Flowchart.png"
        lag_matrix = rep_img_dir / "Autoregressive Environmental Lag Matrix.png"
        harmonic_img = rep_img_dir / "Cyclical Harmonic Transforms Infographic.png"
        flood_mech = rep_img_dir / "Tri-Part Urban Flood Inundation Mechanism.png"

        ed1, ed2 = st.columns(2)
        with ed1:
            if lag_matrix.exists():
                st.image(str(lag_matrix), caption="Figure 1: Autoregressive Environmental Lag Matrix (shift(1) preventing leakage)", use_container_width=True)
            if harmonic_img.exists():
                st.image(str(harmonic_img), caption="Figure 2: Cyclical Harmonic Trigonometric Transforms (24h & 360° wind angles)", use_container_width=True)
        with ed2:
            if flood_mech.exists():
                st.image(str(flood_mech), caption="Figure 3: Tri-Part Urban Flood Inundation Mechanism (JRC + Basalt + IMD)", use_container_width=True)
            if flowchart.exists():
                st.image(str(flowchart), caption="Figure 4: Multimodal Resilience Framework Pipeline Flowchart", use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 5: PLANNER DATA TABLES & EXPORTS
# ------------------------------------------------------------------------------
with tab_data:
    st.markdown("""
    <div style="margin-bottom: 14px;">
        <h2 style="font-size: 1.85rem; margin-bottom: 4px;">Planner Data & Forecast Records</h2>
        <div style="font-size: 0.92rem; color: #74736D; line-height: 1.5;">
            Exportable tabular records supporting municipal administrative workflows, academic verification, and departmental reporting.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if forecast_df is not None and not forecast_df.empty:
        st.markdown("### 6-Month Seasonal Risk Baseline Forecast")
        st.caption("Forecast values formatted to 1 decimal place. Generated via chronological seasonal baselines.")
        
        display_forecast = forecast_df.copy()
        # Round numerical columns
        num_cols = ["composite_climate_resilience_score", "aq_resilience_subscore", "heat_resilience_subscore", "flood_resilience_subscore"]
        for c in num_cols:
            if c in display_forecast.columns:
                display_forecast[c] = display_forecast[c].round(1)

        st.dataframe(
            display_forecast[[
                "year_month", "composite_climate_resilience_score",
                "aq_resilience_subscore", "heat_resilience_subscore",
                "flood_resilience_subscore", "resilience_category",
                "recommended_urban_intervention",
            ]],
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            label="📥 Download 6-Month Forecast (CSV)",
            data=display_forecast.to_csv(index=False).encode('utf-8'),
            file_name="Nigdi_Future_Risk_Baseline_Forecast.csv",
            mime="text/csv",
        )
    else:
        st.info("Future-risk forecast file not found.")

    st.markdown("---")

    st.markdown("### Historical Tri-Hazard Master Scoreboard (PCMC Nigdi)")
    st.caption("Observed 20-month continuous record from February 2025 to September 2026.")

    display_hist = df.copy()
    hist_num_cols = [
        "composite_climate_resilience_score", "aq_resilience_subscore", "heat_resilience_subscore", "flood_resilience_subscore",
        "pm25_mean", "temp_c_max", "precip_max_daily_mm", "soil_saturation_pct"
    ]
    for c in hist_num_cols:
        if c in display_hist.columns:
            display_hist[c] = display_hist[c].round(1)

    display_columns = [
        "year_month", "composite_climate_resilience_score",
        "aq_resilience_subscore", "heat_resilience_subscore", "flood_resilience_subscore",
        "pm25_mean", "temp_c_max", "precip_max_daily_mm", "soil_saturation_pct",
        "resilience_category"
    ]
    existing_disp_cols = [c for c in display_columns if c in display_hist.columns]

    st.dataframe(
        display_hist[existing_disp_cols],
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        label="📥 Download Full Historical Scoreboard (CSV)",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name="Nigdi_TriHazard_Monthly_Master_Scoreboard.csv",
        mime="text/csv",
    )

# ==============================================================================
# 7. CIVIC FOOTER
# ==============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #74736D; font-size: 0.78rem; padding: 10px 0 20px 0; line-height: 1.6;">
    <strong>PCMC Nigdi Urban Climate Resilience AI Platform</strong> · Pimpri Chinchwad College of Engineering (PCCoE)<br>
    Target Pilot Zone: Nigdi Corridor (18.65°N, 73.76°E) · Evaluated against CPCB NAAQS, IMD Heat Action Plan & JRC GSW Hydrology Standards
</div>
""", unsafe_allow_html=True)
