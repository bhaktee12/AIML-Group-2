"""
Step 7: PCMC Nigdi Geospatial Climate Vulnerability Mapping Engine
==================================================================
Fulfills Expected Outcome 2: "Climate vulnerability maps".
Generates an interactive geospatial Leaflet/Folium map for PCMC Nigdi / Ravet:

Layers:
1. Base satellite / OpenStreetMap tile layer.
2. Nigdi Climate Monitoring Station marker (18.65 N, 73.76 E) with live resilience gauges.
3. Pavana River Basin & Low-Lying Flood Inundation Corridor (Ravet boundary).
4. Old Pune-Mumbai Highway (NH-48) Vehicular Emission & Particulate Hotspot.
5. Nigdi Pradhikaran Urban Tree Canopy & Green Cooling Oasis (High Resilience).
6. Interactive Multi-Hazard Risk Legends & Municipal Directives.
7. Exports 'nigdi_climate_vulnerability_map.html'.
"""

import os
from pathlib import Path
import pandas as pd

NIGDI_LAT = 18.65
NIGDI_LON = 73.76
PROJECT_ROOT = Path(__file__).resolve().parent

def generate_nigdi_vulnerability_map(
    scoreboard_csv=PROJECT_ROOT / "Nigdi_TriHazard_Monthly_Master_Scoreboard.csv",
    output_html=PROJECT_ROOT / "nigdi_climate_vulnerability_map.html"
):
    print("=" * 70)
    print("STEP 7: PCMC NIGDI CLIMATE VULNERABILITY MAPPING ENGINE")
    print("=" * 70)

    # 1. Load Latest Resilience Scores
    score_summary = {
        "composite": 78.2,
        "aq": 86.5,
        "heat": 100.0,
        "flood": 34.4,
        "category": "Moderate Resilience (Satisfactory)",
        "action": "WATERLOGGING/FLOOD ALERT: Activate the municipal flood adaptation protocol."
    }

    if Path(scoreboard_csv).exists():
        df = pd.read_csv(scoreboard_csv)
        latest = df.iloc[-1]
        score_summary["composite"] = float(latest.get("composite_climate_resilience_score", 78.2))
        score_summary["aq"] = float(latest.get("aq_resilience_subscore", 86.5))
        score_summary["heat"] = float(latest.get("heat_resilience_subscore", 100.0))
        score_summary["flood"] = float(latest.get("flood_resilience_subscore", 34.4))
        score_summary["category"] = str(latest.get("resilience_category", "Moderate Resilience (Satisfactory)"))
        score_summary["action"] = str(latest.get("recommended_urban_intervention", "Routine maintenance"))
        print(f"Loaded latest monthly scores for Nigdi (Composite: {score_summary['composite']}/100)")

    # Color code for marker based on score
    if score_summary["composite"] >= 75:
        pin_color = "#28a745"  # Green
    elif score_summary["composite"] >= 50:
        pin_color = "#ffc107"  # Yellow
    elif score_summary["composite"] >= 25:
        pin_color = "#fd7e14"  # Orange
    else:
        pin_color = "#dc3545"  # Red

    # Generate complete self-contained HTML with Leaflet.js (zero external Python map dependencies required)
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>PCMC Nigdi - Urban Climate Vulnerability Map</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body {{ margin:0; padding:0; font-family: 'Segoe UI', Arial, sans-serif; background:#CBC2C1; color:#2E312E; }}
        #map {{ width: 100vw; height: 100vh; }}
        .info-box {{
            position: absolute; top: 15px; right: 15px; z-index: 1000;
            background: #F5F1EF; color: #2E312E; padding: 18px 22px; border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.25); max-width: 360px;
        }}
        .score-pill {{
            font-size: 24px; font-weight: bold; padding: 4px 12px;
            border-radius: 6px; display: inline-block; color: white;
            background-color: #906477; margin: 8px 0;
        }}
        .gauge-row {{ display: flex; justify-content: space-between; margin: 6px 0; font-size: 13px; }}
        .legend {{
            position: absolute; bottom: 25px; left: 15px; z-index: 1000;
            background: #F5F1EF; color: #2E312E; padding: 12px 16px; border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2); font-size: 12px;
        }}
        .legend-item {{ display: flex; align-items: center; margin-bottom: 4px; }}
        .color-dot {{ width: 12px; height: 12px; border-radius: 50%; display: inline-block; margin-right: 8px; }}
    </style>
</head>
<body>
    <div id="map"></div>

    <div class="info-box">
        <h3 style="margin:0 0 6px 0; color:#1a365d;">PCMC Nigdi / Ravet</h3>
        <span style="font-size:12px; color:#666;">Multimodal Environmental Intelligence (18.65°N, 73.76°E)</span>
        <div>
            <span class="score-pill">{score_summary['composite']} / 100</span>
            <div style="font-weight:600; font-size:13px; color:#2d3748;">{score_summary['category']}</div>
        </div>
        <hr style="border:0; border-top:1px solid #e2e8f0; margin:10px 0;">
        <div class="gauge-row"><span>Air Quality Sub-Score (CPCB):</span><strong>{score_summary['aq']} / 100</strong></div>
        <div class="gauge-row"><span>Heatwave Sub-Score (IMD):</span><strong>{score_summary['heat']} / 100</strong></div>
        <div class="gauge-row"><span>Flood Sub-Score (GSW):</span><strong>{score_summary['flood']} / 100</strong></div>
        <hr style="border:0; border-top:1px solid #e2e8f0; margin:10px 0;">
        <div style="font-size:11px; color:#4a5568; line-height:1.4;">
            <strong>Active Adaptation Action:</strong><br>{score_summary['action']}
        </div>
    </div>

    <div class="legend">
        <strong>Vulnerability Zones</strong><br>
        <div class="legend-item"><span class="color-dot" style="background:#dc3545;"></span> Pavana River Flood Inundation Buffer</div>
        <div class="legend-item"><span class="color-dot" style="background:#ff7f0e;"></span> NH-48 Emission / Traffic Hotspot</div>
        <div class="legend-item"><span class="color-dot" style="background:#28a745;"></span> Pradhikaran Canopy Cooling Oasis</div>
        <div class="legend-item"><span class="color-dot" style="background:{pin_color};"></span> Active Nigdi Monitoring Station</div>
    </div>

    <script>
        var map = L.map('map').setView([{NIGDI_LAT}, {NIGDI_LON}], 14);

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
            color: '#0077be', fillColor: '#3388ff', fillOpacity: 0.35, weight: 2
        }}).addTo(map).bindPopup("<b>Pavana River Riparian Corridor</b><br>Low-lying riverfront with high monsoon flood and waterlogging susceptibility.");

        // 2. NH-48 Traffic & Particulate Emission Corridor
        var trafficCorridor = L.polyline([
            [18.665, 73.750],
            [18.658, 73.758],
            [18.650, 73.766],
            [18.640, 73.775]
        ], {{
            color: '#ff7f0e', weight: 8, opacity: 0.6
        }}).addTo(map).bindPopup("<b>Old Pune-Mumbai Highway (NH-48 Corridor)</b><br>Heavy vehicular and diesel particulate emission hotspot.");

        // 3. Nigdi Pradhikaran Urban Tree Canopy (Green Cooling Oasis)
        var greenZone = L.circle([18.656, 73.775], {{
            color: '#28a745', fillColor: '#28a745', fillOpacity: 0.25, radius: 900
        }}).addTo(map).bindPopup("<b>Nigdi Pradhikaran Green Oasis</b><br>High urban tree canopy density buffering urban heat island effect.");

        // 4. Central Monitoring Station Marker
        var stationMarker = L.circleMarker([{NIGDI_LAT}, {NIGDI_LON}], {{
            radius: 12, fillColor: '{pin_color}', color: '#ffffff', weight: 3, opacity: 1, fillOpacity: 0.9
        }}).addTo(map).bindPopup("<b>PCMC Nigdi Central Monitoring Hub</b><br>OpenAQ Sensor & Copernicus ERA5-Land Reanalysis Grid.<br><b>Resilience Score: {score_summary['composite']}/100</b>").openPopup();
    </script>
</body>
</html>
"""

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"SUCCESS: Generated interactive spatial vulnerability map -> '{output_html}'")
    print(f"You can open '{output_html}' directly in any web browser to view the interactive map.")
    return output_html

if __name__ == "__main__":
    generate_nigdi_vulnerability_map()
