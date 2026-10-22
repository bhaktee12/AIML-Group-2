# Frontend Redesign Brief --- Urban Climate Resilience AI Dashboard

## Project

Redesign the existing Streamlit application for the **AI-Based Urban
Climate Resilience Decision-Support System --- PCMC Nigdi, Pune**.

The current dashboard already contains working elements such as
resilience score cards, a geospatial vulnerability map,
risk-history/forecast charts, feature-importance charts, and municipal
recommendations. Improve the visual design and information hierarchy
without breaking the existing data pipeline, model outputs, filters, or
interactions.

## Design direction: editorial, warm, modern, climate-focused

Use the supplied **FLEUR dried-flower website screenshot as a visual
mood reference only**. Translate its refined editorial layout, warm
neutral palette, generous whitespace, elegant typography, rounded/arched
image treatments, and restrained borders into a credible environmental
analytics dashboard. Do **not** copy the flower shop branding, product
content, or ecommerce layout.

The result should feel like a polished civic-tech product: calm and
premium, but still data-dense, legible, scientific, and useful to urban
planners.

### Theme tokens

Use a consistent palette throughout the app:

-   **Warm ivory background:** `#F7F3EE`
-   **Soft sand surface:** `#EEE3D8`
-   **Pale clay accent:** `#DCC2AD`
-   **Muted terracotta accent:** `#A96F67`
-   **Deep forest green:** `#25483F`
-   **Sage green:** `#8DA394`
-   **Charcoal text:** `#292D2A`
-   **Muted secondary text:** `#74736D`
-   **White cards:** `#FFFEFC`
-   **Subtle border:** `#D9D0C7`

Use forest green for primary actions and healthy/resilient states,
terracotta for attention and adaptation actions, and amber/red only for
meaningful risk severity. Do not use the same accent colour for every
component. Ensure accessible contrast; never communicate risk using
colour alone.

### Typography

-   Use an elegant serif for the main page title and selected editorial
    headings, such as **DM Serif Display**, **Cormorant Garamond**, or a
    safe serif fallback.
-   Use a clean sans-serif for navigation, metrics, labels, tables,
    controls, and chart text, such as **Inter**, **DM Sans**, or a safe
    sans-serif fallback.
-   Establish clear hierarchy: one strong page title, compact section
    headings, readable body text, and small muted metadata.
-   Avoid oversized text that pushes key dashboard information below the
    fold.

## Application structure and user experience

### 1. Sidebar navigation

Create a refined, compact sidebar with:

-   Product identity: **Urban Resilience**
-   Location chip: **PCMC Nigdi · Pune**
-   Evaluation-month selector, preserving its existing functionality
-   Navigation links/sections for:
    1.  Overview
    2.  Vulnerability Map
    3.  Risk Forecast
    4.  Adaptation Actions
    5.  Model Explainability
-   A small footer label such as `CLIMATE INTELLIGENCE · PCMC PILOT`

Use a deep forest-green or charcoal sidebar with warm ivory text, subtle
dividers, and a restrained active-state indicator. Avoid oversized
icons, excessive decoration, or a large empty sidebar.

If the app currently uses Streamlit tabs rather than multipage
navigation, keep the existing interaction model unless changing it is
straightforward and safe. Do not introduce navigation that breaks
existing functionality.

### 2. Overview / decision-support home

Create a strong opening section with:

-   Eyebrow: `URBAN CLIMATE INTELLIGENCE · PCMC NIGDI`
-   Main title: **Understanding Nigdi's Climate Resilience**
-   Short supporting copy explaining that the dashboard combines
    air-quality, heat, and flood indicators to support local planning.
-   A compact reporting-period/location line and a visible data-period
    or update note when that information is available.
-   A prominent **current municipal advisory** panel, driven by the
    existing recommendation logic and selected evaluation month.

Below the introduction, display four aligned KPI cards:

1.  **Composite Resilience Score** --- score out of 100
2.  **Air Quality** --- sub-score out of 100
3.  **Heatwave Resilience** --- sub-score out of 100
4.  **Flood Resilience** --- sub-score out of 100

Each card should contain a concise label, large readable value, small
descriptor/status, and a subtle visual cue such as a thin progress bar
or small indicator. Keep the existing values and score calculations; do
not hardcode the example numbers from screenshots.

Show a clear resilience category such as **High**, **Moderate**, or
**Low**, with text plus a restrained status colour. Include small help
text or tooltips that explain what each score means and how to interpret
it. Do not imply that a high resilience score guarantees safety.

### 3. Climate vulnerability map

Make the geospatial map a major, easy-to-find section rather than a
buried technical element.

-   Heading: **Where climate risks concentrate**
-   Supporting text: explain that the map presents the Nigdi monitoring
    area and available hazard layers.
-   Use a wide map container with a clean border, rounded corners, and
    enough height to inspect locations.
-   Keep map controls visible and usable.
-   Retain existing layers/markers for the monitoring hub, flood or
    waterlogging-prone areas, traffic/emissions corridor, and
    cooling/green-cover locations when supported by the underlying data.
-   Include a clear, compact legend with labels and colour swatches.
-   Keep the current score and active adaptation action in a readable
    map-side information panel or popup.
-   Make the map responsive. On narrow screens, stack the information
    panel rather than covering most of the map.
-   Use the existing map data and coordinates. Do not invent additional
    sensor stations, boundaries, or measured risk values.

### 4. Risk history and future-risk forecast

Use a clean, wide chart card titled **Seasonal Risk Outlook** or **Risk
history & forecast**.

-   Preserve the existing time series and forecast logic.
-   Clearly distinguish historical observations from forecasts using
    line style and legend labels.
-   Show the resilience thresholds with unobtrusive reference lines and
    explain them in the legend.
-   Keep axis labels, dates, units, tooltips, and legends readable.
-   Use a restrained colour system: charcoal/forest for composite
    resilience, terracotta for air quality, sage for heat, and a
    distinct muted neutral or blue-grey for flood risk.
-   Clearly label forecast periods and include the existing caveat that
    a seasonal/latest-calendar-month baseline is a planning aid, not a
    long-term climate-change projection.
-   Never present generated baseline values as validated climate-change
    predictions.
-   Ensure the chart container does not crop labels or overflow
    horizontally.

Place a compact forecast summary near the chart if the current data
supports it. Summaries must be calculated from actual displayed data,
not written as fixed example values.

### 5. AI-based adaptation recommendations

Create an actionable section titled **Recommended actions for Nigdi**.

Each recommendation card should show:

-   Hazard or focus area (air quality, heat, flood/waterlogging, or
    cross-hazard)
-   Recommendation title
-   Short, plain-language rationale grounded in the displayed score or
    model output
-   Proposed action
-   Priority level, if the existing logic provides one
-   Relevant time horizon or responsible department only if supported by
    the current application data

Examples of action categories may include improving drainage and
waterlogging response, increasing shade and green cover, or reducing
particulate exposure. These are examples of possible recommendations,
not claims that the system has verified a specific site intervention.

Keep the currently selected month connected to the active advisory. Make
the primary action easy to scan, but avoid alarmist language. Do not
invent budgets, deadlines, responsible departments, or measured
intervention impacts.

### 6. AI model explainability (XAI)

Create a dedicated **Why the model predicts this** section.

-   Retain the existing feature-importance charts for the
    PM2.5/air-quality and temperature/heat models.
-   Place each chart in a separate card with a descriptive heading,
    brief explanation, and readable axis labels.
-   Explain that feature importance indicates how strongly a feature
    contributes to model predictions in the fitted model; it does not
    prove causation.
-   Keep percentages and labels legible and avoid excessive chart
    decoration.
-   If a model or explanation is unavailable, show a graceful,
    informative empty state rather than a broken chart.

### 7. Planner-facing data table

Keep the forecast table, but improve its presentation:

-   Use a clear section heading and one-sentence explanation.
-   Format score columns consistently to one decimal place where
    appropriate.
-   Keep month labels readable and preserve all meaningful columns.
-   Use subtle row dividers and restrained highlighting for the selected
    month or important risk categories.
-   Avoid forcing the entire page into a horizontal scroll on normal
    desktop widths; if the table is wide, make the table container
    intentionally scrollable.
-   Preserve any existing download/export functionality.

## Layout and component styling

-   Use a centered main content area with a sensible maximum width and
    consistent page gutters.
-   Use a responsive grid: four KPI cards on wide screens, two per row
    on medium screens, and one per row on narrow screens.
-   Use consistent card padding, corner radius (approximately 12--18
    px), and subtle borders or very soft shadows.
-   Prefer whitespace and alignment over heavy boxes and thick outlines.
-   Use subtle section dividers and a consistent spacing scale.
-   Keep the main title and first KPI row visible without an
    unnecessarily long intro.
-   Avoid a full-screen beige wash that makes all sections blend
    together. Use ivory for the page, white for primary cards, sand for
    occasional highlighted areas, and forest green for key
    navigation/actions.
-   Avoid excessive gradients, neon colours, emojis, decorative stock
    imagery, and generic SaaS dashboard clutter.
-   Do not use flower imagery in the climate dashboard. Use maps, data
    visualizations, and small climate-related line icons only where
    useful.

## Streamlit implementation requirements

1.  Inspect the existing `app_urban_resilience_dashboard.py` and the
    project's current data/model files before editing.
2.  Keep the current calculations, model loading, file paths, filters,
    selected-month behaviour, map data, charts, and recommendations
    intact unless a bug fix is required.
3.  Prefer reusable helper functions for repeated UI patterns, such as
    KPI cards, section headers, status labels, and recommendation cards.
4.  Apply theme CSS carefully through Streamlit-supported patterns.
    Scope CSS selectors to the app where practical and avoid brittle
    selectors tied to generated class names.
5.  Use Streamlit-native components where possible. If HTML is used for
    styling, escape or safely render dynamic text and avoid injecting
    untrusted data.
6.  Do not replace real data with mock values, static screenshots, or
    hardcoded sample charts.
7.  Do not remove existing features just to simplify the layout.
    Preserve useful tabs, interactions, map popups, and data tables.
8.  Handle missing or empty datasets gracefully with helpful messages.
    Avoid silently treating missing values as zero.
9.  Ensure all charts and maps resize correctly and do not overlap
    adjacent components.
10. Keep the application runnable with the existing launch command:
    `python -m streamlit run app_urban_resilience_dashboard.py`
11. If adding a dependency, document it and make sure it is genuinely
    necessary. Prefer not to add dependencies solely for decorative
    styling.
12. After implementation, run the app or perform available checks and
    fix syntax errors, missing imports, broken paths, and layout
    regressions.

## Data integrity and scientific communication

-   Study area: **Nigdi, Pimpri-Chinchwad, Pune**.
-   The platform should communicate five core capabilities:
    1.  Urban Climate Resilience Score for Nigdi
    2.  Climate vulnerability maps
    3.  AI-based adaptation recommendations
    4.  Decision-support dashboard for urban planners
    5.  Prediction/forecasting of future climate risks
-   Preserve the distinction between observed measurements, model
    predictions, and baseline forecasts.
-   Make time ranges and data sources visible where known. Do not imply
    that a station has data for periods it does not cover.
-   Never describe feature importance as causal proof.
-   Do not present the pilot's local scores as official citywide
    measurements.
-   Use clear units for environmental measurements wherever available.
-   Do not invent accuracy metrics, confidence intervals, data coverage,
    or intervention outcomes.

## Suggested page order

1.  Sidebar with location and month selector
2.  Hero title and context
3.  Current municipal advisory
4.  Four resilience KPI cards
5.  Vulnerability map
6.  Risk history and forecast chart
7.  Recommended adaptation actions
8.  Model explainability charts
9.  Forecast/data table and methodology notes

Adjust the order if the current navigation structure makes another
arrangement more usable, but all five core capabilities must remain easy
to discover.

## Acceptance checklist

-   [ ] The dashboard has a cohesive warm ivory, sand, forest-green, and
    muted-terracotta visual identity.
-   [ ] The design feels editorial and polished without looking like a
    flower shop or ecommerce template.
-   [ ] The selected evaluation month still updates the appropriate data
    and recommendations.
-   [ ] All four KPI values are connected to the existing
    data/calculations.
-   [ ] The vulnerability map remains interactive and its legend is
    readable.
-   [ ] Historical data and forecast values are clearly differentiated.
-   [ ] Adaptation recommendations are driven by existing application
    logic.
-   [ ] Feature-importance charts remain visible and scientifically
    explained.
-   [ ] Tables, charts, and cards are responsive and readable.
-   [ ] Existing model/data paths, exports, and interactions continue to
    work.
-   [ ] No fabricated values, locations, or claims have been introduced.
-   [ ] The app launches successfully with the documented Streamlit
    command.

## Final instruction to the coding agent

First inspect the existing code and identify the current page structure
and data dependencies. Then implement the redesign in the existing
application, preserving its working functionality. Make the smallest
safe changes needed to achieve a cohesive, responsive, polished climate
decision-support dashboard. At the end, summarize the files changed, the
visual improvements made, any dependency changes, and the checks
performed.
