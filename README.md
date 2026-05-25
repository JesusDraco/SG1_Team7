# Green Grid Simulator — SG1 Team 7

A neighborhood energy simulation project that analyzes how solar energy, batteries, household consumption, and the electrical grid interact over time.

The simulator compares different energy management strategies and shows how each one affects grid dependence, solar self-consumption, battery usage, energy costs, and overall neighborhood performance.

---

## eam Members

- Diego Sebastián Montoya Rodríguez
- Jesús Abel Gutiérrez Calvillo
- José Bernardo Sandoval Martínez

---

## Main Objective

Compare different energy strategies in a neighborhood equipped with solar panels and batteries, answering questions such as:

- How much energy does the neighborhood generate and consume?
- How much energy is imported from / exported to the grid?
- Which strategy reduces grid dependence the most?
- Which strategy uses solar energy more efficiently?
- How does the battery help balance energy usage?
- How does machine learning improve the solar simulation?

---

## Project Structure

```
SG1_TEAM7/
│
├── ML/
│   └── Simulator/
│       ├── dashboard/
│       │   ├── index.html
│       │   ├── main.js
│       │   └── style.css
│       │
│       ├── data/
│       │   ├── 222628_Actual_DPV_13MW_5m.csv
│       │   ├── 222628_Weather_30m.csv
│       │   ├── 222628_DA_DPV_13MW_60m.csv
│       │   └── 222628_HA4_DPV_13MW_60m.csv
│       │
│       ├── output/
│       │   └── dashboard_data/
│       │
│       ├── config.json
│       ├── greengridsim.py
│       └── ml_solar_model.py
│
└── README.md
```

> **Important:** The main version of the project is located in `ML/Simulator`. Other simulator or dashboard folders are older versions.

---

## Main Files

### `greengridsim.py`
The main simulator. Runs the energy simulation and calculates results for the neighborhood, including:
- Solar generation
- Household consumption
- Battery charge and discharge
- Grid import / export
- Energy costs
- Different seasons and energy strategies

Exports results as CSV and JSON files for the dashboard.

### `ml_solar_model.py`
Contains the machine learning model that predicts solar generation using real weather and solar data. Built with a linear regression model from scratch, using features such as:
- Temperature, humidity, and solar irradiance
- Cloud type and wind speed
- Solar angle, hour of the day, and month

### `dashboard/index.html`
Defines the structure of the dashboard with sections for:
- Executive summary and KPIs
- Duck curve
- Neighborhood analysis
- Machine learning analysis
- Strategy comparison

### `dashboard/main.js`
Loads simulation results and renders charts using **D3.js**. Controls filters, strategy/season/time-scale selection, tabs, duck curve animation, KPI updates, and ML charts.

### `dashboard/style.css`
Visual design of the dashboard: layout, colors, cards, charts, buttons, tabs, and responsive design.

---

## Requirements

Install the required Python libraries:

```bash
pip install simpy pandas matplotlib numpy
```

Or, if using Python 3 explicitly:

```bash
pip3 install simpy pandas matplotlib numpy
```

---

## How to Run

### 1. Navigate to the correct folder

```bash
cd ML/Simulator
```

### 2. Run the simulator

```bash
python greengridsim.py
```

This generates all output files needed by the dashboard.

### 3. Start a local server

```bash
python -m http.server 8000
```

### 4. Open the dashboard

Open your browser and go to:

```
http://localhost:8000/dashboard/index.html
```

> Do **not** open `index.html` by double-clicking it. The dashboard needs a local server to load JSON files from the output folder.

### Quick Reference — Full Run

```bash
cd ML/Simulator
pip install simpy pandas matplotlib numpy
python greengridsim.py
python -m http.server 8000
# Then open: http://localhost:8000/dashboard/index.html
```

---

## Energy Strategies

The simulator compares three strategies:

### `LOAD_PRIORITY`
Solar energy covers household demand first, then charges the battery, then exports any surplus to the grid.
> *Use solar for the house first → battery → export.*

### `CHARGE_PRIORITY`
Solar energy charges the battery first, then covers household demand, then exports any surplus.
> *Charge the battery first → cover demand → export.*

### `PRODUCE_PRIORITY`
Solar energy is exported first. Household demand is then covered using the battery or the grid.
> *Export solar first → cover demand from battery/grid.*

---

## Dashboard Features

| Section | Description |
|---|---|
| KPI Cards | Total generation, consumption, grid import/export, self-consumption ratio, self-sufficiency ratio |
| Duck Curve | Animated visualization of the energy demand curve |
| Household Comparison | Energy behavior by household type |
| Wealth Level Comparison | Energy behavior by wealth level |
| Battery Utilization | How the battery is charged and discharged |
| Energy Surplus & Deficit | When the neighborhood produces more or less than it consumes |
| Cost & Self-Consumption | Cost analysis and solar usage efficiency |
| ML Analysis | Model performance, actual vs predicted generation, forecast comparison |
| Executive Summary | High-level overview of results |

---

## Machine Learning

The ML model was added to make solar generation more realistic. A standard ideal solar curve assumes perfect sunlight every day; in reality, clouds, temperature, and irradiance all affect production.

The model uses real data (irradiance, cloud type, wind speed, solar angle, etc.) to produce a more accurate simulation, leading to better and more meaningful dashboard results.

---

## Output Files

After running the simulator, files are generated inside `output/dashboard_data/`. Examples:

```
kpis_Spring_LOAD_PRIORITY.json
duck_curve_Spring_LOAD_PRIORITY.json
by_household_type_Spring_LOAD_PRIORITY.json
by_wealth_level_Spring_LOAD_PRIORITY.json
costs_self_consumption_Spring_LOAD_PRIORITY.json
ml_model_metrics.json
ml_prediction_sample.json
forecast_comparison.json
```

---

## Common Errors

**`FileNotFoundError: config.json`**
The simulator was run from the wrong folder. Fix:
```bash
cd ML/Simulator
python greengridsim.py
```

**Dashboard shows missing data**
Run the simulator again, then refresh the dashboard:
```bash
python greengridsim.py
```

**Charts do not load**
Use a local server instead of opening the HTML file directly:
```bash
python -m http.server 8000
# Then open: http://localhost:8000/dashboard/index.html
```

---

## Summary

Green Grid Simulator combines **Python simulation**, **machine learning**, and **D3.js visualization** to study how a neighborhood uses solar energy. It models households, solar panels, batteries, grid import/export, and three distinct energy strategies. The ML model predicts solar generation from real weather data, making the simulation more accurate, and the interactive dashboard lets users compare strategies and understand the neighborhood's energy behavior.
