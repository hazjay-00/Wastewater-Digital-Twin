# Smart Wastewater Facility – Process Digital Twin

## Overview
An interactive Streamlit dashboard and Machine Learning digital twin built to predict Chemical Oxygen Demand (COD) levels in wastewater treatment plants using the UCI Machine Learning dataset.

---

## Key Features

* **Interactive Control Knobs:** Real-time sliders allowing operators to simulate plant inflow conditions (flow rate, pH, COD, conductivity, heavy metals, etc.).
* **Predictive ML Simulation:** Forecasts effluent water quality and alerts users to environmental compliance breaches or toxic shock loads.
* **Explainable AI:** Visual feature importance charts detailing which incoming variables drive predictions.
* **Multi-Stage Process Kinetics:** Visualizes water quality across raw inflow, primary settling (transient), and final effluent (steady-state).
* **Data Export:** Capability to download current simulation scenarios into `.CSV` logs for laboratory record-keeping.

---

## Iterative Evolution and Version Progression

* **Version 1 (v1):** Minimal proof-of-concept with 4 basic inputs and a single Random Forest regressor.
* **Version 2 (v2):** Added model evaluation ($R^2$, RMSE), train/test splitting, feature importance charts, and CSV data export.
* **Version 3 (v3):** Recalibrated the model against full baseline historical data for steady-state process compliance.
* **Version 4 (v4):** Introduced a Dual-Pipeline Architecture to simulate multi-stage processing (transient primary settling vs. steady-state final effluent).
* **Version 5 (v5):** Expanded to an 8-dimensional input matrix (including Zinc, BOD, Suspended Solids) and applied hyperparameter tuning (`max_depth`, `min_samples_split`).

---

**Note on Evaluation Methodology:** 
* **Steady-State Models (v3, v4, v5):** Evaluates in-sample fitting against baseline historical operational data to represent continuous equilibrium.
* **Transient Models (v2, v4, v5):** Evaluates out-of-sample generalization via train/test splitting to measure generalizability under dynamic load variations.

---

## Tech Stack & Data Source

### Tech Stack
* **UI / Dashboard:** Streamlit
* **Data Processing & ML:** Pandas, NumPy, Scikit-Learn
* **Data Visualization:** Plotly Express & Graph Objects

### Data Source
* **Dataset:** UCI Machine Learning Repository – Water Treatment Plant Dataset
