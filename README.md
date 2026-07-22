Project Name: Smart Wastewater Facility – Process Digital Twin

Overview: An interactive Streamlit dashboard and Machine Learning digital twin built to predict Chemical Oxygen Demand (COD) levels in wastewater treatment plants using the UCI Machine Learning dataset.

Key Features:

  i) Interactive Control Knobs: Real-time sliders allowing operators to simulate plant inflow conditions (flow rate, pH, COD, conductivity, heavy metals, etc.).
  
  ii) Predictive ML Simulation: Forecasts effluent water quality and alerts users to environmental compliance breaches or toxic shock loads.
  
  iii) Explainable AI: Visual feature importance charts detailing which incoming variables drive predictions.
  
  iv) Multi-Stage Process Kinetics: Visualizes water quality across raw inflow, primary settling (transient), and final effluent (steady-state).
  
  v) Data Export: Capability to download current simulation scenarios into .CSV logs for laboratory record-keeping.

Iterative Evolution and Version Progression:
 
  i) Version 1 (v1): Minimal proof-of-concept with 4 basic inputs and a single Random Forest regressor.
 
  ii) Version 2 (v2): Added model evaluation (R², RMSE), train/test splitting, feature importance charts, and CSV data export. 
  
  iii) Version 3 (v3): Recalibrated the model against full baseline historical data for steady-state process compliance.
  
  iv) Version 4 (v4): Introduced a Dual-Pipeline Architecture to simulate multi-stage processing (transient primary settling vs. steady-state final effluent).
  
  v) Version 5 (v5): Expanded to an 8-dimensional input matrix (including Zinc, BOD, Suspended Solids) and applied hyperparameter tuning (max_depth, min_samples_split).

Tech Stack and Data Source:

UI/Dashboard: Streamlit

Data Processing & ML: Pandas, NumPy, Scikit-Learn

Data Visualization: Plotly Express & Graph Objects

Dataset: UCI Machine Learning Repository – Water Treatment Plant Dataset
