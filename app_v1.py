import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

# PAGE LAYOUT CONFIGURATION
st.set_page_config(page_title="Industrial Digital Twin", layout="wide")

st.title("Smart Wastewater Facility - Process Digital Twin")
st.markdown("---")

# MACHINE LEARNING ENGINE
@st.cache_resource
def run_and_train_engine():
    # Fetching the dataset directly from the UCI Repository
    data_source_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/water-treatment/water-treatment.data"
    
    # Define actual engineering sensor categories based on plant schematics
    columns = [
        'Q_E', 'ZN_E', 'PH_E', 'DBO_E', 'DQO_E', 'SS_E', 'SSV_E', 'SED_E', 'COND_E',
        'PH_P', 'DBO_P', 'SS_P', 'SSV_P', 'COND_P', 'PH_D', 'DBO_D', 'DQO_D', 'SS_D',
        'SSV_D', 'COND_D', 'PH_S', 'DBO_S', 'DQO_S', 'SS_S', 'SSV_S', 'COND_S', 'RD_DBO_P'
    ]
    
    # Read data directly from the web link, identifying "?" characters as empty inputs
    df = pd.read_csv(data_source_url, header=None, names=columns, na_values="?", on_bad_lines="skip")
    
    # Fill missing sensor gaps automatically with the column's mean average
    df = df.fillna(df.mean(numeric_only=True))
    
    # Isolate Process Inputs (X)
    X = df[['Q_E', 'PH_E', 'DQO_E', 'COND_E']]
    
    # Isolate Target Output (y)
    y = df['DQO_S']
    
    # Construct and train the industrial Random Forest predictive ML model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model

# Initialize and lock the compiled AI model engine into memory
twin_model = run_and_train_engine()

# INTERACTIVE CONTROL KNOBS

st.sidebar.header("Physical Plant Control Knobs")
st.sidebar.markdown("Manipulate incoming wastewater parameters to simulate real-time operations.")

# Build interactive slider bounds mapping to real operational historical ranges
input_flow = st.sidebar.slider("Wastewater Inflow Rate (m³/day)", min_value=10000, max_value=60000, value=37000)
input_ph = st.sidebar.slider("Incoming Wastewater pH", min_value=6.5, max_value=8.5, value=7.8, step=0.1)
input_cod = st.sidebar.slider("Incoming Chemical Oxygen Demand (COD mg/L)", min_value=100, max_value=600, value=400)
input_cond = st.sidebar.slider("Incoming Conductivity (µS/cm)", min_value=500, max_value=3000, value=1500)

# 4. DIGITAL TWIN PREDICTIVE SIMULATION

# Convert current physical slider settings into a clean table row for processing
current_status = pd.DataFrame([[input_flow, input_ph, input_cod, input_cond]], columns=['Q_E', 'PH_E', 'DQO_E', 'COND_E'])

# Request the machine learning core to forecast output water quality instantaneously
predicted_effluent_cod = twin_model.predict(current_status)[0]

# Establish strict statutory legal environmental discharge boundary limits
LEGAL_MAX_COD = 120.0

# Split screen workspace layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Operational Status")
    st.metric(label="Predicted Effluent COD Output", value=f"{predicted_effluent_cod:.2f} mg/L")

with col2:
    st.subheader("Safety & Risk Analysis")
    
    # 1. Check for immediate toxic industrial dumping 
    if input_ph < 6.0 or input_ph > 9.0 or input_cod > 300 or input_cond > 2500:
        st.error("TOXIC INFLOW DETECTED: Extreme chemical or pH load will disrupt biological treatment!")
        st.warning("Action Required: Increase aeration or prepare emergency chemical dosing.")
        
    # 2. Check if the model predicts an environmental breach
    elif predicted_effluent_cod > LEGAL_MAX_COD:
        st.error(f"CRITICAL FAILURE PREDICTED: Effluent COD breaks environmental limits ({LEGAL_MAX_COD} mg/L)!")
        
    # 3. If everything passes, the system is stable
    else:
        st.success("SYSTEM STABLE: Plant operating efficiently within standard parameters.")

# KINETICS VISUALIZATION DASHBOARD

st.markdown("### Real-Time Process Kinetics Twin Chart")

# Construct an interactive structural chart comparing initial contamination vs forecasted effluent outputs
fig = go.Figure()
fig.add_trace(go.Bar(
    x=['Incoming Raw Wastewater', 'Predicted Outgoing Effluent'], 
    y=[input_cod, predicted_effluent_cod], 
    marker_color=['#FFA07A', '#20B2AA']
))
fig.update_layout(yaxis_title="Chemical Oxygen Demand (mg/L)", template="plotly_white", height=400)
st.plotly_chart(fig, use_container_width=True)