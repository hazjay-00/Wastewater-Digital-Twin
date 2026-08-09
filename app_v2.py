import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error

# PAGE LAYOUT CONFIGURATION

st.set_page_config(page_title="Industrial Digital Twin - Version 2", layout="wide")

st.title("Smart Wastewater Facility - Process Digital Twin")
st.markdown("---")

# ML ENGINE

@st.cache_resource
def run_and_train_engine():
    # File Pathway Link Configuration
    data_source_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/water-treatment/water-treatment.data"

    # Define actual engineering sensor categories based on plant schematics
    columns = [
        'Q_E', 'ZN_E', 'PH_E', 'DBO_E', 'DQO_E', 'SS_E', 'SSV_E', 'SED_E', 'COND_E',
        'PH_P', 'DBO_P', 'SS_P', 'SSV_P', 'COND_P', 'PH_D', 'DBO_D', 'DQO_D', 'SS_D',
        'SSV_D', 'COND_D', 'PH_S', 'DBO_S', 'DQO_S', 'SS_S', 'SSV_S', 'COND_S', 'RD_DBO_P'
    ]
    
    # Read data directly, identifying "?" characters as empty inputs
    df = pd.read_csv(data_source_url, header=None, names=columns, na_values="?", on_bad_lines="skip")
    
    # Mean Imputation
    df = df.fillna(df.mean(numeric_only=True))
    
    # Isolate Process Inputs (X) and Target Output (y)
    feature_names = ['Q_E', 'PH_E', 'DQO_E', 'COND_E']
    X = df[feature_names]
    y = df['DQO_S']
    
    # Train/Test Split and Model Evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Construct and train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model accuracy metrics
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    
    # Extract Feature Importances 
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'Chemical Parameter': ['Inflow Rate', 'Inflow pH', 'Inflow COD', 'Inflow Conductivity'],
        'Importance Weight': importances
    }).sort_values(by='Importance Weight', ascending=True)
    
    return model, r2, rmse, importance_df

# Initialize and lock the compiled AI model engine into memory
twin_model, model_r2, model_rmse, feature_importance_data = run_and_train_engine()

# INTERACTIVE PROCESS CONTROL KNOBS

st.sidebar.header("Physical Plant Control Knobs")
st.sidebar.markdown("Manipulate incoming wastewater parameters to simulate real-time operations.")

# Build interactive slider bounds
input_flow = st.sidebar.slider("Wastewater Inflow Rate (m³/day)", min_value=10000, max_value=60000, value=37000)
input_ph = st.sidebar.slider("Incoming Wastewater pH", min_value=6.0, max_value=9.0, value=7.8, step=0.1)
input_cod = st.sidebar.slider("Incoming Chemical Oxygen Demand (COD mg/L)", min_value=100, max_value=600, value=400)
input_cond = st.sidebar.slider("Incoming Conductivity (µS/cm)", min_value=500, max_value=3000, value=1500)

# DIGITAL TWIN PREDICTIVE SIMULATION

# Convert current physical slider settings into a clean table row for processing
current_status = pd.DataFrame([[input_flow, input_ph, input_cod, input_cond]], columns=['Q_E', 'PH_E', 'DQO_E', 'COND_E'])

# Request the machine learning core to forecast output water quality
predicted_effluent_cod = twin_model.predict(current_status)[0]

# Establish environmental discharge boundary limits
LEGAL_MAX_COD = 120.0

# Split screen workspace layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Operational Status")
    st.metric(label="Predicted Effluent COD Output", value=f"{predicted_effluent_cod:.2f} mg/L")

with col2:
    st.subheader("Safety & Risk Analysis")
    
    # 1. Rule-Based Override
    if input_ph < 6.5 or input_ph > 8.5 or input_cod > 550 or input_cond > 2600:
        st.error("TOXIC INFLOW ANOMALY: Extreme chemical or pH load detected!")
        st.warning("Biological Risk: Active sludge microbes are dying. Immediate plant stabilization required.")
        
    # 2. Check if calculated effluent levels break regulatory law
    elif predicted_effluent_cod > LEGAL_MAX_COD:
        st.error(f"CRITICAL SYSTEM CRASH PREDICTED: Effluent COD breaks environmental limits ({LEGAL_MAX_COD} mg/L)!")
        
    # 3. Safe State
    else:
        st.success("SYSTEM STABLE: Plant operating efficiently within standard parameters.")

# KINETICS VISUALIZATION DASHBOARD

st.markdown("---")
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown("### Real-Time Process Kinetics Twin Chart")
    # Construct an interactive structural chart comparing initial contamination vs forecasted effluent outputs
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=['Incoming Raw Wastewater', 'Predicted Outgoing Effluent'], 
        y=[input_cod, predicted_effluent_cod], 
        marker_color=['#FFA07A', '#20B2AA']
    ))
    fig.update_layout(yaxis_title="Chemical Oxygen Demand (mg/L)", template="plotly_white", height=350, margin=dict(t=20, b=20, l=20, r=20))
    st.plotly_chart(fig, use_container_width=True)

with chart_col2:
    # Feature Importance Chart
    st.markdown("### Feature Weights")
    fig_importance = px.bar(
        feature_importance_data,
        x='Importance Weight',
        y='Chemical Parameter',
        orientation='h',
        color='Importance Weight',
        color_continuous_scale='Viridis'
    )
    fig_importance.update_layout(height=350, template="plotly_white", showlegend=False, coloraxis_showscale=False, margin=dict(t=20, b=20, l=20, r=20))
    st.plotly_chart(fig_importance, use_container_width=True)

# METRICS SECTION 

st.markdown("---")
st.markdown("### Verification Metrics")
metric_col1, metric_col2, metric_col3 = st.columns(3)

with metric_col1:
    st.metric(label="Model R² Score (Goodness of Fit)", value=f"{model_r2:.4f}")
    st.caption("Proportion of effluent variance predictable from the input parameters.")

with metric_col2:
    st.metric(label="Root Mean Squared Error (RMSE)", value=f"{model_rmse:.2f} mg/L")
    st.caption("Standard deviation of the residuals/prediction errors.")

with metric_col3:
    st.metric(label="Algorithm Engine Type", value="Random Forest")
    st.caption("Ensemble regressor comprising 100 parallelised decision tree nodes.")

# SIMULATION RUN DATA LOGGING & EXPORT

st.markdown("---")
st.markdown("### Simulation Laboratory Data Export")

# Gather current data layout parameters into a row dataset format
log_data = pd.DataFrame([{
    "Timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Inflow_Rate_m3_day": input_flow,
    "Inflow_pH": input_ph,
    "Inflow_COD_mgL": input_cod,
    "Inflow_Conductivity_uS_cm": input_cond,
    "Predicted_Effluent_COD_mgL": round(predicted_effluent_cod, 2),
    "System_Alert_Triggered": "YES" if (input_ph < 6.5 or input_ph > 8.5 or input_cod > 550 or input_cond > 2600 or predicted_effluent_cod > LEGAL_MAX_COD) else "NO"
}])

st.dataframe(log_data, use_container_width=True, hide_index=True)

# Convert log data to downloadable CSV format configuration
csv_buffer = log_data.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Current Simulation Scenario Log File (.CSV)",
    data=csv_buffer,
    file_name="simulation_run_v2.csv",
    mime="text/csv"
)
