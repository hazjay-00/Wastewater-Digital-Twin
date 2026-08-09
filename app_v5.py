import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error

# PAGE LAYOUT CONFIGURATION

st.set_page_config(page_title="Industrial Digital Twin - Version 5", layout="wide")

st.title("Smart Wastewater Facility - Process Digital Twin")
st.markdown("---")

# IMPROVED DUAL-MACHINE LEARNING PIPELINE ENGINE

@st.cache_resource
def run_and_train_engine():
    data_source_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/water-treatment/water-treatment.data"

    columns = [
        'Q_E', 'ZN_E', 'PH_E', 'DBO_E', 'DQO_E', 'SS_E', 'SSV_E', 'SED_E', 'COND_E',
        'PH_P', 'DBO_P', 'SS_P', 'SSV_P', 'COND_P', 'PH_D', 'DBO_D', 'DQO_D', 'SS_D',
        'SSV_D', 'COND_D', 'PH_S', 'DBO_S', 'DQO_S', 'SS_S', 'SSV_S', 'COND_S', 'RD_DBO_P'
    ]
    
    df = pd.read_csv(data_source_url, header=None, names=columns, na_values="?", on_bad_lines="skip")
    df = df.fillna(df.mean(numeric_only=True))
    
    # 8 EXPANDED INPUT FEATURES 
    feature_names = ['Q_E', 'ZN_E', 'PH_E', 'DBO_E', 'DQO_E', 'SS_E', 'SSV_E', 'COND_E']
    X = df[feature_names]
    
    y_transient = df['DQO_D']  # Intermediate Primary/Sediment Settling Output
    y_steady = df['DQO_S']     # Final Outgoing Treated Effluent Quality
    
    # --- Transient Model Training (Out-of-sample Testing) ---
    X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X, y_transient, test_size=0.2, random_state=42)
    model_transient = RandomForestRegressor(n_estimators=150, max_depth=15, min_samples_split=5, random_state=42)
    model_transient.fit(X_train_t, y_train_t)
    
    y_pred_t = model_transient.predict(X_test_t)
    r2_transient = r2_score(y_test_t, y_pred_t)
    rmse_transient = root_mean_squared_error(y_test_t, y_pred_t)
    
    # --- Steady-State Model Training (In-sample Baseline) ---
    model_steady = RandomForestRegressor(n_estimators=150, max_depth=15, min_samples_split=5, random_state=42)
    model_steady.fit(X, y_steady)
    
    y_pred_s = model_steady.predict(X)
    r2_steady = r2_score(y_steady, y_pred_s)
    rmse_steady = root_mean_squared_error(y_steady, y_pred_s)
    
    # Extract Feature Importances (Based on final treated parameters)
    importances = model_steady.feature_importances_
    importance_df = pd.DataFrame({
        'Chemical Parameter': [
            'Inflow Rate (Q_E)', 'Zinc Influx (ZN_E)', 'Inflow pH (PH_E)', 
            'Biological Demand (DBO_E)', 'Chemical Demand (DQO_E)', 
            'Suspended Solids (SS_E)', 'Volatile Solids (SSV_E)', 'Conductivity (COND_E)'
        ],
        'Importance Weight': importances
    }).sort_values(by='Importance Weight', ascending=True)
    
    return model_transient, r2_transient, rmse_transient, model_steady, r2_steady, rmse_steady, importance_df

# Initialize Engine
m_transient, r2_t, rmse_t, m_steady, r2_s, rmse_s, feature_importance_data = run_and_train_engine()

# INTERACTIVE COMPREHENSIVE CONTROL KNOBS

st.sidebar.header("Comprehensive Control Knobs")

input_flow = st.sidebar.slider("Wastewater Inflow Rate (m³/day)", 10000, 60000, 37000)
input_zn = st.sidebar.slider("Incoming Zinc Levels (mg/L)", 0.0, 15.0, 2.5, 0.1)
input_ph = st.sidebar.slider("Incoming Wastewater pH", 6.0, 9.0, 7.8, 0.1)
input_dbo = st.sidebar.slider("Incoming Biological Oxygen Demand (DBO mg/L)", 20, 500, 150)
input_cod = st.sidebar.slider("Incoming Chemical Oxygen Demand (COD mg/L)", 100, 600, 400)
input_ss = st.sidebar.slider("Incoming Suspended Solids (SS mg/L)", 20, 600, 200)
input_ssv = st.sidebar.slider("Incoming Volatile Suspended Solids (%)", 10.0, 90.0, 65.0, 0.5)
input_cond = st.sidebar.slider("Incoming Conductivity (µS/cm)", 500, 3000, 1500)

# Process Data Row Layout
current_status = pd.DataFrame(
    [[input_flow, input_zn, input_ph, input_dbo, input_cod, input_ss, input_ssv, input_cond]], 
    columns=['Q_E', 'ZN_E', 'PH_E', 'DBO_E', 'DQO_E', 'SS_E', 'SSV_E', 'COND_E']
)

# Inference Forecasts
pred_transient = m_transient.predict(current_status)[0]
pred_steady = m_steady.predict(current_status)[0]

LEGAL_MAX_COD = 120.0

# DIGITAL TWIN SIMULATION ANALYTICS

col1, col2 = st.columns(2)

with col1:
    st.subheader("Transient Analysis")
    st.metric(label="Predicted Primary Settling Output COD", value=f"{pred_transient:.2f} mg/L")
    
    # Anomaly Rule Trigger
    if input_ph < 6.5 or input_ph > 8.5 or input_cod > 550 or input_cond > 2600 or input_zn > 10.0:
        st.error("TOXIC INFRASTRUCTURE ALERT: Corrosive influx load detected at the main headworks!")
    else:
        st.success("INFLUX NORMAL: Incoming chemistry safely within primary tank load capacities.")

with col2:
    st.subheader("Steady-State Analysis")
    st.metric(label="Predicted Final Effluent Output COD", value=f"{pred_steady:.2f} mg/L")
    
    if pred_steady > LEGAL_MAX_COD:
        st.error(f"DISCHARGE COMPLIANCE VIOLATION: Final water breaks statutory limit ({LEGAL_MAX_COD} mg/L)!")
    else:
        st.success("OUTFLOW COMPLIANT: Long-term biological assimilation meets legal discharge codes.")

# INTEGRATED VISUALIZATION DASHBOARD

st.markdown("---")
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown("### Real-Time Multi-Stage Process Kinetics")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=['1. Raw Wastewater Inflow', '2. Transient Primary Output', '3. Matured Final Effluent'], 
        y=[input_cod, pred_transient, pred_steady], 
        marker_color=['#FFA07A', '#F4D03F', '#20B2AA']
    ))
    fig.update_layout(yaxis_title="Chemical Oxygen Demand (mg/L)", template="plotly_white", height=350, margin=dict(t=20, b=20, l=20, r=20))
    st.plotly_chart(fig, use_container_width=True)

with chart_col2:
    st.markdown("### 8-Feature Weights")
    fig_importance = px.bar(
        feature_importance_data, x='Importance Weight', y='Chemical Parameter',
        orientation='h', color='Importance Weight', color_continuous_scale='Viridis'
    )
    fig_importance.update_layout(height=350, template="plotly_white", showlegend=False, coloraxis_showscale=False, margin=dict(t=20, b=20, l=20, r=20))
    st.plotly_chart(fig_importance, use_container_width=True)

# MODEL PERFORMANCE EVALUATION MATRIX

st.markdown("---")
st.markdown("### Model Performance Evaluation")
m_col1, m_col2 = st.columns(2)

with m_col1:
    st.markdown("#### **Transient Model Validation (Out-of-Sample Split)**")
    st.metric(label="Testing R² Score", value=f"{r2_t:.4f}")
    st.metric(label="Testing Error (RMSE)", value=f"{rmse_t:.2f} mg/L")

with m_col2:
    st.markdown("#### **Steady-State Model Validation (In-Sample Baseline)**")
    st.metric(label="Baseline R² Score", value=f"{r2_s:.4f}")
    st.metric(label="Baseline Error (RMSE)", value=f"{rmse_s:.2f} mg/L")

# EXPORT INTERFACE

st.markdown("---")
st.markdown("### Simulation Data Logging Archive")

log_data = pd.DataFrame([
    {
        "Timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Inflow_Rate_m3_day": input_flow,
        "Inflow_pH": input_ph,
        "Inflow_COD_mgL": input_cod,
        "Inflow_Conductivity_uS_cm": input_cond,
        "Predicted_Effluent_COD_mgL": round(pred_steady, 2),
    }
])
st.dataframe(log_data, use_container_width=True, hide_index=True)
csv_buffer = log_data.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Export Current 8-Dimensional Scenario Log (.CSV)",
    data=csv_buffer,
    file_name="simulation_run_v5.csv",
    mime="text/csv",
)
