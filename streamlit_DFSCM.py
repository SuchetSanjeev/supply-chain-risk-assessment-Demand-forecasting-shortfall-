import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime


# === Load Model Artifacts ===
@st.cache_data
def load_model_artifacts():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    columns = joblib.load("columns.pkl")
    return model, scaler, columns

model, scaler, reference_columns = load_model_artifacts()

# === Feature Engineering ===
def apply_feature_engineering(df): 
    df['Order_Pressure'] = df['Ordered_Qty'] / df['Committed_Lead_Days']
    df['Vendor_Risk'] = (1 - df['Reliability_score'] / 100) * df['Quality_Rejection_Rate (%)']
    df['Transit_Risk'] = df['Avg_Transit_Days'] * df['Weather_Disruption_Index'] * df['Route Risk Score']
    df['Demand_vs_Reliability'] = df['Ordered_Qty'] / (df['Reliability_score'] + 1)
    df['Stress_Score'] = df['Transit_Risk'] * df['Order_Pressure']
    df['Price_per_Unit_vs_Order'] = df['Price_per_Unit'] / (df['Ordered_Qty'] + 1)

    df['High_Risk_Vendor'] = (df['Vendor_Risk'] > 1.5).astype(int)
    df['High_Order_Pressure'] = (df['Order_Pressure'] > 20).astype(int)
    df['High_Transit_Risk'] = (df['Transit_Risk'] > 25).astype(int)
    df['High_Demand_vs_Reliability'] = (df['Demand_vs_Reliability'] > 8).astype(int)
    df['High_Stress_Score'] = (df['Stress_Score'] > 400).astype(int)
    df['Low_Order_Pressure'] = (df['Order_Pressure'] < 6.2).astype(int)
    df['Low_Stress_Score'] = (df['Stress_Score'] < 40).astype(int)
    df['Low_Demand_vs_Reliability'] = (df['Demand_vs_Reliability'] < 3).astype(int)
    df['Low_Vendor_Risk'] = (df['Vendor_Risk'] < 0.5).astype(int)
    df['Low_Price_per_Unit_vs_Order'] = (df['Price_per_Unit_vs_Order'] < 0.05).astype(int)
    return df

def encode_features(df):
    # === Categorical Encoding ===
    # ID columns
    id_cols = ['Component_ID', 'Vendor_ID', 'Route_ID', 'Source']
    for col in id_cols:
        df[col] = pd.Series(df[col]).astype('category').cat.codes

    # Ordinal encoding
    congestion_map = {'Low': 0, 'Medium': 1, 'High': 2}
    df['Peak_Congestion_Indicator'] = df['Peak_Congestion_Indicator'].map(congestion_map)

    # One-hot encoding manually (as in training)
    df['Mode_Lorry'] = int(df['Mode'].iloc[0] == 'Lorry')
    df['Mode_Train'] = int(df['Mode'].iloc[0] == 'Train')
    df['Backup Route Availability_Yes'] = int(df['Backup Route Availability'].iloc[0] == 'Yes')

    df.drop(columns=['Mode', 'Backup Route Availability'], inplace=True, errors='ignore')

    return df

# === Prediction Function ===
def predict_shortfall(input_dict):
    df = pd.DataFrame([input_dict])
    df = apply_feature_engineering(df)
    df = encode_features(df)
    df_final = df.reindex(columns=reference_columns, fill_value=0)
    df_scaled = scaler.transform(df_final)
    prob = model.predict_proba(df_scaled)[0][1]
    prediction = int(prob >= 0.5)
    return prediction, prob

# === Streamlit UI ===
st.title("📦 Supply Chain Shortfall Risk Predictor")

def save_log(input_data, prediction, probability):
    log_entry = input_data.copy()
    log_entry["Predicted_Prob"] = round(probability, 4)
    log_entry["Predicted_Flag"] = prediction
    log_entry["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_df = pd.DataFrame([log_entry])
    log_path = "prediction_logs.csv"

    if os.path.exists(log_path):
        log_df.to_csv(log_path, mode='a', header=False, index=False)
    else:
        log_df.to_csv(log_path, index=False)


with st.form("prediction_form"):
    Ordered_Qty = st.number_input("Ordered Quantity", value=100)
    Committed_Lead_Days = st.number_input("Committed Lead Days", value=30)
    Reliability_score = st.slider("Reliability Score (%)", 0, 100, 87)
    Quality_Rejection_Rate = st.slider("Quality Rejection Rate (%)", 0, 100, 6)
    collaboration_tenure = st.slider("Collaboration Tenure (years)", 0, 10, 2)
    avg_lead_days = st.number_input("Avg Lead Days", value=7)
    past_incident_count = st.number_input("Past Incident Count", value=1)
    Price_per_Unit = st.number_input("Price per Unit", value=88.0)
    Mode = st.selectbox("Mode", ['Air', 'Lorry', 'Train'])
    Peak_Congestion_Indicator = st.selectbox("Peak Congestion Indicator", ['Low', 'Medium', 'High'])
    Backup_Route = st.selectbox("Backup Route Availability", ['Yes', 'No'])
    Avg_Transit_Days = st.number_input("Avg Transit Days", value=3)
    Weather_Disruption_Index = st.slider("Weather Disruption Index", 0.0, 10.0, 1.0)
    Route_Risk_Score = st.slider("Route Risk Score", 0, 10, 2)
    Component_ID = st.number_input("Component ID", value=1)
    Vendor_ID = st.number_input("Vendor ID", value=5)
    Route_ID = st.number_input("Route ID", value=6)
    Source = st.number_input("Source", value=12)

    submitted = st.form_submit_button("Predict Shortfall")

if submitted:
    input_dict = {
        'Ordered_Qty': Ordered_Qty,
        'Committed_Lead_Days': Committed_Lead_Days,
        'Reliability_score': Reliability_score,
        'Quality_Rejection_Rate (%)': Quality_Rejection_Rate,
        'collaboration_tenure': collaboration_tenure,
        'avg_lead_days': avg_lead_days,
        'past_incident_count': past_incident_count,
        'Price_per_Unit': Price_per_Unit,
        'Mode': Mode,
        'Peak_Congestion_Indicator': Peak_Congestion_Indicator,
        'Backup Route Availability': Backup_Route,
        'Avg_Transit_Days': Avg_Transit_Days,
        'Weather_Disruption_Index': Weather_Disruption_Index,
        'Route Risk Score': Route_Risk_Score,
        'Component_ID': Component_ID,
        'Vendor_ID': Vendor_ID,
        'Route_ID': Route_ID,
        'Source': Source
    }

    pred, prob = predict_shortfall(input_dict)

    st.subheader("🔍 Prediction Result")
    st.write(f"**Shortfall Risk Flag:** {'Yes' if pred else 'No'}")
    st.write(f"**Shortfall Probability:** {prob:.4f}")
    
    # Save prediction log
    save_log(input_dict, pred, prob)

    st.success("Prediction logged!")

    # Show past logs
    st.subheader("📋 Prediction History")
    if os.path.exists("prediction_logs.csv"):
        logs = pd.read_csv("prediction_logs.csv")
        st.dataframe(logs.tail(10), use_container_width=True)
    else:
        st.info("No logs yet.")

    # Clear logs
    if st.button("🗑️ Clear Logs"):
        os.remove("prediction_logs.csv")
        st.success("Logs cleared.")


