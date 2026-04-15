import streamlit as st
import numpy as np
import pandas as pd
import pickle

# =========================
# LOAD FILES
# =========================
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# Load dataset (use CSV to avoid openpyxl issue)
df = pd.read_excel("churn.xlsx")

# Clean column names (same as training)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace(".", "_")

# Take first row as default
default = df.iloc[0]

st.title("📊 Telecom Customer Churn Prediction")

# =========================
# INPUT FIELDS
# =========================

account_length = st.number_input("Account Length", value=float(default['account_length']))

# Yes/No handling
intl_default = 'Yes' if str(default['intl_plan']).lower() == 'yes' else 'No'
voice_default = 'Yes' if str(default['voice_plan']).lower() == 'yes' else 'No'

intl_plan_input = st.selectbox("International Plan", ['No', 'Yes'], index=0 if intl_default=='No' else 1)
voice_plan_input = st.selectbox("Voice Plan", ['No', 'Yes'], index=0 if voice_default=='No' else 1)

# Convert to numeric
intl_plan = 1 if intl_plan_input == 'Yes' else 0
voice_plan = 1 if voice_plan_input == 'Yes' else 0


day_mins = st.number_input("Day Minutes", value=float(default['day_mins']))
day_calls = st.number_input("Day Calls", value=float(default['day_calls']))
day_charge = st.number_input("Day Charge", value=float(default['day_charge']))

eve_mins = st.number_input("Evening Minutes", value=float(default['eve_mins']))
eve_calls = st.number_input("Evening Calls", value=float(default['eve_calls']))
eve_charge = st.number_input("Evening Charge", value=float(default['eve_charge']))

night_mins = st.number_input("Night Minutes", value=float(default['night_mins']))
night_calls = st.number_input("Night Calls", value=float(default['night_calls']))
night_charge = st.number_input("Night Charge", value=float(default['night_charge']))

intl_mins = st.number_input("International Minutes", value=float(default['intl_mins']))
intl_calls = st.number_input("International Calls", value=float(default['intl_calls']))
intl_charge = st.number_input("International Charge", value=float(default['intl_charge']))

customer_calls = st.number_input("Customer Service Calls", value=float(default['customer_calls']))


# =========================
# PREDICTION
# =========================

if st.button("Predict"):

    input_dict = {
        'account_length': account_length,
        'intl_plan': intl_plan,
        'voice_plan': voice_plan,
        'day_mins': day_mins,
        'day_calls': day_calls,
        'day_charge': day_charge,
        'eve_mins': eve_mins,
        'eve_calls': eve_calls,
        'eve_charge': eve_charge,
        'night_mins': night_mins,
        'night_calls': night_calls,
        'night_charge': night_charge,
        'intl_mins': intl_mins,
        'intl_calls': intl_calls,
        'intl_charge': intl_charge,
        'customer_calls': customer_calls
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])

    # Apply encoding (same as training)
    input_df = pd.get_dummies(input_df)

    # Match training columns
    input_df = input_df.reindex(columns=columns, fill_value=0)

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)
    prob = model.predict_proba(input_scaled)[0][1]

    # Output
    if prediction[0] == 1:
        st.error(f"⚠️ Customer is likely to CHURN\n\nProbability: {prob:.2f}")
    else:
        st.success(f"✅ Customer is NOT likely to churn\n\nProbability: {prob:.2f}")