import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.title("🩺 Diabetes Prediction")
st.write("Enter health metrics to predict diabetes risk using a trained logistic regression model.")

# Load model and scaler
try:
    model = joblib.load("logistic_regression_model.joblib")
    scaler = joblib.load("diabetes_scaler.joblib")
except Exception as e:
    st.error(f"❌ Error loading model or scaler: {e}")
    st.stop()

# Load dataset preview with safe encoding


@st.cache_data
def load_data():
    try:
        return pd.read_csv("data.csv", encoding='utf-8')
    except UnicodeDecodeError:
        return pd.read_csv("data.csv", encoding='latin1')


try:
    df = load_data()
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())
except Exception as e:
    st.warning(f"⚠️ Could not load dataset preview: {e}")

# Sidebar input fields (16 features)
st.sidebar.header("🧑‍⚕️ Patient Data")
fields = {
    "Age": (0, 100, 30),
    "Pregnancies": (0, 20, 2),
    "BMI": (0.0, 50.0, 25.0),
    "Glucose": (0.0, 200.0, 100.0),
    "BloodPressure": (0, 130, 70),
    "HbA1c": (0.0, 15.0, 5.0),
    "LDL": (0.0, 200.0, 100.0),
    "HDL": (0.0, 100.0, 50.0),
    "Triglycerides": (0.0, 300.0, 120.0),
    "WaistCircumference": (0.0, 150.0, 90.0),
    "HipCircumference": (0.0, 150.0, 100.0),
    "WHR": (0.0, 2.0, 0.9),
}

# Numeric inputs
input_values = []
for name, (min_val, max_val, default) in fields.items():
    val = st.sidebar.number_input(
        name, min_value=min_val, max_value=max_val, value=default)
    input_values.append(val)

# Categorical binary inputs (assume 1 = Yes, 0 = No)
family_history = st.sidebar.selectbox(
    "Family History of Diabetes?", ("No", "Yes"))
diet_type = st.sidebar.selectbox("Diet Type", ("Normal", "Vegetarian"))
hypertension = st.sidebar.selectbox("Hypertension?", ("No", "Yes"))
medication_use = st.sidebar.selectbox(
    "Currently on Medication?", ("No", "Yes"))

# Encode categorical values
input_values.extend([
    1 if family_history == "Yes" else 0,
    1 if diet_type == "Vegetarian" else 0,
    1 if hypertension == "Yes" else 0,
    1 if medication_use == "Yes" else 0
])

# Convert and scale
features = np.array([input_values])  # shape: (1, 16)
try:
    scaled = scaler.transform(features)
except Exception as e:
    st.error(f"❌ Feature mismatch or scaler error: {e}")
    st.stop()

# Prediction
if st.sidebar.button("🔍 Predict Diabetes"):
    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][1]

    st.subheader("🔮 Prediction Result")
    st.write("**🟥 Diabetic**" if pred == 1 else "**🟩 Non‑Diabetic**")
    st.write(f"**Probability:** {prob*100:.2f}%")
