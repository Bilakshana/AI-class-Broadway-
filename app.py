import streamlit as st
import joblib
import numpy as np

# Load model, scaler, encoder
model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')
encoder = joblib.load('encoder.joblib')

st.title("Breast Cancer Prediction App")
st.write("Adjust the sliders for the patient's measurements:")

# Features and their ranges for sliders
features = {
    'radius_mean': (0.0, 30.0),
    'texture_mean': (0.0, 40.0),
    'perimeter_mean': (0.0, 200.0),
    'area_mean': (0.0, 2500.0),
    'smoothness_mean': (0.0, 0.2),
    'compactness_mean': (0.0, 1.0),
    'concavity_mean': (0.0, 1.0),
    'concave points_mean': (0.0, 0.3),
    'symmetry_mean': (0.0, 0.5),
    'fractal_dimension_mean': (0.0, 0.1),
    'radius_se': (0.0, 5.0),
    'texture_se': (0.0, 5.0),
    'perimeter_se': (0.0, 20.0),
    'area_se': (0.0, 500.0),
    'smoothness_se': (0.0, 0.05),
    'compactness_se': (0.0, 0.2),
    'concavity_se': (0.0, 0.2),
    'concave points_se': (0.0, 0.1),
    'symmetry_se': (0.0, 0.2),
    'fractal_dimension_se': (0.0, 0.05),
    'radius_worst': (0.0, 40.0),
    'texture_worst': (0.0, 50.0),
    'perimeter_worst': (0.0, 250.0),
    'area_worst': (0.0, 3000.0),
    'smoothness_worst': (0.0, 0.2),
    'compactness_worst': (0.0, 2.0),
    'concavity_worst': (0.0, 2.0),
    'concave points_worst': (0.0, 0.5),
    'symmetry_worst': (0.0, 1.0),
    'fractal_dimension_worst': (0.0, 0.1)
}

# Thresholds for high-risk features
risk_thresholds = {
    'radius_mean': 15,
    'perimeter_mean': 100,
    'area_mean': 800,
    'concavity_mean': 0.15,
    'concave points_mean': 0.1
}

# Collect user input
input_data = []
high_risk_features = []
for feature, (min_val, max_val) in features.items():
    val = st.slider(feature.replace('_', ' ').title(),
                    min_val, max_val, (min_val + max_val)/2)
    input_data.append(val)
    if feature in risk_thresholds and val > risk_thresholds[feature]:
        high_risk_features.append(feature)


# Predict button
if st.button("Predict"):
    arr = np.array(input_data).reshape(1, -1)
    scaled = scaler.transform(arr)
    pred = model.predict(scaled)
    label = encoder.inverse_transform(pred)[0]

    if label == 'M':
        st.warning("Diagnosis: M - Malignant")
        st.write("⚠️ The tumor is likely cancerous and needs medical attention.")
    else:
        st.success("Diagnosis: B - Benign")
        st.write("✅ The tumor is likely not cancerous and may only need monitoring.")

    if high_risk_features:
        st.write("High-risk measurements detected:")
        for feat in high_risk_features:
            st.warning(
                f"{feat.replace('_', ' ').title()} is above normal levels!")
