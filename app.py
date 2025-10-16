# app.py

import streamlit as st
import joblib
import numpy as np

model = joblib.load("model.pkl")

st.title("Breast Cancer Classifier")
st.write("Enter the features to predict cancer class.")

features = [
    "mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
    "mean compactness", "mean concavity", "mean concave points", "mean symmetry", "mean fractal dimension"
]

input_data = []
for feature in features:
    val = st.number_input(f"{feature}", min_value=0.0, format="%.4f")
    input_data.append(val)

if st.button("Predict"):
    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    if prediction == 0:
        st.success("Prediction: Malignant")
    else:
        st.success("Prediction: Benign")
