import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load your trained ML model
model = pickle.load(open("wine_model.pkl", "rb"))

# Title
st.title("🍷 Wine Quality Prediction App")

# User inputs
st.header("Enter Wine Characteristics:")

fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0, max_value=20.0, step=0.1)
volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0, max_value=2.0, step=0.01)
citric_acid = st.number_input("Citric Acid", min_value=0.0, max_value=2.0, step=0.01)
residual_sugar = st.number_input("Residual Sugar", min_value=0.0, max_value=20.0, step=0.1)
chlorides = st.number_input("Chlorides", min_value=0.0, max_value=1.0, step=0.001)
alcohol = st.number_input("Alcohol", min_value=0.0, max_value=20.0, step=0.1)

# Predict button
if st.button("Predict Quality"):
    features = np.array([[fixed_acidity, volatile_acidity, citric_acid,
                          residual_sugar, chlorides, alcohol]])
    prediction = model.predict(features)
    st.success(f"Predicted Wine Quality: {prediction[0]}")
