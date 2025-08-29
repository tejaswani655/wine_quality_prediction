import streamlit as st
import numpy as np
import pickle

# Load trained model (make sure wine_model.pkl is in same folder)
model = pickle.load(open("wine Quality.pkl", "rb"))

# Title
st.title("🍷 Wine Quality Prediction App")
st.write("Enter the wine characteristics below to predict its quality:")

# Helper to safely parse floats from text input
def get_float_input(label, default="0.0"):
    val = st.text_input(label, value=default)
    try:
        return float(val)
    except ValueError:
        st.warning(f"⚠️ Please enter a valid number for {label}. Defaulting to {default}.")
        return float(default)

# Collect all 11 features used in training
fixed_acidity = get_float_input("Fixed Acidity", "7.0")
volatile_acidity = get_float_input("Volatile Acidity", "0.7")
citric_acid = get_float_input("Citric Acid", "0.0")
residual_sugar = get_float_input("Residual Sugar", "1.9")
chlorides = get_float_input("Chlorides", "0.076")
free_sulfur_dioxide = get_float_input("Free Sulfur Dioxide", "11.0")
total_sulfur_dioxide = get_float_input("Total Sulfur Dioxide", "34.0")
density = get_float_input("Density", "0.9978")
pH = get_float_input("pH", "3.51")
sulphates = get_float_input("Sulphates", "0.56")
alcohol = get_float_input("Alcohol", "9.4")

# Prediction button
if st.button("Predict Quality"):
    # Arrange inputs in the same order as training
    features = np.array([[fixed_acidity, volatile_acidity, citric_acid,
                          residual_sugar, chlorides, free_sulfur_dioxide,
                          total_sulfur_dioxide, density, pH,
                          sulphates, alcohol]])
    
    prediction = model.predict(features)
    st.success(f"Predicted Wine Quality: {prediction[0]}")
