import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import time

# Title and Header
st.title("🍷 Wine Quality Prediction App")
st.image("https://i.pinimg.com/originals/62/2d/cf/622dcf55a0b3d569f1e5a4c597c51401.gif")
st.header("Model to predict Wine Quality", divider=True)
st.subheader("Enter values for the following features:")

st.sidebar.title("Select Wine Features 🍷")
st.sidebar.image("https://i.pinimg.com/originals/1b/82/c4/1b82c4b515cb7f5d5f12e534fdc4dbd1.jpg")

# Define fixed ranges from UCI Data
ranges = {
    'fixed acidity': (3.80, 15.90),
    'volatile acidity': (0.08, 1.58),
    'citric acid': (0.00, 1.66),
    'residual sugar': (0.60, 65.80),
    'chlorides': (0.009, 0.611),
    'free sulfur dioxide': (1.0, 289.0),
    'total sulfur dioxide': (6, 440),
    'density': (0.987, 1.039),
    'pH': (2.72, 4.01),
    'sulphates': (0.22, 2.00),
    'alcohol': (8.0, 14.9)
}

col = list(ranges.keys())



# Sidebar sliders with fixed data-based ranges
all_values = []
for feature in col:
    min_val, max_val = ranges[feature]
    default_val = (min_val + max_val) / 2
    val = st.sidebar.slider(
        label=f"{feature.capitalize()}",
        min_value=float(min_val),
        max_value=float(max_val),
        value=float(default_val)
    )
    all_values.append(val)

# Load dataset only for scaling
temp_df = pd.read_csv("wineQualityReds.csv")
scaler = StandardScaler()
scaler.fit(temp_df[col])
final_value = scaler.transform([all_values])

# Load trained model
with open("wine Quality.pkl", "rb") as f:
    wine_model = pickle.load(f)

# Predict with progress
prediction = wine_model.predict(final_value)[0]

progress_bar = st.progress(0)
placeholder = st.empty()
placeholder.subheader("Predicting Wine Quality...")
for i in range(100):
    time.sleep(0.01)
    progress_bar.progress(i + 1)
placeholder.empty()

# Display result
if prediction >= 0:
    st.success(f"✅ Predicted Wine Quality: *{prediction}* (scale 0–10)")
else:
    st.warning("Invalid Wine Feature Values")

st.markdown("Designed and developed by: *Tejaswani Raj and Peeyush Sheoran*")
