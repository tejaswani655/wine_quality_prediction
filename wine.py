import streamlit as st
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
import pickle
import time

# ------------------------------
# Title and Header
# ------------------------------
st.title("🍷 Wine Quality Prediction App")

st.image("https://i.pinimg.com/originals/62/2d/cf/622dcf55a0b3d569f1e5a4c597c51401.gif")

st.header("Model to predict Wine Quality", divider=True)

st.subheader("""User must enter values for the following features:
['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 
 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 
 'density', 'pH', 'sulphates', 'alcohol']""")

st.sidebar.title("Select Wine Features 🍷")
st.sidebar.image("https://i.pinimg.com/originals/1b/82/c4/1b82c4b515cb7f5d5f12e534fdc4dbd1.jpg")

# ------------------------------
# Load Dataset (for ranges)
# ------------------------------
temp_df = pd.read_csv("wineQualityReds.csv")   # make sure dataset is uploaded with app
col = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide',
       'density', 'pH', 'sulphates', 'alcohol']

random.seed(42)


# ------------------------------
# Sidebar sliders for input
# ------------------------------
all_values = []

for i in col:
    min_value, max_value = temp_df[i].agg(['min','max'])
    # Create slider for each feature
    var = st.sidebar.slider(f"Select {i} value",
                            float(min_value), float(max_value),
                            float(random.uniform(min_value, max_value)))
    all_values.append(var)

# ------------------------------
# Scaling input values
# ------------------------------
scaler = StandardScaler()
scaler.fit(temp_df[col])   # fit on full dataset
final_value = scaler.transform([all_values])

# ------------------------------
# Load trained model
# ------------------------------
with open("wine Quality.pkl", "rb") as f:
    wine_model = pickle.load(f)

# ------------------------------
# Predict
# ------------------------------
prediction = wine_model.predict(final_value)[0]

# Progress animation
progress_bar = st.progress(0)
placeholder = st.empty()
placeholder.subheader("Predicting Wine Quality...")

for i in range(100):
    time.sleep(0.02)
    progress_bar.progress(i + 1)

placeholder.empty()

# Display result
if prediction > 0:
    st.success(f"✅ Predicted Wine Quality: *{prediction}* (scale 0–10)")
else:
    st.warning("Invalid Wine Feature Values")

st.markdown("Designed and developed by: *Tejaswani Raj and Peeyush Sheoran*")
