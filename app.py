import pickle

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Salary Predictor", page_icon="💰", layout="centered")

# ---------------------------------------------------------------------------
# Load the trained model
# ---------------------------------------------------------------------------
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

# ---------------------------------------------------------------------------
# Title & description
# ---------------------------------------------------------------------------
st.title("💰 Salary Predictor")
st.write(
    "This app predicts an employee's **salary** based on their "
    "**years of experience**, using a Linear Regression model."
)

# ---------------------------------------------------------------------------
# User input
# ---------------------------------------------------------------------------
experience = st.number_input(
    "Years of Experience",
    min_value=0.0,
    max_value=50.0,
    value=5.0,
    step=0.1,
)

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
if st.button("Predict Salary"):
    input_df = pd.DataFrame({"Experience Years": [experience]})
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Salary: ${prediction:,.2f}")