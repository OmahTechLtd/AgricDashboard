import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ==============================
# Paths
# ==============================
MODEL_DIR = "assets/models"
EDA_DIR = "assets/eda"
DATA_DIR = "assets/data"

# ==============================
# Load Models and Scaler
# ==============================
lin_reg = joblib.load(os.path.join(MODEL_DIR, "linear_regression_model.pkl"))
rf_model = joblib.load(os.path.join(MODEL_DIR, "random_forest_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

# ==============================
# Load Dataset
# ==============================
# This dataset will help populate dropdowns dynamically
df = pd.read_csv(os.path.join(DATA_DIR, "nigeria_agriculture.csv"))

# Extract unique values
products = sorted(df['product'].unique())
states = sorted(df['admin_1'].unique())

# ==============================
# Streamlit Page Config
# ==============================
st.set_page_config(page_title="🌾 Nigerian Agriculture Yield Prediction", layout="wide")

st.title("🌾 Nigerian Agriculture Yield Prediction")
st.markdown(
    """
    Predict **crop yields** across Nigeria using Machine Learning.  
    Choose a model and input farm details to get an estimated yield.
    """
)

# ==============================
# Input Features (Main Page)
# ==============================
st.header("Input Farm Details")

col1, col2 = st.columns(2)

with col1:
    product = st.selectbox("Select Crop", products)
    state = st.selectbox("Select State", states)
    planting_year = st.slider("Planting Year", min_value=1999, max_value=2023, value=2020)
    harvest_year = st.slider("Harvest Year", min_value=1999, max_value=2023, value=2021)

with col2:
    area = st.number_input("Area (hectares)", min_value=1.0, value=100.0, step=1.0)
    production = st.number_input("Production (metric tons)", min_value=1.0, value=500.0, step=1.0)
    model_choice = st.radio("Select Model", ("Linear Regression", "Random Forest"))

# ==============================
# Predict Button
# ==============================
if st.button("Predict Yield"):
    # Create dataframe for model input
    input_df = pd.DataFrame([{
        "admin_1": state,
        "product": product,
        "planting_year": planting_year,
        "harvest_year": harvest_year,
        "area": area,
        "production": production
    }])

    # One-hot encode like training
    input_df = pd.get_dummies(input_df)

    # Ensure same columns as training
    training_columns = joblib.load(os.path.join(MODEL_DIR, "columns.pkl"))
    input_df = input_df.reindex(columns=training_columns, fill_value=0)

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict
    if model_choice == "Linear Regression":
        prediction = lin_reg.predict(input_scaled)[0]
    else:
        prediction = rf_model.predict(input_scaled)[0]

    # Display result
    st.success(f"🌱 **Predicted Yield:** {prediction:.2f} tons per hectare")

    # Show encoded input for transparency
    st.markdown("### Encoded Input Data")
    st.write(input_df)

# ==============================
# EDA Section
# ==============================
st.markdown("---")
st.header("Exploratory Data Analysis (EDA)")

st.write("Below are some key insights from the dataset:")

eda_cols = st.columns(2)

with eda_cols[0]:
    st.image(os.path.join(EDA_DIR, "yield_distribution_by_product.png"), caption="Yield Distribution by Product")
    st.image(os.path.join(EDA_DIR, "area_vs_production.png"), caption="Area vs Production")

with eda_cols[1]:
    st.image(os.path.join(EDA_DIR, "correlation_heatmap.png"), caption="Correlation Heatmap")
    st.image(os.path.join(EDA_DIR, "top_products_by_yield.png"), caption="Top Products by Yield")

st.image(os.path.join(EDA_DIR, "feature_importance.png"), caption="Top Feature Importances (Random Forest)")

# ==============================
# Footer
# ==============================
st.markdown("---")
st.markdown("Built with ❤️ by **Omah Tech Ltd**")