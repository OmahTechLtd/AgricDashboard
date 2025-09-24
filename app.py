import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from PIL import Image

# ==============================
# Paths
# ==============================
BASE_DIR = "assets/agriculture"
EDA_DIR = os.path.join(BASE_DIR, "eda_charts")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ==============================
# Load Models and Scaler
# ==============================
@st.cache_resource
def load_models():
    linear_model = joblib.load(os.path.join(MODEL_DIR, "linear_regression_model.pkl"))
    rf_model = joblib.load(os.path.join(MODEL_DIR, "random_forest_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    return linear_model, rf_model, scaler

lin_reg, rf_model, scaler = load_models()

# ==============================
# Load Dataset for Reference
# ==============================
@st.cache_data
def load_data():
    # Use your cleaned Nigerian dataset
    file_path = "drive/MyDrive/nigeria_agriculture_dataset.csv"  # Adjust path if needed
    return pd.read_csv(file_path)

df = load_data()

# ==============================
# Streamlit Page Config
# ==============================
st.set_page_config(page_title="🌾 Nigerian Agriculture Yield Prediction", layout="wide")

st.title("🌾 Nigerian Agriculture Yield Prediction")
st.markdown(
    """
    This dashboard predicts **crop yields** in Nigeria based on agricultural data.  
    Select features, choose a model, and see real-time predictions along with exploratory data analysis (EDA) visuals.
    """
)

# ==============================
# Sidebar Inputs
# ==============================
st.sidebar.header("Configure Prediction")

# Model choice
model_choice = st.sidebar.radio("Choose Prediction Model", ("Linear Regression", "Random Forest"))

# Input features
st.sidebar.subheader("Input Features")
planting_year = st.sidebar.slider("Planting Year", int(df['planting_year'].min()), int(df['planting_year'].max()), 2020)
harvest_year = st.sidebar.slider("Harvest Year", int(df['harvest_year'].min()), int(df['harvest_year'].max()), 2021)
area = st.sidebar.number_input("Area Harvested (hectares)", min_value=10.0, max_value=2_000_000.0, value=50_000.0, step=100.0)
production = st.sidebar.number_input("Production Quantity (metric tons)", min_value=30.0, max_value=6_500_000.0, value=80_000.0, step=100.0)

# ==============================
# Prediction
# ==============================
if st.sidebar.button("Predict Yield"):
    # Prepare data
    input_data = pd.DataFrame([{
        "planting_year": planting_year,
        "harvest_year": harvest_year,
        "area": area,
        "production": production
    }])

    # Scale data
    input_scaled = scaler.transform(input_data)

    # Predict
    if model_choice == "Linear Regression":
        prediction = lin_reg.predict(input_scaled)[0]
    else:
        prediction = rf_model.predict(input_scaled)[0]

    # Display result
    st.success(f"🌱 **Predicted Yield:** {prediction:.2f} metric tons per hectare")

    st.markdown("### Prediction Details")
    st.write(input_data)

# ==============================
# EDA Section
# ==============================
st.markdown("---")
st.header("📊 Exploratory Data Analysis")

st.markdown("Below are key insights from the dataset, generated during the EDA phase.")

# Display EDA Charts
col1, col2 = st.columns(2)

with col1:
    st.subheader("Yield Distribution by Product")
    st.image(os.path.join(EDA_DIR, "yield_distribution_by_product.png"))

    st.subheader("Area vs Production")
    st.image(os.path.join(EDA_DIR, "area_vs_production.png"))

with col2:
    st.subheader("Correlation Heatmap")
    st.image(os.path.join(EDA_DIR, "correlation_heatmap.png"))

    st.subheader("Top Products by Yield")
    st.image(os.path.join(EDA_DIR, "top_products_by_yield.png"))

# ==============================
# Footer
# ==============================
st.markdown("---")
st.markdown("Built with ❤️ by **Omah Tech Ltd** | Empowering Africa with AI")