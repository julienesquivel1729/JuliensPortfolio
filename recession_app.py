import streamlit as st
import pandas as pd
import lightgbm as lgb
import pickle
import os
import urllib.request

# 📌 GitHub URL for model file
MODEL_URL = "https://raw.githubusercontent.com/julienesquivel1729/JuliensMathNotes/main/recession_model.pkl"
MODEL_PATH = "recession_model.pkl"

# 📌 Load the trained model (downloads from GitHub if not found)
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model file...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    try:
        with open(MODEL_PATH, "rb") as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the pre-trained model
model = load_model()

# Streamlit App UI
st.title("📉 Recession Prediction Model")
st.markdown(
    """
    This machine learning model analyzes key economic indicators and predicts 
    the likelihood of a recession.
    
    **How to Use:**
    - Enter the current economic data.
    - Click "Predict Recession" to see the probability.
    """
)

# User Inputs
unrate = st.number_input("Unemployment Rate (%)", min_value=0.0, max_value=20.0, value=4.0, step=0.1)
inflation = st.number_input("Core Inflation Rate (%)", min_value=0.0, max_value=10.0, value=2.5, step=0.1)
gdp_growth = st.number_input("GDP Growth Rate (%)", min_value=-10.0, max_value=10.0, value=2.0, step=0.1)
yield_spread = st.number_input("Yield Curve Spread (%)", min_value=-5.0, max_value=5.0, value=0.5, step=0.1)

# Convert inputs to DataFrame
input_data = pd.DataFrame({
    "UNRATE": [unrate],
    "CORESTICKM159SFRBATL": [inflation],
    "GDP_Growth": [gdp_growth],
    "Yield_Curve_Spread": [yield_spread]
})

# Prediction
if st.button("Predict Recession"):
    if model:
        probability = model.predict_proba(input_data)[0][1]  # Probability of recession
        
        # Display Result
        st.subheader("Prediction Result:")
        if probability > 0.5:
            st.error(f"⚠️ Recession Likely! (Probability: {probability:.2%})")
        else:
            st.success(f"✅ No Recession Expected (Probability: {probability:.2%})")
        
        st.markdown("**Disclaimer:** This model is based on historical economic data and should not be considered financial advice.")
    else:
        st.error("🚨 Model failed to load. Please try again later.")

