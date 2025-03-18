import streamlit as st
import pandas as pd
import lightgbm as lgb
import pickle

# Load the trained model from a saved file
@st.cache_resource
def load_model():
    with open("recession_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

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
    probability = model.predict_proba(input_data)[0][1]  # Probability of recession
    
    # Display Result
    st.subheader("Prediction Result:")
    if probability > 0.5:
        st.error(f"⚠️ Recession Likely! (Probability: {probability:.2%})")
    else:
        st.success(f"✅ No Recession Expected (Probability: {probability:.2%})")
    
    st.markdown("**Disclaimer:** This model is based on historical economic data and should not be considered financial advice.")
