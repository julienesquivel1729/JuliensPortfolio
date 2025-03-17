import pandas as pd
import plotly.express as px
import streamlit as st

# Load dataset
file_path = "DisasterDeclarationsSummaries.csv"  # Ensure this file is accessible

df = pd.read_csv(file_path, low_memory=False)
df['declarationDate'] = pd.to_datetime(df['declarationDate'], errors='coerce')
df['yearDeclared'] = df['declarationDate'].dt.year

df = df.dropna(subset=['state', 'incidentType'])  # Ensure clean data

# Streamlit App Title
st.title("FEMA Disaster Declarations Dashboard")

# Description
st.markdown(
    """
    ### About This Dashboard
    This interactive dashboard provides insights into disaster declarations issued by FEMA.
    Users can explore disaster trends across different U.S. states and filter by disaster types.
    The data is sourced from FEMA's open database, covering major disasters such as hurricanes, floods, wildfires, and more.
    
    **How to Use:**
    - Select a **state** from the dropdown to view disaster trends.
    - Select a **disaster type** to analyze occurrences over time.
    - View visualizations that help understand disaster frequency and distribution across the U.S.
    
    """
)

# State Selection
state = st.selectbox("Select a State:", sorted(df['state'].unique()))

disaster = st.selectbox("Select Disaster Type:", sorted(df['incidentType'].unique()))

# Filter Data
filtered_df = df[(df['state'] == state) & (df['incidentType'] == disaster)]

# Disaster Trend Over Time
st.subheader(f"Trend of {disaster} in {state}")
trend_fig = px.line(
    filtered_df.groupby('yearDeclared').size().reset_index(name='count'),
    x='yearDeclared', y='count', markers=True
)
st.plotly_chart(trend_fig)

# Disaster Count by State (Comparison)
st.subheader(f"{disaster} Count by State")
count_fig = px.bar(
    df[df['incidentType'] == disaster].groupby('state').size().reset_index(name='count'),
    x='state', y='count'
)
st.plotly_chart(count_fig)

# Run using: streamlit run disaster_dashboard.py
