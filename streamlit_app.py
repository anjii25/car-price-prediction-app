import streamlit as st
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Car Price Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("car_price_prediction_.csv")

df = load_data()

# Sidebar navigation
st.sidebar.title("Car Price Prediction")
st.sidebar.markdown("By [Your Names]")
page = st.sidebar.selectbox("Navigate to", [
    "Introduction",
    "Data Visualization", 
    "Price Prediction",
    "Feature Importance",
    "Model Experiments",
    "Conclusion"
])

# Main content area
if page == "Introduction":
    st.title("Car Price Prediction App")
    
    st.markdown("""
    ## Business Problem
    
    The used car market faces significant challenges in accurate pricing:
    
    - **Pricing Inconsistency**: Similar cars priced very differently
    - **Information Asymmetry**: Sellers know more than buyers
    - **Market Volatility**: Prices fluctuate based on many factors
    
    **Our Solution**: A machine learning app that predicts fair market prices 
    based on car specifications.
    """)
    
    st.subheader("Dataset Overview")
    st.dataframe(df.head(10))
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Cars", len(df))
        st.metric("Number of Brands", df['Brand'].nunique())
    
    with col2:
        st.metric("Price Range", f"${df['Price'].min():.0f} - ${df['Price'].max():.0f}")
        st.metric("Average Price", f"${df['Price'].mean():.0f}")
    
    with col3:
        st.metric("Year Range", f"{df['Year'].min()} - {df['Year'].max()}")
        st.metric("Fuel Types", df['Fuel Type'].nunique())

elif page == "Data Visualization":
    st.title("Data Visualization")
    st.info("Visualizations will be added in the next steps!")
    
elif page == "Price Prediction":
    st.title("Price Prediction")
    st.info("Prediction models will be added in the next steps!")
    
elif page == "Feature Importance":
    st.title("Feature Importance")
    st.info("SHAP analysis will be added in the next steps!")
    
elif page == "Model Experiments":
    st.title("Model Experiments")
    st.info("MLflow tracking will be added in the next steps!")
    
elif page == "Conclusion":
    st.title("Conclusion")
    st.info("Project summary will be added in the next steps!")
