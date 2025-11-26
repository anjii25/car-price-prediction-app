import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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
st.sidebar.markdown("By Anji and Jessie")
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
    - **Regional Variations**: Different markets value features differently
    
    **Our Solution**: A machine learning app that predicts fair market prices 
    based on car specifications, helping both buyers and sellers make informed decisions.
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
    
    # Data quality check
    st.subheader("Data Quality")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        st.success("No missing values found")
    else:
        st.warning(f"Missing values detected: {missing[missing > 0]}")

elif page == "Data Visualization":
    st.title("Car Market Insights")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "Price Distribution", 
        "Brand Analysis", 
        "Feature Relationships",
        "Market Trends"
    ])
    
    with tab1:
        st.subheader("Car Price Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df['Price'], bins=50, kde=True, ax=ax)
        ax.set_xlabel("Price ($)")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Car Prices")
        st.pyplot(fig)
        
    with tab2:
        st.subheader("Brand Analysis")
        brand_avg = df.groupby('Brand')['Price'].mean().sort_values(ascending=False)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        brand_avg.plot(kind='bar', ax=ax, color='skyblue')
        ax.set_title("Average Price by Brand")
        ax.set_ylabel("Average Price ($)")
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
    with tab3:
        st.subheader("Feature Relationships")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title("Feature Correlation Heatmap")
        st.pyplot(fig)
        
    with tab4:
        st.subheader("Market Trends Over Time")
        yearly_avg = df.groupby('Year')['Price'].mean()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        yearly_avg.plot(kind='line', ax=ax, marker='o')
        ax.set_title("Average Car Price by Year")
        ax.set_ylabel("Average Price ($)")
        ax.set_xlabel("Year")
        st.pyplot(fig)

elif page == "Price Prediction":
    st.title("🤖 Car Price Prediction")
    
    # Data preprocessing
    df_pred = df.copy()
    
    # Encode categorical variables
    le_brand = LabelEncoder()
    le_fuel = LabelEncoder() 
    le_transmission = LabelEncoder()
    le_condition = LabelEncoder()
    le_model = LabelEncoder()
    
    df_pred['Brand_encoded'] = le_brand.fit_transform(df_pred['Brand'])
    df_pred['Fuel_encoded'] = le_fuel.fit_transform(df_pred['Fuel Type'])
    df_pred['Transmission_encoded'] = le_transmission.fit_transform(df_pred['Transmission'])
    df_pred['Condition_encoded'] = le_condition.fit_transform(df_pred['Condition'])
    df_pred['Model_encoded'] = le_model.fit_transform(df_pred['Model'])
    
    # Feature selection
    feature_cols = ['Year', 'Engine Size', 'Mileage', 'Brand_encoded', 
                   'Fuel_encoded', 'Transmission_encoded', 'Condition_encoded', 'Model_encoded']
    
    X = df_pred[feature_cols]
    y = df_pred['Price']
    
    # Model selection
    st.sidebar.subheader("Model Configuration")
    model_choice = st.sidebar.selectbox(
        "Choose Model",
        ["Linear Regression", "Random Forest"]
    )
    
    test_size = st.sidebar.slider("Test Size (%)", 10, 40, 20)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size/100, random_state=42
    )
    
    # Model training
    if model_choice == "Linear Regression":
        model = LinearRegression()
    elif model_choice == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred) 
    r2 = r2_score(y_test, y_pred)
    
    # Display results
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean Squared Error", f"${mse:,.0f}")
    with col2:
        st.metric("Mean Absolute Error", f"${mae:,.0f}")
    with col3:
        st.metric("R² Score", f"{r2:.3f}")
    
    # Actual vs Predicted plot
    st.subheader("Actual vs Predicted Prices")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    plot_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred,
        'Error': abs(y_test - y_pred)
    })
    
    sns.scatterplot(data=plot_df, x='Actual', y='Predicted', hue='Error', 
                   palette='viridis', alpha=0.7, ax=ax)
    
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    
    ax.set_xlabel("Actual Price ($)")
    ax.set_ylabel("Predicted Price ($)")
    ax.set_title(f"Actual vs Predicted Car Prices ({model_choice})")
    st.pyplot(fig)

elif page == "Feature Importance":
    st.title("🔍 Feature Importance Analysis")
    st.info("SHAP analysis will be implemented in the next iteration!")
    
elif page == "Model Experiments":
    st.title("🧪 Model Experiments")
    st.info("MLflow tracking will be implemented in the next iteration!")
    
elif page == "Conclusion":
    st.title("🎯 Project Conclusion")
    st.markdown("""
    ## Business Impact Summary
    
    Our Car Price Prediction App successfully addresses key challenges in the automotive market.
    
    ### Key Findings:
    - **Vehicle Age** and **Brand** are top price influencers
    - **Machine learning** can predict prices with good accuracy  
    - **Data-driven insights** help both buyers and sellers
    
    ### Next Steps:
    - Add SHAP explainability
    - Implement MLflow experiment tracking
    - Enhance visualizations
    """)