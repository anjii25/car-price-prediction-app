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
import shap
import os
import wandb

wandb.login(key="916eb733271a059e07018432656f6fb084c889b6")
if "experiment_history" not in st.session_state:
    st.session_state.experiment_history = []


# Page configuration
st.set_page_config(
    page_title="Car Price Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load dat
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
    "Hyperparameter Tuning",
    "Conclusion"
])
st.sidebar.image("assets/vroooommmmmm.jpg")

# ========== INTRODUCTION PAGE ==========
if page == "Introduction":
    st.title("Car Price Prediction App")
    col1, col2 = st.columns([2, 1])  # Adjust ratio if needed
    with col1:
        st.markdown("""
        ## Business Problem: Solving Used Car Market Inefficiencies
        
        The $1.2 trillion global used car market suffers from:
        
        - **Pricing Inconsistency**: Identical cars vary by 20-30% in price
        - **Information Asymmetry**: Sellers have more information than buyers
        - **Market Volatility**: Prices change rapidly based on supply/demand
        - **Regional Variations**: Same car, different prices across regions
        
        **Our ML Solution**: A transparent, data-driven pricing tool that:
        - Predicts fair market value using 8 key features
        - Explains what factors drive prices up/down
        - Helps both buyers and sellers make informed decisions
        """)
    with col2:
        st.image("assets/kachow.avif")
    
    st.subheader("Dataset Overview")
    st.dataframe(df.head(10))
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Cars", f"{len(df):,}")
        st.metric("Brands", df['Brand'].nunique())
    with col2:
        st.metric("Price Range", f"${df['Price'].min():,.0f}-${df['Price'].max():,.0f}")
        st.metric("Avg Price", f"${df['Price'].mean():,.0f}")
    with col3:
        st.metric("Year Range", f"{df['Year'].min()}-{df['Year'].max()}")
        st.metric("Fuel Types", df['Fuel Type'].nunique())
    
    # Data quality
    st.subheader("Data Quality Check")
    if df.isnull().sum().sum() == 0:
        st.success("no missing values!")
    else:
        st.warning("Some missing values detected")

# ========== DATA VISUALIZATION PAGE ==========
elif page == "Data Visualization":
    st.title("Car Market Insights & Analytics")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Price Analysis", "Brand Analysis", "Feature Relationships", "Market Trends"])
    
    with tab1:
        st.subheader("Car Price Distribution")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Histogram
        sns.histplot(df['Price'], bins=50, kde=True, ax=ax1, color='skyblue')
        ax1.set_xlabel("Price ($)")
        ax1.set_ylabel("Count")
        ax1.set_title("Price Distribution")
        
        # Box plot by condition
        sns.boxplot(data=df, x='Condition', y='Price', ax=ax2)
        ax2.set_title("Price by Condition")
        ax2.tick_params(axis='x', rotation=45)
        
        st.pyplot(fig)
    
    with tab2:
        st.subheader("Brand Performance Analysis")
        
        # Average price by brand
        brand_stats = df.groupby('Brand').agg({
            'Price': ['mean', 'count']
        }).round(2)
        brand_stats.columns = ['Avg Price', 'Number of Cars']
        brand_stats = brand_stats.sort_values('Avg Price', ascending=False)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar chart
        brand_stats['Avg Price'].head(10).plot(kind='bar', ax=ax1, color='lightcoral')
        ax1.set_title("Top 10 Brands by Average Price")
        ax1.set_ylabel("Average Price ($)")
        ax1.tick_params(axis='x', rotation=45)
        
        # Pie chart
        top_brands = df['Brand'].value_counts().head(8)
        ax2.pie(top_brands.values, labels=top_brands.index, autopct='%1.1f%%')
        ax2.set_title("Brand Distribution")
        
        st.pyplot(fig)
        st.dataframe(brand_stats)
    
    with tab3:
        st.subheader("Feature Correlation Analysis")
        
        # Numeric correlations
        numeric_df = df.select_dtypes(include=[np.number])
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title("Feature Correlation Matrix")
        st.pyplot(fig)
        
        # Feature vs Price
        feature = st.selectbox("Select feature to compare with price:", 
                             ['Year', 'Engine Size', 'Mileage'])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df, x=feature, y='Price', hue='Fuel Type', alpha=0.6, ax=ax)
        ax.set_title(f"{feature} vs Price")
        st.pyplot(fig)
    
    with tab4:
        st.subheader("Market Trends Over Time")
        
        # Price trends
        yearly_trends = df.groupby('Year').agg({
            'Price': 'mean',
            'Car ID': 'count'
        }).rename(columns={'Car ID': 'Count'})
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        yearly_trends['Price'].plot(ax=ax1, marker='o', color='green')
        ax1.set_title("Average Price Trend")
        ax1.set_ylabel("Price ($)")
        
        yearly_trends['Count'].plot(ax=ax2, marker='o', color='orange')
        ax2.set_title("Number of Cars by Year")
        ax2.set_ylabel("Count")
        
        st.pyplot(fig)

# ========== PRICE PREDICTION PAGE ==========
elif page == "Price Prediction":
    st.title("Car Price Prediction by Brand")
    
    st.markdown("""
    ### Brand-Specific Modeling
    Training separate models for each brand for more accurate predictions.
    """)
    
    brand_counts = df['Brand'].value_counts()
    valid_brands = brand_counts[brand_counts >= 30].index.tolist()
    
    selected_brand = st.selectbox(
        "Select a car brand:",
        valid_brands
    )
    

    df_brand = df[df['Brand'] == selected_brand].copy()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(f"{selected_brand} Cars", len(df_brand))
    with col2:
        st.metric("Avg Price", f"${df_brand['Price'].mean():,.0f}")
    with col3:
        st.metric("Price Range", f"${df_brand['Price'].min():,.0f}-${df_brand['Price'].max():,.0f}")
    
    X = df_brand[['Year', 'Engine Size', 'Mileage']]
    y = df_brand['Price']
    
    if len(df_brand) < 30:
        st.error(f"Not enough data for {selected_brand}. Only {len(df_brand)} cars found.")
    else:
        # Step 4: Train/test split
        test_size = st.slider(
            "Test size (%)", 
            min_value=10, 
            max_value=40, 
            value=20, 
            step=5
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size/100, random_state=42
        )
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        st.subheader(f"{selected_brand} Model Performance")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("MSE", f"${mse:,.0f}")
        with col2:
            st.metric("MAE", f"${mae:,.0f}")
        with col3:
            st.metric("R²", f"{r2:.3f}")
        
        st.subheader("Feature Impacts on Price")
        coef_df = pd.DataFrame({
            'Feature': X.columns,
            'Impact per unit': [f"${c:,.2f}" for c in model.coef_],
            'Direction': ['Increases price' if c > 0 else 'Decreases price' for c in model.coef_]
        })
        st.dataframe(coef_df)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        ax1.scatter(y_test, y_pred, alpha=0.6, s=50, color='steelblue')
        
        all_values = np.concatenate([y_test, y_pred])
        min_val = all_values.min() * 0.95
        max_val = all_values.max() * 1.05
        
        ax1.plot([min_val, max_val], [min_val, max_val], 
                'r--', lw=2.5, label='Perfect Prediction')
        
        ax1.set_xlabel("Actual Price ($)", fontsize=12)
        ax1.set_ylabel("Predicted Price ($)", fontsize=12)
        ax1.set_title(f"{selected_brand}: Actual vs Predicted", fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax1.set_xlim([min_val, max_val])
        ax1.set_ylim([min_val, max_val])
        
        most_important_idx = np.argmax(np.abs(model.coef_))
        most_important_feature = X.columns[most_important_idx]
        
        ax2.scatter(df_brand[most_important_feature], df_brand['Price'], 
                   alpha=0.6, s=50, color='coral')
        ax2.set_xlabel(most_important_feature, fontsize=12)
        ax2.set_ylabel("Price ($)", fontsize=12)
        ax2.set_title(f"Price vs {most_important_feature}", fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)

        st.subheader("Try It Yourself: Predict a Price")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            input_year = st.number_input(
                "Year", 
                min_value=int(df_brand['Year'].min()), 
                max_value=int(df_brand['Year'].max()),
                value=int(df_brand['Year'].median())
            )
        with col2:
            input_engine = st.number_input(
                "Engine Size (L)", 
                min_value=float(df_brand['Engine Size'].min()),
                max_value=float(df_brand['Engine Size'].max()),
                value=float(df_brand['Engine Size'].median()),
                step=0.1
            )
        with col3:
            input_mileage = st.number_input(
                "Mileage", 
                min_value=int(df_brand['Mileage'].min()),
                max_value=int(df_brand['Mileage'].max()),
                value=int(df_brand['Mileage'].median())
            )
        
        if st.button("Predict Price", type="primary"):
            input_data = pd.DataFrame({
                'Year': [input_year],
                'Engine Size': [input_engine],
                'Mileage': [input_mileage]
            })
            
            predicted_price = model.predict(input_data)[0]
            
            st.success(f"### Predicted Price: **${predicted_price:,.2f}**")
            
            st.info(f"""
            **Comparison:**
            - Your predicted price: ${predicted_price:,.0f}
            - Average {selected_brand} price: ${df_brand['Price'].mean():,.0f}
            - This is {"above" if predicted_price > df_brand['Price'].mean() else "below"} average
            """)

# ========== FEATURE IMPORTANCE PAGE ==========
elif page == "Feature Importance":
    st.title("Feature Importance & Model Explainability")
    
    # Prepare data
    df_pred = df.copy()
    categorical_cols = ['Brand', 'Fuel Type', 'Transmission', 'Condition', 'Model']
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_pred[f'{col}_encoded'] = le.fit_transform(df_pred[col])
    
    feature_cols = ['Year', 'Engine Size', 'Mileage'] + [f'{col}_encoded' for col in categorical_cols]
    feature_names = ['Year', 'Engine Size', 'Mileage', 'Brand', 'Fuel Type', 'Transmission', 'Condition', 'Model']
    
    X = df_pred[feature_cols]
    y = df_pred['Price']
    
    # Train model for SHAP
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Sample for faster computation
    X_sample = X.sample(min(500, len(X)), random_state=42)
    
    # SHAP analysis
    st.subheader("SHAP Feature Importance")
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)
    
    # Summary plot
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    st.pyplot(fig)
    
    # Feature importance values
    st.subheader("Mean Absolute SHAP Values")
    mean_shap = pd.DataFrame({
        'Feature': feature_names,
        'Importance': np.abs(shap_values.values).mean(0)
    }).sort_values('Importance', ascending=False)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.dataframe(mean_shap, use_container_width=True)
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(data=mean_shap, y='Feature', x='Importance', ax=ax)
        ax.set_title("Feature Importance Ranking")
        st.pyplot(fig)
    
    # Individual prediction explanation
    st.subheader("Individual Prediction Explanation")
    
    example_idx = st.slider("Select example to explain", 0, len(X_sample)-1, 0)
    
    st.markdown("#### Waterfall Plot")
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap_values[example_idx], show=False)
    st.pyplot(fig)


# ========== HYPERPARAMETER TUNING PAGE ==========
elif page == "Hyperparameter Tuning":
    st.title("Hyperparameter Tuning & Experiment Tracking")
    
    st.markdown("""
    ## Weight & Biases Experiment Tracking
    
    This page demonstrates hyperparameter tuning and experiment tracking using W&B.
    Below you can run experiments with different parameters and track their performance.
    """)
    
    # Experiment configuration
    st.subheader("Configure Experiment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_type = st.selectbox("Model Type", ["Random Forest", "Linear Regression", "Gradient Boosting"])
        n_estimators = st.slider("Number of Estimators", 50, 200, 100)
    
    with col2:
        max_depth = st.slider("Max Depth", 5, 50, 20)
        test_size = st.slider("Test Size", 0.1, 0.4, 0.3)
    
    if st.button("Run Experiment"):
        # SPLIT FIRST - CRITICAL!
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        
        # Define features and target
        X = df.drop(['Price', 'Car ID'], axis=1)  # Drop ID column
        y = df['Price']
        
        # Split first
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Define preprocessing
        numeric_features = ['Year', 'Engine Size', 'Mileage']
        categorical_features = ['Brand', 'Fuel Type', 'Transmission', 'Condition', 'Model']
        
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])
        
        # Create model based on selection
        if model_type == "Random Forest":
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == "Linear Regression":
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
        else:  # Gradient Boosting
            from sklearn.ensemble import GradientBoostingRegressor
            model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42
            )
        
        # Create pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        
        # Train and predict
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Display feature importance for tree-based models
        if model_type != "Linear Regression":
            try:
                # Get feature names after one-hot encoding
                feature_names = numeric_features.copy()
                ohe = preprocessor.named_transformers_['cat']
                if hasattr(ohe, 'get_feature_names_out'):
                    cat_features = ohe.get_feature_names_out(categorical_features)
                    feature_names.extend(cat_features)
                
                # Get feature importance
                importances = model.feature_importances_
                importance_df = pd.DataFrame({
                    'Feature': feature_names[:len(importances)],
                    'Importance': importances
                }).sort_values('Importance', ascending=False).head(10)
                
                st.subheader("Top 10 Feature Importances")
                st.dataframe(importance_df)
            except:
                pass
        
        # Save experiment locally
        st.session_state.experiment_history.append({
            "model": model_type,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "test_size": test_size,
            "mse": round(mse, 2),
            "mae": round(mae, 2),
            "r2": round(r2, 4)
        })

        # Display results
        st.success("Experiment Completed!")
        
        col1, col2, col3 = st.columns(3)
        with col1: 
            st.metric("MSE", f"${mse:,.0f}")
            st.write(f"RMSE: ${np.sqrt(mse):,.0f}")
        with col2: 
            st.metric("MAE", f"${mae:,.0f}")
            st.write(f"MAPE: {np.mean(np.abs((y_test - y_pred) / y_test)) * 100:.1f}%")
        with col3: 
            st.metric("R²", f"{r2:.4f}")
            st.write(f"Baseline R²: 0.0")
        
        # Visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Actual vs Predicted
        ax1.scatter(y_test, y_pred, alpha=0.5)
        ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax1.set_xlabel("Actual Price ($)")
        ax1.set_ylabel("Predicted Price ($)")
        ax1.set_title("Actual vs Predicted")
        
        # Residuals
        residuals = y_test - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.5)
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_xlabel("Predicted Price ($)")
        ax2.set_ylabel("Residuals ($)")
        ax2.set_title("Residual Plot")
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Display actual vs predicted table
        st.subheader("Sample Predictions")
        results_df = pd.DataFrame({
            'Actual': y_test.values[:10],
            'Predicted': y_pred[:10],
            'Error': (y_test.values[:10] - y_pred[:10])
        })
        st.dataframe(results_df.style.format({
            'Actual': '${:,.0f}',
            'Predicted': '${:,.0f}',
            'Error': '${:,.0f}'
        }))

# ========== CONCLUSION PAGE ==========
elif page == "Conclusion":
    st.title("Project Conclusion & Limitations")
    
    # Executive Summary
    st.markdown("""
    ## Summary of Findings
    
    This project demonstrates a machine learning pipeline for car price prediction, but faces significant limitations due to data quality issues.
    """)
    
    # Key Findings
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Technical Achievements:**
        - Built complete ML pipeline with multiple models
        - Created interactive visualization dashboard
        - Implemented hyperparameter tuning system
        - Added model explainability with SHAP
        """)
    
    with col2:
        st.markdown("""
        **Critical Limitations:**
        - Dataset contains synthetic, unrealistic data
        - Model shows high error rates (~47%)
        - Some brands have negative R² values
        - Not suitable for production use
        """)
    
    # Data Quality Issues
    st.markdown("""
    ## Data Quality Problems
    
    The dataset has several issues that limit model reliability:
    
    1. **Unrealistic Prices**: Cars from 2000-2005 priced at $90,000+
    2. **Contradictory Features**: Teslas listed with petrol/diesel engines
    3. **Impossible Conditions**: 20+ year old cars marked "New"
    4. **Wrong Models**: Tesla Model 3 listed in 2001 (didn't exist yet)
    5. **Computer-Generated**: All data appears synthetic, not real transactions
    """)
    
    # Model Performance
    st.markdown("""
    ## Model Performance
    
    **Key Metrics:**
    - R² Score: 0.68 (explains 68% of price variance)
    - Average Error: ±$28,147 per prediction
    - Error Rate: 47% relative to average car price
    - Brand Performance: Varies widely, some negative R²
    
    **Interpretation:**
    The model performs worse than simply using average prices for some brands.
    Error rates are too high for financial decision-making.
    """)
    
    # Business Implications
    st.markdown("""
    ## Business Implications
    
    **What This Can Do:**
    - Demonstrate ML methodology and workflow
    - Provide rough directional price guidance
    - Serve as educational/training example
    
    **What This Cannot Do:**
    - Predict actual transaction prices
    - Replace professional appraisals
    - Be used for financial decisions
    - Account for real market conditions
    
    # Recommendations
    st.markdown("""
    ## Recommendations
    
    1. **Fix Data First**: Source real transaction data before further development
    2. **Validate Features**: Remove impossible feature combinations
    3. **Segment Models**: Build separate models for different vehicle types
    4. **Add Real Features**: Include accident history, maintenance records, location data
    5. **Industry Partnerships**: Work with automotive data providers for authentic data
    
    **Minimum Production Requirements:**
    - Real transaction data with verified VINs
    - Accuracy: ±10% error rate maximum
    - Coverage: 90%+ of mainstream vehicles
    - Speed: Sub-second predictions
    """)
    
    **Project Status:** Educational demonstration only
    """)
```

# Run the app
if __name__ == "__main__":
    pass
