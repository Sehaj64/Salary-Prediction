import pandas as pd
import numpy as np
import os
import joblib
import argparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

def load_data(data_dir):
    """
    Loads and merges the main dataset with college and city mappings.
    """
    print("Loading data...")
    main_path = os.path.join(data_dir, "ML case Study.csv")
    colleges_path = os.path.join(data_dir, "Colleges.csv")
    cities_path = os.path.join(data_dir, "cities.csv")

    try:
        df = pd.read_csv(main_path)
        college_df = pd.read_csv(colleges_path)
        cities_df = pd.read_csv(cities_path)
        return df, college_df, cities_df
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None, None, None

def preprocess_data(df, college_df, cities_df):
    """
    Cleans and preprocesses the data for training.
    """
    print("Preprocessing data...")
    
    # --- College Mapping ---
    tier1 = college_df["Tier 1"].dropna().tolist()
    tier2 = college_df["Tier 2"].dropna().tolist()
    tier3 = college_df["Tier 3"].dropna().tolist()

    college_map = {}
    for c in tier1: college_map[c] = 3
    for c in tier2: college_map[c] = 2
    for c in tier3: college_map[c] = 1

    df['College'] = df['College'].map(college_map)
    df['College'] = df['College'].fillna(1) # Default to Tier 1 if unknown

    # --- City Mapping ---
    metro_cities = cities_df["Metro City"].dropna().tolist()
    non_metro_cities = cities_df["non-metro cities"].dropna().tolist()

    city_map = {}
    for c in metro_cities: city_map[c] = 1
    for c in non_metro_cities: city_map[c] = 0

    df['City'] = df['City'].map(city_map)
    df['City'] = df['City'].fillna(0) # Default to Non-Metro

    # --- Role Mapping (Manual One-Hot for consistency) ---
    # The dataset has 'Role' column. We convert it to 'Role_Manager' (1 if Manager, 0 otherwise)
    # This matches the 'get_dummies' logic but is safer for production
    df['Role_Manager'] = df['Role'].apply(lambda x: 1 if 'Manager' in str(x) else 0)

    # --- Feature Selection ---
    # Select features that match the Streamlit app inputs
    # Note: Streamlit app asks for 'Previous CTC', 'Previous job change', 'Graduation Marks', 'EXP (Month)'
    feature_cols = ['College', 'City', 'Role_Manager', 'Previous CTC', 'Previous job change', 'Graduation Marks', 'EXP (Month)']
    
    # Drop rows with missing values in selected columns
    df = df.dropna(subset=feature_cols + ['CTC'])
    
    X = df[feature_cols]
    y = df['CTC']
    
    return X, y

def train_model(X, y, models_dir):
    """
    Trains the Random Forest model and saves artifacts.
    """
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Evaluate
    preds = model.predict(X_test_scaled)
    r2 = r2_score(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    print(f"Model Trained. R2 Score: {r2:.4f}, MSE: {mse:.2f}")

    # Save
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        
    joblib.dump(model, os.path.join(models_dir, 'random_forest_model.joblib'))
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.joblib'))
    print(f"Model and Scaler saved to {models_dir}")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    models_dir = os.path.join(base_dir, 'models')

    df, college_df, cities_df = load_data(data_dir)
    
    if df is not None:
        X, y = preprocess_data(df, college_df, cities_df)
        train_model(X, y, models_dir)

if __name__ == "__main__":
    main()