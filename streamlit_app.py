import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set page configuration
st.set_page_config(page_title="Salary Prediction App", layout="centered")

# Define paths
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(APP_ROOT, 'models')
DATA_DIR = os.path.join(APP_ROOT, 'data')

# Load Model and Scaler
@st.cache_resource
def load_model_and_scaler():
    model_path = os.path.join(MODELS_DIR, 'random_forest_model.joblib')
    scaler_path = os.path.join(MODELS_DIR, 'scaler.joblib')
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_model_and_scaler()

# Load Data for Dropdowns
@st.cache_data
def load_dropdown_data():
    try:
        colleges_df = pd.read_csv(os.path.join(DATA_DIR, 'Colleges.csv'))
        cities_df = pd.read_csv(os.path.join(DATA_DIR, 'cities.csv'))
        return colleges_df, cities_df
    except FileNotFoundError:
        return None, None

colleges_df, cities_df = load_dropdown_data()

st.title("Salary Prediction App")
st.write("Predict the monthly CTC based on candidate profile.")

if model is not None and scaler is not None and colleges_df is not None:
    with st.form("prediction_form"):
        # 1. College Input
        # Get all colleges from Tier 1, 2, 3
        tier1 = colleges_df["Tier 1"].dropna().tolist()
        tier2 = colleges_df["Tier 2"].dropna().tolist()
        tier3 = colleges_df["Tier 3"].dropna().tolist()
        all_colleges = sorted(tier1 + tier2 + tier3)
        
        college_input = st.selectbox("College", all_colleges)
        
        # 2. City Input
        metro = cities_df["Metro City"].dropna().tolist()
        non_metro = cities_df["non-metro cities"].dropna().tolist()
        all_cities = sorted(metro + non_metro)
        
        city_input = st.selectbox("City", all_cities)
        
        # 3. Role Input
        role_input = st.selectbox("Role", ["Manager", "Executive"])
        
        # 4. Numerical Inputs
        previous_ctc = st.number_input("Previous CTC (Annual)", min_value=0.0, step=1000.0, value=50000.0)
        previous_job_change = st.number_input("Previous Job Changes", min_value=0, max_value=20, step=1, value=0)
        graduation_marks = st.slider("Graduation Marks (%)", 0, 100, 60)
        exp_years = st.number_input("Years of Experience", min_value=0.0, max_value=50.0, step=0.1, value=2.0)
        
        submitted = st.form_submit_button("Predict Salary")

    if submitted:
        # Preprocessing inputs to match training logic
        
        # College Mapping
        college_tier = 1 # Default
        if college_input in tier1: college_tier = 3
        elif college_input in tier2: college_tier = 2
        elif college_input in tier3: college_tier = 1
        
        # City Mapping
        city_score = 0 # Default non-metro
        if city_input in metro: city_score = 1
        
        # Role Mapping
        role_manager = 1 if role_input == "Manager" else 0
        
        # Experience to Months
        exp_months = int(exp_years * 12)
        
        # Feature Array: ['College', 'City', 'Role_Manager', 'Previous CTC', 'Previous job change', 'Graduation Marks', 'EXP (Month)']
        features = np.array([[college_tier, city_score, role_manager, previous_ctc, previous_job_change, graduation_marks, exp_months]])
        
        # Scale
        scaled_features = scaler.transform(features)
        
        # Predict
        prediction = model.predict(scaled_features)[0]
        
        st.success(f"Estimated Monthly CTC: ₹ {prediction:,.2f}")

else:
    st.error("Model files or Data files missing. Please run 'train.py' locally first.")