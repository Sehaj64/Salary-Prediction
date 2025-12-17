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
DATA_PATH = os.path.join(APP_ROOT, 'data', 'Salary_Data.csv')

# Load Model and Scaler with caching
@st.cache_resource
def load_model_and_scaler():
    model_path = os.path.join(MODELS_DIR, 'random_forest_model.joblib')
    scaler_path = os.path.join(MODELS_DIR, 'scaler.joblib')
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        st.error("Model or Scaler not found. Please run 'train.py' locally first to generate them.")
        return None, None
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_model_and_scaler()

# Load data for dropdowns
@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        df.dropna(inplace=True)
        return df
    else:
        st.error(f"Data file not found at {DATA_PATH}")
        return pd.DataFrame(columns=['Education Level', 'Job Title'])

df = load_data()

# Title and Description
st.title("Salary Prediction App")
st.write("Enter your details below to predict your estimated monthly salary.")

if not df.empty and model is not None and scaler is not None:
    # Input Form
    with st.form("prediction_form"):
        # Education Level
        unique_education_levels = sorted(df['Education Level'].unique().tolist())
        education_level = st.selectbox("Education Level", unique_education_levels)
        
        # Job Title
        unique_job_titles = sorted(df['Job Title'].unique().tolist())
        job_title = st.selectbox("Job Title", unique_job_titles)
        
        # Years of Experience
        years_of_experience = st.number_input("Years of Experience", min_value=0.0, max_value=50.0, step=0.1, value=1.0)
        
        # Previous CTC
        previous_ctc = st.number_input("Previous CTC (Annual)", min_value=0.0, step=1000.0, value=50000.0)
        
        # Previous Job Changes
        previous_job_change = st.number_input("Previous Job Changes", min_value=0, max_value=20, step=1, value=0)
        
        # Graduation Marks
        graduation_marks = st.slider("Graduation Marks (%)", min_value=0, max_value=100, value=60)
        
        # Submit Button
        submitted = st.form_submit_button("Predict Salary")

    if submitted:
        # Preprocessing Logic (Matching app.py and train.py)
        
        # Map Education to Tier
        education_to_tier = {
            "Bachelor's": 1,
            "Master's": 2,
            "PhD": 3,
            "High School": 0,
            "Associate's Degree": 0
        }
        college_tier = education_to_tier.get(education_level, 0)
        
        # Calculate Exp Months
        exp_months = int(years_of_experience * 12)
        
        # Determine Role
        manager_keywords = ['Manager', 'Director', 'Head', 'Lead', 'VP', 'Chief', 'Principal']
        role = 1 if any(keyword.lower() in job_title.lower() for keyword in manager_keywords) else 0
        
        # City (Random assignment as per app.py logic)
        city = np.random.choice([0, 1])
        
        # Prepare Feature Array
        # Order: 'College', 'City', 'Role', 'Previous CTC', 'Previous job change', 'Graduation Marks', 'EXP (Month)'
        features = np.array([[college_tier, city, role, previous_ctc,
                              previous_job_change, graduation_marks,
                              exp_months]])
        
        # Scale Features
        scaled_features = scaler.transform(features)
        
        # Predict
        prediction = model.predict(scaled_features)
        
        # Display Result
        st.success(f"Predicted Monthly Salary: ₹ {prediction[0]:,.2f}")
        
else:
    st.warning("Application cannot run because dependencies (Model/Data) are missing.")
