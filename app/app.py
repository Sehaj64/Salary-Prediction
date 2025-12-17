from flask import Flask, render_template, request, g
import joblib
import numpy as np
import os
import pandas as pd # Import pandas to use in app.py for mapping

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(APP_ROOT, '..', 'models')
DATA_PATH = "C:\\Users\\Preet\\Salary Data.csv" # Path to the original realistic data

def get_model():
    if 'model' not in g:
        model_path = os.path.join(MODELS_DIR, 'random_forest_model.joblib')
        g.model = joblib.load(model_path)
    return g.model

def get_scaler():
    if 'scaler' not in g:
        scaler_path = os.path.join(MODELS_DIR, 'scaler.joblib')
        g.scaler = joblib.load(scaler_path)
    return g.scaler

# Load the full dataset to get unique values for mapping
try:
    full_df = pd.read_csv(DATA_PATH)
    full_df.dropna(inplace=True)
    full_df = full_df[full_df['Salary'] > 1000] # Consistent with train.py cleaning
except FileNotFoundError:
    print(f"Error: {DATA_PATH} not found. Ensure the data file is in the specified path.")
    full_df = pd.DataFrame() # Create empty DataFrame to prevent further errors

# Precompute mappings for features from the original data
education_to_tier = {
    "Bachelor's": 1,
    "Master's": 2,
    "PhD": 3,
    "High School": 0,
    "Associate's Degree": 0
}

# Collect unique Job Titles and Education Levels for dropdowns if needed, though we map them internally
unique_education_levels = full_df['Education Level'].unique().tolist()
unique_job_titles = full_df['Job Title'].unique().tolist()

@app.route('/')
def home():
    return render_template('index.html',
                           education_levels=unique_education_levels,
                           job_titles=unique_job_titles)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get raw form data
        education_level_input = request.form['education_level']
        job_title_input = request.form['job_title']
        years_of_experience = float(request.form['years_of_experience'])
        previous_ctc = float(request.form['previous_ctc']) # Now directly from user
        previous_job_change = int(request.form['previous_job_change']) # Now directly from user
        graduation_marks = int(request.form['graduation_marks']) # Now directly from user
        
        # Map inputs to numerical values consistent with train.py
        college_tier = education_to_tier.get(education_level_input)
        exp_months = int(years_of_experience * 12)

        manager_keywords = ['Manager', 'Director', 'Head', 'Lead', 'VP', 'Chief', 'Principal']
        role = 1 if any(keyword.lower() in job_title_input.lower() for keyword in manager_keywords) else 0
        
        # For 'City', since original data doesn't have it, we randomly assigned it during training
        # For prediction, we need to provide a value. Let's use a default (e.g., 0 for Non-Metro)
        # or randomly assign to keep consistency with training's random assignment
        city = np.random.choice([0, 1]) # Randomly assign for prediction to match training's random assignment


        if college_tier is None:
            return render_template(
                'index.html',
                prediction_text=f'Invalid Education Level: {education_level_input}.',
                education_levels=unique_education_levels,
                job_titles=unique_job_titles
            )

        # Prepare features in the order the model expects:
        # 'College', 'City', 'Role', 'Previous CTC', 'Previous job change', 'Graduation Marks', 'EXP (Month)'
        features = np.array([[college_tier, city, role, previous_ctc,
                            previous_job_change, graduation_marks,
                            exp_months]])

        scaler = get_scaler()
        scaled_features = scaler.transform(features)

        model = get_model()
        prediction = model.predict(scaled_features)

        return render_template(
            'index.html',
            prediction_text='Predicted Monthly Salary: ₹ {:.2f} INR'.format(
                prediction[0]
            ),
            education_levels=unique_education_levels,
            job_titles=unique_job_titles
        )

    except (ValueError, KeyError) as e:
        return render_template(
            'index.html',
            prediction_text=f'Invalid input. Please check the values you have entered. Error: {e}',
            education_levels=unique_education_levels,
            job_titles=unique_job_titles
        )

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)