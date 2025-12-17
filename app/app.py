from flask import Flask, render_template, request, g
import joblib
import numpy as np
import os
import pandas as pd  # Import pandas for potential data loading

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(APP_ROOT, '..', 'models')
# DATA_PATH is no longer needed for dropdowns as inputs are numerical
# However, keeping it commented for reference or if needed for other purposes
# DATA_PATH = os.path.join(APP_ROOT, '..', 'data', 'Salary_Data.csv')


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


@app.route('/')
def home():
    # Render index.html without passing education_levels or job_titles
    # as inputs will now be numerical or directly mapped in the template.
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get numerical form data directly, consistent with ML-PROJECT
        college_tier = int(request.form['college_tier'])
        city_score = int(request.form['city_score'])
        role_manager = int(request.form['role_manager']) # 0 for Executive, 1 for Manager
        previous_ctc = float(request.form['previous_ctc'])
        previous_job_change = int(request.form['previous_job_change'])
        graduation_marks = int(request.form['graduation_marks'])
        exp_months = int(request.form['exp_months']) # Directly use months

    except (ValueError, KeyError) as e:
        return render_template(
            'index.html',
            prediction_text=f'Invalid input. Please ensure all fields are valid numbers. Error: {e}'
        )

    # Prepare features in the order the model expects:
    # 'College', 'City', 'Role_Manager', 'Previous CTC',
    # 'Previous job change', 'Graduation Marks', 'EXP (Month)'
    features = np.array([[college_tier, city_score, role_manager,
                          previous_ctc, previous_job_change,
                          graduation_marks, exp_months]])

    scaler = get_scaler()
    scaled_features = scaler.transform(features)

    model = get_model()
    prediction = model.predict(scaled_features)

    return render_template(
        'index.html',
        prediction_text='Predicted Monthly Salary: ₹ {:.2f} INR'.format(
            prediction[0]
        )
    )


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)