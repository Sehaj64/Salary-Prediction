from flask import Flask, render_template, request, url_for, current_app, g
import joblib
import numpy as np
import os

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(APP_ROOT, '..', 'models')

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
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        college = int(request.form['college'])
        city = int(request.form['city'])
        role = int(request.form['role'])
        previous_ctc = float(request.form['previous_ctc'])
        previous_job_change = int(request.form['previous_job_change'])
        graduation_marks = int(request.form['graduation_marks'])
        exp_months = int(request.form['exp_months'])
    except ValueError:
        return render_template(
            'index.html',
            prediction_text='Invalid input. Please enter numeric values.'
        )

    features = np.array([[college, city, previous_ctc,
                        previous_job_change, graduation_marks,
                        exp_months, role]])

    scaler = get_scaler()
    scaled_features = scaler.transform(features)

    model = get_model()
    prediction = model.predict(scaled_features)

    return render_template(
        'index.html',
        prediction_text='Predicted Monthly Salary: ₹ {:.2f}'.format(
            prediction[0]
        )
    )


@app.route('/analysis')
def analysis():
    plot_dir = current_app.static_folder
    plots = [
        url_for('static', filename=f)
        for f in os.listdir(plot_dir)
        if f.endswith('.png')
    ]
    return render_template('analysis.html', plots=plots)


if __name__ == '__main__':
    app.run(host="0.0.0.0",debug=True)
    
