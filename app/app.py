from flask import Flask, render_template, request, url_for
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('models/random_forest_model.joblib')
scaler = joblib.load('models/scaler.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()   # or request.get_json(force=True)
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    

def predict():
    # Get the data from the form
    college = int(request.form['college'])
    city = int(request.form['city'])
    role = int(request.form['role'])
    previous_ctc = float(request.form['previous_ctc'])
    previous_job_change = int(request.form['previous_job_change'])
    graduation_marks = int(request.form['graduation_marks'])
    exp_months = int(request.form['exp_months'])

    # Create a feature array for prediction
    features = np.array([[college, city, previous_ctc, previous_job_change, graduation_marks, exp_months, role]])

    # Scale the features
    scaled_features = scaler.transform(features)

    # Make a prediction
    prediction = model.predict(scaled_features)

    return render_template('index.html', prediction_text='Predicted Salary: ₹ {:.2f}'.format(prediction[0]))

@app.route('/analysis')
def analysis():
    # Get the list of plots
    plot_dir = 'app/static'
    plots = [url_for('static', filename=f) for f in os.listdir(plot_dir) if f.endswith('.png')]
    return render_template('analysis.html', plots=plots)

if __name__ == '__main__':
    app.run(debug=True)
