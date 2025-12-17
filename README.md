# Salary Prediction App

## Project Overview
This project is a Machine Learning application that predicts employee salaries based on various features such as Education Level, Job Title, and Years of Experience. It features both a **Flask** web interface and a newly added **Streamlit** dashboard for easy cloud deployment.

## Features
-   **Dual Interface:**
    -   **Streamlit App:** Modern, interactive dashboard hosted on Streamlit Cloud.
    -   **Flask App:** Traditional web interface for local deployment.
-   **Salary Prediction:** Uses a Random Forest Regressor to estimate monthly salary.
-   **Data Analysis:** Analyzes trends based on experience, education, and role.

## Live Demo
Check out the live application on Streamlit Cloud:
[**Salary Prediction App**](https://salary-prediction-n7vjrtmhjvwbh4u9shbcun.streamlit.app/)

## Technologies Used
-   **Frontend:** Streamlit, HTML/CSS (Flask)
-   **Backend:** Python
-   **Machine Learning:** Scikit-learn (RandomForestRegressor)
-   **Data Processing:** Pandas, NumPy

## Setup and Usage

### Prerequisites
-   Python 3.8+
-   `pip`

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Sehaj64/Salary-Prediction.git
    cd Salary-Prediction
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the App

**Option 1: Streamlit (Recommended)**
```bash
streamlit run streamlit_app.py
```

**Option 2: Flask**
```bash
python app/app.py
```

## Model Training
If you need to retrain the model with new data:
1.  Place your data in `data/Salary_Data.csv`.
2.  Run the training script:
    ```bash
    python train.py --data_path "data/Salary_Data.csv"
    ```

## Project Structure
```
.
├── streamlit_app.py        # Main Streamlit application
├── app/
│   ├── templates/          # HTML templates for Flask
│   └── app.py              # Flask application logic
├── data/
│   └── Salary_Data.csv     # Dataset used for training
├── models/
│   ├── random_forest_model.joblib  # Trained Model
│   └── scaler.joblib               # Feature Scaler
├── train.py                # Model training script
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```