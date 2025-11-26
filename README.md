# ML Final Project: Employee Salary Prediction

## Project Overview

This project focuses on building a machine learning model to predict employee salaries. It demonstrates a complete machine learning workflow, from synthetic data generation and preprocessing to model training, evaluation, and hyperparameter tuning. The goal is to identify key factors influencing salary and provide insights into salary prediction.

## Problem Statement

TechWorks Consulting, an IT talent recruitment company, aims to accurately match skilled IT professionals with job opportunities by predicting employee salaries. This project performs a regression task to predict the continuous variable of newly hired employees' salaries. Key aspects include:

-   **Context and Company Background:** TechWorks Consulting specializes in IT talent recruitment.
-   **Data Description:** The dataset contains information about colleges, cities, roles, previous experience, and salary, used for model training and testing.
-   **Regression Task:** The primary objective is to predict a continuous variable (employee salary).
-   **Role of Statistics:** Statistics are crucial for building and verifying model accuracy.
-   **Data Preprocessing:** Involves handling missing values, outliers, categorical variables, normalization, and feature selection.

## Dataset

The project utilizes a synthetically generated dataset that simulates real-world employee data. This approach ensures reproducibility and allows for a clear demonstration of the machine learning pipeline, even without access to original proprietary data.

The `data` directory contains the following CSV files:
-   `ML case Study.csv`: Main dataset containing employee information.
-   `Colleges.csv`: Information about various colleges, categorized by tiers.
-   `cities.csv`: Categorization of cities into metropolitan and non-metropolitan.

**Note on Data:** The original data files for this project (`ML case Study.csv`, `Colleges.csv`, `cities.csv`) were not available. This notebook has been set up to run with **automatically generated placeholder data** that matches the expected schema. While the code demonstrates the full machine learning workflow, the analytical insights and model performance metrics are based on this synthetic data and should not be interpreted as findings from real-world data.

## Project Structure

```
ML-Final-Project/
├── data/
│   ├── cities.csv
│   ├── Colleges.csv
│   └── ML case Study.csv
├── .gitignore
├── generate_data.py
├── modify_notebook.py
├── requirements.txt
└── salary_prediction_analysis.ipynb
```

-   `data/`: Contains the datasets used in the project.
-   `.gitignore`: Specifies intentionally untracked files to ignore.
-   `generate_data.py`: Script to generate synthetic datasets (`ML case Study.csv`, `Colleges.csv`, `cities.csv`).
-   `modify_notebook.py`: Utility script to adjust notebook paths and add data notes.
-   `requirements.txt`: Lists all Python dependencies required to run the project.
-   `salary_prediction_analysis.ipynb`: The main Jupyter Notebook detailing the data analysis, preprocessing, model building, and evaluation.

## Getting Started

Follow these instructions to set up and run the project locally.

### Prerequisites

-   Python 3.x
-   `pip` (Python package installer)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url> # Replace <repository_url> with your GitHub repository URL
    cd ML-Final-Project
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Project

1.  **Generate Synthetic Data:**
    First, generate the necessary data files:
    ```bash
    python generate_data.py
    ```
    This will create `ML case Study.csv`, `Colleges.csv`, and `cities.csv` in the `data/` directory.

2.  **Run the Jupyter Notebook:**
    Launch Jupyter Lab or Jupyter Notebook and open `salary_prediction_analysis.ipynb`:
    ```bash
    jupyter lab
    # or
    jupyter notebook
    ```
    Follow the steps in the notebook to execute the data analysis, preprocessing, and model training.

## Key Findings and Model Performance

The notebook explores several machine learning models for salary prediction, including Linear Regression, Ridge, Lasso, Decision Tree, and Random Forest. Key observations include:

-   **Random Forest** consistently performs well across various scenarios, achieving the highest R-squared scores and indicating a strong fit to the data.
-   **Linear Regression and Lasso** also show good performance, albeit slightly lower than Random Forest.
-   **Decision Tree** models, without ensemble methods, generally exhibit lower performance.
-   **Feature Scaling** (StandardScaler) positively impacts model performance, leading to improved R-squared scores in scenarios where it was applied.

Overall, Random Forest is identified as the top-performing model for this dataset, with R-squared scores reaching approximately 0.67-0.68 in optimized scenarios.

## Future Improvements

To further enhance the model's performance and robustness:

-   **Increase the Number of Trees (Estimators):** Experiment with more trees in the Random Forest model.
-   **Hyperparameter Tuning:** Conduct a more thorough tuning process for parameters like `max_depth`, `min_samples_split`, `min_samples_leaf`, and `max_features` using techniques like Grid Search or Randomized Search.
-   **Feature Engineering/Selection:** Explore creating new features or refining existing ones, and consider more advanced feature selection techniques to optimize the model.
-   **Advanced Models:** Investigate other ensemble methods (e.g., Gradient Boosting, XGBoost) or even deep learning approaches if data complexity and volume warrant it.
-   **Cross-Validation:** Implement more robust cross-validation strategies to ensure model generalization.

## Technologies Used

-   Python 3.x
-   pandas
-   numpy
-   seaborn
-   scikit-learn
-   jupyter / ipykernel

## License

This project is licensed under the MIT License. See the LICENSE file for details. (Assuming MIT license, if a different license is preferred, please specify.)
