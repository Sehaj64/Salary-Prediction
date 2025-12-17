import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os
import argparse
import numpy as np

def load_data(data_path):
    """
    Loads the main dataset from the given path.

    Args:
        data_path (str): Path to the main data CSV file (Salary Data.csv).

    Returns:
        pd.DataFrame: The loaded pandas DataFrame.
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    return df

def preprocess_data(df):
    """
    Performs preprocessing and feature engineering on the data.

    Args:
        df (pd.DataFrame): The main DataFrame.

    Returns:
        tuple: A tuple containing the preprocessed features (X) and target (y).
    """
    print("Preprocessing data...")
    
    # --- Data Cleaning ---
    df.dropna(inplace=True)
    df = df[df['Salary'] > 1000] # Remove outlier
    
    # --- Feature Engineering ---
    
    # Map Education Level to College Tier (1, 2, 3)
    # Bachelor's -> 1, Master's -> 2, PhD -> 3 (Aligns with higher education = higher tier)
    education_to_tier = {
        "Bachelor's": 1,
        "Master's": 2,
        "PhD": 3,
        "High School": 0, # Assuming no specific tier, treat as lowest
        "Associate's Degree": 0 # Assuming no specific tier, treat as lowest
    }
    df['College'] = df['Education Level'].map(education_to_tier)
    # Drop rows where education level couldn't be mapped
    df.dropna(subset=['College'], inplace=True)
    df['College'] = df['College'].astype(int)

    # Convert Years of Experience to EXP (Month)
    df['EXP (Month)'] = (df['Years of Experience'] * 12).astype(int)

    # Simplify Job Title to Role (Manager/Executive) - heuristic approach
    manager_keywords = ['Manager', 'Director', 'Head', 'Lead', 'VP', 'Chief', 'Principal']
    df['Role'] = df['Job Title'].apply(
        lambda x: 1 if any(keyword.lower() in x.lower() for keyword in manager_keywords) else 0
    )
    
    # Synthesize 'Previous CTC' based on current Salary and Years of Experience (if needed for model)
    # Assuming previous CTC is generally lower than current salary, and increases with experience
    # This logic assumes 'Salary' is the current salary and we are deriving a 'Previous CTC'
    df['Previous CTC'] = df['Salary'] / (1 + (df['Years of Experience'] * np.random.uniform(0.02, 0.08))) # Simulate annual raise
    df['Previous CTC'] = df['Previous CTC'].round(2)
    
    # Synthesize 'Previous job change' based on Years of Experience
    # More experience -> potentially more job changes
    df['Previous job change'] = df['Years of Experience'].apply(
        lambda x: np.random.randint(0, max(1, int(x / 5)))
    ).astype(int)
    
    # Synthesize 'Graduation Marks' - assumed to be somewhat correlated with Education Level
    # Random within a range for each education level
    def get_graduation_marks(education_level):
        if education_level == "Bachelor's":
            return np.random.randint(60, 75)
        elif education_level == "Master's":
            return np.random.randint(70, 85)
        elif education_level == "PhD":
            return np.random.randint(75, 90)
        else: # High School, Associate's Degree
            return np.random.randint(40, 60)
            
    df['Graduation Marks'] = df['Education Level'].apply(get_graduation_marks)
    df['Graduation Marks'] = df['Graduation Marks'].apply(
        lambda x: x + np.random.randint(-5, 5)
    ) # Add some noise
    df['Graduation Marks'] = df['Graduation Marks'].clip(0, 100).astype(int) # Ensure marks are within 0-100

    # City (simplification: assign based on Job Title or randomly)
    # For simplicity and to fit existing model input, let's randomly assign Metro/Non-Metro
    df['City'] = np.random.choice([0, 1], size=len(df)) # 0 for Non-Metro, 1 for Metro

    # Ensure all expected columns are present, even if no 'Manager' roles exist (from one-hot encoding logic)
    # Here, 'Role' is already 0 or 1, so no one-hot encoding needed for the model input directly
    
    # Select final features for the model, matching the expected input of the Flask app
    # The Flask app expects: College, City, Role, Previous CTC, Previous job change, Graduation Marks, EXP (Month)
    # Ensure the order and names match
    X = df[[
        'College', 'City', 'Role', 'Previous CTC',
        'Previous job change', 'Graduation Marks', 'EXP (Month)'
    ]]
    y = df['Salary']
    
    return X, y

def train_model(X, y, model_path, scaler_path):
    """
    Trains the RandomForestRegressor model and saves it and the scaler to files.

    Args:
        X (pd.DataFrame): The features.
        y (pd.Series): The target.
        model_path (str): The path to save the model to.
        scaler_path (str): The path to save the scaler to.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training Random Forest Regressor model...")
    model = RandomForestRegressor(
        n_estimators=100, # Increased n_estimators for better performance
        max_features=0.8, # Use a fraction of features for better generalization
        min_samples_leaf=5, # Ensure each leaf has at least 5 samples
        random_state=42, # Added for reproducibility
        n_jobs=-1 # Use all available cores
    )
    model.fit(X_train_scaled, y_train)

    score = model.score(X_test_scaled, y_test)
    print(f"Model trained with R-squared score: {score:.4f}")

    print("Saving model and scaler objects...")
    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    print(f"Training complete. Model saved in '{model_path}' and scaler in '{scaler_path}'")

def main():
    """
    Main function to run the training script.
    """
    parser = argparse.ArgumentParser(description="Train a salary prediction model.")
    parser.add_argument("--data_path", type=str, default="C:\\Users\\Preet\\Salary Data.csv", help="Path to the main data CSV file (Salary Data.csv).")
    parser.add_argument("--model_path", type=str, default="models/random_forest_model.joblib", help="Path to save the trained model.")
    parser.add_argument("--scaler_path", type=str, default="models/scaler.joblib", help="Path to save the scaler object.")
    args = parser.parse_args()

    df = load_data(args.data_path)
    X, y = preprocess_data(df)
    train_model(X, y, args.model_path, args.scaler_path)

if __name__ == "__main__":
    main()
