import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os

# --- Load Data ---
print("Loading data...")
df = pd.read_csv("data/ML case Study.csv")
college = pd.read_csv("data/Colleges.csv")
cities = pd.read_csv("data/cities.csv")

# --- Preprocessing ---
print("Preprocessing data...")

# Create a mapping from college name to tier
college_tier_map = {}
for college_name in college["Tier 1"].dropna():
    college_tier_map[college_name] = 3
for college_name in college["Tier 2"].dropna():
    college_tier_map[college_name] = 2
for college_name in college["Tier 3"].dropna():
    college_tier_map[college_name] = 1

# Apply the mapping
df['College'] = df['College'].replace(college_tier_map)

# Create a mapping from city name to type
city_type_map = {}
for city_name in cities["Metrio City"].dropna():
    city_type_map[city_name] = 1
for city_name in cities["non-metro cities"].dropna():
    city_type_map[city_name] = 0

# Apply the mapping
df['City'] = df['City'].replace(city_type_map)

# One-hot encode 'Role'
df = pd.get_dummies(df, columns=['Role'], drop_first=True, dtype=int)

# --- Feature and Target Split ---
X = df.drop('CTC', axis=1)
y = df['CTC']

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Feature Scaling ---
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Model Training ---
print("Training Random Forest model...")
# Using the best parameters found from GridSearchCV in the notebook
model = RandomForestRegressor(
    n_jobs=-1,
    max_features=4,
    min_samples_split=2
)
model.fit(X_train_scaled, y_train)

# --- Evaluate and Print Score ---
score = model.score(X_test_scaled, y_test)
print(f"Model trained with R-squared score: {score:.4f}")

# --- Save Model and Scaler ---
print("Saving model and scaler objects...")
# Create the models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

joblib.dump(model, "models/random_forest_model.joblib")
joblib.dump(scaler, "models/scaler.joblib")

print("Training complete. Model and scaler saved in 'models/' directory.")
