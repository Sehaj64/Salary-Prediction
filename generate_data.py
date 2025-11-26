import pandas as pd
import numpy as np
import os

# Create the data directory if it doesn't exist
if not os.path.exists('data'):
    os.makedirs('data')

# --- Generate ML case Study.csv ---
num_samples = 1600
roles = ['Executive', 'Manager']
colleges = [
    'SVNIT Surat', 'NIT Bhopal', 'IEM, Kolkata', 'KIIT, Bhubaneswar', 'DTU',
    'IIT Bombay', 'IIT Delhi', 'IIT Kharagpur', 'IIT Madras', 'IIT Kanpur',
    'IIIT Bangalore', 'IIIT Delhi', 'IGDTUW', 'NIT Calicut', 'IIITM Gwalior',
    'Ramaiah Institute of Technology, Bengaluru', 'TIET/Thapar University', 'Manipal Main Campus', 'VIT Vellore', 'SRM Main Campus'
]
cities = [
    'Asansol', 'Ajmer', 'Rajpur Sonarpur', 'Durgapur', 'Mumbai',
    'Delhi', 'Kolkata', 'Chennai', 'Bangalore', 'Dehradun',
    'Rourkela', 'Kozhikode', 'Hyderabad'
]

data = {
    'College': np.random.choice(colleges, num_samples),
    'City': np.random.choice(cities, num_samples),
    'Role': np.random.choice(roles, num_samples, p=[0.8, 0.2]),
    'Previous CTC': np.random.uniform(40000, 75000, num_samples).round(2),
    'Previous job change': np.random.randint(1, 5, num_samples),
    'Graduation Marks': np.random.randint(35, 86, num_samples),
    'EXP (Month)': np.random.randint(18, 65, num_samples),
    'CTC': np.random.uniform(55000, 120000, num_samples).round(2)
}
df = pd.DataFrame(data)
df.to_csv('C:/Users/Preet/ML-Final-Project/data/ML case Study.csv', index=False)

# --- Generate Colleges.csv ---
tier1 = ['IIT Bombay', 'IIT Delhi', 'IIT Kharagpur', 'IIT Madras', 'IIT Kanpur', 'IIT Roorkee', 'IIT Guwahati', 'IIIT Hyderabad', 'BITS Pilani (Pilani Campus)', 'IIT Indore', 'IIT Ropar', 'IIT BHU (Varanasi)', 'IIT ISM Dhanbad', 'DTU', 'NSUT Delhi (NSIT)', 'NIT Tiruchipally (Trichy)', 'NIT Warangal', 'NIT Surathkal (Karnataka)', 'Jadavpur University', 'BITS Pilani (Hyderabad Campus)', 'BITS Pilani (Goa Campus)', 'IIIT Allahabad']
tier2 = ['IIIT Bangalore', 'IIIT Delhi', 'IGDTUW', 'NIT Calicut', 'IIITM Gwalior', 'SVNIT Surat', 'NIT Bhopal']
tier3 = ['IEM, Kolkata', 'KIIT, Bhubaneswar', 'Ramaiah Institute of Technology, Bengaluru', 'TIET/Thapar University', 'Manipal Main Campus', 'VIT Vellore', 'SRM Main Campus']

max_len = max(len(tier1), len(tier2), len(tier3))
tier1.extend([np.nan] * (max_len - len(tier1)))
tier2.extend([np.nan] * (max_len - len(tier2)))
tier3.extend([np.nan] * (max_len - len(tier3)))

colleges_df = pd.DataFrame({
    'Tier 1': tier1,
    'Tier 2': tier2,
    'Tier 3': tier3
})
colleges_df.to_csv('C:/Users/Preet/ML-Final-Project/data/Colleges.csv', index=False)

# --- Generate cities.csv ---
metro_cities = ['Mumbai', 'Delhi', 'Kolkata', 'Chennai', 'Bangalore', 'Hyderabad']
non_metro_cities = ['Asansol', 'Ajmer', 'Rajpur Sonarpur', 'Durgapur', 'Dehradun', 'Rourkela', 'Kozhikode']

max_len = max(len(metro_cities), len(non_metro_cities))
metro_cities.extend([np.nan] * (max_len - len(metro_cities)))
non_metro_cities.extend([np.nan] * (max_len - len(non_metro_cities)))

cities_df = pd.DataFrame({
    'Metrio City': metro_cities,
    'non-metro cities': non_metro_cities
})
cities_df.to_csv('C:/Users/Preet/ML-Final-Project/data/cities.csv', index=False)

print("Synthetic data generated successfully.")
