import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os

if not os.path.exists('plots'):
    os.makedirs('plots')

df = pd.read_csv("data/ML case Study.csv")
college = pd.read_csv("data/Colleges.csv")
cities = pd.read_csv("data/cities.csv")

Tier1 = college["Tier 1"].tolist()
Tier2 = college["Tier 2"].tolist()
Tier3 = college["Tier 3"].tolist()

for item in df.College:
    if item in Tier1:
        df["College"].replace(item, 3, inplace=True)
    elif item in Tier2:
        df["College"].replace(item, 2, inplace=True)
    elif item in Tier3:
        df["College"].replace(item, 1, inplace=True)

metro = cities['Metrio City'].tolist()
non_metro_cities = cities['non-metro cities'].tolist()

for item in df.City:
    if item in metro:
        df['City'].replace(item, 1, inplace=True)
    elif item in non_metro_cities:
        df['City'].replace(item, 0, inplace=True)

df = pd.get_dummies(df, drop_first=True)

sns.boxplot(df['Previous CTC'])
plt.savefig('plots/previous_ctc_boxplot.png')
plt.clf()

sns.boxplot(df['Graduation Marks'])
plt.savefig('plots/graduation_marks_boxplot.png')
plt.clf()

sns.boxplot(df['EXP (Month)'])
plt.savefig('plots/exp_month_boxplot.png')
plt.clf()

sns.boxplot(df['CTC'])
plt.savefig('plots/ctc_boxplot.png')
plt.clf()

corr = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.savefig('plots/correlation_heatmap.png')
plt.clf()

X = df.loc[:, df.columns != 'CTC']
y = df['CTC']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
linear_reg_pred = linear_reg.predict(X_test)

print("Linear Regression:")
print("r2_score:", r2_score(y_test, linear_reg_pred))
print("MAE:", mean_absolute_error(y_test, linear_reg_pred))
print("MSE:", mean_squared_error(y_test, linear_reg_pred))
print("-" * 30)

ridge = Ridge()
ridge.fit(X_train, y_train)
ridge_predict = ridge.predict(X_test)

print("Ridge Regression:")
print("r2_score:", r2_score(y_test, ridge_predict))
print("MAE:", mean_absolute_error(y_test, ridge_predict))
print("MSE:", mean_squared_error(y_test, ridge_predict))
print("-" * 30)

lasso = Lasso()
lasso.fit(X_train, y_train)
lasso_predict = lasso.predict(X_test)

print("Lasso Regression:")
print("r2_score:", r2_score(y_test, lasso_predict))
print("MAE:", mean_absolute_error(y_test, lasso_predict))
print("MSE:", mean_squared_error(y_test, lasso_predict))
print("-" * 30)

print(
    "Analysis complete. All plots have been saved to the 'plots' directory."
)
