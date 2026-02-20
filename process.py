# ================================
# HOUSE PRICE PREDICTION PROJECT
# ================================

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Load Dataset
data = pd.read_csv("Housing.csv")

# Step 3: Display First 5 Rows
print("Dataset Preview:")
print(data.head())

# Step 4: Check Missing Values
print("\nMissing Values:")
print(data.isnull().sum())

# Step 5: Convert Categorical Variables to Numerical
data = pd.get_dummies(data, drop_first=True)

# Step 6: Define Features and Target
X = data.drop("price", axis=1)
y = data["price"]

# Step 7: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================================
# MODEL 1: Linear Regression
# ================================

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print("\nLinear Regression Results:")
print("MSE:", mse_lr)
print("R2 Score:", r2_lr)

# ================================
# MODEL 2: Random Forest
# ================================

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("\nRandom Forest Results:")
print("MSE:", mse_rf)
print("R2 Score:", r2_rf)

# ================================
# Visualization
# ================================

plt.figure()
plt.scatter(y_test, y_pred_rf)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices (Random Forest)")
plt.show()