import pandas as pd
import pickle
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Load dataset
data = pd.read_csv("Housing.csv")

# Convert categorical to numerical
data = pd.get_dummies(data, drop_first=True)

# Features & Target
X = data.drop("price", axis=1)
y = data["price"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Model 1: Linear Regression
# -----------------------------
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

lr_r2 = r2_score(y_test, y_pred_lr)
lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))

lr_cv = cross_val_score(lr, X, y, cv=5, scoring='r2').mean()

# -----------------------------
# Model 2: Random Forest
# -----------------------------
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

rf_r2 = r2_score(y_test, y_pred_rf)
rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))

rf_cv = cross_val_score(rf, X, y, cv=5, scoring='r2').mean()

# Save everything
model_data = {
    "model": rf,  # Best model
    "columns": X.columns,
    "metrics": {
        "Linear Regression": {
            "R2": lr_r2,
            "RMSE": lr_rmse,
            "CrossVal_R2": lr_cv
        },
        "Random Forest": {
            "R2": rf_r2,
            "RMSE": rf_rmse,
            "CrossVal_R2": rf_cv
        }
    }
}

pickle.dump(model_data, open("house_model.pkl", "wb"))

print("Model trained with comparison & cross validation.")