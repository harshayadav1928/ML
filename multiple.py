import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# --- Sample dataset ---
data = {
    'area': [1200, 1500, 1800, 2000, 2400, 3000, 3500],
    'bedrooms': [2, 3, 3, 3, 4, 4, 5],
    'age': [10, 8, 5, 3, 2, 1, 4],
    'price': [200000, 250000, 320000, 350000, 400000, 480000, 550000]
}

df = pd.DataFrame(data)

# --- Split into features (X) and target (y) ---
X = df[['area', 'bedrooms', 'age']]
y = df['price']

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train Model ---
model = LinearRegression()
model.fit(X_train, y_train)

# --- Predict ---
y_pred = model.predict(X_test)

# --- Evaluate ---
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("R² Score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# --- Predict new house ---
new_house = np.array([[2500, 4, 5]])  # [area, bedrooms, age]
predicted_price = model.predict(new_house)
print(f"Predicted Price for new house: ${predicted_price[0]:.2f}")
