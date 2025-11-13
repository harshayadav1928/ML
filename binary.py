# Binary Classification using Logistic Regression

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

# Step 1: Create simple dataset
# Features: [age, salary]
X = np.array([
    [25, 50000],
    [30, 60000],
    [35, 65000],
    [40, 70000],
    [45, 80000],
    [50, 90000],
    [23, 48000],
    [33, 62000],
    [48, 85000],
    [52, 95000]
])

# Labels: 0 = Not Purchased, 1 = Purchased
y = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1])

# Step 2: Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 4: Predict on test data
y_pred = model.predict(X_test)

# Step 5: Evaluate performance
print("Predictions:", y_pred)
print("Actual Labels:", y_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
import matplotlib.pyplot as plt

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.title('Binary Classification (Logistic Regression)')
plt.show()