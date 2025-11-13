import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Step 1: Static dataset (2 features for easy plotting)
X = np.array([
    [1, 2],
    [2, 3],
    [3, 3],
    [6, 5],
    [7, 7],
    [8, 6],
    [9, 8],
    [5, 5],
])
y = np.array([0, 0, 0, 1, 1, 1, 1, 1])  # Two classes (0 and 1)

# Step 2: Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Step 3: KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Step 4: Predict
y_pred = knn.predict(X_test)

print("Actual:", y_test)
print("Predicted:", y_pred)

# Step 5: Simple plot
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', s=80, label='Train Data')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='coolwarm', s=150, marker='*', edgecolor='k', label='Predicted Test')
plt.title("K-Nearest Neighbors (K=3)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.show()