import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Step 1: Define static dataset
X = np.array([
    [1, 2],
    [1.5, 1.8],
    [5, 8],
    [8, 8],
    [1, 0.6],
    [9, 11],
    [8, 2],
    [10, 2],
    [9, 3]
])

# Step 2: Create and fit KMeans model
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Step 3: Get cluster labels and centroids
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

print("Cluster Labels:", labels)
print("Centroids:\n", centroids)

# Step 4: Simple Plot
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=80, label='Data Points')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='X', label='Centroids')
plt.title('K-Means Clustering (k=3)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()