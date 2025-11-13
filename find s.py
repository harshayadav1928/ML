import numpy as np

# Step 1: Define the dataset (features + labels)
data = np.array([
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']
])

# Step 2: Separate features and target
X = data[:, :-1]
y = data[:, -1]

# Step 3: Initialize hypothesis with the most specific values
hypothesis = np.array(['0'] * X.shape[1])

# Step 4: Find-S Algorithm
for i, example in enumerate(X):
    if y[i] == 'Yes':  # Only consider positive examples
        if np.array_equal(hypothesis, np.array(['0'] * X.shape[1])):
            hypothesis = example.copy()
        else:
            for j in range(len(hypothesis)):
                if hypothesis[j] != example[j]:
                    hypothesis[j] = '?'

print("Final Hypothesis:", hypothesis)