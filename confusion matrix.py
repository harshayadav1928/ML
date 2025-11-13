from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Step 1: Define actual and predicted values manually
actual =    [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]  
predicted = [1, 0, 1, 0, 0, 1, 1, 0, 1, 0]

# Step 2: Compute confusion matrix
cm = confusion_matrix(actual, predicted)
print("CONFUSION MATRIX:\n", cm)

# Step 3: Extract parameters (TN, FP, FN, TP)
TN, FP, FN, TP = cm.ravel()

print(f"\nTrue Negatives (TN): {TN}")
print(f"False Positives (FP): {FP}")
print(f"False Negatives (FN): {FN}")
print(f"True Positives (TP): {TP}")

# Step 4: Calculate metrics manually
accuracy  = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
f1_score  = 2 * (precision * recall) / (precision + recall)

print("\n---- METRICS ----")
print(f"Accuracy:  {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall:    {recall:.2f}")
print(f"F1 Score:  {f1_score:.2f}")

# Step 5: Visualize confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Class 0", "Class 1"])
disp.plot(cmap="Oranges")
plt.title("Confusion Matrix with Metrics")
plt.show()