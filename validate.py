# validate.py

import pandas as pd
import joblib
import os
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# -----------------------------
# File paths
# -----------------------------
TEST_PATH = r"C:\Users\hankerz\OneDrive\ml-ops-lab-2-task\test.csv"
MODEL_PATH = r"C:\Users\hankerz\OneDrive\ml-ops-lab-2-task\model.pkl"
METRICS_PATH = r"C:\Users\hankerz\OneDrive\ml-ops-lab-2-task\metrics.json"
CM_PLOT_PATH = r"C:\Users\hankerz\OneDrive\ml-ops-lab-2-task\confusion_matrix.png"

# -----------------------------
# Load test data
# -----------------------------
if not os.path.exists(TEST_PATH):
    raise FileNotFoundError(f"Test file not found: {TEST_PATH}")
    
df_test = pd.read_csv(TEST_PATH)
X_test = df_test.drop("Outcome", axis=1)
y_test = df_test["Outcome"]

# -----------------------------
# Load trained model
# -----------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    
model = joblib.load(MODEL_PATH)

# -----------------------------
# Make predictions
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# Compute metrics
# -----------------------------
metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1_score": f1_score(y_test, y_pred)
}

# Save metrics to JSON
with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=4)

print(f"Metrics saved to {METRICS_PATH}")

# -----------------------------
# Save confusion matrix plot
# -----------------------------
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig(CM_PLOT_PATH)
plt.close()
print(f"Confusion matrix plot saved to {CM_PLOT_PATH}")
