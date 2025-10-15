import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
import os

# -----------------------------
# File paths
# -----------------------------
TRAIN_PATH = r"C:\Users\hankerz\OneDrive\ml-ops-lab-2-task\train.csv"
MODEL_PATH = r"C:\Users\hankerz\OneDrive\ml-ops-lab-2-task\model.pkl"

# -----------------------------
# Load training data
# -----------------------------
if not os.path.exists(TRAIN_PATH):
    raise FileNotFoundError(f"Train file not found: {TRAIN_PATH}")

df = pd.read_csv(TRAIN_PATH)

X = df.drop("Outcome", axis=1)  # Features
y = df["Outcome"]               # Target

# -----------------------------
# Train model
# -----------------------------
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X, y)

# -----------------------------
# Save trained model
# -----------------------------
joblib.dump(model, MODEL_PATH)
print(f"Trained model saved to {MODEL_PATH}")
