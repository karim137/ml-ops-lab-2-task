

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

TRAIN_PATH = r"C:\Users\hankerz\OneDrive\ml-ops-lab-2-task\train.csv"
MODEL_PATH = r"C:\Users\hankerz\OneDrive\ml-ops-lab-2-task\model.pkl"

if not os.path.exists(TRAIN_PATH):
    raise FileNotFoundError(f"Train file not found: {TRAIN_PATH}")

df = pd.read_csv(TRAIN_PATH)

X = df.drop("Outcome", axis=1)  
y = df["Outcome"]              


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# -----------------------------
# Save trained model
# -----------------------------
joblib.dump(model, MODEL_PATH)
print(f"Trained model saved to {MODEL_PATH}")
