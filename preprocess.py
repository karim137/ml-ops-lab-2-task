# preprocess.py

import pandas as pd
from sklearn.model_selection import train_test_split
import os

# -----------------------------
# File paths (Windows-safe)
# -----------------------------
RAW_DATA_PATH = r"C:\Users\hankerz\OneDrive\ml-ops-lab-2-task\diabetes.csv"
TRAIN_PATH = r"C:\Users\hankerz\OneDrive\ml-ops-lab-2-task\train.csv"
TEST_PATH = r"C:\Users\hankerz\OneDrive\ml-ops-lab-2-task\test.csv"

# -----------------------------
# Functions
# -----------------------------
def load_data(path=RAW_DATA_PATH):
    """Load raw CSV data safely."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    return df

def clean_data(df):
    """Clean data by dropping missing values."""
    df_cleaned = df.dropna()
    df_cleaned.reset_index(drop=True, inplace=True)
    return df_cleaned

def split_data(df, test_size=0.2, random_state=42):
    """Split the data into train and test sets."""
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    return train_df, test_df

def save_data(train_df, test_df, train_path=TRAIN_PATH, test_path=TEST_PATH):
    """Save train and test data to CSV."""
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"Saved train data to {train_path} and test data to {test_path}")

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    df = load_data()
    df_cleaned = clean_data(df)
    train_df, test_df = split_data(df_cleaned)
    save_data(train_df, test_df)
