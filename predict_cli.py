import joblib
import numpy as np
import pandas as pd

MODEL_PATH = "models/baseline_lr.joblib"
clf = joblib.load(MODEL_PATH)

print("Enter customer profile (press Enter for default in brackets):")

def ask_int(prompt, default):
    s = input(f"{prompt} [{default}]: ").strip()
    return int(s) if s else default

def ask_float(prompt, default):
    s = input(f"{prompt} [{default}]: ").strip()
    return float(s) if s else default

def ask_cat(prompt, default, choices=None):
    s = input(f"{prompt} [{default}]: ").strip() or default
    if choices and s not in choices:
        print(f"  -> '{s}' not in {choices}. Using default '{default}'.")
        s = default
    return s

# Collect inputs
row = {
    "CreditScore":     ask_int("CreditScore (int)", 650),
    "Geography":       ask_cat("Geography (France/Germany/Spain)", "France",
                               ["France","Germany","Spain"]),
    "Gender":          ask_cat("Gender (Male/Female)", "Male", ["Male","Female"]),
    "Age":             ask_int("Age (int)", 40),
    "Tenure":          ask_int("Tenure (years, int)", 5),
    "Balance":         ask_float("Balance (float)", 90000.0),
    "NumOfProducts":   ask_int("NumOfProducts (int)", 2),
    "HasCrCard":       ask_int("HasCrCard (0/1)", 1),
    "IsActiveMember":  ask_int("IsActiveMember (0/1)", 1),
    "EstimatedSalary": ask_float("EstimatedSalary (float)", 60000.0),
}

# Build 1-row DataFrame and align columns to what the pipeline was trained on
try:
    expected_cols = list(clf.feature_names_in_)  # scikit-learn >=1.0
except AttributeError:
    # Fallback: the columns used in train_baseline.py before encoding
    expected_cols = ["CreditScore","Geography","Gender","Age","Tenure",
                     "Balance","NumOfProducts","HasCrCard","IsActiveMember",
                     "EstimatedSalary"]

X = pd.DataFrame([row])[expected_cols]

# Predict
proba = clf.predict_proba(X)[0, 1]
pred = int(proba >= 0.5)
label = "Churn" if pred == 1 else "Stay"

print(f"\nPrediction: {label}")
print(f"Probability of churn: {proba:.3f}")