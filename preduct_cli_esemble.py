import joblib, pandas as pd

MODEL_PATH = "models/final_model.joblib"
clf = joblib.load(MODEL_PATH)

def ask(prompt, cast, default):
    s = input(f"{prompt} [{default}]: ").strip()
    return cast(s) if s else default

row = {
    "CreditScore":     ask("CreditScore (int)", int, 650),
    "Geography":       input("Geography (France/Germany/Spain) [France]: ").strip() or "France",
    "Gender":          input("Gender (Male/Female) [Male]: ").strip() or "Male",
    "Age":             ask("Age (int)", int, 40),
    "Tenure":          ask("Tenure (years, int)", int, 5),
    "Balance":         ask("Balance (float)", float, 90000.0),
    "NumOfProducts":   ask("NumOfProducts (int)", int, 2),
    "HasCrCard":       ask("HasCrCard (0/1)", int, 1),
    "IsActiveMember":  ask("IsActiveMember (0/1)", int, 1),
    "EstimatedSalary": ask("EstimatedSalary (float)", float, 60000.0),
}

# try to keep original training column order
expected_cols = ["CreditScore","Geography","Gender","Age","Tenure","Balance",
                 "NumOfProducts","HasCrCard","IsActiveMember","EstimatedSalary"]

X = pd.DataFrame([row])[expected_cols]
proba = clf.predict_proba(X)[0,1]
pred = int(proba >= 0.5)
label = "Churn" if pred==1 else "Stay"

print(f"\nModel: {MODEL_PATH}")
print(f"Prediction: {label}")
print(f"Probability of churn: {proba:.3f}")
