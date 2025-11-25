import os, json, joblib, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score, RocCurveDisplay)

RAW = "churn.csv"
FIGDIR = "reports/figures"
MODELDIR = "models"
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(MODELDIR, exist_ok=True)

# 1) Load
df = pd.read_csv(RAW)

# 2) Basic audit to show at demo
print("Shape:", df.shape)
print("Columns:", list(df.columns))
print("Missing values per column:\n", df.isna().sum())

# 3) Keep only the standard columns used in this dataset
keep_cols = ["CreditScore","Geography","Gender","Age","Tenure","Balance",
             "NumOfProducts","HasCrCard","IsActiveMember","EstimatedSalary","Exited"]
df = df[keep_cols].copy()

# 4) EDA figures for demo
# Class balance
ax = df["Exited"].value_counts().sort_index().rename({0:"Stayed",1:"Churned"}).plot(kind="bar")
plt.title("Class distribution (Stayed vs Churned)")
plt.ylabel("Count"); plt.xticks(rotation=0)
plt.tight_layout(); plt.savefig(f"{FIGDIR}/class_distribution.png"); plt.close()

# Churn rate by Geography
geo = (df.groupby("Geography")["Exited"].mean().sort_values()*100)
geo.plot(kind="bar"); plt.ylabel("Churn rate (%)"); plt.title("Churn rate by Geography")
plt.tight_layout(); plt.savefig(f"{FIGDIR}/churn_by_geography.png"); plt.close()

# Churn rate by Age bin
bins = pd.cut(df["Age"], bins=[0,30,40,50,60,120], right=False)
age = df.groupby(bins)["Exited"].mean()*100
age.plot(kind="bar"); plt.ylabel("Churn rate (%)"); plt.title("Churn rate by Age group")
plt.tight_layout(); plt.savefig(f"{FIGDIR}/churn_by_age.png"); plt.close()

# 5) Train/test split (stratified)
X = df.drop(columns=["Exited"])
y = df["Exited"].astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 6) Preprocess: one-hot cat + scale numeric (all inside a pipeline)
numeric = ["CreditScore","Age","Tenure","Balance","NumOfProducts","EstimatedSalary"]
categorical = ["Geography","Gender","HasCrCard","IsActiveMember"]  # last two are 0/1 but treat as categorical for safety

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ], remainder="drop"
)

# 7) Baseline logistic regression (no SMOTE yet for Checkpoint 1)
clf = Pipeline(steps=[
    ("prep", preprocess),
    ("lr", LogisticRegression(max_iter=1000))
])
clf.fit(X_train, y_train)

# 8) Evaluate on test set
proba = clf.predict_proba(X_test)[:,1]
pred = (proba >= 0.5).astype(int)

metrics = {
    "accuracy": round(accuracy_score(y_test, pred), 4),
    "balanced_accuracy": round(balanced_accuracy_score(y_test, pred), 4),
    "precision": round(precision_score(y_test, pred), 4),
    "recall": round(recall_score(y_test, pred), 4),
    "f1": round(f1_score(y_test, pred), 4),
    "roc_auc": round(roc_auc_score(y_test, proba), 4)
}
print("Baseline Logistic Regression (test):", metrics)

# 9) ROC curve figure for demo
RocCurveDisplay.from_predictions(y_test, proba)
plt.title("ROC - Baseline Logistic Regression")
plt.tight_layout(); plt.savefig(f"{FIGDIR}/roc_baseline_lr.png"); plt.close()

# 10) Save pipeline for inference script
joblib.dump(clf, f"{MODELDIR}/baseline_lr.joblib")

# 11) Save metrics for your slide
with open("reports/baseline_metrics.json","w") as f:
    json.dump(metrics, f, indent=2)

print("Saved model -> models/baseline_lr.joblib")
print("Saved figures -> reports/figures/*.png")
print("Saved metrics -> reports/baseline_metrics.json")
