import os, json, joblib, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, RocCurveDisplay, ConfusionMatrixDisplay
)
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# ---------- paths ----------
RAW = "churn.csv"
FIGDIR = "reports/figures"
REPORTDIR = "reports"
MODELDIR = "models"
EVALDIR = "reports/eval"   # <<< NEW

os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(REPORTDIR, exist_ok=True)
os.makedirs(MODELDIR, exist_ok=True)
os.makedirs(EVALDIR, exist_ok=True)   # <<< NEW


# ---------- data ----------
df = pd.read_csv(RAW)
keep_cols = ["CreditScore","Geography","Gender","Age","Tenure","Balance",
             "NumOfProducts","HasCrCard","IsActiveMember","EstimatedSalary","Exited"]
df = df[keep_cols].copy()

X = df.drop(columns=["Exited"])
y = df["Exited"].astype(int)

numeric = ["CreditScore","Age","Tenure","Balance","NumOfProducts","EstimatedSalary"]
categorical = ["Geography","Gender","HasCrCard","IsActiveMember"]

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ],
    remainder="drop"
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# === Export a real background slice (X_test + y_test) ===
BGDIR = "reports/bg"; os.makedirs(BGDIR, exist_ok=True)
bg = X_test.copy(); bg["Exited"] = y_test.values
bg_path = f"{BGDIR}/bg_test.csv"
bg.sample(min(1000, len(bg)), random_state=42).to_csv(bg_path, index=False)
print("saved background ->", bg_path)

# ---------- models + small grids (fast but meaningful) ----------
models = {
    "AdaBoost": (
        AdaBoostClassifier(algorithm="SAMME", random_state=42),
        { "clf__n_estimators": [100, 200],
          "clf__learning_rate": [0.05, 0.1] }
    ),
    "GradientBoosting": (
        GradientBoostingClassifier(random_state=42),
        { "clf__n_estimators": [100, 200],
          "clf__learning_rate": [0.05, 0.1],
          "clf__max_depth": [2, 3] }
    ),
    "LightGBM": (
        LGBMClassifier(random_state=42, force_row_wise=True),
        { "clf__n_estimators": [200, 400],
          "clf__learning_rate": [0.05, 0.1],
          "clf__num_leaves": [15, 31] }
    ),
    "CatBoost": (
        CatBoostClassifier(random_state=42, verbose=False),
        { "clf__iterations": [300, 500],
          "clf__depth": [4, 6],
          "clf__learning_rate": [0.03, 0.1] }
    ),
}

# ---------- helpers ----------
def evaluate(y_true, proba, pred):
    return {
        "accuracy": round(accuracy_score(y_true, pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_true, pred), 4),
        "precision": round(precision_score(y_true, pred), 4),
        "recall": round(recall_score(y_true, pred), 4),
        "f1": round(f1_score(y_true, pred), 4),
        "roc_auc": round(roc_auc_score(y_true, proba), 4),
    }


def run_track(track_name, use_smote: bool):
    """
    track_name: "imbalanced" or "smote"
    use_smote:  whether to insert SMOTE (train-only) before the classifier
    """
    metrics_rows = []
    cv_rows = []  # >>> NEW (CV SUMMARY): collect per-model CV mean/std for best params

    roc_fig = plt.figure()
    ax = plt.gca()

    best_model_name = None
    best_score = -np.inf
    best_obj = None
    best_metrics = None

    for name, (clf, grid) in models.items():
        print(f"\n[{track_name}] tuning {name} ...")
        if use_smote:
            pipe = ImbPipeline([
                ("prep", preprocess),
                ("smote", SMOTE(random_state=42)),
                ("clf", clf)
            ])
        else:
            pipe = Pipeline([
                ("prep", preprocess),
                ("clf", clf)
            ])

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        gscv = GridSearchCV(
            estimator=pipe,
            param_grid=grid,
            cv=cv,
            scoring="balanced_accuracy",
            n_jobs=-1,
            verbose=0,
            return_train_score=False
        )
        gscv.fit(X_train, y_train)

        print(f"  best params: {gscv.best_params_}")
        print(f"  best CV balanced_acc (mean): {gscv.best_score_:.4f}")

        # >>> NEW (CV SUMMARY): compute std across splits for the best-ranked param set
        # identify rows for rank 1 (best)
        ranks = gscv.cv_results_["rank_test_score"]
        mask_best = (ranks == 1)
        # collect split scores for that row
        split_scores = []
        for i in range(cv.get_n_splits()):
            split_scores.append(gscv.cv_results_[f"split{i}_test_score"][mask_best][0])
        cv_mean = float(np.mean(split_scores))
        cv_std  = float(np.std(split_scores))
        print(f"  CV mean ± std: {cv_mean:.4f} ± {cv_std:.4f}")

        # record CV summary row
        cv_rows.append({
            "track": track_name,
            "model": name,
            "cv_mean_balanced_accuracy": round(cv_mean, 4),
            "cv_std_balanced_accuracy": round(cv_std, 4),
        })
        # <<< END NEW (CV SUMMARY)

        # evaluate on held-out test set (untouched)
        best = gscv.best_estimator_
        proba = best.predict_proba(X_test)[:,1]
        pred  = (proba >= 0.5).astype(int)
        m = evaluate(y_test, proba, pred)
        print(f"  test metrics: {m}")

        # add to test metrics table
        row = {"track": track_name, "model": name, **m}
        metrics_rows.append(row)

        # ROC curve overlay
        RocCurveDisplay.from_predictions(y_test, proba, name=name, ax=ax)

        # Keep best by balanced accuracy, then F1 as tie-break
        score = (m["balanced_accuracy"], m["f1"], m["roc_auc"])
        if score > (best_metrics["balanced_accuracy"], best_metrics["f1"], best_metrics["roc_auc"]) if best_metrics else True:
            best_model_name = name
            best_obj = best
            best_metrics = m

        # Confusion matrix figure per model
        cm_fig = plt.figure()
        ConfusionMatrixDisplay.from_predictions(y_test, pred)
        plt.title(f"Confusion Matrix - {name} ({track_name})")
        plt.tight_layout()
        cm_path = os.path.join(FIGDIR, f"cm_{name.lower()}_{track_name}.png")
        plt.savefig(cm_path, dpi=150)
        plt.close(cm_fig)
        print(f"  saved {cm_path}")

    # finalize ROC figure
    plt.plot([0,1],[0,1], linestyle="--")
    plt.title(f"ROC Curves - {track_name.upper()}")
    plt.tight_layout()
    roc_path = os.path.join(FIGDIR, f"roc_{track_name}.png")
    plt.savefig(roc_path, dpi=150)
    plt.close(roc_fig)
    print(f"saved {roc_path}")

    # save test metrics table
    dfm = pd.DataFrame(metrics_rows)
    csv_path = os.path.join(REPORTDIR, f"metrics_{track_name}.csv")
    dfm.to_csv(csv_path, index=False)
    print(f"saved {csv_path}")

    # >>> NEW (CV SUMMARY): save CV table
    dfcv = pd.DataFrame(cv_rows)
    cv_csv_path = os.path.join(REPORTDIR, f"cv_summary_{track_name}.csv")
    dfcv.to_csv(cv_csv_path, index=False)
    print(f"saved {cv_csv_path}")
    # <<< END NEW

    # save best model for the track
    model_path = os.path.join(MODELDIR, f"best_{track_name}.joblib")
    joblib.dump(best_obj, model_path)
    print(f"saved {model_path}")

    return best_model_name, best_metrics, model_path, csv_path, cv_csv_path

# ---------- run both tracks ----------
best_imb_name, best_imb_metrics, best_imb_path, csv_imb, cv_csv_imb = run_track("imbalanced", use_smote=False)
best_sm_name,  best_sm_metrics,  best_sm_path,  csv_sm,  cv_csv_sm  = run_track("smote", use_smote=True)

# pick final by balanced accuracy then F1 then AUC
def key(m): return (m["balanced_accuracy"], m["f1"], m["roc_auc"])
final = ("imbalanced", best_imb_name, best_imb_metrics, best_imb_path)
if key(best_sm_metrics) > key(best_imb_metrics):
    final = ("smote", best_sm_name, best_sm_metrics, best_sm_path)

final_track, final_name, final_metrics, final_model_path = final

# copy/save final selection info
final_info = {
    "final_track": final_track,
    "final_model_name": final_name,
    "final_metrics": final_metrics,
    "source_model_path": final_model_path
}
with open(os.path.join(REPORTDIR, "final_selection.json"), "w") as f:
    json.dump(final_info, f, indent=2)

# also save a canonical copy of the chosen model
final_out = os.path.join(MODELDIR, "final_model.joblib")
joblib.dump(joblib.load(final_model_path), final_out)
print("\n=== FINAL SELECTION ===")
print(json.dumps(final_info, indent=2))
print(f"saved best model -> {final_out}")

# ---------- create eval_predictions.csv for Streamlit dashboard ----------
from pathlib import Path

print("\nCreating eval_predictions.csv for dashboard ...")
final_model = joblib.load(final_out)   # load the chosen best model (with preprocess inside)

# Use the same X_test, y_test defined earlier in this file
proba = final_model.predict_proba(X_test)[:, 1]
y_pred = (proba >= 0.5).astype(int)   # or final_model.predict(X_test)

# Build evaluation dataframe
df_eval = X_test.copy()
df_eval["y_true"] = y_test.values
df_eval["y_pred"] = y_pred
df_eval["y_prob"] = proba

eval_path = os.path.join(EVALDIR, "eval_predictions.csv")
df_eval.to_csv(eval_path, index=False)
print(f"saved evaluation predictions -> {eval_path}")
# ------------------------------------------------------------------------

# bar chart comparison (Balanced Acc & F1) from both CSVs
df_all = pd.concat([pd.read_csv(csv_imb), pd.read_csv(csv_sm)], ignore_index=True)

for metric in ["balanced_accuracy","f1","roc_auc"]:
    plt.figure()
    # pivot to bars: rows=model, cols=track
    pivot = df_all.pivot(index="model", columns="track", values=metric).reindex(["AdaBoost","GradientBoosting","LightGBM","CatBoost"])
    pivot.plot(kind="bar")
    plt.title(f"{metric.upper()} by Model and Track")
    plt.ylabel(metric)
    plt.tight_layout()
    outp = os.path.join(FIGDIR, f"bar_{metric}.png")
    plt.savefig(outp, dpi=150)
    plt.close()
    print(f"saved {outp}")

df_eval = pd.read_csv("reports/eval/eval_predictions.csv")
print(df_eval.head())
print(df_eval.columns)