# app.py
# Streamlit UI for ChurnPredict — single-customer input → prediction + probability
# Run: streamlit run app.py

import os
import time
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# -----------------------------
# Paths & setup
# -----------------------------
MODEL_PATHS_TRY = [
    "models/final_model.joblib",
    "models/best_smote.joblib",
    "models/best_imbalanced.joblib",
    "models/baseline_lr.joblib",
]
REPORTS_DIR = "reports"
LOG_PATH = os.path.join(REPORTS_DIR, "prediction_logs.csv")
os.makedirs(REPORTS_DIR, exist_ok=True)

EXPECTED_COLS = [
    "CreditScore","Geography","Gender","Age","Tenure","Balance",
    "NumOfProducts","HasCrCard","IsActiveMember","EstimatedSalary"
]

GEO_OPTS = ["France","Germany","Spain"]
GENDER_OPTS = ["Male","Female"]

# -----------------------------
# Utilities
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    for p in MODEL_PATHS_TRY:
        if os.path.exists(p):
            try:
                clf = joblib.load(p)
                return clf, p
            except Exception:
                continue
    st.warning("No trained model found. Please train and save a model to models/final_model.joblib")
    return None, None

def make_dataframe_from_inputs(inputs: dict) -> pd.DataFrame:
    # Ensure correct column order
    row = {k: inputs.get(k) for k in EXPECTED_COLS}
    return pd.DataFrame([row], columns=EXPECTED_COLS)

def risk_label(prob: float) -> tuple[str, str]:
    """Return (label, color) for a given churn probability."""
    if prob >= 0.70:
        return "High Risk", "🔴"
    if prob >= 0.40:
        return "Medium Risk", "🟠"
    return "Low Risk", "🟢"

def safe_float(x, default):
    try:
        return float(x)
    except Exception:
        return default

def safe_int(x, default):
    try:
        return int(x)
    except Exception:
        return default

def append_log(record: dict):
    try:
        df = pd.DataFrame([record])
        if not os.path.exists(LOG_PATH):
            df.to_csv(LOG_PATH, index=False)
        else:
            df.to_csv(LOG_PATH, mode="a", header=False, index=False)
    except PermissionError:
        st.warning("Could not write to prediction log (file may be open elsewhere).")

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="ChurnPredict", page_icon="📉", layout="centered")
st.title("ChurnPredict — Single Customer Scoring")

with st.sidebar:
    st.header("About")
    st.write(
        "Enter customer attributes and press **Predict**.\n\n"
        "This app uses your trained pipeline (preprocessing + model) to produce a churn probability."
    )
    model_obj, model_path = load_model()
    if model_obj and model_path:
        st.success(f"Model loaded: `{model_path}`")
    else:
        st.error("Model not loaded. Train models and rerun this app.")
    st.caption("Tip: Try adjusting Age, Balance, and Active Member to see probability changes.")

st.subheader("Input customer attributes")

# --- Form for inputs ---
with st.form("input_form", clear_on_submit=False):
    c1, c2 = st.columns(2)

    with c1:
        geography = st.selectbox("Geography", GEO_OPTS, index=0)
        gender = st.selectbox("Gender", GENDER_OPTS, index=0)
        age = st.number_input("Age", min_value=18, max_value=100, value=40, step=1)
        tenure = st.number_input("Tenure (years)", min_value=0, max_value=50, value=5, step=1)
        num_products = st.number_input("NumOfProducts", min_value=1, max_value=10, value=2, step=1)

    with c2:
        credit_score = st.number_input("CreditScore", min_value=300, max_value=900, value=650, step=1)
        balance = st.number_input("Balance", min_value=0.0, max_value=1000000.0, value=90000.0, step=100.0, format="%.2f")
        estimated_salary = st.number_input("EstimatedSalary", min_value=0.0, max_value=1000000.0, value=60000.0, step=100.0, format="%.2f")
        has_cr_card = st.selectbox("HasCrCard", options=[0, 1], index=1)
        is_active = st.selectbox("IsActiveMember", options=[0, 1], index=1)

    submitted = st.form_submit_button("Predict", use_container_width=True)

# --- On submit ---
if submitted:
    if model_obj is None:
        st.error("No model available. Train and save a model first.")
    else:
        inputs = {
            "CreditScore": safe_int(credit_score, 650),
            "Geography": geography,
            "Gender": gender,
            "Age": safe_int(age, 40),
            "Tenure": safe_int(tenure, 5),
            "Balance": safe_float(balance, 90000.0),
            "NumOfProducts": safe_int(num_products, 2),
            "HasCrCard": safe_int(has_cr_card, 1),
            "IsActiveMember": safe_int(is_active, 1),
            "EstimatedSalary": safe_float(estimated_salary, 60000.0),
        }
        X = make_dataframe_from_inputs(inputs)

        with st.spinner("Scoring..."):
            try:
                proba = float(model_obj.predict_proba(X)[0, 1])
                pred = int(proba >= 0.5)
            except Exception as e:
                st.exception(e)
                st.stop()

        label = "Churn" if pred == 1 else "Stay"
        tag, emoji = risk_label(proba)

        st.markdown("---")
        st.subheader("Prediction")
        st.metric(label="Churn Probability", value=f"{proba:.2%}")
        st.write(f"**Predicted class:** {label}  {emoji}  •  **Risk level:** {tag}")

        # Simple visual bar
        st.progress(min(max(proba, 0.0), 1.0))

        # Log it
        log_record = {
            **inputs,
            "pred_label": label,
            "pred_proba": round(proba, 4),
            "timestamp": int(time.time()),
            "model_path": model_path or "N/A",
        }
        append_log(log_record)

        with st.expander("View request payload"):
            st.json(inputs)

        with st.expander("Recent predictions (from log)"):
            try:
                if os.path.exists(LOG_PATH):
                    df_log = pd.read_csv(LOG_PATH)
                    st.dataframe(df_log.tail(10), use_container_width=True)
                else:
                    st.info("No logs yet.")
            except Exception:
                st.info("Log not available (file may be in use).")

st.markdown("---")
st.caption("© 2025 ChurnPredict — Streamlit demo")

