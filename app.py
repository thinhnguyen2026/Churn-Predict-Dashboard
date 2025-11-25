import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# -----------------------------
# xAI helpers
# -----------------------------
def prepare_background(X_row: pd.DataFrame, user_row: dict, schema: dict):
    """Return a background DataFrame aligned to X_row's columns & dtypes."""
    bg_real = load_background()
    if bg_real is not None:
        bg = bg_real.copy()
    else:
        bg = make_background(user_row, schema, n=400, jitter=0.03, seed=42)

    # keep only expected columns, add any missing ones from the user_row
    expected = list(X_row.columns)
    bg = bg.reindex(columns=expected)

    for c in expected:
        if c not in bg or bg[c].isna().all():
            bg[c] = user_row[c]

    # cast numeric/categorical types to be safe
    for c, spec in schema["numeric"].items():
        if c in bg:
            # if the user input is int, keep int; else float
            bg[c] = bg[c].astype(int if isinstance(user_row[c], int) else float)

    for c in schema["categorical"].keys():
        if c in bg:
            # keep as object/string so OneHotEncoder sees categories
            bg[c] = bg[c].astype("object")

    # small sample for speed
    return bg.sample(min(400, len(bg)), random_state=42).reset_index(drop=True)


BG_PATH = "reports/bg/bg_test.csv"


@st.cache_resource(show_spinner=False)
def load_background(path=BG_PATH):
    if os.path.isfile(path):
        df = pd.read_csv(path)
        if "Exited" in df.columns:
            df = df.drop(columns=["Exited"])
        return df
    return None


def make_background(user_row: dict, schema: dict, n=300, jitter=0.03, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        r = dict(user_row)
        # jitter numeric
        for k, spec in schema["numeric"].items():
            v = r[k]
            lo, hi, _default = spec
            if v != 0:
                v2 = float(v) * (1 + rng.normal(0, jitter))
            else:
                v2 = float(v) + rng.normal(0, (hi - lo) * jitter * 0.1)
            v2 = float(np.clip(v2, lo, hi))
            r[k] = int(v2) if isinstance(v, int) else v2
        # occasionally flip categoricals
        for k, choices in schema["categorical"].items():
            if len(choices) > 1 and rng.random() < 0.5:
                r[k] = rng.choice(choices)
        rows.append(r)
    return pd.DataFrame(rows)


def _proba_like(estimator, X_df: pd.DataFrame) -> np.ndarray:
    """Return a probability-like 0..1 score for the positive class."""
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X_df)[:, 1]
    elif hasattr(estimator, "decision_function"):
        z = estimator.decision_function(X_df)
        return 1 / (1 + np.exp(-z))
    p = estimator.predict(X_df).astype(float)
    if set(np.unique(p)) == {-1.0, 1.0}:
        p = (p + 1.0) / 2.0
    return p


def show_plot(fig):
    """Single place to render Plotly charts."""
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": False}
    )


def local_importance_signed(
    model, X_row: pd.DataFrame, bg: pd.DataFrame, draws=300, seed=42
):
    """
    For each feature, replace ONLY that feature in X_row with a value sampled from bg,
    recompute the probability, and average the signed change.
    Returns a Series in [-1, 1] where sign shows direction (↑ increases churn risk).
    """
    rng = np.random.default_rng(seed)
    base = float(_proba_like(model, X_row)[0])
    cols = list(X_row.columns)

    deltas = {c: [] for c in cols}
    # Pre-sample donor indices for speed/reproducibility
    idxs = rng.integers(0, len(bg), size=draws)

    for j in idxs:
        donor = bg.iloc[j]
        for c in cols:
            Xm = X_row.copy()
            Xm[c] = donor[c]
            p = float(_proba_like(model, Xm)[0])
            deltas[c].append(p - base)

    imp = pd.Series({c: np.mean(v) for c, v in deltas.items()})
    m = np.max(np.abs(imp.values)) or 1.0
    return imp / m


# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="ChurnPredict", layout="wide")
MODEL_PATH = "models/final_model.joblib"
SCHEMA_PATH = "data/schema.json"  # optional
EVAL_PATH = "reports/eval/eval_predictions.csv"

# Default UI ranges (used if no schema.json)
DEFAULT_SCHEMA = {
    "categorical": {
        "Geography": ["France", "Germany", "Spain"],
        "Gender": ["Male", "Female"],
        "HasCrCard": [0, 1],
        "IsActiveMember": [0, 1],
    },
    "numeric": {
        "CreditScore": [300, 900, 650],
        "Age": [18, 92, 40],
        "Tenure": [0, 10, 5],
        "Balance": [0.0, 250000.0, 90000.0],
        "NumOfProducts": [1, 4, 2],
        "EstimatedSalary": [0.0, 200000.0, 60000.0],
    },
}

# -----------------------------
# Utils
# -----------------------------


@st.cache_resource(show_spinner=False)
def load_model(path):
    if not os.path.isfile(path):
        return None
    return joblib.load(path)


def load_schema(path):
    if os.path.isfile(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return DEFAULT_SCHEMA


@st.cache_data(show_spinner=False)
def load_eval_df(path=EVAL_PATH):
    if os.path.isfile(path):
        return pd.read_csv(path)
    return None


def risk_bucket(prob):
    if prob < 0.25:
        return "Low", "🟢"
    if prob < 0.55:
        return "Medium", "🟡"
    return "High", "🔴"


def plot_gauge(prob):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(prob * 100, 1),
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"thickness": 0.3},
                "steps": [
                    {"range": [0, 25], "color": "#d6f5d6"},
                    {"range": [25, 55], "color": "#fff3cd"},
                    {"range": [55, 100], "color": "#f8d7da"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.75,
                    "value": prob * 100,
                },
            },
            title={"text": "Churn Probability"},
        )
    )
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def simple_recommendations(x: dict):
    tips = []
    # you can tune these thresholds based on your dataset distribution
    if x.get("IsActiveMember", 1) == 0:
        tips.append(
            "Customer is not active: enroll in engagement program (app onboarding, monthly check-ins)."
        )
    if x.get("CreditScore", 650) < 600:
        tips.append(
            "Low credit score: offer credit-education resources and small credit-builder products."
        )
    if x.get("Balance", 0) > 100000 and x.get("NumOfProducts", 1) <= 1:
        tips.append(
            "High balance but few products: propose savings/investment bundles or premium account."
        )
    if x.get("Tenure", 0) >= 5 and x.get("IsActiveMember", 1) == 0:
        tips.append(
            "Long tenure but low activity: targeted loyalty perks to re-activate."
        )
    if not tips:
        tips.append(
            "Maintain relationship with periodic check-ins and personalized offers."
        )
    return tips


# -----------------------------
# Load model + schema + eval
# -----------------------------
model = load_model(MODEL_PATH)
schema = load_schema(SCHEMA_PATH)
df_eval = load_eval_df()

# -----------------------------
# Sidebar (about + status)
# -----------------------------
with st.sidebar:
    st.header("About ChurnPredict")
    st.write(
        "This dashboard uses a trained machine learning model to estimate "
        "the probability that a retail banking customer will churn.\n\n"
        "Use the *Single Customer* tab for a detailed explanation of one case, "
        "and the *Portfolio Dashboard* tab for an overview of many customers."
    )
    if model is None:
        st.warning("No trained model found. Save a pipeline to `models/final_model.joblib`.")
    st.caption("Author: Thinh Nguyen")

# -----------------------------
# Layout: title + tabs
# -----------------------------
st.markdown(
    """
    <div style="
        background: linear-gradient(90deg, #4f46e5, #6366f1);
        padding: 0.8rem 1.2rem;
        border-radius: 12px;
        text-align: center;
        color: white;
        margin-bottom: 0.6rem;
    ">
        <h1 style="margin-bottom: 0.2rem;">ChurnPredict</h1>
        <p style="margin: 0;">An application for Customer Churn Risk Dashboard was made by Thinh Nguyen.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()


tab_single, tab_dashboard = st.tabs(["🔮 Single Customer", "📊 Portfolio Dashboard"])

# -----------------------------
# Tab 1: Single customer view
# -----------------------------
with tab_single:
    # ----- INPUT BLOCK (light blue) -----
    with st.container():
        st.markdown(
            "<div style='background-color:#f0f4ff; padding:1rem; border-radius:10px;'>",
            unsafe_allow_html=True,
        )

        st.subheader("Enter customer attributes")

        with st.form("customer_form"):
            col1, col2 = st.columns(2)

            with col1:
                Geography = st.selectbox("Geography", schema["categorical"]["Geography"])
                Gender = st.selectbox("Gender", schema["categorical"]["Gender"])
                CreditScore = st.number_input(
                    "CreditScore",
                    min_value=int(schema["numeric"]["CreditScore"][0]),
                    max_value=int(schema["numeric"]["CreditScore"][1]),
                    value=int(schema["numeric"]["CreditScore"][2]),
                    step=1,
                )
                Age = st.number_input(
                    "Age",
                    min_value=int(schema["numeric"]["Age"][0]),
                    max_value=int(schema["numeric"]["Age"][1]),
                    value=int(schema["numeric"]["Age"][2]),
                    step=1,
                )
                Tenure = st.number_input(
                    "Tenure (years)",
                    min_value=int(schema["numeric"]["Tenure"][0]),
                    max_value=int(schema["numeric"]["Tenure"][1]),
                    value=int(schema["numeric"]["Tenure"][2]),
                    step=1,
                )

            with col2:
                Balance = st.number_input(
                    "Balance",
                    min_value=float(schema["numeric"]["Balance"][0]),
                    max_value=float(schema["numeric"]["Balance"][1]),
                    value=float(schema["numeric"]["Balance"][2]),
                    step=100.0,
                    format="%.2f",
                )
                EstimatedSalary = st.number_input(
                    "EstimatedSalary",
                    min_value=float(schema["numeric"]["EstimatedSalary"][0]),
                    max_value=float(schema["numeric"]["EstimatedSalary"][1]),
                    value=float(schema["numeric"]["EstimatedSalary"][2]),
                    step=100.0,
                    format="%.2f",
                )
                NumOfProducts = st.number_input(
                    "NumOfProducts",
                    min_value=int(schema["numeric"]["NumOfProducts"][0]),
                    max_value=int(schema["numeric"]["NumOfProducts"][1]),
                    value=int(schema["numeric"]["NumOfProducts"][2]),
                    step=1,
                )
                HasCrCard = st.selectbox(
                    "HasCrCard", schema["categorical"]["HasCrCard"]
                )
                IsActiveMember = st.selectbox(
                    "IsActiveMember", schema["categorical"]["IsActiveMember"]
                )

            center_col = st.columns([1, 1, 1])[1]
            with center_col:
                submitted = st.form_submit_button(
                    "Predict churn risk",
                    type="primary",
                    use_container_width=True,
                )


        st.markdown("</div>", unsafe_allow_html=True)  # close input block div

    if submitted:
        user_row = {
            "Geography": Geography,
            "Gender": Gender,
            "CreditScore": int(CreditScore),
            "Age": int(Age),
            "Tenure": int(Tenure),
            "Balance": float(Balance),
            "NumOfProducts": int(NumOfProducts),
            "HasCrCard": int(HasCrCard),
            "IsActiveMember": int(IsActiveMember),
            "EstimatedSalary": float(EstimatedSalary),
        }

        if model is None:
            st.error(
                "No model available. Train and save `models/final_model.joblib` first."
            )
        else:
            with st.spinner("Calculating churn risk..."):
                X_row = pd.DataFrame([user_row])
                prob = float(model.predict_proba(X_row)[:, 1][0])
                label = int(prob >= 0.5)
                bucket, dot = risk_bucket(prob)
                st.session_state["user_row"] = user_row
                st.session_state["prob"] = prob
                st.session_state["label"] = label

            st.success("Prediction completed")

            # ----- PREDICTION + GAUGE BLOCK (soft yellow) -----
            with st.container():
                st.markdown(
                    "<div style='background-color:#fff7e6; padding:1rem; border-radius:10px; margin-top:0.8rem;'>",
                    unsafe_allow_html=True,
                )

                st.markdown(
                    "<h4 style='text-align:center; margin-top:0.2rem;'>Customer churn risk</h4>",
                    unsafe_allow_html=True,
                )

                left, center, right = st.columns([1, 2, 1])
                with center:
                    st.metric(
                        label=f"Risk level {dot}",
                        value=bucket,
                        delta=f"{prob*100:.1f}%",
                    )
                    show_plot(plot_gauge(prob))

                st.subheader("Prediction details")
                st.write(f"**Churn prediction:** {'Yes' if label == 1 else 'No'}")
                st.write(f"**Churn probability:** {prob:.3f}")

                st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    # ----- FEATURE IMPORTANCE + RECOMMENDATIONS BLOCK (light grey) -----
    st.subheader("Why this prediction? (Feature importance)")

    if "user_row" in st.session_state and model is not None:
        with st.container():
            st.markdown(
                "<div style='background-color:#f5f5f5; padding:1rem; border-radius:10px;'>",
                unsafe_allow_html=True,
            )

            user_row = st.session_state["user_row"]
            X_row = pd.DataFrame([user_row])
            bg = prepare_background(X_row, user_row, schema)

            imp = local_importance_signed(model, X_row, bg, draws=300, seed=42)
            top = imp.reindex(imp.abs().sort_values(ascending=False).head(5).index)
            top_df = (
                top.rename("Relative impact (±)")
                .reset_index()
                .rename(columns={"index": "Feature"})
            )

            st.dataframe(top_df, width="stretch", hide_index=True)

            # Color by direction: red = increases churn, green = decreases churn
            colors = [
                "#e74c3c" if v > 0 else "#2ecc71"
                for v in top_df["Relative impact (±)"]
            ]
            bar = go.Figure(
                go.Bar(
                    x=top_df["Feature"],
                    y=top_df["Relative impact (±)"],
                    marker_color=colors,
                )
            )
            bar.update_layout(
                height=330,
                margin=dict(l=10, r=10, t=30, b=90),
                xaxis_tickangle=-30,
            )
            show_plot(bar)

            # Divider line between explanation and recommendations
            st.markdown(
                "<div style='border-top:1px solid #d0d0d0; margin:1.2rem 0 0.8rem 0;'></div>",
                unsafe_allow_html=True,
            )

            # Inner white card for recommendations
            st.markdown(
                "<div style='background-color:#ffffff; padding:0.8rem 1rem; border-radius:8px;'>",
                unsafe_allow_html=True,
            )

            st.subheader("Recommended next steps")
            for tip in simple_recommendations(user_row):
                st.write("• " + tip)

            st.markdown("</div>", unsafe_allow_html=True)   # close white card
            st.markdown("</div>", unsafe_allow_html=True)   # close grey outer block

    else:
        st.info(
            "Fill the form and click **Predict churn risk** to see explanation and recommendations."
        )

# -----------------------------
# Tab 2: Portfolio dashboard
# -----------------------------
with tab_dashboard:
    st.subheader("Portfolio overview")

    if df_eval is None:
        st.info(
            "No evaluation file found at `reports/eval/eval_predictions.csv`.\n\n"
            "Save a CSV with columns like `y_true`, `y_pred`, `y_prob` to enable the dashboard."
        )
    elif not all(col in df_eval.columns for col in ["y_true", "y_pred", "y_prob"]):
        st.warning(
            "Evaluation file is loaded but is missing one of the required "
            "columns: `y_true`, `y_pred`, `y_prob`."
        )
        st.dataframe(df_eval.head())
    else:
        total = len(df_eval)
        churn_rate = df_eval["y_true"].mean()
        acc = (df_eval["y_true"] == df_eval["y_pred"]).mean()

        # ----- METRICS BLOCK (light green) -----
        with st.container():
            st.markdown(
                "<div style='background-color:#e8f7f0; padding:1rem; border-radius:10px; margin-bottom:0.8rem;'>",
                unsafe_allow_html=True,
            )

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total customers", total)
            with col2:
                st.metric("Churn rate", f"{churn_rate*100:.1f}%")
            with col3:
                st.metric("Model accuracy", f"{acc*100:.1f}%")

            st.markdown("</div>", unsafe_allow_html=True)

        # ----- CHURN DISTRIBUTION BLOCK (lavender) -----
        with st.container():
            st.markdown(
                "<div style='background-color:#f4f0ff; padding:1rem; border-radius:10px; margin-bottom:0.8rem;'>",
                unsafe_allow_html=True,
            )

            st.markdown("### Churn distribution")
            counts = df_eval["y_true"].value_counts().reindex([0, 1]).fillna(0)
            fig_churn = go.Figure(
                go.Bar(
                    x=["No churn", "Churn"],
                    y=[counts[0], counts[1]],
                    marker_color=["#2ecc71", "#e74c3c"],  # green = stayed, red = churned
                )
            )
            fig_churn.update_layout(margin=dict(l=10, r=10, t=30, b=40))
            show_plot(fig_churn)

            st.caption("How many customers in the evaluation set actually churned.")

            st.markdown("</div>", unsafe_allow_html=True)

        # ----- PROBABILITY HISTOGRAM BLOCK (soft red) -----
        with st.container():
            st.markdown(
                "<div style='background-color:#fdf2f2; padding:1rem; border-radius:10px;'>",
                unsafe_allow_html=True,
            )

            st.markdown("### Predicted probability distribution")
            fig_hist = go.Figure()
            fig_hist.add_histogram(
                x=df_eval["y_prob"],
                nbinsx=20,
                marker_color="#9b59b6"  # purple tone, different from bars above
            )
            fig_hist.update_layout(
                xaxis_title="Predicted churn probability",
                yaxis_title="Number of customers",
                margin=dict(l=10, r=10, t=30, b=40),
            )
            show_plot(fig_hist)

            st.caption("How confident the model is across all customers.")

            st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Footer (Warm Color Version)
# -----------------------------
st.markdown(
    """
    <div style="
        margin-top: 2rem;
        padding: 1.5rem;
        border-radius: 12px;
        background: linear-gradient(135deg, #f7c59f, #f99f92);
        color: #2c2c2c;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    ">
        <h3 style="margin-bottom: 0.4rem;">
            Questions about this dashboard?
        </h3>
        <p style="margin: 0.2rem 0; font-size: 0.95rem;">
            You can contact <b>Thinh Nguyen</b> at
            <a href="mailto:thinhnguyen_2026@depauw.edu" style="color:#4c2e05; font-weight:bold;">
                thinhnguyen_2026@depauw.edu
            </a>.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
