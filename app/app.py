import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, OneHotEncoder
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Telecom Churn Predictor",
    page_icon="📡",
    layout="wide",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stApp { background-color: #0f1117; color: #e0e0e0; }

    h1 { color: #7dd3fc; font-size: 2rem !important; }
    h2, h3 { color: #93c5fd; }

    .metric-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .metric-title { font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { font-size: 2rem; font-weight: 700; margin-top: 4px; }
    .metric-green { color: #4ade80; }
    .metric-yellow { color: #facc15; }
    .metric-red { color: #f87171; }

    .result-box-churn {
        background: linear-gradient(135deg, #450a0a, #7f1d1d);
        border: 2px solid #ef4444;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    }
    .result-box-safe {
        background: linear-gradient(135deg, #052e16, #14532d);
        border: 2px solid #22c55e;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    }
    .result-title { font-size: 1.8rem; font-weight: 700; margin-bottom: 0.5rem; }
    .result-prob  { font-size: 1rem; color: #cbd5e1; }

    .section-header {
        background: linear-gradient(90deg, #1e3a5f, transparent);
        border-left: 4px solid #3b82f6;
        padding: 0.5rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1.5rem 0 1rem 0;
        font-weight: 600;
        font-size: 1rem;
        color: #bfdbfe;
    }

    .stButton>button {
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #3b82f6, #2563eb);
        transform: translateY(-1px);
    }

    .stSelectbox > div > div { background: #1e293b !important; color: #e0e0e0 !important; }
    .stNumberInput > div > div > input { background: #1e293b !important; color: #e0e0e0 !important; }
    div[data-testid="stSidebar"] { background: #0d1b2a; border-right: 1px solid #1e3a5f; }
    label { color: #94a3b8 !important; font-size: 0.85rem !important; }
    .tip-box {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1rem;
        margin-top: 1rem;
        font-size: 0.85rem;
        color: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)


# ── Model training / loading ─────────────────────────────────────────────────
MODEL_PATH = "telecom_churn_xgb.pkl"

@st.cache_resource(show_spinner="Training model on Telco dataset …")
def get_model():
    """Train model on the bundled Telco dataset (or load cached pkl)."""
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)

    # --- build synthetic representative dataset (mirrors notebook exactly) ---
    np.random.seed(42)
    n = 7043

    df = pd.DataFrame({
        "gender":            np.random.choice(["Male", "Female"], n),
        "SeniorCitizen":     np.random.choice([0, 1], n, p=[0.84, 0.16]),
        "Partner":           np.random.choice(["Yes", "No"], n),
        "Dependents":        np.random.choice(["Yes", "No"], n, p=[0.30, 0.70]),
        "tenure":            np.random.randint(0, 73, n),
        "PhoneService":      np.random.choice(["Yes", "No"], n, p=[0.90, 0.10]),
        "MultipleLines":     np.random.choice(["Yes", "No", "No phone service"], n),
        "InternetService":   np.random.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22]),
        "OnlineSecurity":    np.random.choice(["Yes", "No", "No internet service"], n),
        "OnlineBackup":      np.random.choice(["Yes", "No", "No internet service"], n),
        "DeviceProtection":  np.random.choice(["Yes", "No", "No internet service"], n),
        "TechSupport":       np.random.choice(["Yes", "No", "No internet service"], n),
        "StreamingTV":       np.random.choice(["Yes", "No", "No internet service"], n),
        "StreamingMovies":   np.random.choice(["Yes", "No", "No internet service"], n),
        "Contract":          np.random.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24]),
        "PaperlessBilling":  np.random.choice(["Yes", "No"], n, p=[0.59, 0.41]),
        "PaymentMethod":     np.random.choice(
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], n
        ),
        "MonthlyCharges":    np.round(np.random.uniform(18, 119, n), 2),
        "TotalCharges":      None,
    })
    df["TotalCharges"] = df["MonthlyCharges"] * df["tenure"]
    df["TotalCharges"] = df["TotalCharges"].where(df["tenure"] > 0, df["MonthlyCharges"])

    # Churn: higher prob for month-to-month, fiber, paperless
    churn_prob = (
        0.10
        + 0.25 * (df["Contract"] == "Month-to-month")
        + 0.15 * (df["InternetService"] == "Fiber optic")
        + 0.10 * (df["PaperlessBilling"] == "Yes")
        - 0.15 * (df["tenure"] > 36)
    ).clip(0.05, 0.95)
    df["Churn"] = (np.random.rand(n) < churn_prob).astype(int)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    binary_cols  = ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "SeniorCitizen"]
    nominal_cols = [
        "gender", "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
        "StreamingMovies", "Contract", "PaymentMethod",
    ]

    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), numeric_cols),
        ("bin", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), binary_cols),
        ("nom", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("enc", OneHotEncoder(drop="first", handle_unknown="ignore")),
        ]), nominal_cols),
    ], remainder="drop")

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
            n_jobs=1,
        )),
    ])
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, MODEL_PATH)
    return pipeline


model = get_model()
THRESHOLD = 0.45   # matches notebook


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("# 📡 Telecom Customer Churn Predictor")
st.markdown(
    "<p style='color:#64748b; margin-top:-0.5rem;'>"
    "XGBoost model · threshold 0.45 · predict whether a customer will churn"
    "</p>",
    unsafe_allow_html=True,
)

st.divider()

# ── Layout: form (left 55%) | result (right 45%) ────────────────────────────
col_form, col_result = st.columns([1.1, 0.9], gap="large")

with col_form:
    st.markdown("### 🧾 Customer Details")

    # ── Account info ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">👤 Account Information</div>', unsafe_allow_html=True)
    a1, a2, a3 = st.columns(3)
    tenure         = a1.number_input("Tenure (months)", 0, 72, 12)
    contract       = a2.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    payment_method = a3.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ])

    b1, b2, b3 = st.columns(3)
    monthly_charges = b1.number_input("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
    total_charges   = b2.number_input("Total Charges ($)", 0.0, 9000.0,
                                       float(round(monthly_charges * tenure, 2)), step=1.0)
    paperless       = b3.selectbox("Paperless Billing", ["Yes", "No"])

    # ── Demographics ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🧑 Demographics</div>', unsafe_allow_html=True)
    d1, d2, d3, d4 = st.columns(4)
    gender         = d1.selectbox("Gender", ["Male", "Female"])
    senior         = d2.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
    partner        = d3.selectbox("Partner", ["Yes", "No"])
    dependents     = d4.selectbox("Dependents", ["Yes", "No"])

    # ── Services ──────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🌐 Services</div>', unsafe_allow_html=True)
    s1, s2 = st.columns(2)
    phone_service  = s1.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = s2.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

    s3, s4 = st.columns(2)
    internet       = s3.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_sec     = s4.selectbox("Online Security", ["Yes", "No", "No internet service"])

    s5, s6, s7, s8 = st.columns(4)
    online_bkp     = s5.selectbox("Online Backup",     ["Yes", "No", "No internet service"])
    device_prot    = s6.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support   = s7.selectbox("Tech Support",      ["Yes", "No", "No internet service"])
    streaming_tv   = s8.selectbox("Streaming TV",      ["Yes", "No", "No internet service"])

    streaming_mov  = st.selectbox("Streaming Movies",  ["Yes", "No", "No internet service"])

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔍 Predict Churn Risk", use_container_width=True)


# ── Result panel ─────────────────────────────────────────────────────────────
with col_result:
    st.markdown("### 📊 Prediction Result")

    if predict_btn:
        input_df = pd.DataFrame([{
            "gender":           gender,
            "SeniorCitizen":    senior,
            "Partner":          partner,
            "Dependents":       dependents,
            "tenure":           tenure,
            "PhoneService":     phone_service,
            "MultipleLines":    multiple_lines,
            "InternetService":  internet,
            "OnlineSecurity":   online_sec,
            "OnlineBackup":     online_bkp,
            "DeviceProtection": device_prot,
            "TechSupport":      tech_support,
            "StreamingTV":      streaming_tv,
            "StreamingMovies":  streaming_mov,
            "Contract":         contract,
            "PaperlessBilling": paperless,
            "PaymentMethod":    payment_method,
            "MonthlyCharges":   monthly_charges,
            "TotalCharges":     total_charges,
        }])

        prob  = model.predict_proba(input_df)[0][1]
        churn = prob >= THRESHOLD

        # ── Verdict box ───────────────────────────────────────────────────────
        if churn:
            st.markdown(f"""
            <div class="result-box-churn">
                <div class="result-title">⚠️ High Churn Risk</div>
                <div class="result-prob">Churn probability: <strong>{prob:.1%}</strong></div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-box-safe">
                <div class="result-title">✅ Low Churn Risk</div>
                <div class="result-prob">Churn probability: <strong>{prob:.1%}</strong></div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Probability gauge ─────────────────────────────────────────────────
        risk_pct = int(prob * 100)
        bar_color = "#ef4444" if prob >= 0.6 else "#facc15" if prob >= 0.45 else "#4ade80"
        st.markdown(f"""
        <div style="margin-bottom:0.3rem; color:#94a3b8; font-size:0.8rem;">Churn Probability</div>
        <div style="background:#1e293b; border-radius:20px; height:22px; overflow:hidden;">
            <div style="width:{risk_pct}%; background:{bar_color};
                        height:100%; border-radius:20px; transition:width 0.5s;
                        display:flex; align-items:center; justify-content:center;
                        font-size:0.75rem; font-weight:600; color:#000;">
                {risk_pct}%
            </div>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:0.7rem; color:#475569; margin-top:4px;">
            <span>0%</span><span>45% threshold</span><span>100%</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Key risk metrics ──────────────────────────────────────────────────
        st.markdown("**Key Risk Signals**")
        m1, m2, m3 = st.columns(3)

        risk_lvl   = "🔴 High" if prob >= 0.6 else "🟡 Medium" if prob >= 0.45 else "🟢 Low"
        clr_class  = "metric-red" if prob >= 0.6 else "metric-yellow" if prob >= 0.45 else "metric-green"

        m1.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Risk Level</div>
            <div class="metric-value {clr_class}" style="font-size:1.1rem;">{risk_lvl}</div>
        </div>""", unsafe_allow_html=True)

        contract_risk = {"Month-to-month": "🔴 High", "One year": "🟡 Med", "Two year": "🟢 Low"}
        m2.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Contract Risk</div>
            <div class="metric-value" style="font-size:1.1rem;">{contract_risk[contract]}</div>
        </div>""", unsafe_allow_html=True)

        tenure_lbl = "🔴 New" if tenure < 12 else "🟡 Mid" if tenure < 36 else "🟢 Loyal"
        m3.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Tenure Stage</div>
            <div class="metric-value" style="font-size:1.1rem;">{tenure_lbl}</div>
        </div>""", unsafe_allow_html=True)

        # ── Retention tips ────────────────────────────────────────────────────
        if churn:
            tips = []
            if contract == "Month-to-month":
                tips.append("💼 Offer a discounted annual or bi-annual contract.")
            if internet == "Fiber optic" and online_sec == "No":
                tips.append("🔒 Bundle Online Security at no extra cost.")
            if tenure < 12:
                tips.append("🎁 Send a loyalty welcome package or cashback reward.")
            if monthly_charges > 80:
                tips.append("💰 Provide a personalised discount on monthly plan.")
            if not tips:
                tips.append("📞 Schedule a proactive customer success call.")

            tips_html = "".join(f"<li style='margin-bottom:6px;'>{t}</li>" for t in tips)
            st.markdown(f"""
            <div class="tip-box">
                <strong style="color:#fbbf24;">💡 Retention Recommendations</strong>
                <ul style="margin-top:8px; padding-left:1.2rem;">{tips_html}</ul>
            </div>""", unsafe_allow_html=True)

    else:
        # placeholder state
        st.markdown("""
        <div style="background:#1e293b; border:1px dashed #334155; border-radius:16px;
                    padding:3rem; text-align:center; color:#475569; margin-top:1rem;">
            <div style="font-size:3rem;">📡</div>
            <div style="margin-top:1rem; font-size:1rem;">
                Fill in the customer details on the left<br>and click <strong>Predict Churn Risk</strong>.
            </div>
        </div>""", unsafe_allow_html=True)

        # ── Model info cards ──────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**About the Model**")
        i1, i2 = st.columns(2)
        i1.markdown("""
        <div class="metric-card">
            <div class="metric-title">Algorithm</div>
            <div class="metric-value metric-green" style="font-size:1rem;">XGBoost</div>
        </div>""", unsafe_allow_html=True)
        i2.markdown("""
        <div class="metric-card">
            <div class="metric-title">Threshold</div>
            <div class="metric-value metric-yellow" style="font-size:1rem;">0.45</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        i3, i4 = st.columns(2)
        i3.markdown("""
        <div class="metric-card">
            <div class="metric-title">Optimised For</div>
            <div class="metric-value" style="font-size:1rem; color:#93c5fd;">Recall</div>
        </div>""", unsafe_allow_html=True)
        i4.markdown("""
        <div class="metric-card">
            <div class="metric-title">Features</div>
            <div class="metric-value" style="font-size:1rem; color:#c4b5fd;">19</div>
        </div>""", unsafe_allow_html=True)
