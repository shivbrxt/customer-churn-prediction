import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, OneHotEncoder
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    average_precision_score, RocCurveDisplay, PrecisionRecallDisplay
)

import warnings
warnings.filterwarnings("ignore")

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Telecom Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f1117; }
    .main-header {
        background: linear-gradient(135deg, #1a1f35 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid #2d3561;
    }
    .main-header h1 { color: #e94560; margin: 0; font-size: 2.2rem; }
    .main-header p  { color: #8892b0; margin: 0.4rem 0 0; font-size: 1rem; }

    .metric-card {
        background: #1a1f35;
        border: 1px solid #2d3561;
        border-radius: 12px;
        padding: 1.2rem 1rem;
        text-align: center;
    }
    .metric-card .label { color: #8892b0; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; }
    .metric-card .value { color: #e94560; font-size: 2rem; font-weight: 700; margin: 0.3rem 0 0; }

    .churn-alert {
        background: linear-gradient(135deg, #3d0000, #6b0000);
        border: 1px solid #e94560;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        color: #fff;
    }
    .no-churn-alert {
        background: linear-gradient(135deg, #003d1a, #006b2f);
        border: 1px solid #00c96b;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        color: #fff;
    }
    .section-header {
        color: #e94560;
        font-size: 1.1rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 1.5rem 0 0.8rem;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid #2d3561;
    }
    div[data-testid="stSidebarContent"] { background-color: #0d1117; }
    .stSelectbox label, .stSlider label, .stNumberInput label { color: #8892b0 !important; }
    .stButton>button {
        background: linear-gradient(135deg, #e94560, #c73652);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        width: 100%;
    }
    .stButton>button:hover { background: linear-gradient(135deg, #c73652, #a62d44); }
    .stTabs [data-baseweb="tab"] { color: #8892b0; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { color: #e94560; border-bottom-color: #e94560; }
</style>
""", unsafe_allow_html=True)

# ── Column Definitions ───────────────────────────────────────────────────────
NUMERIC_COLS  = ["tenure", "MonthlyCharges", "TotalCharges"]
BINARY_COLS   = ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "SeniorCitizen"]
NOMINAL_COLS  = [
    "gender", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaymentMethod"
]
THRESHOLD     = 0.45

# ── Data & Model Helpers ─────────────────────────────────────────────────────
@st.cache_data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df = df.drop(columns=["customerID"], errors="ignore")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"] * df["tenure"])
    le = LabelEncoder()
    df["Churn"] = le.fit_transform(df["Churn"])
    return df

@st.cache_resource
def train_model(df):
    X = df.drop(["Churn"], axis=1)
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    numeric_transformer = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    binary_transformer  = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])
    nominal_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, NUMERIC_COLS),
        ("bin", binary_transformer,  BINARY_COLS),
        ("nom", nominal_transformer, NOMINAL_COLS)
    ], remainder="drop")

    scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

    model_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        ))
    ])
    model_pipeline.fit(X_train, y_train)
    return model_pipeline, X_test, y_test

def get_metrics(model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= THRESHOLD).astype(int)
    return {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "f1":        f1_score(y_test, y_pred),
        "roc_auc":   roc_auc_score(y_test, y_prob),
        "pr_auc":    average_precision_score(y_test, y_prob),
        "y_prob":    y_prob,
        "y_pred":    y_pred,
    }

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📡 Churn Predictor")
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload Telco CSV", type=["csv"],
                                     help="Upload Telco-Customer-Churn.csv")
    st.markdown("---")
    st.markdown("**Prediction Threshold**")
    threshold_val = st.slider("", 0.1, 0.9, THRESHOLD, 0.05,
                              help="Lower = catch more churners (higher recall)")
    st.markdown("---")
    st.markdown("""
    <div style='color:#8892b0;font-size:0.8rem'>
    <b style='color:#e94560'>Model:</b> XGBoost Classifier<br>
    <b style='color:#e94560'>Target:</b> Customer Churn<br>
    <b style='color:#e94560'>Default Threshold:</b> 0.45
    </div>
    """, unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>📡 Telecom Customer Churn Prediction</h1>
  <p>XGBoost-powered churn analytics — upload your data, train the model, and predict at-risk customers.</p>
</div>
""", unsafe_allow_html=True)

# ── No File State ─────────────────────────────────────────────────────────────
if uploaded_file is None:
    st.info("👈  Upload your **Telco-Customer-Churn.csv** in the sidebar to get started.")
    with st.expander("📋 Expected CSV Columns"):
        st.code("""customerID, gender, SeniorCitizen, Partner, Dependents, tenure,
PhoneService, MultipleLines, InternetService, OnlineSecurity,
OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies,
Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges, Churn""")
    st.stop()

# ── Load & Train ─────────────────────────────────────────────────────────────
df = load_data(uploaded_file)

with st.spinner("🔧 Training XGBoost model..."):
    model, X_test, y_test = train_model(df)

# Recalculate metrics with sidebar threshold
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= threshold_val).astype(int)
metrics = {
    "accuracy":  accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall":    recall_score(y_test, y_pred),
    "f1":        f1_score(y_test, y_pred),
    "roc_auc":   roc_auc_score(y_test, y_prob),
    "pr_auc":    average_precision_score(y_test, y_prob),
    "y_prob": y_prob,
    "y_pred": y_pred,
}

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Model Performance", "🔍 Predict Customer", "📈 Data Insights", "⭐ Feature Importance"])

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 – Model Performance
# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### Model Metrics")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    for col, label, key in zip(
        [c1, c2, c3, c4, c5, c6],
        ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "PR-AUC"],
        ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    ):
        col.markdown(f"""
        <div class="metric-card">
          <div class="label">{label}</div>
          <div class="value">{metrics[key]:.3f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_cm, col_curves = st.columns([1, 2])

    with col_cm:
        st.markdown("**Confusion Matrix**")
        fig, ax = plt.subplots(figsize=(4, 3.5))
        fig.patch.set_facecolor("#1a1f35")
        ax.set_facecolor("#1a1f35")
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", ax=ax,
                    xticklabels=["No Churn", "Churn"],
                    yticklabels=["No Churn", "Churn"])
        ax.set_xlabel("Predicted", color="#8892b0")
        ax.set_ylabel("Actual", color="#8892b0")
        ax.tick_params(colors="#8892b0")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_curves:
        st.markdown("**ROC Curve & PR Curve**")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))
        for ax in (ax1, ax2):
            fig.patch.set_facecolor("#1a1f35")
            ax.set_facecolor("#1a1f35")
            ax.tick_params(colors="#8892b0")
            ax.xaxis.label.set_color("#8892b0")
            ax.yaxis.label.set_color("#8892b0")
            ax.title.set_color("#e94560")
            for spine in ax.spines.values():
                spine.set_edgecolor("#2d3561")

        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax1,
                                       color="#e94560", name="XGBoost")
        ax1.set_title("ROC Curve")
        ax1.grid(True, alpha=0.2)

        PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax2,
                                              color="#e94560", name="XGBoost")
        ax2.set_title("Precision-Recall Curve")
        ax2.grid(True, alpha=0.2)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with st.expander("📋 Full Classification Report"):
        report = classification_report(y_test, y_pred, target_names=["No Churn", "Churn"])
        st.code(report)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 – Predict Customer
# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### Predict Churn for a Single Customer")
    st.markdown("Fill in the customer details below and click **Predict**.")

    with st.form("predict_form"):
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.markdown('<div class="section-header">Demographics</div>', unsafe_allow_html=True)
            gender         = st.selectbox("Gender", ["Male", "Female"])
            senior         = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner        = st.selectbox("Partner", ["Yes", "No"])
            dependents     = st.selectbox("Dependents", ["Yes", "No"])

            st.markdown('<div class="section-header">Account</div>', unsafe_allow_html=True)
            tenure         = st.slider("Tenure (months)", 0, 72, 12)
            contract       = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless      = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment        = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"
            ])

        with col_b:
            st.markdown('<div class="section-header">Phone Services</div>', unsafe_allow_html=True)
            phone_svc      = st.selectbox("Phone Service", ["Yes", "No"])
            multi_lines    = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

            st.markdown('<div class="section-header">Internet Services</div>', unsafe_allow_html=True)
            internet_svc   = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
            online_sec     = st.selectbox("Online Security",   ["No", "Yes", "No internet service"])
            online_bkp     = st.selectbox("Online Backup",     ["No", "Yes", "No internet service"])
            device_prot    = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            tech_supp      = st.selectbox("Tech Support",      ["No", "Yes", "No internet service"])
            stream_tv      = st.selectbox("Streaming TV",      ["No", "Yes", "No internet service"])
            stream_movies  = st.selectbox("Streaming Movies",  ["No", "Yes", "No internet service"])

        with col_c:
            st.markdown('<div class="section-header">Charges</div>', unsafe_allow_html=True)
            monthly        = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0, 0.5)
            total          = st.number_input("Total Charges ($)",   0.0, 10000.0, monthly * max(tenure, 1), 1.0)

            st.markdown("<br><br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("🔮 Predict Churn")

    if submitted:
        input_data = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_svc,
            "MultipleLines": multi_lines,
            "InternetService": internet_svc,
            "OnlineSecurity": online_sec,
            "OnlineBackup": online_bkp,
            "DeviceProtection": device_prot,
            "TechSupport": tech_supp,
            "StreamingTV": stream_tv,
            "StreamingMovies": stream_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
        }])

        prob = model.predict_proba(input_data)[0, 1]
        pred = int(prob >= threshold_val)

        col_res, col_gauge = st.columns([1, 1])
        with col_res:
            if pred == 1:
                st.markdown(f"""
                <div class="churn-alert">
                  <h2>⚠️ HIGH CHURN RISK</h2>
                  <p>This customer is likely to churn.</p>
                  <h3>Churn Probability: {prob:.1%}</h3>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="no-churn-alert">
                  <h2>✅ LOW CHURN RISK</h2>
                  <p>This customer is likely to stay.</p>
                  <h3>Churn Probability: {prob:.1%}</h3>
                </div>""", unsafe_allow_html=True)

        with col_gauge:
            fig, ax = plt.subplots(figsize=(4, 3))
            fig.patch.set_facecolor("#1a1f35")
            ax.set_facecolor("#1a1f35")
            color = "#e94560" if pred == 1 else "#00c96b"
            ax.barh(["Churn Prob"], [prob], color=color, height=0.4)
            ax.barh(["Churn Prob"], [1 - prob], left=[prob], color="#2d3561", height=0.4)
            ax.axvline(threshold_val, color="white", linestyle="--", alpha=0.6, label=f"Threshold ({threshold_val})")
            ax.set_xlim(0, 1)
            ax.set_xlabel("Probability", color="#8892b0")
            ax.tick_params(colors="#8892b0")
            ax.legend(fontsize=8, labelcolor="#8892b0", facecolor="#1a1f35")
            for spine in ax.spines.values():
                spine.set_edgecolor("#2d3561")
            ax.set_title(f"Score: {prob:.3f}", color="#8892b0")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # Retention Tips
        if pred == 1:
            st.markdown("### 💡 Retention Recommendations")
            tips = []
            if contract == "Month-to-month":
                tips.append("📄 Offer a discounted **annual or two-year contract**")
            if internet_svc == "Fiber optic" and monthly > 80:
                tips.append("💸 Consider a **loyalty discount** on fiber plan")
            if tech_supp == "No":
                tips.append("🛠️ Offer **free tech support** for 3 months")
            if online_sec == "No":
                tips.append("🔒 Bundle **online security** at no extra cost")
            if payment == "Electronic check":
                tips.append("💳 Incentivise switching to **automatic payment**")
            if tenure < 12:
                tips.append("🎁 Send a **new customer loyalty reward**")
            if not tips:
                tips.append("📞 Assign a dedicated **customer success manager**")
            for tip in tips:
                st.markdown(f"- {tip}")

# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 – Data Insights
# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    churn_rate = df["Churn"].mean()
    c1.metric("Total Customers", f"{len(df):,}")
    c2.metric("Churn Rate",      f"{churn_rate:.1%}")
    c3.metric("Features",        str(df.shape[1] - 1))
    c4.metric("Avg Tenure",      f"{df['tenure'].mean():.1f} mo")

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("**Churn Distribution**")
        fig, ax = plt.subplots(figsize=(4, 3))
        fig.patch.set_facecolor("#1a1f35")
        ax.set_facecolor("#1a1f35")
        counts = df["Churn"].value_counts()
        ax.bar(["No Churn", "Churn"], counts.values, color=["#00c96b", "#e94560"])
        ax.set_ylabel("Count", color="#8892b0")
        ax.tick_params(colors="#8892b0")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2d3561")
        for i, v in enumerate(counts.values):
            ax.text(i, v + 20, str(v), ha="center", color="#8892b0", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_r:
        st.markdown("**Tenure Distribution by Churn**")
        fig, ax = plt.subplots(figsize=(4, 3))
        fig.patch.set_facecolor("#1a1f35")
        ax.set_facecolor("#1a1f35")
        df[df["Churn"] == 0]["tenure"].plot(kind="hist", ax=ax, bins=30, alpha=0.6,
                                              color="#00c96b", label="No Churn")
        df[df["Churn"] == 1]["tenure"].plot(kind="hist", ax=ax, bins=30, alpha=0.6,
                                              color="#e94560", label="Churn")
        ax.legend(facecolor="#1a1f35", labelcolor="#8892b0")
        ax.set_xlabel("Tenure (months)", color="#8892b0")
        ax.tick_params(colors="#8892b0")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2d3561")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("**Monthly Charges vs Churn**")
    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_facecolor("#1a1f35")
    ax.set_facecolor("#1a1f35")
    df[df["Churn"] == 0]["MonthlyCharges"].plot(kind="hist", ax=ax, bins=30, alpha=0.7,
                                                  color="#00c96b", label="No Churn")
    df[df["Churn"] == 1]["MonthlyCharges"].plot(kind="hist", ax=ax, bins=30, alpha=0.7,
                                                  color="#e94560", label="Churn")
    ax.legend(facecolor="#1a1f35", labelcolor="#8892b0")
    ax.set_xlabel("Monthly Charges ($)", color="#8892b0")
    ax.tick_params(colors="#8892b0")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2d3561")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("**Contract Type vs Churn Rate**")
    contract_churn = df.groupby("Contract")["Churn"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6, 2.5))
    fig.patch.set_facecolor("#1a1f35")
    ax.set_facecolor("#1a1f35")
    ax.barh(contract_churn.index, contract_churn.values, color=["#e94560", "#f4a261", "#00c96b"])
    ax.set_xlabel("Churn Rate", color="#8892b0")
    ax.tick_params(colors="#8892b0")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2d3561")
    for i, v in enumerate(contract_churn.values):
        ax.text(v + 0.005, i, f"{v:.1%}", va="center", color="#8892b0", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    with st.expander("📄 Raw Data Sample"):
        st.dataframe(df.head(20), use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 4 – Feature Importance
# ──────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("### Top 20 Predictive Features")
    xgb_importances  = model.named_steps["model"].feature_importances_
    feature_names     = model.named_steps["preprocessor"].get_feature_names_out()
    importance_df     = pd.DataFrame({
        "Feature": feature_names,
        "Importance": xgb_importances
    }).sort_values("Importance", ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#1a1f35")
    ax.set_facecolor("#1a1f35")
    colors = [
        "#e94560" if i < 5 else "#f4a261" if i < 10 else "#8892b0"
        for i in range(len(importance_df))
    ]
    ax.barh(importance_df["Feature"][::-1], importance_df["Importance"][::-1], color=colors[::-1])
    ax.set_xlabel("Feature Importance (Gain)", color="#8892b0")
    ax.tick_params(colors="#8892b0")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2d3561")
    ax.set_title("XGBoost Feature Importances (Top 20)", color="#e94560", pad=12)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("**Full Importance Table**")
    st.dataframe(
        importance_df.reset_index(drop=True).style.background_gradient(
            subset=["Importance"], cmap="Reds"
        ),
        use_container_width=True
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#8892b0;font-size:0.8rem'>"
    "Telecom Churn Predictor · XGBoost · Threshold = "
    f"{threshold_val:.2f}</p>",
    unsafe_allow_html=True
)
