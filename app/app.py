import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #080c14 !important;
    color: #d0d7e3;
    font-family: 'Inter', sans-serif;
}
[data-testid="stAppViewContainer"] > .main { padding: 0; }
[data-testid="stMainBlockContainer"] { padding: 0 !important; max-width: 100% !important; }
[data-testid="stSidebar"] { display: none; }
section.main > div { padding: 0 !important; }

/* ── Layout shell ── */
.shell {
    display: grid;
    grid-template-columns: 420px 1fr;
    min-height: 100vh;
}

/* ── LEFT PANEL ── */
.left-panel {
    background: #0d1220;
    border-right: 1px solid #1c2333;
    padding: 36px 28px 40px;
    overflow-y: auto;
}
.brand {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 32px;
}
.brand-icon {
    width: 40px; height: 40px;
    background: linear-gradient(135deg, #3b82f6, #6366f1);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.2rem;
}
.brand-name {
    font-size: 1rem;
    font-weight: 700;
    color: #e2e8f0;
    letter-spacing: -0.3px;
}
.brand-sub {
    font-size: 0.72rem;
    color: #4b5a72;
    margin-top: 1px;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.05em;
}

/* Section dividers */
.sec-head {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #3b82f6;
    margin: 24px 0 14px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.sec-head::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #1c2333;
}

/* ── RIGHT PANEL ── */
.right-panel {
    background: #080c14;
    padding: 48px 52px;
    display: flex;
    flex-direction: column;
}
.right-top {
    margin-bottom: 36px;
}
.page-eyebrow {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #3b82f6;
    margin-bottom: 10px;
    font-family: 'JetBrains Mono', monospace;
}
.page-title {
    font-size: 2.4rem;
    font-weight: 800;
    color: #e2e8f0;
    letter-spacing: -1px;
    line-height: 1.1;
    margin-bottom: 10px;
}
.page-desc {
    font-size: 0.9rem;
    color: #4b5a72;
    max-width: 480px;
    line-height: 1.6;
}

/* Result area */
.result-idle {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-direction: column;
    gap: 16px;
    color: #1c2840;
}
.result-idle .idle-icon { font-size: 4rem; opacity: 0.3; }
.result-idle .idle-text {
    font-size: 0.9rem;
    color: #2a3a52;
    font-family: 'JetBrains Mono', monospace;
}

/* Churn result card */
.result-churn, .result-stay {
    border-radius: 16px;
    padding: 36px 40px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.result-churn {
    background: linear-gradient(135deg, #120508 0%, #1a0812 100%);
    border: 1px solid #4a0a14;
}
.result-stay {
    background: linear-gradient(135deg, #020e09 0%, #061510 100%);
    border: 1px solid #0a3a1e;
}
.result-churn::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #ef4444, #f97316);
}
.result-stay::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #22c55e, #10b981);
}
.result-label {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 8px;
}
.result-churn .result-label { color: #f87171; }
.result-stay  .result-label { color: #4ade80; }

.verdict {
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -1px;
    margin-bottom: 6px;
}
.result-churn .verdict { color: #fca5a5; }
.result-stay  .verdict { color: #86efac; }

.verdict-sub {
    font-size: 0.85rem;
    color: #4b5a72;
    margin-bottom: 28px;
}

/* Big probability number */
.prob-display {
    display: flex;
    align-items: baseline;
    gap: 6px;
    margin-bottom: 12px;
}
.prob-number {
    font-size: 3.8rem;
    font-weight: 800;
    letter-spacing: -2px;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1;
}
.result-churn .prob-number { color: #ef4444; }
.result-stay  .prob-number { color: #22c55e; }
.prob-unit {
    font-size: 1.4rem;
    font-weight: 600;
    color: #4b5a72;
}
.prob-caption {
    font-size: 0.75rem;
    color: #4b5a72;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.05em;
    margin-bottom: 20px;
}

/* Progress bar */
.pbar-wrap { margin-bottom: 4px; }
.pbar-track {
    background: #111827;
    border-radius: 4px;
    height: 8px;
    overflow: hidden;
    margin-bottom: 6px;
}
.pbar-fill-churn {
    height: 8px;
    border-radius: 4px;
    background: linear-gradient(90deg, #ef4444, #f97316);
    transition: width 0.6s cubic-bezier(.4,0,.2,1);
}
.pbar-fill-stay {
    height: 8px;
    border-radius: 4px;
    background: linear-gradient(90deg, #22c55e, #10b981);
    transition: width 0.6s cubic-bezier(.4,0,.2,1);
}
.pbar-labels {
    display: flex;
    justify-content: space-between;
    font-size: 0.7rem;
    color: #2a3a52;
    font-family: 'JetBrains Mono', monospace;
}

/* Stats row */
.stats-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-top: 28px;
}
.stat-box {
    background: #0d1220;
    border: 1px solid #1c2333;
    border-radius: 10px;
    padding: 14px 16px;
}
.stat-box .s-label {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #2a3a52;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 6px;
}
.stat-box .s-val {
    font-size: 1.1rem;
    font-weight: 700;
    color: #e2e8f0;
}

/* Risk badge */
.risk-row { margin-top: 20px; }
.risk-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.05em;
}
.risk-low  { background: #052814; color: #4ade80; border: 1px solid #166534; }
.risk-med  { background: #1c1200; color: #fbbf24; border: 1px solid #78350f; }
.risk-high { background: #1a0508; color: #f87171; border: 1px solid #7f1d1d; }

/* Retention tip */
.tip-card {
    background: #0d1220;
    border: 1px solid #1c2333;
    border-left: 3px solid #3b82f6;
    border-radius: 0 10px 10px 0;
    padding: 16px 20px;
    margin-top: 20px;
    font-size: 0.85rem;
    color: #94a3b8;
    line-height: 1.6;
}
.tip-card strong { color: #93c5fd; }

/* Streamlit widget overrides */
div[data-testid="stSelectbox"] > label,
div[data-testid="stSlider"] > label,
div[data-testid="stNumberInput"] > label {
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    color: #64748b !important;
    letter-spacing: 0.01em !important;
    margin-bottom: 2px !important;
}
div[data-testid="stSelectbox"] > div > div {
    background: #0d1525 !important;
    border: 1px solid #1c2840 !important;
    border-radius: 8px !important;
    color: #d0d7e3 !important;
    font-size: 0.85rem !important;
}
div[data-testid="stNumberInput"] input {
    background: #0d1525 !important;
    border: 1px solid #1c2840 !important;
    border-radius: 8px !important;
    color: #d0d7e3 !important;
    font-size: 0.85rem !important;
}
div[data-testid="stSlider"] div[data-baseweb="slider"] div[role="slider"] {
    background: #3b82f6 !important;
}

/* Predict button */
div.stButton > button {
    background: linear-gradient(135deg, #2563eb, #4f46e5) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 14px 24px !important;
    font-size: 0.9rem !important;
    font-weight: 700 !important;
    width: 100% !important;
    letter-spacing: 0.02em !important;
    margin-top: 8px !important;
    transition: opacity 0.2s !important;
    box-shadow: 0 4px 20px rgba(37,99,235,0.35) !important;
}
div.stButton > button:hover { opacity: 0.88 !important; }

/* Hide streamlit chrome */
#MainMenu, footer, header { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("telecom_customer_churn_model_XGB.pkl")

try:
    model = load_model()
    model_ok = True
except FileNotFoundError:
    model_ok = False

# ── Two-column layout via columns ──────────────────────────────────────────────
col_form, col_result = st.columns([420, 780], gap="small")

# ═══════════════════════════════════════════════════════
# LEFT — INPUT FORM
# ═══════════════════════════════════════════════════════
with col_form:
    st.markdown("""
    <div class="left-panel">
      <div class="brand">
        <div class="brand-icon">📡</div>
        <div>
          <div class="brand-name">ChurnSense</div>
          <div class="brand-sub">XGBoost · v1.0</div>
        </div>
      </div>
    """, unsafe_allow_html=True)

    # — Demographics —
    st.markdown('<div class="sec-head">Demographics</div>', unsafe_allow_html=True)
    gender         = st.selectbox("Gender", ["Male", "Female"], key="gender")
    senior         = st.selectbox("Senior Citizen", ["No", "Yes"], key="senior")
    partner        = st.selectbox("Partner", ["No", "Yes"], key="partner")
    dependents     = st.selectbox("Dependents", ["No", "Yes"], key="dependents")

    # — Account —
    st.markdown('<div class="sec-head">Account</div>', unsafe_allow_html=True)
    tenure         = st.slider("Tenure (months)", 0, 72, 12, key="tenure")
    contract       = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], key="contract")
    paperless      = st.selectbox("Paperless Billing", ["No", "Yes"], key="paperless")
    payment        = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ], key="payment")

    # — Charges —
    st.markdown('<div class="sec-head">Charges</div>', unsafe_allow_html=True)
    monthly        = st.number_input("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5, key="monthly")
    total          = st.number_input("Total Charges ($)", 0.0, 9000.0,
                                     float(round(monthly * max(tenure, 1), 2)), step=1.0, key="total")

    # — Phone —
    st.markdown('<div class="sec-head">Phone Services</div>', unsafe_allow_html=True)
    phone_svc      = st.selectbox("Phone Service", ["Yes", "No"], key="phone")
    multi_lines    = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"], key="multilines")

    # — Internet —
    st.markdown('<div class="sec-head">Internet Services</div>', unsafe_allow_html=True)
    internet       = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], key="internet")
    no_inet        = "No internet service"
    inet_3         = ["No", "Yes", no_inet] if internet != "No" else [no_inet]

    online_sec     = st.selectbox("Online Security",   inet_3, key="osec")
    online_bkp     = st.selectbox("Online Backup",     inet_3, key="obkp")
    device_prot    = st.selectbox("Device Protection", inet_3, key="dprot")
    tech_supp      = st.selectbox("Tech Support",      inet_3, key="tsupp")
    stream_tv      = st.selectbox("Streaming TV",      inet_3, key="stv")
    stream_movies  = st.selectbox("Streaming Movies",  inet_3, key="smov")

    st.markdown("</div>", unsafe_allow_html=True)   # close left-panel div

    st.markdown("<br>", unsafe_allow_html=True)
    predict = st.button("Predict Churn →", key="predict_btn")

# ═══════════════════════════════════════════════════════
# RIGHT — RESULT
# ═══════════════════════════════════════════════════════
with col_result:
    st.markdown("""
    <div class="right-panel">
      <div class="right-top">
        <div class="page-eyebrow">Churn Intelligence</div>
        <div class="page-title">Will this customer<br>stay or leave?</div>
        <div class="page-desc">
          Fill in the customer profile on the left and click Predict.
          The model returns a churn probability and a risk verdict instantly.
        </div>
      </div>
    """, unsafe_allow_html=True)

    if not model_ok:
        st.error("**Model file not found.** Place `telecom_customer_churn_model_XGB.pkl` beside `app.py` and restart.")

    elif not predict:
        st.markdown("""
        <div class="result-idle">
          <div class="idle-icon">◎</div>
          <div class="idle-text">awaiting prediction…</div>
        </div>
        """, unsafe_allow_html=True)

    else:
        # Build input row
        input_df = pd.DataFrame([{
            "gender":           gender,
            "SeniorCitizen":    1 if senior == "Yes" else 0,
            "Partner":          partner,
            "Dependents":       dependents,
            "tenure":           tenure,
            "PhoneService":     phone_svc,
            "MultipleLines":    multi_lines,
            "InternetService":  internet,
            "OnlineSecurity":   online_sec,
            "OnlineBackup":     online_bkp,
            "DeviceProtection": device_prot,
            "TechSupport":      tech_supp,
            "StreamingTV":      stream_tv,
            "StreamingMovies":  stream_movies,
            "Contract":         contract,
            "PaperlessBilling": paperless,
            "PaymentMethod":    payment,
            "MonthlyCharges":   monthly,
            "TotalCharges":     total,
        }])

        THRESHOLD = 0.45
        prob  = model.predict_proba(input_df)[0, 1]
        churn = prob >= THRESHOLD
        pct   = round(prob * 100, 1)
        stay  = round(100 - pct, 1)

        if pct < 30:
            badge_cls, risk_label, risk_dot = "risk-low",  "Low Risk",    "🟢"
        elif pct < 55:
            badge_cls, risk_label, risk_dot = "risk-med",  "Medium Risk", "🟡"
        else:
            badge_cls, risk_label, risk_dot = "risk-high", "High Risk",   "🔴"

        if churn:
            card_cls    = "result-churn"
            verdict_txt = "Will Churn"
            verdict_sub = "Customer is likely to cancel their subscription."
            fill_cls    = "pbar-fill-churn"
        else:
            card_cls    = "result-stay"
            verdict_txt = "Will Stay"
            verdict_sub = "Customer is likely to remain active next cycle."
            fill_cls    = "pbar-fill-stay"

        # Retention tip
        if churn:
            if contract == "Month-to-month":
                tip = "Offer a discounted <strong>annual or two-year contract</strong> — monthly customers churn at 3× the rate."
            elif tenure < 12:
                tip = "New customer at risk. Send a <strong>proactive onboarding check-in</strong> or loyalty welcome reward."
            elif internet == "Fiber optic" and monthly > 80:
                tip = "High fiber bill is a churn driver. A <strong>bundle discount or speed upgrade</strong> could retain them."
            elif tech_supp == "No":
                tip = "Customer has no tech support. Offer <strong>complimentary support access</strong> for 3 months."
            else:
                tip = "Assign a dedicated <strong>customer success manager</strong> and schedule a retention call this week."
            tip_html = f'<div class="tip-card">💡 {tip}</div>'
        else:
            tip_html = '<div class="tip-card">✅ <strong>No action needed.</strong> Customer is stable — monitor at the next billing cycle.</div>'

        st.markdown(f"""
        <div class="{card_cls}">
          <div class="result-label">Prediction Result</div>
          <div class="verdict">{verdict_txt}</div>
          <div class="verdict-sub">{verdict_sub}</div>

          <div class="prob-display">
            <div class="prob-number">{pct}</div>
            <div class="prob-unit">%</div>
          </div>
          <div class="prob-caption">churn probability (threshold {int(THRESHOLD*100)}%)</div>

          <div class="pbar-wrap">
            <div class="pbar-track">
              <div class="{fill_cls}" style="width:{pct}%"></div>
            </div>
            <div class="pbar-labels"><span>0%</span><span>50%</span><span>100%</span></div>
          </div>

          <div class="stats-row">
            <div class="stat-box">
              <div class="s-label">Churn Prob</div>
              <div class="s-val">{pct}%</div>
            </div>
            <div class="stat-box">
              <div class="s-label">Stay Prob</div>
              <div class="s-val">{stay}%</div>
            </div>
            <div class="stat-box">
              <div class="s-label">Tenure</div>
              <div class="s-val">{tenure} mo</div>
            </div>
          </div>

          <div class="risk-row">
            <span class="risk-badge {badge_cls}">{risk_dot} {risk_label}</span>
          </div>

          {tip_html}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # close right-panel

