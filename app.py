"""Credit Scoring AI - Live Demo.   Run:  streamlit run app.py"""
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from features import add_features

st.set_page_config(page_title="Credit Scoring AI", page_icon="💳", layout="wide")
st.markdown("""
<style>
.block-container{padding-top:2rem;max-width:1200px}
.hero{background:linear-gradient(135deg,#1e3a8a,#2563eb);padding:1.5rem 2rem;border-radius:14px;color:#fff;margin-bottom:1rem}
.hero h1{margin:0;font-size:2rem;color:#fff}.hero p{margin:.3rem 0 0;opacity:.9}
.card{padding:1.3rem 1.6rem;border-radius:14px;border:1px solid}
.low{background:#ecfdf5;border-color:#10b981;color:#065f46}
.mid{background:#fffbeb;border-color:#f59e0b;color:#92400e}
.high{background:#fef2f2;border-color:#ef4444;color:#991b1b}
.score{font-size:3.2rem;font-weight:800;line-height:1}
.track{background:#e5e7eb;border-radius:999px;height:14px;margin-top:.9rem}
.fill{height:14px;border-radius:999px}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def load():
    with open("models/meta.json") as f:
        return joblib.load("models/credit_model.joblib"), json.load(f)


try:
    model, meta = load()
except FileNotFoundError:
    st.error("Model nahi mila. Pehle `python train.py` chalao.")
    st.stop()

RAW = meta["raw_columns"]
BASE = {c: meta["cat_mode"].get(c, meta["num_ranges"].get(c, [0, 0, 0])[2]) for c in RAW}
LABELS = {"duration": "Loan duration (months)", "credit_amount": "Loan amount",
          "installment_commitment": "Installment rate (% of income)",
          "checking_status": "Checking account", "savings_status": "Savings account",
          "other_parties": "Guarantors", "existing_credits": "Existing credits",
          "residence_since": "Years at residence", "num_dependents": "Dependents"}
GROUPS = {"📄 Loan details": ["duration", "credit_amount", "purpose", "installment_commitment", "other_payment_plans"],
          "💰 Financial history": ["checking_status", "savings_status", "credit_history", "existing_credits",
                                  "property_magnitude", "other_parties"],
          "👤 Personal profile": ["age", "personal_status", "employment", "job", "housing", "residence_since",
                                 "num_dependents", "own_telephone", "foreign_worker"]}
lab = lambda c: LABELS.get(c, c.replace("_", " ").title())


def widget(c):
    if c in meta["cat_options"]:
        return st.selectbox(lab(c), meta["cat_options"][c], key=c)
    lo, hi, med = meta["num_ranges"][c]
    if hi - lo > 100:
        return st.number_input(lab(c), 0.0, float(hi * 2), float(med), 100.0, key=c)
    return st.slider(lab(c), int(lo), int(hi), int(med), key=c)


def prob(df):
    return model.predict_proba(add_features(df[RAW]))[:, 1]


def score(p):  # illustrative 300-850 scale (lower default prob = higher score)
    return np.clip(850 - 550 * np.asarray(p), 300, 850).astype(int)


def band(p, thr):
    if p >= thr:
        return "high", "🔴 Reject / manual review"
    if p >= thr * 0.6:
        return "mid", "🟡 Approve with extra verification"
    return "low", "🟢 Approve"


def drivers(row):
    """Local explanation: har feature ko 'typical' value se badal ke dekho risk kitna badla."""
    alts = []
    for c in RAW:
        a = row.copy()
        a[c] = BASE[c]
        alts.append(a)
    p0 = prob(pd.DataFrame([row]))[0]
    d = p0 - prob(pd.DataFrame(alts))
    out = pd.Series(d, index=[lab(c) for c in RAW]).sort_values()
    return p0, pd.concat([out.head(4), out.tail(4)])


# ---------------- sidebar
best = meta["metrics"][meta["best_model"]]
with st.sidebar:
    st.header("Model card")
    st.write(f"**{meta['best_model']}**")
    st.metric("ROC-AUC", f"{best['ROC-AUC']:.3f}")
    st.metric("Recall (defaulters)", f"{best['Recall']:.3f}")
    st.metric("Precision", f"{best['Precision']:.3f}")
    st.divider()
    thr = st.slider("Risk threshold", 0.05, 0.95, float(meta["cost_threshold"]), 0.01,
                    help="Default = business-cost optimal threshold (FN costs 5x FP).")
    st.caption(f"Cost-optimal threshold: **{meta['cost_threshold']:.2f}**")
    st.caption("CodeAlpha ML Internship · Task 1 · German Credit (UCI)")

st.markdown('<div class="hero"><h1>💳 Credit Scoring AI</h1>'
            '<p>Credit score, risk decision aur explanation - ek jagah.</p></div>', unsafe_allow_html=True)
t1, t2, t3 = st.tabs(["🔮 Single applicant", "📂 Batch scoring", "📊 Model performance"])

# ---------------- single
with t1:
    row = {}
    for title, cols in GROUPS.items():
        cols = [c for c in cols if c in RAW]
        with st.expander(title, expanded=True):
            g = st.columns(3)
            for i, c in enumerate(cols):
                with g[i % 3]:
                    row[c] = widget(c)
    if st.button("Analyse applicant", type="primary"):
        p, drv = drivers(row)
        css, decision = band(p, thr)
        color = {"high": "#ef4444", "mid": "#f59e0b", "low": "#10b981"}[css]
        a, b = st.columns([1, 1.3])
        with a:
            sc = int(score(p))
            L = 3.14159 * 90
            fill = (sc - 300) / 550 * L
            st.markdown(f'''<div class="card {css}" style="text-align:center">
            <svg viewBox="0 0 200 118" width="100%" style="max-width:340px">
              <path d="M10 100 A90 90 0 0 1 190 100" fill="none" stroke="#e5e7eb" stroke-width="16" stroke-linecap="round"/>
              <path d="M10 100 A90 90 0 0 1 190 100" fill="none" stroke="{color}" stroke-width="16" stroke-linecap="round"
                    stroke-dasharray="{fill:.1f} {L:.1f}"/>
              <text x="100" y="88" text-anchor="middle" font-size="40" font-weight="800" fill="{color}">{sc}</text>
              <text x="100" y="108" text-anchor="middle" font-size="9" fill="#6b7280">CREDIT SCORE  (300 - 850)</text>
            </svg>
            <div style="font-size:1.15rem"><b>{decision}</b></div>
            <div>Default probability <b>{p:.1%}</b> · threshold {thr:.0%}</div></div>''', unsafe_allow_html=True)
        with b:
            st.markdown("**Why this result?** (risk change vs typical applicant)")
            st.bar_chart(drv * 100, horizontal=True, color="#2563eb")
            st.caption("Positive = ye feature risk badha raha hai · Negative = risk ghata raha hai (percentage points)")

# ---------------- batch
with t2:
    st.write("Bahut saare applicants ek saath score karo. Pehle template download karo, bharo, phir upload karo.")
    st.download_button("⬇️ CSV template", pd.DataFrame([BASE]).to_csv(index=False), "template.csv", "text/csv")
    up = st.file_uploader("Applicants CSV", type="csv")
    if up:
        df = pd.read_csv(up)
        miss = [c for c in RAW if c not in df.columns]
        if miss:
            st.error(f"Ye columns missing hain: {', '.join(miss)}")
        else:
            p = prob(df)
            res = df.copy()
            res["default_probability"] = p.round(3)
            res["credit_score"] = score(p)
            res["decision"] = [band(x, thr)[1] for x in p]
            c = st.columns(3)
            c[0].metric("Applicants", len(res))
            c[1].metric("Avg score", int(res["credit_score"].mean()))
            c[2].metric("High risk", f"{(p >= thr).mean():.0%}")
            st.dataframe(res)
            st.download_button("⬇️ Download results", res.to_csv(index=False), "scored.csv", "text/csv")

# ---------------- performance
with t3:
    st.dataframe(pd.DataFrame(meta["metrics"]).T.round(3))
    st.caption(f"Best model ROC-AUC se chuna gaya. Cost matrix: missed defaulter = {meta['cost_matrix']['FN']}x, "
               f"wrongly rejected good customer = {meta['cost_matrix']['FP']}x.")
    x, y = st.columns(2)
    x.image("outputs/roc_curve.png", caption="ROC curves")
    y.image("outputs/cost_curve.png", caption="Cost vs threshold")
    z, w = st.columns(2)
    z.image("outputs/feature_importance.png", caption="Feature importance (Random Forest)")
    w.image("outputs/confusion_matrix.png", caption="Confusion matrix (best model)")
    q, _ = st.columns(2)
    q.image("outputs/calibration.png", caption="Calibration: predicted vs observed default rate")
