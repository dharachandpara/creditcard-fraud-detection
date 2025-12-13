import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="🛡️", layout="wide")

st.title("🛡️ Credit Card Fraud Detection")
st.caption("Portfolio demo by Dhara Chandpara, upload transactions, score fraud probability, download results.")

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

EXPECTED_COLS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

st.sidebar.header("Settings")
threshold = st.sidebar.slider("Fraud threshold", 0.01, 0.99, 0.50, 0.01)

uploaded = st.file_uploader("Upload a CSV (Time, V1..V28, Amount)", type=["csv"])

if not uploaded:
    st.info("Upload a CSV to begin. If you use the Kaggle dataset, keep columns: Time, V1..V28, Amount.")
    st.stop()

df = pd.read_csv(uploaded)

# If the file contains the label column, drop it for prediction
if "Class" in df.columns:
    df = df.drop(columns=["Class"])

st.subheader("Data preview")
st.dataframe(df.head(20), use_container_width=True)

missing = [c for c in EXPECTED_COLS if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

X = df[EXPECTED_COLS].copy()

if hasattr(model, "predict_proba"):
    proba = model.predict_proba(X)[:, 1]
else:
    proba = model.predict(X)

pred = (proba >= threshold).astype(int)

out = df.copy()
out["fraud_probability"] = proba
out["fraud_pred"] = pred

st.subheader("Results")
c1, c2, c3 = st.columns(3)
c1.metric("Rows scored", len(out))
c2.metric("Flagged fraud", int(out["fraud_pred"].sum()))
c3.metric("Flag rate", f"{(out['fraud_pred'].mean() * 100):.2f}%")

st.dataframe(out.head(50), use_container_width=True)

st.download_button(
    "Download predictions CSV",
    out.to_csv(index=False).encode("utf-8"),
    file_name="fraud_predictions.csv",
    mime="text/csv",
)

st.markdown("---")
st.markdown("**Note:** This is a demo model for learning/portfolio purposes.")
