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

# Sidebar controls
st.sidebar.header("Settings")
threshold = st.sidebar.slider("Fraud threshold", 0.0, 1.0, 0.50, 0.01)

# Upload
uploaded = st.file_uploader("Upload a CSV (Time, V1..V28, Amount)", type=["csv"])

if uploaded is None:
    st.info("Upload a CSV to begin. If you use the Kaggle dataset, keep columns: Time, V1..V28, Amount.")
    st.stop()

# Read CSV
df = pd.read_csv(uploaded)

max_rows = st.sidebar.number_input("Max rows to score", min_value=100, max_value=50000, value=5000, step=100)
df = df.head(int(max_rows))


# If label exists, drop it
if "Class" in df.columns:
    df = df.drop(columns=["Class"])

# Normalize column names
df.columns = [str(c) for c in df.columns]

# Determine expected column order
fallback_cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
expected_cols = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else fallback_cols

missing = [c for c in expected_cols if c not in df.columns]
if missing:
    st.error(
        "Your uploaded CSV does not match what this trained model expects.\n\n"
        f"Missing required columns: {missing}\n\n"
        "Tip: Upload the Kaggle creditcard.csv file and do not rename columns."
    )
    st.stop()

X = df[expected_cols].copy()

# Predict probabilities
if hasattr(model, "predict_proba"):
    proba = model.predict_proba(X)[:, 1]
else:
    # fallback if model does not support predict_proba
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
st.markdown("**Note:** Demo model for learning/portfolio purposes.")
