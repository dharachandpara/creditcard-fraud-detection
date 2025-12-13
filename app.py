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

# If the file contains the label column, drop it for prediction
if "Class" in df.columns:
    df = df.drop(columns=["Class"])

# Normalize column names to strings
df.columns = [str(c) for c in df.columns]

# Use the exact feature names the model was trained on (prevents mismatch errors)
model_expected_cols = None
if hasattr(model, "feature_names_in_"):
    model_expected_cols = list(model.feature_names_in_)

# Fallback to full Kaggle schema if feature_names_in_ is not available
fallback_cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
required_cols = model_expected_cols if model_expected_cols else fallback_cols

missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(
        "Your uploaded CSV does not match the columns this trained model expects.\n\n"
        f"Missing required columns: {missing}\n\n"
        "Tip: Upload the Kaggle creditcard.csv file, and do not rename columns."
    )
    st.stop()

# Select and order columns exactly as expected by the model
X = df[required_cols].copy()


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
