import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

HERE = Path(__file__).parent
model = joblib.load(HERE / "model.joblib")
threshold = json.load(open(HERE / "config.json"))["threshold"]
ex = joblib.load(HERE / "explain.joblib")

prep = model.named_steps["prep"]
coef = model.named_steps["model"].coef_[0]

st.set_page_config(page_title="Churn Risk", page_icon="📉")
st.title("📉 Customer churn risk")
st.caption("Enter a customer's details to get a churn probability and the main reasons.")

c1, c2 = st.columns(2)
with c1:
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly = st.slider("Monthly charges", 18.0, 120.0, 70.0)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
    payment = st.selectbox("Payment method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"])
    paperless = st.selectbox("Paperless billing", ["Yes", "No"])
with c2:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])
    phone = st.selectbox("Phone service", ["Yes", "No"])

    def multiple_lines():
        return "No phone service" if phone == "No" else st.selectbox("Multiple lines", ["No", "Yes"])

    def addon(label):
        return "No internet service" if internet == "No" else st.selectbox(label, ["No", "Yes"])

    lines = multiple_lines()
    security = addon("Online security")
    backup = addon("Online backup")
    protection = addon("Device protection")
    tech = addon("Tech support")
    tv = addon("Streaming TV")
    movies = addon("Streaming movies")

row = pd.DataFrame([{
    "gender": gender, "SeniorCitizen": senior, "Partner": partner,
    "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
    "MultipleLines": lines, "InternetService": internet,
    "OnlineSecurity": security, "OnlineBackup": backup,
    "DeviceProtection": protection, "TechSupport": tech,
    "StreamingTV": tv, "StreamingMovies": movies, "Contract": contract,
    "PaperlessBilling": paperless, "PaymentMethod": payment,
    "MonthlyCharges": monthly,
    "TotalCharges": tenure * monthly,  # approximation: not asked from the user
}])

p = model.predict_proba(row)[0, 1]

st.divider()
st.metric("Churn probability", f"{p:.0%}")
if p >= threshold:
    st.error(f"High risk (above the {threshold:.0%} cutoff): recommend a retention offer.")
else:
    st.success(f"Lower risk (below the {threshold:.0%} cutoff): no offer needed.")

# Reasons: coefficient * (value - training mean), grouped by original feature
x = prep.transform(row)
x = x.toarray()[0] if hasattr(x, "toarray") else x[0]
contrib = pd.Series(coef * (x - ex["means"])).groupby(ex["groups"]).sum()
top = contrib.reindex(contrib.abs().sort_values(ascending=False).index)[:5]

st.subheader("Top reasons for this score")
for feature, value in top.items():
    arrow = "🔴 pushes toward churn" if value > 0 else "🟢 pushes toward staying"
    st.write(f"**{feature}**: {arrow} ({value:+.2f})")
st.caption("Contributions are in log-odds. Tenure, monthly and total charges are correlated, so read them together.")