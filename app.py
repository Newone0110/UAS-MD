"""Antarmuka web Streamlit untuk prediksi credit score.
Memuat model hasil training SageMaker (model.joblib) dan menampilkan hasil prediksi."""
import joblib
import pandas as pd
import streamlit as st
from aws_pipeline import DataPreprocessor

st.set_page_config(page_title="Credit Score Classifier", layout="wide")
st.markdown("""<style>
.section-label{text-transform:uppercase;letter-spacing:1px;font-size:.85rem;color:#8B8F9A;
font-weight:600;margin:1.75rem 0 1rem 0;padding-bottom:.5rem;border-bottom:1px solid #2A2E39;}
</style>""", unsafe_allow_html=True)

@st.cache_resource
def load_bundle():
    return joblib.load("model.joblib")

def predict(row):
    bundle = load_bundle()
    pipe, inv = bundle["pipeline"], bundle["label_map"]
    df = DataPreprocessor().clean_raw(pd.DataFrame([row]))
    proba = pipe.predict_proba(df)[0]
    return {"prediction": inv[int(proba.argmax())],
            "probabilities": {inv[i]: round(float(p), 4) for i, p in enumerate(proba)}}

st.title("Credit Score Classifier")
st.caption("Prediksi performa kredit nasabah berdasarkan data keuangan. Powered by AWS SageMaker.")
st.divider()

c1, c2 = st.columns(2, gap="large")
with c1:
    st.markdown('<p class="section-label">Profil Nasabah</p>', unsafe_allow_html=True)
    age = st.number_input("Age", 18, 100, 35)
    occupation = st.selectbox("Occupation",
        ["Engineer","Lawyer","Architect","Developer","Doctor","Teacher","Accountant",
         "Entrepreneur","Scientist","Mechanic","Musician","Writer","Journalist","Media_Manager"])
    annual_income = st.number_input("Annual Income (USD)", 0.0, 1e7, 50000.0, step=1000.0)
    inhand = st.number_input("Monthly Inhand Salary (USD)", 0.0, 1e6, 4000.0, step=100.0)
    balance = st.number_input("Monthly Balance (USD)", 0.0, 1e5, 300.0, step=50.0)
    st.markdown('<p class="section-label">Informasi Kredit</p>', unsafe_allow_html=True)
    n_bank = st.slider("Num Bank Accounts", 0, 20, 4)
    n_card = st.slider("Num Credit Cards", 0, 20, 5)
    n_loan = st.slider("Num of Loans", 0, 20, 3)
    loan_type = st.multiselect("Type of Loan (boleh pilih lebih dari satu)",
        ["Auto Loan","Credit-Builder Loan","Debt Consolidation Loan","Home Equity Loan",
         "Mortgage Loan","Payday Loan","Personal Loan","Student Loan","Not Specified"],
        default=["Personal Loan"])
    util = st.slider("Credit Utilization Ratio (%)", 0.0, 100.0, 32.0)
    hist = st.text_input("Credit History Age", value="15 Years and 4 Months",
        help="Format wajib: 'X Years and Y Months'. Contoh: '15 Years and 4 Months'.")
    st.caption("Format wajib: X Years and Y Months (mis. 15 Years and 4 Months)")
with c2:
    st.markdown('<p class="section-label">Perilaku Pembayaran</p>', unsafe_allow_html=True)
    interest = st.slider("Interest Rate (%)", 0, 50, 12)
    delay = st.slider("Delay from Due Date (days)", -10, 100, 10)
    n_delayed = st.slider("Num of Delayed Payments", 0, 100, 5)
    chg_limit = st.number_input("Changed Credit Limit", -50.0, 50.0, 5.0, step=0.5)
    inquiries = st.slider("Num Credit Inquiries", 0, 50, 4)
    credit_mix = st.selectbox("Credit Mix", ["Good","Standard","Bad"])
    min_amt = st.selectbox("Payment of Min Amount", ["Yes","No","NM"])
    debt = st.number_input("Outstanding Debt (USD)", 0.0, 1e5, 800.0, step=50.0)
    emi = st.number_input("Total EMI per Month (USD)", 0.0, 1e5, 100.0, step=50.0)
    invested = st.number_input("Amount Invested Monthly (USD)", 0.0, 1e5, 200.0, step=50.0)
    behaviour = st.selectbox("Payment Behaviour",
        ["High_spent_Large_value_payments","High_spent_Medium_value_payments",
         "High_spent_Small_value_payments","Low_spent_Large_value_payments",
         "Low_spent_Medium_value_payments","Low_spent_Small_value_payments"])

st.divider()
if st.button("Predict", type="primary", use_container_width=True):
    row = {"Age":age,"Occupation":occupation,"Annual_Income":annual_income,
           "Monthly_Inhand_Salary":inhand,"Num_Bank_Accounts":n_bank,"Num_Credit_Card":n_card,
           "Interest_Rate":interest,"Num_of_Loan":n_loan,
           "Type_of_Loan":", ".join(loan_type) if loan_type else "",
           "Delay_from_due_date":delay,"Num_of_Delayed_Payment":n_delayed,
           "Changed_Credit_Limit":chg_limit,"Num_Credit_Inquiries":inquiries,
           "Credit_Mix":credit_mix,"Outstanding_Debt":debt,"Credit_Utilization_Ratio":util,
           "Credit_History_Age":hist,"Payment_of_Min_Amount":min_amt,"Total_EMI_per_month":emi,
           "Amount_invested_monthly":invested,"Payment_Behaviour":behaviour,"Monthly_Balance":balance}
    out = predict(row)
    color = {"Good":"green","Standard":"orange","Poor":"red"}[out["prediction"]]
    st.markdown(f"## Prediksi: :{color}[{out['prediction']}]")
    st.write("**Probabilitas tiap kelas:**")
    for label, prob in out["probabilities"].items():
        st.write(f"- {label}: {prob:.2%}")
