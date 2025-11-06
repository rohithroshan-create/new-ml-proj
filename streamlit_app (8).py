import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import google.generativeai as genai

# ----- Gemini config -----
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
gemini_model = None
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")

def is_blockchain_question(q):
    bc_keywords = [
        "blockchain","crypto","bitcoin","ethereum","smart contract","web3","defi","nft","token",
        "consensus","mining","ledger","solidity","wallet","gas fee","polygon","cardano","staking",
        "hash","metamask","decentralized","chain","binance","stablecoin"
    ]
    return any(kw in q.lower() for kw in bc_keywords)

def gemini_blockchain_chatbot(question):
    if not gemini_model:
        return "Gemini API Key not configured."
    prompt = (
        "You are an expert blockchain consultant. "
        "Answer ONLY if this is blockchain/crypto/Web3 related. "
        "If not, reply: 'Sorry, I only answer blockchain-related questions.'\n\n"
        f"User: {question}\n\n"
        "Answer:"
    )
    resp = gemini_model.generate_content(prompt)
    return resp.text if hasattr(resp, "text") else str(resp)

# ----- Load models -----
def load_model(path):
    return joblib.load(path) if os.path.exists(path) else None

rf_reg=load_model("rf_regressor_pipeline.pkl")
rf_cls=load_model("rf_classifier_pipeline.pkl")
m2_forecast=load_model("module2_forecast.pkl")
cat_churn=load_model("catboost_customer_churn.pkl")
cat_supplier=load_model("catboost_supplier_reliability.pkl")

# ----- App -----
st.title("Supply Chain AI + Gemini Blockchain Chatbot")

tab1, tab2, tab3, tab4 = st.tabs([
    "Module 1: Delivery",
    "Module 2: Forecast",
    "Module 3: Churn & Supplier",
    "Blockchain Chatbot"
])

with tab1:
    st.header("Delivery Prediction")
    file = st.file_uploader("Upload CSV/XLSX", type=["csv","xlsx"], key="mod1")
    if file:
        df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
        st.info(f"Data: {df.shape[0]} rows, {df.shape[1]} columns")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Risk Classification"):
                if rf_cls:
                    pred = rf_cls.predict(df)
                    st.success(f"High risk count: {int(pred.sum())} / {len(pred)}")
                else:
                    st.error("Classifier model not found.")
        with col2:
            if st.button("Delay Regression"):
                if rf_reg:
                    pred = rf_reg.predict(df)
                    st.success(f"Average delay: {np.mean(pred):.2f}, Max: {np.max(pred):.2f}, Min: {np.min(pred):.2f}")
                else:
                    st.error("Regressor model not found.")

with tab2:
    st.header("Demand Forecasting")
    file = st.file_uploader("Upload Forecast Data", type=["csv","xlsx"], key="mod2")
    if file:
        df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
        st.info(f"Data: {df.shape[0]} rows, {df.shape[1]} columns")
        if st.button("Run Forecast"):
            if m2_forecast:
                pred = m2_forecast.predict(df)
                st.success(f"Forecast avg: {np.mean(pred):.2f}, Max: {np.max(pred):.2f}, Min: {np.min(pred):.2f}")
            else:
                st.error("Forecast model not found.")

with tab3:
    st.header("Churn & Supplier Analysis")
    ch1, ch2 = st.columns(2)
    with ch1:
        file = st.file_uploader("Customer Data", type=["csv","xlsx"], key="mod3-churn")
        if file:
            df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
            if st.button("Predict Churn"):
                if cat_churn:
                    pred = cat_churn.predict(df)
                    st.success(f"Churned: {int(pred.sum())} / {len(pred)}  ({100*pred.sum()/len(pred):.1f}%)")
                else:
                    st.error("Churn model not found.")
    with ch2:
        file = st.file_uploader("Supplier Data", type=["csv","xlsx"], key="mod3-supplier")
        if file:
            df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
            if st.button("Score Suppliers"):
                if cat_supplier:
                    pred = cat_supplier.predict(df)
                    st.success(f"Average score: {np.mean(pred):.2f} / 10")
                else:
                    st.error("Supplier reliability model not found.")

with tab4:
    st.header("Blockchain Knowledge Chatbot (Gemini AI)")
    q = st.text_area("Ask your blockchain question here", key="bc")
    if st.button("Get Answer", key="bc-submit"):
        if is_blockchain_question(q):
            answer = gemini_blockchain_chatbot(q)
        else:
            answer = "Sorry, I only answer blockchain-related questions."
        st.write(answer)
