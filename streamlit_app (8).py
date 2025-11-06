import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

try:
    import google.generativeai as genai
except ImportError:
    genai = None

st.title("Supply Chain ML + Gemini Blockchain Chatbot")

def try_load_model(path):
    try:
        if os.path.exists(path):
            return joblib.load(path)
    except Exception as e:
        st.warning(f"Could not load '{path}': {e}")
        return None
    return None

# Model Loaders (do not crash)
rf_reg = try_load_model("rf_regressor_pipeline.pkl")
rf_cls = try_load_model("rf_classifier_pipeline.pkl")
m2_forecast = try_load_model("module2_forecast.pkl")
cat_churn = try_load_model("catboost_customer_churn.pkl")
cat_supplier = try_load_model("catboost_supplier_reliability.pkl")

tab1, tab2, tab3, tab4 = st.tabs([
    "Delivery", "Forecast", "Churn & Supplier", "Blockchain Chatbot"
])

with tab1:
    st.header("Module 1: Delivery")
    file = st.file_uploader("Upload CSV/XLSX", type=["csv","xlsx"], key="mod1")
    if file:
        df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
        st.info(f"Data: {df.shape}")
        if st.button("Run Risk Classifier"):
            if rf_cls:
                try:
                    pred = rf_cls.predict(df)
                    st.success(f"High risk: {int(pred.sum())}/{len(pred)}")
                except Exception as e:
                    st.error(f"Classifier error: {e}")
            else:
                st.error("Classifier model not loaded.")
        if st.button("Run Delay Regressor"):
            if rf_reg:
                try:
                    pred = rf_reg.predict(df)
                    st.success(f"Avg delay: {np.mean(pred):.2f}, Max: {np.max(pred):.2f}")
                except Exception as e:
                    st.error(f"Regressor error: {e}")
            else:
                st.error("Regressor model not loaded.")

with tab2:
    st.header("Module 2: Forecast")
    file = st.file_uploader("Upload Forecast Data", type=["csv","xlsx"], key="mod2")
    if file:
        df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
        if st.button("Forecast"):
            if m2_forecast:
                try:
                    pred = m2_forecast.predict(df)
                    st.success(f"Forecast avg: {np.mean(pred):.2f}")
                except Exception as e:
                    st.error(f"Forecast error: {e}")
            else:
                st.error("Forecast model not loaded.")

with tab3:
    st.header("Module 3: Churn & Supplier")
    file1 = st.file_uploader("Customer Data", type=["csv","xlsx"], key="churn")
    if file1 and st.button("Predict Churn"):
        if cat_churn:
            try:
                df1 = pd.read_csv(file1) if file1.name.endswith('.csv') else pd.read_excel(file1)
                pred = cat_churn.predict(df1)
                st.success(f"Churned: {int(pred.sum())}/{len(pred)}")
            except Exception as e:
                st.error(f"Churn model error: {e}")
        else:
            st.error("Churn model not loaded.")
    file2 = st.file_uploader("Supplier Data", type=["csv","xlsx"], key="supplier")
    if file2 and st.button("Score Suppliers"):
        if cat_supplier:
            try:
                df2 = pd.read_csv(file2) if file2.name.endswith('.csv') else pd.read_excel(file2)
                pred = cat_supplier.predict(df2)
                st.success(f"Supplier avg score: {np.mean(pred):.2f}")
            except Exception as e:
                st.error(f"Supplier model error: {e}")
        else:
            st.error("Supplier model not loaded.")

with tab4:
    st.header("Gemini Blockchain Chatbot")
    if genai is None:
        st.warning("Gemini/Google GenerativeAI module not installed!")
    else:
        GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
        gemini_model = None
        if GEMINI_API_KEY:
            try:
                genai.configure(api_key=GEMINI_API_KEY)
                gemini_model = genai.GenerativeModel("gemini-1.5-flash")
            except Exception as e:
                st.warning(f"Could not initialize Gemini: {e}")
        q = st.text_area("Blockchain Q:", "")
        if st.button("Get Blockchain Answer"):
            def is_blockchain(q):
                bc_list = [
                    "blockchain","crypto","bitcoin","ethereum","smart contract","web3","defi",
                    "nft","token","ledger","solidity","wallet","gas fee","mining","polygon","staking"
                ]
                return any(kw in q.lower() for kw in bc_list)
            if not is_blockchain(q):
                st.info("Sorry, I only answer blockchain-related questions.")
            elif gemini_model:
                prompt = (
                    "You are a blockchain expert. Only answer real blockchain, Web3, crypto, smart contract, DeFi, or NFT questions."
                    "\nUser: " + q
                )
                try:
                    answer = gemini_model.generate_content(prompt)
                    st.write(answer.text if hasattr(answer, 'text') else str(answer))
                except Exception as e:
                    st.error(f"Gemini error: {e}")
            else:
                st.warning("Gemini API not configured.")

st.info("If you see 'Model not loaded' or a Python error above, ensure:\n"
        "- All .pkl files match your current Python & scikit-learn versions\n"
        "- Any custom pipeline steps are defined/imported here before loading the models.")
