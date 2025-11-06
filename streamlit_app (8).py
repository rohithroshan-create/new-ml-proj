import streamlit as st
import pandas as pd
import numpy as np
import pickle  # Switched from joblib for consistency with your notebook
import os
from prophet import Prophet
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai
import warnings
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings('ignore')

# ========== PAGE CONFIG ==========\
st.set_page_config(
    page_title="🏭 Supply Chain AI Pro",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ========== CUSTOM CSS ==========\
st.markdown("""
    <style>
    * { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    .main { background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%); color: #1a1a1a; }
    .stApp { background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%); }
    .header-box {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 50%, #c44569 100%);
        color: white; padding: 20px; border-radius: 15px;
        box-shadow: 0 8px 32px rgba(255, 107, 107, 0.3);
        margin: 20px 0; border: 1px solid rgba(255,255,255,0.2);
        display: flex; align-items: center; justify-content: space-between;
    }
    .header-box h1 { color: white; margin: 0; font-weight: 600; font-size: 2.5em; }
    .header-box img { height: 70px; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background: #f0f2f6;
        border-radius: 8px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 8px;
        color: #4f4f4f;
        font-weight: 500;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        color: #ff6b6b;
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .stMetric {
        background: #ffffff;
        border: 1px solid #e6e6e6;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.03);
    }
    .stMetric > label { font-weight: 500; color: #4f4f4f; }
    .stMetric > div { font-weight: 700; color: #ff6b6b; }
    .success-box { background: #e8f5e9; color: #2e7d32; padding: 12px; border-radius: 8px; margin: 10px 0; border-left: 5px solid #4caf50; font-weight: 500; }
    .error-box { background: #ffebee; color: #c62828; padding: 12px; border-radius: 8px; margin: 10px 0; border-left: 5px solid #ff6b6b; font-weight: 500; }
    .chat-message-user { background: #e3f2fd; color: #0d47a1; padding: 12px; border-radius: 10px; margin-bottom: 5px; max-width: 80%; float: right; clear: both; }
    .chat-message-ai { background: #f1f8e9; color: #33691e; padding: 12px; border-radius: 10px; margin-bottom: 5px; max-width: 80%; float: left; clear: both; }
    .chat-message-reject { background: #ffebee; color: #c62828; padding: 12px; border-radius: 10px; margin-bottom: 5px; max-width: 80%; float: left; clear: both; }
    .chat-container { height: 500px; overflow-y: auto; padding: 10px; background: #ffffff; border: 1px solid #e6e6e6; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

# ========== HEADER ==========\
col1, col2 = st.columns([0.85, 0.15])
with col1:
    st.markdown("""
        <div class="header-box">
            <div>
                <h1 style="font-size: 2.2em;">🏭 Supply Chain AI Pro</h1>
                <span style="font-size: 1.1em;">Advanced Analytics for Delivery, Demand, and Churn</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
with col2:
    if os.path.exists("image_b2c860.png"):
        st.image("image_b2c860.png", width=120)
    else:
        st.markdown(" ") # Placeholder if image is missing

# ========== LOAD MODELS ==========\
@st.cache_resource
def load_model(path):
    try:
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file not found at {path}. Please make sure it's in the correct directory.")
        return None
    except Exception as e:
        st.error(f"Error loading model {path}: {e}")
        return None

# --- Module 1 Models (The new RF ones) ---
clf_pipeline = load_model("rf_classifier_pipeline.pkl")
reg_pipeline = load_model("rf_regressor_pipeline.pkl")

# --- Module 2 & 3 Models (From your original app) ---
prophet_model = None # Prophet models are trained on the fly
churn_model = load_model("churn_model.pkl")
supplier_model = load_model("supplier_model.pkl")

# ========== TABS ==========\
tab1, tab2, tab3, tab4 = st.tabs([
    "📦 Delivery Risk & Delay",
    "📈 Demand Forecasting",
    "👥 Churn & Supplier",
    "🤖 Blockchain Chatbot"
])

# ===== TAB 1: DELIVERY RISK & DELAY (NEW) =====
with tab1:
    st.header("📦 Delivery Risk & Delay Prediction")
    st.markdown("Upload your shipping data to predict delivery risks and potential delays using our Random Forest models.")

    uploaded_file_1 = st.file_uploader("Upload CSV for Delivery Prediction", type="csv", key="tab1_uploader")

    if uploaded_file_1 and clf_pipeline and reg_pipeline:
        try:
            df = pd.read_csv(uploaded_file_1)
            st.markdown("---")
            st.subheader("🔮 Predictions")

            # --- Check required columns ---
            # These are the columns I identified in our previous chat
            cols_clf = [
                'Days for shipment (scheduled)', 'Order Item Discount', 'Order Item Discount Rate', 
                'Order Item Product Price', 'Order Item Profit Ratio', 'Order Item Quantity', 
                'Product Price', 'Order_Month', 'Order_DayOfWeek', 'Order_Hour', 'Type', 
                'Delivery Status', 'Category Name', 'Customer Segment', 'Department Name', 
                'Market', 'Order Region', 'Order Status', 'Product Name', 'Shipping Mode'
            ]
            cols_reg = [
                'Days for shipment (scheduled)', 'Order Item Discount', 'Order Item Discount Rate', 
                'Order Item Product Price', 'Order Item Profit Ratio', 'Order Item Quantity', 
                'Product Price', 'Order_Month', 'Order_DayOfWeek', 'Order_Hour', 'Type', 
                'Category Name', 'Customer Segment', 'Department Name', 'Market', 
                'Order Region', 'Product Name', 'Shipping Mode'
            ]

            # Check for missing columns
            missing_clf_cols = [col for col in cols_clf if col not in df.columns]
            missing_reg_cols = [col for col in cols_reg if col not in df.columns]

            if missing_clf_cols or missing_reg_cols:
                if missing_clf_cols:
                    st.error(f"File is missing required columns for Risk Prediction: {', '.join(missing_clf_cols)}")
                if missing_reg_cols:
                    st.error(f"File is missing required columns for Delay Prediction: {', '.join(missing_reg_cols)}")
            else:
                # --- Run Predictions ---
                risk_predictions = clf_pipeline.predict(df[cols_clf])
                risk_probabilities = clf_pipeline.predict_proba(df[cols_clf])[:, 1]
                delay_predictions = reg_pipeline.predict(df[cols_reg])

                df_results = df.copy()
                df_results['Predicted_Late_Risk'] = risk_predictions
                df_results['Predicted_Late_Probability'] = risk_probabilities
                df_results['Predicted_Shipping_Days'] = delay_predictions.round(1)

                # --- Display KPIs ---
                high_risk_count = df_results['Predicted_Late_Risk'].sum()
                avg_delay = df_results['Predicted_Shipping_Days'].mean()

                col1, col2, col3 = st.columns(3)
                col1.metric("Total Shipments", f"{len(df_results)}")
                col2.metric("High Risk Shipments", f"{high_risk_count}", f"{100 * high_risk_count / len(df_results):.1f}%")
                col3.metric("Avg. Predicted Delay", f"{avg_delay:.1f} days")
                st.markdown("---")

                # --- Display Graphs ---
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Delivery Risk Distribution")
                    risk_counts = df_results['Predicted_Late_Risk'].value_counts().reset_index()
                    risk_counts['index'] = risk_counts['index'].map({0: 'On Time', 1: 'Late Risk'})
                    fig_pie = px.pie(risk_counts, names='index', values='count', 
                                     title="Risk Prediction Overview",
                                     color='index', color_discrete_map={'On Time':'green', 'Late Risk':'red'})
                    st.plotly_chart(fig_pie, use_container_width=True)

                with col2:
                    st.subheader("Predicted Shipping Delays")
                    fig_hist = px.histogram(df_results, x='Predicted_Shipping_Days', 
                                            title="Histogram of Predicted Delays",
                                            color_discrete_sequence=['#ff6b6b'])
                    st.plotly_chart(fig_hist, use_container_width=True)

                # --- Display Data ---
                st.subheader("Prediction Results")
                with st.expander("View full prediction data"):
                    st.dataframe(df_results[[
                        'Order Id', 'Predicted_Late_Risk', 'Predicted_Late_Probability', 
                        'Predicted_Shipping_Days', 'Days for shipment (scheduled)'
                    ] + cols_clf].head(100), use_container_width=True)
        
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

# ===== TAB 2: DEMAND FORECASTING (PORTED) =====
with tab2:
    st.header("📈 Demand Forecasting")
    st.markdown("Upload your historical sales data to generate a future demand forecast using Prophet.")

    uploaded_file_2 = st.file_uploader("Upload CSV for Demand Forecasting", type="csv", key="tab2_uploader")

    if uploaded_file_2:
        try:
            df = pd.read_csv(uploaded_file_2)
            st.info("File uploaded. Please select the date and metric columns.")

            col1, col2 = st.columns(2)
            date_col = col1.selectbox("Select Date Column (ds)", df.columns)
            metric_col = col2.selectbox("Select Metric Column (y)", [col for col in df.columns if np.issubdtype(df[col].dtype, np.number)])
            
            periods = st.slider("Select forecast period (days)", 30, 365, 90)

            if st.button("🚀 Generate Forecast", key="prophet_button"):
                with st.spinner("🧠 Training model and forecasting..."):
                    df[date_col] = pd.to_datetime(df[date_col])
                    df_prophet = df[[date_col, metric_col]].rename(columns={date_col: 'ds', metric_col: 'y'})
                    
                    m = Prophet()
                    m.fit(df_prophet)
                    future = m.make_future_dataframe(periods=periods)
                    forecast = m.predict(future)
                    
                    st.session_state['forecast'] = forecast
                    st.session_state['prophet_model'] = m
                    
                    st.success("Forecast generated successfully!")

            if 'forecast' in st.session_state:
                st.markdown("---")
                st.subheader("Forecast Results")
                
                fig1 = st.session_state['prophet_model'].plot(st.session_state['forecast'])
                st.pyplot(fig1)
                
                fig2 = st.session_state['prophet_model'].plot_components(st.session_state['forecast'])
                st.pyplot(fig2)
                
                with st.expander("View forecast data"):
                    st.dataframe(st.session_state['forecast'].tail(periods))

        except Exception as e:
            st.error(f"An error occurred: {e}")

# ===== TAB 3: CHURN & SUPPLIER (PORTED) =====
with tab3:
    st.header("👥 Customer Churn & Supplier Reliability")
    st.markdown("Predict customer churn and assess supplier reliability scores from your data.")
    
    uploaded_file_3 = st.file_uploader("Upload CSV for Churn/Supplier Analysis", type="csv", key="tab3_uploader")

    if uploaded_file_3 and churn_model and supplier_model:
        try:
            df = pd.read_csv(uploaded_file_3)
            st.markdown("---")
            
            # --- Churn Prediction ---
            st.subheader("🧑‍💼 Customer Churn")
            # --- This is a placeholder. You need to know the *exact* columns your model was trained on ---
            # --- I am guessing based on the `streamlit_app (8).py` file's hints ---
            churn_features = [col for col in df.columns if col not in ['CustomerID', 'CustomerName', 'Churn_Status']] # Example
            
            # A real implementation would need to check/preprocess features
            # This is a simplified example
            try:
                # Assuming churn_model is a pipeline that handles preprocessing
                churn_preds = churn_model.predict(df)
                churn_proba = churn_model.predict_proba(df)[:, 1]
                
                churn_count = np.sum(churn_preds)
                total_customers = len(df)
                churn_pct = 100 * churn_count / total_customers

                col1, col2 = st.columns(2)
                col1.metric("Total Customers", f"{total_customers}")
                col2.metric("Predicted Churn", f"{churn_count}", f"{churn_pct:.1f}%")
                
                st.session_state['predictions'] = {'churn': {'count': churn_count, 'total': total_customers, 'churn_pct': churn_pct}}

            except Exception as e:
                st.warning(f"Could not run churn prediction. Model may expect different columns. Error: {e}")

            # --- Supplier Reliability ---
            st.subheader("🚚 Supplier Reliability")
            try:
                # Again, assuming supplier_model is a pipeline
                supplier_scores = supplier_model.predict(df) # This is likely a regressor
                df['Reliability_Score'] = supplier_scores
                avg_score = df['Reliability_Score'].mean()

                col1, col2 = st.columns(2)
                col1.metric("Total Suppliers", f"{df['SupplierID'].nunique()}") # Assuming a 'SupplierID' column
                col2.metric("Avg. Reliability Score", f"{avg_score:.2f} / 10") # Assuming 1-10 scale

                fig_supplier = px.histogram(df, x='Reliability_Score', title="Supplier Score Distribution", color_discrete_sequence=['#ff6b6b'])
                st.plotly_chart(fig_supplier, use_container_width=True)

                st.session_state['predictions'] = {'supplier': {'avg_score': avg_score}}
            
            except Exception as e:
                st.warning(f"Could not run supplier prediction. Model may expect different columns. Error: {e}")
        
        except Exception as e:
            st.error(f"An error occurred: {e}")

# ===== TAB 4: BLOCKCHAIN CHATBOT (NEW) =====
with tab4:
    st.header("🤖 Blockchain & Supply Chain Chatbot")
    st.markdown("Ask me anything about **blockchain** or general **supply chain/delivery tips**.")

    # --- Gemini API Setup ---
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model = genai.GenerativeModel('gemini-1.5-pro-latest')
    except Exception:
        st.error("Could not configure Gemini API. Please add your `GEMINI_API_KEY` to Streamlit secrets.", icon="🚨")
        model = None

    # --- System Prompt (The Guardrail) ---
    SYSTEM_PROMPT = """
    You are a specialized AI assistant for a supply chain project.
    
    ALLOWED TOPICS:
    1.  Blockchain Technology: Answer any questions about what blockchain is, how it works, its use cases in supply chain, ledgers, hashes, etc.
    2.  General Supply Chain & Delivery Tips: Answer general knowledge questions about logistics, inventory management, last-mile delivery, etc.
    
    STRICT REJECTION RULES:
    1.  DO NOT answer questions about any other topic (e.g., marketing, finance, coding, history, the weather, celebrities, etc.).
    2.  DO NOT have access to any data, uploads, or predictions from the other tabs. If the user asks "what was my churn rate?" or "how many shipments are high risk?", you MUST state that you do not have access to that information.
    3.  If the user asks a question outside your allowed topics, you MUST politely decline and remind them of your purpose.
    
    EXAMPLE REJECTION: "I'm sorry, I can only answer questions related to blockchain technology and general supply chain tips. I cannot help with [User's Off-Topic Subject]."
    """

    # --- Chat History Initialization ---
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = [
            {"role": "ai", "content": "Hello! Ask me about blockchain or supply chain tips.", "type": "ai"}
        ]

    # --- Chat Display ---
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state['chat_history']:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-message-user"><b>You:</b><br/>{msg["content"]}</div>', unsafe_allow_html=True)
            elif msg["type"] == "ai":
                st.markdown(f'<div class="chat-message-ai"><b>Bot:</b><br/>{msg["content"]}</div>', unsafe_allow_html=True)
            elif msg["type"] == "reject":
                st.markdown(f'<div class="chat-message-reject"><b>Bot:</b><br/>{msg["content"]}</div>', unsafe_allow_html=True)

    # --- Chat Input ---
    user_input = st.chat_input("Ask your question...")

    if user_input and model:
        st.session_state['chat_history'].append({"role": "user", "content": user_input, "type": "user"})
        with chat_container:
            st.markdown(f'<div class="chat-message-user"><b>You:</b><br/>{user_input}</div>', unsafe_allow_html=True)
        
        with st.spinner("🤖 Thinking..."):
            try:
                # Construct the full prompt
                full_prompt = SYSTEM_PROMPT + "\n\n" + "User question: " + user_input
                
                response = model.generate_content(full_prompt)
                bot_response = response.text
                
                # Simple check for rejection (you can make this smarter)
                if "I'm sorry" in bot_response or "I cannot help" in bot_response or "I can only answer" in bot_response:
                    msg_type = "reject"
                else:
                    msg_type = "ai"

                st.session_state['chat_history'].append({"role": "ai", "content": bot_response, "type": msg_type})
                st.rerun()

            except Exception as e:
                st.error(f"Error communicating with AI: {e}")
