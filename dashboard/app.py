import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sqlite3
import os
from fpdf import FPDF
import datetime
import time

st.set_page_config(page_title="AutoValuate Pro", page_icon="🏎️", layout="wide", initial_sidebar_state="expanded")

# Inject premium custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

    /* Global Font & Animated Gradient Background */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif !important;
    }
    
    .stApp {
        background: linear-gradient(-45deg, #09090b, #18181b, #27272a, #09090b);
        background-size: 400% 400%;
        animation: gradientBG 20s ease infinite;
        color: #f8fafc;
    }
    
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: rgba(9, 9, 11, 0.8) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(255, 255, 255, 0.03);
        padding: 8px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 8px;
        color: #a1a1aa;
        padding: 0 16px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6) !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
    }

    /* Animated Button with Hover Glow */
    .stButton>button {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        color: white;
        border-radius: 30px;
        padding: 0.75rem 2.5rem;
        border: none;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        width: 100%;
        margin-top: 1rem;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 8px 25px rgba(139, 92, 246, 0.6);
        background: linear-gradient(90deg, #60a5fa, #a78bfa);
        color: white;
    }

    /* Glowing Metric Cards */
    .metric-card {
        background: rgba(24, 24, 27, 0.6);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
        text-align: center;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    }
    
    .metric-card:hover {
        transform: translateY(-8px);
        border-color: rgba(139, 92, 246, 0.4);
        box-shadow: 0 20px 40px -10px rgba(139, 92, 246, 0.3);
    }

    /* Text Gradients */
    .gradient-text {
        background: linear-gradient(to right, #60a5fa, #c084fc, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem;
        letter-spacing: -1px;
        margin-bottom: 0;
    }
    
    /* Input field styling */
    .stTextInput>div>div>input, .stNumberInput>div>div>input {
        background-color: rgba(0, 0, 0, 0.2) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.3s ease;
    }
    
    .stTextInput>div>div>input:focus, .stNumberInput>div>div>input:focus {
        border-color: #8b5cf6 !important;
        box-shadow: 0 0 12px rgba(139, 92, 246, 0.3) !important;
        background-color: rgba(0, 0, 0, 0.4) !important;
    }

    /* Selectbox styling */
    div[data-baseweb="select"] > div {
        background-color: rgba(0, 0, 0, 0.2) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
        transition: all 0.3s ease;
    }
    
    div[data-baseweb="select"] > div:hover {
        border-color: #8b5cf6 !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper function for PDF Generation
def generate_pdf(req_data, price, lower, upper):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16, style='B')
    pdf.cell(200, 15, txt="AUTOVALUATE PRO", ln=1, align='C')
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Official Vehicle Valuation Certificate", ln=1, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", size=11)
    pdf.cell(200, 10, txt=f"Date of Valuation: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=1, align='L')
    pdf.ln(5)
    
    pdf.set_font("Arial", size=12, style='B')
    pdf.cell(200, 10, txt="Vehicle Specifications:", ln=1, align='L')
    pdf.set_font("Arial", size=11)
    for key, value in req_data.items():
        pdf.cell(200, 8, txt=f"  - {key.replace('_', ' ').title()}: {value}", ln=1, align='L')
        
    pdf.ln(10)
    pdf.set_font("Arial", size=12, style='B')
    pdf.cell(200, 10, txt="Valuation Result (80% Confidence):", ln=1, align='L')
    pdf.set_font("Arial", size=11)
    pdf.cell(200, 10, txt=f"Estimated Market Value: INR {price:,.2f}", ln=1, align='L')
    pdf.cell(200, 10, txt=f"Fair Market Range: INR {lower:,.2f} to INR {upper:,.2f}", ln=1, align='L')
    
    return bytes(pdf.output())

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3202/3202926.png", width=60)
    st.markdown("### 🏎️ AutoValuate Pro")
    st.markdown("<span style='color:#a1a1aa; font-size: 0.9rem;'>Enterprise Pricing Engine</span>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("#### 🔐 API Authentication")
    api_key = st.text_input("Enter your API Key", value="AUTOVAL-DEMO-KEY", type="password")
    
    st.markdown("---")
    st.markdown("#### 🟢 System Status")
    try:
        health_resp = requests.get("http://localhost:8000/health")
        if health_resp.status_code == 200:
            st.success("Core ML API: Online")
        else:
            st.error("Core ML API: Error")
    except:
        st.error("Core ML API: Offline")

# Main Header
st.markdown("<h1 class='gradient-text'>AutoValuate Pro</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #a1a1aa; font-size: 1.2rem; margin-bottom: 2rem;'>Intelligent, Uncertainty-Aware Vehicle Pricing System</p>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["✨ Valuation Engine", "📈 MLOps & Drift Monitor"])

with tab1:
    col_form, col_result = st.columns([1.1, 1.9], gap="large")

    with col_form:
        st.markdown("### 📋 Enter Vehicle Specs")
        with st.form("vehicle_form"):
            name = st.text_input("Car Make, Model, & Trim", "Hyundai Creta SX Opt", help="Include make, model, and trim for highest accuracy")
            
            c1, c2 = st.columns(2)
            with c1:
                year = st.number_input("Manufacture Year", min_value=1990, max_value=2026, value=2019, step=1)
            with c2:
                km_driven = st.number_input("Kilometers Driven", min_value=0, value=35000, step=1000)
                
            c3, c4 = st.columns(2)
            with c3:
                fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "Electric", "LPG"])
            with c4:
                transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
                
            c5, c6 = st.columns(2)
            with c5:
                owner = st.selectbox("Owner Status", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above"])
            with c6:
                seller_type = st.selectbox("Seller Type", ["Individual", "Dealer", "Trustmark Dealer"])
            
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("Generate Valuation 🚀")

    with col_result:
        if not submitted:
            st.info("👈 Fill out the vehicle specifications and hit Generate Valuation to view the AI-driven pricing analysis.")
            
        if submitted:
            payload = {
                "name": name,
                "year": int(year),
                "km_driven": int(km_driven),
                "fuel": fuel,
                "seller_type": seller_type,
                "transmission": transmission,
                "owner": "Fourth & Above Owner" if owner == "Fourth & Above" else owner
            }
            
            with st.spinner("Engaging XGBoost & LightGBM Ensembles..."):
                time.sleep(0.5) # Slight delay for dramatic UX effect
                try:
                    headers = {"X-API-Key": api_key}
                    response = requests.post("http://localhost:8000/predict", json=payload, headers=headers)
                    if response.status_code == 200:
                        data = response.json()
                        st.toast("Valuation Generated Successfully!", icon="✅")
                        
                        price = data['predicted_price']
                        lower = data['confidence_lower_80']
                        upper = data['confidence_upper_80']
                        
                        # 1. Price Metrics
                        m1, m2, m3 = st.columns(3)
                        m1.markdown(f"<div class='metric-card'><h5>80% Lower Bound</h5><h2 style='color:#ef4444; margin-top: 10px;'>₹{lower:,.0f}</h2></div>", unsafe_allow_html=True)
                        m2.markdown(f"<div class='metric-card' style='border-color: #8b5cf6;'><h5>Point Estimate</h5><h2 style='color:#a78bfa; margin-top: 10px;'>₹{price:,.0f}</h2></div>", unsafe_allow_html=True)
                        m3.markdown(f"<div class='metric-card'><h5>80% Upper Bound</h5><h2 style='color:#10b981; margin-top: 10px;'>₹{upper:,.0f}</h2></div>", unsafe_allow_html=True)
                        
                        # PDF Download Button under metrics
                        pdf_bytes = generate_pdf(payload, price, lower, upper)
                        st.download_button(
                            label="⬇️ Download Official Certificate",
                            data=pdf_bytes,
                            file_name=f"Valuation_{name.replace(' ', '_')}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                        
                        st.markdown("<hr style='border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
                        
                        viz_col1, viz_col2 = st.columns(2)
                        
                        # 2. Gauge Chart
                        with viz_col1:
                            st.markdown("#### 🎯 Market Position")
                            fig_gauge = go.Figure(go.Indicator(
                                mode = "gauge+number",
                                value = price,
                                number = {'prefix': "₹", 'valueformat': ",.0f", 'font': {'color': 'white'}},
                                domain = {'x': [0, 1], 'y': [0, 1]},
                                gauge = {
                                    'axis': {'range': [lower * 0.7, upper * 1.3], 'tickwidth': 1, 'tickcolor': "white"},
                                    'bar': {'color': "#8b5cf6"},
                                    'bgcolor': "rgba(255,255,255,0.05)",
                                    'borderwidth': 0,
                                    'steps': [
                                        {'range': [lower, upper], 'color': "rgba(16, 185, 129, 0.15)"},
                                    ],
                                }
                            ))
                            fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"}, height=250, margin=dict(l=20, r=20, t=20, b=20))
                            st.plotly_chart(fig_gauge, use_container_width=True)

                        # 3. SHAP Waterfall
                        with viz_col2:
                            st.markdown("#### 🔍 Value Drivers (SHAP)")
                            if data['top_3_shap_features']:
                                features = [f['feature'].replace('_', ' ').title() for f in data['top_3_shap_features']]
                                contributions = [f['contribution'] for f in data['top_3_shap_features']]
                                
                                fig_shap = go.Figure(go.Bar(
                                    x=contributions,
                                    y=features,
                                    orientation='h',
                                    marker_color=['#10b981' if c > 0 else '#ef4444' for c in contributions],
                                    text=[f"+₹{c:,.0f}" if c > 0 else f"₹{c:,.0f}" for c in contributions],
                                    textposition='auto'
                                ))
                                fig_shap.update_layout(
                                    paper_bgcolor="rgba(0,0,0,0)", 
                                    plot_bgcolor="rgba(0,0,0,0)",
                                    font={'color': "#a1a1aa"},
                                    xaxis=dict(showgrid=False, zeroline=True, zerolinecolor="rgba(255,255,255,0.2)"),
                                    yaxis=dict(showgrid=False),
                                    height=250,
                                    margin=dict(l=0, r=0, t=10, b=0)
                                )
                                st.plotly_chart(fig_shap, use_container_width=True)
                        
                        # 4. AI Negotiation Intelligence
                        if 'negotiation_insights' in data and data['negotiation_insights']:
                            st.markdown("#### 🧠 AI Negotiation Intelligence")
                            for insight in data['negotiation_insights']:
                                if insight.startswith("Buyer Tactic"):
                                    st.error(insight, icon="📉")
                                elif insight.startswith("Seller Leverage"):
                                    st.success(insight, icon="📈")
                                else:
                                    st.info(insight, icon="💡")
                    elif response.status_code == 401:
                        st.error("🚫 Unauthorized: Invalid API Key. Please check the sidebar.")
                    else:
                        st.error(f"Error {response.status_code}: {response.text}")
                except Exception as e:
                    st.error(f"Failed to connect to API: {str(e)}")

with tab2:
    st.markdown("### 📡 Live Production Telemetry")
    st.markdown("<p style='color: #a1a1aa;'>Monitoring data drift and production metrics from SQLite event store.</p>", unsafe_allow_html=True)
    
    db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'prediction_logs.db')
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            df_logs = pd.read_sql("SELECT * FROM prediction_logs ORDER BY timestamp DESC", conn)
            conn.close()
            
            if len(df_logs) > 0:
                df_logs['timestamp'] = pd.to_datetime(df_logs['timestamp'])
                
                # Top metrics
                mc1, mc2, mc3 = st.columns(3)
                mc1.markdown(f"<div class='metric-card'><h5>Total Queries</h5><h2 style='color:#60a5fa;'>{len(df_logs)}</h2></div>", unsafe_allow_html=True)
                mc2.markdown(f"<div class='metric-card'><h5>Avg Valuation</h5><h2 style='color:#34d399;'>₹{df_logs['predicted_price'].mean():,.0f}</h2></div>", unsafe_allow_html=True)
                mc3.markdown(f"<div class='metric-card'><h5>Most Queried Fuel</h5><h2 style='color:#f472b6;'>{df_logs['fuel'].mode()[0]}</h2></div>", unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("#### 📈 Predicted Price Trend")
                    fig_trend = px.line(df_logs, x='timestamp', y='predicted_price', markers=True)
                    fig_trend.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': "#a1a1aa"}, xaxis_title="", yaxis_title="Price (₹)")
                    fig_trend.update_traces(line_color="#8b5cf6")
                    st.plotly_chart(fig_trend, use_container_width=True)
                
                with c2:
                    st.markdown("#### ⛽ Query Distribution by Fuel")
                    fig_pie = px.pie(df_logs, names='fuel', hole=0.6)
                    fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': "#a1a1aa"}, showlegend=False)
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label', marker=dict(colors=['#3b82f6', '#8b5cf6', '#f472b6', '#10b981']))
                    st.plotly_chart(fig_pie, use_container_width=True)
                    
                st.markdown("#### 📓 Recent Evaluation Logs")
                st.dataframe(
                    df_logs[['timestamp', 'name', 'year', 'km_driven', 'predicted_price']].head(10),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("Database exists but no predictions have been logged yet.")
        except Exception as e:
            st.error(f"Error reading database: {e}")
    else:
        st.warning("No prediction logs found. Make a prediction in the Valuation Engine tab first!")
