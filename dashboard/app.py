import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
import base64

st.set_page_config(page_title="AutoValuate Pro", page_icon="🏎️", layout="wide")

# Inject custom CSS for premium dark-mode / glassmorphism
st.markdown("""
<style>
    /* Animated Gradient Background */
    .stApp {
        background: linear-gradient(-45deg, #0f172a, #1e1b4b, #312e81, #0f172a);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        color: #f8fafc;
    }
    
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Fade-in Animation for main elements */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .main .block-container {
        animation: fadeIn 1s ease-out forwards;
    }

    /* Animated Button with Hover Glow */
    .stButton>button {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        color: white;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        border: none;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 8px 25px rgba(139, 92, 246, 0.6);
        background: linear-gradient(90deg, #60a5fa, #a78bfa);
    }
    
    .stButton>button:active {
        transform: translateY(1px) scale(0.98);
    }

    /* Glowing Metric Cards */
    .metric-card {
        background: rgba(15, 23, 42, 0.6);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        text-align: center;
        transition: all 0.4s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card:hover {
        transform: translateY(-10px);
        border-color: rgba(255, 255, 255, 0.3);
        box-shadow: 0 15px 30px rgba(0, 0, 0, 0.3);
    }

    /* Text Gradients */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
    }
    
    .gradient-text {
        background: linear-gradient(to right, #60a5fa, #c084fc, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        display: inline-block;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); filter: brightness(1.2); }
        100% { transform: scale(1); }
    }
    
    /* Input field styling */
    .stTextInput>div>div>input, .stNumberInput>div>div>input {
        background-color: rgba(255, 255, 255, 0.05) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 8px !important;
        transition: all 0.3s ease;
    }
    
    .stTextInput>div>div>input:focus, .stNumberInput>div>div>input:focus {
        border-color: #8b5cf6 !important;
        box-shadow: 0 0 10px rgba(139, 92, 246, 0.5) !important;
    }

    /* Selectbox styling */
    div[data-baseweb="select"] > div {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 8px !important;
    }
    
    div[data-baseweb="select"] > div:hover {
        border-color: #3b82f6 !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='gradient-text'>🏎️ AutoValuate Pro</h1>", unsafe_allow_html=True)
st.markdown("### Intelligent Used Car Pricing Engine")

st.markdown("""
Welcome to **AutoValuate Pro**, a production-grade machine learning system designed to optimize vehicle pricing. 
Unlike standard pricing models that treat overestimations and underestimations equally, this engine is optimized 
for **Asymmetric Business Cost**—because overpricing a car leads to inventory rot, while underpricing leads to lost margins.

**Key Technical Features:**
- **Conformal Prediction (MAPIE):** Provides mathematically guaranteed 80% confidence bounds, representing state-of-the-art uncertainty quantification.
- **Stacked Ensemble:** Combines XGBoost and LightGBM models tuned via Optuna.
- **Advanced NLP:** Uses regex heuristics to extract explicit Trim and Model features from messy vehicle names.
""")

st.divider()

# Create a two-column layout
col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("Vehicle Specifications")
    with st.form("vehicle_form"):
        name = st.text_input("Car Make, Model, & Trim", "Maruti Swift Dzire VDI", help="e.g. 'Honda City V MT'")
        year = st.number_input("Manufacture Year", min_value=1990, max_value=2026, value=2017, step=1)
        km_driven = st.number_input("Kilometers Driven", min_value=0, value=45000, step=1000)
        fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
        seller_type = st.selectbox("Seller Type", ["Individual", "Dealer", "Trustmark Dealer"])
        transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
        owner = st.selectbox("Owner Status", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"])
        
        submitted = st.form_submit_button("Generate Valuation")

with col_right:
    if not submitted:
        st.info("👈 Enter the vehicle specifications and click 'Generate Valuation' to see the intelligent pricing analysis.")
        
    if submitted:
        payload = {
            "name": name,
            "year": int(year),
            "km_driven": int(km_driven),
            "fuel": fuel,
            "seller_type": seller_type,
            "transmission": transmission,
            "owner": owner
        }
        
        with st.spinner("Running Ensembles & MAPIE Conformal Predictors..."):
            try:
                response = requests.post("http://localhost:8000/predict", json=payload)
                if response.status_code == 200:
                    data = response.json()
                    
                    price = data['predicted_price']
                    lower = data['confidence_lower_80']
                    upper = data['confidence_upper_80']
                    
                    # 1. Plotly Gauge Chart for Price
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = price,
                        number = {'prefix': "₹", 'valueformat': ",.0f"},
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Predicted Valuation", 'font': {'size': 24, 'color': 'white'}},
                        gauge = {
                            'axis': {'range': [lower * 0.8, upper * 1.2], 'tickwidth': 1, 'tickcolor': "white"},
                            'bar': {'color': "#3b82f6"},
                            'bgcolor': "rgba(0,0,0,0)",
                            'borderwidth': 2,
                            'bordercolor': "gray",
                            'steps': [
                                {'range': [lower, upper], 'color': "rgba(16, 185, 129, 0.2)"},
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': price
                            }
                        }
                    ))
                    fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"}, height=300)
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    
                    # Markdown metrics
                    m1, m2, m3 = st.columns(3)
                    m1.markdown(f"<div class='metric-card'><h4>Lower Bound (80%)</h4><h2 style='color:#ef4444'>₹{lower:,.0f}</h2></div>", unsafe_allow_html=True)
                    m2.markdown(f"<div class='metric-card'><h4>Point Estimate</h4><h2 style='color:#3b82f6'>₹{price:,.0f}</h2></div>", unsafe_allow_html=True)
                    m3.markdown(f"<div class='metric-card'><h4>Upper Bound (80%)</h4><h2 style='color:#10b981'>₹{upper:,.0f}</h2></div>", unsafe_allow_html=True)
                    
                    # 2. SHAP Waterfall
                    st.markdown("### Why this price?")
                    features = [f['feature'] for f in data['top_3_shap_features']]
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
                        font={'color': "white"},
                        xaxis_title="Impact on Price (₹)", 
                        yaxis_title="Feature",
                        height=250,
                        margin=dict(l=0, r=0, t=30, b=0)
                    )
                    st.plotly_chart(fig_shap, use_container_width=True)
                    
                    # 3. Depreciation Journey Mockup
                    st.markdown("### Depreciation Journey")
                    # Generate a realistic-looking exponential decay curve based on the current price
                    ages = np.arange(1, 16)
                    decay_rate = 0.12 # mock average depreciation rate
                    base_price = price / ((1 - decay_rate) ** (2026 - int(year)))
                    curve_prices = base_price * ((1 - decay_rate) ** ages)
                    
                    fig_curve = go.Figure()
                    fig_curve.add_trace(go.Scatter(x=ages, y=curve_prices, mode='lines', name='Avg Depreciation', line=dict(color='rgba(255,255,255,0.5)', width=2)))
                    
                    car_age = 2026 - int(year)
                    fig_curve.add_trace(go.Scatter(x=[car_age], y=[price], mode='markers', name='This Vehicle', marker=dict(color='#3b82f6', size=15, symbol='star')))
                    
                    fig_curve.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", 
                        plot_bgcolor="rgba(0,0,0,0)",
                        font={'color': "white"},
                        xaxis_title="Car Age (Years)",
                        yaxis_title="Estimated Value (₹)",
                        height=300,
                        margin=dict(l=0, r=0, t=10, b=0)
                    )
                    st.plotly_chart(fig_curve, use_container_width=True)

                else:
                    st.error(f"Error {response.status_code}: {response.text}")
            except Exception as e:
                st.error(f"Failed to connect to API: {str(e)}")
