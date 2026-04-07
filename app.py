import streamlit as st
import requests

# Set page configuration
st.set_page_config(page_title="SentinelScore", page_icon="🛡️", layout="centered")

st.title("🛡️ SentinelScore Risk Engine")
st.markdown("Enter applicant details below to calculate real-time credit default risk.")

st.divider()

# Create two columns for a clean form layout
col1, col2 = st.columns(2)

with col1:
    income = st.number_input("Total Income (USD)", min_value=10000, value=50000, step=5000)
    credit = st.number_input("Requested Credit Amount", min_value=10000, value=200000, step=10000)
    age_years = st.number_input("Age (Years)", min_value=18, max_value=100, value=35)

with col2:
    employed_years = st.number_input("Years Employed", min_value=0, max_value=60, value=5)
    ext_source_2 = st.slider("External Credit Score 2", 0.0, 1.0, 0.5)
    ext_source_3 = st.slider("External Credit Score 3", 0.0, 1.0, 0.5)

st.divider()

# The Submission Button
if st.button("Calculate Risk Score", type="primary", use_container_width=True):
    
    # 1. Format the data mathematically the way the model expects (Days as negative integers)
    payload = {
        "features": {
            "AMT_INCOME_TOTAL": income,
            "AMT_CREDIT": credit,
            "DAYS_BIRTH": int(age_years * -365),
            "DAYS_EMPLOYED": int(employed_years * -365),
            "EXT_SOURCE_2": ext_source_2,
            "EXT_SOURCE_3": ext_source_3
        }
    }
    
    # 2. Call the live Docker API
    try:
        with st.spinner("Consulting XGBoost Engine..."):
            response = requests.post("http://localhost:8000/predict", json=payload)
            response.raise_for_status() # Raise an error if the API crashes
            result = response.json()
            
        # 3. Display the Results Beautifully
        status = result["applicant_status"]
        prob = result["default_probability"] * 100
        tier = result["risk_tier"]
        
        st.subheader("Verdict:")
        
        if status == "APPROVE":
            st.success(f"✅ **{status}**")
            st.info(f"**Risk Tier:** {tier} | **Default Probability:** {prob:.2f}%")
        else:
            st.error(f"❌ **{status}**")
            st.warning(f"**Risk Tier:** {tier} | **Default Probability:** {prob:.2f}%")
            
    except requests.exceptions.ConnectionError:
        st.error("🚨 API Offline: Could not connect to localhost:8000. Is your Docker container running?")