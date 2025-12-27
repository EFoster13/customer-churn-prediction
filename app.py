import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# Load the trained model and scaler
@st.cache_resource
def load_model():
    with open('models/churn_model_logistic.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

# Title and description
st.title("Customer Churn Prediction System")

st.markdown("""
<div style='background: linear-gradient(135deg, #10b981 0%, #059669 100%); 
            padding: 25px; border-radius: 12px; text-align: center; 
            margin-bottom: 30px; box-shadow: 0 4px 15px rgba(16, 185, 129, 0.2);'>
    <h2 style='color: white; margin: 0; font-weight: 600;'>
        AI-Powered Customer Retention Intelligence
    </h2>
    <p style='color: #d1fae5; margin: 8px 0 0 0; font-size: 16px;'>
        Predict Risk • Analyze Patterns • Drive Retention
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
This application predicts whether a telecom customer will churn based on their profile and services.
**Model Performance:** 84.6% ROC-AUC | Built with Logistic Regression
""")


st.markdown("""
<style>
    /* Main container */
    .main {
        background-color: #f0fdf4;
    }
    
    /* Title styling */
    h1 {
        color: #065f46;
        text-align: center;
        padding: 20px 0;
        font-size: 3em;
        background: linear-gradient(90deg, #10b981, #059669);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Subheaders */
    h2 {
        color: #047857;
        margin-top: 20px;
        border-left: 4px solid #10b981;
        padding-left: 15px;
    }
    
    h3 {
        color: #059669;
    }
    
    /* Prediction button */
    .stButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        font-weight: 600;
        border-radius: 12px;
        padding: 15px 30px;
        font-size: 18px;
        border: none;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4);
    }
    
    /* Metrics styling */
    [data-testid="stMetricValue"] {
        font-size: 32px;
        color: #065f46;
        font-weight: bold;
    }
    
    [data-testid="stMetricLabel"] {
        color: #047857;
        font-weight: 500;
    }
    
    /* Success/Error boxes */
    .stAlert {
        border-radius: 10px;
        padding: 20px;
        font-weight: 500;
    }
    
    /* Dividers */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #10b981, transparent);
        margin: 30px 0;
    }
    
    /* Input fields */
    .stSelectbox label, .stNumberInput label, .stSlider label {
        color: #065f46;
        font-weight: 500;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #065f46 0%, #047857 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: #f0fdf4 !important;
    }
    
    /* Cards effect for columns */
    div[data-testid="column"] {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(6, 95, 70, 0.1);
        border-left: 3px solid #10b981;
        margin: 10px 5px;
    }
    
    /* Markdown text */
    .markdown-text-container {
        color: #047857;
    }
            
    /* Slider styling */
    .stSlider [data-baseweb="slider"] {
        background-color: transparent;
    }
    
    /* Slider track */
    .stSlider [data-baseweb="slider"] [data-testid="stTickBar"] {
        background-color: #d1fae5 !important;
    }
    
    /* Slider filled portion */
    .stSlider [data-baseweb="slider"] > div:first-child > div:first-child {
        background-color: #10b981 !important;
    }
    
    /* Slider thumb */
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background-color: #059669 !important;
        border: 4px solid #10b981 !important;
        width: 24px !important;
        height: 24px !important;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.4) !important;
    }
    
    /* Slider thumb on hover/drag */
    .stSlider [data-baseweb="slider"] [role="slider"]:hover,
    .stSlider [data-baseweb="slider"] [role="slider"]:active {
        background-color: #047857 !important;
        box-shadow: 0 0 0 10px rgba(16, 185, 129, 0.15) !important;
    }
             
</style>
""", unsafe_allow_html=True)

# Create input fields
col1, col2 = st.columns(2)

with col1:
    st.subheader("Demographics")
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Has Partner?", ["No", "Yes"])
    dependents = st.selectbox("Has Dependents?", ["No", "Yes"])

with col2:
    st.subheader("Account Information")
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
    payment_method = st.selectbox("Payment Method", 
                                  ["Electronic check", "Mailed check", 
                                   "Bank transfer (automatic)", "Credit card (automatic)"])

st.markdown("---")

col3, col4 = st.columns(2)

with col3:
    st.subheader("Services")
    phone_service = st.selectbox("Phone Service", ["No", "Yes"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    
with col4:
    st.subheader("Charges")
    monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 150.0, 50.0, step=0.5)
    total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 500.0, step=10.0)

# Additional services (only if has internet)
if internet_service != "No":
    st.subheader("Internet Services")
    col5, col6 = st.columns(2)
    
    with col5:
        online_security = st.selectbox("Online Security", ["No", "Yes"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes"])
    
    with col6:
        tech_support = st.selectbox("Tech Support", ["No", "Yes"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])
else:
    online_security = "No"
    online_backup = "No"
    device_protection = "No"
    tech_support = "No"
    streaming_tv = "No"
    streaming_movies = "No"

st.markdown("---")

# Prediction button
if st.button("Predict Churn", type="primary", use_container_width=True):
    
    # Prepare input data (same preprocessing as training)
    input_data = {
        'gender': 1 if gender == "Male" else 0,
        'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
        'Partner': 1 if partner == "Yes" else 0,
        'Dependents': 1 if dependents == "Yes" else 0,
        'tenure': tenure,
        'PhoneService': 1 if phone_service == "Yes" else 0,
        'MultipleLines': 1 if multiple_lines == "Yes" else 0,
        'OnlineSecurity': 1 if online_security == "Yes" else 0,
        'OnlineBackup': 1 if online_backup == "Yes" else 0,
        'DeviceProtection': 1 if device_protection == "Yes" else 0,
        'TechSupport': 1 if tech_support == "Yes" else 0,
        'StreamingTV': 1 if streaming_tv == "Yes" else 0,
        'StreamingMovies': 1 if streaming_movies == "Yes" else 0,
        'PaperlessBilling': 1 if paperless_billing == "Yes" else 0,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'InternetService_Fiber optic': 1 if internet_service == "Fiber optic" else 0,
        'InternetService_No': 1 if internet_service == "No" else 0,
        'Contract_One year': 1 if contract == "One year" else 0,
        'Contract_Two year': 1 if contract == "Two year" else 0,
        'PaymentMethod_Credit card (automatic)': 1 if payment_method == "Credit card (automatic)" else 0,
        'PaymentMethod_Electronic check': 1 if payment_method == "Electronic check" else 0,
        'PaymentMethod_Mailed check': 1 if payment_method == "Mailed check" else 0,
    }
    
    # Add engineered features
    input_data['avg_monthly_cost'] = total_charges / (tenure + 1)
    input_data['is_new_customer'] = 1 if tenure <= 12 else 0
    input_data['high_value_customer'] = 1 if monthly_charges > 70 else 0
    
    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Scale the features that were scaled during training
    cols_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges', 'avg_monthly_cost']
    input_df[cols_to_scale] = scaler.transform(input_df[cols_to_scale])
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]
    
    # Display results
    st.markdown("---")
    st.subheader("Prediction Results")
    
    col_result1, col_result2, col_result3 = st.columns(3)
    
    with col_result1:
        if prediction == 1:
            st.error("HIGH RISK - Customer Likely to Churn")
        else:
            st.success("LOW RISK - Customer Likely to Stay")
    
    with col_result2:
        st.metric("Churn Probability", f"{probability[1]*100:.1f}%")
    
    with col_result3:
        st.metric("Retention Probability", f"{probability[0]*100:.1f}%")
    
    # Risk assessment and recommendations
    st.markdown("---")
    st.subheader("Risk Assessment & Recommendations")
    
    # Generate personalized recommendations
    recommendations = []
    
    if contract == "Month-to-month":
        recommendations.append("**High Risk Factor:** Month-to-month contract (42.7% churn rate)")
        recommendations.append("**Recommendation:** Offer 15-20% discount for switching to 1-year or 2-year contract")
    
    if internet_service == "Fiber optic":
        recommendations.append("**High Risk Factor:** Fiber optic service (41.9% churn rate)")
        recommendations.append("**Recommendation:** Enroll in Fiber Loyalty Bundle with discounted rates")
    
    if tenure <= 6:
        recommendations.append("**High Risk Factor:** New customer (most churn in first 6 months)")
        recommendations.append("**Recommendation:** Activate '90-Day Success Program' with proactive check-ins")
    
    if monthly_charges > 70:
        recommendations.append("**Moderate Risk:** High monthly charges ($70+)")
        recommendations.append("**Recommendation:** Review service bundle for cost optimization opportunities")
    
    if len(recommendations) == 0:
        st.success("This customer has low churn risk factors. Continue standard service.")
    else:
        for rec in recommendations:
            st.markdown(rec)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>Customer Churn Prediction System</strong></p>
    <p>Model: Logistic Regression | ROC-AUC: 84.6% | Dataset: 7,043 customers</p>
    <p>Built with Python, Scikit-learn, and Streamlit</p>
</div>
""", unsafe_allow_html=True)