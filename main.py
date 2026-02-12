import joblib
import pandas as pd
import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="Churn Predictor",
    layout="wide",
    page_icon="📞",
)



# 2. Load your best model (the Logistic Regression Pipeline)
@st.cache_resource  # This keeps the model in memory so it doesn't reload every click
def load_model():
    return joblib.load("models/logistic/model.joblib")


model = load_model()

# 3. The UI Layout
st.title("📞 Customer Churn Prediction Dashboard")
st.markdown("Use this tool to predict if a customer will leave based on their profile.")

# st.file_uploader("Choose a file")

# We use columns to make the UI look professional
col1, col2 = st.columns([1, 2])

with col1:
    with st.container(border=True):
        st.subheader("Customer Details")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Is Senior Citizen?", [0, 1])
        partner = st.selectbox("Has Partner?", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
        tenure = st.slider("Tenure (Months)", 0, 72, 12)

    with st.container(border=True):
        st.subheader("Financials")
        monthly = st.number_input("Monthly Charges ($)", 0.0, 150.0, 70.0)
        total = st.number_input("Total Charges ($)", 0.0, 9000.0, 800.0)
        st.divider()

with col2:
    with st.container(border=True):
        st.subheader("Services & Contract")
        sub_col1, sub_col2 = st.columns([1, 2], border=True)
        with sub_col1:
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multi_service = st.selectbox("Multiple Service", ["Yes", "No"])
            online_service = st.selectbox("Online Service", ["Yes", "No"])
        with sub_col2:
            device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            streaming_service = st.selectbox("Streaming Service", ["Yes", "No"])



        with st.container(border=True):
            contract = st.radio("Contract Type", ["Month-to-month", "One year", "Two year"])
            internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
            tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            payment = st.selectbox("Payment Method",
                                   ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])

            # Logic to calculate the "Feature Engineered" columns we made earlier
            service_list = [tech_support, internet]  # simplified for example
            total_services = 1 if internet != "No" else 0
            if tech_support == "Yes": total_services += 1

    # 4. The Prediction Button
    if st.button(
            "Predict Churn Risk",
            use_container_width=True,
            type="primary",
            help="Click to predict if this customer is likely to churn."
    ):
        # Create the dataframe for the model (Must match x_train columns exactly)
        input_df = pd.DataFrame({
            'gender': [gender], 'SeniorCitizen': [senior], 'Partner': [partner],
            'Dependents': [dependents], 'tenure': [tenure], 'PhoneService': [phone_service],
            'MultipleLines': [multi_service], 'InternetService': [internet], 'OnlineSecurity': [online_service],
            'OnlineBackup': [online_service], 'DeviceProtection': ['No'], 'TechSupport': [tech_support],
            'StreamingTV': [streaming_service], 'StreamingMovies': [streaming_service], 'Contract': [contract],
            'PaperlessBilling': ['Yes'], 'PaymentMethod': [payment],
            'MonthlyCharges': [monthly], 'TotalCharges': [total],
            'Total_Services': [total_services]  # Our engineered feature!
        })

        # Make Prediction
        probability = model.predict_proba(input_df)[0][1]

        # Display Result
        st.divider()
        if probability > 0.5:
            st.error(f"### ⚠️ High Risk: {probability:.1%}")
            st.write("This customer is likely to leave. Consider offering a contract discount.")
        else:
            st.success(f"### ✅ Low Risk: {probability:.1%}")
            st.write("This customer is likely to stay.")
