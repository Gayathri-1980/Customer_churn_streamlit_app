import streamlit as st
import numpy as np
import pickle
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# -------------------------------------------
# Load Model + Scaler
# -------------------------------------------
@st.cache_resource
def load_model():
    with open("models/churn_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

# -------------------------------------------
# Streamlit Page Setup
# -------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide"
)

st.title("Customer Churn Prediction Dashboard")
st.write("Use the controls on the left to estimate the likelihood of customer churn.")

# -------------------------------------------
# Sidebar Inputs
# -------------------------------------------
st.sidebar.header("Customer Information")

age = st.sidebar.slider("Age", 18, 90, 30)
income = st.sidebar.number_input("Annual Income ($)", 10000, 200000, 50000, step=1000)
tenure = st.sidebar.slider("Tenure (Years)", 0, 20, 5)
support_calls = st.sidebar.slider("Support Calls", 0, 15, 2)

predict_btn = st.sidebar.button("Predict Churn")

# -------------------------------------------
# KPI Metrics (Static)
# -------------------------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Customers", "500")
col2.metric("Overall Churn Rate", "31.2%")
col3.metric("Average Income", "$68,168")
col4.metric("Average Tenure (Years)", "5.0")

st.divider()

# -------------------------------------------
# Prediction Section
# -------------------------------------------
st.subheader("Prediction Result")

if predict_btn:
    input_data = np.array([[age, income, tenure, support_calls]])
    scaled = scaler.transform(input_data)

    prediction = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0][1]  # prob of churn

    # -------------------------------
    # Probability Gauge Meter
    # -------------------------------
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={"text": "Churn Probability (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#003f5c"},
            "steps": [
                {"range": [0, 30], "color": "#d4f4dd"},
                {"range": [30, 60], "color": "#fff3cd"},
                {"range": [60, 100], "color": "#f8d7da"},
            ],
        },
    ))

    st.plotly_chart(gauge_fig, use_container_width=True)

    # Result Message
    if prediction == 1:
        st.error(f"Customer is LIKELY to churn. Predicted Probability: {probability:.2f}")
    else:
        st.success(f"Customer is NOT likely to churn. Predicted Probability: {probability:.2f}")

else:
    st.info("Adjust the inputs on the left and click **Predict Churn**.")

st.divider()

# -------------------------------------------
# Feature Importance Section
# -------------------------------------------
st.subheader("Feature Importance (Logistic Regression Coefficients)")

feature_names = ["Age", "Income", "Tenure", "Support Calls"]
coefficients = model.coef_[0]

fi_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients
})

fig = px.bar(
    fi_df,
    x="Feature",
    y="Coefficient",
    title="Impact of Each Feature on Churn",
    color="Coefficient",
    color_continuous_scale="RdBu",
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------
# Explanation Section
# -------------------------------------------
st.subheader("How the Model Makes Decisions")
st.write("""
- Customers with **high support call frequency** tend to churn more.  
- Customers with **lower income levels** show increased churn likelihood.  
- **Shorter tenure** (new customers) increases the chance of churn.  
- **Younger customers** show slightly higher churn tendency.  
""")

st.info("Model Used: Logistic Regression trained on the generated synthetic dataset.")
