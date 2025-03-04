import pandas as pd
import joblib
import streamlit as st
from sklearn.preprocessing import StandardScaler

# Load the saved model
model_path = 'svc_wine_quality_model.pkl'  # Replace with the correct path
model = joblib.load(model_path)

# Streamlit App
st.title("Wine Quality Prediction")

st.markdown("Enter the physicochemical properties of the wine to predict its quality.")

# Create a form to input features
with st.form("wine_quality_form"):
    fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0, step=0.1, format="%.1f")
    volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0, step=0.01, format="%.2f")
    citric_acid = st.number_input("Citric Acid", min_value=0.0, step=0.01, format="%.2f")
    residual_sugar = st.number_input("Residual Sugar", min_value=0.0, step=0.1, format="%.1f")
    chlorides = st.number_input("Chlorides", min_value=0.0, step=0.0001, format="%.4f")
    free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=0.0, step=1.0, format="%.0f")
    total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=0.0, step=1.0, format="%.0f")
    density = st.number_input("Density", min_value=0.0, step=0.0001, format="%.4f")
    pH = st.number_input("pH", min_value=0.0, step=0.01, format="%.2f")
    sulphates = st.number_input("Sulphates", min_value=0.0, step=0.01, format="%.2f")
    alcohol = st.number_input("Alcohol", min_value=0.0, step=0.1, format="%.1f")

    # Submit button
    submitted = st.form_submit_button("Predict Quality")

# Perform prediction if the form is submitted
if submitted:
    # Combine inputs into a DataFrame for prediction
    input_data = pd.DataFrame({
        'fixed acidity': [fixed_acidity],
        'volatile acidity': [volatile_acidity],
        'citric acid': [citric_acid],
        'residual sugar': [residual_sugar],
        'chlorides': [chlorides],
        'free sulfur dioxide': [free_sulfur_dioxide],
        'total sulfur dioxide': [total_sulfur_dioxide],
        'density': [density],
        'pH': [pH],
        'sulphates': [sulphates],
        'alcohol': [alcohol]
    })

    # Standardize the input data (assuming the same scaler was used during training)
    # scaler = StandardScaler()
    scaler_path = 'standard_scaler.pkl'  # Ensure this file exists
    scaler = joblib.load(scaler_path)
    input_data_scaled = scaler.transform(input_data)
    # st.title(input_data)
    # st.title(input_data_scaled)
    # Predict quality
    prediction = model.predict(input_data_scaled)

    # Display the result
    st.write(f"Predicted Wine Quality: {prediction[0]}")
