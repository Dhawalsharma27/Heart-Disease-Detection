import streamlit as st
import numpy as np
import joblib

# Load the saved model
model = joblib.load("heart_disease_model.pkl")

st.set_page_config(page_title="Heart Disease Prediction", layout="centered")

st.title("❤️ Heart Disease Prediction System")
st.write("Enter patient details below to check the risk of heart disease.")

st.markdown("""
        <style>
            .main{
            background-color: #F5F5F5;
            }
""", unsafe_allow_html=True)

# Input fields for 13 features
age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex (1 = Male, 0 = Female)", [0, 1])
cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 50, 250, 120)
chol = st.number_input("Cholesterol (mg/dl)", 50, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = Yes, 0 = No)", [0, 1])
restecg = st.selectbox("Resting ECG (0–2)", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved", 50, 250, 150)
exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1])
oldpeak = st.number_input("Oldpeak (ST depression)", 0.0, 10.0, 1.0, step=0.1)
slope = st.selectbox("Slope (0–2)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0–4)", [0, 1, 2, 3, 4])
thal = st.selectbox("Thal (0 = Normal, 1 = Fixed Defect, 2 = Reversible)", [0, 1, 2])

# Predict button
if st.button("Predict ➜"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])

    prediction = model.predict(input_data)

    if prediction[0] == 0:
        st.success("🟢 The person does NOT have heart disease.")
    else:
        st.error("🔴 The person HAS heart disease. Consult a doctor immediately.")
