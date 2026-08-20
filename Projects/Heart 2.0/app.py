import numpy as np
import streamlit as st
import pandas as pd
import joblib

import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "logRegression.pkl")

model = joblib.load(model_path)
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")
scaler = joblib.load(scaler_path)
columns_path = os.path.join(BASE_DIR, "columns.pkl")
columns = joblib.load(columns_path)


st.title("Heart Disease Detection")
st.markdown("Provide the following details")

age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("SEX", ['M', 'F'])
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
RestingBP = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
FastingBS = st.selectbox("Fasting Blood Sugar >120 mg/dL", [0, 1])
RestingECG = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
MaxHR = st.slider("MAX Heart Rate", 60, 220, 150)
ExerciseAngina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
OldPeak = st.slider("OldPeak(ST Depression)", 0.0, 6.0, 1.0)
ST_Slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

if st.button("Predict"):
    raw_input = {
        "Age": age,
        "RestingBP": RestingBP,
        "Cholesterol": cholesterol,
        "FastingBS": FastingBS,
        "MaxHR": MaxHR,
        "Oldpeak": OldPeak,
        "Sex_" + sex: 1,
        "ChestPainType_" + chest_pain: 1,
        "RestingECG_" + RestingECG: 1,
        "ExerciseAngina_" + ExerciseAngina: 1,
        "ST_Slope_" + ST_Slope: 1
    }
    input_df = pd.DataFrame([raw_input])

    for column in columns:
        if column not in input_df.columns:
            input_df[column] = 0
    input_df = input_df[columns]

    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.error("High risk of Heart Disease")
    else:
        st.success("Low risk of Heart Disease")
