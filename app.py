import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load("Heart_disease_model.pkl")

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")
st.title("❤️ Heart Disease Prediction App")
st.markdown("Enter patient details to predict the risk of heart disease.")

# --- Define category mappings ---
sex_dict = {0: "Female", 1: "Male"}
cp_dict = {1: "Typical Angina", 2: "Atypical Angina", 3: "Non-anginal Pain", 4: "Asymptomatic"}
fbs_dict = {0: "No", 1: "Yes"}
restecg_dict = {0: "Normal", 1: "ST-T Wave Abnormality", 2: "Left Ventricular Hypertrophy"}
exang_dict = {0: "No", 1: "Yes"}
slope_dict = {1: "Upsloping", 2: "Flat", 3: "Downsloping"}
thal_dict = {1: "Normal", 2: "Fixed Defect", 3: "Reversible Defect"}

# --- Input Fields ---
age = st.number_input("Age", min_value=10, max_value=120, value=50)
sex = st.selectbox("Sex", options=list(sex_dict.values()))
cp = st.selectbox("Chest Pain Type", options=list(cp_dict.values()))
trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=250, value=120)
chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=list(fbs_dict.values()))
restecg = st.selectbox("Resting ECG Results", options=list(restecg_dict.values()))
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
exang = st.selectbox("Exercise Induced Angina", options=list(exang_dict.values()))
oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
slope = st.selectbox("Slope of Peak Exercise ST Segment", options=list(slope_dict.values()))
ca = st.slider("Number of Major Vessels (ca)", min_value=0, max_value=4, value=0)
thal = st.selectbox("Thalassemia (thal)", options=list(thal_dict.values()))

# --- Predict Button ---
if st.button("🔍 Predict Heart Disease"):
    # Convert selected string values back to numbers using dict lookup
    sex_val = [k for k, v in sex_dict.items() if v == sex][0]
    cp_val = [k for k, v in cp_dict.items() if v == cp][0]
    fbs_val = [k for k, v in fbs_dict.items() if v == fbs][0]
    restecg_val = [k for k, v in restecg_dict.items() if v == restecg][0]
    exang_val = [k for k, v in exang_dict.items() if v == exang][0]
    slope_val = [k for k, v in slope_dict.items() if v == slope][0]
    thal_val = [k for k, v in thal_dict.items() if v == thal][0]

    # Prepare input for model
    user_data = np.array([[age, sex_val, cp_val, trestbps, chol, fbs_val,
                           restecg_val, thalach, exang_val, oldpeak, slope_val, ca, thal_val]])

    # Predict
    prediction = model.predict(user_data)[0]
    probability = model.predict_proba(user_data)[0][1]

    # --- Output Section ---
    st.subheader("Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Disease (Confidence: {probability*100:.2f}%)")
    else:
        st.success(f"✅ No Heart Disease Detected (Confidence: {(1 - probability)*100:.2f}%)")

st.markdown("---")
st.caption("Developed with ❤️ using Streamlit and AdaBoost Classifier")
