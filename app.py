import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load data and preprocess
data = pd.read_csv(r"C:\Users\LENOVO\Desktop\Diabetes\Diabeties-Pridiction\diabetes_prediction_dataset.csv")
data["gender"] = data["gender"].map({"Male": 1, "Female": 2, "Other": 3})
data["smoking_history"] = data["smoking_history"].map({
    "never": 1, "No Info": 2, "current": 3, "former": 4, "ever": 5, "not current": 6
})

# Splitting dataset
X = data.drop("diabetes", axis=1)
y = data["diabetes"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# CSS for styling
st.markdown("""
    <style>
    body {
        font-family: 'Poppins', sans-serif;
        background: linear-gradient(to bottom, #e0eafc, #cfdef3);
        color: #333;
    }
    .header {
        text-align: center;
        padding: 20px;
        margin-bottom: 20px;
        border-radius: 12px;
        background: #007BFF;
        color: white;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
    }
    .form-container {
        max-width: 600px;
        margin: auto;
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
    }
    .footer {
        text-align: center;
        padding: 10px;
        margin-top: 20px;
        background: #333;
        color: white;
        border-radius: 12px;
    }
    .stButton>button {
        background-color: #007BFF;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("""
<div class="header">
    <h1>Diabetes Prediction App</h1>
    <p>Enter your details to predict diabetes risk with ML-powered accuracy.</p>
</div>
""", unsafe_allow_html=True)

# Form Section
st.markdown('<div class="form-container">', unsafe_allow_html=True)
with st.form("prediction_form"):
    gender = st.selectbox("Gender", ["Select", "Male", "Female", "Other"])
    age = st.slider("Age", 1, 120, 25)
    hypertension = st.radio("Hypertension", ["No", "Yes"])
    heart_disease = st.radio("Heart Disease", ["No", "Yes"])
    smoking_history = st.selectbox("Smoking History", ["Select", "Never", "No Info", "Current", "Former", "Ever", "Not Current"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
    hba1c = st.number_input("HbA1c Level", min_value=4.0, max_value=15.0, value=5.5)
    blood_glucose = st.number_input("Blood Glucose Level", min_value=50, max_value=250, value=100)
    submit = st.form_submit_button("Predict")

st.markdown('</div>', unsafe_allow_html=True)

# Prediction
if submit:
    if gender == "Select" or smoking_history == "Select":
        st.error("Please select valid options for Gender and Smoking History.")
    else:
        input_data = np.array([[age, {"Male": 1, "Female": 2, "Other": 3}[gender],
                                1 if hypertension == "Yes" else 0, 1 if heart_disease == "Yes" else 0,
                                {"Never": 1, "No Info": 2, "Current": 3, "Former": 4, "Ever": 5, "Not Current": 6}[smoking_history],
                                bmi, hba1c, blood_glucose]])
        prediction = model.predict(input_data)[0]
        result = "Diabetic" if prediction == 1 else "Not Diabetic"
        color = "red" if prediction == 1 else "green"
        st.markdown(f"<h3 style='text-align:center; color:{color}'>{result}</h3>", unsafe_allow_html=True)

# Footer Section
st.markdown("""
<div class="footer">
    <p>© 2024 Diabetes Prediction App. Powered by Streamlit & ML.</p>
</div>
""", unsafe_allow_html=True)
