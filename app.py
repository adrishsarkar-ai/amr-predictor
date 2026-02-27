import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="AI AMR Predictor", layout="centered")

# --------- Custom CSS Styling ----------
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #1e3c72, #2a5298);
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stButton>button {
        background-color: #2a5298;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #1e3c72;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🧬 AI-Based AMR Prediction System")
st.markdown("### Clinical Decision Support Tool")

# --------------------------------------------
# DATA GENERATION
# --------------------------------------------
np.random.seed(42)
n = 1000

data = pd.DataFrame({
    "Age": np.random.randint(18, 85, n),
    "WBC": np.random.randint(4000, 20000, n),
    "Hospital_Days": np.random.randint(1, 20, n),
    "Previous_Antibiotic": np.random.randint(0, 2, n),
    "ICU_Stay": np.random.randint(0, 2, n),
    "Infection_Type": np.random.choice(["UTI", "Pneumonia", "Sepsis"], n),
    "Antibiotic": np.random.choice(["Ceftriaxone", "Meropenem", "Ciprofloxacin"], n)
})

data["Resistance"] = (
    (data["Previous_Antibiotic"] == 1) &
    (data["Hospital_Days"] > 7) &
    (data["ICU_Stay"] == 1)
).astype(int)

le_inf = LabelEncoder()
le_ab = LabelEncoder()

data["Infection_Type"] = le_inf.fit_transform(data["Infection_Type"])
data["Antibiotic"] = le_ab.fit_transform(data["Antibiotic"])

X = data.drop("Resistance", axis=1)
y = data["Resistance"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))

st.sidebar.title("Model Information")
st.sidebar.success(f"Accuracy: {round(accuracy*100,2)}%")
st.sidebar.info("Model: Random Forest (200 Trees)")
st.sidebar.info("Dataset: Synthetic (Demo Purpose)")

# --------------------------------------------
# INPUT SECTION
# --------------------------------------------
st.header("Patient Clinical Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 100)
    wbc = st.number_input("WBC Count", 4000, 20000)
    hospital_days = st.number_input("Hospital Stay (Days)", 1, 30)

with col2:
    prev_ab = st.selectbox("Previous Antibiotic Use", ["No", "Yes"])
    icu = st.selectbox("ICU Stay", ["No", "Yes"])
    infection = st.selectbox("Infection Type", ["UTI", "Pneumonia", "Sepsis"])
    antibiotic = st.selectbox("Antibiotic", ["Ceftriaxone", "Meropenem", "Ciprofloxacin"])

if st.button("Predict Resistance"):

    prev_ab_val = 1 if prev_ab == "Yes" else 0
    icu_val = 1 if icu == "Yes" else 0

    infection_enc = le_inf.transform([infection])[0]
    antibiotic_enc = le_ab.transform([antibiotic])[0]

    patient = pd.DataFrame([{
        "Age": age,
        "WBC": wbc,
        "Hospital_Days": hospital_days,
        "Previous_Antibiotic": prev_ab_val,
        "ICU_Stay": icu_val,
        "Infection_Type": infection_enc,
        "Antibiotic": antibiotic_enc
    }])

    probability = model.predict_proba(patient)[0][1]

    st.markdown("---")
    st.subheader(f"Resistance Probability: {round(probability*100,2)}%")

    if probability > 0.7:
        st.error("⚠ High Risk – Avoid this antibiotic.")
    elif probability > 0.3:
        st.warning("⚠ Moderate Risk – Use with monitoring.")
    else:
        st.success("✅ Likely Effective.")

st.markdown("---")
st.caption("⚠ For academic/demo purposes only. Not for clinical use.")