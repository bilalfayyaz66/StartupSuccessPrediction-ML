import pandas as pd
import streamlit as st
import joblib

from config import FINAL_MODEL_PATH, IMPUTER_PATH

st.set_page_config(page_title="Startup Success Predictor", layout="centered")

st.title("Startup Success Predictor")
st.write("Enter startup details to estimate probability of success.")

model = joblib.load(FINAL_MODEL_PATH)
imputer = joblib.load(IMPUTER_PATH)

relationships = st.number_input("Relationships", min_value=0, value=10)
funding_rounds = st.number_input("Funding Rounds", min_value=0, value=2)
funding_total_usd = st.number_input("Funding Total USD", min_value=0.0, value=5000000.0)
milestones = st.number_input("Milestones", min_value=0, value=3)
avg_participants = st.number_input("Average Participants", min_value=0.0, value=3.0)

is_CA = st.selectbox("Is California based?", [0, 1], index=1)
is_NY = st.selectbox("Is New York based?", [0, 1], index=0)
is_MA = st.selectbox("Is Massachusetts based?", [0, 1], index=0)
is_TX = st.selectbox("Is Texas based?", [0, 1], index=0)
is_otherstate = st.selectbox("Other State?", [0, 1], index=0)

category_code = st.number_input("Category Code", min_value=0, value=5)

is_software = st.selectbox("Software?", [0, 1], index=1)
is_web = st.selectbox("Web?", [0, 1], index=0)
is_mobile = st.selectbox("Mobile?", [0, 1], index=0)
is_enterprise = st.selectbox("Enterprise?", [0, 1], index=1)
is_advertising = st.selectbox("Advertising?", [0, 1], index=0)
is_gamesvideo = st.selectbox("Games/Video?", [0, 1], index=0)
is_ecommerce = st.selectbox("E-commerce?", [0, 1], index=0)
is_biotech = st.selectbox("Biotech?", [0, 1], index=0)
is_consulting = st.selectbox("Consulting?", [0, 1], index=0)
is_othercategory = st.selectbox("Other Category?", [0, 1], index=0)

has_VC = st.selectbox("Has VC funding?", [0, 1], index=1)
has_angel = st.selectbox("Has angel funding?", [0, 1], index=1)
has_roundA = st.selectbox("Has Round A?", [0, 1], index=1)
has_roundB = st.selectbox("Has Round B?", [0, 1], index=0)
has_roundC = st.selectbox("Has Round C?", [0, 1], index=0)
has_roundD = st.selectbox("Has Round D?", [0, 1], index=0)
is_top500 = st.selectbox("Is Top 500?", [0, 1], index=0)

if st.button("Predict"):
    sample_data = pd.DataFrame([{
        "relationships": relationships,
        "funding_rounds": funding_rounds,
        "funding_total_usd": funding_total_usd,
        "milestones": milestones,
        "is_CA": is_CA,
        "is_NY": is_NY,
        "is_MA": is_MA,
        "is_TX": is_TX,
        "is_otherstate": is_otherstate,
        "category_code": category_code,
        "is_software": is_software,
        "is_web": is_web,
        "is_mobile": is_mobile,
        "is_enterprise": is_enterprise,
        "is_advertising": is_advertising,
        "is_gamesvideo": is_gamesvideo,
        "is_ecommerce": is_ecommerce,
        "is_biotech": is_biotech,
        "is_consulting": is_consulting,
        "is_othercategory": is_othercategory,
        "has_VC": has_VC,
        "has_angel": has_angel,
        "has_roundA": has_roundA,
        "has_roundB": has_roundB,
        "has_roundC": has_roundC,
        "has_roundD": has_roundD,
        "avg_participants": avg_participants,
        "is_top500": is_top500
    }])

    sample_data["funding_per_round"] = sample_data["funding_total_usd"] / (sample_data["funding_rounds"] + 1)
    sample_data["participants_per_round"] = sample_data["avg_participants"] / (sample_data["funding_rounds"] + 1)
    sample_data["milestones_per_round"] = sample_data["milestones"] / (sample_data["funding_rounds"] + 1)
    sample_data["network_funding_strength"] = sample_data["relationships"] * sample_data["avg_participants"]
    sample_data["rounds_x_top500"] = sample_data["funding_rounds"] * sample_data["is_top500"]

    sample_data = pd.DataFrame(imputer.transform(sample_data), columns=sample_data.columns)

    prediction = model.predict(sample_data)[0]
    probability = model.predict_proba(sample_data)[0][1]

    st.subheader("Prediction Result")
    st.write(f"Predicted Class: {prediction}")
    st.write(f"Success Probability: {probability:.2%}")

    if prediction == 1:
        st.success("This startup is predicted as likely successful.")
    else:
        st.error("This startup is predicted as less likely to succeed.")