import streamlit as st
import joblib
import pandas as pd
import xgboost as xgb

# ====================== CONFIG ======================
st.set_page_config(
    page_title="🕵️ Fake Job Detector",
    page_icon="🔍",
    layout="centered"
)

# ====================== LOAD MODEL ======================
@st.cache_resource
def load_components():
    try:
        preprocessor = joblib.load('preprocessor.pkl')
        booster = xgb.Booster()
        booster.load_model('xgb_model.json')
        return preprocessor, booster
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {str(e)}")
        st.stop()

preprocessor, booster = load_components()

# ====================== TITLE ======================
st.title("🕵️ Fake Job Posting Detector")
st.markdown("Deteksi lowongan kerja palsu menggunakan Machine Learning")
st.divider()

# ====================== TOMBOL CONTOH ======================
if st.button("📋 Gunakan Contoh Lowongan Palsu", use_container_width=True, type="secondary"):
    # Simpan contoh ke session state
    st.session_state.update({
        "title": "Earn $5000/month - Work From Home",
        "location": "Remote",
        "employment_type": "Full-time",
        "required_experience": "Entry level",
        "department": "Marketing",
        "required_education": "High School",
        "industry": "Marketing",
        "has_company_logo": "No",
        "has_questions": "No",
        "company_profile": "We are a fast growing international company offering remote opportunities.",
        "description": "No experience required! Start immediately and get paid every week. Work from home easily.",
        "requirements": "Only need a laptop and stable internet connection. No degree needed.",
        "benefits": "High salary, flexible working hours, work from anywhere."
    })
    st.rerun()

# ====================== INPUT FORM ======================
col1, col2 = st.columns(2)

with col1:
    title = st.text_input("Job Title", key="title")
    location = st.text_input("Location", key="location")
    employment_type = st.selectbox("Employment Type", 
                                   ["Full-time", "Part-time", "Internship", "Contract", "Freelance", "Unknown"], 
                                   key="employment_type")
    required_experience = st.selectbox("Required Experience", 
                                       ["Entry level", "Mid level", "Senior level", "Unknown"], 
                                       key="required_experience")

with col2:
    department = st.text_input("Department", key="department")
    required_education = st.text_input("Required Education", key="required_education")
    industry = st.text_input("Industry", key="industry")
    has_company_logo = st.radio("Has Company Logo?", ["Yes", "No"], horizontal=True, key="has_company_logo")
    has_questions = st.radio("Has Screening Questions?", ["Yes", "No"], horizontal=True, key="has_questions")

company_profile = st.text_area("Company Profile", height=100, key="company_profile")
description = st.text_area("Job Description", height=150, key="description")
requirements = st.text_area("Requirements", height=120, key="requirements")
benefits = st.text_area("Benefits", height=100, key="benefits")

# ====================== PREDICTION ======================
if st.button("🔍 Prediksi Sekarang", type="primary", use_container_width=True):
    if not description or not description.strip():
        st.warning("⚠️ Mohon isi Job Description terlebih dahulu.")
    else:
        input_data = pd.DataFrame([{
            'title': title,
            'location': location,
            'department': department,
            'company_profile': company_profile,
            'description': description,
            'requirements': requirements,
            'benefits': benefits,
            'telecommuting': 1 if "remote" in str(location).lower() else 0,
            'has_company_logo': 1 if has_company_logo == "Yes" else 0,
            'has_questions': 1 if has_questions == "Yes" else 0,
            'employment_type': employment_type,
            'required_experience': required_experience,
            'required_education': required_education,
            'industry': industry,
            'function': department,
            'text_combined': f"{title} {company_profile} {description} {requirements} {benefits}"
        }])

        X_transformed = preprocessor.transform(input_data)
        dmatrix = xgb.DMatrix(X_transformed)
        prob_fake = float(booster.predict(dmatrix)[0])
        prob_fake = max(0.0, min(1.0, prob_fake))

        prediction = 1 if prob_fake > 0.5 else 0

        if prediction == 1:
            st.error("🚨 **LOWONGAN INI DIDUGA PALSU**")
            st.progress(prob_fake)
            st.metric("Probabilitas Scam", f"{prob_fake:.1%}")
        else:
            st.success("✅ **LOWONGAN INI TERDETEKSI ASLI**")
            st.progress(1.0 - prob_fake)
            st.metric("Probabilitas Asli", f"{(1.0 - prob_fake):.1%}")

        st.caption("⚠️ Model ini hanya alat bantu. Selalu verifikasi manual.")

st.divider()
st.caption("Fake Job Detection | Portfolio Data Scientist")