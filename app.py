import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ====================== CONFIGURATION ======================
st.set_page_config(
    page_title="Fake Job Detector",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Load model
@st.cache_resource
def load_model():
    try:
        # Load preprocessor
        preprocessor = joblib.load('preprocessor.pkl')
        
        # Load XGBoost model (JSON format - lebih stabil)
        booster = xgb.Booster()
        booster.load_model('xgb_model.json')
        
        st.success("✅ Model berhasil dimuat (XGBoost 3.2.0)")
        return preprocessor, booster
    except Exception as e:
        st.error(f"Gagal memuat model: {str(e)}")
        st.error("Pastikan preprocessor.pkl dan xgb_model.json ada di root folder")
        st.stop()

# Load model sekali di awal
preprocessor, booster = load_model()

# ====================== TITLE & DESCRIPTION ======================
st.title("Fake Job Posting Detector")
st.markdown("""
**Deteksi Lowongan Kerja Palsu**  
Aplikasi ini menggunakan **XGBoost + TF-IDF** untuk memprediksi apakah sebuah lowongan kerja **asli** atau **palsu** (scam).  

Masukkan detail lowongan di bawah ini, lalu klik **Prediksi**.
""")

st.divider()

# ====================== INPUT FORM ======================
col1, col2 = st.columns(2)

with col1:
    title = st.text_input("Job Title", placeholder="Senior Python Developer")
    location = st.text_input("Location", placeholder="Jakarta, Indonesia / Remote")
    employment_type = st.selectbox(
        "Employment Type",
        ["Full-time", "Part-time", "Contract", "Freelance", "Internship", "Unknown"]
    )
    required_experience = st.selectbox(
        "Required Experience",
        ["Entry level", "Mid level", "Senior level", "Executive", "Unknown"]
    )

with col2:
    department = st.text_input("Department", placeholder="Engineering")
    required_education = st.text_input("Required Education", placeholder="Bachelor's Degree")
    industry = st.text_input("Industry", placeholder="Information Technology")
    has_company_logo = st.radio("Has Company Logo?", ["Yes", "No"], horizontal=True)
    has_questions = st.radio("Has Screening Questions?", ["Yes", "No"], horizontal=True)

# Text area yang lebih besar
st.subheader("Company Profile")
company_profile = st.text_area("Company Profile", height=120, placeholder="Kami adalah perusahaan teknologi yang sedang berkembang pesat...")

st.subheader("Job Description")
description = st.text_area("Job Description", height=150, placeholder="Kami mencari kandidat yang bertanggung jawab untuk...")

st.subheader("Requirements")
requirements = st.text_area("Requirements", height=120, placeholder="Minimal 2 tahun pengalaman di Python...")

st.subheader("Benefits")
benefits = st.text_area("Benefits", height=100, placeholder="Gaji kompetitif, asuransi kesehatan, WFH...")

# Tombol contoh
if st.button("Gunakan Contoh Lowongan Palsu"):
    title = "Work From Home - Earn $5000/month"
    location = "Remote"
    employment_type = "Full-time"
    required_experience = "Entry level"
    department = "Marketing"
    required_education = "High School"
    industry = "Marketing"
    has_company_logo = "No"
    has_questions = "No"
    company_profile = "We are a fast growing international company."
    description = "No experience required. Start immediately and get paid weekly. Just send your CV!"
    requirements = "Only need a laptop and internet connection."
    benefits = "Flexible hours, high salary, work from anywhere."

# ====================== PREDICTION ======================
if st.button("Prediksi Sekarang", type="primary", use_container_width=True):
    if not description.strip():
        st.warning("Mohon isi Job Description terlebih dahulu.")
    else:
        # Siapkan data input
        input_data = pd.DataFrame([{
            'title': title,
            'location': location,
            'department': department,
            'company_profile': company_profile,
            'description': description,
            'requirements': requirements,
            'benefits': benefits,
            'telecommuting': 1 if "remote" in location.lower() else 0,
            'has_company_logo': 1 if has_company_logo == "Yes" else 0,
            'has_questions': 1 if has_questions == "Yes" else 0,
            'employment_type': employment_type,
            'required_experience': required_experience,
            'required_education': required_education,
            'industry': industry,
            'function': department,  # approximate
            'text_combined': f"{title} {company_profile} {description} {requirements} {benefits}"
        }])

        X_transformed = preprocessor.transform(input_data)

        # Prediksi dengan XGBoost booster
        dmatrix = xgb.DMatrix(X_transformed)
        prob_fake = float(booster.predict(dmatrix)[0])
        prob_fake = max(0.0, min(1.0, prob_fake))

        prediction = 1 if prob_fake > 0.5 else 0
        
        # Hasil
        if prediction == 1:
            st.error(f"**LOWONGAN INI DIDUGA PALSU**")
            st.progress(probability)
            st.metric("Probabilitas Scam", f"{probability:.1%}")
        else:
            st.success(f"**LOWONGAN INI TERDETEKSI ASLI**")
            st.progress(1.0 - probability)
            st.metric("Probabilitas Asli", f"{1.0-probability:.1%}")

        st.caption("Catatan: Model ini bukan 100% akurat. Gunakan sebagai alat bantu saja.")

# ====================== FOOTER ======================
st.divider()
st.markdown("""
**Tips:**  
- Lowongan palsu sering menggunakan kata-kata seperti “no experience required”, “start immediately”, “earn money fast”, “work from home easily”.  
- Semakin lengkap informasi yang kamu masukkan, semakin akurat prediksi.
""")