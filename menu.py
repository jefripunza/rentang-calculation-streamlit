
import streamlit as st

# 1. Load CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# 2. Pastikan CSS berada di folder yang sama
local_css("cyberpunk.css")

def render_sidebar():
    st.sidebar.header("📌 Main Menu")

    # Home / Introduction
    st.sidebar.page_link(
        "app.py",
        label="🏠 Introduction"
    )

    # Pages dari folder pages/
    st.sidebar.markdown("### 📊 Analysis & Calculation")
    st.sidebar.page_link(
        "pages/pages1_Rain_Thiessen.py",
        label="1️⃣ Rain & Thiessen"
    )
    st.sidebar.page_link(
        "pages/pages2_NFR_Calculation.py",
        label="2️⃣ NFR Calculation"
    )
    st.sidebar.page_link(
        "pages/pages3_BranchArea_Calc.py",
        label="3️⃣ Branch Area Calc"
    )
    st.sidebar.page_link(
        "pages/pages4_Water_Requirement_Calculation.py",
        label="4️⃣ Water Requirement Calc"
    )
    st.sidebar.page_link(
        "pages/pages5_Qp_Adjustment.py",
        label="5️⃣ Qp Adjustment"
    )
    st.sidebar.page_link(
        "pages/pages6_Print_Report.py",
        label="6️⃣ Print Report"
    )
