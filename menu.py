
import streamlit as st

def render_sidebar():
    st.sidebar.header("📌 Menu Utama")

    # Home / Introduction
    st.sidebar.page_link(
        "app.py",
        label="🏠 Introduction"
    )

    # Pages dari folder pages/
    st.sidebar.markdown("### 📊 Analisis & Kalkulasi")
    st.sidebar.page_link(
        "pages/pages1_Rain_Thiessen.py",
        label="1. Rain & Thiessen"
    )
    st.sidebar.page_link(
        "pages/pages2_NFR_Calculation.py",
        label="2. NFR Calculation"
    )
    st.sidebar.page_link(
        "pages/pages3_BranchArea_Calc.py",
        label="3. Branch Area Calc"
    )
    st.sidebar.page_link(
        "pages/pages4_Water_Requirement_Calculation.py",
        label="4. Water Requirement Calc"
    )
    st.sidebar.page_link(
        "pages/pages5_Qp_Adjustment.py",
        label="5. Qp Adjustment"
    )
    st.sidebar.page_link(
        "pages/pages6_Print_Report.py",
        label="6. Print Report"
    )
