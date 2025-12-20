
import streamlit as st

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
