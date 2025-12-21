import streamlit as st
from util import local_css
import requests

PASS = "rentang@uyee"

def check_auth():
    # cek apakah sudah login
    if st.session_state.get("logged_in", False):
        return True

    # -------------------------------- #
    # tampilkan form login kalau belum #
    # -------------------------------- #
    st.title("🔐 Aplikasi Perencanaan Irigasi")
    st.write("### Silakan Masukkan Password untuk melanjutkan")
    password = st.text_input("Password", type="password", key="auth_password")
    if st.button("Login"):
        # nembah DSS
        if password == PASS:
            st.session_state["logged_in"] = True
            st.rerun()
        else:
            st.error("❌ Password salah")
    local_css("style.css")
    return st.session_state.get("logged_in", False)
