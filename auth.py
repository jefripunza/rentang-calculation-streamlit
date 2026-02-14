import streamlit as st
from util import local_css
import requests

def check_auth():
    local_css("style.css")
    # cek apakah sudah login
    is_logged_in = st.session_state.get("logged_in", False)
    if not is_logged_in:
        # -------------------------------- #
        # tampilkan form login kalau belum #
        # -------------------------------- #
        st.title("🔐 Aplikasi Perencanaan Irigasi")
        st.write("### Silakan Masukkan Password untuk melanjutkan")
        default_pass = st.query_params.get("pass", "")
        password = st.text_input("Password", type="password", key="auth_password", value=default_pass)
        if st.button("Login"):
            # nembah DSS
            if password == PASS:
                st.session_state["logged_in"] = True
                st.rerun()
            else:
                st.error("❌ Password salah")
        return False
    return True
