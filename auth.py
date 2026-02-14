import streamlit as st
from util import local_css
import requests
from environment import host_dss_url

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
            try:
                response = requests.get(host_dss_url + "/api/process/session/" + password)
                if response.status_code == 200:
                    resp = response.json()
                    # response: data.is_connected
                    if resp.get("data", {}).get("is_connected", False):
                        st.session_state["logged_in"] = True
                        st.rerun()
                    else:
                        st.error("❌ Password salah")
                else:
                    st.error("❌ Password salah")
            except Exception as e:
                st.error(f"❌ Gagal menghubungi server: {e}")
        return False
    return True
