import streamlit as st

PASS = "rentang@uyee"

def check_auth():
    # cek apakah sudah login
    if st.session_state.get("logged_in", False):
        return True

    # ---------------------------------
    # tampilkan form login kalau belum
    # ---------------------------------
    st.title("🔐 Aplikasi Perencanaan Irigasi")
    st.write("### Silakan Masukkan Password untuk melanjutkan")

    password = st.text_input("Password", type="password", key="auth_password")

    if st.button("Login"):
        if password == PASS:
            st.session_state["logged_in"] = True
            st.rerun()
        else:
            st.error("❌ Password salah")
    return st.session_state.get("logged_in", False)
