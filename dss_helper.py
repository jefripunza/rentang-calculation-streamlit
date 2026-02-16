import streamlit as st
import requests
from environment import host_dss_url


def send_to_dss(context: str, dataframe, button_label: str = "Send to DSS"):
    """
    Reusable function to send dataframe to DSS API with loading state.
    
    Args:
        context: Context name for the API endpoint (e.g., 're', 'nfr')
        dataframe: Pandas DataFrame to send
        button_label: Custom button label (default: "Send to DSS")
    """
    session_id = st.session_state.get("session_id", "")
    send_key = f"sending_{context}"
    is_sending = st.session_state.get(send_key, False)

    if st.button(button_label, disabled=is_sending, key=f"btn_{context}"):
        st.session_state[send_key] = True
        st.rerun()

    if is_sending:
        with st.spinner(f"Mengirim data {context.upper()} ke DSS..."):
            try:
                url = f"{host_dss_url}/api/process/streamlit/{context}/{session_id}"
                payload = dataframe.to_dict(orient="records")
                resp = requests.post(url, json=payload)
                if resp.status_code == 200:
                    st.success(f"✅ Data {context.upper()} berhasil dikirim ke DSS")
                else:
                    st.error(f"❌ Gagal mengirim data: {resp.status_code}")
            except Exception as e:
                st.error(f"❌ Gagal menghubungi server: {e}")
            finally:
                st.session_state[send_key] = False
