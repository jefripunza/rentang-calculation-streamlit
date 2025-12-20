import datetime as dt
from pathlib import Path

import pandas as pd
import streamlit as st
from auth import check_auth
from menu import render_sidebar

# ========================================
# ===== Autentikasi Basic Auth ===========
# ========================================
if not check_auth():
    st.stop()
render_sidebar()

# ========================================
# Paths: project root, CSV folder, IMG folder, AUDIO folder
# ========================================
BASE_DIR = Path(__file__).resolve().parent

CSV_DIR = BASE_DIR / "csv"
CSV_DIR.mkdir(exist_ok=True)  # For local write (treated as temporary folder in cloud)

IMG_DIR = BASE_DIR / "img"    # Folder to place General_Qp_Calculation.png
AUDIO_DIR = BASE_DIR / "audio" # [Added] Folder for audio files (create if necessary)

st.set_page_config(page_title="Irrigation Tool", layout="wide")

# ---------- Common: table renderer with borders & 1-based row index ----------
# def render_table(df: pd.DataFrame, header_bg: str = "#e5f0ff", caption: str | None = None):
#     if caption:
#         st.caption(caption)
#     if df is None or df.empty:
#         st.write("(no data)")
#         return

#     df_show = df.copy()
#     df_show.index = range(1, len(df_show) + 1)  # 1-based index

#     styler = (
#         df_show
#         .style
#         .set_properties(**{
#             "text-align": "center",
#             "padding": "2px 4px",
#             "font-size": "12px",
#             "white-space": "nowrap",
#         })
#         .set_table_styles([
#             {"selector": "th.col_heading",
#              "props": [("background-color", header_bg),
#                        ("font-weight", "bold"),
#                        ("border", "1px solid #999"),
#                        ("padding", "2px 4px"),
#                        ("white-space", "nowrap"),
#                        ("position", "sticky"),
#                        ("top", "0"),
#                        ("z-index", "2")]},
#             {"selector": "th.row_heading",
#              "props": [("background-color", "#f5f5f5"),
#                        ("font-weight", "bold"),
#                        ("border", "1px solid #999"),
#                        ("padding", "2px 4px"),
#                        ("white-space", "nowrap")]},
#             {"selector": "th.blank",
#              "props": [("border", "1px solid #999"),
#                        ("padding", "2px 4px")]},
#             {"selector": "td",
#              "props": [("border", "1px solid #999"),
#                        ("padding", "2px 4px"),
#                        ("white-space", "nowrap")]},
#         ])
#     )

#     html = styler.to_html()
#     html = (
#         '<div style="overflow-x:auto; text-align:left;">'
#         f"{html}"
#         "</div>"
#     )
#     st.markdown(html, unsafe_allow_html=True)


# def find_latest_csv(pattern_func):
#     files = [p for p in CSV_DIR.glob("*.csv") if pattern_func(p)]
#     if not files:
#         return None
#     return max(files, key=lambda p: p.stat().st_mtime)


# ========================================
# 1. Overview
# ========================================
st.title("Irrigation Analysis Tool")

# ---------------------------------------------------------
# [Added] Explanation Audio Section
# ---------------------------------------------------------
# * Specify path or URL for audio files
AUDIO_INDO_PATH = "https://raw.githubusercontent.com/rimpkmatsu/public-img/main/img_waterrequiement_public/General%20Naration_Bahasa2.m4a"  # or AUDIO_DIR / "indo.mp3"
AUDIO_JP_PATH = "https://raw.githubusercontent.com/rimpkmatsu/public-img/main/img_waterrequiement_public/General_Naration_JP.m4a"      # or AUDIO_DIR / "jp.mp3"

st.write("### App Explanation / Penjelasan Aplikasi")
col_audio1, col_audio2 = st.columns(2)

with col_audio1:
    st.markdown("**Bahasa Indonesia**")
    # Add error handling (try-except) if you want to show only when file exists
    st.audio(str(AUDIO_INDO_PATH), format="audio/mp3")

with col_audio2:
    st.markdown("**日本語**")
    st.audio(str(AUDIO_JP_PATH), format="audio/mp3")

st.markdown("---")


# ---------------------------------------------------------
# Image Definitions
# ---------------------------------------------------------

# [Added] URL or path for the new image to be added on top
TOP_IMG_URL = (
    "https://raw.githubusercontent.com/rimpkmatsu/public-img/refs/heads/main/img_waterrequiement_public/General_p1-p5.png"
)

# Existing image URL
# Replace with raw URL of your public assets repo
Qp_IMG_URL = (
    "https://raw.githubusercontent.com/rimpkmatsu/public-img/refs/heads/main/img_waterrequiement_public/General_Qp_Calculation.png"
)

st.markdown("""
This tool consists of the following modules:

1. **Rain & Thiessen** Estimate areal rainfall using Thiessen polygons and compute effective rainfall.

2. **NFR Calculation** Calculate crop water requirement and Net Field Water Requirement (NFR).

3. **Branch Area & Canal Filter** Analyze the canal network, accumulate downstream TB areas by branch,
   and view results per main / secondary canal.

4. **Water Requirement (Qp)** Use NFR and canal conveyance efficiencies to compute 5-day Qp for each canal reach.
""")

st.markdown("### Conceptual diagram of Qp calculation flow")

# ---------------------------------------------------------
# [Added] Display the new image at the top
# ---------------------------------------------------------
# use_container_width=True (or use_column_width=True) to fit container width
st.image(TOP_IMG_URL, caption="Overview Diagram", use_container_width=True)

# Spacer between images
st.write("") 

# ---------------------------------------------------------
# Display existing image
# ---------------------------------------------------------
# Note: 'width="stretch"' might not work in some browsers; 'use_container_width=True' is recommended.
st.image(Qp_IMG_URL, caption="Qp Calculation Flow", use_container_width=True)

# Link below the image (opens in new tab)
st.markdown(
    f"[Open Qp calculation diagram in a new tab]({Qp_IMG_URL})",
    unsafe_allow_html=False,
)

st.markdown("---")