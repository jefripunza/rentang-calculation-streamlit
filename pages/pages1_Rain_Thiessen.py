from pathlib import Path
from typing import Optional
import streamlit as st
import pandas as pd
import numpy as np
from auth import check_auth
from dss_helper import send_to_dss
from menu import render_sidebar

# ========================================
# ===== Autentikasi Basic Auth ===========
# ========================================
if not check_auth():
    st.stop()
render_sidebar()

st.markdown(
    """
    <style>
    /* ==== 読み込み済み CSV を示す緑のボックス ==== */
    .loaded-box{
        border:1px solid #16a34a;
        background:#ecfdf3;
        color:#166534;
        padding:8px 10px;
        margin:6px 0 4px;
        border-radius:6px;
        font-size:0.95rem;
    }
    .loaded-main{
        font-weight:700;
        font-size:1.0rem;
        display:block;
        margin-bottom:2px;
    }
    .loaded-sub{
        font-size:0.9rem;
    }

    /* ==== アップロードが必要なときの赤ボックス ==== */
    .need-upload-box{
        border:1px solid #b91c1c;
        background:#fef2f2;
        color:#7f1d1d;
        padding:10px 12px;
        margin:6px 0 4px;
        border-radius:6px;
        font-size:0.95rem;
    }
    .need-upload-main{
        font-weight:700;
        font-size:1.0rem;
        display:block;
        margin-bottom:2px;
    }
    .need-upload-sub{
        font-size:0.9rem;
    }

    /* 小さめアップローダ用 */
    .small-uploader > div[data-testid="stFileUploader"]{
        padding-top:4px;
        padding-bottom:4px;
    }
    .small-uploader label{
        font-size:0.85rem;
        margin-bottom:2px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def show_loaded_csv(
    kind: str,
    filename: str,
    source: str
):
    """読み込み済み CSV を緑ボックス＋✅付きで表示"""
    st.markdown(
        f"""
        <div class="loaded-box">
          <span class="loaded-main">✅ {kind} loaded: {filename}</span>
          <span class="loaded-sub">{source}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_need_upload(
    label: str,
    detail: str
):
    """アップロードが必要なときだけ表示する赤いボックス"""
    st.markdown(
        f"""
        <div class="need-upload-box">
          <span class="need-upload-main">{label}</span>
          <span class="need-upload-sub">{detail}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ========================================
# Paths: project root, CSV folder, IMG folder
# ========================================
# __file__ = pages/1_Rain_Thiessen.py
# parent      → pages
# parent.parent → project_root (Discharge Calculation)
BASE_DIR = Path(__file__).resolve().parent.parent
CSV_DIR = BASE_DIR / "csv"
IMG_DIR = BASE_DIR / "img"

# ★ ローカル判定：プロジェクト直下の .local_write があれば「ローカル環境」
IS_LOCAL = (BASE_DIR / ".local_write").exists()

rain_path = CSV_DIR / "RainData_wide.csv"
thiessen_path = CSV_DIR / "Thiessen.csv"

# ---------- 共通：罫線付きテーブル描画（左上セルも含む・行番号1始まり） ----------
def render_table(
    df: pd.DataFrame,
    header_bg: str = "#e5f0ff",
    caption: Optional[str] = None,
    header_font_size: str = "12px",
    header_min_width: Optional[str] = None,
    header_max_width: Optional[str] = None,
    header_white_space: str = "nowrap",
    number_format: Optional[str] = None,    # 数値書式（任意）
    max_height_px: Optional[int] = None,    # 縦の最大高さ（px, 任意）
    sticky_first_data_row: bool = False, # ★ 追加：最初のデータ行も固定するかどうか
):
    if caption:
        st.caption(caption)
    if df is None or df.empty:
        st.write("(no data)")
        return

    df_show = df.copy()
    # 行番号は 1始まり
    df_show.index = range(1, len(df_show) + 1)

    # ヘッダー用スタイル
    th_props = [
        ("background-color", header_bg),
        ("font-weight", "bold"),
        ("border", "1px solid #999"),
        ("padding", "2px 4px"),
        ("font-size", header_font_size),
        ("text-align", "center"),
    ]
    if header_min_width is not None:
        th_props.append(("min-width", header_min_width))
    if header_max_width is not None:
        th_props.append(("max-width", header_max_width))
    th_props.append(("white-space", header_white_space))

    table_styles = [
        {
            "selector": "table",
            "props": [
                ("border-collapse", "collapse"),
                ("border", "1px solid #999"),
            ],
        },
        {
            "selector": "th.col_heading",
            "props": th_props,
        },
        {
            "selector": "th.row_heading",
            "props": [
                ("background-color", "#f5f5f5"),
                ("font-weight", "bold"),
                ("border", "1px solid #999"),
                ("padding", "2px 4px"),
                ("white-space", "nowrap"),
                ("text-align", "center"),
            ],
        },
        {
            "selector": "th.blank",
            "props": [
                ("border", "1px solid #999"),
                ("padding", "2px 4px"),
            ],
        },
        {
            "selector": "td",
            "props": [
                ("border", "1px solid #999"),
                ("padding", "2px 4px"),
                ("white-space", "nowrap"),
                ("text-align", "center"),
            ],
        },
    ]

    # ★ 1行目（最初のデータ行）も固定したい場合
    if sticky_first_data_row:
        # ヘッダーの直下に張り付くように top を少し下げる
        table_styles.append(
            {
                "selector": "tbody tr:first-child",
                "props": [
                    ("position", "sticky"),
                    ("top", "24px"),          # ヘッダーの高さぶんだけ下げる（必要なら微調整）
                    ("z-index", "1"),
                    ("background-color", "#ffffff"),
                ],
            }
        )

    styler = (
        df_show
        .style
        .set_properties(
            **{
                "text-align": "center",
                "padding": "2px 4px",
                "font-size": "12px",
                "white-space": "nowrap",
            }
        )
        .set_table_styles(table_styles)
    )

    # 数値列だけフォーマット指定がある場合は適用
    if number_format is not None:
        numeric_cols = df_show.select_dtypes(include="number").columns
        if len(numeric_cols) > 0:
            styler = styler.format(number_format, subset=numeric_cols)

    html_table = styler.to_html()

    # 縦スクロール用ラッパー
    wrapper_style = "text-align:left; overflow-x:auto;"
    if max_height_px is not None:
        wrapper_style += f" max-height:{max_height_px}px; overflow-y:auto;"
    else:
        wrapper_style += " overflow-y:auto;"

    html = f"<div style='{wrapper_style}'>{html_table}</div>"
    st.markdown(html, unsafe_allow_html=True)


# ========================================
# Basic settings & CSS (slightly smaller headings)
# ========================================
st.markdown(
    """
    <style>
    h1 {font-size: 1.5rem;}
    h2 {font-size: 1.3rem;}
    h3 {font-size: 1.1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Rainfall Analysis & Effective Rainfall ($Re$)")

st.markdown("""
This page determines the **Effective Rainfall** required for irrigation planning based on observed data.

1.  **Data Loading**
    Loads 5-day rainfall data (`RainData_wide.csv`) and station weights (`Thiessen.csv`).

2.  **Weighted Rainfall Calculation**
    Computes the area average rainfall for each irrigation group (Bank/Golongan) using Thiessen polygon weights.

3.  **Probability Analysis (Ranking)**
    Analyzes the past 5 years of data to rank rainfall events for every 5-day step.

4.  **Effective Rainfall ($Re$) Output**
    Identifies the dependable rainfall (specifically **Rank 2**, representing approx. 80% probability) to calculate $Re$ for Paddy and Palawija crops, generating the `Re_5Day.csv` file for the next steps.
""")

# ========================================
# 1. Load RainData_wide (wide format)
# ========================================
st.header("1. Load RainData (wide format)")

# ローカルPC: rain_path があれば自動読み込み
rain_wide = None
rain_source_label = None
rain_source_detail = None

if rain_path.exists():
    try:
        rain_wide = pd.read_csv(rain_path)
        rain_source_label = rain_path.name
        rain_source_detail = "Loaded automatically from local `csv` folder."
    except Exception as e:
        st.error(f"Failed to read local RainData_wide.csv: {e}")
        rain_wide = None

# まだ読み込めていなければ、アップロードを促す（赤ボックス）
if rain_wide is None:
    show_need_upload(
        "RainData_wide.csv is required.",
        "Please upload a rainfall CSV with columns `Year, 5Day_ID, <Station_ID...>`."
    )

uploaded_rain = st.file_uploader(
    "Upload RainData_wide.csv (will override local file if provided)",
    type=["csv"],
    key="rain_csv"
)

if uploaded_rain is not None:
    try:
        rain_wide = pd.read_csv(uploaded_rain)
        rain_source_label = uploaded_rain.name
        rain_source_detail = "Uploaded manually on Page1."
        st.success("Uploaded RainData_wide.csv is used instead of local file.")
    except Exception as e:
        st.error(f"Failed to read uploaded RainData_wide.csv: {e}")
        rain_wide = None

# 最終チェック
if rain_wide is None:
    st.stop()

# ✅ 読み込み済み情報を強調表示
if rain_source_label is None:
    rain_source_label = "RainData_wide.csv"
    rain_source_detail = "Loaded in current session."

show_loaded_csv("RainData_wide.csv", rain_source_label, rain_source_detail)

# 必須列チェック
required = {"Year", "5Day_ID"}
if not required.issubset(rain_wide.columns):
    st.error(f"Missing columns: {required}\nFound columns: {set(rain_wide.columns)}")
    st.stop()

# 5Day_ID → Month, MonthNo 付与（ここ以降は従来どおり）
season_index = (rain_wide["5Day_ID"] - 1) // 6  # 0..71 -> 0..11
rain_wide["Month"]   = ((season_index + 10) % 12) + 1  # +10 → start at 11 (Nov)
rain_wide["MonthNo"] = ((rain_wide["5Day_ID"] - 1) % 6) + 1

station_cols = [c for c in rain_wide.columns if c not in ["Year", "Month", "MonthNo", "5Day_ID"]]
rain_wide = rain_wide[["Year", "Month", "MonthNo", "5Day_ID"] + station_cols]


# ========================================
# 2. Load Thiessen weights
# ========================================
st.header("2. Load Thiessen weights")

th = None
thi_source_label  = None
thi_source_detail = None

# ローカル CSV
if thiessen_path.exists():
    try:
        th = pd.read_csv(thiessen_path)
        thi_source_label  = thiessen_path.name
        thi_source_detail = "Loaded automatically from local `csv` folder."
    except Exception as e:
        st.error(f"Failed to read local Thiessen.csv: {e}")
        th = None

# 無ければ赤ボックスで促す
if th is None:
    show_need_upload(
        "Thiessen.csv is required.",
        "Please upload a Thiessen weight CSV with columns `Bank, Golongan/Prop_gol, ID, Nama_Stasi, Ratio`."
    )

uploaded_th = st.file_uploader(
    "Upload Thiessen.csv (will override local file if provided)",
    type=["csv"],
    key="thiessen_csv"
)

if uploaded_th is not None:
    try:
        th = pd.read_csv(uploaded_th)
        thi_source_label  = uploaded_th.name
        thi_source_detail = "Uploaded manually on Page1."
        st.success("Uploaded Thiessen.csv is used instead of local file.")
    except Exception as e:
        st.error(f"Failed to read uploaded Thiessen.csv: {e}")
        th = None

if th is None:
    st.stop()

# ✅ 読み込み済み情報を表示
if thi_source_label is None:
    thi_source_label  = "Thiessen.csv"
    thi_source_detail = "Loaded in current session."
show_loaded_csv("Thiessen.csv", thi_source_label, thi_source_detail)

# ここから先は既存のクリーニング処理をそのまま維持
# Column name variations
if "Nama_Stasi" not in th.columns and "Nama_Stati" in th.columns:
    th = th.rename(columns={"Nama_Stati": "Nama_Stasi"})
if "Golongan" not in th.columns and "Prop_gol" in th.columns:
    th = th.rename(columns={"Prop_gol": "Golongan"})

required_th = {"Bank", "Golongan", "ID", "Ratio"}
if not required_th.issubset(th.columns):
    st.error(f"Missing columns: {required_th}\nFound: {set(th.columns)}")
    st.stop()

# Clean
th["ID"]       = th["ID"].astype(str).str.strip()
th["Bank"]     = th["Bank"].astype(str).str.strip()
th["Golongan"] = th["Golongan"].astype(str).str.strip()


# ===== Correct Ratio → weight conversion =====
ratio_str = (
    th["Ratio"].astype(str)
    .str.replace("%", "", regex=False)
    .str.strip()
)
th["ratio_raw"] = pd.to_numeric(ratio_str, errors="coerce")
# Always treat as percent: 0.3% → 0.003, 12.8% → 0.128
th["weight"] = th["ratio_raw"] / 100.0

# Group name = Bank_Golongan
th["group"] = th["Bank"] + "_" + th["Golongan"]

# Mapping ID → Nama
id_to_name = th.drop_duplicates(subset="ID").set_index("ID")["Nama_Stasi"].to_dict()

# ========================================
# Station Filter (multiselect)
# ========================================
st.subheader("Station Filter")

station_display = {sid: f"{sid}_{id_to_name.get(str(sid),'')}" for sid in station_cols}
station_labels = list(station_display.values())

years = sorted(rain_wide["Year"].unique())
selected_year = st.selectbox("Select Year (for table view):", years, index=0)
rain_year = rain_wide[rain_wide["Year"] == selected_year].copy()

# Init session state
if "station_selected" not in st.session_state:
    st.session_state["station_selected"] = station_labels[:]

col_all, col_none = st.columns([1, 1])
with col_all:
    if st.button("Select All Stations"):
        st.session_state["station_selected"] = station_labels[:]
with col_none:
    if st.button("Clear All Stations"):
        st.session_state["station_selected"] = []

selected_station_labels = st.multiselect(
    "Stations",
    options=station_labels,
    key="station_selected"
)

selected_station_ids = [
    sid for sid in station_cols
    if station_display[sid] in st.session_state["station_selected"]
]
if not selected_station_ids:
    selected_station_ids = station_cols

rename_map = {sid: station_display[sid] for sid in selected_station_ids}
rain_display = rain_year[
    ["Year", "Month", "MonthNo", "5Day_ID"] + selected_station_ids
].rename(columns=rename_map)

st.subheader("Filtered RainData")

# ▼ 最大化フラグ（初期値 False）
if "rain_table_maximized" not in st.session_state:
    st.session_state["rain_table_maximized"] = False

# ▼ ボタンを右側に配置
col_rt_lbl, col_rt_btn = st.columns([4, 1])
with col_rt_btn:
    if st.button(
        "Maximize" if not st.session_state["rain_table_maximized"] else "Restore",
        key="btn_rain_table_toggle"
    ):
        st.session_state["rain_table_maximized"] = not st.session_state["rain_table_maximized"]

# ▼ 最大化時は高さ制限なし、通常時は約15行分（例：480px）に制限
if st.session_state["rain_table_maximized"]:
    max_h = None
else:
    max_h = 480  # 行高×15行くらいのイメージ

# ▼ 整数表示用に数値列を Int64 に揃える
rain_display_int = rain_display.copy()
num_cols = rain_display_int.select_dtypes(include="number").columns
if len(num_cols) > 0:
    rain_display_int[num_cols] = (
        rain_display_int[num_cols].round(0).astype("Int64")
    )

# ▼ 共通レンダラで表示（ヘッダー固定＋罫線＋縦スクロール）
render_table(
    rain_display_int,
    header_bg="#e0f2ff",
    caption=f"RainData_wide.csv – Year {selected_year}",
    header_font_size="11px",
    header_min_width="50px",
    header_max_width="80px",
    header_white_space="normal",   # ヘッダー折り返し許可
    number_format="{:.0f}",        # 数値列は整数表示
    max_height_px=max_h,           # 縦スクロール or フル表示
)


# ========================================
# 2. Thiessen Weight Matrix
# ========================================
st.header("2. Thiessen Weight Matrix")

# Use original Ratio (weight) for display
thiessen_matrix = th.pivot_table(
    index=["ID", "Nama_Stasi"],
    columns="group",
    values="weight",           # 0–1
    aggfunc="sum"
).fillna(0.0)

group_cols = list(thiessen_matrix.columns)

# Convert 0–1 to %
matrix_pct = thiessen_matrix * 100.0
totals = (thiessen_matrix.sum(axis=0) * 100.0).round(0).astype(int)

matrix_disp = matrix_pct.round(1).reset_index()
total_row = {"ID": "TOTAL", "Nama_Stasi": "Total"}
for g in group_cols:
    total_row[g] = totals[g]
matrix_disp = pd.concat([matrix_disp, pd.DataFrame([total_row])], ignore_index=True)

# Build display dataframe (0% → blank, total row = integer %, others = 1 decimal)
display_df = matrix_disp.copy()
for g in group_cols:
    display_df[g] = ""
    nonzero = matrix_disp[g] != 0
    total = matrix_disp["ID"] == "TOTAL"

    display_df.loc[nonzero & ~total, g] = matrix_disp.loc[nonzero & ~total, g].map(
        lambda v: f"{v:.1f}%"
    )
    display_df.loc[nonzero & total, g] = matrix_disp.loc[nonzero & total, g].map(
        lambda v: f"{int(v)}%"
    )

# ---- layout: left = table, right = images ----
col_table, col_img = st.columns([2, 1])

with col_table:
    st.subheader("Thiessen Weights (%)")
    render_table(
        display_df,
        header_bg="#e5f0ff",
        header_font_size="11px",
        header_min_width="40px",
        header_max_width="70px",
        header_white_space="normal",  # 折り返し許可
    )

with col_img:
    map_path = IMG_DIR / "ThiessenMap.png"
    ratio_path = IMG_DIR / "ThiessenRatioMap.png"

    # Thiessen Map（thumbnail）
    if map_path.exists():
        st.image(str(map_path), caption="Thiessen Map", width=350)
        if st.button("Enlarge Thiessen Map", key="btn_map_enlarge"):
            st.session_state["show_map_full"] = True
    else:
        st.info(f"No image file found at: {map_path}")

    # Thiessen Ratio Map（thumbnail）
    if ratio_path.exists():
        st.image(str(ratio_path), caption="Thiessen Ratio Map", width=350)
        if st.button("Enlarge Thiessen Ratio Map", key="btn_ratio_enlarge"):
            st.session_state["show_ratio_full"] = True
    else:
        st.info(f"No image file found at: {ratio_path}")

# ========================================
# Full-size image display
# ========================================
map_path_full = IMG_DIR / "ThiessenMap.png"
ratio_path_full = IMG_DIR / "ThiessenRatioMap.png"

if st.session_state.get("show_map_full", False) and map_path_full.exists():
    st.subheader("Thiessen Map (Full size)")
    st.image(str(map_path_full))
    if st.button("Close Thiessen Map", key="btn_map_close"):
        st.session_state["show_map_full"] = False

if st.session_state.get("show_ratio_full", False) and ratio_path_full.exists():
    st.subheader("Thiessen Ratio Map (Full size)")
    st.image(str(ratio_path_full))
    if st.button("Close Thiessen Ratio Map", key="btn_ratio_close"):
        st.session_state["show_ratio_full"] = False

# ========================================
# 3. Compute weighted rainfall (using weight)
# ========================================
# ※ 計算は全期間分で実施（Year フィルタしない）
id_vars = ["Year", "5Day_ID"]
sta_for_calc = [sid for sid in station_cols if sid in selected_station_ids]

rain_long = rain_wide.melt(
    id_vars=id_vars,
    value_vars=sta_for_calc,
    var_name="STA",
    value_name="Rain"
)
rain_long["Year"] = rain_long["Year"].astype(int)
rain_long["5Day_ID"] = rain_long["5Day_ID"].astype(int)
rain_long["STA"] = rain_long["STA"].astype(str).str.strip()
rain_long["Rain"] = pd.to_numeric(rain_long["Rain"], errors="coerce").fillna(0.0)

th_join = th[["ID", "group", "weight"]].copy()
th_join["ID"] = th_join["ID"].astype(str).str.strip()

merged = pd.merge(
    rain_long, th_join,
    left_on="STA", right_on="ID",
    how="inner"
)
merged["Rain_weighted"] = merged["Rain"] * merged["weight"]

group_5day = (
    merged
    .groupby(["Year", "5Day_ID", "group"], as_index=False)["Rain_weighted"]
    .sum()
    .rename(columns={"Rain_weighted": "Rain_mm"})
)

# 5Day_ID → Month(1〜12, Nov-start) and MonthNo(1〜6) again
season_index_g = (group_5day["5Day_ID"] - 1) // 6
group_5day["Month"] = ((season_index_g + 10) % 12) + 1
group_5day["MonthNo"] = ((group_5day["5Day_ID"] - 1) % 6) + 1

group_5day = group_5day.sort_values(["Year", "Month", "MonthNo", "group"])

# ========================================
# 4. Group × 5-day Rainfall (Data Bar Table, November-start water year)
# ========================================
st.header("3. Group × 5-Day Rainfall (Data Bar Table)")

years_pivot = sorted(group_5day["Year"].unique())
pivot_year = st.selectbox("Select Year for 5-day table (water year):", years_pivot, index=0)

subset = group_5day[group_5day["Year"] == pivot_year].copy()

months_available = sorted(subset["Month"].unique())
water_month_order = [11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
months_ordered = [m for m in water_month_order if m in months_available]
month_labels = [str(m) for m in months_ordered]

if "month_selected" not in st.session_state:
    st.session_state["month_selected"] = month_labels[:]

col_m_all, col_m_none = st.columns([1, 1])
with col_m_all:
    if st.button("Select All Months"):
        st.session_state["month_selected"] = month_labels[:]
with col_m_none:
    if st.button("Clear All Months"):
        st.session_state["month_selected"] = []

selected_month_labels = st.multiselect(
    "Months (water year, starting from November)",
    options=month_labels,
    key="month_selected"
)

if st.session_state["month_selected"]:
    month_filter = [int(m) for m in st.session_state["month_selected"]]
else:
    month_filter = months_ordered

subset = subset[subset["Month"].isin(month_filter)].copy()

table = subset.pivot_table(
    index="group",
    columns=["Year", "Month", "MonthNo"],
    values="Rain_mm"
).round(1)

cols = list(table.columns)

def col_sort_key(c):
    year, month, monthno = c
    try:
        m_idx = water_month_order.index(month)
    except ValueError:
        m_idx = 99
    return (year, m_idx, monthno)

cols_sorted = sorted(cols, key=col_sort_key)
table = table[cols_sorted]

table = table.reset_index()
num_cols = table.columns[1:]

styled = (
    table
    .style
    .format("{:.1f}", subset=num_cols)
    .bar(subset=num_cols, axis=None, color="#99ccff")
    .set_properties(subset=pd.IndexSlice[:, num_cols], **{
        "padding": "2px 4px",
        "line-height": "1.0"
    })
    .set_properties(subset=pd.IndexSlice[:, ["group"]], **{
        "min-width": "150px",
        "padding": "2px 6px"
    })
    .set_table_styles([
        {"selector": "th.col_heading",
         "props": [("background-color", "#e5f0ff"),
                   ("font-weight", "bold"),
                   ("border", "1px solid #999"),
                   ("padding", "2px 4px"),
                   ("white-space", "nowrap"),
                   ("position", "sticky"),
                   ("top", "0"),
                   ("z-index", "2")]},
        {"selector": "th.row_heading",
         "props": [("background-color", "#f5f5f5"),
                   ("font-weight", "bold"),
                   ("border", "1px solid #999"),
                   ("padding", "2px 4px"),
                   ("white-space", "nowrap")]},
        {"selector": "th.blank",
         "props": [("border", "1px solid #999"),
                   ("padding", "2px 4px")]},
        {"selector": "td",
         "props": [("border", "1px solid #999"),
                   ("padding", "2px 4px"),
                   ("white-space", "nowrap")]},
    ])
)

try:
    styled = styled.hide(axis="index")
except Exception:
    styled = styled.hide_index()

html_table = styled.to_html()
html_table = (
    '<div style="overflow-x:auto; text-align:left; font-size:13px;">'
    f'{html_table}'
    '</div>'
)
st.markdown(html_table, unsafe_allow_html=True)

# ========================================
# 5. Yearly Rainfall Summary per Group (water year, 5-day steps)
#    & 6. Ranked 5-day rainfall per group + 7. Re paddy / Re palawija
# ========================================

st.header("4. Yearly Rainfall Summary per Group (water year, 5-day steps)")

years_all = sorted(group_5day["Year"].unique())
last5_years = years_all[-5:] if len(years_all) > 5 else years_all

summary = group_5day[group_5day["Year"].isin(last5_years)].copy()
summary = summary.sort_values(["group", "Year", "5Day_ID"])

rank_df = summary.copy()
rank_df["Rank"] = (
    rank_df
    .groupby(["group", "5Day_ID"])["Rain_mm"]
    .rank(method="first", ascending=True)
    .astype(int)
)

days_in_month = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
                 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}

def make_month_step_and_days(id_index: pd.Index):
    ids = id_index.to_numpy()
    season_idx = (ids - 1) // 6
    months = ((season_idx + 10) % 12) + 1
    steps = ((ids - 1) % 6) + 1

    col_multi = pd.MultiIndex.from_arrays(
        [months, steps],
        names=["Month", "Step"]
    )

    day_counts = []
    for m, s in zip(months, steps):
        if s <= 5:
            day_counts.append(5)
        else:
            day_counts.append(days_in_month[m] - 25)
    return col_multi, pd.Series(day_counts, index=col_multi)

# ---------- 4. Yearly Rainfall Summary per Group ----------

year_table = summary.pivot_table(
    index=["group", "Year"],
    columns="5Day_ID",
    values="Rain_mm"
).round(1)

col_multi, day_series = make_month_step_and_days(year_table.columns)
year_table.columns = col_multi

rank_matrix = rank_df.pivot_table(
    index=["group", "Year"],
    columns="5Day_ID",
    values="Rank"
)
rank_matrix.columns = col_multi

year_table_disp = year_table.where(pd.notna(year_table), "")

days_index = pd.MultiIndex.from_tuples(
    list(year_table_disp.index) + [("Days", "")],
    names=year_table_disp.index.names
)
year_table_disp = year_table_disp.reindex(days_index)
year_table_disp.loc[("Days", "")] = day_series

def style_year_table(data: pd.DataFrame) -> pd.DataFrame:
    styles = pd.DataFrame("", index=data.index, columns=data.columns)
    ranks = rank_matrix.reindex(index=data.index, columns=data.columns)
    styles = styles.mask(ranks == 2, "background-color: #fff2b2;")
    styles = styles.mask(ranks == 3, "background-color: #c2f0c2;")

    prev_group = None
    for idx in data.index:
        g = idx[0]
        if g == "Days":
            styles.loc[idx, :] = styles.loc[idx, :].astype(str) + \
                "border-top: 2px solid #000; border-bottom: 2px solid #000;"
            continue
        if prev_group is None or g != prev_group:
            styles.loc[idx, :] = styles.loc[idx, :].astype(str) + "border-top: 2px solid #000;"
        prev_group = g
    return styles

styled_year = (
    year_table_disp
    .style
    .format("{:.1f}")
    .apply(style_year_table, axis=None)
    .set_properties(**{
        "padding": "2px 4px",
        "line-height": "1.0",
        "font-size": "13px"
    })
    .set_table_styles([
        {"selector": "th.col_heading",
         "props": [("background-color", "#e5f0ff"),
                   ("font-weight", "bold"),
                   ("border", "1px solid #999"),
                   ("padding", "2px 4px"),
                   ("white-space", "nowrap"),
                   ("position", "sticky"),
                   ("top", "0"),
                   ("z-index", "2")]},
        {"selector": "th.row_heading",
         "props": [("background-color", "#f5f5f5"),
                   ("font-weight", "bold"),
                   ("border", "1px solid #999"),
                   ("padding", "2px 4px"),
                   ("white-space", "nowrap")]},
        {"selector": "th.blank",
         "props": [("border", "1px solid #999"),
                   ("padding", "2px 4px")]},
        {"selector": "td",
         "props": [("border", "1px solid #999"),
                   ("padding", "2px 4px"),
                   ("white-space", "nowrap")]},
    ])
)

html_year_table = styled_year.to_html()
html_year_table = (
    '<div style="overflow-x:auto; text-align:left;">'
    f'{html_year_table}'
    '</div>'
)
st.markdown(html_year_table, unsafe_allow_html=True)

# ---------- 5. Ranked 5-Day Rainfall per Group (mm/day) ----------
st.header("5. Ranked 5-Day Rainfall per Group (last 5 years, per 5-day ID, mm/day)")

rank_table = rank_df.pivot_table(
    index=["group", "Rank"],
    columns="5Day_ID",
    values="Rain_mm"
)

col_multi_rank, day_series_rank = make_month_step_and_days(rank_table.columns)
rank_table.columns = col_multi_rank

# mm/day に変換
rank_table_perday = rank_table.divide(day_series_rank, axis=1).round(1)
rank_table_disp = rank_table_perday.where(pd.notna(rank_table_perday), "")

days_index_rank = pd.MultiIndex.from_tuples(
    list(rank_table_disp.index) + [("Days", 0)],
    names=rank_table_disp.index.names
)
rank_table_disp = rank_table_disp.reindex(days_index_rank)
rank_table_disp.loc[("Days", 0)] = day_series_rank

def style_rank_table(data: pd.DataFrame) -> pd.DataFrame:
    styles = pd.DataFrame("", index=data.index, columns=data.columns)
    groups_idx = data.index.get_level_values("group")
    ranks_idx = data.index.get_level_values("Rank")

    prev_group = None
    for idx, g, r in zip(data.index, groups_idx, ranks_idx):
        if g == "Days":
            styles.loc[idx, :] = "border-top: 2px solid #000; border-bottom: 2px solid #000;"
            continue
        if r == 2:
            base = "background-color: #fff2b2;"
        elif r == 3:
            base = "background-color: #c2f0c2;"
        else:
            base = ""
        if prev_group is None or g != prev_group:
            styles.loc[idx, :] = base + "border-top: 2px solid #000;"
        else:
            styles.loc[idx, :] = base
        prev_group = g
    return styles

styled_rank = (
    rank_table_disp
    .style
    .format("{:.1f}")
    .apply(style_rank_table, axis=None)
    .set_properties(**{
        "padding": "2px 4px",
        "line-height": "1.0",
        "font-size": "13px"
    })
    .set_table_styles([
        {"selector": "th.col_heading",
         "props": [("background-color", "#e5f0ff"),
                   ("font-weight", "bold"),
                   ("border", "1px solid #999"),
                   ("padding", "2px 4px"),
                   ("white-space", "nowrap"),
                   ("position", "sticky"),
                   ("top", "0"),
                   ("z-index", "2")]},
        {"selector": "th.row_heading",
         "props": [("background-color", "#f5f5f5"),
                   ("font-weight", "bold"),
                   ("border", "1px solid #999"),
                   ("padding", "2px 4px"),
                   ("white-space", "nowrap")]},
        {"selector": "th.blank",
         "props": [("border", "1px solid #999"),
                   ("padding", "2px 4px")]},
        {"selector": "td",
         "props": [("border", "1px solid #999"),
                   ("padding", "2px 4px"),
                   ("white-space", "nowrap")]},
    ])
)

html_rank = styled_rank.to_html()
html_rank = (
    '<div style="overflow-x:auto; text-align:left;">'
    f'{html_rank}'
    '</div>'
)
st.markdown(html_rank, unsafe_allow_html=True)

# ---------- 6. Effective Rainfall Re (Paddy / Palawija, mm/day) ----------
st.header("6. Effective Rainfall Re (Paddy / Palawija, mm/day)")

# Rank=2 の行から Re_paddy を作成
if "Rank" in rank_table_perday.index.names:
    re_paddy = rank_table_perday.xs(2, level="Rank", drop_level=False)
else:
    mask = [idx[1] == 2 for idx in rank_table_perday.index]
    re_paddy = rank_table_perday[mask]

re_long = None  # 後で存在チェック用

if not re_paddy.empty:
    # --- Re_paddy / Re_palawija のテーブルを作成 ---
    re_paddy_vals = re_paddy.copy()
    re_pal_vals   = re_paddy_vals * 0.5

    groups_idx = re_paddy_vals.index.get_level_values("group")
    re_paddy_vals.index = groups_idx
    re_pal_vals.index   = groups_idx

    rows = []
    idx  = []
    for g in groups_idx.unique():
        idx.append((g, "Re_paddy"))
        rows.append(re_paddy_vals.loc[g])
        idx.append((g, "Re_palawija"))
        rows.append(re_pal_vals.loc[g])

    idx_mi = pd.MultiIndex.from_tuples(idx, names=["group", "Crop"])
    data   = pd.DataFrame(rows, index=idx_mi).round(1)

    # 他ページ用に session_state に保存
    st.session_state["re_table"]    = data
    st.session_state["re_paddy"]    = data.xs("Re_paddy",    level="Crop")
    st.session_state["re_palawija"] = data.xs("Re_palawija", level="Crop")

    # ========= Re_5Day 用の long 形式 (group,Crop,Month,Step,Re) を作成 =========
    rows_long = []
    for (group, crop), row in data.iterrows():
        for (month, step), val in row.items():
            rows_long.append({
                "group": group,
                "Crop":  crop,
                "Month": int(month),
                "Step":  int(step),
                "Re":    float(val),
            })

    re_long = pd.DataFrame(rows_long)

    # ---- 表示（罫線＋データバー）----
    styled_re = (
        data
        .style
        .format("{:.1f}")
        .bar(axis=None, color="#4d88ff")
        .set_properties(**{
            "padding": "2px 4px",
            "line-height": "1.0",
            "font-size": "13px",
        })
        .set_table_styles([
            {"selector": "th.col_heading",
             "props": [("background-color", "#e5f0ff"),
                       ("font-weight", "bold"),
                       ("border", "1px solid #999"),
                       ("padding", "2px 4px"),
                       ("white-space", "nowrap"),
                       ("position", "sticky"),
                       ("top", "0"),
                       ("z-index", "2")]},
            {"selector": "th.row_heading",
             "props": [("background-color", "#f5f5f5"),
                       ("font-weight", "bold"),
                       ("border", "1px solid #999"),
                       ("padding", "2px 4px"),
                       ("white-space", "nowrap")]},
            {"selector": "th.blank",
             "props": [("border", "1px solid #999"),
                       ("padding", "2px 4px")]},
            {"selector": "td",
             "props": [("border", "1px solid #999"),
                       ("padding", "2px 4px"),
                       ("white-space", "nowrap")]},
        ])
    )

    html_re = (
        '<div style="overflow-x:auto; text-align:left;">'
        f'{styled_re.to_html()}'
        '</div>'
    )
    st.markdown(html_re, unsafe_allow_html=True)

else:
    st.info("No Rank=2 rows found to build Re paddy / Re palawija table.")


# ========= Re_5Day.csv の書き出し／ダウンロード =========
if re_long is not None:
    # ---- ローカル環境だけ csv/ に書き出し ----
    if IS_LOCAL:
        re_csv_path = CSV_DIR / "Re_5Day.csv"
        try:
            re_long.to_csv(re_csv_path, index=False)
            st.success(f"Re 5-day table has been saved to CSV: `{re_csv_path.name}`")
        except Exception as e:
            st.error(f"Failed to save Re CSV: {e}")
    else:
        # Cloud / 共有環境ではディスクに書かない
        st.info(
            "Cloud / shared environment: `Re_5Day.csv` is **not** written to disk. "
            "Use the download button below to save it locally."
        )

    # ---- ローカル／Cloud 共通：ダウンロードボタン ----
    # st.download_button(
    #     "Download Re 5-day CSV (Re in mm/day)",
    #     re_long.to_csv(index=False).encode("utf-8-sig"),
    #     file_name="Re_5Day.csv",
    #     mime="text/csv",
    # )

    send_to_dss("re", re_long, "Send to DSS: Re 5-day")
