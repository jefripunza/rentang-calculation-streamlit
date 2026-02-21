# pages/4_WaterRequirement_Calculation.py

from pathlib import Path
from collections import defaultdict
import datetime as dt
import calendar
import re

import numpy as np
import pandas as pd
import streamlit as st
from auth import check_auth
from menu import render_sidebar

from typing import Optional
# ========================================
# ===== Autentikasi Basic Auth ===========
# ========================================
if not check_auth():
    st.stop()
render_sidebar()

# ---------- 共通：CSV 状態表示用スタイル（読み込み済み / アップロード必要） ----------
st.markdown(
    """
    <style>
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
    </style>
    """,
    unsafe_allow_html=True,
)

def show_loaded_csv(kind: str, label: str, detail: str):
    """読み込み済み CSV を緑ボックス＋✅で表示"""
    st.markdown(
        f"""
        <div class="loaded-box">
          <span class="loaded-main">✅ {kind} loaded: {label}</span>
          <span class="loaded-sub">{detail}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

def show_need_upload(label: str, detail: str):
    """アップロードが必要なときだけ赤ボックスで強調"""
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
# Paths: project root, CSV folder
# ========================================
BASE_DIR = Path(__file__).resolve().parent.parent
CSV_DIR = BASE_DIR / "csv"

# ★ ローカル判定：プロジェクト直下の .local_write があれば「ローカル環境」
IS_LOCAL = (BASE_DIR / ".local_write").exists()

# ---------- 共通：小さめ・罫線付きテーブル描画 ----------
def render_simple_table(df: pd.DataFrame, header_bg: str = "#e5f0ff", caption: Optional[str] = None):
    if caption:
        st.caption(caption)
    if df is None or df.empty:
        st.write("(no data)")
        return

    styler = (
        df.style
        .set_properties(**{
            "text-align": "center",
            "padding": "2px 4px",
            "font-size": "12px",
            "white-space": "nowrap",
        })
        .set_table_styles([
            # 列ヘッダー
            {"selector": "th.col_heading",
             "props": [("background-color", header_bg),
                       ("font-weight", "bold"),
                       ("border", "1px solid #999"),
                       ("padding", "2px 4px"),
                       ("white-space", "nowrap"),
                       ("position", "sticky"),
                       ("top", "0"),
                       ("z-index", "2")]},
            # 行ヘッダー
            {"selector": "th.row_heading",
             "props": [("background-color", "#f5f5f5"),
                       ("font-weight", "bold"),
                       ("border", "1px solid #999"),
                       ("padding", "2px 4px"),
                       ("white-space", "nowrap")]},
            # 左上の空セル
            {"selector": "th.blank",
             "props": [("border", "1px solid #999"),
                       ("padding", "2px 4px")]},
            # データセル
            {"selector": "td",
             "props": [("border", "1px solid #999"),
                       ("padding", "2px 4px"),
                       ("white-space", "nowrap")]},
        ])
    )

    html = styler.to_html()
    html = (
        '<div style="overflow-x:auto; text-align:left;">'
        f'{html}'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


# ---------- 5Day インデックス計算（today 用） ----------
water_month_order = [11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def date_to_5day_index(d: dt.date) -> int:
    """今日の日付から、0-based の 5Day インデックス (0..71) を返す。"""
    m = d.month
    try:
        m_idx = water_month_order.index(m)
    except ValueError:
        return -1

    day = d.day
    if day <= 5:
        step = 1
    elif day <= 10:
        step = 2
    elif day <= 15:
        step = 3
    elif day <= 20:
        step = 4
    elif day <= 25:
        step = 5
    else:
        step = 6

    return m_idx * 6 + (step - 1)  # 0..71


# ========================================
# Title
# ========================================
st.title("Water Requirement Calculation (Qp)")

st.markdown("""
This page:

1. Selects a **network CSV** (files whose name contains `Diagram`) from the `csv` folder.  
2. Selects an **NFR CSV** (files whose name contains `NFR`, e.g. `NFR_5Day.csv`) from the same folder.  
3. Uses **NFR (L/s/ha)** and **Eff (canal conveyance efficiencies)** to compute 5-day Qp for each reach.  
""")

# ========================================
# 1. Select Network CSV (Diagram*)
# ========================================
st.header("1. Select network CSV (Diagram)")

network_candidates = sorted(
    [p for p in CSV_DIR.glob("*.csv") if "diagram" in p.name.lower()],
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)

if not network_candidates:
    st.error(f"No network CSV found in `{CSV_DIR}` whose filename contains 'Diagram'.")
    st.stop()

net_labels = [
    f"{p.name}  (updated: {dt.datetime.fromtimestamp(p.stat().st_mtime).strftime('%Y-%m-%d %H:%M')})"
    for p in network_candidates
]

selected_net_label = st.selectbox(
    "Select network CSV",
    options=net_labels,
    index=0,
)
net_path = network_candidates[net_labels.index(selected_net_label)]

st.info(f"Using network CSV: `{net_path.name}`")

try:
    # pages3 と同じく、2行目がヘッダーなら header=1
    df_net_raw = pd.read_csv(net_path, header=1)
except Exception as e:
    st.error(f"Failed to read network CSV: {e}")
    st.stop()

st.caption(f"(network CSV preview omitted – rows: {len(df_net_raw)})")


# ========================================
# 2. Load NFR CSV (NFR*, unit = L/s/ha)
# ========================================
st.header("2. Load NFR CSV (5-day, NFR in L/s/ha)")

df_nfr_raw = None
nfr_source_label  = None
nfr_source_detail = None

# まず、Page2 で計算済みの NFR が session_state にあればそれを優先
nfr_5day_session = st.session_state.get("nfr_5day", None)
if isinstance(nfr_5day_session, pd.DataFrame):
    df_nfr_raw = nfr_5day_session.copy()
    nfr_source_label  = "session (Page2)"
    nfr_source_detail = "NFR_5Day table calculated on Page2 in this session."

# session に無ければ、csv フォルダから NFR*.csv を探す
if df_nfr_raw is None:
    nfr_candidates = sorted(
        [p for p in CSV_DIR.glob("*.csv") if "nfr" in p.name.lower()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if nfr_candidates:
        nfr_path = nfr_candidates[0]
        try:
            df_nfr_raw = pd.read_csv(nfr_path)
            nfr_source_label  = nfr_path.name
            nfr_source_detail = "Loaded automatically from local `csv` folder (NFR*)."
        except Exception as e:
            st.error(f"Failed to read NFR CSV: {e}")
            df_nfr_raw = None

# ここまでで NFR が無ければ、アップロードを強調して促す
if df_nfr_raw is None:
    show_need_upload(
        "NFR CSV is required.",
        "Please either calculate NFR on Page2 (to create `NFR_5Day.csv`), "
        "or upload an NFR CSV with columns `Bank, Golongan, Group, Season, Month, Step, NFR`."
    )

uploaded_nfr = st.file_uploader(
    "Upload NFR CSV (Bank, Golongan, Group, Season, Month, Step, NFR)",
    type=["csv"],
    key="nfr_csv",
)

if uploaded_nfr is not None:
    try:
        df_nfr_raw = pd.read_csv(uploaded_nfr)
        nfr_source_label  = uploaded_nfr.name
        nfr_source_detail = "Uploaded manually on Page4."
        st.success("Uploaded NFR CSV is used instead of local file/session data.")
    except Exception as e:
        st.error(f"Failed to read uploaded NFR CSV: {e}")
        df_nfr_raw = None

# 最終チェック
if df_nfr_raw is None:
    st.stop()

# ✅ 読み込み済み情報を明示
if nfr_source_label is None:
    nfr_source_label  = "NFR_5Day.csv"
    nfr_source_detail = "Loaded in current session."
show_loaded_csv("NFR 5-day (L/s/ha)", nfr_source_label, nfr_source_detail)

st.caption("(Using NFR 5-day table: Bank, Golongan, Group, Season, Month, Step, NFR)")

# ========================================
#  Prepare Network Data & Graph Structure
# ========================================
# （内部処理のみ。プレビュー表は省略）

required_cols = {
    "ParentKey":   "Parent Key",
    "ChildKey":    "Child Key",
    "ParentName":  "Parent Name",
    "ChildName":   "Child Name",
    "Class":       "Class",
    "Area":        "Area",
    "Golongan_TB": "Golongan_TB",
    "Canal":       "Canal",
    "CanalKey":    "Canal Key",
    "SectionName": "Section",
}

missing = [v for v in required_cols.values() if v not in df_net_raw.columns]
if missing:
    st.error(f"Required columns missing in network CSV: {missing}")
    st.stop()

df = pd.DataFrame({
    "ParentKey":   df_net_raw[required_cols["ParentKey"]],
    "ChildKey":    df_net_raw[required_cols["ChildKey"]],
    "ParentName":  df_net_raw[required_cols["ParentName"]],
    "ChildName":   df_net_raw[required_cols["ChildName"]],
    "Class":       df_net_raw[required_cols["Class"]],
    "Area":        pd.to_numeric(df_net_raw[required_cols["Area"]], errors="coerce").fillna(0.0),
    "Golongan_TB": df_net_raw[required_cols["Golongan_TB"]],
    "Canal":       df_net_raw[required_cols["Canal"]],
    "CanalKey":    df_net_raw[required_cols["CanalKey"]],
    "SectionName": df_net_raw[required_cols["SectionName"]],
})

for col in [
    "ParentKey", "ChildKey", "ParentName", "ChildName",
    "Class", "Golongan_TB", "Canal", "CanalKey", "SectionName"
]:
    df[col] = df[col].astype(str).str.strip()

df["SectionName"] = df["SectionName"].replace("nan", "").fillna("")
df["Bank"] = df["ChildKey"].str[0].map({"L": "Left", "R": "Right"}).fillna("")
df["BranchID"] = df["ParentKey"] + "→" + df["ChildKey"]

children_map = defaultdict(list)
rows_by_child = defaultdict(list)
parent_map = {}

for idx, row in df.iterrows():
    p = row["ParentKey"]
    c = row["ChildKey"]
    children_map[p].append(c)
    rows_by_child[c].append(idx)
    if c not in parent_map:
        parent_map[c] = (p, idx)


# --- TB 面積の累積（total & per Golongan） ---
area_cache = {}
area_gol_cache = {}


def _parse_golongan_no(g):
    s = str(g).strip()
    if not s or s.lower() == "nan":
        return None
    m = re.search(r"(\d+)", s)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def compute_area(node_key: str) -> float:
    if node_key in area_cache:
        return area_cache[node_key]
    total = 0.0
    if node_key in rows_by_child:
        for idx in rows_by_child[node_key]:
            if df.at[idx, "Class"] == "TB":
                total += df.at[idx, "Area"]
    for child in children_map.get(node_key, []):
        total += compute_area(child)
    area_cache[node_key] = total
    return total


def compute_area_gol(node_key: str):
    if node_key in area_gol_cache:
        return area_gol_cache[node_key]
    g1 = g2 = g3 = 0.0
    if node_key in rows_by_child:
        for idx in rows_by_child[node_key]:
            if df.at[idx, "Class"] == "TB":
                a = df.at[idx, "Area"]
                g_no = _parse_golongan_no(df.at[idx, "Golongan_TB"])
                if g_no == 1:
                    g1 += a
                elif g_no == 2:
                    g2 += a
                elif g_no == 3:
                    g3 += a
    for child in children_map.get(node_key, []):
        cg1, cg2, cg3 = compute_area_gol(child)
        g1 += cg1
        g2 += cg2
        g3 += cg3
    area_gol_cache[node_key] = (g1, g2, g3)
    return area_gol_cache[node_key]


branch_areas = []
g1_list = []
g2_list = []
g3_list = []

for _, row in df.iterrows():
    child = row["ChildKey"]
    branch_areas.append(compute_area(child))
    g1, g2, g3 = compute_area_gol(child)
    g1_list.append(g1)
    g2_list.append(g2)
    g3_list.append(g3)

df["BranchArea"] = branch_areas
df["Area_G1"] = g1_list   # Golongan 1
df["Area_G2"] = g2_list   # Golongan 2
df["Area_G3"] = g3_list   # Golongan 3

# ★ ここを追加：0 のセルは表示上 NaN（空欄）にする
for col in ["BranchArea", "Area_G1", "Area_G2", "Area_G3"]:
    df[col] = df[col].where(df[col] != 0, np.nan)

st.caption("Cumulative TB areas per branch have been computed internally (total and per Golongan).")


# --- CanalKey の解析（内部用） ---
def parse_canal_key(ck: str):
    if ck is None:
        return "", "", ""
    ck = str(ck).strip()
    if not ck or ck.lower() == "nan":
        return "", "", ""
    parts = ck.split("_")
    if len(parts) < 2:
        return "", "", ""
    bank_char = parts[0][0] if parts[0] else ""
    bank = {"L": "Left", "R": "Right"}.get(bank_char, "")
    main_key = f"{parts[0]}_{parts[1]}"
    if len(parts) == 2:
        level = "Main"
    elif len(parts) >= 3:
        level = "Secondary"
    else:
        level = ""
    return bank, level, main_key


df["CanalBank"] = ""
df["CanalLevel"] = ""
df["MainKey"] = ""

for idx, row in df.iterrows():
    bank, level, main_key = parse_canal_key(row["CanalKey"])
    df.at[idx, "CanalBank"] = bank
    df.at[idx, "CanalLevel"] = level
    df.at[idx, "MainKey"] = main_key


# ========================================
# Build NFR 5-day arrays (Bank×Golongan×5Day, NFR in L/s/ha)
# ========================================

# NFR 列を柔軟に検出
raw_cols = [c for c in df_nfr_raw.columns]
nfr_col_candidates = [c for c in raw_cols if c.strip().lower() == "nfr"]
if not nfr_col_candidates:
    st.error(
        "Could not find 'NFR' column in NFR CSV.\n\n"
        f"Available columns: {raw_cols}"
    )
    st.stop()
nfr_col_name = nfr_col_candidates[0]

expected_nfr_cols = {"Bank", "Golongan", "Month", "Step", nfr_col_name}
if not expected_nfr_cols.issubset(set(df_nfr_raw.columns)):
    st.error(f"NFR CSV must contain at least: Bank, Golongan, Month, Step, {nfr_col_name}.\n"
             f"Actual columns: {raw_cols}")
    st.stop()

df_nfr = df_nfr_raw.copy()
df_nfr["Bank"] = df_nfr["Bank"].astype(str).str.strip()
df_nfr["Golongan"] = df_nfr["Golongan"].apply(_parse_golongan_no)
df_nfr["Month"] = pd.to_numeric(df_nfr["Month"], errors="coerce").astype("Int64")
df_nfr["Step"] = pd.to_numeric(df_nfr["Step"], errors="coerce").astype("Int64")
df_nfr["NFR"] = pd.to_numeric(df_nfr[nfr_col_name], errors="coerce").fillna(0.0)
df_nfr = df_nfr[df_nfr["Golongan"].isin([1, 2, 3])].copy()


def agg_nfr_ignore_zero(series: pd.Series) -> float:
    pos = series[series > 1e-9]
    if pos.empty:
        return 0.0
    return float(pos.mean())


df_nfr_aggr = (
    df_nfr
    .groupby(["Bank", "Golongan", "Month", "Step"], as_index=False)
    .agg(NFR=("NFR", agg_nfr_ignore_zero))
)

n_steps = 72
banks = ["Left", "Right"]
nfr_arr = {b: {g: np.zeros(n_steps, dtype=float) for g in [1, 2, 3]} for b in banks}

for _, r in df_nfr_aggr.iterrows():
    b = r["Bank"]
    g = int(r["Golongan"])
    m = int(r["Month"])
    s = int(r["Step"])
    if b not in nfr_arr or g not in nfr_arr[b]:
        continue
    if m not in water_month_order or not (1 <= s <= 6):
        continue
    m_idx = water_month_order.index(m)
    idx = m_idx * 6 + (s - 1)
    if 0 <= idx < n_steps:
        nfr_arr[b][g][idx] = r["NFR"]

labels = []
for i in range(n_steps):
    m = water_month_order[i // 6]
    step = (i % 6) + 1
    labels.append(f"{calendar.month_abbr[m]}-{step}")


# ========================================
# 3. NFR overview (L/s/ha, 6 patterns)
# ========================================
st.header("3. NFR (L/s/ha) by Bank × Golongan")

rows = []
idx_labels = []
for bank in banks:
    for g in [1, 2, 3]:
        rows.append(nfr_arr[bank][g])
        idx_labels.append(f"{bank}_G{g}")

nfr_table = pd.DataFrame(rows, index=idx_labels, columns=labels)

today = dt.date.today()
today_idx = date_to_5day_index(today)
if 0 <= today_idx < len(labels):
    today_label_nfr = labels[today_idx]
else:
    today_label_nfr = None


def highlight_today(col: pd.Series):
    """NFR テーブル用：本日の列だけハイライト＋太字。"""
    if today_label_nfr is not None and col.name == today_label_nfr:
        return ['background-color: #fff2a8; font-weight: bold'] * len(col)
    return [''] * len(col)


# ★ 0 を空欄にするフォーマッタ
def nfr_zero_blank_formatter(v):
    if pd.isna(v) or abs(v) < 1e-9:
        return ""
    return f"{v:.2f}"


styled_nfr = (
    nfr_table
    .style
    .format(nfr_zero_blank_formatter)   # ← ここだけ "{:.2f}" から変更
    .apply(highlight_today, axis=0)
    .set_properties(**{
        "text-align": "center",
        "padding": "2px 4px",
        "font-size": "12px",
        "white-space": "nowrap",
    })
    .set_table_styles([
        {"selector": "th.col_heading",
         "props": [("background-color", "#fee2e2"),
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

html_nfr = (
    '<div style="overflow-x:auto; text-align:left;">'
    f'{styled_nfr.to_html()}'
    '</div>'
)
st.markdown(html_nfr, unsafe_allow_html=True)

st.info("NFR values (L/s/ha) are imported from Page2's NFR_5Day.csv for each Bank × Golongan.")


# ========================================
# 4. Set canal efficiencies (Eff)
# ========================================
st.header("4. Canal conveyance efficiencies (Eff)")

eff_rows = [
    ["L_M", "Left bank Main Canal",      0.90],
    ["L_S", "Left bank Secondary Canal", 0.90],
    ["L_T", "Left bank Tertiary level",  0.80],
    ["R_M", "Right bank Main Canal",     0.90],
    ["R_S", "Right bank Secondary Canal",0.90],
    ["R_T", "Right bank Tertiary level", 0.80],
]

eff_df = pd.DataFrame(eff_rows, columns=["Code", "Description", "Value"])

col_eff_edit, col_eff_space = st.columns([2, 3])

with col_eff_edit:
    st.markdown("**Editing table – Eff parameters (draft)**")
    eff_edited = st.data_editor(
        eff_df,
        num_rows="fixed",
        hide_index=True,
        width="stretch",          # ★ 追加
        column_config={
            "Code": st.column_config.Column("Code", disabled=True),
            "Description": st.column_config.Column("Description", disabled=True),
            "Value": st.column_config.NumberColumn("Value", min_value=0.1, max_value=1.0, step=0.01),
        },
    )


# 計算用に辞書化
eff_map_raw = eff_edited.set_index("Code")["Value"].to_dict()


def get_eff(bank: str, level: str) -> float:
    if bank == "Left":
        key = f"L_{level}"
    else:
        key = f"R_{level}"
    return float(eff_map_raw.get(key, 1.0))


# 罫線付きのコンパクトなプレビュー表
st.markdown("**Calculation table – Eff used in Qp (compact view)**")

eff_preview = eff_edited.copy()
eff_preview.index = np.arange(1, len(eff_preview) + 1)  # ★ 1,2,3,... にする

styled_eff = (
    eff_preview
    .style
    .format({"Value": "{:.2f}"})
    .set_properties(**{
        "text-align": "center",
        "padding": "2px 4px",
        "font-size": "12px",
        "white-space": "nowrap",
    })
    .set_table_styles([
        {"selector": "th.col_heading",
         "props": [("background-color", "#e5f0ff"),
                   ("font-weight", "bold"),
                   ("border", "1px solid #999"),
                   ("padding", "2px 4px"),
                   ("white-space", "nowrap")]},
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

st.markdown(
    styled_eff.to_html(),
    unsafe_allow_html=True
)


def compute_qp_from_nfr(
    df: pd.DataFrame,
    nfr_arr: dict[str, dict[int, np.ndarray]],
) -> pd.DataFrame:
    """
    TB の NFR (L/s/ha) と効率 (Eff_T/S/M) を使って、
    各セグメントの Qp (5-day, L/s) を計算して df に 72 列を追加する。

    - df には Page4 で作ったネットワーク表（Class, Area, Golongan_TB,
      Bank, CanalLevel, ParentKey, ChildKey など）が入っている前提。
    - nfr_arr は Page4 で作った {Bank -> {Golongan -> 72要素の配列}}。
    - Eff は、既に定義済みの get_eff(bank, level) を内部で使用。
    - parent_map, children_map, labels, _parse_golongan_no も既存のものを使用。
    """

    n_rows = len(df)
    n_steps = len(labels)

    # T, S, M それぞれのレベルでの流量 [L/s]
    flows_T = np.zeros((n_rows, n_steps), dtype=float)
    flows_S = np.zeros((n_rows, n_steps), dtype=float)
    flows_M = np.zeros((n_rows, n_steps), dtype=float)

    # TB 行を抽出
    tb_rows = df[df["Class"] == "TB"].copy()

    for idx, row in tb_rows.iterrows():
        bank = row["Bank"]
        if bank not in nfr_arr:
            continue

        g_no = _parse_golongan_no(row["Golongan_TB"])
        if g_no not in [1, 2, 3]:
            continue

        area_tb = float(row["Area"])
        if area_tb <= 0:
            continue

        nfr_vec = nfr_arr[bank].get(g_no)
        if nfr_vec is None:
            continue

        # 田面需要 [L/s] → T レベル入口
        demand_vec = area_tb * nfr_vec  # L/s
        eff_T = get_eff(bank, "T")
        if eff_T <= 0:
            continue

        q_T = demand_vec / eff_T  # T レベル入口流量

        # ① TB 行に T レベル流量を計上
        seg_tb_idx = idx  # idx は元の df のインデックス
        flows_T[seg_tb_idx, :] += q_T

        # この TB の経路上の Secondary / Main セグメントを親方向にたどる
        sec_indices: list[int] = []
        main_indices: list[int] = []

        child_key = row["ChildKey"]
        while child_key in parent_map:
            parent_key, seg_idx = parent_map[child_key]
            level = df.at[seg_idx, "CanalLevel"]

            if level == "Secondary":
                sec_indices.append(seg_idx)
            elif level == "Main":
                main_indices.append(seg_idx)

            child_key = parent_key

        # ② Secondary セグメント：T+S を考慮
        if sec_indices:
            eff_S = get_eff(bank, "S")
            if eff_S > 0:
                q_S = q_T / eff_S
                for seg_idx in sec_indices:
                    flows_S[seg_idx, :] += q_S

        # ③ Main セグメント：T+S+M or T+M
        if main_indices:
            eff_M = get_eff(bank, "M")
            if eff_M <= 0:
                continue

            if sec_indices:
                eff_S = get_eff(bank, "S")
                if eff_S <= 0:
                    continue
                q_M = q_T / (eff_S * eff_M)  # T+S+M
            else:
                q_M = q_T / eff_M           # T+M

            for seg_idx in main_indices:
                flows_M[seg_idx, :] += q_M

    # 各行に表示する最終 Qp[L/s] を組み立て
    flows_mat_lps = np.zeros((n_rows, n_steps), dtype=float)
    for i, row in df.iterrows():
        level = row["CanalLevel"]
        cls = str(row["Class"]).strip()

        if cls == "TB":
            flows_mat_lps[i, :] = flows_T[i, :]
        elif level == "Secondary":
            flows_mat_lps[i, :] = flows_S[i, :]
        elif level == "Main":
            flows_mat_lps[i, :] = flows_M[i, :]
        else:
            flows_mat_lps[i, :] = 0.0

    # df に 5-day 列を追加（単位は L/s のまま）
    for j, lab in enumerate(labels):
        df[lab] = flows_mat_lps[:, j]

    return df

# ========================================
# Compute Qp (5-day, using TB demand and canal efficiencies)
# ========================================
# Page4 の Qp 計算ロジック（TB → Secondary → Main の積み上げ）を実行
df = compute_qp_from_nfr(df, nfr_arr)

# ★ Page5 でも使えるように、ネットワーク（Qp[L/s] 付き）と Eff を保存
#    - df は L/s 単位の Qp 列が 72 個付いたネットワーク表
#    - eff_map_raw は Eff 編集テーブルから作った辞書（Code → Value）
st.session_state["qp_df_network_base"] = df.copy()
st.session_state["qp_eff_map_raw"] = eff_map_raw


#===================================================
# --- Canal 親子関係の構築（Division 行ベース） ---
#===================================================

def build_canal_edges(df: pd.DataFrame) -> pd.DataFrame:
    """
    1行 = 水路区間（構造物ノード間のリンク）という前提で、
    Class = 'Division' 行を起点に、ParentCanal → ChildCanal の関係を作る。
    """
    df_loc = df.copy()
    for col in [
        "ParentKey", "ChildKey", "ParentName", "ChildName",
        "Canal", "Class", "CanalBank", "Bank"
    ]:
        if col in df_loc.columns:
            df_loc[col] = df_loc[col].astype(str).str.strip()

    df_loc["Class_upper"] = df_loc["Class"].str.upper()

    edges = []

    # Class = Division の行（各水路の始点）
    div_df = df_loc[df_loc["Class_upper"] == "DIVISION"].copy()

    for idx, r in div_df.iterrows():
        child_canal = r["Canal"]
        if not child_canal or child_canal.lower() == "nan":
            continue

        # Bank は CanalBank があればそれを優先
        bank = r.get("CanalBank", "") or r.get("Bank", "")
        bank = str(bank).strip()

        candidates = [
            (r["ParentKey"], r["ParentName"]),
            (r["ChildKey"],  r["ChildName"]),
        ]

        parent_canal = ""
        junction_key = ""
        junction_name = ""

        for node_key, node_name in candidates:
            if not node_key or str(node_key).lower() == "nan":
                continue

            rows_here = df_loc[
                (df_loc["ParentKey"] == node_key) |
                (df_loc["ChildKey"] == node_key)
            ]
            if rows_here.empty:
                continue

            parents_here = rows_here[
                (rows_here["Canal"] != child_canal) &
                (rows_here["Canal"] != "") &
                (rows_here["Canal"].str.lower() != "nan")
            ]
            if parents_here.empty:
                continue

            prim = parents_here[parents_here["Class_upper"] != "DIVISION"]
            target = prim if not prim.empty else parents_here

            target = target.sort_values(["Canal", "ParentKey", "ChildKey"])

            parent_canal = target["Canal"].iloc[0]
            junction_key = node_key
            junction_name = node_name
            break

        edges.append({
            "Bank": bank,
            "ParentCanal": parent_canal,
            "ChildCanal": child_canal,
            "JunctionKey": junction_key,
            "JunctionName": junction_name,
            "DivisionRowIndex": idx,
        })

    return pd.DataFrame(edges)

# df, BranchArea, Area_G1/2/3 を作った直後あたりで呼び出し
canal_edges = build_canal_edges(df)


# ========================================
# 5. Qp table (per reach, with 5-day flows, unit = m³/s)
# ========================================
st.header("5. Qp table (per reach, with 5-day flows, unit = m³/s)")

# --- Build (Bank, Golongan, Season) -> 5Day index from Page2 ---
def build_mt_start_map_from_page2() -> dict[tuple[str, int, str], int]:
    """
    Build (Bank, Golongan, Season) -> 0-based 5Day index (0..71)
    from Page2's season start dates.

    Seasons: "MT-1", "MT-2", "MT-3".
    For MT-3, rows with MT3_active = False are ignored
    (no orange highlight for MT-3 in those cases).
    """
    mt_start_map: dict[tuple[str, int, str], int] = {}

    lp = st.session_state.get("landprep_df", None)
    if isinstance(lp, pd.DataFrame) and not lp.empty and "Group" in lp.columns:
        has_mt3 = "MT3_start" in lp.columns
        has_mt3_active = "MT3_active" in lp.columns

        for _, r in lp.iterrows():
            grp = str(r["Group"])
            if "_" not in grp:
                continue
            bank_part, rest = grp.split("_", 1)
            bank = bank_part.strip()
            m_g = re.search(r"(\d+)", rest)
            if not m_g:
                continue
            gol_idx = int(m_g.group(1))

            # MT-1 / MT-2 / MT-3
            for season, col in [
                ("MT-1", "MT1_start"),
                ("MT-2", "MT2_start"),
                ("MT-3", "MT3_start"),
            ]:
                if col not in lp.columns:
                    continue

                # MT-3: skip if MT3_active is False
                if season == "MT-3" and has_mt3_active:
                    active_flag = r.get("MT3_active", True)
                    try:
                        active_flag = bool(active_flag)
                    except Exception:
                        active_flag = True
                    if not active_flag:
                        continue

                v = r.get(col, None)
                if v is None or pd.isna(v):
                    continue

                try:
                    d = pd.to_datetime(v).date()
                except Exception:
                    continue

                idx = date_to_5day_index(d)   # 0..71
                if 0 <= idx < 72:
                    key = (bank, gol_idx, season)
                    # keep the first value for each key
                    if key not in mt_start_map:
                        mt_start_map[key] = idx

    # Fallback: mt*_common from Page2 if landprep_df was not available
    if not mt_start_map:
        mt1_common = st.session_state.get("mt1_common", None)
        mt2_common = st.session_state.get("mt2_common", None)
        mt3_common = st.session_state.get("mt3_common", None)

        def common_idx(v):
            if v is None or pd.isna(v):
                return None
            try:
                d = pd.to_datetime(v).date()
            except Exception:
                return None
            idx = date_to_5day_index(d)
            return idx if 0 <= idx < 72 else None

        mt1_idx = common_idx(mt1_common)
        mt2_idx = common_idx(mt2_common)
        mt3_idx = common_idx(mt3_common)

        for bank in ["Left", "Right"]:
            for g in [1, 2, 3]:
                if mt1_idx is not None:
                    mt_start_map[(bank, g, "MT-1")] = mt1_idx
                if mt2_idx is not None:
                    mt_start_map[(bank, g, "MT-2")] = mt2_idx
                if mt3_idx is not None:
                    mt_start_map[(bank, g, "MT-3")] = mt3_idx

    return mt_start_map


mt_start_map = build_mt_start_map_from_page2()


def earliest_idx_for_mt(bank: str, season: str, gol_candidates: list[int]) -> Optional[int]:
    """For a given Bank / Season, return the earliest 5Day index among the candidate Golongan."""
    idx_list: list[int] = []
    for g in gol_candidates:
        idx = mt_start_map.get((bank, g, season), None)
        if idx is not None:
            idx_list.append(idx)
    if not idx_list:
        return None
    return min(idx_list)


# --- Maximize / Restore toggle for Qp table ---
if "qp_table_maximized" not in st.session_state:
    st.session_state["qp_table_maximized"] = False

# Short explanation for Qp table
st.markdown("""
- **Row visibility**: use the checkboxes below  
  (*Show TB / Show Branch Canal / Show Target Canal / Show Golongan 1 / 2 / 3 columns*)  
- **Orange cells**: indicate the start **5Day_ID** of **MT-1 / MT-2 / MT-3**  
  for each row (based on the season settings defined on Page2).
""")

# ---------- 1. Bank / Canal level / Canal ----------
st.markdown("### 1. Select Bank / Canal level / Canal")

row1_col_bank, row1_col_level, row1_col_view = st.columns([1.4, 1.8, 0.8])

with row1_col_bank:
    st.markdown("**Bank**")
    selected_bank = st.radio(
        label="Select Bank",
        options=["Left", "Right"],
        index=0,
        horizontal=True,
        label_visibility="collapsed",
    )

with row1_col_level:
    st.markdown("**Canal level**")
    level_sel = st.radio(
        label="Select Canal Level",
        options=["Main Canal", "Secondary Canal"],
        index=0,
        horizontal=True,
        label_visibility="collapsed",
    )

with row1_col_view:
    st.markdown("**View**")
    label = "Restore" if st.session_state["qp_table_maximized"] else "Maximize"
    if st.button(label, key="btn_qp_table_toggle"):
        st.session_state["qp_table_maximized"] = not st.session_state["qp_table_maximized"]
        st.rerun()

# Height limit: around 20 rows in normal mode, unlimited in Maximize mode
if st.session_state["qp_table_maximized"]:
    max_height_px = None
else:
    max_height_px = 640  # previously 480

# ---------- 2. Row visibility (TB / Branch / Target) ----------
st.markdown("### 2. Row visibility (TB / Branch / Target)")

col_tb, col_branch, col_target, col_gol = st.columns(4)

with col_tb:
    show_tb = st.checkbox("Show TB", value=False)
with col_branch:
    show_branch = st.checkbox("Show Branch Canal", value=True)
with col_target:
    show_target = st.checkbox("Show Target Canal", value=True)
with col_gol:
    show_golongan = st.checkbox("Show Golongan 1 / 2 / 3 columns", value=False)

# --- Bank / Canal level filter ---
df_bank = df[df["Bank"] == selected_bank].copy()

if level_sel == "Secondary Canal":
    level_name = "Secondary"
else:
    level_name = "Main"

df_level = df_bank[df_bank["CanalLevel"] == level_name].copy()

if df_level.empty:
    st.info(f"No rows found for Bank={selected_bank}, level={level_name}.")
    st.stop()

# --- Canal candidates sorted by CanalKey ---
order_canal = (
    df_level
    .sort_values("CanalKey")
    [["Canal", "CanalKey"]]
    .drop_duplicates(subset=["Canal"])
)
canal_names = order_canal["Canal"].tolist()

st.markdown(f"**Select {level_name} (sorted by CanalKey)**")
selected_canal = st.selectbox(
    label="Select Canal",
    options=canal_names,
)

# Reaches belonging to this canal
df_sel = df_level[df_level["Canal"] == selected_canal].copy()

if df_sel.empty:
    st.info("No rows for selected canal.")
    st.stop()

# --- Aggregate (internal segments + branch rows) ---
group_cols = [
    "Bank", "Canal",
    "ParentKey", "ChildKey",
    "ParentName", "ChildName",
    "Class", "SectionName",
]
agg_cols_base = ["BranchArea", "Area_G1", "Area_G2", "Area_G3"]
agg_cols_q = labels

# 1) internal segments
internal_summary = (
    df_sel
    .groupby(group_cols, as_index=False)[agg_cols_base + agg_cols_q]
    .sum()
)
internal_summary["BranchCanalName"] = ""

# 2) Division rows for branches from this canal
branches_for_this = pd.DataFrame()
if "canal_edges" in globals() and not canal_edges.empty:
    branches_for_this = canal_edges[
        (canal_edges["ParentCanal"] == selected_canal) &
        (canal_edges["Bank"] == selected_bank) &
        (canal_edges["ParentCanal"] != "")
    ].copy()

if not branches_for_this.empty:
    branch_records = []
    for _, b in branches_for_this.iterrows():
        idx_div = int(b["DivisionRowIndex"])

        record = {
            "Bank":       selected_bank,
            "Canal":      selected_canal,
            "ParentKey":  b["JunctionKey"],
            "ChildKey":   df.at[idx_div, "ChildKey"],
            "ParentName": b["JunctionName"],
            "ChildName":  b["ChildCanal"],
            "Class":      "Division",
            "SectionName": "",
            "BranchArea": df.at[idx_div, "BranchArea"],
            "Area_G1":    df.at[idx_div, "Area_G1"],
            "Area_G2":    df.at[idx_div, "Area_G2"],
            "Area_G3":    df.at[idx_div, "Area_G3"],
            "BranchCanalName": b["ChildCanal"],
        }
        for lab in labels:
            record[lab] = df.at[idx_div, lab]
        branch_records.append(record)

    branch_summary = pd.DataFrame(branch_records)
else:
    branch_summary = internal_summary.iloc[0:0].copy()
    branch_summary["BranchCanalName"] = ""

# 3) internal + branch
summary = pd.concat([internal_summary, branch_summary], ignore_index=True)

branch_flag = summary["BranchCanalName"].astype(str).str.strip()
is_branch = np.where(
    (summary["Class"].astype(str).str.strip() == "Division") &
    (branch_flag != "") &
    (branch_flag.str.lower() != "nan"),
    1,
    0,
)

summary["SortKey"] = np.where(
    is_branch == 0,
    summary["ChildKey"],
    summary["ParentKey"],
)
summary["SortIsBranch"] = is_branch

summary = (
    summary
    .sort_values(["SortKey", "SortIsBranch", "ChildName", "ParentKey", "ChildKey"])
    .reset_index(drop=True)
)

# Class filters
if "Class" in summary.columns:
    if not show_tb:
        summary = summary[summary["Class"] != "TB"]
    if not show_branch:
        summary = summary[summary["Class"] != "Division"]
    if not show_target:
        summary = summary[summary["Class"].isin(["TB", "Division"])]

# === Per-row MT-1 / MT-2 / MT-3 start indices ===
mt1_idx_rows: list[Optional[int]] = []
mt2_idx_rows: list[Optional[int]] = []
mt3_idx_rows: list[Optional[int]] = []

for _, r in summary.iterrows():
    bank = r["Bank"]
    areas = [
        r.get("Area_G1", 0.0),
        r.get("Area_G2", 0.0),
        r.get("Area_G3", 0.0),
    ]

    # Golongan candidates with non-zero area
    gol_candidates = [
        g for g, a in enumerate(areas, start=1)
        if not (pd.isna(a) or abs(a) < 1e-9)
    ]
    if not gol_candidates:
        gol_candidates = [1, 2, 3]

    mt1_idx_rows.append(earliest_idx_for_mt(bank, "MT-1", gol_candidates))
    mt2_idx_rows.append(earliest_idx_for_mt(bank, "MT-2", gol_candidates))
    mt3_idx_rows.append(earliest_idx_for_mt(bank, "MT-3", gol_candidates))

# ---------- 3. Planned Qp (fixed start) ----------
st.markdown("### 3. Planned Qp (fixed start)")

# Calendar labels: always Nov-1 .. Oct-6
ordered_labels = labels[:]

# Build display DataFrame
base_cols = [
    "Bank", "Canal", "ParentName",
    "ChildName", "Class", "SectionName",
    "BranchArea", "Area_G1", "Area_G2", "Area_G3",
    "BranchCanalName",
]
display_cols = base_cols + ordered_labels

display_df = summary[display_cols].copy()
display_df = display_df.rename(columns={
    "BranchArea": "Area (ha)",
    "Area_G1": "Golongan 1",
    "Area_G2": "Golongan 2",
    "Area_G3": "Golongan 3",
})

if not show_golongan:
    for gcol in ["Golongan 1", "Golongan 2", "Golongan 3"]:
        if gcol in display_df.columns:
            display_df = display_df.drop(columns=gcol)

for col in ["Bank", "Canal", "ParentName"]:
    if col in display_df.columns:
        display_df[col] = display_df[col].where(
            display_df[col].ne(display_df[col].shift()), ""
        )

# Convert L/s → m³/s for display
for lab in ordered_labels:
    if lab in display_df.columns:
        display_df[lab] = display_df[lab] / 1000.0

display_for_style = display_df.copy()
display_for_style.index = np.arange(1, len(display_for_style) + 1)


def highlight_qp(row: pd.Series) -> list[str]:
    """Highlight text cells for TB rows and Division (branch) rows."""
    cls = str(row.get("Class", "")).strip()
    branch_name = str(row.get("BranchCanalName", "")).strip()
    is_branch_row = branch_name not in ("", "nan", "NaN")

    text_cols = {
        "Bank", "Canal", "ParentKey", "ChildKey",
        "ParentName", "ChildName", "Class", "SectionName",
        "Area (ha)", "Golongan 1", "Golongan 2", "Golongan 3",
        "BranchCanalName",
    }
    branch_green_cols = {"ChildName", "Class", "SectionName", "Area (ha)"}

    styles: list[str] = []
    for col in row.index:
        if col not in text_cols:
            styles.append("")
            continue

        color = ""
        if is_branch_row and col in branch_green_cols:
            color = "#d9f99d"
        elif cls == "TB":
            color = "#ffe6cc"

        styles.append(f"background-color: {color}" if color else "")
    return styles


def style_qp(
    df_sty: pd.DataFrame,
    mt1_idx_rows: list[Optional[int]],
    mt2_idx_rows: list[Optional[int]],
    mt3_idx_rows: list[Optional[int]],
):
    cols = set(df_sty.columns)

    # number formatting
    fmt_dict = {}
    if "Area (ha)" in cols:
        fmt_dict["Area (ha)"] = "{:,.0f}"
    if "Golongan 1" in cols:
        fmt_dict["Golongan 1"] = lambda v: "" if (pd.isna(v) or abs(v) < 1e-9) else f"{v:,.0f}"
    if "Golongan 2" in cols:
        fmt_dict["Golongan 2"] = lambda v: "" if (pd.isna(v) or abs(v) < 1e-9) else f"{v:,.0f}"
    if "Golongan 3" in cols:
        fmt_dict["Golongan 3"] = lambda v: "" if (pd.isna(v) or abs(v) < 1e-9) else f"{v:,.0f}"

    qp_cols = [lab for lab in ordered_labels if lab in cols]
    for lab in qp_cols:
        fmt_dict[lab] = (lambda v, lab=lab: "" if (v == 0 or pd.isna(v)) else f"{v:,.3f}")

    # today's column label
    today = dt.date.today()
    t_idx = date_to_5day_index(today)
    today_label_local = labels[t_idx] if 0 <= t_idx < len(labels) else None

    # row-wise MT start highlight
    def highlight_mt_start(row: pd.Series) -> list[str]:
        pos = int(row.name) - 1  # display_for_style index is 1-based

        idxs: list[int] = []
        for idx in (mt1_idx_rows[pos], mt2_idx_rows[pos], mt3_idx_rows[pos]):
            if idx is not None and 0 <= idx < len(labels):
                idxs.append(idx)

        mt_labels = {
            labels[i] for i in idxs
            if 0 <= i < len(labels) and labels[i] in df_sty.columns
        }

        styles: list[str] = []
        for col in row.index:
            if col in mt_labels:
                styles.append("background-color: #ffcc66")  # deeper orange
            else:
                styles.append("")
        return styles

    styler = df_sty.style

    # 1) TB / Division text highlight
    styler = styler.apply(highlight_qp, axis=1)

    # 2) number format
    if fmt_dict:
        styler = styler.format(fmt_dict)

    # 3) data bars for Qp columns
    if qp_cols:
        styler = styler.bar(subset=qp_cols, axis=None, color="#99ccff")

    # 4) MT start cells (per row)
    styler = styler.apply(highlight_mt_start, axis=1)

    # 5) common cell style
    styler = styler.set_properties(**{
        "text-align": "center",
        "padding": "2px 4px",
        "font-size": "12px",
        "white-space": "nowrap",
    })

    # 6) thick border to the right of Area (ha)
    if "Area (ha)" in df_sty.columns:
        styler = styler.set_properties(
            subset=["Area (ha)"],
            **{"border-right": "2px solid #666"},
        )

    # 7) bold for Bank / Canal / ParentName / ChildName / Area (ha)
    bold_cols = [c for c in ["Bank", "Canal", "ParentName", "ChildName", "Area (ha)"]
                 if c in df_sty.columns]
    if bold_cols:
        styler = styler.set_properties(
            subset=bold_cols,
            **{"font-weight": "bold"},
        )

    # 8) table borders / headers
    styler = styler.set_table_styles([
        {
            "selector": "table",
            "props": [
                ("border-collapse", "collapse"),
                ("border", "1px solid #999"),
            ],
        },
        {
            "selector": "th.col_heading",
            "props": [
                ("background-color", "#e5ffe5" if level_name == "Main" else "#e5f0ff"),
                ("font-weight", "bold"),
                ("white-space", "nowrap"),
                ("padding", "2px 4px"),
                ("border", "1px solid #999"),
                ("position", "sticky"),
                ("top", "0"),
                ("z-index", "2"),
            ],
        },
        {
            "selector": "th.row_heading",
            "props": [
                ("background-color", "#f5f5f5"),
                ("font-weight", "bold"),
                ("border", "1px solid #999"),
                ("padding", "2px 4px"),
                ("white-space", "nowrap"),
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
            ],
        },
    ])

    # 9) today's column (yellow, overriding orange)
    if today_label_local is not None and today_label_local in df_sty.columns:
        styler = styler.set_properties(
            subset=[today_label_local],
            **{"background-color": "#fff2a8", "font-weight": "bold"},
        )

    return styler


# --- apply styling and render Qp table ---
styled_qp = style_qp(display_for_style, mt1_idx_rows, mt2_idx_rows, mt3_idx_rows)

wrapper_style = "text-align:left; overflow-x:auto; margin-top:0.5rem;"
if max_height_px is not None:
    wrapper_style += f" max-height:{max_height_px}px; overflow-y:auto;"
else:
    wrapper_style += " overflow-y:auto;"

html_qp = f"<div style='{wrapper_style}'>{styled_qp.to_html()}</div>"
st.markdown(html_qp, unsafe_allow_html=True)

# --- Legend for orange / yellow cells ---
st.markdown(
    """
    <div style="margin-top:4px; font-size:0.9rem;">
      <div style="display:flex; flex-wrap:wrap; gap:16px; align-items:center;">
        <div>
          <span style="display:inline-block;width:14px;height:14px;
                       background-color:#ffcc66;border:1px solid #999;
                       margin-right:6px;vertical-align:middle;"></span>
          <strong>Orange</strong>: Start 5Day_ID of <strong>MT-1 / MT-2 / MT-3</strong> for each row
        </div>
        <div>
          <span style="display:inline-block;width:14px;height:14px;
                       background-color:#fff2a8;border:1px solid #999;
                       margin-right:6px;vertical-align:middle;"></span>
          <strong>Yellow</strong>: <strong>Today</strong> (current 5-day step)
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------- CSV 出力用 DataFrame（全データ対象） ----------
# ※ 画面のフィルタ（Bank / Canal / TB / Golongan 表示）は無視して、
#    ネットワーク全体の Qp を CSV に出力する。

export_cols = [
    "Bank", "Canal", "CanalKey",
    "ParentKey", "ChildKey",
    "ParentName", "ChildName",
    "Class", "SectionName",
    "BranchArea", "Area_G1", "Area_G2", "Area_G3",
] + labels  # labels = 72 個の 5Day ラベル (Nov-1 .. Oct-6)

export_df = df[export_cols].copy()

# Qp 列は L/s → m³/s に変換して出力
for lab in labels:
    export_df[lab] = export_df[lab] / 1000.0

# ★ サーバ側の csvフォルダにも保存
csv_path = CSV_DIR / "Qp_all_reaches_m3s.csv"
export_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

# ★ ダウンロード用（これはクライアントPCのダウンロードフォルダに落ちる）
csv_bytes = export_df.to_csv(index=False).encode("utf-8-sig")
csv_name = "Qp_all_reaches_m3s.csv"
st.download_button(
    "Download Qp CSV (all reaches, m³/s)",
    csv_bytes,
    file_name=csv_name,
    mime="text/csv",
)

