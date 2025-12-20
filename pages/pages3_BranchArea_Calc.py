from pathlib import Path
import pandas as pd
import streamlit as st
from collections import defaultdict
import numpy as np
import re
import datetime as dt
from auth import check_auth
from menu import render_sidebar

# ========================================
# ===== Autentikasi Basic Auth ===========
# ========================================
if not check_auth():
    st.stop()
render_sidebar()

# ========================================
# Paths: project root, CSV folder
# ========================================
BASE_DIR = Path(__file__).resolve().parent.parent
CSV_DIR = BASE_DIR / "csv"

st.title("Branch Area Aggregation & Canal Filter")

st.markdown("""
This page reads a **water system network sheet** (parent–child nodes) from an uploaded CSV,
and computes **accumulated area per branch** from downstream TB nodes.

Expected column headers in the sheet:

- `Parent Key`   : parent node key (D column)
- `Child Key`    : child node key (E column)
- `Parent Name`  : parent node name
- `Child Name`   : child node name
- `Class`        : node class (TB = terminal fields)
- `Area`         : area (only TB rows have meaningful values)
- `Golongan_TB`  : Golongan for TB nodes (Y column)
- `Canal`        : canal name
- `Canal Key`    : canal key
- `Section`      : section name / ID

Notes:

- Column I (BranchID) is **not used**; instead we generate `BranchID = ParentKey → ChildKey`.
- `Child Key` starts with `L` or `R`: `L` → `Left` bank, `R` → `Right` bank.
- `Canal Key` pattern:
  - Main Canal: `"L_M2"`         (Bank + "_" + MainCode)
  - Secondary : `"L_M2_S01"`     (Bank + "_" + MainCode + "_" + SecondaryCode)
""")

# ========================================
# 1. Network CSV (Diagram) – from csv folder or uploaded file
# ========================================
st.header("1. Network CSV (Diagram)")

# ---- 1-1. csv フォルダ内の Diagram* を候補として選択 ----
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
    "Select network CSV from `csv` folder (file names containing 'Diagram')",
    options=net_labels,
    index=0,
)
net_path = network_candidates[net_labels.index(selected_net_label)]

st.caption(f"Default network data source: `{net_path.name}` in `csv` folder.")

# ---- 1-2. 任意で、このページで CSV をアップロードして上書き（保存はしない） ----
uploaded_file = st.file_uploader(
    "Optionally upload another network CSV (used only in this session)",
    type=["csv"],
)

# ---- 1-3. 実際に読むファイルを決定 ----
if uploaded_file is not None:
    src_desc = f"uploaded file: `{uploaded_file.name}`"
    try:
        # 2 行目がヘッダー → header=1 （Diagram エクスポートと同じ前提）
        df_raw = pd.read_csv(uploaded_file, header=1)
    except Exception as e:
        st.error(f"Failed to read uploaded network CSV: {e}")
        st.stop()
else:
    src_desc = f"`{net_path.name}` in `csv` folder"
    try:
        df_raw = pd.read_csv(net_path, header=1)
    except Exception as e:
        st.error(f"Failed to read network CSV from csv folder: {e}")
        st.stop()

st.info(f"Using network CSV: {src_desc}")

st.subheader("Raw network data (head)")
st.dataframe(df_raw.head())


# ========================================
# 2. 必要列を抽出
# ========================================
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

missing = [v for v in required_cols.values() if v not in df_raw.columns]
if missing:
    st.error(f"Required columns missing in uploaded CSV: {missing}")
    st.stop()

df = pd.DataFrame({
    "ParentKey":   df_raw[required_cols["ParentKey"]],
    "ChildKey":    df_raw[required_cols["ChildKey"]],
    "ParentName":  df_raw[required_cols["ParentName"]],
    "ChildName":   df_raw[required_cols["ChildName"]],
    "Class":       df_raw[required_cols["Class"]],
    "Area":        pd.to_numeric(df_raw[required_cols["Area"]], errors="coerce").fillna(0.0),
    "Golongan_TB": df_raw[required_cols["Golongan_TB"]],
    "Canal":       df_raw[required_cols["Canal"]],
    "CanalKey":    df_raw[required_cols["CanalKey"]],
    "SectionName": df_raw[required_cols["SectionName"]],
})

# 整形：文字列化 & strip
for col in ["ParentKey", "ChildKey", "ParentName", "ChildName",
            "Class", "Golongan_TB", "Canal", "CanalKey", "SectionName"]:
    df[col] = df[col].astype(str).str.strip()

# SectionName の "nan" などは空白に
df["SectionName"] = df["SectionName"].replace("nan", "").fillna("")

# 子Keyの先頭文字から Bank を判定
df["Bank"] = df["ChildKey"].str[0].map({"L": "Left", "R": "Right"}).fillna("")

# BranchID を ParentKey→ChildKey から生成
df["BranchID"] = df["ParentKey"] + "→" + df["ChildKey"]

st.subheader("Network core columns")
st.dataframe(df[["ParentKey", "ChildKey", "ParentName", "ChildName",
                 "BranchID", "Class", "Area",
                 "Golongan_TB", "Bank", "Canal", "CanalKey", "SectionName"]].head(30))

# ========================================
# 3. グラフ構造を構築（Parent -> Child の隣接リスト）
# ========================================
children_map = defaultdict(list)
rows_by_child = defaultdict(list)

for idx, row in df.iterrows():
    p = row["ParentKey"]
    c = row["ChildKey"]
    children_map[p].append(c)
    rows_by_child[c].append(idx)

# ========================================
# 4. Node ごとの下流面積を再帰的に計算
# ========================================
area_cache = {}

def compute_area(node_key: str) -> float:
    """指定ノード以下の TB 面積合計を返す（メモ化つき）。"""
    if node_key in area_cache:
        return area_cache[node_key]

    total = 0.0

    # 自分自身が TB ノードとして現れる行の Area を足す
    if node_key in rows_by_child:
        for idx in rows_by_child[node_key]:
            if df.at[idx, "Class"] == "TB":
                total += df.at[idx, "Area"]

    # 子ノードへ再帰
    for child in children_map.get(node_key, []):
        total += compute_area(child)

    area_cache[node_key] = total
    return total

# ========================================
# 4.5 Golongan 別の TB 面積を Node ごとに再帰集計
# ========================================
area_gol_cache = {}

def _parse_golongan_no(g):
    """Golongan_TB の文字列から 1 / 2 / 3 を取り出す。例: 'G1', 'Golongan 2' など。"""
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

def compute_area_gol(node_key: str):
    """指定ノード以下の TB 面積を Golongan 1/2/3 別に返す。"""
    if node_key in area_gol_cache:
        return area_gol_cache[node_key]

    g1 = g2 = g3 = 0.0

    # 自身が TB ノードとして現れる行の Area を Golongan 別に足す
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

    # 子ノードへ再帰
    for child in children_map.get(node_key, []):
        cg1, cg2, cg3 = compute_area_gol(child)
        g1 += cg1
        g2 += cg2
        g3 += cg3

    area_gol_cache[node_key] = (g1, g2, g3)
    return area_gol_cache[node_key]


# ========================================
# 4.5 Division 行から Canal 同士の親子関係を作る
# ========================================
def build_canal_edges(df: pd.DataFrame) -> pd.DataFrame:
    """
    1行 = 水路区間（構造物ノード間のリンク）という前提で、
    Class = 'Division' 行を起点に、
    ParentCanal → ChildCanal の関係を作る。

    - 子水路: Division 行の Canal
    - 親水路: 同じ構造物ノードを通る行のうち、
              Canal != 子水路 かつ Class != Division を優先して選ぶ。
      （幹線側の reach を優先）
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

        # Division 行が接続している 2 つの構造物ノード
        candidates = [
            (r["ParentKey"], r["ParentName"]),
            (r["ChildKey"],  r["ChildName"]),
        ]

        parent_canal = ""
        junction_key = ""
        junction_name = ""

        # どちらかのノードで、親水路候補を探す
        for node_key, node_name in candidates:
            if not node_key or str(node_key).lower() == "nan":
                continue

            # この構造物ノードを通る全ての行
            rows_here = df_loc[
                (df_loc["ParentKey"] == node_key) |
                (df_loc["ChildKey"] == node_key)
            ]
            if rows_here.empty:
                continue

            # 子水路以外の Canal
            parents_here = rows_here[
                (rows_here["Canal"] != child_canal) &
                (rows_here["Canal"] != "") &
                (rows_here["Canal"].str.lower() != "nan")
            ]

            if parents_here.empty:
                continue

            # ① 最優先: Class != Division の Canal（幹線側）
            prim = parents_here[parents_here["Class_upper"] != "DIVISION"]

            if not prim.empty:
                target = prim
            else:
                # ② どうしても無ければ、子水路以外の Canal から選ぶ
                target = parents_here

            # 安定のため、ソートして先頭を採用
            target = target.sort_values(
                ["Canal", "ParentKey", "ChildKey"]
            )

            parent_canal = target["Canal"].iloc[0]
            junction_key = node_key
            junction_name = node_name
            break

        edges.append({
            "Bank": bank,
            "ParentCanal": parent_canal,        # "" → 親水路なし（最上流）
            "ChildCanal": child_canal,
            "JunctionKey": junction_key,        # 分水構造物ノード
            "JunctionName": junction_name,
            "DivisionRowIndex": idx,            # df 内の行番号（BranchArea 参照用）
        })

    return pd.DataFrame(edges)


# ========================================
# 5. 各ブランチ（行）に BranchArea & Golongan 別面積を割り当て
# ========================================
branch_areas = []
g1_list = []
g2_list = []
g3_list = []

for idx, row in df.iterrows():
    child = row["ChildKey"]

    # 合計 TB 面積
    branch_areas.append(compute_area(child))

    # Golongan 1/2/3 別 TB 面積
    g1, g2, g3 = compute_area_gol(child)
    g1_list.append(g1)
    g2_list.append(g2)
    g3_list.append(g3)

df["BranchArea"] = branch_areas
df["Area_G1"] = g1_list   # Golongan 1
df["Area_G2"] = g2_list   # Golongan 2
df["Area_G3"] = g3_list   # Golongan 3

st.info("""
`BranchArea` is computed as the sum of TB areas downstream of each **Child** node.
It is used below for Main/Secondary canal summaries.
""")


# ========================================
# 6. TB 行の Bank × Golongan_TB 別面積集計
# ========================================
st.header("6. Area summary by Bank × Golongan_TB (TB nodes only)")

tb_rows = df[df["Class"] == "TB"].copy()

if tb_rows.empty:
    st.info("No TB rows found (Class == 'TB').")
else:
    summary = (
        tb_rows
        .groupby(["Bank", "Golongan_TB"], dropna=False)["Area"]
        .sum()
        .reset_index()
    )

    pivot = summary.pivot(index="Golongan_TB", columns="Bank", values="Area").fillna(0.0)

    for bank in ["Left", "Right"]:
        if bank not in pivot.columns:
            pivot[bank] = 0.0
    pivot = pivot[["Left", "Right"]]
    pivot["Total_LR"] = pivot["Left"] + pivot["Right"]

    total_row = pivot.sum(axis=0).to_frame().T
    total_row.index = ["TOTAL"]
    pivot_all = pd.concat([pivot, total_row])

    # ★ 表示用に列名を変更（Total_LR → Total）
    pivot_all = pivot_all.rename(columns={"Total_LR": "Total"})

    # ★ Styler でフォーマット・罫線・TOTAL 行太字を設定
    def highlight_total(row):
        # 行名が TOTAL の行だけ太字
        if row.name == "TOTAL":
            return ["font-weight: bold" for _ in row]
        return ["" for _ in row]

    styler = (
        pivot_all
        .style
        # 数値は整数＋3桁区切り
        .format("{:,.0f}")
        # TOTAL 行だけ太字
        .apply(highlight_total, axis=1)
        # テーブル全体の罫線など
        .set_table_styles([
            {
                "selector": "table",
                "props": [
                    ("border-collapse", "collapse"),
                    ("margin", "0 auto"),
                ],
            },
            {
                "selector": "th",
                "props": [
                    ("border", "1px solid #999"),
                    ("padding", "2px 6px"),
                    ("background-color", "#e5f0ff"),
                    ("font-weight", "bold"),
                    ("text-align", "center"),
                ],
            },
            {
                "selector": "td",
                "props": [
                    ("border", "1px solid #999"),
                    ("padding", "2px 6px"),
                    ("text-align", "right"),
                ],
            },
            {
                "selector": "th.row_heading",
                "props": [
                    ("border", "1px solid #999"),
                    ("padding", "2px 6px"),
                    ("background-color", "#f5f5f5"),
                    ("font-weight", "bold"),
                ],
            },
        ])
    )

    html_pivot = styler.to_html()

    html_pivot = (
        "<div style='display:flex; justify-content:center; margin-top:0.5rem;'>"
        "<div style='width:fit-content; text-align:center;'>"
        f"{html_pivot}"
        "</div></div>"
    )
    st.markdown(html_pivot, unsafe_allow_html=True)


    st.info("""
The table above shows TB area (sum) for each `Golongan_TB`, split by **Left/Right banks**,
and their combined total (`Total_LR`).  
The last row (`TOTAL`) is the overall sum across all Golongan_TB.
""")

# ========================================
# 7. Canal 情報を使ったフィルタテーブル
# ========================================
st.header("7. Canal filter view (Main / Secondary)")

# CanalKey から Bank / Level / MainKey を判定
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


# CanalKey から CanalBank / CanalLevel / MainKey を設定
df["CanalBank"] = ""
df["CanalLevel"] = ""
df["MainKey"] = ""

for idx, row in df.iterrows():
    bank, level, main_key = parse_canal_key(row["CanalKey"])
    df.at[idx, "CanalBank"] = bank
    df.at[idx, "CanalLevel"] = level
    df.at[idx, "MainKey"] = main_key

# Division 行から Canal 親子テーブルを構築
canal_edges = build_canal_edges(df)

# TB が 1 行も無いケースだけメッセージ
tb_rows = df[df["Class"] == "TB"].copy()
if tb_rows.empty:
    st.info("No TB rows found (Class == 'TB'). But canal table is still available.")

# --- Bank & level フィルタ ---
col_bank, col_level = st.columns(2)

with col_bank:
    bank_options = ["Left", "Right"]
    selected_bank = st.radio(
        "Bank filter",
        bank_options,
        index=0,
        horizontal=True,
    )

with col_level:
    level_sel = st.radio(
        "Canal level",
        options=["Main Canal", "Secondary Canal"],
        index=0,
        horizontal=True,
    )

# Bank フィルタ（ChildKey ベースの Bank）
df_bank = df[df["Bank"] == selected_bank].copy()

# --------------------------------------------------
#  Secondary Canal Table
# --------------------------------------------------
if level_sel == "Secondary Canal":
    sec_candidates = df_bank[df_bank["CanalLevel"] == "Secondary"]

    if sec_candidates.empty:
        st.info("No Secondary Canal names found for the selected bank.")
    else:
        # Secondary 名の並び（上流側 ParentKey が小さい順）
        order_sec = (
            sec_candidates
            .groupby("Canal")["ParentKey"].min()
            .reset_index()
            .sort_values("ParentKey")
        )
        sec_names = order_sec["Canal"].tolist()

        sec_name_sel = st.selectbox("Select Secondary Canal Name", sec_names)

        # この Secondary Canal に属する区間（内部 reach + TB）
        sec_rows = df_bank[df_bank["Canal"] == sec_name_sel].copy()

        if sec_rows.empty:
            st.info("No rows found for the selected Secondary Canal.")
        else:
            # --- ① Secondary 内部区間 ---
            group_cols = [
                "Bank", "Canal",
                "ParentKey", "ChildKey",
                "ParentName", "ChildName", "Class", "SectionName"
            ]

            internal_summary = (
                sec_rows
                .groupby(group_cols, as_index=False)[
                    ["BranchArea", "Area_G1", "Area_G2", "Area_G3"]
                ]
                .sum()
            )

            internal_summary["SectionName"] = (
                internal_summary["SectionName"]
                .replace("nan", "")
                .fillna("")
            )
            internal_summary["BranchCanalName"] = ""   # 初期値（通常行）

            # --- ② Secondary からさらに下流へ分岐する Canal 行 ---
            branches_for_sec = pd.DataFrame()
            if not canal_edges.empty:
                branches_for_sec = canal_edges[
                    (canal_edges["ParentCanal"] == sec_name_sel) &
                    (canal_edges["Bank"] == selected_bank) &
                    (canal_edges["ParentCanal"] != "")
                ].copy()

            if not branches_for_sec.empty:
                branch_records = []
                for _, b in branches_for_sec.iterrows():
                    idx_div = int(b["DivisionRowIndex"])
                    branch_records.append({
                        "Bank": selected_bank,
                        "Canal": sec_name_sel,
                        "ParentKey": b["JunctionKey"],
                        "ChildKey": df.at[idx_div, "ChildKey"],
                        "ParentName": b["JunctionName"],
                        "ChildName": b["ChildCanal"],      # 表示は「下流側 Canal 名」
                        "Class": "Division",
                        "SectionName": "",
                        "BranchArea": df.at[idx_div, "BranchArea"],
                        "Area_G1": df.at[idx_div, "Area_G1"],
                        "Area_G2": df.at[idx_div, "Area_G2"],
                        "Area_G3": df.at[idx_div, "Area_G3"],
                        "BranchCanalName": b["ChildCanal"],
                    })
                branch_summary = pd.DataFrame(branch_records)
            else:
                branch_summary = internal_summary.iloc[0:0].copy()
                branch_summary["BranchCanalName"] = ""

            # --- ③ 内部区間 + 分岐行を結合 ---
            summary_sec = pd.concat([internal_summary, branch_summary], ignore_index=True)

            # 並べ替え（通常行:ChildKey, 分岐行:ParentKey）
            branch_flag = summary_sec["BranchCanalName"].astype(str).str.strip()
            is_branch = np.where(
                (branch_flag != "") & (branch_flag.str.lower() != "nan"),
                1,  # 分岐行
                0,  # 通常行
            )
            summary_sec["SortKey"] = np.where(
                is_branch == 0,
                summary_sec["ChildKey"],
                summary_sec["ParentKey"],
            )
            summary_sec["SortIsBranch"] = is_branch
            summary_sec = (
                summary_sec
                .sort_values(
                    ["SortKey", "SortIsBranch", "ChildName", "ParentKey", "ChildKey"]
                )
                .reset_index(drop=True)
            )

            # 表示用：ParentKey は出さない
            display_sec = summary_sec[[
                "Bank", "Canal", "ParentName",
                "ChildName", "Class", "SectionName",
                "BranchArea", "Area_G1", "Area_G2", "Area_G3",
                "BranchCanalName"
            ]].copy()

            display_sec = display_sec.rename(columns={
                "BranchArea": "Area (ha)",
                "Area_G1": "Golongan 1",
                "Area_G2": "Golongan 2",
                "Area_G3": "Golongan 3",
            })

            for col in ["Bank", "Canal", "ParentName"]:
                display_sec[col] = display_sec[col].where(
                    display_sec[col].ne(display_sec[col].shift()), ""
                )

            # --- 行ハイライト（Secondary 用） ---
            def highlight_sec(row: pd.Series):
                cls = str(row["Class"]).strip()
                branch_name = str(row.get("BranchCanalName", "")).strip()
                is_branch_row = branch_name not in ("", "nan", "NaN")

                styles = []
                for col in row.index:
                    # Golongan 列は常にグレー系
                    if col in ("Golongan 1", "Golongan 2", "Golongan 3"):
                        if cls == "TB" or is_branch_row:
                            color = "#d0d0d0"  # 濃いめグレー
                        else:
                            color = "#f2f2f2"  # 薄めグレー
                    else:
                        if cls == "TB":
                            color = "#ffe6cc"
                        elif is_branch_row:
                            color = "#cce6ff"
                        else:
                            color = ""
                    styles.append(f"background-color: {color}" if color else "")
                return styles

            def style_sec(df_sty: pd.DataFrame):
                styler = (
                    df_sty
                    .style
                    .apply(highlight_sec, axis=1)
                    .format({
                        "Area (ha)": "{:,.0f}",
                        "Golongan 1": lambda v: "" if v == 0 else f"{v:,.0f}",
                        "Golongan 2": lambda v: "" if v == 0 else f"{v:,.0f}",
                        "Golongan 3": lambda v: "" if v == 0 else f"{v:,.0f}",
                    })
                    .set_properties(**{"text-align": "center", "padding": "2px 6px"})
                    .set_properties(
                        subset=["Area (ha)"],
                        **{"border-right": "2px solid #666"}  # Area と Golongan の仕切り線
                    )
                    .set_properties(
                        subset=["ParentName", "ChildName", "Area (ha)"],
                        **{"font-weight": "bold"}
                    )
                    .set_table_styles([
                        {"selector": "table",
                         "props": [("border-collapse", "collapse"),
                                   ("border", "1px solid #999")]},
                        {"selector": "th.col_heading",
                         "props": [("background-color", "#e5f0ff"),
                                   ("font-weight", "bold"),
                                   ("border", "1px solid #999"),
                                   ("padding", "2px 4px"),
                                   ("white-space", "nowrap"),
                                   ("text-align", "center")]},
                        {"selector": "th.row_heading",
                         "props": [("background-color", "#e5f0ff"),
                                   ("font-weight", "bold"),
                                   ("border", "1px solid #999"),
                                   ("padding", "2px 4px"),
                                   ("white-space", "nowrap"),
                                   ("text-align", "center")]},
                        {"selector": "th.blank",
                         "props": [("border", "1px solid #999"),
                                   ("padding", "2px 4px")]},
                        {"selector": "td",
                         "props": [("border", "1px solid #999"),
                                   ("padding", "2px 4px"),
                                   ("white-space", "nowrap"),
                                   ("text-align", "center")]},
                        # ヘッダー側 Area(ha) 右にも太線
                        {"selector": "th.col_heading.level0.col6",
                         "props": [("border-right", "2px solid #666")]},
                    ])
                )
                return styler

            df_sec_for_style = display_sec.copy()
            styled_sec = style_sec(df_sec_for_style)

            # BranchCanalName 列は非表示（ハイライト判定だけに使用）
            try:
                styled_sec = styled_sec.hide(axis="columns", subset=["BranchCanalName"])
            except Exception:
                df_sec_for_style = df_sec_for_style.drop(columns=["BranchCanalName"])
                styled_sec = style_sec(df_sec_for_style)

            html_sec = (
                "<div style='display:flex; justify-content:center; margin-top:0.5rem;'>"
                "<div style='width:fit-content; text-align:center;'>"
                f"{styled_sec.to_html()}"
                "</div></div>"
            )

            st.subheader("Secondary Canal Table (within selected Secondary)")
            st.markdown(html_sec, unsafe_allow_html=True)

# --------------------------------------------------
#  Main Canal Table
# --------------------------------------------------
else:
    main_candidates = df_bank[df_bank["CanalLevel"] == "Main"]

    if main_candidates.empty:
        st.info("No Main Canal names found for the selected bank.")
    else:
        order_main = (
            main_candidates
            .groupby("Canal")["ParentKey"].min()
            .reset_index()
            .sort_values("ParentKey")
        )
        main_names = order_main["Canal"].tolist()

        main_name_sel = st.selectbox("Select Main Canal Name", main_names)

        # この Main Canal に属する区間
        main_rows = df_bank[df_bank["Canal"] == main_name_sel].copy()

        if main_rows.empty:
            st.info("No rows found for the selected Main Canal.")
        else:
            # --- ① Main 内部区間 ---
            internal_rows = main_rows.copy()

            group_cols_main = [
                "Bank", "Canal",
                "ParentKey", "ChildKey",
                "ParentName", "ChildName", "Class", "SectionName"
            ]

            internal_summary = (
                internal_rows
                .groupby(group_cols_main, as_index=False)[
                    ["BranchArea", "Area_G1", "Area_G2", "Area_G3"]
                ]
                .sum()
            )
            internal_summary["SectionName"] = (
                internal_summary["SectionName"]
                .replace("nan", "")
                .fillna("")
            )
            internal_summary["BranchCanalName"] = ""   # 初期値（通常行）

            # --- ② Main → 他 Canal 分岐行（Division ベース） ---
            branches_for_main = pd.DataFrame()
            if not canal_edges.empty:
                branches_for_main = canal_edges[
                    (canal_edges["ParentCanal"] == main_name_sel) &
                    (canal_edges["Bank"] == selected_bank) &
                    (canal_edges["ParentCanal"] != "")
                ].copy()

            if not branches_for_main.empty:
                branch_records = []
                for _, b in branches_for_main.iterrows():
                    idx_div = int(b["DivisionRowIndex"])
                    branch_records.append({
                        "Bank": selected_bank,
                        "Canal": main_name_sel,
                        "ParentKey": b["JunctionKey"],
                        "ChildKey": df.at[idx_div, "ChildKey"],
                        "ParentName": b["JunctionName"],
                        "ChildName": b["ChildCanal"],   # 表示は Canal 名
                        "Class": "Division",
                        "SectionName": "",
                        "BranchArea": df.at[idx_div, "BranchArea"],
                        "Area_G1": df.at[idx_div, "Area_G1"],
                        "Area_G2": df.at[idx_div, "Area_G2"],
                        "Area_G3": df.at[idx_div, "Area_G3"],
                        "BranchCanalName": b["ChildCanal"],
                    })
                branch_summary = pd.DataFrame(branch_records)
            else:
                branch_summary = internal_summary.iloc[0:0].copy()
                branch_summary["BranchCanalName"] = ""

            # --- ③ 内部区間 + 分岐行を結合 ---
            summary_main = pd.concat([internal_summary, branch_summary], ignore_index=True)

            # 並べ替え（通常行:ChildKey, 分岐行:ParentKey）
            branch_flag = summary_main["BranchCanalName"].astype(str).str.strip()
            is_branch = np.where(
                (branch_flag != "") & (branch_flag.str.lower() != "nan"),
                1,  # 分岐行
                0,  # 通常行
            )
            summary_main["SortKey"] = np.where(
                is_branch == 0,
                summary_main["ChildKey"],
                summary_main["ParentKey"],
            )
            summary_main["SortIsBranch"] = is_branch
            summary_main = (
                summary_main
                .sort_values(
                    ["SortKey", "SortIsBranch", "ChildName", "ParentKey", "ChildKey"]
                )
                .reset_index(drop=True)
            )

            # 表示用
            display_main = summary_main[[
                "Bank", "Canal", "ParentName",
                "ChildName", "Class", "SectionName",
                "BranchArea", "Area_G1", "Area_G2", "Area_G3",
                "BranchCanalName"
            ]].copy()

            display_main = display_main.rename(columns={
                "BranchArea": "Area (ha)",
                "Area_G1": "Golongan 1",
                "Area_G2": "Golongan 2",
                "Area_G3": "Golongan 3",
            })

            for col in ["Bank", "Canal", "ParentName"]:
                display_main[col] = display_main[col].where(
                    display_main[col].ne(display_main[col].shift()), ""
                )

            # --- 行ハイライト（Main 用）---
            def highlight_main(row: pd.Series):
                cls = str(row["Class"]).strip()
                branch_name = str(row.get("BranchCanalName", "")).strip()
                is_branch_row = branch_name not in ("", "nan", "NaN")

                styles = []
                for col in row.index:
                    if col in ("Golongan 1", "Golongan 2", "Golongan 3"):
                        if cls == "TB" or is_branch_row:
                            color = "#d0d0d0"
                        else:
                            color = "#f2f2f2"
                    else:
                        if cls == "TB":
                            color = "#ffe6cc"
                        elif is_branch_row:
                            color = "#cce6ff"
                        else:
                            color = ""
                    styles.append(f"background-color: {color}" if color else "")
                return styles

            def style_main(df_sty: pd.DataFrame):
                styler = (
                    df_sty
                    .style
                    .apply(highlight_main, axis=1)
                    .format({
                        "Area (ha)": "{:,.0f}",
                        "Golongan 1": lambda v: "" if v == 0 else f"{v:,.0f}",
                        "Golongan 2": lambda v: "" if v == 0 else f"{v:,.0f}",
                        "Golongan 3": lambda v: "" if v == 0 else f"{v:,.0f}",
                    })
                    .set_properties(**{"text-align": "center", "padding": "2px 6px"})
                    .set_properties(
                        subset=["Area (ha)"],
                        **{"border-right": "2px solid #666"}
                    )
                    .set_properties(
                        subset=["ParentName", "ChildName", "Area (ha)"],
                        **{"font-weight": "bold"}
                    )
                    .set_table_styles([
                        {"selector": "table",
                         "props": [("border-collapse", "collapse"),
                                   ("border", "1px solid #999")]},
                        {"selector": "th.col_heading",
                         "props": [("background-color", "#e5ffe5"),
                                   ("font-weight", "bold"),
                                   ("border", "1px solid #999"),
                                   ("padding", "2px 4px"),
                                   ("white-space", "nowrap"),
                                   ("text-align", "center")]},
                        {"selector": "th.row_heading",
                         "props": [("background-color", "#e5ffe5"),
                                   ("font-weight", "bold"),
                                   ("border", "1px solid #999"),
                                   ("padding", "2px 4px"),
                                   ("white-space", "nowrap"),
                                   ("text-align", "center")]},
                        {"selector": "th.blank",
                         "props": [("border", "1px solid #999"),
                                   ("padding", "2px 4px")]},
                        {"selector": "td",
                         "props": [("border", "1px solid #999"),
                                   ("padding", "2px 4px"),
                                   ("white-space", "nowrap"),
                                   ("text-align", "center")]},
                        {"selector": "th.col_heading.level0.col6",
                         "props": [("border-right", "2px solid #666")]},
                    ])
                )
                return styler

            df_main_for_style = display_main.copy()
            styled_main = style_main(df_main_for_style)

            try:
                styled_main = styled_main.hide(axis="columns", subset=["BranchCanalName"])
            except Exception:
                df_main_for_style = df_main_for_style.drop(columns=["BranchCanalName"])
                styled_main = style_main(df_main_for_style)

            html_main = (
                "<div style='display:flex; justify-content:center; margin-top:0.5rem;'>"
                "<div style='width:fit-content; text-align:center;'>"
                f"{styled_main.to_html()}"
                "</div></div>"
            )

            st.subheader("Main Canal Table (branches within selected Main)")
            st.markdown(html_main, unsafe_allow_html=True)

            st.info("""
- Each row is a **reach (Parent structure → Child structure)** belonging to the selected Main Canal.

- `Area (ha)` is the sum of TB areas downstream of the **Child structure**.

- `Golongan 1 / 2 / 3` columns show how the TB area is composed by Golongan,
  based on the `Golongan_TB` values of downstream TB nodes.

- Rows where `BranchCanalName` is **non-empty**
  → branches from this Main Canal into a **Secondary (or other) canal**
  (these rows are highlighted in light blue).

- Rows with `Class == "TB"` are highlighted in light orange.
""")
