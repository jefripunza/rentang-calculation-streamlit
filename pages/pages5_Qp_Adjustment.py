from pathlib import Path
import datetime as dt
import calendar
import re

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from auth import check_auth
from menu import render_sidebar

# ========================================
# ===== Autentikasi Basic Auth ===========
# ========================================
if not check_auth():
    st.stop()
render_sidebar()

def _parse_golongan_no_simple(g) -> int | None:
    """文字列から Golongan 番号（1/2/3）だけを取り出す簡易関数。"""
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

# =========================
# 共通テーブル描画ヘルパー
# =========================
def render_simple_table(
    df: pd.DataFrame,
    header_bg: str = "#e5f0ff",
    caption: str | None = None,
    col_width: str | None = None,
    row_highlight=None,
) -> None:
    """
    Render a compact, bordered table (pandas Styler) inside Streamlit.
    """
    if caption:
        st.caption(caption)
    if df is None or df.empty:
        st.write("(no data)")
        return

    col_heading_props = [
        ("background-color", header_bg),
        ("font-weight", "bold"),
        ("border", "1px solid #999"),
        ("padding", "2px 4px"),
        ("white-space", "nowrap"),
        ("position", "sticky"),
        ("top", "0"),
        ("z-index", "2"),
    ]
    td_props = [
        ("border", "1px solid #999"),
        ("padding", "2px 4px"),
        ("white-space", "nowrap"),
    ]
    if col_width is not None:
        col_heading_props.append(("min-width", col_width))
        col_heading_props.append(("max-width", col_width))
        td_props.append(("min-width", col_width))
        td_props.append(("max-width", col_width))

    styler = (
        df.style
        .set_properties(**{
            "text-align": "center",
            "padding": "2px 4px",
            "font-size": "16px",
            "white-space": "nowrap",
        })
        .set_table_styles([
            {"selector": "th.col_heading", "props": col_heading_props},
            {"selector": "th.row_heading",
             "props": [("background-color", "#f5f5f5"),
                       ("font-weight", "bold"),
                       ("border", "1px solid #999"),
                       ("padding", "2px 4px"),
                       ("white-space", "nowrap")]},
            {"selector": "th.blank",
             "props": [("border", "1px solid #999"),
                       ("padding", "2px 4px")]},
            {"selector": "td", "props": td_props},
        ])
    )

    if row_highlight is not None:
        styler = styler.apply(row_highlight, axis=1)

    html = (
        '<div style="overflow-x:auto; text-align:left;">'
        f'{styler.to_html()}'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_adjusted_qp_table(
    df: pd.DataFrame,
    mt_start_labels: set[str],
    caption: str | None = None,
) -> None:
    """
    Adjusted Qp シナリオ用のテーブル描画（3番の見た目に揃えた版）。
    """
    if caption:
        st.caption(caption)
    if df is None or df.empty:
        st.write("(no data)")
        return

    cols = list(df.columns)
    fmt_dict: dict[str, object] = {}

    # Area (ha): 整数表示（NaN / 0 は空欄）
    if "Area (ha)" in cols:
        fmt_dict["Area (ha)"] = (
            lambda v: "" if (pd.isna(v) or abs(v) < 1e-9) else f"{v:,.0f}"
        )

    # Golongan 1/2/3: 整数表示（NaN / 0 は空欄）
    for gcol in ["Golongan 1", "Golongan 2", "Golongan 3"]:
        if gcol in cols:
            fmt_dict[gcol] = (
                lambda v, gcol=gcol:
                    "" if (pd.isna(v) or abs(v) < 1e-9) else f"{v:,.0f}"
            )

    # Qp 列: 小数点3桁（NaN / 0 は空欄）
    qp_cols: list[str] = []
    for lab in labels:
        if lab in cols:
            qp_cols.append(lab)
            fmt_dict[lab] = (
                lambda v, lab=lab:
                    "" if (pd.isna(v) or abs(v) < 1e-9) else f"{v:.3f}"
            )

    # TB / Division 色付け
    def _highlight_row(row: pd.Series) -> list[str]:
        cls = str(row.get("Class", "")).strip()
        styles: list[str] = []

        text_cols = {
            "Bank", "Canal",
            "ParentName", "ChildName",
            "Class", "SectionName",
            "Area (ha)", "Golongan 1", "Golongan 2", "Golongan 3",
        }
        branch_green_cols = {"ChildName", "SectionName", "Area (ha)"}
        sec_name = str(row.get("SectionName", "")).strip()

        for col in row.index:
            bg = ""
            if col in text_cols:
                if cls == "TB":
                    bg = "#ffe6cc"
                # Sec01 のときは緑対象から除外
                elif cls == "Division" and col in branch_green_cols and sec_name != "Sec01":
                    bg = "#d9f99d"
            styles.append(f"background-color:{bg}" if bg else "")
        return styles

    styler = df.style.format(fmt_dict)

    # データバー（薄い赤）
    if qp_cols:
        vals = pd.to_numeric(df[qp_cols].stack(), errors="coerce")
        vals = vals.replace([np.inf, -np.inf], np.nan).dropna()
        if not vals.empty:
            vmin = float(vals.min())
            vmax = float(vals.max())
            if vmin != vmax:
                styler = styler.bar(
                    subset=pd.IndexSlice[:, qp_cols],
                    axis=0,
                    color="#ffc0cb",
                    vmin=vmin,
                    vmax=vmax,
                )

    # TB / Division
    styler = styler.apply(_highlight_row, axis=1)

    # MT開始列（MT-1/2/3 の 5Day_ID 列）をオレンジ
    if mt_start_labels:
        styler = styler.set_properties(
            subset=[lab for lab in mt_start_labels if lab in df.columns],
            **{"background-color": "#ffcc66"}
        )

    # 今日列（薄黄＋太字）
    today = dt.date.today()
    t_idx = date_to_5day_index(today)
    today_label = labels[t_idx] if 0 <= t_idx < len(labels) else None
    if today_label is not None and today_label in df.columns:
        styler = styler.set_properties(
            subset=[today_label],
            **{"background-color": "#fff2a8", "font-weight": "bold"}
        )

    # 共通セルスタイル
    styler = styler.set_properties(**{
        "text-align": "center",
        "padding": "3px 5px",
        "font-size": "13px",
        "white-space": "nowrap",
    })

    # Qp 列を太字
    if qp_cols:
        styler = styler.set_properties(
            subset=qp_cols,
            **{"font-weight": "bold"}
        )

    # Bank / Canal / ParentName / ChildName / Area (ha) も太字
    strong_cols = [c for c in ["Bank", "Canal", "ParentName", "ChildName", "Area (ha)"] if c in df.columns]
    if strong_cols:
        styler = styler.set_properties(
            subset=strong_cols,
            **{"font-weight": "bold"}
        )

    # ヘッダー
    styler = styler.set_table_styles([
        {
            "selector": "th.col_heading",
            "props": [
                ("background-color", "#ffe4e6"),
                ("font-weight", "bold"),
                ("font-size", "12px"),
                ("border", "1px solid #999"),
                ("padding", "2px 4px"),
                ("white-space", "nowrap"),
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

    # 列幅を Planned テーブルと共通化
    styler = apply_common_col_widths(styler, df)

    html = (
        '<div style="overflow-x:auto; text-align:left; max-height:420px; overflow-y:auto;">'
        f'{styler.to_html()}'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)

    # ★ Adjusted Qp 用の凡例をテーブル直下に表示
    st.markdown(
        """
        <div style="margin-top:4px; font-size:0.9rem;">
          <div style="display:flex; flex-wrap:wrap; gap:16px; align-items:center;">
            <div>
              <span style="display:inline-block;width:14px;height:14px;
                           background-color:#ffcc66;border:1px solid #999;
                           margin-right:6px;vertical-align:middle;"></span>
              <strong>Orange</strong>: Start 5Day_ID of <strong>MT-1 / MT-2 / MT-3</strong> after adjustment
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


def recompute_nfr_5day_for_landprep(landprep_case: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """
    Recompute NFR 5-day table (Bank, Golongan, Group, Season, Month, Step, NFR[L/s/ha])
    from a given land-preparation schedule (landprep_case).

    Returns:
        (nfr_long_case, comp_long_case)

        nfr_long_case: DataFrame
            ["Bank", "Golongan", "Group", "Season", "Month", "Step", "NFR"]

        comp_long_case: DataFrame
            ["Bank","Golongan","Group","Season","Month","Step",
             "LP","P","WLr","ETc","Re","NFR_mm","NFR_lps"]
    """

    # ---------- 0. Prerequisites from Page2 ----------
    try:
        base_dir = BASE_DIR
        csv_dir  = CSV_DIR
    except NameError:
        st.error("BASE_DIR / CSV_DIR are not defined in this page.")
        return None

    thiessen_path = csv_dir / "Thiessen.csv"
    if not thiessen_path.exists():
        st.error(f"`Thiessen.csv` not found.\nPath: {thiessen_path}")
        return None

    th = pd.read_csv(thiessen_path)

    if "Golongan" not in th.columns and "Prop_gol" in th.columns:
        th = th.rename(columns={"Prop_gol": "Golongan"})
    if "Nama_Stasi" not in th.columns and "Nama_Stati" in th.columns:
        th = th.rename(columns={"Nama_Stati": "Nama_Stasi"})

    th["Bank"]     = th["Bank"].astype(str).str.strip()
    th["Golongan"] = th["Golongan"].astype(str).str.strip()
    th["group"]    = th["Bank"] + "_" + th["Golongan"]

    groups = sorted(th["group"].unique())

    th_unique = th.drop_duplicates("group")[["group", "Golongan"]]

    def extract_gol_idx(text: str) -> int:
        m = re.search(r"\d+", text)
        return int(m.group()) if m else 1

    th_unique["Gol_idx"] = th_unique["Golongan"].map(extract_gol_idx)
    group_to_gol_idx = th_unique.set_index("group")["Gol_idx"].to_dict()

    bank_map      = th.drop_duplicates("group")[["group", "Bank"]]
    group_to_bank = bank_map.set_index("group")["Bank"].to_dict()

    # Basic / Kc / ETo / Re / LP from Page2
    basic_edited = st.session_state.get("basic_edited", None)
    kc_edited    = st.session_state.get("kc_edited", None)
    eto_df       = st.session_state.get("eto_df", None)
    lp_df        = st.session_state.get("lp_df", None)
    re_paddy     = st.session_state.get("re_paddy", None)
    re_palawija  = st.session_state.get("re_palawija", None)

    if any(x is None for x in [basic_edited, kc_edited, eto_df, lp_df, re_paddy, re_palawija]):
        st.warning(
            "Basic / Kc / ETo / LP / Re are not ready in session_state. "
            "Please open Page2 and complete the NFR calculation first."
        )
        return None

    # ---------- 1. Shared parameters (same as Page2 section 7) ----------
    basic_vals = basic_edited.set_index("Variable")["Value"]

    section_n = int(basic_vals.get("Section", 1))
    section_n = max(section_n, 1)

    durlp_days    = float(basic_vals.get("DurLP", 20))
    secshift_days = float(basic_vals.get("SecShift", 5))
    unitP         = float(basic_vals.get("UnitP", 2.0))
    durPmt1_days  = float(basic_vals.get("DurPmt1", 95))
    durPmt2_days  = float(basic_vals.get("DurPmt2", 90))
    unitWLr       = float(basic_vals.get("UnitWLr", 1.67))

    kc_vals = kc_edited.set_index("Variable")["Value"]
    Kc1 = float(kc_vals.get("Kc1", 1.10))
    Kc2 = float(kc_vals.get("Kc2", 1.05))
    Kc3 = float(kc_vals.get("Kc3", 0.95))

    Kc1_mt3 = float(kc_vals.get("Kc1_mt3", 0.50))
    Kc2_mt3 = float(kc_vals.get("Kc2_mt3", 0.59))
    Kc3_mt3 = float(kc_vals.get("Kc3_mt3", 0.96))
    Kc4_mt3 = float(kc_vals.get("Kc4_mt3", 1.05))
    Kc5_mt3 = float(kc_vals.get("Kc5_mt3", 1.02))
    Kc6_mt3 = float(kc_vals.get("Kc6_mt3", 0.95))

    # LP base: 5Day_ID -> LP(mm/day)
    lp_base = lp_df.copy()
    lp_base["5Day_ID"] = eto_df["5Day_ID"].to_numpy()
    lp_base = lp_base.set_index("5Day_ID")["LP"]

    ids     = lp_base.index.to_numpy()
    n_steps = len(lp_base)

    season_index_ids = (ids - 1) // 6
    months = ((season_index_ids + 10) % 12) + 1
    steps  = ((ids - 1) % 6) + 1
    col_multi = pd.MultiIndex.from_arrays(
        [months, steps],
        names=["Month", "Step"],
    )

    STEP_DAYS     = 5.0
    durLP_steps   = max(1, int(round(durlp_days / STEP_DAYS)))
    offset_steps  = max(1, int(round(secshift_days / STEP_DAYS)))
    ETo_arr       = eto_df["ETo"].to_numpy()

    # ---------- 2. Helper: Kc by DAT ----------
    def kc_for_dat(season: str, dat_mid: float) -> float:
        if season in ("MT-1", "MT-2"):
            if 1 <= dat_mid <= 30:
                return Kc1
            elif 31 <= dat_mid <= 60:
                return Kc2
            elif season == "MT-1" and 61 <= dat_mid <= 80:
                return Kc3
            elif season == "MT-2" and 61 <= dat_mid <= 75:
                return Kc3
            else:
                return 0.0
        else:  # MT-3
            if 1 <= dat_mid <= 15:
                return Kc1_mt3
            elif 16 <= dat_mid <= 30:
                return Kc2_mt3
            elif 31 <= dat_mid <= 45:
                return Kc3_mt3
            elif 46 <= dat_mid <= 55:
                return Kc4_mt3
            elif 56 <= dat_mid <= 65:
                return Kc5_mt3
            elif 66 <= dat_mid <= 70:
                return Kc6_mt3
            else:
                return 0.0

    seasons_all = ["MT-1", "MT-2", "MT-3"]

    # ---------- 3. Distributions per season (using landprep_case) ----------
    def compute_distributions_for_season(season: str):
        if season == "MT-1":
            start_col = "MT1_start"
            durP_days = durPmt1_days
            max_DAT   = 80
        elif season == "MT-2":
            start_col = "MT2_start"
            durP_days = durPmt2_days
            max_DAT   = 75
        else:  # MT-3
            start_col = "MT3_start"
            durP_days = 0.0
            max_DAT   = 70

        durP_steps  = max(0, int(round(durP_days / STEP_DAYS)))
        durWL_days  = 60.0
        durWL_steps = max(1, int(round(durWL_days / STEP_DAYS)))

        active_LP  = {g: np.zeros(n_steps, dtype=int) for g in groups}
        active_P   = {g: np.zeros(n_steps, dtype=int) for g in groups}
        active_WLr = {g: np.zeros(n_steps, dtype=int) for g in groups}
        ETc_dist   = {g: np.zeros(n_steps, dtype=float) for g in groups}

        # MT-3 active / days
        if "MT3_active" in landprep_case.columns:
            mt3_active_map = dict(zip(landprep_case["Group"], landprep_case["MT3_active"]))
        else:
            mt3_active_map = {g: True for g in groups}

        if "MT3_days" in landprep_case.columns:
            mt3_days_map = dict(zip(landprep_case["Group"], landprep_case["MT3_days"]))
        else:
            mt3_days_map = {g: 70 for g in groups}

        for _, row in landprep_case.iterrows():
            g       = row["Group"]
            gol_idx = group_to_gol_idx.get(g, 1)

            if season == "MT-3":
                active_flag = mt3_active_map.get(g, True)
                if gol_idx == 3 or not active_flag:
                    continue
                max_DAT_local = float(mt3_days_map.get(g, 70))
            else:
                max_DAT_local = max_DAT

            start_date = row[start_col]
            base_id    = date_to_5day_id(start_date)  # 1..72
            base_index = base_id - 1                  # 0..71

            act_LP  = active_LP[g]
            act_P   = active_P[g]
            act_WLr = active_WLr[g]
            etc_vec = ETc_dist[g]

            for sec in range(section_n):
                if season in ("MT-1", "MT-2"):
                    # LP
                    start_idx_lp = base_index + sec * offset_steps
                    for k in range(durLP_steps):
                        raw_idx = start_idx_lp + k
                        if raw_idx >= n_steps:
                            break
                        act_LP[raw_idx] += 1

                    last_lp_idx = start_idx_lp + durLP_steps - 1
                    if last_lp_idx >= n_steps:
                        last_lp_idx = n_steps - 1

                    # P
                    if durP_steps > 0:
                        for k in range(durP_steps):
                            raw_idx = last_lp_idx + k
                            if raw_idx >= n_steps:
                                break
                            act_P[raw_idx] += 1

                    # WLr
                    for k in range(durWL_steps):
                        raw_idx = last_lp_idx + 1 + k
                        if raw_idx >= n_steps:
                            break
                        act_WLr[raw_idx] += 1

                    # ETc
                    j = 0
                    while True:
                        dat_mid = j * 5 + 3
                        if dat_mid > max_DAT_local:
                            break
                        raw_idx = last_lp_idx + j
                        if raw_idx >= n_steps:
                            break
                        idx_e = raw_idx

                        Kc_val = kc_for_dat(season, dat_mid)
                        if Kc_val > 0:
                            etc_val = Kc_val * ETo_arr[idx_e]
                            etc_vec[idx_e] += etc_val / float(section_n)
                        j += 1

                else:  # MT-3: only ETc
                    start_idx_etc = base_index + sec * offset_steps
                    j = 0
                    while True:
                        dat_mid = j * 5 + 3
                        if dat_mid > max_DAT_local:
                            break
                        raw_idx = start_idx_etc + j
                        idx_e   = raw_idx % n_steps

                        Kc_val = kc_for_dat(season, dat_mid)
                        if Kc_val > 0:
                            etc_val = Kc_val * ETo_arr[idx_e]
                            etc_vec[idx_e] += etc_val / float(section_n)
                        j += 1

        base_vals = lp_base.to_numpy()
        LP_dist   = {}
        P_dist    = {}
        WLr_dist  = {}

        for g in groups:
            frac_LP  = active_LP[g]  / float(section_n)
            frac_P   = active_P[g]   / float(section_n)
            frac_WLr = active_WLr[g] / float(section_n)

            LP_dist[g]  = base_vals * frac_LP
            P_dist[g]   = unitP     * frac_P
            WLr_dist[g] = unitWLr   * frac_WLr

        dist_LP_table = pd.DataFrame(
            [LP_dist[g] for g in groups],
            index=groups,
            columns=col_multi,
        )
        dist_P_table = pd.DataFrame(
            [P_dist[g] for g in groups],
            index=groups,
            columns=col_multi,
        )
        dist_WLr_table = pd.DataFrame(
            [WLr_dist[g] for g in groups],
            index=groups,
            columns=col_multi,
        )
        dist_ETc_table = pd.DataFrame(
            [ETc_dist[g] for g in groups],
            index=groups,
            columns=col_multi,
        )

        return dist_LP_table, dist_P_table, dist_WLr_table, dist_ETc_table

    # distributions for all seasons
    LP_tables   = {}
    P_tables    = {}
    WLr_tables  = {}
    ETc_tables  = {}

    for season in seasons_all:
        LP_tables[season], P_tables[season], WLr_tables[season], ETc_tables[season] = \
            compute_distributions_for_season(season)

    dist_LP_all  = pd.concat(LP_tables.values(),   keys=seasons_all, names=["Season", "Group"])
    dist_P_all   = pd.concat(P_tables.values(),    keys=seasons_all, names=["Season", "Group"])
    dist_WLr_all = pd.concat(WLr_tables.values(),  keys=seasons_all, names=["Season", "Group"])
    dist_ETc_all = pd.concat(ETc_tables.values(),  keys=seasons_all, names=["Season", "Group"])

    # ---------- 4. Build Re(mm/day) for this case (same style as Page2 section 8) ----------
    rows_re = []
    idx_list = []
    for season in seasons_all:
        for g in groups:
            gol_idx = group_to_gol_idx.get(g, 1)

            if season in ("MT-1", "MT-2"):
                src = re_paddy
                if g in src.index:
                    rows_re.append(src.loc[g])
                else:
                    rows_re.append(pd.Series(0.0, index=src.columns))
            else:
                # MT-3
                active_flag = bool(
                    landprep_case.loc[landprep_case["Group"] == g, "MT3_active"].iloc[0]
                ) if "MT3_active" in landprep_case.columns and g in list(landprep_case["Group"]) else True

                if gol_idx == 3 or not active_flag:
                    rows_re.append(pd.Series(0.0, index=re_palawija.columns))
                else:
                    src = re_palawija
                    if g in src.index:
                        rows_re.append(src.loc[g])
                    else:
                        rows_re.append(pd.Series(0.0, index=src.columns))

            idx_list.append((season, g))

    re_all = pd.DataFrame(
        rows_re,
        index=pd.MultiIndex.from_tuples(idx_list, names=["Season", "Group"]),
        columns=re_palawija.columns,
    )
    re_all = re_all.reindex(columns=col_multi)

    # ---------- 5. NFR(mm/day) & L/s/ha ----------
    total_req = dist_LP_all.add(dist_P_all,   fill_value=0) \
                           .add(dist_WLr_all, fill_value=0) \
                           .add(dist_ETc_all, fill_value=0)

    NFR_mm  = (total_req - re_all).clip(lower=0)
    MM_TO_LPS_PER_HA = 10000.0 / 86400.0
    NFR_lps = NFR_mm * MM_TO_LPS_PER_HA

    # ---------- 6. Long-form NFR table ----------
    nfr_rows = []
    for (season, group), row_mm in NFR_mm.iterrows():
        for (month, step), _ in row_mm.items():
            nfr_rows.append({
                "Season":   season,
                "Group":    group,
                "Month":    int(month),
                "Step":     int(step),
                "NFR":      float(NFR_lps.loc[(season, group), (month, step)]),
            })
    nfr_long = pd.DataFrame(nfr_rows)
    nfr_long["Bank"]     = nfr_long["Group"].map(group_to_bank).fillna("")
    nfr_long["Golongan"] = nfr_long["Group"].map(group_to_gol_idx)

    nfr_long = nfr_long[["Bank", "Golongan", "Group",
                         "Season", "Month", "Step", "NFR"]]

    # ---------- 7. Component breakdown (LP, P, WLr, ETc, Re, NFR) ----------
    comp_rows = []
    for (season, group), _ in dist_LP_all.iterrows():
        bank = group_to_bank.get(group, "")
        gol  = group_to_gol_idx.get(group, None)

        lp_row   = dist_LP_all.loc[(season, group)]
        p_row    = dist_P_all.loc[(season, group)]
        wlr_row  = dist_WLr_all.loc[(season, group)]
        etc_row  = dist_ETc_all.loc[(season, group)]
        re_row   = re_all.loc[(season, group)]
        nfr_mm_row  = NFR_mm.loc[(season, group)]
        nfr_lps_row = NFR_lps.loc[(season, group)]

        for (month, step) in col_multi:
            comp_rows.append({
                "Bank":     bank,
                "Golongan": gol,
                "Group":    group,
                "Season":   season,
                "Month":    int(month),
                "Step":     int(step),
                "LP":       float(lp_row[(month, step)]),
                "P":        float(p_row[(month, step)]),
                "WLr":      float(wlr_row[(month, step)]),
                "ETc":      float(etc_row[(month, step)]),
                "Re":       float(re_row[(month, step)]),
                "NFR_mm":   float(nfr_mm_row[(month, step)]),
                "NFR_lps":  float(nfr_lps_row[(month, step)]),
            })

    comp_long = pd.DataFrame(comp_rows)

    return nfr_long, comp_long

def render_nfr_pivot_base_vs_case(
    nfr_base: pd.DataFrame,
    nfr_case: pd.DataFrame,
    bank_sel: str,
    gol_sel: int,
    landprep_base: pd.DataFrame,
    landprep_case: pd.DataFrame,
) -> None:
    """
    Bank×Golongan に対する NFR (Base vs Case) pivot を表示し、
    各行の「設定された MT 開始日の 5Day_ID」に対応するセルをオレンジにハイライトする。
    """

    # ---- 対象 Bank×Golongan の行だけ抽出 ----
    base_view = nfr_base[
        (nfr_base["Bank"] == bank_sel) &
        (nfr_base["Golongan"] == gol_sel)
    ].copy()
    case_view = nfr_case[
        (nfr_case["Bank"] == bank_sel) &
        (nfr_case["Golongan"] == gol_sel)
    ].copy()

    if base_view.empty or case_view.empty:
        st.info(
            f"No NFR rows for Bank={bank_sel}, Golongan={gol_sel} "
            "in either base or scenario table."
        )
        return

    base_view = base_view[["Season", "Month", "Step", "NFR"]].rename(
        columns={"NFR": "NFR_base"}
    )
    case_view = case_view[["Season", "Month", "Step", "NFR"]].rename(
        columns={"NFR": "NFR_case"}
    )

    cmp = (
        base_view
        .merge(case_view, on=["Season", "Month", "Step"], how="outer")
        .sort_values(["Season", "Month", "Step"], ignore_index=True)
    )

    # 5Day_ID はデバッグ用
    def _month_step_to_id(m, s):
        try:
            m_idx = water_month_order.index(int(m))
        except Exception:
            return None
        try:
            stp = int(s)
        except Exception:
            return None
        return m_idx * 6 + stp  # 1..72

    cmp["5Day_ID"] = cmp.apply(
        lambda r: _month_step_to_id(r["Month"], r["Step"]), axis=1
    )

    # Long → pivot: rows = Season×Scenario, cols = Month×Step, value = NFR
    cmp_long = cmp.melt(
        id_vars=["Season", "Month", "Step"],
        value_vars=["NFR_base", "NFR_case"],
        var_name="Scenario",
        value_name="NFR",
    )
    cmp_long["Scenario"] = cmp_long["Scenario"].map(
        {"NFR_base": "Base", "NFR_case": "Case"}
    )

    pivot = (
        cmp_long
        .pivot_table(
            index=["Season", "Scenario"],
            columns=["Month", "Step"],
            values="NFR",
            aggfunc="mean",
        )
        .sort_index()
    )

    # 列順：水年順 (Nov,Dec,Jan..Oct) × Step(1..6)
    def _col_sort_key(col):
        m, s = col
        try:
            m_idx = water_month_order.index(int(m))
        except Exception:
            m_idx = 99
        try:
            step = int(s)
        except Exception:
            step = 99
        return (m_idx, step)

    pivot = pivot.reindex(
        columns=sorted(pivot.columns, key=_col_sort_key)
    )

    # ---- Bank×Golongan×Season ごとの開始 5Day_ID を Base / Case それぞれ求める ----
    def _start_id_for_bank_gol(lp_df, bank, gol, season):
        """
        landprep_df から Bank×Golongan×Season の開始日を取り、
        5Day_ID(1..72) に変換して返す。
        同じ Bank×Golongan 内に複数行がある場合は、最も早い日付を採用。
        """
        if not isinstance(lp_df, pd.DataFrame) or lp_df.empty:
            return None

        col_map = {"MT-1": "MT1_start", "MT-2": "MT2_start", "MT-3": "MT3_start"}
        col = col_map.get(season)
        if col not in lp_df.columns:
            return None

        def _split_group(s: str):
            s = str(s)
            if "_" not in s:
                return None, None
            bank_part, rest = s.split("_", 1)
            m = re.search(r"(\d+)", rest)
            gol_idx = int(m.group(1)) if m else None
            return bank_part, gol_idx

        mask = lp_df["Group"].astype(str).apply(
            lambda s: _split_group(s) == (bank, gol)
        )
        if not mask.any():
            return None

        # MT-3 は active 行だけを見る
        if season == "MT-3" and "MT3_active" in lp_df.columns:
            try:
                active = lp_df["MT3_active"].astype(bool)
                mask = mask & active
            except Exception:
                pass

        vals = lp_df.loc[mask, col].dropna()
        if vals.empty:
            return None

        try:
            dates = pd.to_datetime(vals).dt.date
        except Exception:
            return None

        d_min = dates.min()
        return date_to_5day_id(d_min)

    def _id_to_month_step(id_):
        """5Day_ID(1..72) → (Month, Step)"""
        if id_ is None or pd.isna(id_):
            return None
        try:
            id_int = int(id_)
        except Exception:
            return None
        if not (1 <= id_int <= 72):
            return None
        i0 = id_int - 1
        m = water_month_order[i0 // 6]
        s = (i0 % 6) + 1
        return (m, s)

    seasons_in_pivot = sorted(set(pivot.index.get_level_values("Season")))
    start_id_base_map = {
        season: _start_id_for_bank_gol(landprep_base, bank_sel, gol_sel, season)
        for season in seasons_in_pivot
    }
    start_id_case_map = {
        season: _start_id_for_bank_gol(landprep_case, bank_sel, gol_sel, season)
        for season in seasons_in_pivot
    }

    # 各行 (Season, Scenario) ごとに「この行で塗るべき (Month, Step)」を求める
    start_col_old = {}
    for old_idx in pivot.index:
        season, scenario = old_idx
        if scenario == "Base":
            id_ = start_id_base_map.get(season)
        else:  # Case
            id_ = start_id_case_map.get(season)
        ms = _id_to_month_step(id_)
        if ms is not None and ms in pivot.columns:
            start_col_old[old_idx] = ms

    # 行ラベルを "MT-1 – Base" 形式に変更して start_col_map を作る
    new_index_labels = [f"{idx[0]} – {idx[1]}" for idx in pivot.index]
    start_col_map = {
        new_label: start_col_old.get(old_idx)
        for old_idx, new_label in zip(pivot.index, new_index_labels)
    }
    pivot.index = new_index_labels
    pivot.columns = pd.MultiIndex.from_tuples(
        pivot.columns, names=["Month", "Step"]
    )

    # 表示用フォーマット：0 → 空文字、それ以外は小数3桁
    def _fmt(v):
        if pd.isna(v) or abs(v) < 1e-9:
            return ""
        return f"{v:.3f}"

    pivot_disp = pivot.copy().map(_fmt)

    # 行ごとに「start_col_map で指定された列だけオレンジ」
    def _highlight_start(row: pd.Series):
        idx_label = row.name  # "MT-1 – Base" など
        target_col = start_col_map.get(idx_label)
        styles: list[str] = []
        for col in row.index:
            if target_col is not None and col == target_col:
                styles.append("background-color:#ffcc66;")
            else:
                styles.append("")
        return styles

    render_simple_table(
        pivot_disp,
        header_bg="#e5f0ff",
        caption=(
            f"Base vs scenario NFR (L/s/ha) – Bank={bank_sel}, "
            f"Golongan={gol_sel}"
        ),
        col_width="46px",
        row_highlight=_highlight_start,
    )


# ========================================
# Common: 5-day settings
# ========================================
BASE_DIR = Path(__file__).resolve().parent.parent
CSV_DIR = BASE_DIR / "csv"

# 5Day_ID always starts from November (water year)
water_month_order = [11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# ★追加: "Nov","Dec","Jan",...,"Oct" をチャートの並び順に使う
month_order_abbr = [calendar.month_abbr[m] for m in water_month_order]

def build_5day_labels() -> list[str]:
    labels = []
    for i in range(72):
        m = water_month_order[i // 6]
        step = (i % 6) + 1
        labels.append(f"{calendar.month_abbr[m]}-{step}")
    return labels


def date_to_5day_index(d: dt.date) -> int:
    """Convert a calendar date to 0-based 5Day index (0..71)."""
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
    return m_idx * 6 + (step - 1)


def date_to_5day_id(d: dt.date) -> int:
    """
    Convert a calendar date to a 1-based 5Day_ID (1..72)
    for a water year starting in November (11, 12, 1..10).

    This uses the same convention as the NFR calculation on Page2.
    """
    m = d.month
    try:
        m_idx = water_month_order.index(m)
    except ValueError:
        m_idx = 0  # 念のため

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

    return m_idx * 6 + step  # 1..72

def _five_day_id_to_date_from_base(base_date: dt.date, id_5d: int) -> dt.date | None:
    """
    5Day_ID (1..72) を、base_date と同じ water-year 上の日付に変換する。

    11,12月 → water-year 開始年、それ以外 → 開始年+1年。
    """
    if id_5d is None or id_5d < 1 or id_5d > 72:
        return None

    idx0 = id_5d - 1
    month = water_month_order[idx0 // 6]   # 11,12,1..10
    step  = (idx0 % 6) + 1                 # 1..6

    # base_date から water-year の開始年を決める
    if base_date.month >= 11:
        wy_start = base_date.year
    else:
        wy_start = base_date.year - 1

    year = wy_start if month in (11, 12) else wy_start + 1

    # その 5-day step の先頭日（1,6,11,...）
    day = (step - 1) * 5 + 1
    last_day = calendar.monthrange(year, month)[1]
    if day > last_day:
        day = last_day

    return dt.date(year, month, day)

def shift_date_by_5day_steps_str(base, adj_steps) -> str:
    """
    base      : MT*_start の元の日付（Timestamp / datetime / date / 文字列）
    adj_steps : 5Day_ID の増減（整数）

    - base を 5Day_ID(1..72) に変換
    - 5Day_ID を adj_steps 分だけ足し引き（1..72 にクリップ）
    - 同じ water-year 上の日付に戻して 'yy-mm-dd' で返す
    """
    if base is None or pd.isna(base):
        return ""

    try:
        base_date = pd.to_datetime(base).date()
    except Exception:
        return ""

    # 1..72 の 5Day_ID に変換
    base_id = date_to_5day_id(base_date)
    if not (1 <= base_id <= 72):
        return ""

    # 5Day_ID ベースでシフト
    new_id = base_id + int(adj_steps)
    new_id = max(1, min(72, new_id))

    # water-year 内のカレンダー日付に戻す
    new_date = _five_day_id_to_date_from_base(base_date, new_id)
    if new_date is None:
        return ""

    return new_date.strftime("%y-%m-%d")


def build_nfr_arr_from_long(nfr_long: pd.DataFrame) -> dict[str, dict[int, np.ndarray]]:
    """
    Page2/5 の NFR 5-day テーブル
      (Bank, Golongan, Season, Month, Step, NFR)
    から、Page4 と同じ形式の
      nfr_arr[Bank][Golongan] = 長さ72の配列
    を作る。

    ※ Season はここで集約してしまう（Bank×Golongan×Month×Step で合算/平均）。
    """

    n_steps = len(labels)
    banks = ["Left", "Right"]
    arr = {b: {g: np.zeros(n_steps, dtype=float) for g in [1, 2, 3]} for b in banks}

    if nfr_long is None or nfr_long.empty:
        return arr

    df = nfr_long.copy()

    # 列の正規化
    df["Bank"] = df["Bank"].astype(str).str.strip()
    df["Golongan"] = df["Golongan"].apply(_parse_golongan_no_simple)
    df["Month"] = pd.to_numeric(df["Month"], errors="coerce").astype("Int64")
    df["Step"]  = pd.to_numeric(df["Step"],  errors="coerce").astype("Int64")
    df["NFR"]   = pd.to_numeric(df["NFR"],   errors="coerce").fillna(0.0)

    # 1,2,3 以外のゴロンガンは除外
    df = df[df["Golongan"].isin([1, 2, 3])].copy()

    # Page4 と同じ：「0 は無視して平均」を使った集約関数
    def agg_nfr_ignore_zero(series: pd.Series) -> float:
        pos = series[series > 1e-9]
        if pos.empty:
            return 0.0
        return float(pos.mean())

    # ★ Season や Group は無視して、Bank×Golongan×Month×Step で集約
    df_aggr = (
        df
        .groupby(["Bank", "Golongan", "Month", "Step"], as_index=False)
        .agg(NFR=("NFR", agg_nfr_ignore_zero))
    )

    # 配列に書き込む
    for _, r in df_aggr.iterrows():
        bank = str(r["Bank"])
        if bank not in arr:
            continue

        g = int(r["Golongan"])
        if g not in arr[bank]:
            continue

        m = int(r["Month"])
        s = int(r["Step"])
        if m not in water_month_order or not (1 <= s <= 6):
            continue

        m_idx = water_month_order.index(m)
        idx = m_idx * 6 + (s - 1)
        if 0 <= idx < n_steps:
            arr[bank][g][idx] = float(r["NFR"])

    return arr

# ========================================
# Read MT-1 / MT-2 / MT-3 start (Bank × Golongan) from Page2
# ========================================
def build_mt_start_map_from_page2():
    """
    Build (Bank, Golongan, Season) -> 0-based 5Day index (0..71)
    from Page2's season start dates.

    Seasons: "MT-1", "MT-2", "MT-3"
    Golongan is parsed from Group like "Left_Golongan 1", "Right_Golongan 3", etc.
    """
    mt_start_map = {}  # key: (bank, gol_idx, season) -> idx(0..71)

    lp = st.session_state.get("landprep_df", None)

    # Prefer landprep_df if available
    if isinstance(lp, pd.DataFrame) and not lp.empty and \
            "Group" in lp.columns and \
            "MT1_start" in lp.columns and \
            "MT2_start" in lp.columns and \
            "MT3_start" in lp.columns:
        for _, r in lp.iterrows():
            grp = str(r["Group"])
            parts = grp.split("_", 1)
            if len(parts) != 2:
                continue
            bank = parts[0].strip()
            rest = parts[1]

            m = re.search(r"\d+", rest)
            if not m:
                continue
            gol_idx = int(m.group())

            for season, col in [
                ("MT-1", "MT1_start"),
                ("MT-2", "MT2_start"),
                ("MT-3", "MT3_start"),
            ]:
                # ★ MT-3 is only valid when MT3_active is True (if the column exists)
                if season == "MT-3" and "MT3_active" in lp.columns:
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

                if isinstance(v, str):
                    try:
                        d = pd.to_datetime(v).date()
                    except Exception:
                        continue
                elif isinstance(v, (dt.date, dt.datetime, pd.Timestamp)):
                    d = pd.to_datetime(v).date()
                else:
                    continue

                idx = date_to_5day_index(d)
                if 0 <= idx < 72:
                    key = (bank, gol_idx, season)
                    if key not in mt_start_map:
                        mt_start_map[key] = idx


    # If nothing from landprep_df, fallback to common mt*_common values
    if not mt_start_map:
        mt1_common = st.session_state.get("mt1_common", None)
        mt2_common = st.session_state.get("mt2_common", None)
        mt3_common = st.session_state.get("mt3_common", None)

        def common_idx(v):
            if v is None or pd.isna(v):
                return None
            if isinstance(v, str):
                try:
                    d = pd.to_datetime(v).date()
                except Exception:
                    return None
            elif isinstance(v, (dt.date, dt.datetime, pd.Timestamp)):
                d = pd.to_datetime(v).date()
            else:
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


# ========================================
# CanalKey parsing and canal-edge relations (same logic as Page4)
# ========================================
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


def build_canal_edges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Same idea as Page4:
    For Class='Division' rows, build ParentCanal -> ChildCanal relations.
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

    div_df = df_loc[df_loc["Class_upper"] == "DIVISION"].copy()

    for idx, r in div_df.iterrows():
        child_canal = r["Canal"]
        if not child_canal or child_canal.lower() == "nan":
            continue

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

def compute_qp_from_nfr_for_df(
    df_net: pd.DataFrame,
    nfr_arr: dict[str, dict[int, np.ndarray]],
    eff_map_raw: dict[str, float] | None,
) -> pd.DataFrame:
    """
    Page4 の Qp 計算ロジックと同じことを、
    任意の df_net ＋ nfr_arr ＋ Eff で実行するエンジン。

    df_net:
        Page4 の df と同じ構造を想定
        (ParentKey, ChildKey, Class, Area, Golongan_TB, CanalLevel, Bank, ...)

    nfr_arr:
        build_nfr_arr_from_long で作った
        {Bank -> {Golongan -> 長さ72(NFR[L/s/ha])}} の辞書。

    eff_map_raw:
        Page4 の eff_edited から作った map
        {"L_M":0.9, "L_S":0.9, "L_T":0.8, ...}
        None の場合はデフォルト値(0.9/0.8)を使用。
    """

    df = df_net.copy()
    n_rows = len(df)
    n_steps = len(labels)

    # --- Eff を取得するローカル関数 ---
    def get_eff_local(bank: str, level: str) -> float:
        if eff_map_raw is None:
            default = {
                ("Left", "M"): 0.90,
                ("Left", "S"): 0.90,
                ("Left", "T"): 0.80,
                ("Right", "M"): 0.90,
                ("Right", "S"): 0.90,
                ("Right", "T"): 0.80,
            }
            return default.get((bank, level), 1.0)
        if bank == "Left":
            key = f"L_{level}"
        else:
            key = f"R_{level}"
        return float(eff_map_raw.get(key, 1.0))

    # --- parent_map: child_key -> (parent_key, seg_idx) ---
    parent_map_local: dict[str, tuple[str, int]] = {}
    for idx, row in df.iterrows():
        child = str(row.get("ChildKey", "")).strip()
        parent = str(row.get("ParentKey", "")).strip()
        if child and child not in parent_map_local:
            parent_map_local[child] = (parent, idx)

    # --- 各レベルの流量配列 [L/s] ---
    flows_T = np.zeros((n_rows, n_steps), dtype=float)
    flows_S = np.zeros((n_rows, n_steps), dtype=float)
    flows_M = np.zeros((n_rows, n_steps), dtype=float)

    # TB 行だけ抽出
    tb_rows = df[df["Class"] == "TB"].copy()

    for idx, row in tb_rows.iterrows():
        bank = str(row.get("Bank", "")).strip()
        if bank not in nfr_arr:
            continue

        g_no = _parse_golongan_no_simple(row.get("Golongan_TB", ""))
        if g_no not in (1, 2, 3):
            continue

        try:
            area_tb = float(row.get("Area", 0.0))
        except Exception:
            area_tb = 0.0
        if area_tb <= 0:
            continue

        nfr_vec = nfr_arr[bank].get(g_no, None)
        if nfr_vec is None:
            continue

        # 田面需要 [L/s] → T レベル入口
        demand_vec = area_tb * nfr_vec
        eff_T = get_eff_local(bank, "T")
        if eff_T <= 0:
            continue
        q_T = demand_vec / eff_T  # T レベル入口流量

        # ① TB 行に T レベル流量を計上
        seg_tb_idx = idx
        flows_T[seg_tb_idx, :] += q_T

        # この TB の経路上の Secondary / Main セグメントを親方向にたどる
        sec_indices: list[int] = []
        main_indices: list[int] = []

        child_key = str(row.get("ChildKey", "")).strip()
        while child_key in parent_map_local:
            parent_key, seg_idx = parent_map_local[child_key]
            level = str(df.at[seg_idx, "CanalLevel"]).strip()

            if level == "Secondary":
                sec_indices.append(seg_idx)
            elif level == "Main":
                main_indices.append(seg_idx)

            child_key = parent_key

        # ② Secondary セグメント：T+S
        if sec_indices:
            eff_S = get_eff_local(bank, "S")
            if eff_S > 0:
                q_S = q_T / eff_S
                for seg_idx in sec_indices:
                    flows_S[seg_idx, :] += q_S

        # ③ Main セグメント：T+S+M or T+M
        if main_indices:
            eff_M = get_eff_local(bank, "M")
            if eff_M <= 0:
                continue

            if sec_indices:
                eff_S = get_eff_local(bank, "S")
                if eff_S <= 0:
                    continue
                q_M = q_T / (eff_S * eff_M)
            else:
                q_M = q_T / eff_M

            for seg_idx in main_indices:
                flows_M[seg_idx, :] += q_M

    # --- 最終 Qp[L/s] を行ごとに決定 ---
    flows_mat_lps = np.zeros((n_rows, n_steps), dtype=float)
    for i, row in df.iterrows():
        level = str(row.get("CanalLevel", "")).strip()
        cls = str(row.get("Class", "")).strip()

        if cls == "TB":
            flows_mat_lps[i, :] = flows_T[i, :]
        elif level == "Secondary":
            flows_mat_lps[i, :] = flows_S[i, :]
        elif level == "Main":
            flows_mat_lps[i, :] = flows_M[i, :]
        else:
            flows_mat_lps[i, :] = 0.0

    # --- df に 72 列を追加（単位は L/s のまま） ---
    for j, lab in enumerate(labels):
        df[lab] = flows_mat_lps[:, j]

    return df


# ========================================
# Title
# ========================================
st.title("Water Requirement Adjustment (Qp vs Adjusted Qp)")

st.markdown("""
This page compares **planned Qp** from Page4 with **shifted Qp** by 5-day steps.

- Top table: Planned Qp (fixed crop calendar)  
- Bottom table: Adjusted Qp (shifted sowing dates, per 5-day step)  
- Row visibility: use **Show TB / Show Branch Canal / Show Target Canal**  
- Orange cells: **start 5Day_ID of MT-1 / MT-2 / MT-3** (per row, based on Page2)
""")

# ========================================
# 1. Load Qp CSV
# ========================================
qp_csv_path = CSV_DIR / "Qp_all_reaches_m3s.csv"
if not qp_csv_path.exists():
    st.error(
        f"`{qp_csv_path.name}` not found.\n\n"
        "Please calculate Qp on Page4 and save the CSV into the `csv` folder first."
    )
    st.stop()

try:
    df_qp_raw = pd.read_csv(qp_csv_path)
except Exception as e:
    st.error(f"Failed to read Qp CSV: {e}")
    st.stop()

st.caption(f"Loaded Qp CSV: `{qp_csv_path.name}`  (rows: {len(df_qp_raw)})")

labels = build_5day_labels()
missing_labels = [lab for lab in labels if lab not in df_qp_raw.columns]
if missing_labels:
    st.error(
        "Qp CSV is missing some 5Day columns.\n"
        f"Missing labels (first few): {missing_labels[:10]}"
    )
    st.stop()

# Base dataframe, close to Page4's df structure
df = df_qp_raw.copy()

for col in [
    "ParentKey", "ChildKey", "ParentName", "ChildName",
    "Class", "Golongan_TB", "Canal", "CanalKey", "SectionName", "Bank"
]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()
    else:
        df[col] = ""

df["SectionName"] = df["SectionName"].replace("nan", "").fillna("")

df["CanalBank"] = ""
df["CanalLevel"] = ""
df["MainKey"] = ""
for idx, row in df.iterrows():
    bank, level, main_key = parse_canal_key(row["CanalKey"])
    df.at[idx, "CanalBank"] = bank
    df.at[idx, "CanalLevel"] = level
    df.at[idx, "MainKey"] = main_key

df["Bank"] = df["Bank"].replace("", np.nan)
df["Bank"] = df["Bank"].fillna(df["CanalBank"]).fillna("")

df["Class_upper"] = df["Class"].str.upper()

for col in ["BranchArea", "Area_G1", "Area_G2", "Area_G3"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    else:
        df[col] = np.nan

canal_edges = build_canal_edges(df)

# ========================================
# 2. Select Bank / Level / Canal
# ========================================
st.header("1. Select Bank / Canal level / Canal")

col_bank, col_level = st.columns(2)

with col_bank:
    selected_bank = st.radio(
        "Bank",
        options=["Left", "Right"],
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

if level_sel == "Secondary Canal":
    level_name = "Secondary"
else:
    level_name = "Main"

df_bank = df[df["Bank"] == selected_bank].copy()
df_level = df_bank[df_bank["CanalLevel"] == level_name].copy()

if df_level.empty:
    st.info(f"No rows for Bank = {selected_bank}, Level = {level_name}.")
    st.stop()

order_canal = (
    df_level
    .sort_values("CanalKey")
    [["Canal", "CanalKey"]]
    .drop_duplicates(subset=["Canal"])
)
canal_names = order_canal["Canal"].tolist()

selected_canal = st.selectbox(
    f"Select {level_name} (sorted by CanalKey)",
    canal_names,
)

df_sel_raw = df_level[df_level["Canal"] == selected_canal].copy()
if df_sel_raw.empty:
    st.info("No rows for selected canal.")
    st.stop()

# ========================================
# 3. Row visibility controls
# ========================================
st.header("2. Row visibility (TB / Branch / Target)")

col_tb, col_branch, col_target, col_gol = st.columns(4)

with col_tb:
    show_tb = st.checkbox("Show TB", value=False)

with col_branch:
    show_branch = st.checkbox("Show Branch Canal", value=False)

with col_target:
    show_target = st.checkbox("Show Target Canal", value=True)

with col_gol:
    show_golongan = st.checkbox("Show Golongan 1 / 2 / 3 columns", value=False)


def apply_class_filters(df_in: pd.DataFrame) -> pd.DataFrame:
    if "Class_upper" not in df_in.columns:
        return df_in

    df_tmp = df_in.copy()
    cls = df_tmp["Class_upper"]

    mask_tb = (cls == "TB")
    mask_div = (cls == "DIVISION")
    mask_target = ~(mask_tb | mask_div)

    keep_mask = (
        (show_tb & mask_tb) |
        (show_branch & mask_div) |
        (show_target & mask_target)
    )
    return df_tmp[keep_mask].copy()


# ========================================
# 共通: Bank / Level / Canal ごとの summary を作る関数
# ========================================
def build_summary_for_canal(
    df_source: pd.DataFrame,
    selected_bank: str,
    level_name: str,
    selected_canal: str,
) -> pd.DataFrame:
    """
    3番・4番テーブル共通のサマリを作る。

    - df_source: Qp を含むネットワーク DataFrame（元 df / df_qp_case どちらでもOK）
    - selected_bank / level_name / selected_canal:
        画面上で選択された Bank / Canal level / Canal 名
    - Row visibility (TB / Branch / Target) は apply_class_filters() で共通適用
    """
    # 1) Bank / Canal level / Canal で絞り込み
    df_bank = df_source[df_source["Bank"] == selected_bank].copy()
    df_level = df_bank[df_bank["CanalLevel"] == level_name].copy()
    if df_level.empty:
        return df_level.iloc[0:0].copy()

    df_sel_raw = df_level[df_level["Canal"] == selected_canal].copy()
    if df_sel_raw.empty:
        return df_sel_raw.iloc[0:0].copy()

    # 2) 内部セグメントの groupby サマリ
    group_cols = [
        "Bank", "Canal",
        "ParentKey", "ChildKey",
        "ParentName", "ChildName",
        "Class", "SectionName",
    ]
    agg_cols_base = ["BranchArea", "Area_G1", "Area_G2", "Area_G3"]
    agg_cols_q = labels  # 72 個の 5-day 列

    internal_summary = (
        df_sel_raw
        .groupby(group_cols, as_index=False)[agg_cols_base + agg_cols_q]
        .sum()
    )

    # 3) Division 行（branch rows）を canal_edges から追加
    branches_for_this = pd.DataFrame()
    if canal_edges is not None and not canal_edges.empty:
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
                "ChildKey":   df_source.at[idx_div, "ChildKey"],
                "ParentName": b["JunctionName"],
                "ChildName":  b["ChildCanal"],
                "Class":      "Division",
                "SectionName": df_source.at[idx_div, "SectionName"],
                "BranchArea": df_source.at[idx_div, "BranchArea"],
                "Area_G1":    df_source.at[idx_div, "Area_G1"],
                "Area_G2":    df_source.at[idx_div, "Area_G2"],
                "Area_G3":    df_source.at[idx_div, "Area_G3"],
            }
            for lab in labels:
                record[lab] = df_source.at[idx_div, lab]
            branch_records.append(record)

        branch_summary = pd.DataFrame(branch_records)
    else:
        branch_summary = internal_summary.iloc[0:0].copy()

    # 4) internal + branch を結合して並び替え
    summary = pd.concat([internal_summary, branch_summary], ignore_index=True)

    is_branch = np.where(
        summary["Class"].astype(str).str.strip() == "Division",
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
        .sort_values(
            ["SortKey", "SortIsBranch", "ChildName", "ParentKey", "ChildKey"]
        )
        .reset_index(drop=True)
    )

    summary["Class_upper"] = summary["Class"].astype(str).str.upper()

    # 5) TB / Branch / Target のチェックボックスを共通適用
    summary = apply_class_filters(summary)
    return summary

def _get_reach_row(df_source: pd.DataFrame, bank: str, canal: str,
                   parent_name: str, child_name: str) -> pd.Series | None:
    """Bank / Canal / ParentName / ChildName で 1 行を特定（存在しなければ None）。"""
    mask = (
        df_source["Bank"].astype(str).str.strip().eq(bank) &
        df_source["Canal"].astype(str).str.strip().eq(canal) &
        df_source["ParentName"].astype(str).str.strip().eq(parent_name) &
        df_source["ChildName"].astype(str).str.strip().eq(child_name)
    )
    hit = df_source[mask]
    if hit.empty:
        return None
    return hit.iloc[0]


def _build_5day_qp_pair(
    df_base: pd.DataFrame,
    df_case: pd.DataFrame | None,
    bank: str,
    canal: str,
    parent_name: str,
    child_name: str,
) -> pd.DataFrame | None:
    """
    指定した 1 reach について、5日ごとの Qp（Planned / Adjusted）を作る。

    ・df_base : Qp_all_reaches_m3s.csv 読み込み結果（単位 m³/s）
    ・df_case : df_qp_case（単位 L/s） → 1000 で割って m³/s に変換
    戻り値: ["StepLabel","Qp_planned","Qp_adjusted"]
    """
    row_base = _get_reach_row(df_base, bank, canal, parent_name, child_name)
    if row_base is None:
        return None

    row_case = _get_reach_row(df_case, bank, canal, parent_name, child_name) \
        if df_case is not None else None

    records = []
    for lab in labels:  # "Nov-1" .. "Oct-6"
        if lab not in row_base.index:
            continue

        # Planned Qp（既に m³/s）
        try:
            qp_plan = float(row_base.get(lab, 0.0))
        except Exception:
            qp_plan = 0.0

        # Adjusted Qp（L/s → m³/s）
        qp_adj = np.nan
        if row_case is not None and lab in row_case.index:
            v = row_case.get(lab)
            try:
                qp_adj = float(v) / 1000.0   # ★ ここで 1000 分の 1
            except Exception:
                qp_adj = np.nan

        records.append({
            "StepLabel": lab,
            "Qp_planned": qp_plan,
            "Qp_adjusted": qp_adj,
        })

    return pd.DataFrame(records)

def render_5day_qp_bar_with_cards(df_5day: pd.DataFrame | None, title: str) -> None:
    """
    5日ごとの Qp (Planned / Adjusted) を棒グラフで描き、
    下に Max 値 (m³/s) と積算量 Sum (百万 m³) をカード風に表示する。
    Adjusted Qp のカードは背景色を付けて強調する。
    """
    if df_5day is None or df_5day.empty:
        st.write(f"(no data for {title})")
        return

    # --- 補助: df_5day と列名から体積[m3]を積算する ---
    def _volume_m3_from_df(df_local: pd.DataFrame, col_name: str) -> float:
        total = 0.0
        for _, row in df_local.iterrows():
            step_label = row["StepLabel"]  # "Nov-1" など
            q = row[col_name]
            if pd.isna(q):
                continue
            try:
                idx = labels.index(step_label)  # 0..71
            except ValueError:
                continue

            month = water_month_order[idx // 6]   # 11,12,1..10
            step  = (idx % 6) + 1                 # 1..6

            # その月の日数（うるう年は考えず、2001年など適当な平年を使用）
            days_in_month = calendar.monthrange(2001, month)[1]
            if step <= 5:
                days = 5
            else:
                days = days_in_month - 25  # 6番目のstepだけ日数が変動

            # Q [m3/s] × 秒数 → m3
            total += float(q) * days * 86400.0
        return total

    # ---- Max / Sum とそのラベル ----
    # Planned
    max_plan = df_5day["Qp_planned"].max()
    lab_plan = df_5day.loc[df_5day["Qp_planned"].idxmax(), "StepLabel"]
    vol_plan_m3 = _volume_m3_from_df(df_5day, "Qp_planned")
    vol_plan_mill = vol_plan_m3 / 1e6  # 百万 m3

    # Adjusted
    if df_5day["Qp_adjusted"].notna().any():
        max_adj = df_5day["Qp_adjusted"].max()
        lab_adj = df_5day.loc[df_5day["Qp_adjusted"].idxmax(), "StepLabel"]
        df_adj_for_sum = df_5day.copy()
        df_adj_for_sum["Qp_adjusted"] = df_adj_for_sum["Qp_adjusted"].fillna(0.0)
        vol_adj_m3 = _volume_m3_from_df(df_adj_for_sum, "Qp_adjusted")
        vol_adj_mill = vol_adj_m3 / 1e6
    else:
        max_adj = np.nan
        lab_adj = "-"
        vol_adj_mill = np.nan

    # ---- Altair 用に縦長に変換 ----
    df_long = df_5day.melt(
        id_vars="StepLabel",
        value_vars=["Qp_planned", "Qp_adjusted"],
        var_name="Scenario",
        value_name="Qp",
    )
    label_map = {
        "Qp_planned": "Planned Qp",
        "Qp_adjusted": "Adjusted Qp",
    }
    df_long["ScenarioLabel"] = df_long["Scenario"].map(label_map)

    # x 軸ラベルは 72 ステップ全部だとつぶれるので、6 ステップごとに表示
    tick_values = [lab for i, lab in enumerate(labels) if i % 6 == 0]

    # ★ 軸ラベル・目盛のフォントサイズも大きめに
    chart = (
        alt.Chart(df_long)
        .mark_bar()
        .encode(
            x=alt.X(
                "StepLabel:N",
                sort=labels,
                title="5-day step",
                axis=alt.Axis(
                    values=tick_values,
                    labelFontSize=12,
                    titleFontSize=14,
                ),
            ),
            xOffset="ScenarioLabel:N",
            y=alt.Y(
                "Qp:Q",
                title="Q (m³/s)",
                axis=alt.Axis(
                    labelFontSize=12,
                    titleFontSize=14,
                ),
            ),
            color=alt.Color(
                "ScenarioLabel:N",
                scale=alt.Scale(
                    domain=["Planned Qp", "Adjusted Qp"],
                    range=["#60a5fa", "#f97373"],  # 青 / 赤
                ),
                legend=alt.Legend(
                    title=None,
                    orient="top",
                    labelFontSize=12,
                ),
            ),
            tooltip=[
                "StepLabel:N",
                "ScenarioLabel:N",
                alt.Tooltip("Qp:Q", format=".3f"),
            ],
        )
        .properties(
            title=title,
            width=650,
            height=260,
        )
    )

    # Streamlit v1.41 以降は width を使う
    st.altair_chart(chart, width="stretch")

    # ---- 下にカード風の Max / Sum 情報 ----
    card_left, card_right = st.columns(2)

    # Planned 用カード
    card_html_planned = """
    <div style="
        border-radius:8px;
        border:1px solid #e5e7eb;
        padding:6px 10px;
        margin-top:4px;
        background:#f9fafb;">
      <div style="font-size:0.9rem;color:#4b5563;margin-bottom:2px;">{label}</div>
      <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:12px;">
        <div>
          <div style="font-size:0.8rem;color:#6b7280;">Max</div>
          <div style="font-size:1.1rem;font-weight:bold;">{max_val:.1f} m³/s</div>
          <div style="font-size:0.8rem;color:#6b7280;">at {step}</div>
        </div>
        <div style="text-align:right;">
          <div style="font-size:0.8rem;color:#6b7280;font-weight:bold;">Sum</div>
          <div style="font-size:1.1rem;font-weight:bold;">{sum_val:,.1f} mill. m³</div>
          <div style="font-size:0.75rem;color:#6b7280;visibility:hidden;">dummy line</div>
        </div>
      </div>
    </div>
    """

    # Adjusted 用カード（背景を薄い赤で強調）
    card_html_adjusted = """
    <div style="
        border-radius:8px;
        border:1px solid #fecaca;
        padding:6px 10px;
        margin-top:4px;
        background:#fef2f2;">
      <div style="font-size:0.95rem;color:#b91c1c;font-weight:bold;margin-bottom:2px;">
        {label}
      </div>
      <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:12px;">
        <div>
          <div style="font-size:0.85rem;color:#7f1d1d;">Max</div>
          <div style="font-size:1.2rem;font-weight:bold;color:#b91c1c;">{max_val:.1f} m³/s</div>
          <div style="font-size:0.8rem;color:#7f1d1d;">at {step}</div>
        </div>
        <div style="text-align:right;">
          <div style="font-size:0.85rem;color:#7f1d1d;font-weight:bold;">Sum</div>
          <div style="font-size:1.2rem;font-weight:bold;color:#b91c1c;">{sum_val:,.1f} mill. m³</div>
          <div style="font-size:0.75rem;color:#7f1d1d;visibility:hidden;">dummy line</div>
        </div>
      </div>
    </div>
    """

    with card_left:
        st.markdown(
            card_html_planned.format(
                label="Planned Qp",
                max_val=max_plan,
                step=lab_plan,
                sum_val=vol_plan_mill,
            ),
            unsafe_allow_html=True,
        )

    with card_right:
        if np.isnan(max_adj):
            # Adjusted シナリオなし
            st.markdown(
                """
                <div style="
                    border-radius:8px;
                    border:1px solid #e5e7eb;
                    padding:6px 10px;
                    margin-top:4px;
                    background:#fef2f2;">
                  <div style="font-size:0.9rem;color:#b91c1c;font-weight:bold;">Adjusted Qp</div>
                  <div style="font-size:0.9rem;color:#9ca3af;">(no scenario)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                card_html_adjusted.format(
                    label="Adjusted Qp",
                    max_val=max_adj,
                    step=lab_adj,
                    sum_val=vol_adj_mill,
                ),
                unsafe_allow_html=True,
            )


def render_annual_summary_and_detail_block(
    left_5day: pd.DataFrame | None,
    right_5day: pd.DataFrame | None,
    df_case_disp: pd.DataFrame | None,
    mt_start_labels: set[str] | None,
    bank_sel: str,
    sec_name: str,
    gol_sel: int,
) -> None:
    """
    Section 4 の 5-day 縦棒グラフの直下に出す年間サマリ表と、
    Secondary × Golongan の Adjusted Qp 詳細表をまとめて描画する。

    年間サマリは行順:
      Planned:  Left, Right, Total
      Adjusted: Left, Right, Total

    Qp_max の Max 列は「月別 Max の合計」ではなく、
    5Day_ID（5-day step）ごとの合計流量から求めた年間最大値を表示する。
    Adjusted 行は Plan よりフォントを少し大きく & 太字で強調する。
    Total 行は Bank 列の背景色を少し変えて見やすくする。
    セル内バー: Planned=薄青, Adjusted=薄赤, 90% をフルスケールとして表示。
    """
    # どちらの系列も無ければ何もしない
    if (left_5day is None or left_5day.empty) and (right_5day is None or right_5day.empty):
        return

    months_num = water_month_order[:]              # [11,12,1,...,10]
    months_abbr = [calendar.month_abbr[m] for m in months_num]

    def _monthly_stats(df_5day_single: pd.DataFrame | None, col_name: str):
        """月別 max(Q) [m³/s] と volume [million m³] を返す。"""
        qmax = {m: np.nan for m in months_num}
        vol  = {m: 0.0     for m in months_num}

        if df_5day_single is None or df_5day_single.empty:
            return qmax, vol

        for _, row in df_5day_single.iterrows():
            step_label = row["StepLabel"]
            q = row[col_name]
            if pd.isna(q):
                continue
            try:
                idx = labels.index(step_label)
            except ValueError:
                continue

            month = water_month_order[idx // 6]
            step  = (idx % 6) + 1

            days_in_month = calendar.monthrange(2001, month)[1]
            days = 5 if step <= 5 else (days_in_month - 25)

            # 月内最大 Q
            if np.isnan(qmax[month]) or q > qmax[month]:
                qmax[month] = float(q)

            # 体積[m³]
            vol[month] += float(q) * days * 86400.0

        # m³ → million m³
        for m in months_num:
            vol[m] /= 1e6

        return qmax, vol

    def _nan_to_zero(v: float) -> float:
        return 0.0 if pd.isna(v) else float(v)

    # ===== Left bank =====
    if left_5day is not None and not left_5day.empty:
        max5_L_P = float(left_5day["Qp_planned"].max()) if left_5day["Qp_planned"].notna().any() else np.nan
        qmax_L_P, vol_L_P = _monthly_stats(left_5day, "Qp_planned")

        if left_5day["Qp_adjusted"].notna().any():
            max5_L_A = float(left_5day["Qp_adjusted"].max())
            dfL_adj = left_5day.copy()
            dfL_adj["Qp_adjusted"] = dfL_adj["Qp_adjusted"].fillna(0.0)
            qmax_L_A, vol_L_A = _monthly_stats(dfL_adj, "Qp_adjusted")
        else:
            max5_L_A = np.nan
            qmax_L_A = {m: np.nan for m in months_num}
            vol_L_A  = {m: 0.0     for m in months_num}
    else:
        max5_L_P = max5_L_A = np.nan
        qmax_L_P = {m: np.nan for m in months_num}
        qmax_L_A = {m: np.nan for m in months_num}
        vol_L_P  = {m: 0.0     for m in months_num}
        vol_L_A  = {m: 0.0     for m in months_num}

    # ===== Right bank =====
    if right_5day is not None and not right_5day.empty:
        max5_R_P = float(right_5day["Qp_planned"].max()) if right_5day["Qp_planned"].notna().any() else np.nan
        qmax_R_P, vol_R_P = _monthly_stats(right_5day, "Qp_planned")

        if right_5day["Qp_adjusted"].notna().any():
            max5_R_A = float(right_5day["Qp_adjusted"].max())
            dfR_adj = right_5day.copy()
            dfR_adj["Qp_adjusted"] = dfR_adj["Qp_adjusted"].fillna(0.0)
            qmax_R_A, vol_R_A = _monthly_stats(dfR_adj, "Qp_adjusted")
        else:
            max5_R_A = np.nan
            qmax_R_A = {m: np.nan for m in months_num}
            vol_R_A  = {m: 0.0     for m in months_num}
    else:
        max5_R_P = max5_R_A = np.nan
        qmax_R_P = {m: np.nan for m in months_num}
        qmax_R_A = {m: np.nan for m in months_num}
        vol_R_P  = {m: 0.0     for m in months_num}
        vol_R_A  = {m: 0.0     for m in months_num}

    # ===== Total (Left+Right) 5Day step ベースの Max 用 =====
    # Planned total
    plan_series_list = []
    if left_5day is not None and not left_5day.empty:
        plan_series_list.append(left_5day.set_index("StepLabel")["Qp_planned"].fillna(0.0))
    if right_5day is not None and not right_5day.empty:
        plan_series_list.append(right_5day.set_index("StepLabel")["Qp_planned"].fillna(0.0))

    if plan_series_list:
        total_plan = plan_series_list[0]
        for s in plan_series_list[1:]:
            total_plan = total_plan.add(s, fill_value=0.0)
        max5_T_P = float(total_plan.max())
    else:
        max5_T_P = np.nan

    # Adjusted total
    adj_series_list = []
    if left_5day is not None and not left_5day.empty and left_5day["Qp_adjusted"].notna().any():
        adj_series_list.append(left_5day.set_index("StepLabel")["Qp_adjusted"].fillna(0.0))
    if right_5day is not None and not right_5day.empty and right_5day["Qp_adjusted"].notna().any():
        adj_series_list.append(right_5day.set_index("StepLabel")["Qp_adjusted"].fillna(0.0))

    if adj_series_list:
        total_adj = adj_series_list[0]
        for s in adj_series_list[1:]:
            total_adj = total_adj.add(s, fill_value=0.0)
        max5_T_A = float(total_adj.max())
    else:
        max5_T_A = np.nan

    # Total の月別 qmax / volume は左右岸の和
    qmax_T_P = {}
    qmax_T_A = {}
    vol_T_P  = {}
    vol_T_A  = {}
    for m in months_num:
        qmax_T_P[m] = _nan_to_zero(qmax_L_P[m]) + _nan_to_zero(qmax_R_P[m])
        qmax_T_A[m] = _nan_to_zero(qmax_L_A[m]) + _nan_to_zero(qmax_R_A[m])
        vol_T_P[m]  = vol_L_P[m] + vol_R_P[m]
        vol_T_A[m]  = vol_L_A[m] + vol_R_A[m]

    # ===== 行データ作成（Planned → Adjusted, Left/Right/Total） =====
    rows_max: list[dict] = []
    rows_sum: list[dict] = []

    def _append_rows(case_label: str,
                     qmax_L, qmax_R, qmax_T,
                     vol_L, vol_R, vol_T,
                     max5_L: float, max5_R: float, max5_T: float) -> None:
        """
        1ケース分（Planned / Adjusted）について、
        Left / Right / Total の3行を rows_max / rows_sum に追加する。
        Max 列は 5Day_ID レベルの合計値から求めた年間最大値を使う。
        """
        for bank_label, qdict, vdict, max_val_5d in [
            ("Left",  qmax_L, vol_L, max5_L),
            ("Right", qmax_R, vol_R, max5_R),
            ("Total", qmax_T, vol_T, max5_T),
        ]:
            # Qp_max テーブル用
            row_max = {"Type": "Qp_max (m³/s)", "Case": case_label, "Bank": bank_label}
            for m, abbr in zip(months_num, months_abbr):
                row_max[abbr] = qdict[m]
            row_max["Max"] = max_val_5d
            rows_max.append(row_max)

            # ΣQp テーブル用
            row_sum = {"Type": "ΣQp (Mill. m³)", "Case": case_label, "Bank": bank_label}
            vols = []
            for m, abbr in zip(months_num, months_abbr):
                vv = vdict[m]
                row_sum[abbr] = vv
                vols.append(vv)
            total_vol = sum(vols)
            row_sum["Sum"] = total_vol if abs(total_vol) > 1e-9 else np.nan
            rows_sum.append(row_sum)

    # Planned ブロック
    _append_rows("Planned",
                 qmax_L_P, qmax_R_P, qmax_T_P,
                 vol_L_P, vol_R_P, vol_T_P,
                 max5_L_P, max5_R_P, max5_T_P)

    # Adjusted ブロック
    _append_rows("Adjusted",
                 qmax_L_A, qmax_R_A, qmax_T_A,
                 vol_L_A, vol_R_A, vol_T_A,
                 max5_L_A, max5_R_A, max5_T_A)

    if not rows_max:
        return

    # 列順: No, Type, Case, Bank, Month..., Max / Sum
    cols_max = ["Type", "Case", "Bank"] + months_abbr + ["Max"]
    cols_sum = ["Type", "Case", "Bank"] + months_abbr + ["Sum"]
    df_max = pd.DataFrame(rows_max)[cols_max]
    df_sum = pd.DataFrame(rows_sum)[cols_sum]

    # No. 列と Type/Bank のまとめ
    df_max.insert(0, "No.", np.arange(1, len(df_max) + 1))
    df_sum.insert(0, "No.", np.arange(1, len(df_sum) + 1))

    df_max_disp = df_max.copy()
    df_sum_disp = df_sum.copy()
    for col in ["Type", "Bank"]:
        df_max_disp[col] = df_max_disp[col].where(df_max_disp[col].ne(df_max_disp[col].shift()), "")
        df_sum_disp[col] = df_sum_disp[col].where(df_sum_disp[col].ne(df_sum_disp[col].shift()), "")

    # ハイライト関数（行ごと）
    def _make_highlighter(df_for_style: pd.DataFrame):
        def _highlight(row: pd.Series):
            styles = [""] * len(row)
            case_label = row["Case"]
            vals = [row[m] for m in months_abbr]
            arr = np.array(vals, dtype=float)
            row_max = np.nanmax(arr) if np.any(~np.isnan(arr)) else None

            for j, col in enumerate(df_for_style.columns):
                # 月別 Max（Planned→青, Adjusted→赤）
                if col in months_abbr and row_max is not None:
                    v = row[col]
                    if not pd.isna(v) and abs(v - row_max) < 1e-9:
                        if case_label == "Planned":
                            styles[j] = "color:#1d4ed8;font-weight:bold;"
                        else:
                            styles[j] = "color:#b91c1c;font-weight:bold;"
                # 集計列 Max / Sum は黄色ハイライト
                if col in ["Max", "Sum"]:
                    base = styles[j]
                    extra = "background-color:#fff59d;font-weight:bold;"
                    styles[j] = (base + extra) if base else extra
            return styles
        return _highlight

    # ---------------- Qp_max テーブル ----------------
    fmt_max = {m: "{:.1f}" for m in months_abbr}
    fmt_max["Max"] = "{:.1f}"

    sty_max = df_max_disp.style.format(fmt_max, na_rep="")

    # セル内バー（Planned: 青, Adjusted: 赤）
    vals_all_max = df_max_disp[months_abbr].to_numpy().astype(float)
    vmax_max = np.nanmax(vals_all_max) if np.any(~np.isnan(vals_all_max)) else 0.0
    if vmax_max > 0:
        vmax_max_for_bar = vmax_max / 0.9
        planned_rows = df_max_disp.index[df_max_disp["Case"] == "Planned"]
        adjusted_rows = df_max_disp.index[df_max_disp["Case"] == "Adjusted"]
        sty_max = sty_max.bar(
            subset=pd.IndexSlice[planned_rows, months_abbr],
            color="#bfdbfe",  # 薄青
            vmin=0,
            vmax=vmax_max_for_bar,
        )
        sty_max = sty_max.bar(
            subset=pd.IndexSlice[adjusted_rows, months_abbr],
            color="#fecaca",  # 薄赤
            vmin=0,
            vmax=vmax_max_for_bar,
        )

    # Adjusted 行のインデックス
    adj_rows_max = df_max_disp.index[df_max_disp["Case"] == "Adjusted"]

    sty_max = (
        sty_max
        .apply(_make_highlighter(df_max_disp), axis=1)
        .hide(axis="index")
        .set_properties(**{
            "text-align": "center",
            "font-size": "11px",
            "padding": "2px 4px",
            "white-space": "nowrap",
        })
        # Adjusted 行だけフォント大きめ & 太字
        .set_properties(
            subset=pd.IndexSlice[adj_rows_max, :],
            **{"font-size": "12px", "font-weight": "bold"}
        )
        # ★ 罫線スタイルを追加
        .set_table_styles([
            {
                "selector": "table",
                "props": [
                    ("border-collapse", "collapse"),
                    ("border", "1px solid #999"),
                ],
            },
            {
                "selector": "th",
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
                ],
            },
        ])
    )


    # Adjusted 行を強調
    adj_rows_max = df_max_disp.index[df_max_disp["Case"] == "Adjusted"]
    sty_max = sty_max.set_properties(
        subset=pd.IndexSlice[adj_rows_max, :],
        **{"font-weight": "bold", "font-size": "13.5px"}
    )

    # Total 行の Bank 列だけ少し色付け
    total_rows_max = df_max_disp.index[df_max_disp["Bank"] == "Total"]
    if len(total_rows_max) > 0:
        sty_max = sty_max.set_properties(
            subset=pd.IndexSlice[total_rows_max, ["Bank"]],
            **{"background-color": "#fef3c7"}
        )

    # ---------------- ΣQp テーブル ----------------
    fmt_sum = {m: "{:,.1f}" for m in months_abbr}
    fmt_sum["Sum"] = "{:,.1f}"

    sty_sum = df_sum_disp.style.format(fmt_sum, na_rep="")

    vals_all_sum = df_sum_disp[months_abbr].to_numpy().astype(float)
    vmax_sum = np.nanmax(vals_all_sum) if np.any(~np.isnan(vals_all_sum)) else 0.0
    if vmax_sum > 0:
        vmax_sum_for_bar = vmax_sum / 0.9
        planned_rows_s = df_sum_disp.index[df_sum_disp["Case"] == "Planned"]
        adjusted_rows_s = df_sum_disp.index[df_sum_disp["Case"] == "Adjusted"]
        sty_sum = sty_sum.bar(
            subset=pd.IndexSlice[planned_rows_s, months_abbr],
            color="#bfdbfe",
            vmin=0,
            vmax=vmax_sum_for_bar,
        )
        sty_sum = sty_sum.bar(
            subset=pd.IndexSlice[adjusted_rows_s, months_abbr],
            color="#fecaca",
            vmin=0,
            vmax=vmax_sum_for_bar,
        )

    adj_rows_sum = df_sum_disp.index[df_sum_disp["Case"] == "Adjusted"]

    sty_sum = (
        sty_sum
        .apply(_make_highlighter(df_sum_disp), axis=1)
        .hide(axis="index")
        .set_properties(**{
            "text-align": "center",
            "font-size": "11px",
            "padding": "2px 4px",
            "white-space": "nowrap",
        })
        .set_properties(
            subset=pd.IndexSlice[adj_rows_sum, :],
            **{"font-size": "12px", "font-weight": "bold"}
        )
        .set_table_styles([
            {
                "selector": "table",
                "props": [
                    ("border-collapse", "collapse"),
                    ("border", "1px solid #999"),
                ],
            },
            {
                "selector": "th",
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
                ],
            },
        ])
    )


    # Adjusted 行を強調
    adj_rows_sum = df_sum_disp.index[df_sum_disp["Case"] == "Adjusted"]
    sty_sum = sty_sum.set_properties(
        subset=pd.IndexSlice[adj_rows_sum, :],
        **{"font-weight": "bold", "font-size": "13.5px"}
    )

    # Total 行の Bank 列だけ少し色付け
    total_rows_sum = df_sum_disp.index[df_sum_disp["Bank"] == "Total"]
    if len(total_rows_sum) > 0:
        sty_sum = sty_sum.set_properties(
            subset=pd.IndexSlice[total_rows_sum, ["Bank"]],
            **{"background-color": "#fef3c7"}
        )

    # ==== 実際の描画 ====
    # タイトルは左寄せ、表自体は中央寄せ（flexでラップ）
    st.markdown(
        "<div style='font-weight:bold;font-size:1.05rem;margin:6px 0 4px;text-align:left;'>"
        "Annual summary – Qp_max (m³/s)"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='display:flex; justify-content:center; margin:0.25rem 0;'>"
        "<div style='min-width:960px; max-width:100%;'>"
        f"{sty_max.to_html()}"
        "</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='font-size:0.8rem;color:#4b5563;margin-top:2px;margin-bottom:6px;text-align:left;'>"
        "Note: The 'Max' column is <strong>not</strong> the sum of monthly peaks. "
        "It is the maximum of the total discharge Q over all 5-day steps in the water year. "
        "In particular, for the 'Total' row, Left and Right bank peaks may occur in different months, "
        "but the Max value is taken from the 5-day step where (Left + Right) is highest."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div style='font-weight:bold;font-size:0.9rem;margin:6px 0 2px;text-align:left;'>"
        "Annual summary – ΣQp (Mill. m³)"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='display:flex; justify-content:center; margin:0.25rem 0;'>"
        "<div style='min-width:960px; max-width:100%;'>"
        f"{sty_sum.to_html()}"
        "</div></div>",
        unsafe_allow_html=True,
    )

    # ==== Secondary × Golongan の詳細表 ==== 
    if df_case_disp is not None and mt_start_labels is not None and not df_case_disp.empty:
        st.markdown(
            "<div style='font-weight:bold;font-size:0.9rem;margin:4px 0;text-align:left;'>"
            f"Adjusted Qp table – Bank={bank_sel}, Secondary={sec_name}, Golongan={gol_sel}"
            "</div>",
            unsafe_allow_html=True,
        )
        render_adjusted_qp_table(
            df_case_disp,
            mt_start_labels=mt_start_labels,
            caption=None,
        )


def render_main_adjusted_table(
    df_qp_case: pd.DataFrame,
    selected_bank: str,
    level_name: str,
    selected_canal: str,
    display_cols: list[str],
    show_golongan: bool,
    mt_start_labels: set[str] | None = None,
) -> None:
    """
    Section 4 のメインテーブル（Adjusted Qp, m³/s）を描画する。

    - df_qp_case : compute_qp_from_nfr_for_df で再計算したネットワーク（単位 L/s）
    - selected_* : Section1〜3 で選んだ Bank / Level / Canal
    - display_cols: 表3と同じベース列 + 5-day 列のリスト
    - show_golongan: ゴロンガン列を出すかどうか
    - mt_start_labels: このシナリオにおける MT-1/2/3 の開始 5Day ラベル集合
                       （例 {"Nov-3","Dec-1","Feb-2"}）。
                       None のときはハイライト無し。
    """
    # --- Bank / Level / Canal でサマリを作成 ---
    summary_adj = build_summary_for_canal(
        df_qp_case,
        selected_bank=selected_bank,
        level_name=level_name,
        selected_canal=selected_canal,
    )

    if summary_adj.empty:
        with adj_top_container:
            st.write("(no adjusted Qp rows for selected canal)")
        return

    display_adj = summary_adj[display_cols].copy()

    # L/s → m³/s に変換
    for lab in labels:
        if lab in display_adj.columns:
            display_adj[lab] = display_adj[lab] / 1000.0

    # 列名を表3と合わせる
    display_adj = display_adj.rename(columns={
        "BranchArea": "Area (ha)",
        "Area_G1": "Golongan 1",
        "Area_G2": "Golongan 2",
        "Area_G3": "Golongan 3",
    })

    # ゴロンガン列の表示/非表示
    if not show_golongan:
        for gcol in ["Golongan 1", "Golongan 2", "Golongan 3"]:
            if gcol in display_adj.columns:
                display_adj = display_adj.drop(columns=gcol)

    # Bank / Canal / ParentName の連続行は空欄に（Planned と同じ）
    for col in ["Bank", "Canal", "ParentName"]:
        if col in display_adj.columns:
            display_adj[col] = display_adj[col].where(
                display_adj[col].ne(display_adj[col].shift()), ""
            )

    # 行番号を 1 始まりに
    display_adj_for_style = display_adj.copy()
    display_adj_for_style.index = np.arange(1, len(display_adj_for_style) + 1)

    # シナリオ開始列ハイライト用のラベルセット（None → 空集合）
    mt_labels: set[str] = set(mt_start_labels) if mt_start_labels is not None else set()

    # ★レポート用に保持
    st.session_state["report_display_adj_for_style"] = display_adj_for_style.copy()
    st.session_state["report_mt_start_labels"] = set(mt_labels)

    # ★ 実際の描画：列単位で mt_labels に一致する 5Day 列をオレンジ塗り
    with adj_top_container:
        render_adjusted_qp_table(
            display_adj_for_style,
            mt_start_labels=mt_labels,
            caption=(
                f"Adjusted Qp (m³/s) – Bank={selected_bank}, "
                f"{level_name} = {selected_canal}"
            ),
        )


def _get_reach_row(df_source: pd.DataFrame, bank: str, canal: str,
                   parent_name: str, child_name: str) -> pd.Series | None:
    """Bank / Canal / ParentName / ChildName で 1 行を特定（存在しなければ None）。"""
    mask = (
        df_source["Bank"].astype(str).str.strip().eq(bank) &
        df_source["Canal"].astype(str).str.strip().eq(canal) &
        df_source["ParentName"].astype(str).str.strip().eq(parent_name) &
        df_source["ChildName"].astype(str).str.strip().eq(child_name)
    )
    hit = df_source[mask]
    if hit.empty:
        return None
    return hit.iloc[0]


def _build_monthly_qp_for_reach(
    df_source: pd.DataFrame,
    bank: str,
    canal: str,
    parent_name: str,
    child_name: str,
    unit_scale: float,
) -> pd.DataFrame | None:
    """
    指定した 1 つの reach について、Month ごとの平均 Qp を計算する。
      unit_scale: 1.0 → そのまま, 1/1000 → L/s → m³/s 変換など
    戻り値: columns = ["MonthAbbr","Qp"]
    """
    row = _get_reach_row(df_source, bank, canal, parent_name, child_name)
    if row is None:
        return None

    vals = []
    months = []
    for lab in labels:          # "Nov-1" など 72 列
        if lab not in row.index:
            continue
        try:
            v = float(row[lab]) * unit_scale
        except Exception:
            v = 0.0
        vals.append(v)
        mon_abbr = lab.split("-")[0]   # "Nov-1" → "Nov"
        months.append(mon_abbr)

    df_tmp = pd.DataFrame({"MonthAbbr": months, "Qp": vals})
    monthly = (
        df_tmp.groupby("MonthAbbr", as_index=False)["Qp"].mean()
              .set_index("MonthAbbr")
              .reindex(month_order_abbr, fill_value=0.0)
              .reset_index()
    )
    return monthly


def _build_monthly_qp_pair(
    df_base: pd.DataFrame,
    df_case: pd.DataFrame | None,
    bank: str,
    canal: str,
    parent_name: str,
    child_name: str,
) -> pd.DataFrame | None:
    """
    同じ reach について、Planned Qp と Adjusted Qp の Month 別データを作る。
    戻り値: ["MonthAbbr","Qp_planned","Qp_adjusted"]
    """
    # Planned Qp（Qp_all_reaches_m3s.csv は既に m³/s）
    base = _build_monthly_qp_for_reach(
        df_base, bank, canal, parent_name, child_name, unit_scale=1.0
    )
    if base is None:
        return None
    base = base.rename(columns={"Qp": "Qp_planned"})

    # Adjusted Qp（df_qp_case は L/s なので 1000 分の 1）
    if df_case is None:
        base["Qp_adjusted"] = np.nan
        return base

    adj = _build_monthly_qp_for_reach(
        df_case, bank, canal, parent_name, child_name, unit_scale=1.0 / 1000.0
    )
    if adj is None:
        base["Qp_adjusted"] = np.nan
        return base

    out = base.merge(
        adj[["MonthAbbr", "Qp"]].rename(columns={"Qp": "Qp_adjusted"}),
        on="MonthAbbr",
        how="left",
    )
    return out

def compute_mt_start_labels_for_scenario(
    landprep_case: pd.DataFrame,
    bank_sel: str,
    gol_sel: int,
) -> set[str]:
    """
    landprep_case（Section 6 で作ったシナリオ用 landprep）から、
    Bank × Golongan ごとの MT-1/2/3 の開始 5Day 列ラベル集合を返す。
      例: {"Nov-3", "Dec-1", "Feb-2"}
    """
    if landprep_case is None or landprep_case.empty:
        return set()

    col_map = {
        "MT-1": "MT1_start",
        "MT-2": "MT2_start",
        "MT-3": "MT3_start",
    }

    def _split_group_key_local(s: str):
        s = str(s)
        if "_" not in s:
            return None, None
        bank_part, rest = s.split("_", 1)
        m = re.search(r"(\d+)", rest)
        g = int(m.group(1)) if m else None
        return bank_part, g

    labels_set: set[str] = set()

    for season, col in col_map.items():
        if col not in landprep_case.columns:
            continue

        # このシナリオの Bank × Golongan 行のみが対象
        mask = landprep_case["Group"].astype(str).apply(
            lambda s: _split_group_key_local(s) == (bank_sel, gol_sel)
        )
        if not mask.any():
            continue

        # MT-3 の場合は active 行だけ
        if season == "MT-3" and "MT3_active" in landprep_case.columns:
            try:
                active = landprep_case["MT3_active"].astype(bool)
                mask = mask & active
            except Exception:
                pass

        vals = landprep_case.loc[mask, col].dropna()
        if vals.empty:
            continue

        try:
            dates = pd.to_datetime(vals).dt.date
        except Exception:
            continue

        d_min = dates.min()
        id_ = date_to_5day_id(d_min)
        idx = id_ - 1
        if 0 <= idx < len(labels):
            labels_set.add(labels[idx])

    return labels_set



def render_monthly_qp_bar(df_month: pd.DataFrame | None, title: str) -> None:
    """Month 別 Qp（Planned / Adjusted）の棒グラフを 1 つ描画。"""
    if df_month is None or df_month.empty:
        st.write(f"(no data for {title})")
        return

    df_long = df_month.melt(
        id_vars="MonthAbbr",
        value_vars=["Qp_planned", "Qp_adjusted"],
        var_name="Scenario",
        value_name="Qp",
    )
    scenario_label_map = {
        "Qp_planned": "Planned Qp",
        "Qp_adjusted": "Adjusted Qp",
    }
    df_long["ScenarioLabel"] = df_long["Scenario"].map(scenario_label_map)

    chart = (
        alt.Chart(df_long)
        .mark_bar()
        .encode(
            x=alt.X(
                "MonthAbbr:N",
                sort=month_order_abbr,
                title="Month",
            ),
            xOffset="ScenarioLabel:N",
            y=alt.Y("Qp:Q", title="Q (m³/s)"),
            color=alt.Color(
                "ScenarioLabel:N",
                scale=alt.Scale(
                    domain=["Planned Qp", "Adjusted Qp"],
                    range=["#60a5fa", "#f97373"],
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=["MonthAbbr:N", "ScenarioLabel:N", alt.Tooltip("Qp:Q", format=".3f")],
        )
        .properties(
            title=title,
            width=350,
            height=260,
        )
    )
    # ★ use_container_width → width="stretch"
    st.altair_chart(chart, width="stretch")



# ========================================
# 4. Build summary (internal segments + branch rows)
# ========================================
group_cols = [
    "Bank", "Canal",
    "ParentKey", "ChildKey",
    "ParentName", "ChildName",
    "Class", "SectionName",
]# ========================================
# 4. Build summary (internal segments + branch rows)
# ========================================
summary = build_summary_for_canal(df, selected_bank, level_name, selected_canal)

if summary.empty:
    st.info("No rows left after applying TB/Branch/Target filters.")
    st.stop()

# === MT-1 / MT-2 / MT-3 start indices (per row) ===
# Rule:
#   For each row, consider all Golongan (1–3) that have Area_G* > 0.
#   For each MT, highlight the earliest start 5Day_ID among those Golongan.
#   If no Area_G* > 0, fallback to all Golongan 1–3 at Bank level.
mt_start_map = build_mt_start_map_from_page2()

mt1_idx_rows: list[int | None] = []
mt2_idx_rows: list[int | None] = []
mt3_idx_rows: list[int | None] = []


def earliest_idx_for_mt(bank: str, season: str, gol_candidates: list[int]) -> int | None:
    """Return the minimum 5Day index for the given Bank / Season among gol_candidates."""
    idx_list: list[int] = []
    for g in gol_candidates:
        idx = mt_start_map.get((bank, g, season), None)
        if idx is not None:
            idx_list.append(idx)
    if not idx_list:
        return None
    return min(idx_list)


for _, row in summary.iterrows():
    bank = row["Bank"]
    areas = [
        row.get("Area_G1", 0.0),
        row.get("Area_G2", 0.0),
        row.get("Area_G3", 0.0),
    ]

    # 1) Golongan candidates from Area_G* (>0)
    gol_candidates = [
        g for g, area_val in enumerate(areas, start=1)
        if not (pd.isna(area_val) or abs(area_val) < 1e-9)
    ]

    # 2) If no Area_G* > 0, fallback to all Golongan 1–3 at Bank level
    if not gol_candidates:
        gol_candidates = [1, 2, 3]

    mt1_idx = earliest_idx_for_mt(bank, "MT-1", gol_candidates)
    mt2_idx = earliest_idx_for_mt(bank, "MT-2", gol_candidates)
    mt3_idx = earliest_idx_for_mt(bank, "MT-3", gol_candidates)

    mt1_idx_rows.append(mt1_idx)
    mt2_idx_rows.append(mt2_idx)
    mt3_idx_rows.append(mt3_idx)

# ========================================
# 3. Planned Qp table (top)
# ========================================
st.header("3. Planned Qp (fixed start)")

ordered_labels = labels[:]  # Nov-1 .. Oct-6

base_cols = [
    "Bank", "Canal", "ParentName",
    "ChildName", "Class", "SectionName",
    "BranchArea", "Area_G1", "Area_G2", "Area_G3",
]
display_cols = base_cols + ordered_labels


display_plan = summary[display_cols].copy()
display_plan = display_plan.rename(columns={
    "BranchArea": "Area (ha)",
    "Area_G1": "Golongan 1",
    "Area_G2": "Golongan 2",
    "Area_G3": "Golongan 3",
})

if not show_golongan:
    for gcol in ["Golongan 1", "Golongan 2", "Golongan 3"]:
        if gcol in display_plan.columns:
            display_plan = display_plan.drop(columns=gcol)

for col in ["Bank", "Canal", "ParentName"]:
    if col in display_plan.columns:
        display_plan[col] = display_plan[col].where(
            display_plan[col].ne(display_plan[col].shift()), ""
        )

display_plan_for_style = display_plan.copy()
display_plan_for_style.index = np.arange(1, len(display_plan_for_style) + 1)

# ★レポート用に保持
st.session_state["report_display_plan_for_style"] = display_plan_for_style.copy()

def make_highlight_func(
    qp_cols,
    mt1_idx_rows,
    mt2_idx_rows,
    mt3_idx_rows,
    shift_steps: int,
):
    """
    Highlight only the start 5Day_ID cells of MT-1 / MT-2 / MT-3 as orange,
    per row. shift_steps is given in 5-day steps (for Adjust Qp).

    Branch Canal 行（Class = "Division"）は、テキスト系の列を淡い緑 (#d9f99d) で着色する。
    """
    def _highlight(row: pd.Series):
        pos = int(row.name) - 1
        mt_indices: list[int] = []

        # MT-1 / MT-2 / MT-3 の開始 5Day_ID を行ごとにずらして着色対象にする
        for base in [mt1_idx_rows[pos], mt2_idx_rows[pos], mt3_idx_rows[pos]]:
            if base is None:
                continue
            idx = (base + shift_steps) % 72
            mt_indices.append(idx)

        mt_cols_this_row = {
            ordered_labels[i] for i in mt_indices
            if 0 <= i < len(ordered_labels)
        }

        # 行種別の判定
        cls_raw = str(row.get("Class", "")).strip()
        cls_lower = cls_raw.lower()
        # ★ Branch Canal 行: Class が "Division" なら一律 Branch 扱い
        is_branch_row = (cls_lower == "division")

        text_cols = {
            "Bank", "Canal", "ParentName", "ChildName",
            "Class", "SectionName", "Area (ha)",
            "Golongan 1", "Golongan 2", "Golongan 3",
        }
        branch_green_cols = {"ChildName", "Class", "SectionName", "Area (ha)"}

        styles: list[str] = []
        for col in row.index:
            color = ""

            # MT 開始セルはオレンジ
            if col in qp_cols and col in mt_cols_this_row:
                color = "#ffa500"  # orange (MT start)

            # テキスト系セルの色づけ
            elif col in text_cols:
                if is_branch_row and col in branch_green_cols:
                    # Branch Canal 行 → 緑
                    color = "#d9f99d"
                elif cls_lower == "tb":
                    # TB 行 → オレンジがかった背景
                    color = "#ffe6cc"

            styles.append(f"background-color: {color}" if color else "")

        return styles

    return _highlight

def apply_common_col_widths(styler: pd.io.formats.style.Styler,
                            df_for_cols: pd.DataFrame) -> pd.io.formats.style.Styler:
    """
    表3・表4で列幅をそろえるための共通設定。
    列名ベースで min-width / max-width を固定する。
    """

    # 文字系の列（少し広め）
    text_cols = ["Bank", "Canal", "ParentName", "ChildName", "Class", "SectionName"]
    # 面積・Golongan 列
    num_cols  = ["Area (ha)", "Golongan 1", "Golongan 2", "Golongan 3"]

    # 幅はお好みで調整可
    text_w = "110px"
    num_w  = "90px"
    qp_w   = "60px"   # 5-day 列

    for c in text_cols:
        if c in df_for_cols.columns:
            styler = styler.set_properties(
                subset=[c],
                **{"min-width": text_w, "max-width": text_w}
            )

    for c in num_cols:
        if c in df_for_cols.columns:
            styler = styler.set_properties(
                subset=[c],
                **{"min-width": num_w, "max-width": num_w}
            )

    # 5-day 列（Nov-1 … Oct-6）
    qp_cols = [lab for lab in labels if lab in df_for_cols.columns]
    if qp_cols:
        styler = styler.set_properties(
            subset=qp_cols,
            **{"min-width": qp_w, "max-width": qp_w}
        )

    return styler



def style_qp_table(
    df_sty: pd.DataFrame,
    level_name: str,
    highlight_func,
    bar_color: str,
):
    cols = set(df_sty.columns)

    fmt_dict: dict[str, object] = {}

    # ---- 数値フォーマット設定 ----
    if "Area (ha)" in cols:
        fmt_dict["Area (ha)"] = "{:,.0f}"

    if "Golongan 1" in cols:
        fmt_dict["Golongan 1"] = (
            lambda v: "" if (pd.isna(v) or abs(v) < 1e-9) else f"{v:,.0f}"
        )
    if "Golongan 2" in cols:
        fmt_dict["Golongan 2"] = (
            lambda v: "" if (pd.isna(v) or abs(v) < 1e-9) else f"{v:,.0f}"
        )
    if "Golongan 3" in cols:
        fmt_dict["Golongan 3"] = (
            lambda v: "" if (pd.isna(v) or abs(v) < 1e-9) else f"{v:,.0f}"
        )

    # Qp 列（5日ステップ）のフォーマット
    qp_cols = [lab for lab in ordered_labels if lab in cols]
    for lab in qp_cols:
        fmt_dict[lab] = (
            lambda v, lab=lab: "" if (v == 0 or pd.isna(v)) else f"{v:,.3f}"
        )

    # 今日の 5Day_ID → ラベル（例: Dec-2）
    today = dt.date.today()
    t_idx = date_to_5day_index(today)
    today_label_local = labels[t_idx] if 0 <= t_idx < len(labels) else None

    # ---- Styler 構築 ----
    styler = df_sty.style.format(fmt_dict)

    # ★ データバー適用部
    if qp_cols:
        vals = pd.to_numeric(
            df_sty[qp_cols].stack(),
            errors="coerce"
        )
        vals = vals.replace([np.inf, -np.inf], np.nan).dropna()

        if not vals.empty:
            vmin = float(vals.min())
            vmax = float(vals.max())
            if vmin != vmax:
                styler = styler.bar(
                    subset=pd.IndexSlice[:, qp_cols],
                    axis=0,
                    color=bar_color,
                    vmin=vmin,
                    vmax=vmax,
                )

    # MT 開始セル / TB 行 / Branch 行のハイライトなど
    styler = (
        styler
        .apply(highlight_func, axis=1)
        .set_properties(**{
            "text-align": "center",
            "padding": "2px 4px",
            "font-size": "12px",
            "white-space": "nowrap",
        })
        .set_properties(
            subset=["Area (ha)"],
            **{"border-right": "2px solid #666"}
        )
        .set_properties(
            subset=[c for c in ["Canal", "ParentName", "ChildName", "Area (ha)"]
                    if c in df_sty.columns],
            **{"font-weight": "bold"}
        )
        .set_table_styles([
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
                    ("background-color",
                     "#e5ffe5" if level_name == "Main" else "#e5f0ff"),
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
    )

    # 今日の列（Dec-2 など）は最後に黄色で上書き
    if today_label_local is not None and today_label_local in df_sty.columns:
        styler = styler.set_properties(
            subset=[today_label_local],
            **{"background-color": "#fff2a8", "font-weight": "bold"}
        )

    # 列幅を表4と共通化
    styler = apply_common_col_widths(styler, df_sty)

    return styler

qp_cols_plan = [lab for lab in ordered_labels if lab in display_plan_for_style.columns]

highlight_plan = make_highlight_func(
    qp_cols_plan,
    mt1_idx_rows,
    mt2_idx_rows,
    mt3_idx_rows,
    shift_steps=0,
)

styled_plan = style_qp_table(
    display_plan_for_style,
    level_name,
    highlight_plan,
    bar_color="#99ccff",   # planned Qp: light blue
)

wrapper_style = "text-align:left; overflow-x:auto; max-height:420px; overflow-y:auto;"
html_plan = f"<div style='{wrapper_style}'>{styled_plan.to_html()}</div>"

st.markdown(html_plan, unsafe_allow_html=True)

# Legend for Planned Qp table
st.markdown(
    """
    <div style="margin-top:4px; font-size:0.9rem;">
      <div style="display:flex; flex-wrap:wrap; gap:16px; align-items:center;">
        <div>
          <span style="display:inline-block;width:14px;height:14px;
                       background-color:#ffcc66;border:1px solid #999;
                       margin-right:6px;vertical-align:middle;"></span>
          <strong>Orange</strong>: Start 5Day_ID of <strong>MT-1 / MT-2 / MT-3</strong> (planned schedule)
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

# ========================================
# 4. Adjusted Qp (shifted start)
# ========================================
st.header("4. Adjusted Qp (shifted start)")

st.markdown("""
This section shows **Adjusted Qp (m³/s)** using the NFR scenario
selected in **6. NFR scenario preview (per Secondary Canal)**.

If no scenario is selected yet, the table will still be based on the
original planned Qp.
""")

# ★ 上部 Adjusted Qp テーブル用のコンテナ（中身はセクション6で描画）
adj_top_container = st.container()

# ★ 5-day Qp グラフ用コンテナ（中身は 6 番で描画）
qp_graphs_container = st.container()

st.info("""
Top table: Planned Qp (fixed schedule).  
Bottom table: Adjusted Qp (sowing date shifted by 5-day steps).  
Orange cells: start 5Day_ID of MT-1 / MT-2 / MT-3 for each row.
""")


st.markdown("<hr style='margin:1.5rem 0; border:0; border-top:2px solid #e5e7eb;'>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="
        padding:8px 12px;
        margin-bottom:8px;
        border-radius:6px;
        background:#eef2ff;
        border:1px solid #c7d2fe;
        font-weight:bold;
        font-size:0.95rem;
        ">
      Advanced section – Condition & Detail of NFR Calculation Result
    </div>
    """,
    unsafe_allow_html=True,
)


# ========================================
# 5. Condition settings for Qp/NFR scenario (per Canal)
# ========================================
with st.expander("5. Condition settings for Qp/NFR scenario (per Canal)", expanded=False):
    st.subheader("5. Condition settings for Qp/NFR scenario (per Canal)")

    st.markdown("""
    Adjust start dates **per Canal × Golongan** using 5-day steps.

    - **Scope**: switch between **Secondary Canals** and **Main-level TB (Main × Golongan)**  
    - **MT-1/2/3 Base**: base dates imported from Page2 (Bank × Golongan)  
    - **Adj MT-1/2/3 (5d)**: shift in 5-day steps (negative = earlier, positive = later)  
    - **MT-1/2/3 Adj**: adjusted dates (YY-MM-DD)
    """)

    # ---------- Secondary / Main TB source rows ----------
    sec_rows = df[df["CanalLevel"] == "Secondary"].copy()
    main_rows = df[df["CanalLevel"] == "Main"].copy()

    if sec_rows.empty and main_rows.empty:
        st.info("No canals found for start-date settings.")
    else:
        # --- Main canal name map: (Bank, MainKey) -> MainCanal name ---
        main_name_map = {}
        for _, r in main_rows.iterrows():
            b = r["Bank"]
            mk = r["MainKey"]
            if not mk:
                continue
            key = (b, mk)
            if key not in main_name_map:
                main_name_map[key] = r["Canal"]

        # Unique Secondary rows (Bank, MainKey, Canal)
        sec_unique = (
            sec_rows[["Bank", "MainKey", "Canal"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        prev_cfg = st.session_state.get("sec_start_config", None)

        # ---------- Build Bank×Golongan×Season -> base dates (MT-1/2/3) from Page2 ----------
        lp = st.session_state.get("landprep_df", None)
        bank_gol_to_mt: dict[tuple[str, int], dict[str, dt.date]] = {}

        if isinstance(lp, pd.DataFrame) and not lp.empty and \
                "Group" in lp.columns and \
                "MT1_start" in lp.columns and \
                "MT2_start" in lp.columns and \
                "MT3_start" in lp.columns:
            for _, r in lp.iterrows():
                grp = str(r["Group"])
                parts = grp.split("_", 1)
                if len(parts) != 2:
                    continue
                bank = parts[0].strip()
                rest = parts[1]

                m_g = re.search(r"(\d+)", rest)
                if not m_g:
                    continue
                gol_idx = int(m_g.group(1))

                mt_map = bank_gol_to_mt.setdefault((bank, gol_idx), {})

                # MT-1 / MT-2 は常に対象
                for season, col in [("MT-1", "MT1_start"), ("MT-2", "MT2_start")]:
                    v = r.get(col, None)
                    if v is None or pd.isna(v):
                        continue
                    try:
                        d = pd.to_datetime(v).date()
                    except Exception:
                        continue
                    mt_map[season] = d

                # MT-3: active のときだけ Base を持つ
                if "MT3_start" in lp.columns:
                    active_flag = True
                    if "MT3_active" in lp.columns:
                        active_flag = r.get("MT3_active", True)
                        try:
                            active_flag = bool(active_flag)
                        except Exception:
                            active_flag = True
                    if active_flag:
                        v3 = r.get("MT3_start", None)
                        if v3 is not None and not pd.isna(v3):
                            try:
                                d3 = pd.to_datetime(v3).date()
                                mt_map["MT-3"] = d3
                            except Exception:
                                pass

        # Fallback: use mt*_common if nothing found
        if not bank_gol_to_mt:
            mt1_common = st.session_state.get("mt1_common", None)
            mt2_common = st.session_state.get("mt2_common", None)
            mt3_common = st.session_state.get("mt3_common", None)

            def to_date(v):
                if v is None or pd.isna(v):
                    return None
                try:
                    return pd.to_datetime(v).date()
                except Exception:
                    return None

            d1 = to_date(mt1_common)
            d2 = to_date(mt2_common)
            d3 = to_date(mt3_common)

            for bank in ["Left", "Right"]:
                for gol in [1, 2, 3]:
                    mt_map = bank_gol_to_mt.setdefault((bank, gol), {})
                    if d1 is not None:
                        mt_map.setdefault("MT-1", d1)
                    if d2 is not None:
                        mt_map.setdefault("MT-2", d2)
                    if d3 is not None:
                        mt_map.setdefault("MT-3", d3)

        # ---------- Build base config rows: Secondary + Main TB ----------
        cfg_rows = []

        # Secondary-level rows (Scope = "Secondary")
        for _, r in sec_unique.iterrows():
            bank = r["Bank"]
            main_key = r["MainKey"]
            sec_name = r["Canal"]

            parent_name = ""
            parent_key = ""
            sec_canal_key = ""
            sec_child_key = ""

            cand_sec = sec_rows[
                (sec_rows["Bank"] == bank) &
                (sec_rows["Canal"] == sec_name)
            ]
            if not cand_sec.empty:
                parent_name = str(cand_sec.iloc[0]["ParentName"])
                parent_key = str(cand_sec.iloc[0]["ParentKey"])
                sec_canal_key = str(cand_sec.iloc[0]["CanalKey"])
                sec_child_key = str(cand_sec.iloc[0]["ChildKey"])

            main_name = main_name_map.get((bank, main_key), "")

            for gol in [1, 2, 3]:
                col_area = f"Area_G{gol}"
                has_area = False
                if col_area in df.columns:
                    mask = (
                        (df["Bank"] == bank) &
                        (df["Canal"] == sec_name) &
                        (df[col_area].fillna(0.0) > 0)
                    )
                    has_area = bool(mask.any())

                if not has_area:
                    continue

                mt_map = bank_gol_to_mt.get((bank, gol), {})
                mt1_base = mt_map.get("MT-1", None)
                mt2_base = mt_map.get("MT-2", None)
                mt3_base = mt_map.get("MT-3", None)

                cfg_rows.append({
                    "Scope": "Secondary",
                    "Bank": bank,
                    "MainCanal": main_name,
                    "ParentName": parent_name,
                    "ParentKey": parent_key,
                    "SecondaryCanal": sec_name,
                    "Golongan": gol,
                    "SecCanalKey": sec_canal_key,
                    "SecChildKey": sec_child_key,
                    "MT1_base": mt1_base,
                    "MT2_base": mt2_base,
                    "MT3_base": mt3_base,
                    "Adj_MT1": 0,
                    "Adj_MT2": 0,
                    "Adj_MT3": 0,
                })

        # Main-level TB rows (Scope = "Main")
        main_unique = (
            main_rows[["Bank", "MainKey", "Canal"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        for _, r in main_unique.iterrows():
            bank = r["Bank"]
            main_key = r["MainKey"]
            main_canal = r["Canal"]

            for gol in [1, 2, 3]:
                # --- 1) この Main × Golongan に TB 面積があるかチェック ---
                col_area = f"Area_G{gol}"
                has_area_direct = False
                if col_area in df.columns:
                    mask = (
                        (df["Bank"] == bank) &
                        (df["Canal"] == main_canal) &
                        (df["Class"] == "TB") &
                        (df[col_area].fillna(0.0) > 0)
                    )
                    has_area_direct = bool(mask.any())

                # TB が全く無ければ、この Golongan 行は作らない
                if not has_area_direct:
                    continue

                # --- 2) Bank×Golongan レベルの Base 日付を取得 ---
                mt_map = bank_gol_to_mt.get((bank, gol), {})
                mt1_base = mt_map.get("MT-1", None)
                mt2_base = mt_map.get("MT-2", None)
                mt3_base = mt_map.get("MT-3", None)

                # Base が 1つも無ければ意味がないのでスキップ
                if mt1_base is None and mt2_base is None and mt3_base is None:
                    continue

                cfg_rows.append({
                    "Scope": "Main",
                    "Bank": bank,
                    "MainCanal": main_canal,
                    "ParentName": "",
                    "ParentKey": "",
                    "SecondaryCanal": "",
                    "Golongan": gol,
                    "SecCanalKey": main_key,
                    "SecChildKey": "",
                    "MT1_base": mt1_base,
                    "MT2_base": mt2_base,
                    "MT3_base": mt3_base,
                    "Adj_MT1": 0,
                    "Adj_MT2": 0,
                    "Adj_MT3": 0,
                })

        if cfg_rows:
            base_cfg = pd.DataFrame(cfg_rows)
            base_cfg = (
                base_cfg
                .sort_values(["Scope", "Bank", "SecCanalKey", "SecChildKey", "Golongan"])
                .reset_index(drop=True)
            )
        else:
            base_cfg = pd.DataFrame(
                columns=[
                    "Scope",
                    "Bank", "MainCanal", "ParentName", "ParentKey", "SecondaryCanal",
                    "Golongan", "SecCanalKey", "SecChildKey",
                    "MT1_base", "MT2_base", "MT3_base",
                    "Adj_MT1", "Adj_MT2", "Adj_MT3",
                ]
            )

        # ---------- Merge previous adjustments (if any) ----------
        if isinstance(prev_cfg, pd.DataFrame) and not prev_cfg.empty:
            key_cols = [
                "Scope", "Bank", "MainCanal", "ParentName", "ParentKey",
                "SecondaryCanal", "Golongan", "SecCanalKey", "SecChildKey",
            ]
            cols_to_merge = key_cols + [c for c in ["Adj_MT1", "Adj_MT2", "Adj_MT3"] if c in prev_cfg.columns]
            base_cfg = base_cfg.merge(
                prev_cfg[cols_to_merge],
                on=key_cols,
                how="left",
                suffixes=("", "_old"),
            )

            for col in ["Adj_MT1", "Adj_MT2", "Adj_MT3"]:
                old_col = col + "_old"
                if old_col in base_cfg.columns:
                    base_cfg[col] = base_cfg[old_col].where(base_cfg[old_col].notna(), base_cfg[col])
                    base_cfg = base_cfg.drop(columns=[old_col])

        st.session_state["sec_start_config"] = base_cfg.copy()
        sec_cfg_all = st.session_state["sec_start_config"]

        if sec_cfg_all.empty:
            st.info("No rows found for canal start settings.")
        else:
            # ---------- 共通フィルタ (Scope / Bank / Main / Golongan) ----------
            st.markdown("### 5-A. Filters for canal start settings and scenario")
            st.markdown(
                "These filters apply to **both** the table below and "
                "**6. NFR scenario preview (per Secondary Canal)**."
            )

            col_scope, col_f_bank, col_f_main, col_f_gol = st.columns([1.4, 1.2, 2.0, 1.2])

            with col_scope:
                st.markdown("**Filter – Scope (table)**")
                scope_sel = st.radio(
                    "Scope (for start-date table)",
                    options=["Secondary", "Main"],
                    horizontal=True,
                    key="sec_cfg_scope_filter",
                )

            cfg_scope = sec_cfg_all[sec_cfg_all["Scope"] == scope_sel].copy()

            with col_f_bank:
                st.markdown("**Filter – Bank**")
                bank_sel = st.radio(
                    "Bank (Left / Right)",
                    options=["Left", "Right"],
                    horizontal=True,
                    key="sec_cfg_bank_filter",
                )

            cfg_bank = cfg_scope[cfg_scope["Bank"] == bank_sel].copy()
            if cfg_bank.empty:
                st.info(f"No settings for {bank_sel} bank in this scope.")
            else:
                with col_f_main:
                    st.markdown("**Filter – Main canal**")
                    main_order = (
                        df[(df["Bank"] == bank_sel) & (df["CanalLevel"] == "Main")]
                        .sort_values("CanalKey")
                        [["Canal"]]
                        .drop_duplicates()
                    )
                    main_list = main_order["Canal"].tolist()
                    main_label = st.selectbox(
                        "Main canal (for filtering)",
                        options=["(All)"] + main_list,
                        index=0,
                        key="sec_cfg_main_filter",
                    )

                with col_f_gol:
                    st.markdown("**Filter – Golongan**")
                    gol_list = sorted(cfg_bank["Golongan"].unique())
                    gol_label = st.selectbox(
                        "Golongan (for filtering)",
                        options=["(All)"] + [str(g) for g in gol_list],
                        index=0,
                        key="sec_cfg_gol_filter",
                    )

                # この後の cfg_bank にフィルタを適用
                if main_label != "(All)":
                    cfg_bank = cfg_bank[cfg_bank["MainCanal"] == main_label]
                if gol_label != "(All)":
                    cfg_bank = cfg_bank[cfg_bank["Golongan"] == int(gol_label)]

                if cfg_bank.empty:
                    st.info("No rows after applying filters.") 

                else:
                    if "SecCanalKey" in cfg_bank.columns and "SecChildKey" in cfg_bank.columns:
                        cfg_bank = cfg_bank.sort_values(
                            ["SecCanalKey", "SecChildKey", "Golongan"],
                            kind="mergesort"
                        )

                    cfg_bank = cfg_bank.reset_index()
                    original_index = cfg_bank["index"]

                    # ★ 調整量は常に整数として扱う
                    for col_name in ["Adj_MT1", "Adj_MT2", "Adj_MT3"]:
                        if col_name in cfg_bank.columns:
                            cfg_bank[col_name] = cfg_bank[col_name].fillna(0).astype(int)

                    def fmt_date(v):
                        if pd.isna(v):
                            return ""
                        try:
                            return pd.to_datetime(v).strftime("%y-%m-%d")
                        except Exception:
                            return ""

                    def apply_adj(base, adj_steps):
                        return shift_date_by_5day_steps_str(base, adj_steps)

                    cfg_bank["MT1_base_str"] = cfg_bank["MT1_base"].apply(fmt_date)
                    cfg_bank["MT2_base_str"] = cfg_bank["MT2_base"].apply(fmt_date)
                    cfg_bank["MT3_base_str"] = cfg_bank["MT3_base"].apply(fmt_date)
                    cfg_bank["MT1_adj_str"] = [
                        apply_adj(b, a) for b, a in zip(cfg_bank["MT1_base"], cfg_bank["Adj_MT1"])
                    ]
                    cfg_bank["MT2_adj_str"] = [
                        apply_adj(b, a) for b, a in zip(cfg_bank["MT2_base"], cfg_bank["Adj_MT2"])
                    ]
                    cfg_bank["MT3_adj_str"] = [
                        apply_adj(b, a) for b, a in zip(cfg_bank["MT3_base"], cfg_bank["Adj_MT3"])
                    ]

                    cfg_bank["MainCanal_disp"] = cfg_bank["MainCanal"]
                    cfg_bank.loc[cfg_bank["MainCanal_disp"].duplicated(), "MainCanal_disp"] = ""

                    st.markdown("**Editing table – canal start adjustments (visible rows)**")

                    edit_df = cfg_bank[[
                        "MainCanal_disp", "ParentName", "SecondaryCanal", "Golongan",
                        "MT1_base_str", "MT1_adj_str", "Adj_MT1",
                        "MT2_base_str", "MT2_adj_str", "Adj_MT2",
                        "MT3_base_str", "MT3_adj_str", "Adj_MT3",
                    ]].copy()
                    edit_df.insert(0, "No.", np.arange(1, len(edit_df) + 1))

                    col_top_left, col_top_right = st.columns([4, 1])
                    with col_top_right:
                        if st.button("Reset Adj (visible)", key="btn_reset_adj_visible"):
                            sec_cfg_all.loc[original_index, ["Adj_MT1", "Adj_MT2", "Adj_MT3"]] = 0
                            st.session_state["sec_start_config"] = sec_cfg_all
                            st.rerun()

                    # Editing table 全体のフォントサイズを少し大きくする（data_editor 用）
                    st.markdown(
                        """
                        <style>
                        [data-testid="stDataFrame"] table td,
                        [data-testid="stDataFrame"] table th {
                            font-size: 0.95rem;
                        }
                        </style>
                        """,
                        unsafe_allow_html=True,
                    )

                    edited = st.data_editor(
                        edit_df,
                        num_rows="fixed",
                        hide_index=True,
                        key="sec_cfg_editor",
                        width="content",
                        column_config={
                            "No.": st.column_config.NumberColumn("No.", disabled=True),
                            "MainCanal_disp": st.column_config.Column("Main Canal", disabled=True),
                            "ParentName": st.column_config.Column("Parent Name", disabled=True),
                            "SecondaryCanal": st.column_config.Column("Secondary Canal", disabled=True),
                            "Golongan": st.column_config.NumberColumn("Golongan", disabled=True),
                            "MT1_base_str": st.column_config.Column("MT-1 Base", disabled=True),
                            "MT1_adj_str": st.column_config.Column("MT-1 Adj", disabled=True),
                            "Adj_MT1": st.column_config.NumberColumn("Adj MT-1 (5d)", step=1, format="%d"),
                            "MT2_base_str": st.column_config.Column("MT-2 Base", disabled=True),
                            "MT2_adj_str": st.column_config.Column("MT-2 Adj", disabled=True),
                            "Adj_MT2": st.column_config.NumberColumn("Adj MT-2 (5d)", step=1, format="%d"),
                            "MT3_base_str": st.column_config.Column("MT-3 Base", disabled=True),
                            "MT3_adj_str": st.column_config.Column("MT-3 Adj", disabled=True),
                            "Adj_MT3": st.column_config.NumberColumn("Adj MT-3 (5d)", step=1, format="%d"),
                        },
                    )

                    st.markdown("**Apply and bulk adjustments (for visible rows)**")
                    col_apply = st.columns([1])[0]
                    with col_apply:
                        if st.button("Apply adjustments (manual edits)", key="btn_apply_sec_manual"):
                            sec_cfg_all.loc[original_index, "Adj_MT1"] = edited["Adj_MT1"].astype(int).to_numpy()
                            sec_cfg_all.loc[original_index, "Adj_MT2"] = edited["Adj_MT2"].astype(int).to_numpy()
                            sec_cfg_all.loc[original_index, "Adj_MT3"] = edited["Adj_MT3"].astype(int).to_numpy()
                            st.session_state["sec_start_config"] = sec_cfg_all
                            st.success("Adjustments have been updated.")
                            st.rerun()

                    # ★ 調整単位の説明（赤字）
                    st.markdown(
                        "<span style='color:#b91c1c; font-size:1.2rem;'>* The values ​​of Adj MT-1 / MT-2 / MT-3 are all integers in 5-day steps.</span>",
                        unsafe_allow_html=True,
                    )

                    # bulk ±1 buttons for MT-1 / MT-2 / MT-3（全て 1 行）
                    col_btn1, col_btn2, col_btn3, col_btn4, col_btn5, col_btn6 = st.columns(6)
                    with col_btn1:
                        if st.button("🔵 −1 MT-1 (visible)", key="btn_dec_mt1"):
                            sec_cfg_all.loc[original_index, "Adj_MT1"] -= 1
                            st.session_state["sec_start_config"] = sec_cfg_all
                            st.rerun()
                    with col_btn2:
                        if st.button("🔴 +1 MT-1 (visible)", key="btn_inc_mt1"):
                            sec_cfg_all.loc[original_index, "Adj_MT1"] += 1
                            st.session_state["sec_start_config"] = sec_cfg_all
                            st.rerun()
                    with col_btn3:
                        if st.button("🔵 −1 MT-2 (visible)", key="btn_dec_mt2"):
                            sec_cfg_all.loc[original_index, "Adj_MT2"] -= 1
                            st.session_state["sec_start_config"] = sec_cfg_all
                            st.rerun()
                    with col_btn4:
                        if st.button("🔴 +1 MT-2 (visible)", key="btn_inc_mt2"):
                            sec_cfg_all.loc[original_index, "Adj_MT2"] += 1
                            st.session_state["sec_start_config"] = sec_cfg_all
                            st.rerun()
                    with col_btn5:
                        if st.button("🔵 −1 MT-3 (visible)", key="btn_dec_mt3"):
                            sec_cfg_all.loc[original_index, "Adj_MT3"] -= 1
                            st.session_state["sec_start_config"] = sec_cfg_all
                            st.rerun()
                    with col_btn6:
                        if st.button("🔴 +1 MT-3 (visible)", key="btn_inc_mt3"):
                            sec_cfg_all.loc[original_index, "Adj_MT3"] += 1
                            st.session_state["sec_start_config"] = sec_cfg_all
                            st.rerun()

                    st.markdown("**Preview (Adj>0 red, Adj<0 blue – YY-MM-DD)**")

                    cfg_bank_latest = st.session_state["sec_start_config"]
                    cfg_bank_latest = cfg_bank_latest[
                        (cfg_bank_latest["Scope"] == scope_sel) &
                        (cfg_bank_latest["Bank"] == bank_sel)
                    ].copy()
                    if main_label != "(All)":
                        cfg_bank_latest = cfg_bank_latest[cfg_bank_latest["MainCanal"] == main_label]
                    if gol_label != "(All)":
                        cfg_bank_latest = cfg_bank_latest[cfg_bank_latest["Golongan"] == int(gol_label)]

                    if cfg_bank_latest.empty:
                        st.info("No rows to preview after filters.")
                    else:
                        if "SecCanalKey" in cfg_bank_latest.columns and "SecChildKey" in cfg_bank_latest.columns:
                            cfg_bank_latest = cfg_bank_latest.sort_values(
                                ["SecCanalKey", "SecChildKey", "Golongan"],
                                kind="mergesort"
                            )

                        cfg_bank_latest = cfg_bank_latest.reset_index(drop=True)

                        # ★ Preview 用にも調整量は整数にそろえる
                        for col_name in ["Adj_MT1", "Adj_MT2", "Adj_MT3"]:
                            if col_name in cfg_bank_latest.columns:
                                cfg_bank_latest[col_name] = cfg_bank_latest[col_name].fillna(0).astype(int)

                        cfg_bank_latest["MT1_base_str"] = cfg_bank_latest["MT1_base"].apply(fmt_date)
                        cfg_bank_latest["MT2_base_str"] = cfg_bank_latest["MT2_base"].apply(fmt_date)
                        cfg_bank_latest["MT3_base_str"] = cfg_bank_latest["MT3_base"].apply(fmt_date)
                        cfg_bank_latest["MT1_adj_str"] = [
                            apply_adj(b, a) for b, a in zip(cfg_bank_latest["MT1_base"], cfg_bank_latest["Adj_MT1"])
                        ]
                        cfg_bank_latest["MT2_adj_str"] = [
                            apply_adj(b, a) for b, a in zip(cfg_bank_latest["MT2_base"], cfg_bank_latest["Adj_MT2"])
                        ]
                        cfg_bank_latest["MT3_adj_str"] = [
                            apply_adj(b, a) for b, a in zip(cfg_bank_latest["MT3_base"], cfg_bank_latest["Adj_MT3"])
                        ]
                        cfg_bank_latest["MainCanal_disp"] = cfg_bank_latest["MainCanal"]
                        cfg_bank_latest.loc[cfg_bank_latest["MainCanal_disp"].duplicated(), "MainCanal_disp"] = ""

                        preview = cfg_bank_latest[[
                            "MainCanal_disp", "ParentName", "SecondaryCanal", "Golongan",
                            "MT1_base_str", "MT1_adj_str", "Adj_MT1",
                            "MT2_base_str", "MT2_adj_str", "Adj_MT2",
                            "MT3_base_str", "MT3_adj_str", "Adj_MT3",
                        ]].copy()
                        preview.index = np.arange(1, len(preview) + 1)

                        def adj_color_factory(df_local: pd.DataFrame):
                            adj_mt1 = df_local["Adj_MT1"]
                            adj_mt2 = df_local["Adj_MT2"]
                            adj_mt3 = df_local["Adj_MT3"]

                            def _adj_color(col: pd.Series):
                                styles = [""] * len(col)
                                if col.name in ["Adj_MT1", "MT1_adj_str"]:
                                    for i, v in enumerate(adj_mt1):
                                        if v > 0:
                                            styles[i] = "background-color: #fecaca"
                                        elif v < 0:
                                            styles[i] = "background-color: #bfdbfe"
                                elif col.name in ["Adj_MT2", "MT2_adj_str"]:
                                    for i, v in enumerate(adj_mt2):
                                        if v > 0:
                                            styles[i] = "background-color: #fecaca"
                                        elif v < 0:
                                            styles[i] = "background-color: #bfdbfe"
                                elif col.name in ["Adj_MT3", "MT3_adj_str"]:
                                    for i, v in enumerate(adj_mt3):
                                        if v > 0:
                                            styles[i] = "background-color: #fecaca"
                                        elif v < 0:
                                            styles[i] = "background-color: #bfdbfe"
                                return styles

                            return _adj_color

                        bold_cols = [
                            "MainCanal_disp",
                            "ParentName", "SecondaryCanal",
                            "MT1_adj_str", "Adj_MT1",
                            "MT2_adj_str", "Adj_MT2",
                            "MT3_adj_str", "Adj_MT3",
                        ]

                        mt1_left_col = "MT1_base_str"
                        mt1_right_col = "Adj_MT1"
                        mt2_left_col = "MT2_base_str"
                        mt2_right_col = "Adj_MT2"
                        mt3_left_col = "MT3_base_str"
                        mt3_right_col = "Adj_MT3"

                        # ★ Adj列用のフォーマット辞書（整数表示）
                        fmt_adj = {
                            col: "{:d}"
                            for col in ["Adj_MT1", "Adj_MT2", "Adj_MT3"]
                            if col in preview.columns
                        }

                        styled_prev = (
                            preview
                            .style
                            .apply(adj_color_factory(preview), axis=0)
                            .set_properties(**{
                                "font-size": "11px",
                                "white-space": "nowrap",
                                "padding": "2px 4px",
                                "text-align": "center",
                            })
                            .set_properties(
                                subset=[c for c in bold_cols if c in preview.columns],
                                **{"font-size": "15px", "font-weight": "bold"}
                            )
                            .set_properties(
                                subset=[mt1_left_col],
                                **{"border-left": "3px double #666"}
                            )
                            .set_properties(
                                subset=[mt1_right_col],
                                **{"border-right": "3px double #666"}
                            )
                            .set_properties(
                                subset=[mt2_left_col],
                                **{"border-left": "3px double #666"}
                            )
                            .set_properties(
                                subset=[mt2_right_col],
                                **{"border-right": "3px double #666"}
                            )
                            .set_properties(
                                subset=[mt3_left_col],
                                **{"border-left": "3px double #666"}
                            )
                            .set_properties(
                                subset=[mt3_right_col],
                                **{"border-right": "3px double #666"}
                            )
                            .set_table_styles([
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
                                        ("background-color", "#e5f0ff"),
                                        ("font-weight", "bold"),
                                        ("border", "1px solid #999"),
                                        ("padding", "2px 4px"),
                                        ("white-space", "nowrap"),
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
                                    "selector": "td",
                                    "props": [
                                        ("border", "1px solid #999"),
                                        ("padding", "2px 4px"),
                                        ("white-space", "nowrap"),
                                    ],
                                },
                            ])
                        )

                        st.markdown(
                            '<div style="overflow-x:auto; text-align:left;">'
                            f'{styled_prev.to_html()}'
                            '</div>',
                            unsafe_allow_html=True,
                        )

# ========================================
# 6. NFR scenario preview (per Secondary Canal)
# ========================================
with st.expander("6. Detail of NFR calculation result (per Secondary Canal)", expanded=False):
    st.subheader("6. Detail of NFR calculation result (per Secondary Canal)")

    st.markdown("""
    This section recomputes **NFR 5-day values** for one Secondary-canal × Golongan
    scenario and compares it with the base NFR.

    The filters in **5. Canal start settings** (Bank / Main canal / Golongan)
    are also applied here.
    """)

    # ★ ここを追加 ★
    df_case_disp: pd.DataFrame | None = None
    mt_start_labels: set[str] | None = None

    # Canal start settings and base land-prep
    sec_cfg_all   = st.session_state.get("sec_start_config", None)
    landprep_base = st.session_state.get("landprep_df", None)

    if sec_cfg_all is None or landprep_base is None:
        st.info(
            "To use this preview:\n"
            "• Compute NFR on Page2 (so that `nfr_5day` and `landprep_df` exist in session_state).\n"
            "• Configure canal start settings above (so that `sec_start_config` exists)."
        )
    else:

        # --------------------------------------------------------
        # 9.1 Base NFR/Component tables
        #    ※ 毎回 landprep_base から再計算する
        # --------------------------------------------------------
        nfr_base, comp_base = recompute_nfr_5day_for_landprep(landprep_base)
        # 必要であれば最新値をキャッシュとして保持するだけにする
        st.session_state["nfr_5day_base_recomputed"] = (nfr_base, comp_base)


        # ★ Base NFR 配列 (Bank×Golongan×5Day) – Page4 と同じ形式
        nfr_arr_base = build_nfr_arr_from_long(nfr_base)

        # ★ Page4 で計算済みのベースネットワーク ＋ Eff
        df_net_base = st.session_state.get("qp_df_network_base", None)
        eff_map_raw = st.session_state.get("qp_eff_map_raw", None)

        # ★ 5-A の Golongan フィルタだけ利用（Bank / Canal は 3 番と揃える）
        gol_filter  = st.session_state.get("sec_cfg_gol_filter", "(All)")

        cfg_sec = sec_cfg_all[sec_cfg_all["Scope"] == "Secondary"].copy()

        # 3. Planned Qp で選択している Bank に合わせる
        cfg_sec = cfg_sec[cfg_sec["Bank"] == selected_bank]

        # 3. Planned Qp で選択している Canal に合わせる（Secondary のとき）
        if level_sel == "Secondary Canal":
            cfg_sec = cfg_sec[cfg_sec["SecondaryCanal"] == selected_canal]

        # Golongan フィルタ（5-A）を適用
        if gol_filter != "(All)":
            try:
                g_filter_int = int(gol_filter)
                cfg_sec = cfg_sec[cfg_sec["Golongan"] == g_filter_int]
            except Exception:
                pass


        if cfg_sec.empty:
            st.info("No rows with current filters in the canal-start settings table (Scope=Secondary).")
        else:
            # ------------------------------------------------------------------
            # 9.2 Choose one Secondary-canal × Golongan scenario
            # ------------------------------------------------------------------
            option_labels = []
            for _, r in cfg_sec.iterrows():
                bank = str(r["Bank"])
                main = str(r["MainCanal"])
                sec  = str(r["SecondaryCanal"])
                g    = int(r["Golongan"])
                option_labels.append(f"{bank} – {main} / {sec}  (G{g})")


            sel_label = st.selectbox(
                "Select a Secondary canal × Golongan for scenario calculation",
                options=option_labels,
                index=0,
            )
            sel_row = cfg_sec.iloc[option_labels.index(sel_label)]

            bank_sel = str(sel_row["Bank"])
            gol_sel  = int(sel_row["Golongan"])

            # Build scenario land-prep from base
            landprep_case = landprep_base.copy()

            # Group key is like "Left_Golongan 1" → (Bank, 1)
            def _split_group_key(s: str):
                s = str(s)
                if "_" not in s:
                    return None, None
                bank_part, rest = s.split("_", 1)
                m = re.search(r"(\d+)", rest)
                g = int(m.group(1)) if m else None
                return bank_part, g

            # All rows in landprep that belong to this Bank × Golongan
            mask_group = landprep_case["Group"].astype(str).apply(
                lambda s: _split_group_key(s) == (bank_sel, gol_sel)
            )

            # Shift helper (5Day_ID ベースでシフト)
            def _shift_start(base, adj_steps):
                """
                base : MT*_start の元の日付
                adj_steps : 5Day_ID の増減（整数）

                1. base の 5Day_ID を求める
                2. 5Day_ID を adj_steps だけ足し引き（1..72 にクリップ）
                3. その 5Day_ID を water-year 上の日付に戻す
                """
                if base is None or pd.isna(base):
                    return None
                try:
                    base_date = pd.to_datetime(base).date()
                except Exception:
                    return None

                # 1..72 の 5Day_ID に変換
                base_id = date_to_5day_id(base_date)
                if not (1 <= base_id <= 72):
                    return None

                # 5Day_ID ベースでシフト（ここが以前と違う）
                new_id = base_id + int(adj_steps)
                new_id = max(1, min(72, new_id))

                # water-year 内のカレンダー日付に戻す
                return _five_day_id_to_date_from_base(base_date, new_id)


            # New start dates for this scenario
            new_mt1 = _shift_start(sel_row["MT1_base"], sel_row["Adj_MT1"])
            new_mt2 = _shift_start(sel_row["MT2_base"], sel_row["Adj_MT2"])
            new_mt3 = _shift_start(sel_row["MT3_base"], sel_row["Adj_MT3"])

            if mask_group.any():
                if new_mt1 is not None:
                    landprep_case.loc[mask_group, "MT1_start"] = new_mt1
                if new_mt2 is not None:
                    landprep_case.loc[mask_group, "MT2_start"] = new_mt2
                if new_mt3 is not None:
                    landprep_case.loc[mask_group, "MT3_start"] = new_mt3

            st.markdown("**Scenario season-start dates for the selected Bank × Golongan**")

            # landprep_base / landprep_case の対象行を取得（同じ mask_group で揃える）
            start_base = landprep_base[mask_group].copy()
            start_case = landprep_case[mask_group].copy()

            def _to_5day_id_safe(v):
                if v is None or pd.isna(v):
                    return None
                try:
                    d = pd.to_datetime(v).date()
                except Exception:
                    return None
                return date_to_5day_id(d)

            # MT1 / MT2 / MT3 について「5Day_ID 差分（Case - Base）」を計算
            for col in ["MT1_start", "MT2_start", "MT3_start"]:
                if col in start_base.columns and col in start_case.columns:
                    diffs = []
                    for b, c in zip(start_base[col], start_case[col]):
                        id_b = _to_5day_id_safe(b)
                        id_c = _to_5day_id_safe(c)
                        if id_b is None or id_c is None:
                            diffs.append(None)
                        else:
                            diffs.append(id_c - id_b)  # 5Day_ID の増減値
                    # 例: MT1_start → MT1_diff_5d という列名で追加
                    start_case[col.replace("_start", "_diff_5d")] = diffs

            # Case 側のテーブルを表示（Start 日付 + 差分列）
            st.dataframe(start_case.reset_index(drop=True))

            # === このシナリオの MT-1/2/3 開始 5Day ラベル集合（表4・詳細表のハイライト用） ===
            def _start_id_for_season(lp_df: pd.DataFrame, bank: str, gol: int, season: str):
                col_map = {
                    "MT-1": "MT1_start",
                    "MT-2": "MT2_start",
                    "MT-3": "MT3_start",
                }
                col = col_map.get(season)
                if col not in lp_df.columns:
                    return None

                def _split_group(s: str):
                    s = str(s)
                    if "_" not in s:
                        return None, None
                    bank_part, rest = s.split("_", 1)
                    m = re.search(r"(\d+)", rest)
                    g_val = int(m.group(1)) if m else None
                    return bank_part, g_val

                mask_local = lp_df["Group"].astype(str).apply(
                    lambda s: _split_group(s) == (bank, gol)
                )
                if not mask_local.any():
                    return None

                # MT-3 のときだけ active 行を考慮
                if season == "MT-3" and "MT3_active" in lp_df.columns:
                    try:
                        active = lp_df["MT3_active"].astype(bool)
                        mask_local = mask_local & active
                    except Exception:
                        pass

                vals_local = lp_df.loc[mask_local, col].dropna()
                if vals_local.empty:
                    return None

                try:
                    dates_local = pd.to_datetime(vals_local).dt.date
                except Exception:
                    return None

                d_min_local = dates_local.min()
                return date_to_5day_id(d_min_local)

            # ★ 実際のラベル集合（例 {"Dec-2","Jun-1","Aug-3"}）
            mt_start_labels = set()
            for season_name in ["MT-1", "MT-2", "MT-3"]:
                sid = _start_id_for_season(landprep_case, bank_sel, gol_sel, season_name)
                if sid is None or pd.isna(sid):
                    continue
                idx = int(sid) - 1
                if 0 <= idx < len(labels):
                    mt_start_labels.add(labels[idx])


            # ------------------------------------------------------------------
            # 9.3 Recompute NFR for this scenario
            # ------------------------------------------------------------------
            nfr_case, comp_case = recompute_nfr_5day_for_landprep(landprep_case)

            if nfr_case is None or nfr_case.empty:
                st.info("Failed to recompute NFR for this scenario (see messages above).")
            else:
                # -----------------------------
                # 9.3.1 NFR (L/s/ha) pivot: Base vs Case
                # -----------------------------
                render_nfr_pivot_base_vs_case(
                    nfr_base,
                    nfr_case,
                    bank_sel,
                    gol_sel,
                    landprep_base,
                    landprep_case,
                )

                # -----------------------------
                # 9.3.2 Component breakdown (Base vs Case, side-by-side, vertical calendar)
                # -----------------------------
                st.subheader("Component breakdown for this scenario (LP, P, WLr, ETc, Re, NFR)")

                comp_base_sel = comp_base[
                    (comp_base["Bank"] == bank_sel) &
                    (comp_base["Golongan"] == gol_sel)
                ].copy()
                comp_case_sel = comp_case[
                    (comp_case["Bank"] == bank_sel) &
                    (comp_case["Golongan"] == gol_sel)
                ].copy()

                if comp_base_sel.empty or comp_case_sel.empty:
                    st.info("Component tables for this Bank × Golongan are not available.")
                else:
                    # 1) まず同じ Group を 1つ選ぶ（Bank×Golongan で通常1つ）
                    target_group = comp_base_sel["Group"].iloc[0]
                    comp_base_sel = comp_base_sel[comp_base_sel["Group"] == target_group]
                    comp_case_sel = comp_case_sel[comp_case_sel["Group"] == target_group]

                    # 2) MT を切り替えるフィルタ（元の仕様に戻す）
                    seasons_available = sorted(
                        set(comp_base_sel["Season"].unique()) |
                        set(comp_case_sel["Season"].unique())
                    )
                    season_sel = st.radio(
                        "Season",
                        options=seasons_available,
                        horizontal=True,
                        key="comp_season_filter",
                    )

                    cb = comp_base_sel[comp_base_sel["Season"] == season_sel].copy()
                    cc = comp_case_sel[comp_case_sel["Season"] == season_sel].copy()

                    if cb.empty and cc.empty:
                        st.info(f"No component rows for Season={season_sel}.")
                    else:
                        # 3) 開始 5Day_ID（Base / Case）を取得 → NFR_lps の着色に使う
                        def _start_id_from_lp(lp_df, group_key, col_name):
                            mask = (lp_df["Group"].astype(str) == group_key)
                            if not mask.any() or col_name not in lp_df.columns:
                                return None
                            v = lp_df.loc[mask, col_name].iloc[0]
                            if pd.isna(v):
                                return None
                            try:
                                return date_to_5day_id(pd.to_datetime(v).date())
                            except Exception:
                                return None

                        def _id_to_month_step(id_):
                            """5Day_ID(1..72) → (Month, Step)"""
                            if id_ is None or pd.isna(id_):
                                return None, None
                            i = int(id_) - 1
                            if i < 0 or i >= 72:
                                return None, None
                            m = water_month_order[i // 6]
                            s = (i % 6) + 1
                            return m, s

                        start_id_base = {
                            "MT-1": _start_id_from_lp(landprep_base, target_group, "MT1_start"),
                            "MT-2": _start_id_from_lp(landprep_base, target_group, "MT2_start"),
                            "MT-3": _start_id_from_lp(landprep_base, target_group, "MT3_start"),
                        }.get(season_sel)

                        start_id_case = {
                            "MT-1": _start_id_from_lp(landprep_case, target_group, "MT1_start"),
                            "MT-2": _start_id_from_lp(landprep_case, target_group, "MT2_start"),
                            "MT-3": _start_id_from_lp(landprep_case, target_group, "MT3_start"),
                        }.get(season_sel)

                        start_ms_base = _id_to_month_step(start_id_base)
                        start_ms_case = _id_to_month_step(start_id_case)

                        # 4) 縦方向のカレンダー用テーブルを作成
                        def make_vertical_table(df_src: pd.DataFrame) -> pd.DataFrame:
                            df_v = df_src.copy()
                            # 5Day_ID を計算（調整の確認用）
                            def _to_id(r):
                                try:
                                    m = int(r["Month"])
                                    s = int(r["Step"])
                                except Exception:
                                    return None
                                try:
                                    m_idx = water_month_order.index(m)
                                except ValueError:
                                    return None
                                return m_idx * 6 + s  # 1..72

                            df_v["5Day_ID"] = df_v.apply(_to_id, axis=1)

                            # 表示する列だけに絞る
                            cols_order = [
                                "Month", "Step", "5Day_ID",
                                "LP", "P", "WLr", "ETc", "Re", "NFR_mm", "NFR_lps",
                            ]
                            df_v = df_v[cols_order]

                            # 5Day_ID で昇順にソート
                            df_v = df_v.sort_values("5Day_ID", ignore_index=True)

                            return df_v

                        base_vert = make_vertical_table(cb)
                        case_vert = make_vertical_table(cc)

                        # 5) スタイリング（開始日の NFR_lps をオレンジで着色）
                        num_cols = ["LP", "P", "WLr", "ETc", "Re", "NFR_mm", "NFR_lps"]

                        def fmt_component(v):
                            if pd.isna(v) or abs(v) < 1e-9:
                                return ""
                            return f"{v:.3f}"

                        def style_vertical(df_v: pd.DataFrame, start_ms: tuple[int | None, int | None]):
                            start_month, start_step = start_ms

                            def _highlight(row: pd.Series):
                                styles = []
                                for col in row.index:
                                    if (
                                        col == "NFR_lps"
                                        and start_month is not None
                                        and start_step is not None
                                        and int(row.get("Month", -1)) == int(start_month)
                                        and int(row.get("Step", -1)) == int(start_step)
                                    ):
                                        styles.append("background-color:#ffcc66;")
                                    else:
                                        styles.append("")
                                return styles

                            fmt_dict = {col: fmt_component for col in num_cols if col in df_v.columns}

                            styler = (
                                df_v.style
                                .format(fmt_dict)
                                .apply(_highlight, axis=1)
                                .set_properties(**{
                                    "text-align": "center",
                                    "padding": "3px 4px",
                                    "font-size": "13px",      # ★ 少し大きく
                                    "white-space": "nowrap",
                                })
                                # NFR_lps 列は太字
                                .set_properties(
                                    subset=["NFR_lps"],
                                    **{"font-weight": "bold"}
                                )
                                .set_table_styles([
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
                                            ("background-color", "#e5f0ff"),
                                            ("font-weight", "bold"),
                                            ("border", "1px solid #999"),
                                            ("padding", "2px 4px"),
                                            ("white-space", "nowrap"),
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
                                            ("padding", "2px 3px"),
                                            ("white-space", "nowrap"),
                                        ],
                                    },
                                ])
                            )
                            return styler

                        sty_base = style_vertical(base_vert, start_ms_base)
                        sty_case = style_vertical(case_vert, start_ms_case)

                        col_left, col_right = st.columns(2)

                        with col_left:
                            st.markdown("**Base (vertical calendar)**")
                            st.markdown(sty_base.to_html(), unsafe_allow_html=True)

                        with col_right:
                            st.markdown("**Case (vertical calendar)**")
                            st.markdown(sty_case.to_html(), unsafe_allow_html=True)

                # -----------------------------
                # 9.3.3 Adjusted Qp (この NFR シナリオに基づく Qp 再計算)
                # -----------------------------
                if df_net_base is None:
                    st.info(
                        "Adjusted Qp を計算するには、先に Page4 で Qp 計算を実行しておく必要があります。"
                    )
                else:
                    # シナリオ側の NFR 配列を構築（全 Bank×Golongan 分）
                    nfr_arr_case_all = build_nfr_arr_from_long(nfr_case)

                    # Base 配列をコピーして、選択した Bank×Golongan だけ Case 値で上書き
                    nfr_arr_adj: dict[str, dict[int, np.ndarray]] = {}
                    for b, gmap in nfr_arr_base.items():
                        nfr_arr_adj[b] = {g: v.copy() for g, v in gmap.items()}

                    nfr_vec_case = nfr_arr_case_all.get(bank_sel, {}).get(gol_sel, None)
                    if (
                        nfr_vec_case is not None
                        and bank_sel in nfr_arr_adj
                        and gol_sel in nfr_arr_adj[bank_sel]
                    ):
                        nfr_arr_adj[bank_sel][gol_sel] = nfr_vec_case

                    # Page4 と同じロジックで Qp(L/s) を再計算
                    df_qp_case = compute_qp_from_nfr_for_df(df_net_base, nfr_arr_adj, eff_map_raw)
                    st.session_state["qp_df_network_case"] = df_qp_case

                    # ★ Section4 のメインテーブル（Adjusted Qp, m³/s）をここで描画
                    render_main_adjusted_table(
                        df_qp_case=df_qp_case,
                        selected_bank=selected_bank,
                        level_name=level_name,
                        selected_canal=selected_canal,
                        display_cols=display_cols,
                        show_golongan=show_golongan,
                        mt_start_labels=mt_start_labels,
                    )

                    # --- 4-A 用グラフ + 年間サマリ + 詳細表の描画（最新の df_qp_case を使用） ---
                    with qp_graphs_container:
                        st.subheader("4-A. 5-day Qp comparison (key Main Canals)")

                        # (1) Left – Cipelang MC. / HW – Excluder Cipelang
                        left_5day = _build_5day_qp_pair(
                            df_base=df,              # ベースは CSV 読み込み済みの df（m³/s）
                            df_case=df_qp_case,      # 今回シナリオで再計算したネットワーク（L/s）
                            bank="Left",
                            canal="Cipelang MC.",
                            parent_name="HW",
                            child_name="Excluder Cipelang",
                        )

                        # (2) Right – Sindupraja MC. / HW – Excluder Sindupraja
                        right_5day = _build_5day_qp_pair(
                            df_base=df,
                            df_case=df_qp_case,
                            bank="Right",
                            canal="Sindupraja MC.",
                            parent_name="HW",
                            child_name="Excluder Sindupraja",
                        )

                        col_left, col_right = st.columns(2)
                        with col_left:
                            render_5day_qp_bar_with_cards(
                                left_5day,
                                title="Left – Cipelang MC. / HW – Excluder Cipelang",
                            )
                        with col_right:
                            render_5day_qp_bar_with_cards(
                                right_5day,
                                title="Right – Sindupraja MC. / HW – Excluder Sindupraja",
                            )

                        # ★ 年間サマリ + Secondary×Golongan 詳細表をここで描画
                        #   mt_start_labels がまだ None の場合は、詳細表の MT 開始ハイライトだけスキップされます。
                        render_annual_summary_and_detail_block(
                            left_5day,
                            right_5day,
                            df_case_disp,
                            mt_start_labels,
                            bank_sel,
                            sec_name,
                            gol_sel,
                        )


                    # === 4番テーブル用: Selected Bank / Level / Canal の Adjusted summary ===
                    summary_adj = build_summary_for_canal(
                        df_qp_case,
                        selected_bank=selected_bank,
                        level_name=level_name,
                        selected_canal=selected_canal,
                    )

                    # 3番と同じ列構成に整形（L/s → m³/s に変換）
                    if not summary_adj.empty:
                        display_adj = summary_adj[display_cols].copy()

                        # ★ Qp 列は L/s → m³/s
                        for lab in labels:
                            if lab in display_adj.columns:
                                display_adj[lab] = display_adj[lab] / 1000.0

                        display_adj = display_adj.rename(columns={
                            "BranchArea": "Area (ha)",
                            "Area_G1": "Golongan 1",
                            "Area_G2": "Golongan 2",
                            "Area_G3": "Golongan 3",
                        })

                        if not show_golongan:
                            for gcol in ["Golongan 1", "Golongan 2", "Golongan 3"]:
                                if gcol in display_adj.columns:
                                    display_adj = display_adj.drop(columns=gcol)

                        # Bank / Canal / ParentName の連続行は空欄に（3番と同じ）
                        for col in ["Bank", "Canal", "ParentName"]:
                            if col in display_adj.columns:
                                display_adj[col] = display_adj[col].where(
                                    display_adj[col].ne(display_adj[col].shift()), ""
                                )

                        display_adj_for_style = display_adj.copy()
                        display_adj_for_style.index = np.arange(1, len(display_adj_for_style) + 1)
                    else:
                        display_adj_for_style = summary_adj  # 空の DataFrame


                    # 対象 Bank の行を抽出
                    df_case_bank = df_qp_case[df_qp_case["Bank"] == bank_sel].copy()

                    # Secondary レベルのみ
                    df_case_sec = df_case_bank[df_case_bank["CanalLevel"] == "Secondary"].copy()

                    # 選択された Secondary Canal
                    sec_name = str(sel_row.get("SecondaryCanal", ""))

                    df_case_canal = df_case_sec[df_case_sec["Canal"] == sec_name].copy()

                    if df_case_canal.empty:
                        st.info("このシナリオに対する Qp の再計算結果に、対象の Secondary Canal の行がありません。")
                    else:
                        # === 3 と同じ Class フィルタを適用（TB / Branch / Target） ===
                        df_case_canal["Class_upper"] = df_case_canal["Class"].astype(str).str.upper()
                        df_case_canal = apply_class_filters(df_case_canal)

                        if df_case_canal.empty:
                            st.info("No rows left after applying TB/Branch/Target filters.")
                        else:
                            # === 3 と同じ並び順 (ChildKey / ParentKey ベース) ===
                            is_branch = np.where(
                                df_case_canal["Class"].astype(str).str.strip() == "Division",
                                1,
                                0,
                            )
                            df_case_canal["SortKey"] = np.where(
                                is_branch == 0,
                                df_case_canal["ChildKey"],
                                df_case_canal["ParentKey"],
                            )
                            df_case_canal["SortIsBranch"] = is_branch
                            df_case_canal = (
                                df_case_canal
                                .sort_values(
                                    ["SortKey", "SortIsBranch", "ChildName", "ParentKey", "ChildKey"]
                                )
                                .reset_index(drop=True)
                            )

                            # === 表示用 DataFrame（ParentKey / ChildKey は不要） ===
                            display_cols_case = [
                                "Bank", "Canal", "ParentName",
                                "ChildName", "Class", "SectionName",
                                "BranchArea", "Area_G1", "Area_G2", "Area_G3",
                            ] + labels

                            display_cols_case = [c for c in display_cols_case if c in df_case_canal.columns]
                            df_case_disp = df_case_canal[display_cols_case].copy()

                            # L/s → m³/s
                            for lab in labels:
                                if lab in df_case_disp.columns:
                                    df_case_disp[lab] = df_case_disp[lab] / 1000.0

                            # 列名を 3 番と合わせる
                            df_case_disp = df_case_disp.rename(columns={
                                "BranchArea": "Area (ha)",
                                "Area_G1": "Golongan 1",
                                "Area_G2": "Golongan 2",
                                "Area_G3": "Golongan 3",
                            })

                            # Golongan 列の表示/非表示
                            if not show_golongan:
                                for gcol in ["Golongan 1", "Golongan 2", "Golongan 3"]:
                                    if gcol in df_case_disp.columns:
                                        df_case_disp = df_case_disp.drop(columns=gcol)

                            # Bank / Canal / ParentName の連続行を空欄に（3 と同じ）
                            for col in ["Bank", "Canal", "ParentName"]:
                                if col in df_case_disp.columns:
                                    df_case_disp[col] = df_case_disp[col].where(
                                        df_case_disp[col].ne(df_case_disp[col].shift()), ""
                                    )

                            # 行番号を 1 始まりに
                            df_case_disp.index = np.arange(1, len(df_case_disp) + 1)


# =====================================================
# Reporting / Print layout (A3)
# =====================================================

def render_report_qp_comparison_section(df_base: pd.DataFrame) -> None:
    """
    1. Summary of Qp Comparison
    4-A の 5-day Qp グラフ + カード + Annual summary テーブルを印刷レイアウト向けに描画。
    """
    st.markdown("### 1. Summary of Qp Comparison")

    df_case = st.session_state.get("qp_df_network_case", None)
    if df_case is None or df_case.empty:
        st.info("Adjusted Qp scenario has not been computed yet in Section 6.")
        return

    # Left – Cipelang
    left_5day = _build_5day_qp_pair(
        df_base=df_base,
        df_case=df_case,
        bank="Left",
        canal="Cipelang MC.",
        parent_name="HW",
        child_name="Excluder Cipelang",
    )

    # Right – Sindupraja
    right_5day = _build_5day_qp_pair(
        df_base=df_base,
        df_case=df_case,
        bank="Right",
        canal="Sindupraja MC.",
        parent_name="HW",
        child_name="Excluder Sindupraja",
    )

    col_left, col_right = st.columns(2)
    with col_left:
        render_5day_qp_bar_with_cards(
            left_5day,
            title="Left – Cipelang MC. / HW – Excluder Cipelang",
        )
    with col_right:
        render_5day_qp_bar_with_cards(
            right_5day,
            title="Right – Sindupraja MC. / HW – Excluder Sindupraja",
        )

    # カードの直下に Annual summary テーブルを 2 つ表示
    render_annual_summary_and_detail_block(
        left_5day,
        right_5day,
        df_case_disp=None,      # 詳細表は出さない
        mt_start_labels=None,
        bank_sel="",            # 使われないダミー
        sec_name="",
        gol_sel=0,
    )


# ---------- 2. Condition settings table (per Canal) ----------

def _build_report_condition_table_for_bank(bank: str) -> pd.DataFrame:
    """
    sec_start_config 全体から、指定 Bank の Main/Secondary をまとめた
    レポート用テーブルを作る。
    """
    sec_cfg_all = st.session_state.get("sec_start_config", None)
    if sec_cfg_all is None or sec_cfg_all.empty:
        return pd.DataFrame()

    cfg_bank = sec_cfg_all[sec_cfg_all["Bank"] == bank].copy()
    if cfg_bank.empty:
        return pd.DataFrame()

    # 調整量を整数にそろえる
    for col in ["Adj_MT1", "Adj_MT2", "Adj_MT3"]:
        if col in cfg_bank.columns:
            cfg_bank[col] = cfg_bank[col].fillna(0).astype(int)

    def apply_adj(base, adj_steps):
        return shift_date_by_5day_steps_str(base, adj_steps)


    cfg_bank["MT1_adj_str"] = [
        apply_adj(b, a) for b, a in zip(cfg_bank["MT1_base"], cfg_bank["Adj_MT1"])
    ]
    cfg_bank["MT2_adj_str"] = [
        apply_adj(b, a) for b, a in zip(cfg_bank["MT2_base"], cfg_bank["Adj_MT2"])
    ]
    cfg_bank["MT3_adj_str"] = [
        apply_adj(b, a) for b, a in zip(cfg_bank["MT3_base"], cfg_bank["Adj_MT3"])
    ]

    # 並び順は Scope / SecCanalKey / SecChildKey / Golongan ベース
    if "SecCanalKey" in cfg_bank.columns and "SecChildKey" in cfg_bank.columns:
        cfg_bank = cfg_bank.sort_values(
            ["Scope", "SecCanalKey", "SecChildKey", "Golongan"],
            kind="mergesort",
        )

    # 連続行の重複値を空欄にしてグルーピングを見やすく
    cfg_bank["MainCanal_disp"] = cfg_bank["MainCanal"]
    cfg_bank.loc[cfg_bank["MainCanal_disp"].duplicated(), "MainCanal_disp"] = ""

    for col in ["Scope", "MainCanal_disp", "ParentName", "SecondaryCanal"]:
        if col in cfg_bank.columns:
            cfg_bank[col] = cfg_bank[col].where(cfg_bank[col].ne(cfg_bank[col].shift()), "")

    preview = cfg_bank[[
        "Scope",
        "MainCanal_disp",
        "ParentName",
        "SecondaryCanal",
        "Golongan",
        "MT1_adj_str", "Adj_MT1",
        "MT2_adj_str", "Adj_MT2",
        "MT3_adj_str", "Adj_MT3",
    ]].copy()

    preview = preview.rename(columns={
        "Scope": "Scope",
        "MainCanal_disp": "Main canal",
        "ParentName": "Parent",
        "SecondaryCanal": "Secondary canal",
        "Golongan": "Gol",
        "MT1_adj_str": "MT-1 Adj",
        "Adj_MT1": "ΔID MT-1",
        "MT2_adj_str": "MT-2 Adj",
        "Adj_MT2": "ΔID MT-2",
        "MT3_adj_str": "MT-3 Adj",
        "Adj_MT3": "ΔID MT-3",
    })

    preview.index = np.arange(1, len(preview) + 1)
    return preview


def _style_report_condition_table(df_cond: pd.DataFrame) -> pd.io.formats.style.Styler:
    """
    Condition table を preview と同じ色付けルールでスタイルする：
      - ΔID > 0: 薄赤
      - ΔID < 0: 薄青
      - 日付列(MT-* Adj) は太字
      - ヘッダーのフォントは少し小さめ
    """
    if df_cond.empty:
        return df_cond.style

    adj_mt1 = df_cond["ΔID MT-1"]
    adj_mt2 = df_cond["ΔID MT-2"]
    adj_mt3 = df_cond["ΔID MT-3"]

    def _adj_color(col: pd.Series):
        styles = [""] * len(col)
        name = col.name
        series = None
        if name in ["ΔID MT-1", "MT-1 Adj"]:
            series = adj_mt1
        elif name in ["ΔID MT-2", "MT-2 Adj"]:
            series = adj_mt2
        elif name in ["ΔID MT-3", "MT-3 Adj"]:
            series = adj_mt3
        else:
            return styles

        for i, v in enumerate(series):
            if v > 0:
                styles[i] = "background-color:#fecaca;"
            elif v < 0:
                styles[i] = "background-color:#bfdbfe;"
        return styles

    fmt_adj = {c: "{:+d}" for c in ["ΔID MT-1", "ΔID MT-2", "ΔID MT-3"] if c in df_cond.columns}

    sty = (
        df_cond.style
        .apply(_adj_color, axis=0)
        .format(fmt_adj)
        .set_properties(**{
            "font-size": "11px",
            "white-space": "nowrap",
            "padding": "2px 4px",
            "text-align": "center",
        })
        # 日付列を太字
        .set_properties(
            subset=[c for c in ["MT-1 Adj", "MT-2 Adj", "MT-3 Adj"] if c in df_cond.columns],
            **{"font-weight": "bold"}
        )
        .set_table_styles([
            {
                "selector": "table",
                "props": [
                    ("border-collapse", "collapse"),
                    ("border", "1px solid #999"),
                ],
            },
            {
                "selector": "th",
                "props": [
                    ("border", "1px solid #999"),
                    ("padding", "2px 4px"),
                    ("background-color", "#e5f0ff"),
                    ("font-weight", "bold"),
                    ("white-space", "nowrap"),
                    ("font-size", "10px"),       # ★ ヘッダーフォント少し小さく
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
    )
    return sty


def render_report_condition_section() -> None:
    """
    2. Condition settings for Qp/NFR scenario (per Canal)
    Left / Right を左右に並べて表示。
    """
    st.markdown("### 2. Condition settings for Qp/NFR scenario (per Canal)")

    sec_cfg_all = st.session_state.get("sec_start_config", None)
    if sec_cfg_all is None or sec_cfg_all.empty:
        st.info("Canal start settings are not available yet (Section 5 has not been used).")
        return

    col_left, col_right = st.columns(2)

    left_tbl = _build_report_condition_table_for_bank("Left")
    right_tbl = _build_report_condition_table_for_bank("Right")

    with col_left:
        st.markdown("**Left bank – main/secondary canals**")
        if left_tbl.empty:
            st.write("(no rows)")
        else:
            sty = _style_report_condition_table(left_tbl)
            st.markdown(
                "<div style='overflow-x:auto; text-align:left;'>"
                f"{sty.to_html()}"
                "</div>",
                unsafe_allow_html=True,
            )

    with col_right:
        st.markdown("**Right bank – main/secondary canals**")
        if right_tbl.empty:
            st.write("(no rows)")
        else:
            sty = _style_report_condition_table(right_tbl)
            st.markdown(
                "<div style='overflow-x:auto; text-align:left;'>"
                f"{sty.to_html()}"
                "</div>",
                unsafe_allow_html=True,
            )


# ---------- 3. Adjusted Schedule (per bank, main canal) ----------

def _build_adjusted_schedule_table_for_bank(
    df_qp_case: pd.DataFrame,
    bank: str,
    main_canal: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    指定 Bank × Main Canal について、
    Main+Branch(Target/Branch) の Adjusted Qp スケジュールテーブルを作る。

    - df_qp_case : compute_qp_from_nfr_for_df で再計算したネットワーク（単位 L/s）
    - bank       : "Left" / "Right"
    - main_canal : 対象 Main Canal 名（例 "Cipelang MC."）

    戻り値:
      df_sched: Parent / Child / Area / Gol1-3 + 72列(Nov-1..Oct-6) の DataFrame（単位 m³/s）
      row_is_branch: 各行が Branch(Division) 行かどうかのフラグ Series
    """
    if df_qp_case is None or df_qp_case.empty:
        return pd.DataFrame(), pd.Series(dtype=bool)

    # Target + Branch だけを表示するため、一時的にフラグを書き換え
    global show_tb, show_branch, show_target
    old_tb, old_branch, old_target = show_tb, show_branch, show_target
    show_tb, show_branch, show_target = False, True, True
    try:
        summary = build_summary_for_canal(
            df_qp_case,
            selected_bank=bank,
            level_name="Main",
            selected_canal=main_canal,
        )
    finally:
        show_tb, show_branch, show_target = old_tb, old_branch, old_target

    if summary.empty:
        return pd.DataFrame(), pd.Series(dtype=bool)

    # ▼▼▼ ソートロジックの適用（ここから追加・修正） ▼▼▼
    
    # 必要な列が存在しない場合は空文字で埋め、文字列型に統一
    for col in ["Class", "ParentKey", "ChildKey", "CanalKey"]:
        if col not in summary.columns:
            summary[col] = ""
        summary[col] = summary[col].astype(str).str.strip()

    summary["Class_upper"] = summary["Class"].str.upper()
    is_branch = np.where(summary["Class_upper"] == "DIVISION", 1, 0)

    # 1. 挿入位置 (SortKey)
    #    Target行は ChildKey、Branch行は ParentKey を使うことで、
    #    「分岐点」の直後に「分岐水路」が並ぶようにする
    summary["SortKey"] = np.where(
        is_branch == 0,
        summary["ChildKey"]
        #summary["ParentKey"]
    )

    # 2. 本流・分岐の順序 (SortIsBranch)
    #    同じ場所なら、本流(0)が先、分岐(1)が後
    summary["SortIsBranch"] = is_branch

    # 3. 分岐水路同士の並び順 (BranchSortKey)
    #    Canal Key と Parent Key の組み合わせでソート
    summary["BranchSortKey"] = summary["CanalKey"] + "_" + summary["ParentKey"]

    # ソート実行
    summary = (
        summary.sort_values(["SortKey", "SortIsBranch", "BranchSortKey"])
        .reset_index(drop=True)
    )

    # ▲▲▲ ソートロジックここまで ▲▲▲

    # Branch 行かどうか（Class = Division）
    # ソート後の順序で再評価
    row_is_branch = summary["Class"].astype(str).str.strip().str.lower().eq("division")

    # 表示列を選択（Class / SectionName は削除）
    cols = [
        "ParentName", "ChildName",
        "BranchArea", "Area_G1", "Area_G2", "Area_G3",
    ] + [lab for lab in ordered_labels if lab in summary.columns]

    df_sched = summary[cols].copy()
    df_sched = df_sched.rename(columns={
        "ParentName": "Parent",
        "ChildName": "Child",
        "BranchArea": "Area (ha)",
        "Area_G1": "Gol 1",
        "Area_G2": "Gol 2",
        "Area_G3": "Gol 3",
    })

    # L/s → m³/s（実値はそのままにして、フォーマットで 1 桁表示）
    for lab in ordered_labels:
        if lab in df_sched.columns:
            df_sched[lab] = df_sched[lab] / 1000.0

    df_sched.index = np.arange(1, len(df_sched) + 1)
    row_is_branch = row_is_branch.reset_index(drop=True)
    return df_sched, row_is_branch



def _style_adjusted_schedule_table(
    df_sched: pd.DataFrame,
    row_is_branch: pd.Series,
) -> pd.io.formats.style.Styler:
    """
    Adjusted Schedule テーブルのスタイル:
      - Qp 列は小数点 1 桁
      - Branch 行のテキスト列を淡い緑 (#d9f99d)
      - データバーは薄赤
    """
    if df_sched.empty:
        return df_sched.style

    cols = set(df_sched.columns)
    text_cols = {
        "Parent", "Child",
        "Area (ha)", "Gol 1", "Gol 2", "Gol 3",
    }
    qp_cols = [lab for lab in ordered_labels if lab in cols]

    fmt_dict: dict[str, object] = {}

    if "Area (ha)" in cols:
        fmt_dict["Area (ha)"] = "{:,.0f}"
    for gcol in ["Gol 1", "Gol 2", "Gol 3"]:
        if gcol in cols:
            fmt_dict[gcol] = (
                lambda v, gcol=gcol: "" if (pd.isna(v) or abs(v) < 1e-9) else f"{v:,.0f}"
            )

    for lab in qp_cols:
        fmt_dict[lab] = (
            lambda v, lab=lab: "" if (pd.isna(v) or abs(v) < 1e-9) else f"{v:,.1f}"
        )

    def _highlight(row: pd.Series):
        pos = int(row.name) - 1
        is_branch = bool(row_is_branch.iloc[pos]) if pos < len(row_is_branch) else False
        styles: list[str] = []
        for col in row.index:
            color = ""
            if col in text_cols and is_branch:
                color = "#d9f99d"
            styles.append(f"background-color:{color}" if color else "")
        return styles

    styler = df_sched.style.format(fmt_dict)

    # データバー（Qp 列のみ薄赤）
    if qp_cols:
        vals = pd.to_numeric(df_sched[qp_cols].stack(), errors="coerce")
        vals = vals.replace([np.inf, -np.inf], np.nan).dropna()
        if not vals.empty:
            vmin = float(vals.min())
            vmax = float(vals.max())
            if vmin != vmax:
                styler = styler.bar(
                    subset=pd.IndexSlice[:, qp_cols],
                    axis=0,
                    color="#fecaca",
                    vmin=vmin,
                    vmax=vmax,
                )

    styler = (
        styler
        .apply(_highlight, axis=1)
        .set_properties(**{
            "text-align": "center",
            "padding": "2px 4px",
            "font-size": "11px",
            "white-space": "nowrap",
        })
        .set_properties(
            subset=["Area (ha)"],
            **{"border-right": "2px solid #666"}
        )
        .set_properties(
            subset=[c for c in ["Parent", "Child", "Area (ha)"] if c in df_sched.columns],
            **{"font-weight": "bold"}
        )
        .set_table_styles([
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
                    ("background-color", "#ffe4e6"),
                    ("font-weight", "bold"),
                    ("font-size", "11px"),
                    ("border", "1px solid #999"),
                    ("padding", "2px 4px"),
                    ("white-space", "nowrap"),
                    ("position", "sticky"),
                    ("top", "0"),
                    ("z-index", "2"),
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
    )
    return styler


def render_report_adjusted_schedule_section() -> None:
    """
    3. Adjusted Schedule

    左右に分けず、1ページ幅を使って、
    Left bank / Right bank の全 Main Canal について
    Target+Branch 行をまとめた 72列テーブルを順番に表示する。
    """
    st.markdown("### 3. Adjusted Schedule")

    df_case = st.session_state.get("qp_df_network_case", None)
    if df_case is None or df_case.empty:
        st.info("Adjusted Qp scenario has not been computed yet in Section 6.")
        return

    # Main level の行から、Bank ごとの Main Canal 一覧を取得
    if "CanalLevel" in df_case.columns:
        main_rows = df_case[df_case["CanalLevel"] == "Main"].copy()
    else:
        main_rows = df_case.copy()

    if "CanalKey" in main_rows.columns:
        sort_cols = ["Bank", "CanalKey", "Canal"]
    else:
        sort_cols = ["Bank", "Canal"]

    main_rows = main_rows.sort_values(sort_cols)

    left_main_list = (
        main_rows[main_rows["Bank"] == "Left"][["Canal"]]
        .drop_duplicates()["Canal"]
        .tolist()
    )
    right_main_list = (
        main_rows[main_rows["Bank"] == "Right"][["Canal"]]
        .drop_duplicates()["Canal"]
        .tolist()
    )

    # -------- Left bank --------
    if left_main_list:
        st.markdown("#### 3.1 Left bank")
        for mc in left_main_list:
            df_sched, flags = _build_adjusted_schedule_table_for_bank(
                df_qp_case=df_case,
                bank="Left",
                main_canal=mc,
            )
            st.markdown(f"**Main Canal: {mc} (Target & Branch)**")
            if df_sched.empty:
                st.write("(no rows)")
            else:
                sty = _style_adjusted_schedule_table(df_sched, flags)
                st.markdown(
                    "<div style='overflow-x:auto; text-align:left;'>"
                    f"{sty.to_html()}"
                    "</div>",
                    unsafe_allow_html=True,
                )
            # 少し間隔を空ける
            st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)
    else:
        st.markdown("#### 3.1 Left bank")
        st.write("(no main canals for Left bank)")

    st.markdown("---")

    # -------- Right bank --------
    if right_main_list:
        st.markdown("#### 3.2 Right bank")
        for mc in right_main_list:
            df_sched, flags = _build_adjusted_schedule_table_for_bank(
                df_qp_case=df_case,
                bank="Right",
                main_canal=mc,
            )
            st.markdown(f"**Main Canal: {mc} (Target & Branch)**")
            if df_sched.empty:
                st.write("(no rows)")
            else:
                sty = _style_adjusted_schedule_table(df_sched, flags)
                st.markdown(
                    "<div style='overflow-x:auto; text-align:left;'>"
                    f"{sty.to_html()}"
                    "</div>",
                    unsafe_allow_html=True,
                )
            st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)
    else:
        st.markdown("#### 3.2 Right bank")
        st.write("(no main canals for Right bank)")


# ---------- Print layout entry point ----------

def render_print_layout_summary(df_base: pd.DataFrame) -> None:
    """
    A4 縦想定：
    1. Summary of Qp Comparison だけを印刷用に描画するレイアウト。
    """
    today_str = dt.date.today().strftime("%Y-%m-%d")

    st.markdown("## Water Requirement Adjustment (Qp vs Adjusted Qp) Scenario – Summary")
    st.markdown(f"**Date:** {today_str}")
    st.markdown("---")

    # 1. Summary of Qp comparison
    render_report_qp_comparison_section(df_base=df_base)


def render_print_layout_conditions() -> None:
    """
    A3 縦想定：
    2. Condition settings for Qp/NFR scenario (per Canal) だけを印刷用に描画するレイアウト。
    """
    today_str = dt.date.today().strftime("%Y-%m-%d")

    st.markdown("## Water Requirement Adjustment (Qp vs Adjusted Qp) Scenario – Conditions")
    st.markdown(f"**Date:** {today_str}")
    st.markdown("---")

    # 2. Condition settings (per Canal)
    render_report_condition_section()


# ---------- Print layout entry point ----------

def render_print_layout(level_name: str) -> None:
    """
    印刷用レイアウト全体を描画する。

    現状の構成:
      1. Summary of Qp Comparison
      2. Condition settings for Qp/NFR scenario (per Canal)
      3. Adjusted Schedule
    """
    today_str = dt.date.today().strftime("%Y-%m-%d")

    # タイトル & 日付
    st.markdown("## Water Requirement Adjustment (Qp vs Adjusted Qp) Scenario")
    st.markdown(f"**Date:** {today_str}")
    st.markdown("---")

    # 1. Summary of Qp comparison
    #   （5-day Qp グラフ + カード + Annual summary）
    render_report_qp_comparison_section(df_base=df)
    st.markdown("---")

    # 2. Condition settings (per Canal)
    render_report_condition_section()
    st.markdown("---")

    # 3. Adjusted Schedule
    render_report_adjusted_schedule_section()
