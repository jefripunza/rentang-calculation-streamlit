from pathlib import Path
import datetime as dt
import calendar
import re

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ========================================
# 共通設定・5day 系
# ========================================

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_DIR = BASE_DIR / "csv"

# 5Day_ID always starts from November (water year)
water_month_order = [11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
month_order_abbr = [calendar.month_abbr[m] for m in water_month_order]


def build_5day_labels() -> list[str]:
    labels = []
    for i in range(72):
        m = water_month_order[i // 6]
        step = (i % 6) + 1
        labels.append(f"{calendar.month_abbr[m]}-{step}")
    return labels


def date_to_5day_id(d: dt.date) -> int:
    m = d.month
    try:
        m_idx = water_month_order.index(m)
    except ValueError:
        m_idx = 0

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

def date_to_5day_index(d: dt.date) -> int:
    """
    カレンダー日付 → 0-based 5Day index (0..71) に変換する。
    Page5 本体のロジックと同じ。
    """
    m = d.month
    try:
        m_idx = water_month_order.index(m)  # 0..11 (Nov..Oct)
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

    # 0-based index (0..71)
    return m_idx * 6 + (step - 1)


def build_mt_start_map_from_page2() -> dict[tuple[str, int, str], int]:
    """
    Page2 で設定した Bank×Golongan×Season 別の MT 開始日から、
    0-based 5Day index (0..71) を求める。
    Page5 と同じロジック。
    """
    mt_start_map: dict[tuple[str, int, str], int] = {}

    lp = st.session_state.get("landprep_df", None)

    # landprep_df がある場合はそちらを優先
    if (
        isinstance(lp, pd.DataFrame)
        and not lp.empty
        and {"Group", "MT1_start", "MT2_start", "MT3_start"}.issubset(lp.columns)
    ):
        for _, r in lp.iterrows():
            grp = str(r["Group"])
            parts = grp.split("_", 1)
            if len(parts) != 2:
                continue
            bank = parts[0].strip()
            rest = parts[1]

            m = re.search(r"(\d+)", rest)
            if not m:
                continue
            gol_idx = int(m.group(1))

            for season, col in [
                ("MT-1", "MT1_start"),
                ("MT-2", "MT2_start"),
                ("MT-3", "MT3_start"),
            ]:
                # MT-3 は MT3_active が False の行は無視
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

    # landprep_df が無いときは共通 MT1/2/3 を使うフォールバック
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


def _five_day_id_to_date_from_base(base_date: dt.date, id_5d: int) -> dt.date | None:
    """
    5Day_ID (1..72) を、base_date と同じ water-year 上の日付に変換する。
    11,12月 → water-year 開始年、それ以外 → 開始年+1年。
    """
    if id_5d is None or id_5d < 1 or id_5d > 72:
        return None

    idx0 = id_5d - 1
    month = water_month_order[idx0 // 6]      # 11,12,1..10
    step  = (idx0 % 6) + 1                    # 1..6

    # base_date から water-year の開始年を決定
    if base_date.month >= 11:
        wy_start = base_date.year
    else:
        wy_start = base_date.year - 1

    year = wy_start if month in (11, 12) else wy_start + 1

    # ステップの先頭日（1,6,11,...）
    day = (step - 1) * 5 + 1
    last_day = calendar.monthrange(year, month)[1]
    if day > last_day:
        day = last_day

    return dt.date(year, month, day)


def _get_mt_duration_steps() -> tuple[int, int, int]:
    """
    Page2 の設定から MT-1/2/3 の期間（日数）を取得し、5Day ステップ数に変換する。
    DurPmt1, DurPmt2 は basic_edited から、
    MT-3 は landprep_df.MT3_days の中央値（なければ 70 日）を使う。
    """
    basic_edited = st.session_state.get("basic_edited", None)
    # デフォルト（日数）
    dur1_days, dur2_days, dur3_days = 95.0, 90.0, 70.0

    if isinstance(basic_edited, pd.DataFrame) and not basic_edited.empty:
        vals = basic_edited.set_index("Variable")["Value"]
        dur1_days = float(vals.get("DurPmt1", dur1_days))
        dur2_days = float(vals.get("DurPmt2", dur2_days))

    lp = st.session_state.get("landprep_df", None)
    if isinstance(lp, pd.DataFrame) and "MT3_days" in lp.columns:
        s = pd.to_numeric(lp["MT3_days"], errors="coerce").dropna()
        if not s.empty:
            dur3_days = float(s.median())

    dur1_steps = max(1, int(round(dur1_days / 5.0)))
    dur2_steps = max(1, int(round(dur2_days / 5.0)))
    dur3_steps = max(1, int(round(dur3_days / 5.0)))
    return dur1_steps, dur2_steps, dur3_steps



labels = build_5day_labels()
ordered_labels = labels[:]

# ========================================
# Qp CSV 読み込み（Page5 と同じ）
# ========================================

st.set_page_config(page_title="Page5 – Qp Report", layout="wide")

# --- 印刷時のレイアウト調整 ---
st.markdown(
    """
    <style>
    /* 印刷時だけ適用されるスタイル */
    @media print {
      /* 左側のページ一覧（サイドバー）を非表示 */
      section[data-testid="stSidebar"] {
        display: none;
      }

      /* 余白を少し詰める（お好みで調整） */
      div.block-container {
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
      }

      /* .no-print クラスの要素は印刷時は非表示 */
      .no-print {
        display: none !important;
      }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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

df_base = df_qp_raw.copy()

# 文字列系は trim しておく（Bank / Canal / ParentName / ChildName）
for col in ["Bank", "Canal", "ParentName", "ChildName"]:
    if col in df_base.columns:
        df_base[col] = df_base[col].astype(str).str.strip()


# ========================================
# 1. Summary of Qp Comparison
# ========================================

def _get_reach_row(
    df_source: pd.DataFrame, bank: str, canal: str, parent_name: str, child_name: str
) -> pd.Series | None:
    mask = (
        df_source["Bank"].astype(str).str.strip().eq(bank)
        & df_source["Canal"].astype(str).str.strip().eq(canal)
        & df_source["ParentName"].astype(str).str.strip().eq(parent_name)
        & df_source["ChildName"].astype(str).str.strip().eq(child_name)
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
    row_base = _get_reach_row(df_base, bank, canal, parent_name, child_name)
    if row_base is None:
        return None

    row_case = _get_reach_row(df_case, bank, canal, parent_name, child_name) \
        if df_case is not None else None

    records = []
    for lab in labels:
        if lab not in row_base.index:
            continue

        try:
            qp_plan = float(row_base.get(lab, 0.0))
        except Exception:
            qp_plan = 0.0

        qp_adj = np.nan
        if row_case is not None and lab in row_case.index:
            v = row_case.get(lab)
            try:
                qp_adj = float(v) / 1000.0  # L/s → m3/s
            except Exception:
                qp_adj = np.nan

        records.append(
            {"StepLabel": lab, "Qp_planned": qp_plan, "Qp_adjusted": qp_adj}
        )

    return pd.DataFrame(records)


def render_5day_qp_bar_with_cards(df_5day: pd.DataFrame | None, title: str) -> None:
    if df_5day is None or df_5day.empty:
        st.write(f"(no data for {title})")
        return

    def _volume_m3_from_df(df_local: pd.DataFrame, col_name: str) -> float:
        total = 0.0
        for _, row in df_local.iterrows():
            step_label = row["StepLabel"]
            q = row[col_name]
            if pd.isna(q):
                continue
            try:
                idx = labels.index(step_label)
            except ValueError:
                continue

            month = water_month_order[idx // 6]
            step = (idx % 6) + 1
            days_in_month = calendar.monthrange(2001, month)[1]
            days = 5 if step <= 5 else (days_in_month - 25)
            total += float(q) * days * 86400.0
        return total

    max_plan = df_5day["Qp_planned"].max()
    lab_plan = df_5day.loc[df_5day["Qp_planned"].idxmax(), "StepLabel"]
    vol_plan_mill = _volume_m3_from_df(df_5day, "Qp_planned") / 1e6

    if df_5day["Qp_adjusted"].notna().any():
        max_adj = df_5day["Qp_adjusted"].max()
        lab_adj = df_5day.loc[df_5day["Qp_adjusted"].idxmax(), "StepLabel"]
        df_adj_for_sum = df_5day.copy()
        df_adj_for_sum["Qp_adjusted"] = df_adj_for_sum["Qp_adjusted"].fillna(0.0)
        vol_adj_mill = _volume_m3_from_df(df_adj_for_sum, "Qp_adjusted") / 1e6
    else:
        max_adj = np.nan
        lab_adj = "-"
        vol_adj_mill = np.nan

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

    tick_values = [lab for i, lab in enumerate(labels) if i % 6 == 0]

    chart = (
        alt.Chart(df_long)
        .mark_bar()
        .encode(
            x=alt.X(
                "StepLabel:N",
                sort=labels,
                title="5-day step",
                axis=alt.Axis(
                    values=tick_values, labelFontSize=12, titleFontSize=14
                ),
            ),
            xOffset="ScenarioLabel:N",
            y=alt.Y(
                "Qp:Q",
                title="Q (m³/s)",
                axis=alt.Axis(labelFontSize=12, titleFontSize=14),
            ),
            color=alt.Color(
                "ScenarioLabel:N",
                scale=alt.Scale(
                    domain=["Planned Qp", "Adjusted Qp"],
                    range=["#60a5fa", "#f97373"],
                ),
                legend=alt.Legend(title=None, orient="top", labelFontSize=12),
            ),
            tooltip=[
                "StepLabel:N",
                "ScenarioLabel:N",
                alt.Tooltip("Qp:Q", format=".3f"),
            ],
        )
        .properties(title=title, width=650, height=260)
    )
    st.altair_chart(chart, width="stretch")

    card_left, card_right = st.columns(2)

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
        </div>
      </div>
    </div>
    """

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


# 年間サマリ表は、元の Page5 と同じ render_annual_summary_and_detail_block を
# ここにコピペしても良いですが、長くなるので割愛します。
# （今の Page5 で正しく動いているバージョンをそのままこのファイルに移植してください）

from copy import deepcopy

# ========= ここから：Condition table / Adjusted Schedule は
#           今のファイルの _build_report_condition_table_for_bank,
#           _style_report_condition_table,
#           render_report_condition_section,
#           _build_adjusted_schedule_table_for_bank,
#           _style_adjusted_schedule_table,
#           render_report_adjusted_schedule_section
#           を丸ごとコピペして下さい。
#           （長いので、このメッセージでは途中までにしますが、
#            ロジックは今の Page5 のものをそのまま使えます）
# =========

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
    5-day Qp グラフの下に出す Annual summary 表だけを描画する簡易版。
    Secondary×Golongan の詳細表は、このレポートページでは表示しない。
    """

    # 両方とも空なら何もしない
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

            if np.isnan(qmax[month]) or q > qmax[month]:
                qmax[month] = float(q)

            vol[month] += float(q) * days * 86400.0

        for m in months_num:
            vol[m] /= 1e6  # m³ → million m³

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

    qmax_T_P = {}
    qmax_T_A = {}
    vol_T_P  = {}
    vol_T_A  = {}
    for m in months_num:
        qmax_T_P[m] = _nan_to_zero(qmax_L_P[m]) + _nan_to_zero(qmax_R_P[m])
        qmax_T_A[m] = _nan_to_zero(qmax_L_A[m]) + _nan_to_zero(qmax_R_A[m])
        vol_T_P[m]  = vol_L_P[m] + vol_R_P[m]
        vol_T_A[m]  = vol_L_A[m] + vol_R_A[m]

    rows_max: list[dict] = []
    rows_sum: list[dict] = []

    def _append_rows(case_label, qmax_L, qmax_R, qmax_T, vol_L, vol_R, vol_T,
                     max5_L, max5_R, max5_T):
        for bank_label, qdict, vdict, max_val_5d in [
            ("Left",  qmax_L, vol_L, max5_L),
            ("Right", qmax_R, vol_R, max5_R),
            ("Total", qmax_T, vol_T, max5_T),
        ]:
            row_max = {"Type": "Qp_max (m³/s)", "Case": case_label, "Bank": bank_label}
            for m, abbr in zip(months_num, months_abbr):
                row_max[abbr] = qdict[m]
            row_max["Max"] = max_val_5d
            rows_max.append(row_max)

            row_sum = {"Type": "ΣQp (Mill. m³)", "Case": case_label, "Bank": bank_label}
            vols = []
            for m, abbr in zip(months_num, months_abbr):
                vv = vdict[m]
                row_sum[abbr] = vv
                vols.append(vv)
            total_vol = sum(vols)
            row_sum["Sum"] = total_vol if abs(total_vol) > 1e-9 else np.nan
            rows_sum.append(row_sum)

    _append_rows("Planned", qmax_L_P, qmax_R_P, qmax_T_P,
                 vol_L_P, vol_R_P, vol_T_P,
                 max5_L_P, max5_R_P, max5_T_P)
    _append_rows("Adjusted", qmax_L_A, qmax_R_A, qmax_T_A,
                 vol_L_A, vol_R_A, vol_T_A,
                 max5_L_A, max5_R_A, max5_T_A)

    if not rows_max:
        return

    cols_max = ["Type", "Case", "Bank"] + months_abbr + ["Max"]
    cols_sum = ["Type", "Case", "Bank"] + months_abbr + ["Sum"]
    df_max = pd.DataFrame(rows_max)[cols_max]
    df_sum = pd.DataFrame(rows_sum)[cols_sum]

    df_max.insert(0, "No.", np.arange(1, len(df_max) + 1))
    df_sum.insert(0, "No.", np.arange(1, len(df_sum) + 1))

    df_max_disp = df_max.copy()
    df_sum_disp = df_sum.copy()
    for col in ["Type", "Bank"]:
        df_max_disp[col] = df_max_disp[col].where(df_max_disp[col].ne(df_max_disp[col].shift()), "")
        df_sum_disp[col] = df_sum_disp[col].where(df_sum_disp[col].ne(df_sum_disp[col].shift()), "")

    def _make_highlighter(df_for_style: pd.DataFrame):
        def _highlight(row: pd.Series):
            styles = [""] * len(row)
            case_label = row["Case"]
            vals = [row[m] for m in months_abbr]
            arr = np.array(vals, dtype=float)
            row_max = np.nanmax(arr) if np.any(~np.isnan(arr)) else None

            for j, col in enumerate(df_for_style.columns):
                if col in months_abbr and row_max is not None:
                    v = row[col]
                    if not pd.isna(v) and abs(v - row_max) < 1e-9:
                        if case_label == "Planned":
                            styles[j] = "color:#1d4ed8;font-weight:bold;"
                        else:
                            styles[j] = "color:#b91c1c;font-weight:bold;"
                if col in ["Max", "Sum"]:
                    base = styles[j]
                    extra = "background-color:#fff59d;font-weight:bold;"
                    styles[j] = (base + extra) if base else extra
            return styles
        return _highlight

    fmt_max = {m: "{:.1f}" for m in months_abbr}
    fmt_max["Max"] = "{:.1f}"
    sty_max = df_max_disp.style.format(fmt_max, na_rep="")

    vals_all_max = df_max_disp[months_abbr].to_numpy().astype(float)
    vmax_max = np.nanmax(vals_all_max) if np.any(~np.isnan(vals_all_max)) else 0.0
    if vmax_max > 0:
        vmax_max_for_bar = vmax_max / 0.9
        planned_rows = df_max_disp.index[df_max_disp["Case"] == "Planned"]
        adjusted_rows = df_max_disp.index[df_max_disp["Case"] == "Adjusted"]
        sty_max = sty_max.bar(
            subset=pd.IndexSlice[planned_rows, months_abbr],
            color="#bfdbfe", vmin=0, vmax=vmax_max_for_bar,
        )
        sty_max = sty_max.bar(
            subset=pd.IndexSlice[adjusted_rows, months_abbr],
            color="#fecaca", vmin=0, vmax=vmax_max_for_bar,
        )

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
        .set_properties(
            subset=pd.IndexSlice[adj_rows_max, :],
            **{"font-size": "13.5px", "font-weight": "bold"}
        )
        .set_table_styles([
            {"selector": "table", "props": [("border-collapse", "collapse"), ("border", "1px solid #999")]},
            {"selector": "th", "props": [("border", "1px solid #999"), ("padding", "2px 4px")]},
            {"selector": "td", "props": [("border", "1px solid #999"), ("padding", "2px 4px")]},
        ])
    )

    total_rows_max = df_max_disp.index[df_max_disp["Bank"] == "Total"]
    if len(total_rows_max) > 0:
        sty_max = sty_max.set_properties(
            subset=pd.IndexSlice[total_rows_max, ["Bank"]],
            **{"background-color": "#fef3c7"}
        )

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
            color="#bfdbfe", vmin=0, vmax=vmax_sum_for_bar,
        )
        sty_sum = sty_sum.bar(
            subset=pd.IndexSlice[adjusted_rows_s, months_abbr],
            color="#fecaca", vmin=0, vmax=vmax_sum_for_bar,
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
            **{"font-size": "13.5px", "font-weight": "bold"}
        )
        .set_table_styles([
            {"selector": "table", "props": [("border-collapse", "collapse"), ("border", "1px solid #999")]},
            {"selector": "th", "props": [("border", "1px solid #999"), ("padding", "2px 4px")]},
            {"selector": "td", "props": [("border", "1px solid #999"), ("padding", "2px 4px")]},
        ])
    )

    total_rows_sum = df_sum_disp.index[df_sum_disp["Bank"] == "Total"]
    if len(total_rows_sum) > 0:
        sty_sum = sty_sum.set_properties(
            subset=pd.IndexSlice[total_rows_sum, ["Bank"]],
            **{"background-color": "#fef3c7"}
        )

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


# ---------- 2. Condition settings table (per Canal) ----------

def _build_report_condition_table_for_bank(bank: str) -> pd.DataFrame:
    """
    sec_start_config 全体から、指定 Bank の Main/Secondary をまとめた
    レポート用テーブルを作る。

    MT-1/2/3 については:
      - Adj は 5Day_ID ベースでシフト
      - それぞれの終了日 (End) 列も追加
    """
    sec_cfg_all = st.session_state.get("sec_start_config", None)
    if sec_cfg_all is None or sec_cfg_all.empty:
        return pd.DataFrame()

    cfg_bank = sec_cfg_all[sec_cfg_all["Bank"] == bank].copy()
    if cfg_bank.empty:
        return pd.DataFrame()

    # 調整量は常に整数
    for col in ["Adj_MT1", "Adj_MT2", "Adj_MT3"]:
        if col in cfg_bank.columns:
            cfg_bank[col] = pd.to_numeric(cfg_bank[col], errors="coerce").fillna(0).astype(int)
        else:
            cfg_bank[col] = 0

    mt1_steps, mt2_steps, mt3_steps = _get_mt_duration_steps()

    def _adj_and_end(base_val, adj_steps, n_steps):
        """
        base_val(日付) と adj_steps(5Day_ID 差分) から、調整後開始日と終了日を返す。
        - 5Day_ID は 1..72 で計算・クリップ
        - 日付の表示は yy-mm-dd
        """
        if base_val is None or pd.isna(base_val):
            return "", ""
        try:
            base_date = pd.to_datetime(base_val).date()
        except Exception:
            return "", ""
        base_id = date_to_5day_id(base_date)  # 1..72
        if base_id < 1 or base_id > 72:
            return "", ""

        new_id = base_id + int(adj_steps)
        new_id = max(1, min(72, new_id))
        end_id = new_id + n_steps - 1
        end_id = max(1, min(72, end_id))

        adj_date = _five_day_id_to_date_from_base(base_date, new_id)
        end_date = _five_day_id_to_date_from_base(base_date, end_id)
        if adj_date is None or end_date is None:
            return "", ""

        return adj_date.strftime("%y-%m-%d"), end_date.strftime("%y-%m-%d")

    # 各 MT について Adj / End を計算
    mt1_adj_str, mt1_end_str = [], []
    mt2_adj_str, mt2_end_str = [], []
    mt3_adj_str, mt3_end_str = [], []

    for _, r in cfg_bank.iterrows():
        s1, e1 = _adj_and_end(r.get("MT1_base"), r.get("Adj_MT1", 0), mt1_steps)
        s2, e2 = _adj_and_end(r.get("MT2_base"), r.get("Adj_MT2", 0), mt2_steps)
        s3, e3 = _adj_and_end(r.get("MT3_base"), r.get("Adj_MT3", 0), mt3_steps)
        mt1_adj_str.append(s1); mt1_end_str.append(e1)
        mt2_adj_str.append(s2); mt2_end_str.append(e2)
        mt3_adj_str.append(s3); mt3_end_str.append(e3)

    cfg_bank["MT1_adj_str"] = mt1_adj_str
    cfg_bank["MT1_end_str"] = mt1_end_str
    cfg_bank["MT2_adj_str"] = mt2_adj_str
    cfg_bank["MT2_end_str"] = mt2_end_str
    cfg_bank["MT3_adj_str"] = mt3_adj_str
    cfg_bank["MT3_end_str"] = mt3_end_str

    # 並び順は Scope / SecCanalKey / SecChildKey / Golongan ベース
    if "SecCanalKey" in cfg_bank.columns and "SecChildKey" in cfg_bank.columns:
        cfg_bank = cfg_bank.sort_values(
            ["Scope", "SecCanalKey", "SecChildKey", "Golongan"],
            kind="mergesort",
        )

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
        "MT1_adj_str", "MT1_end_str", "Adj_MT1",
        "MT2_adj_str", "MT2_end_str", "Adj_MT2",
        "MT3_adj_str", "MT3_end_str", "Adj_MT3",
    ]].copy()

    preview = preview.rename(columns={
        "Scope": "Scope",
        "MainCanal_disp": "Main canal",
        "ParentName": "Parent",
        "SecondaryCanal": "Secondary canal",
        "Golongan": "Gol",
        "MT1_adj_str": "MT-1 Adj",
        "MT1_end_str": "MT-1 End",
        "Adj_MT1": "ΔID MT-1",
        "MT2_adj_str": "MT-2 Adj",
        "MT2_end_str": "MT-2 End",
        "Adj_MT2": "ΔID MT-2",
        "MT3_adj_str": "MT-3 Adj",
        "MT3_end_str": "MT-3 End",
        "Adj_MT3": "ΔID MT-3",
    })

    # Left/Right を同じ表に入れやすくするため、Bank 列を先頭に追加
    preview.insert(0, "Bank", bank)

    preview.index = np.arange(1, len(preview) + 1)
    return preview


def _style_report_condition_table(df_cond: pd.DataFrame) -> pd.io.formats.style.Styler:
    """
    Condition table を preview と同じ色付けルールでスタイルする：
      - ΔID > 0: 薄赤
      - ΔID < 0: 薄青
      - 日付列(MT-* Adj) は太字
      - ヘッダー・値とも中央揃え、フォント少し大きめ
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
            "font-size": "12px",       # ★ 少し大きく
            "white-space": "nowrap",
            "padding": "3px 5px",
            "text-align": "center",    # ★ 中央揃え
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
                    ("padding", "3px 5px"),
                    ("background-color", "#e5f0ff"),
                    ("font-weight", "bold"),
                    ("white-space", "nowrap"),
                    ("font-size", "11px"),   # ★ ヘッダーもやや大きめ
                    ("text-align", "center"),
                ],
            },
            {
                "selector": "td",
                "props": [
                    ("border", "1px solid #999"),
                    ("padding", "3px 5px"),
                    ("white-space", "nowrap"),
                    ("text-align", "center"),
                ],
            },
        ])
    )
    return sty


def render_report_condition_section() -> None:
    """
    2. Condition settings for Qp/NFR scenario (per Canal)
    Left / Right を 1 つのテーブルにまとめて表示。
    """
    st.markdown("### 2. Condition settings for Qp/NFR scenario (per Canal)")

    left_tbl  = _build_report_condition_table_for_bank("Left")
    right_tbl = _build_report_condition_table_for_bank("Right")

    if left_tbl.empty and right_tbl.empty:
        st.info("Canal start settings are not available yet (Section 5 has not been used).")
        return

    df_all = pd.concat([left_tbl, right_tbl], ignore_index=True)

    sty = _style_report_condition_table(df_all)
    st.markdown(
        "<div style='overflow-x:auto; text-align:left;'>"
        f"{sty.to_html()}"
        "</div>",
        unsafe_allow_html=True,
    )

def _build_adjusted_mt_index_map() -> dict[tuple[str, int, str], int]:
    """
    sec_start_config と Page2 の base start から、
    (Bank, Golongan, Season) → Adjusted 5Day index(0..71) のマップを作る。
    Season: "MT-1", "MT-2", "MT-3"
    """
    sec_cfg_all = st.session_state.get("sec_start_config", None)
    if not isinstance(sec_cfg_all, pd.DataFrame) or sec_cfg_all.empty:
        return {}

    mt_start_map = build_mt_start_map_from_page2()
    adj_map: dict[tuple[str, int, str], int] = {}

    for _, r in sec_cfg_all.iterrows():
        bank = str(r.get("Bank", "")).strip()
        try:
            gol = int(r.get("Golongan", 0))
        except Exception:
            continue

        for season, adj_col in [
            ("MT-1", "Adj_MT1"),
            ("MT-2", "Adj_MT2"),
            ("MT-3", "Adj_MT3"),
        ]:
            base_idx = mt_start_map.get((bank, gol, season), None)
            if base_idx is None:
                continue
            try:
                adj = int(r.get(adj_col, 0))
            except Exception:
                adj = 0

            new_idx = base_idx + adj
            if 0 <= new_idx < len(ordered_labels):
                adj_map[(bank, gol, season)] = new_idx

    return adj_map


def _get_report_mt_start_labels(bank: str) -> set[str]:
    """
    sec_start_config と Page2 の base start を見て、
    指定 Bank の調整後 MT-1/2/3 開始 5Day 列ラベル集合を返す。
    """
    sec_cfg_all = st.session_state.get("sec_start_config", None)
    if not isinstance(sec_cfg_all, pd.DataFrame) or sec_cfg_all.empty:
        return set()

    mt_start_map = build_mt_start_map_from_page2()  # (bank, gol, season) -> idx(0..71)
    label_set: set[str] = set()

    # 指定 Bank だけを見る
    for _, r in sec_cfg_all[sec_cfg_all["Bank"] == bank].iterrows():
        try:
            gol = int(r.get("Golongan", 0))
        except Exception:
            continue

        for season, adj_col in [("MT-1", "Adj_MT1"),
                                ("MT-2", "Adj_MT2"),
                                ("MT-3", "Adj_MT3")]:
            base_idx = mt_start_map.get((bank, gol, season), None)
            if base_idx is None:
                continue

            try:
                adj = int(r.get(adj_col, 0))
            except Exception:
                adj = 0

            new_idx = base_idx + adj
            if 0 <= new_idx < len(ordered_labels):
                label_set.add(ordered_labels[new_idx])

    return label_set


def _build_canal_edges_for_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Page5 の build_canal_edges と同じロジック。
    Class='Division' 行から ParentCanal -> ChildCanal 関係を抽出し、
    Junction の Key/Name と Division 行の index をまとめたテーブルを返す。
    """
    df_loc = df.copy()

    # 文字列列を整形
    for col in [
        "ParentKey", "ChildKey", "ParentName", "ChildName",
        "Canal", "Class", "CanalBank", "Bank"
    ]:
        if col in df_loc.columns:
            df_loc[col] = df_loc[col].astype(str).str.strip()
        else:
            df_loc[col] = ""

    df_loc["Class_upper"] = df_loc["Class"].str.upper()
    edges: list[dict[str, object]] = []

    # Division 行だけ抜き出し
    div_df = df_loc[df_loc["Class_upper"] == "DIVISION"].copy()

    for idx, r in div_df.iterrows():
        child_canal = r["Canal"]
        if not child_canal or child_canal.lower() == "nan":
            continue

        bank = r.get("CanalBank", "") or r.get("Bank", "")
        bank = str(bank).strip()

        # Junction 候補（ParentKey / ChildKey のどちらか）
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

            # 自分自身と同じ Canal は除外し、他の Canal を親候補にする
            parents_here = rows_here[
                (rows_here["Canal"] != child_canal) &
                (rows_here["Canal"] != "") &
                (rows_here["Canal"].str.lower() != "nan")
            ]
            if parents_here.empty:
                continue

            # できれば Division 以外を優先（Main / Secondary など）
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

# ---------- 3. Adjusted Schedule (per bank, main canal) ----------

def _build_adjusted_schedule_table_for_bank(
    df_qp_case: pd.DataFrame,
    bank: str,
    main_canal: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    指定 Bank × Main Canal について、
    Main 内の Target + Branch(Division) 行をまとめた 72 列テーブルを作る。

    - df_qp_case : Page5 で作成済みのネットワークケース（単位 L/s）
    - 戻り値:
        df_sched    : Parent / Child / Area / Gol1-3 + 72 列 (m³/s)
        row_is_branch: 各行が Branch(Division) 行かどうか
    """
    if df_qp_case is None or df_qp_case.empty:
        return pd.DataFrame(), pd.Series(dtype=bool)

    df = df_qp_case.copy()
    
    # CanalKey がない場合は空文字で列作成（エラー防止）
    if "CanalKey" not in df.columns:
        df["CanalKey"] = ""

    for col in ["Bank", "Canal", "CanalKey", "Class", "SectionName"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
        else:
            df[col] = ""

    # ----- Main level だけ抽出 -----
    if "CanalLevel" in df.columns:
        df = df[df["CanalLevel"].astype(str).str.strip() == "Main"]
    df = df[(df["Bank"] == bank) & (df["Canal"] == main_canal)].copy()
    if df.empty:
        return pd.DataFrame(), pd.Series(dtype=bool)

    # groupby 用の列
    group_cols = [
        "Bank", "Canal", "CanalKey",
        "ParentKey", "ChildKey",
        "ParentName", "ChildName",
        "Class", "SectionName",
    ]
    agg_cols_base = ["BranchArea", "Area_G1", "Area_G2", "Area_G3"]
    agg_cols_q = ordered_labels

    for c in agg_cols_base:
        if c not in df.columns:
            df[c] = np.nan
    for lab in agg_cols_q:
        if lab not in df.columns:
            df[lab] = 0.0

    internal_summary = (
        df.groupby(group_cols, as_index=False)[agg_cols_base + agg_cols_q].sum()
    )

    # ----- Branch 行: canal_edges から Division 行を追加 -----
    canal_edges = _build_canal_edges_for_report(df_qp_case)
    branches_for_this = pd.DataFrame()
    if canal_edges is not None and not canal_edges.empty:
        branches_for_this = canal_edges[
            (canal_edges["ParentCanal"] == main_canal)
            & (canal_edges["Bank"] == bank)
            & (canal_edges["ParentCanal"] != "")
        ].copy()

    if not branches_for_this.empty:
        branch_records = []
        for _, b in branches_for_this.iterrows():
            idx_div = int(b["DivisionRowIndex"])

            c_key = ""
            if "CanalKey" in df_qp_case.columns:
                val = df_qp_case.at[idx_div, "CanalKey"]
                c_key = str(val).strip() if pd.notna(val) else ""

            # ★ Page5 と同じ：ChildName は ChildCanal 名を使う
            record = {
                "Bank":       bank,
                "Canal":      main_canal,
                "CanalKey":   c_key,
                "ParentKey":  b["JunctionKey"],
                "ChildKey":   df_qp_case.at[idx_div, "ChildKey"],
                "ParentName": b["JunctionName"],
                "ChildName":  b["ChildCanal"],
                "Class":      "Division",
                "SectionName": df_qp_case.at[idx_div, "SectionName"],
                "BranchArea": df_qp_case.at[idx_div, "BranchArea"],
                "Area_G1":    df_qp_case.at[idx_div, "Area_G1"],
                "Area_G2":    df_qp_case.at[idx_div, "Area_G2"],
                "Area_G3":    df_qp_case.at[idx_div, "Area_G3"],
            }
            for lab in ordered_labels:
                record[lab] = df_qp_case.at[idx_div, lab]
            branch_records.append(record)

        branch_summary = pd.DataFrame(branch_records)
    else:
        branch_summary = internal_summary.iloc[0:0].copy()

    summary = pd.concat([internal_summary, branch_summary], ignore_index=True)

    # ----- 並び順ロジックの適用 -----
    summary["Class_upper"] = summary["Class"].astype(str).str.upper()
    is_branch = np.where(summary["Class_upper"] == "DIVISION", 1, 0)

    # 1. 挿入位置の決定 (SortKey)
    #    Target行は ChildKey、Branch行は ParentKey を使うことで、
    #    「分岐点」の直後に「分岐水路」が並ぶようにする。
    summary["SortKey"] = np.where(
        is_branch == 0,
        summary["ChildKey"].astype(str),   # Target: ChildKey
        summary["ParentKey"].astype(str),  # Branch: ParentKey
    )
    
    # 2. 本流・分岐の順序 (SortIsBranch)
    #    同じ場所(SortKey)なら、本流(0)が先、分岐(1)が後。
    summary["SortIsBranch"] = is_branch

    # 3. ご指定のソートキー (BranchSortKey)
    #    CanalKey と ParentKey を連結。
    #    ※ 分岐が複数ある場合、このキーの昇順で並ぶことになります。
    summary["BranchSortKey"] = (
        summary["CanalKey"].astype(str) + "_" + summary["ParentKey"].astype(str)
    )

    # ▼▼▼ ソート実行 ▼▼▼
    summary = (
        summary.sort_values(
            ["SortKey", "SortIsBranch", "BranchSortKey"]
        )
        .reset_index(drop=True)
    )
    # ▲▲▲ 修正箇所ここまで ▲▲▲

    # ----- TB 行は除外（Target + Branch のみ） -----
    cls = summary["Class_upper"]
    mask_tb = cls.eq("TB")
    summary = summary[~mask_tb].copy()

    row_is_branch = summary["Class_upper"].eq("DIVISION").reset_index(drop=True)

    # ----- 表示用テーブルへ整形（L/s → m³/s） -----
    cols = [
        "ParentName", "ChildName",
        "BranchArea", "Area_G1", "Area_G2", "Area_G3",
    ] + [lab for lab in ordered_labels if lab in summary.columns]

    df_sched = summary[cols].copy()
    df_sched = df_sched.rename(columns={
        "ParentName": "Parent",
        "ChildName":  "Child",
        "BranchArea": "Area (ha)",
        "Area_G1":    "Gol 1",
        "Area_G2":    "Gol 2",
        "Area_G3":    "Gol 3",
    })

    for lab in ordered_labels:
        if lab in df_sched.columns:
            df_sched[lab] = df_sched[lab] / 1000.0  # L/s → m³/s

    df_sched.index = np.arange(1, len(df_sched) + 1)
    return df_sched, row_is_branch

def _apply_schedule_col_widths(
    styler: pd.io.formats.style.Styler,
    df_for_cols: pd.DataFrame,
) -> pd.io.formats.style.Styler:
    """Adjusted schedule 用の列幅を全テーブルでそろえる。"""
    # テキスト列
    text_cols = ["Parent", "Child"]
    # 面積・Gol 列
    num_cols = ["Area (ha)", "Gol 1", "Gol 2", "Gol 3"]

    text_w = "120px"
    num_w  = "90px"
    qp_w   = "55px"   # 72 個の 5-day 列

    for c in text_cols:
        if c in df_for_cols.columns:
            styler = styler.set_properties(
                subset=[c],
                **{"min-width": text_w, "max-width": text_w},
            )

    for c in num_cols:
        if c in df_for_cols.columns:
            styler = styler.set_properties(
                subset=[c],
                **{"min-width": num_w, "max-width": num_w},
            )

    qp_cols = [lab for lab in ordered_labels if lab in df_for_cols.columns]
    if qp_cols:
        styler = styler.set_properties(
            subset=qp_cols,
            **{"min-width": qp_w, "max-width": qp_w},
        )

    return styler

def _style_adjusted_schedule_table(
    df_sched: pd.DataFrame,
    row_is_branch: pd.Series,
    bank: str,
) -> pd.io.formats.style.Styler:
    """
    Adjusted Schedule テーブルのスタイル:
      - Qp 列は小数点 3 桁
      - Branch 行のテキスト列を淡い緑 (#d9f99d)
      - 行ごとの MT-1/2/3 Adjusted 開始 5Day セルだけオレンジ (#ffcc66)
      - データバーは薄赤
      - 列幅・フォント・中央揃えは共通
    """
    if df_sched.empty:
        return df_sched.style

    cols = set(df_sched.columns)
    text_cols = {"Parent", "Child", "Area (ha)", "Gol 1", "Gol 2", "Gol 3"}
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
            lambda v, lab=lab: "" if (pd.isna(v) or abs(v) < 1e-9) else f"{v:,.3f}"
        )

    # ==== Bank×Golongan×Season → Adjusted index のマップ ====
    adj_index_map = _build_adjusted_mt_index_map()

    # 行ごとの MT-1/2/3 Adjusted index を計算
    mt1_idx_rows: list[int | None] = []
    mt2_idx_rows: list[int | None] = []
    mt3_idx_rows: list[int | None] = []

    for _, row in df_sched.iterrows():
        areas = [
            row.get("Gol 1", 0.0),
            row.get("Gol 2", 0.0),
            row.get("Gol 3", 0.0),
        ]
        gol_candidates = [
            g for g, a in enumerate(areas, start=1)
            if not (pd.isna(a) or abs(a) < 1e-9)
        ]
        if not gol_candidates:
            gol_candidates = [1, 2, 3]

        def _earliest(season: str) -> int | None:
            idx_list: list[int] = []
            for g in gol_candidates:
                idx = adj_index_map.get((bank, g, season), None)
                if idx is not None:
                    idx_list.append(idx)
            if not idx_list:
                return None
            return min(idx_list)

        mt1_idx_rows.append(_earliest("MT-1"))
        mt2_idx_rows.append(_earliest("MT-2"))
        mt3_idx_rows.append(_earliest("MT-3"))

    # Branch 行＋MT 開始セルのハイライト
    def _highlight(row: pd.Series):
        pos = int(row.name) - 1
        is_branch = bool(row_is_branch.iloc[pos]) if pos < len(row_is_branch) else False

        mt_indices: list[int] = []
        for idx in (mt1_idx_rows[pos], mt2_idx_rows[pos], mt3_idx_rows[pos]):
            if idx is not None:
                mt_indices.append(idx)

        mt_cols_this_row = {
            ordered_labels[i]
            for i in mt_indices
            if 0 <= i < len(ordered_labels)
        }

        styles: list[str] = []
        for col in row.index:
            color = ""
            if col in qp_cols and col in mt_cols_this_row:
                color = "#ffcc66"
            elif is_branch and col in text_cols:
                color = "#d9f99d"
            styles.append(f"background-color:{color}" if color else "")
        return styles

    styler = df_sched.style.format(fmt_dict)

    # データバー
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

    styler = styler.apply(_highlight, axis=1)

    # 列幅＆テーブル枠
    styler = _apply_schedule_col_widths(styler, df_sched)
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
                ("background-color", "#ffe4e6"),
                ("font-weight", "bold"),
                ("font-size", "11px"),
                ("border", "1px solid #999"),
                ("padding", "3px 5px"),
                ("white-space", "nowrap"),
                ("position", "sticky"),
                ("top", "0"),
                ("z-index", "2"),
                ("text-align", "center"),
            ],
        },
        {
            "selector": "td",
            "props": [
                ("border", "1px solid #999"),
                ("padding", "3px 5px"),
                ("white-space", "nowrap"),
                ("text-align", "center"),
            ],
        },
    ]).set_properties(**{"font-size": "12px"})

    return styler


def render_report_adjusted_schedule_section() -> None:
    """
    3. Adjusted Schedule  (A3 landscape 用)
    Left bank / Right bank の全 Main Canal について
    Target+Branch 行をまとめた 72列テーブルを順番に表示する。
    """
    st.markdown("### 3. Adjusted Schedule")

    df_case = st.session_state.get("qp_df_network_case", None)
    if df_case is None or df_case.empty:
        st.info("Adjusted Qp scenario has not been computed yet in Page5.")
        return

    # Main level の行から、Bank ごとの Main Canal 一覧を取得
    if "CanalLevel" in df_case.columns:
        main_rows = df_case[df_case["CanalLevel"].astype(str).str.strip() == "Main"].copy()
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
                sty = _style_adjusted_schedule_table(df_sched, flags, bank="Left")
                st.markdown(
                    "<div style='overflow-x:auto; text-align:left;'>"
                    f"{sty.to_html()}"
                    "</div>",
                    unsafe_allow_html=True,
                )
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
                sty = _style_adjusted_schedule_table(df_sched, flags, bank="Right")
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

# ========================================
# ページ本体
# ========================================

today_str = dt.date.today().strftime("%Y-%m-%d")
st.markdown("## Water Requirement Adjustment (Qp vs Adjusted Qp) Scenario – Report")
st.markdown(f"**Date:** {today_str}")

st.markdown(
    "<hr style='margin:0.5rem 0 0.75rem 0; border:0; border-top:1px solid #e5e7eb;'>",
    unsafe_allow_html=True,
)

# ---- レイアウト選択（1 行で表示：画面上だけ・印刷時は非表示）----
st.markdown(
    """
    <div class="no-print">
    <style>
      /* Report layout ラジオボタンの行間・余白を少し詰める */
      div[data-testid="stRadio"] > label {
        margin-bottom: 0.15rem;
      }
      div[data-testid="stRadio"] div[role="radiogroup"] {
        gap: 0.6rem;
        margin-top: 0.1rem;
        margin-bottom: 0.1rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("**Report layout**")
layout = st.radio(
    "Select Report Layout",
    options=("Summary (A4 portrait)", "Conditions (A3 portrait)", "Adjusted schedule (A3 landscape)"),
    horizontal=True,
)
# 下の余白も少しだけにする
st.markdown("<div style='height:0.25rem;'></div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)



# 選んだレイアウトに応じて @page の size をヒントとして指定（ブラウザによっては無視されます）
if layout.startswith("Summary"):
    page_size_css = "@page { size: A4 portrait; }"
elif layout.startswith("Conditions"):
    page_size_css = "@page { size: A3 portrait; }"
else:
    page_size_css = "@page { size: A3 landscape; }"

st.markdown(
    f"""
    <style>
    @media print {{
      {page_size_css}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# Page5 で Adjusted シナリオを計算済みであることを前提とする
df_case = st.session_state.get("qp_df_network_case", None)
if df_case is None or df_case.empty:
    st.info("Adjusted Qp scenario has not been computed on Page5 yet.")
    st.stop()

# ===== 1. Summary レイアウト =====
if layout.startswith("Summary"):

    st.markdown("### 1. Summary of Qp Comparison")

    left_5day = _build_5day_qp_pair(
        df_base=df_base,
        df_case=df_case,
        bank="Left",
        canal="Cipelang MC.",
        parent_name="HW",
        child_name="Excluder Cipelang",
    )
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

    render_annual_summary_and_detail_block(
        left_5day,
        right_5day,
        df_case_disp=None,
        mt_start_labels=None,
        bank_sel="",
        sec_name="",
        gol_sel=0,
    )

# ===== 2. Conditions レイアウト (A3 縦想定) =====
elif layout.startswith("Conditions"):
    render_report_condition_section()

# ===== 3. Adjusted Schedule レイアウト (A3 横想定) =====
else:
    render_report_adjusted_schedule_section()


# ------- 画面上だけ出したい説明（印刷時は非表示） -------
st.markdown(
    """
    <div class="no-print"
         style="margin-top:0.5rem; padding:8px 12px;
                border-radius:6px; background:#eef2ff;
                border:1px solid #c7d2fe; font-size:0.9rem;">
      Use your browser's print dialog (Ctrl+P) and choose paper size/orientation
      (e.g. A4 / A3, portrait / landscape) to print this page.
    </div>
    """,
    unsafe_allow_html=True,
)


from io import BytesIO

def make_conditions_excel() -> bytes:
    """
    Conditions テーブル（Left / Right）を色付き・罫線付きのまま
    Excel に書き出す。数値は小数点3桁まで。
    """
    sec_cfg_all = st.session_state.get("sec_start_config", None)
    if sec_cfg_all is None or sec_cfg_all.empty:
        return b""

    left_tbl  = _build_report_condition_table_for_bank("Left")
    right_tbl = _build_report_condition_table_for_bank("Right")

    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        workbook = writer.book
        border_fmt = workbook.add_format({"border": 1})

        # Left
        if not left_tbl.empty:
            sty_left = _style_report_condition_table(left_tbl)
            sty_left.to_excel(writer, sheet_name="Left", index=True)
            ws = writer.sheets["Left"]
            nrows, ncols = left_tbl.shape
            # index + header を含めた範囲に罫線
            ws.conditional_format(0, 0, nrows, ncols, {
                "type": "no_errors",
                "format": border_fmt,
            })

        # Right
        if not right_tbl.empty:
            sty_right = _style_report_condition_table(right_tbl)
            sty_right.to_excel(writer, sheet_name="Right", index=True)
            ws = writer.sheets["Right"]
            nrows, ncols = right_tbl.shape
            ws.conditional_format(0, 0, nrows, ncols, {
                "type": "no_errors",
                "format": border_fmt,
            })

    output.seek(0)
    return output.getvalue()


st.markdown("#### Export tables to Excel", unsafe_allow_html=True)
st.download_button(
    "Download Conditions (Excel)",
    data=make_conditions_excel(),
    file_name="Qp_Conditions.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

def make_schedule_excel() -> bytes:
    """
    Adjusted schedule（Main × Target+Branch 行）と
    Tertiary Block 情報を色付きのまま Excel に書き出す。
    """
    df_case = st.session_state.get("qp_df_network_case", None)
    if df_case is None or df_case.empty:
        return b""

    # === Main canal の一覧（Left / Right） ===
    if "CanalLevel" in df_case.columns:
        main_rows = df_case[df_case["CanalLevel"].astype(str).str.strip() == "Main"].copy()
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

    # === TB 用基礎データ ===
    tb_df = df_case.copy()
    for c in ["Bank", "Canal", "Class", "ParentKey", "ChildKey",
              "ParentName", "ChildName"]:
        if c in tb_df.columns:
            tb_df[c] = tb_df[c].astype(str).str.strip()
        else:
            tb_df[c] = ""

    tb_df["Class_upper"] = tb_df.get("Class", "").astype(str).str.upper()
    tb_rows = tb_df[tb_df["Class_upper"] == "TB"].copy()

    # 親子関係マップ
    parent_map: dict[str, tuple[str, int]] = {}
    for idx, r in tb_df.iterrows():
        child = str(r.get("ChildKey", "")).strip()
        parent = str(r.get("ParentKey", "")).strip()
        if child and child not in parent_map:
            parent_map[child] = (parent, idx)

    canal_level_series = df_case.get("CanalLevel")
    has_canal_level = canal_level_series is not None

    def _trace_to_secondary_main(child_key: str) -> tuple[str, str]:
        if not has_canal_level:
            return "", ""

        sec_name = ""
        main_name = ""
        visited = set()
        cur_key = child_key

        while cur_key and cur_key not in visited and cur_key in parent_map:
            visited.add(cur_key)
            parent_key, idx_row = parent_map[cur_key]
            rowp = df_case.iloc[idx_row]

            lvl = str(rowp.get("CanalLevel", "")).strip()
            canal_name = str(rowp.get("Canal", "")).strip()

            if lvl == "Secondary" and not sec_name:
                sec_name = canal_name
            if lvl == "Main" and not main_name:
                main_name = canal_name

            if sec_name and main_name:
                break

            cur_key = str(rowp.get("ParentKey", "")).strip()

        return sec_name, main_name

    tb_records: list[dict] = []
    for _, r in tb_rows.iterrows():
        bank = r.get("Bank", "")
        child_key = str(r.get("ChildKey", "")).strip()
        sec_name, main_name = _trace_to_secondary_main(child_key)
        if not main_name:
            main_name = str(r.get("Canal", "")).strip()

        rec = {
            "Bank": bank,
            "Main canal": main_name,
            "Secondary canal": sec_name,
            "TB name": str(r.get("ChildName", "")),
            "Parent": str(r.get("ParentName", "")),
        }

        rec["Area (ha)"] = r.get("Area", np.nan) if "Area" in r.index else r.get("BranchArea", np.nan)
        rec["Golongan_TB"] = str(r.get("Golongan_TB", ""))

        # ★ TB についても 72 ステップの流量（m³/s）を Excel 出力
        for lab in ordered_labels:
            val = r.get(lab, np.nan)
            try:
                rec[lab] = float(val) / 1000.0  # L/s → m³/s
            except Exception:
                rec[lab] = np.nan

        tb_records.append(rec)

    tb_all = pd.DataFrame(tb_records)

    # === Excel 書き出し ===
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        workbook = writer.book
        border_fmt = workbook.add_format({"border": 1})

        # ----- 1) Adjusted schedule: Left bank -----
        for mc in left_main_list:
            df_sched, flags = _build_adjusted_schedule_table_for_bank(df_case, "Left", mc)
            if df_sched.empty:
                continue

            # 小数点3桁に丸めてから Styler 適用
            df_x = df_sched.copy()
            qp_cols = [lab for lab in ordered_labels if lab in df_x.columns]
            if qp_cols:
                df_x[qp_cols] = df_x[qp_cols].astype(float).round(3)

            sty = _style_adjusted_schedule_table(df_x, flags, bank="Left")
            sheet_name = f"L_{mc[:20]}" or "Left"
            sty.to_excel(writer, sheet_name=sheet_name, index=True)

            ws = writer.sheets[sheet_name]
            nrows, ncols = df_x.shape
            ws.conditional_format(0, 0, nrows, ncols, {
                "type": "no_errors",
                "format": border_fmt,
            })

        # ----- 2) Adjusted schedule: Right bank -----
        for mc in right_main_list:
            df_sched, flags = _build_adjusted_schedule_table_for_bank(df_case, "Right", mc)
            if df_sched.empty:
                continue

            df_x = df_sched.copy()
            qp_cols = [lab for lab in ordered_labels if lab in df_x.columns]
            if qp_cols:
                df_x[qp_cols] = df_x[qp_cols].astype(float).round(3)

            sty = _style_adjusted_schedule_table(df_x, flags, bank="Right")
            sheet_name = f"R_{mc[:20]}" or "Right"
            sty.to_excel(writer, sheet_name=sheet_name, index=True)

            ws = writer.sheets[sheet_name]
            nrows, ncols = df_x.shape
            ws.conditional_format(0, 0, nrows, ncols, {
                "type": "no_errors",
                "format": border_fmt,
            })

        # ----- 3) Tertiary Block sheets (TB_<Main>) -----
        if not tb_all.empty:
            # Bank×Golongan×Season → Adjusted 5Day index のマップ
            adj_index_map = _build_adjusted_mt_index_map()

            def _gol_from_str(s: str) -> int | None:
                # 修正: r"\d+" -> r"(\d+)" に変更して group(1) が使えるようにしました
                m = re.search(r"(\d+)", str(s))
                return int(m.group(1)) if m else None

            for main_name, grp in tb_all.groupby("Main canal"):
                sheet_name = f"TB_{str(main_name)[:25]}" if str(main_name) else "TB_Unknown"

                cols_order = [
                    "Bank", "Main canal", "Secondary canal",
                    "TB name", "Parent", "Area (ha)", "Golongan_TB",
                ] + ordered_labels

                df_tb = grp[cols_order].copy()
                df_tb.index = np.arange(1, len(df_tb) + 1)

                # 面積・流量を 3桁丸め
                num_cols = ["Area (ha)"] + ordered_labels
                for c in num_cols:
                    if c in df_tb.columns:
                        df_tb[c] = pd.to_numeric(df_tb[c], errors="coerce").round(3)

                fmt_dict = {c: "{:,.3f}" for c in num_cols}

                # 行ごとの MT 開始セルを決定
                def _highlight_tb(row: pd.Series):
                    bank_row = str(row.get("Bank", "")).strip()
                    gol_str = row.get("Golongan_TB", "")
                    gol_idx = _gol_from_str(gol_str)
                    labels_for_row: set[str] = set()

                    if gol_idx is not None:
                        for season in ("MT-1", "MT-2", "MT-3"):
                            idx = adj_index_map.get((bank_row, gol_idx, season), None)
                            if idx is not None and 0 <= idx < len(ordered_labels):
                                labels_for_row.add(ordered_labels[idx])

                    styles: list[str] = []
                    for col in row.index:
                        color = ""
                        if col in labels_for_row:
                            color = "#ffcc66"  # Adjusted start
                        styles.append(f"background-color:{color}" if color else "")
                    return styles

                sty_tb = (
                    df_tb.style
                    .format(fmt_dict)
                    .apply(_highlight_tb, axis=1)
                    .set_properties(**{
                        "font-size": "12px",
                        "white-space": "nowrap",
                        "padding": "3px 5px",
                        "text-align": "center",
                    })
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
                                ("padding", "3px 5px"),
                                ("background-color", "#eef2ff"),
                                ("font-weight", "bold"),
                                ("white-space", "nowrap"),
                                ("font-size", "11px"),
                                ("text-align", "center"),
                            ],
                        },
                        {
                            "selector": "td",
                            "props": [
                                ("border", "1px solid #999"),
                                ("padding", "3px 5px"),
                                ("white-space", "nowrap"),
                                ("text-align", "center"),
                            ],
                        },
                    ])
                )

                sty_tb.to_excel(writer, sheet_name=sheet_name, index=True)
                ws = writer.sheets[sheet_name]
                nrows, ncols = df_tb.shape
                ws.conditional_format(0, 0, nrows, ncols, {
                    "type": "no_errors",
                    "format": border_fmt,
                })

    output.seek(0)
    return output.getvalue()


st.download_button(
    "Download Adjusted schedule (Excel)",
    data=make_schedule_excel(),
    file_name="Qp_AdjustedSchedule.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)