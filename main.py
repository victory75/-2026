import streamlit as st
from datetime import datetime
from zoneinfo import ZoneInfo

st.set_page_config(
    page_title="Streamlit Test",
    page_icon="✅"
)

st.title("✅ Streamlit 연결 테스트")

st.write("이 화면이 보이면 GitHub와 Streamlit이 정상적으로 연결되었습니다.")

st.divider()

st.write("⏰ 현재 시간 (한국 시간):")
st.write(datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S"))

st.caption("페이지를 새로고침했을 때 시간이 바뀌면 정상입니다.")

st.success("연결 성공!")

import io
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


@st.cache_data
def read_csv(file):
    """UploadedFile에서 바이트를 꺼내 utf-8-sig → cp949 순으로 읽기"""
    raw = file.getvalue() if hasattr(file, "getvalue") else file.read()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()


def std_cols(df):
    """열 이름 표준화 + 숫자형 변환 + time 파싱"""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    rename = {}
    for c in df.columns:
        k = c.lower().replace(" ", "")
        if k in ("time", "categories", "시간", "날짜", "datetime", "date"):
            rename[c] = "time"
        elif k in ("temperature", "온도"):
            rename[c] = "temperature"
        elif k in ("humidity", "습도"):
            rename[c] = "humidity"
        elif k == "ph":
            rename[c] = "ph"
        elif k == "ec":
            rename[c] = "ec"
        elif "생중량" in k:
            rename[c] = "생중량"
        elif "잎" in k:
            rename[c] = "잎 수"
        elif "지상부" in k and "길이" in k:
            rename[c] = "지상부 길이"
        elif "지하부" in k and "길이" in k:
            rename[c] = "지하부 길이"
        elif "지상부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지상부 질량"
        elif "지하부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지하부 질량"
        elif "개체" in k:
            rename[c] = "개체번호"
    df = df.rename(columns=rename)

    for c in df.columns:
        if c in ("time", "개체번호"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            num = pd.to_numeric(
                df[c].astype(str).str.replace(",", "")
                .str.extract(r"(-?\d+\.?\d*)")[0],
                errors="coerce",
            )
            if num.notna().mean() >= 0.5:
                df[c] = num

    if "생중량" not in df.columns and {"지상부 질량", "지하부 질량"} <= set(df.columns):
        df["생중량"] = df["지상부 질량"] + df["지하부 질량"]

    if "time" in df.columns:
        t = df["time"].astype(str).str.strip()
        parsed = pd.to_datetime(t, errors="coerce", format="mixed")
        bad = parsed.isna() & t.str.match(r"^\d{8}_\d{1,2}$")
        if bad.any():
            parts = t[bad].str.split("_", expand=True)
            parsed[bad] = (
                pd.to_datetime(parts[0], format="%Y%m%d")
                + pd.to_timedelta(parts[1].astype(int), unit="h")
            )
        df["time"] = parsed
    return df


def bar_chart_growth(df, col, title):
    """개체별 평균 막대 그래프 (상위·하위 25% 강조)"""
    v = df[col].dropna()
    if len(v) < 2:
        return None
    q75, q25 = v.quantile(0.75), v.quantile(0.25)
    df2 = df.dropna(subset=[col]).copy()
    df2["group"] = np.where(
        df2[col] >= q75, "상위 25%",
        np.where(df2[col] <= q25, "하위 25%", "중간"),
    )
    colors = {"상위 25%": "#2e7d32", "하위 25%": "#c62828", "중간": "#90a4ae"}

    fig = go.Figure()
    for grp, sub in df2.groupby("group"):
        fig.add_trace(
            go.Bar(
                x=sub["개체번호"],
                y=sub[col],
                name=grp,
                marker_color=colors[grp],
                hovertemplate="%{x}<br>%{y:.2f}<extra></extra>",
            )
        )
    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left", y=0.95),
        xaxis_title="개체번호",
        yaxis_title=title,
        barmode="overlay",
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
        margin=dict(l=40, r=20, t=50, b=40),
        height=280,
    )
    return fig


def box_plot(df, col, title):
    """생중량 분포 박스플롯"""
    v = df[col].dropna()
    if len(v) < 2:
        return None
    fig = go.Figure()
    fig.add_trace(
        go.Box(
            y=v,
            name=title,
            fillcolor="#1565c0",
            line_color="#0d47a1",
            boxmean=True,
            hovertemplate="%{y:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left", y=0.95),
        xaxis_title="",
        yaxis_title=title,
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
        margin=dict(l=40, r=20, t=50, b=40),
        height=280,
    )
    return fig


def scatter_trend(df, xcol, ycol, title):
    """산점도 + np.polyfit 추세선"""
    d = df.dropna(subset=[xcol, ycol]).copy()
    if len(d) < 2:
        return None
    r_val = d[xcol].corr(d[ycol])
    slope, intercept = np.polyfit(d[xcol], d[ycol], 1)
    x_fit = np.array([d[xcol].min(), d[xcol].max()])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=d[xcol],
            y=d[ycol],
            mode="markers",
            marker=dict(size=7, color="#1565c0", opacity=0.7),
            hovertemplate="%{x:.2f}, %{y:.2f}<extra></extra>",
            name="개체",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_fit,
            y=slope * x_fit + intercept,
            mode="lines",
            line=dict(color="#c62828", width=2, dash="dash"),
            name=f"추세선 (r={r_val:.3f})",
        )
    )
    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left", y=0.95),
        xaxis_title=xcol,
        yaxis_title=ycol,
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
        margin=dict(l=40, r=20, t=50, b=40),
        height=300,
    )
    return fig, r_val


def download_csv(df, filename):
    """한글 깨짐 없는 CSV 다운로드 버튼"""
    csv_data = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label=f"📥 {filename} 다운로드",
        data=csv_data,
        file_name=filename,
        mime="text/csv",
    )


def cors_card(col, r_val):
    """상관계수 카드"""
    color = (
        "#2e7d32" if abs(r_val) >= 0.7
        else ("#f9a825" if abs(r_val) >= 0.4 else "#c62828")
    )
    st.markdown(
        f"""
        <div style="background:#f5f5f5; border-left:4px solid {color};
        padding:8px 12px; border-radius:4px; margin-bottom:8px;">
        <b>{col} vs 생중량 상관계수 (r): {r_val:.3f}</b><br>
        <small>
        r ≥ 0.7 → 강한 양의 상관 &nbsp;|&nbsp;
        0.4 ≤ |r| &lt; 0.7 → 중간 상관 &nbsp;|&nbsp;
        |r| &lt; 0.4 → 약한 상관
        </small>
        </div>
        """,
        unsafe_allow_html=True,
    )


def fmt_id(val):
    """개체번호를 소수점 없이 표시"""
    if isinstance(val, float) and val == int(val):
        return str(int(val))
    return str(val)


# ──────────────────────────────────────────────
# 페이지 설정 및 기본 CSS
st.set_page_config(page_title="극지식물 생육 대시보드", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
   font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("🌱 우리 학교 극지식물 개체별 생육 차이 분석")


# ── 사이드바: 파일 업로드 ──────────────────────
with st.sidebar:
    st.header("📁 데이터 업로드")
    env_file = st.file_uploader("환경 데이터 CSV", type="csv")
    growth_file = st.file_uploader("생육 결과 데이터 CSV", type="csv")

    st.divider()
    st.subheader("📏 생육 측정 항목")
    growth_cols_opt = {
        "잎 수": "잎 수",
        "지상부 길이": "지상부 길이",
        "지하부 길이": "지하부 길이",
        "생중량": "생중량",
        "지상부 질량": "지상부 질량",
        "지하부 질량": "지하부 질량",
    }
    selected_growth = st.multiselect(
        "차트에 표시할 항목",
        options=list(growth_cols_opt.keys()),
        default=["생중량", "잎 수", "지상부 길이", "지하부 길이"],
    )
    selected_growth_cols = [growth_cols_opt[c] for c in selected_growth]

    st.divider()
    env_cols_opt = {
        "온도": "temperature",
        "습도": "humidity",
        "pH": "ph",
        "EC": "ec",
    }
    selected_env = st.multiselect(
        "환경 차트 항목",
        options=list(env_cols_opt.keys()),
        default=["온도", "습도", "pH", "EC"],
    )
    selected_env_cols = [env_cols_opt[c] for c in selected_env]

    st.divider()
    # 학교명: 파일명 앞부분에서 추출
    uploaded_name = "업로드하지 않음"
    if env_file is not None:
        try:
            uploaded_name = os.path.splitext(os.path.basename(env_file.name))[0]
            uploaded_name = uploaded_name.split("_")[0]
        except Exception:
            uploaded_name = env_file.name
    st.caption(f"🏫 학교명: **{uploaded_name}**")


# ── 데이터 로드 (한 번만) ──────────────────────
env_df = pd.DataFrame()
growth_df = pd.DataFrame()

if env_file is not None:
    with st.spinner("환경 데이터를 읽고 있습니다..."):
        env_df = std_cols(read_csv(env_file))

if growth_file is not None:
    with st.spinner("생육 결과 데이터를 읽고 있습니다..."):
        growth_df = std_cols(read_csv(growth_file))


# ── 탭 구성 ─────────────────────────────────────
tab1, tab2, tab3 = st.tabs(
    ["📖 실험 개요", "🌡️ 환경 데이터", "📊 생육 결과"]
)


# ══════════════════════════════════════════════
# TAB 1 — 실험 개요
# ══════════════════════════════════════════════
with tab1:
    try:
        st.subheader("🔬 연구 배경 및 목적")
        st.markdown(
            """
극지식물은 낮은 온도와 건조한 환경에서 살아남기 위해 특별한 적응 전략을 가집니다.
이 대시보드에서는 우리 학교 극지식물 **개체별 생육 차이**를 분석하고,
**잎 수·지상부 길이와 생중량 사이의 관계**를 살펴보며
**상위 25%와 하위 25% 개체**를 비교합니다.

> 💡 **주요 용어:**
> - **평균(mean)**: 모든 값을 더해서 개수로 나눈 대표값
> - **표준편차(standard deviation)**: 평균에서 얼마나 퍼져 있는지 나타내는 값 (클수록 들쭉날쭉)
> - **변이계수(CV, coefficient of variation)**: 표준편차를 평균으로 나눈 값 (단위가 다른 그룹끼리 비교 가능)
> - **상관계수(r, Pearson correlation coefficient)**: 두 변수의 선형 관계 강도 (-1~1, 1에 가까울수록 함께 증가)
            """
        )

        st.subheader("📋 데이터 요약")

        n_growth = len(growth_df) if not growth_df.empty else 0
        n_env = len(env_df) if not env_df.empty else 0

        # 측정 기간 (환경)
        if "time" in env_df.columns and not env_df.empty:
            t_sorted = env_df["time"].dropna().sort_values()
            if len(t_sorted) > 0:
                period_start = t_sorted.min().strftime("%Y-%m-%d")
                period_end = t_sorted.max().strftime("%Y-%m-%d")
                period_str = f"{period_start} ~ {period_end}"
            else:
                period_str = "정보 없음"
        else:
            period_str = "파일 없음"

        env_items = [
            c for c in ["temperature", "humidity", "ph", "ec"]
            if c in env_df.columns
        ]
        env_items_str = ", ".join(env_items) if env_items else "없음"

        growth_items = [
            c for c in [
                "잎 수", "지상부 길이", "지하부 길이",
                "생중량", "지상부 질량", "지하부 질량",
            ]
            if c in growth_df.columns
        ]
        growth_items_str = ", ".join(growth_items) if growth_items else "없음"

        summary_rows = [["구분", "내용"]]
        if n_growth > 0:
            summary_rows.append(["총 개체 수", f"{n_growth}개"])
        if not env_df.empty and len(env_df.get("time", pd.Series())) > 0:
            summary_rows.append(["측정 기간 (환경)", period_str])
        if n_env > 0:
            summary_rows.append(["환경 데이터 행 수", f"{n_env}행"])
        if env_items:
            summary_rows.append(["환경 측정 항목", env_items_str])
        if growth_items:
            summary_rows.append(["생육 측정 항목", growth_items_str])

        if len(summary_rows) > 1:
            st.table(
                pd.DataFrame(summary_rows[1:], columns=summary_rows[0])
            )
        else:
            st.info("📌 데이터 요약 정보를 표시할 수 없어요. CSV를 업로드해 주세요.")

        st.divider()

        # ── 주요 지표 카드 4개 ───────────────
        st.subheader("📊 주요 지표")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if n_growth > 0:
                st.metric("총 개체수", f"{n_growth}개")
            else:
                st.info("📌 생육 데이터가 업로드되지 않았어요.")

        with col2:
            if "temperature" in env_df.columns:
                v = env_df["temperature"].dropna()
                if len(v) >= 1:
                    st.metric(
                        "평균 온도",
                        f"{v.mean():.1f} ℃",
                    )
                else:
                    st.info("📌 온도 데이터가 없어요.")
            else:
                st.info("📌 온도 열이 없어요.")

        with col3:
            if "humidity" in env_df.columns:
                v = env_df["humidity"].dropna()
                if len(v) >= 1:
                    st.metric(
                        "평균 습도",
                        f"{v.mean():.1f} %",
                    )
                else:
                    st.info("📌 습도 데이터가 없어요.")
            else:
                st.info("📌 습도 열이 없어요.")

        with col4:
            growth_val_cols = [
                c for c in ["생중량", "지상부 질량", "지하부 질량"]
                if c in growth_df.columns
            ]
            if growth_val_cols:
                v = growth_df[growth_val_cols[0]].dropna()
                if len(v) >= 1:
                    st.metric(
                        "평균 생중량",
                        f"{v.mean():.2f} g",
                    )
                else:
                    st.info("📌 생중량 데이터가 없어요.")
            else:
                st.info("📌 생중량 열이 없어요.")

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


# ══════════════════════════════════════════════
# TAB 2 — 환경 데이터
# ══════════════════════════════════════════════
with tab2:
    try:
        if env_df.empty:
            st.info("📌 환경 데이터 CSV를 사이드바에서 업로드해 주세요.")
        else:
            st.subheader("🌡️ 환경 데이터 요약")

            env_sum_cols = [
                c for c in ["temperature", "humidity", "ph", "ec"]
                if c in env_df.columns
            ]
            if env_sum_cols:
                summary = (
                    env_df[env_sum_cols]
                    .dropna()
                    .agg(["mean", "min", "max"])
                    .round(2)
                )
                summary.index = ["평균", "최소", "최대"]
                st.dataframe(summary, width="stretch")
            else:
                st.info(
                    "📌 평균·최소·최대를 계산할 환경 열이 하나도 없어요."
                )

            st.divider()

            st.subheader("📈 날짜별 환경 변화 (일별 평균)")

            env_chart_cols = [
                c for c in selected_env_cols if c in env_df.columns
            ]
            chart_colors = {
                "temperature": "#e53935",
                "humidity": "#1e88e5",
                "ph": "#7cb342",
                "ec": "#8e24aa",
            }
            chart_titles = {
                "temperature": "온도 변화 (℃)",
                "humidity": "습도 변화 (%)",
                "ph": "pH 변화",
                "ec": "EC 변화 (mS/cm)",
            }

            if env_chart_cols:
                n_rows = len(env_chart_cols)
                fig_sub = make_subplots(
                    rows=n_rows,
                    cols=1,
                    subplot_titles=[
                        chart_titles[c] for c in env_chart_cols
                    ],
                    vertical_spacing=0.12,
                )
                for i, col in enumerate(env_chart_cols):
                    d = env_df.dropna(subset=[col, "time"]).copy()
                    if len(d) < 2:
                        continue
                    d["date"] = d["time"].dt.date
                    daily = d.groupby("date")[col].mean().reset_index()
                    if len(daily) < 2:
                        continue
                    fig_sub.add_trace(
                        go.Scatter(
                            x=daily["date"],
                            y=daily[col],
                            mode="lines",
                            name=chart_titles[col],
                            line=dict(
                                color=chart_colors.get(col, "#333"), width=2
                            ),
                            showlegend=False,
                            hovertemplate="%{x}<br>%{y:.2f}<extra></extra>",
                        ),
                        row=i + 1,
                        col=1,
                    )
                    fig_sub.update_xaxes(
                        title_text="날짜", row=i + 1, col=1
                    )
                    fig_sub.update_yaxes(
                        title_text=chart_titles[col], row=i + 1, col=1
                    )

                fig_sub.update_layout(
                    font=dict(
                        family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"
                    ),
                    height=180 + 180 * n_rows,
                    margin=dict(l=40, r=20, t=30, b=40),
                    showlegend=False,
                )
                st.plotly_chart(fig_sub, width="stretch")
            else:
                st.info("📌 차트에 표시할 환경 열이 없어요.")

            st.divider()

            with st.expander("📋 환경 데이터 원본 테이블 + CSV 다운로드"):
                st.dataframe(env_df, width="stretch")
                download_csv(env_df, "환경데이터_원본.csv")

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


# ══════════════════════════════════════════════
# TAB 3 — 생육 결과
# ══════════════════════════════════════════════
with tab3:
    try:
        if growth_df.empty:
            st.info(
                "📌 생육 결과 데이터 CSV를 사이드바에서 업로드해 주세요."
            )
        else:
            # ── 🥇 핵심 결과 카드 ─────────────
            st.subheader("🥇 핵심 결과")

            growth_val = [
                c for c in ["생중량", "지상부 질량", "지하부 질량"]
                if c in growth_df.columns
            ]
            main_val_col = growth_val[0] if growth_val else None

            cards_row = st.columns(3)
            if main_val_col:
                v = growth_df[main_val_col].dropna()
                mean_val = v.mean()
                min_idx = v.idxmin()
                max_idx = v.idxmax()
                min_id_val = growth_df.loc[min_idx, "개체번호"]
                max_id_val = growth_df.loc[max_idx, "개체번호"]
                min_val = v.min()
                max_val = v.max()

                cards_row[0].metric(
                    label=f"📈 평균 {main_val_col}",
                    value=f"{mean_val:.2f} g",
                )
                cards_row[1].metric(
                    label="🏆 생중량 최대 개체",
                    value=f"개체 {fmt_id(max_id_val)}",
                    delta=f"{max_val:.2f} g",
                )
                cards_row[2].metric(
                    label="📉 생중량 최소 개체",
                    value=f"개체 {fmt_id(min_id_val)}",
                    delta=f"{min_val:.2f} g",
                )
                st.caption(
                    "※ 생중량이 없는 학교는 지상부 질량 + 지하부 질량으로 자동 계산해요."
                )
            else:
                st.info("📌 생중량 관련 열이 없어요.")

            st.divider()

            # ── 개체별 생육 비교 (막대 그래프) ──
            st.subheader("📊 개체별 생육 비교")
            bar_cols = [
                c for c in selected_growth_cols if c in growth_df.columns
            ]
            if bar_cols:
                for col in bar_cols:
                    fig = bar_chart_growth(growth_df, col, f"개체별 평균 {col}")
                    if fig:
                        st.plotly_chart(fig, width="stretch")
                    else:
                        st.info(
                            f"📌 {col} 차트는 데이터가 부족해 표시할 수 없어요."
                        )
            else:
                st.info(
                    "📌 표시할 생육 측정 항목이 없어요. 사이드바에서 선택해 주세요."
                )

            st.divider()

            # ── 생중량 분포 (박스플롯) ──────────
            st.subheader("📦 생중량 분포")
            if main_val_col:
                fig_box = box_plot(
                    growth_df, main_val_col, f"{main_val_col} 분포"
                )
                if fig_box:
                    st.plotly_chart(fig_box, width="stretch")
                else:
                    st.info("📌 분포 차트를 표시할 데이터가 부족해요.")
            else:
                st.info("📌 생중량 열이 없어요.")

            st.divider()

            # ── 변이계수(CV) 안내 ───────────────
            if main_val_col:
                v = growth_df[main_val_col].dropna()
                if len(v) >= 2 and v.mean() != 0:
                    cv = v.std() / v.mean()
                    st.info(
                        f"💡 **변이계수(CV) = {cv:.3f}**  "
                        f"( 표준편차 ÷ 평균 )<br>"
                        f"CV가 클수록 개체 간 생중량 차이가 크다는 뜻이에요.",
                        icon="💡",
                    )

            st.divider()

            # ── 상관관계 분석 (산점도) ─────────
            st.subheader("🔗 상관관계 분석")

            corr_targets = []
            for sel_name, sel_col in growth_cols_opt.items():
                if (
                    sel_col in growth_df.columns
                    and sel_col != "개체번호"
                    and sel_col not in {"생중량", "지상부 질량", "지하부 질량"}
                ):
                    corr_targets.append((sel_name, sel_col))

            if corr_targets and main_val_col:
                for sel_name, sel_col in corr_targets:
                    result = scatter_trend(
                        growth_df, sel_col, main_val_col,
                        f"{sel_name} vs {main_val_col}"
                    )
                    if result is None:
                        st.info(
                            f"📌 {sel_name} vs 생중량 산점도는 데이터가 부족해요."
                        )
                    else:
                        fig, r_val = result
                        cors_card(sel_name, r_val)
                        st.plotly_chart(fig, width="stretch")
            else:
                st.info(
                    "📌 상관관계 분석을 할 수 있는 측정 항목이 없어요."
                )

            st.divider()

            with st.expander("📋 생육 데이터 원본 + CSV 다운로드"):
                st.dataframe(growth_df, width="stretch")
                download_csv(growth_df, "생육결과_원본.csv")

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")
