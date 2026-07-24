"""LCK 경기 데이터 분석 리포트 에이전트 - Streamlit UI

실행:
    cd lol_analyst
    streamlit run app.py
"""

from datetime import datetime

import streamlit as st

import analysis as A
from agent import run_analyst

st.set_page_config(page_title="LCK 분석 에이전트", page_icon="📊", layout="wide")

# ---------------------------------------------------------------
# 데이터 로드 (캐시)
# ---------------------------------------------------------------

@st.cache_data(show_spinner="데이터 불러오는 중...")
def load():
    d = A.load_data()
    return d["teams"], d["players"], d["champs"]


try:
    teams, players, champs = load()
except FileNotFoundError as e:
    st.error(f"{e}\n\n먼저 prep_lol.py를 실행해 data/ 폴더를 만들어 주세요.")
    st.stop()

# ---------------------------------------------------------------
# 세션 상태
# ---------------------------------------------------------------

for key, default in [("messages", []), ("history", []), ("last_report", "")]:
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------------------------------------------
# 사이드바
# ---------------------------------------------------------------

with st.sidebar:
    st.header("데이터")

    years = sorted(teams["year"].dropna().unique())
    n_games = teams["gameid"].nunique()
    st.caption(
        f"LCK {int(years[0])}~{int(years[-1])} · {n_games:,}경기 · "
        f"{teams['teamname'].nunique()}팀 · {players['playername'].nunique()}명"
    )

    with st.expander("연도별 경기 수"):
        st.dataframe(
            teams.groupby("year")["gameid"].nunique().rename("경기 수"),
            width='stretch',
        )

    with st.expander("팀 목록"):
        st.write(", ".join(sorted(teams["teamname"].dropna().unique())))

    st.divider()
    st.header("분석 기록")
    hist = st.session_state.history
    if hist:
        st.caption(f"이번 세션 {len(hist)}건")
        for i, h in enumerate(hist, 1):
            label = h.get("intent") or h["type"]
            st.write(f"{i}. {label} — {h['question'][:24]}")
    else:
        st.caption("아직 없음")

    st.divider()
    if st.session_state.last_report:
        st.download_button(
            "리포트 txt 다운로드",
            data=st.session_state.last_report.encode("utf-8-sig"),
            file_name=f"lck_report_{datetime.now():%Y%m%d_%H%M}.txt",
            mime="text/plain",
            width='stretch',
        )

    if st.button("대화 초기화", width='stretch'):
        st.session_state.messages = []
        st.session_state.history = []
        st.session_state.last_report = ""
        st.rerun()

# ---------------------------------------------------------------
# 본문
# ---------------------------------------------------------------

st.title("LCK 경기 데이터 분석 에이전트")
st.caption(
    "통계 계산은 pandas가 하고 LLM은 해석만 합니다. "
    "표본이 부족한 항목은 단정적으로 서술하지 않습니다."
)

if not st.session_state.messages:
    st.markdown("**이런 걸 물어볼 수 있어요**")
    cols = st.columns(4)
    samples = [
        "이 데이터 상태 어때?",
        "승패에 영향 주는 지표는?",
        "T1이랑 젠지 2026 상대전적",
        "저평가된 챔피언 있어?",
    ]
    for col, s in zip(cols, samples):
        col.button(s, width='stretch', key=f"sample_{s}",
                   on_click=lambda q=s: st.session_state.update(pending=q))

# 이전 대화 출력
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("stats"):
            with st.expander("계산 결과 원본 (pandas 출력)"):
                st.code(m["stats"], language=None)
        if m.get("chart_spec"):
            try:
                fig = A.make_chart(m["chart_spec"]["kind"], m["chart_spec"]["data"])
                st.pyplot(fig)
            except Exception as e:
                st.caption(f"차트를 그리지 못했습니다: {e}")

# ---------------------------------------------------------------
# 입력 처리
# ---------------------------------------------------------------

user_input = st.chat_input("무엇이 궁금하신가요?")
if not user_input and st.session_state.get("pending"):
    user_input = st.session_state.pop("pending")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("분석 중..."):
            try:
                res = run_analyst(user_input, st.session_state.history)
                answer = res["answer"]
                st.session_state.history = res["history"]
                chart_spec = res.get("chart_spec") or {}
                stats = res.get("stats_text", "")
                if res.get("intent") == "" and "리포트" in answer[:200]:
                    st.session_state.last_report = answer
            except Exception as e:
                answer, chart_spec, stats = f"오류가 발생했습니다: {e}", {}, ""

        st.markdown(answer)

        if stats:
            with st.expander("계산 결과 원본 (pandas 출력)"):
                st.code(stats, language=None)

        if chart_spec:
            try:
                fig = A.make_chart(chart_spec["kind"], chart_spec["data"])
                st.pyplot(fig)
            except Exception as e:
                st.caption(f"차트를 그리지 못했습니다: {e}")

    st.session_state.messages.append({
        "role": "assistant", "content": answer,
        "stats": stats, "chart_spec": chart_spec,
    })

    # 리포트는 언제든 내려받을 수 있게 보관
    if len(answer) > 500 and "##" in answer:
        st.session_state.last_report = answer

    st.rerun()