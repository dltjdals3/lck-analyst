"""LCK 경기 데이터 분석 리포트 에이전트 - LangGraph Supervisor

설계 원칙
  1. 모든 통계 계산은 analysis.py의 pandas 함수가 한다.
  2. LLM은 계산된 숫자를 해석해 문장으로 옮기는 일만 한다.
  3. 어떤 분석을 할지, 표본이 충분한지, 어떤 팀/선수를 가리키는지 같은
     판단도 최대한 파이썬이 하고 LLM에게는 확정된 결과만 넘긴다.
  4. LLM이 코드를 생성해 실행하는 방식(exec)은 쓰지 않는다.

단독 테스트:
    python agent.py
"""

import json
import os
import re
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

import analysis as A

load_dotenv(Path(__file__).parent.parent / ".env")

# 모델명은 여기서만 바꾸면 된다.
MODEL = os.getenv("LCK_AGENT_MODEL", "gpt-4o-mini")

llm = ChatOpenAI(model=MODEL, temperature=0.3)

GLOSSARY = """[용어] 아래 컬럼명은 다음 뜻이다. 임의로 번역하지 말고 이 표현을 쓸 것.
firstblood 선취(퍼스트블러드) / firstdragon 첫 드래곤 / firsttower 첫 타워
firstbaron 첫 바론 / firstherald 첫 전령 / firstmidtower 첫 미드 타워
golddiffat15 15분 골드 격차 / xpdiffat15 15분 경험치 격차 / csdiffat15 15분 CS 격차
dpm 분당 딜량 / damageshare 딜 지분 / cspm 분당 CS / visionscore 시야 점수
gamelength_min 경기 시간(분) / result 승패 / picks 픽 수 / banpick_rate 밴픽률
"""

SYSTEM_RULES = """당신은 e스포츠 데이터 분석가입니다.

반드시 지킬 규칙:
1. 아래 [분석 결과]에 제시된 수치만 근거로 삼는다. 없는 숫자를 만들어내지 않는다.
2. [서술 지침]과 [해석 지침]이 있으면 그 지시를 그대로 따른다.
3. 상관관계를 인과관계로 단정하지 않는다.
4. 표본이 부족하다고 표시된 항목은 해석하지 않고 부족하다고 밝힌다.
5. 쓸 내용이 없으면 없다고 쓴다. 분량을 채우려고 추측하지 않는다.
6. 롤 도메인 지식으로 데이터에 없는 내용을 보충하지 않는다.
   (예: 특정 챔피언의 성능, 선수의 평판, 경기 외적 사정)
7. 분석 대상 기간을 답변 앞부분에 반드시 밝힌다.
   결과에 '기간=전체 기간(2024~2026)'이라고 적혀 있으면 3년 합산임을 명시한다.
   특정 연도만 본 것처럼 쓰지 않는다.
8. 질문이 요구한 내용이 결과에 없으면 '표본이 부족하다'고 하지 않는다.
   그 대신 '이 분석은 그 항목을 다루지 않는다'고 정확히 밝히고,
   어떤 질문으로 다시 물으면 되는지 알려준다.
   표본 부족은 데이터는 있으나 건수가 적을 때만 쓰는 표현이다.
9. 직접 계산하지 않는다. 덧셈, 평균, 비율 산출을 스스로 하지 말고
   결과에 이미 있는 값만 인용한다. 필요한 합계가 없으면 없다고 말한다.
10. 등급 표기를 그대로 존중한다.
   '표본참고수준-단정금지'가 붙은 항목은 단정적으로 서술하지 않는다.
   '표본부족-해석금지'가 붙은 항목은 해석하지 않는다.
   표본이 충분한 항목이 하나도 없으면 그 사실을 밝힌다.
""" + GLOSSARY

# ---------------------------------------------------------------
# 분석 메뉴
# ---------------------------------------------------------------
# LLM에게 자유롭게 컬럼을 고르게 하면 KeyError나 무의미한 조합이 나온다.
# 가능한 분석을 미리 정의하고 그 중에서만 고르게 한다.

ANALYSIS_MENU = {
    "quality": {
        "desc": "데이터 품질 진단. 결측, 이상치, 자료형, 표본 구조",
        "needs": [],
    },
    "win_factor": {
        "desc": "승패에 영향을 주는 지표 분석. 상관관계, 오브젝트 선취 효과",
        "needs": [],
    },
    "champion": {
        "desc": "챔피언 전반 분석. 저평가/과대평가 챔피언 탐지 (특정 챔피언 지목 없을 때)",
        "needs": [],
    },
    "champion_detail": {
        "desc": "특정 챔피언 하나의 픽/밴/승률 조회",
        "needs": ["champion"],
    },
    "player": {
        "desc": "선수 개인 지표 또는 포지션별 선수 비교",
        "needs": [],
    },
    "player_champion": {
        "desc": "특정 선수의 챔피언별 전적",
        "needs": ["player"],
    },
    "head_to_head": {
        "desc": "두 팀의 상대전적과 경기 양상",
        "needs": ["two_teams"],
    },
    "draft": {
        "desc": "진영과 픽 순서 조합별 승률",
        "needs": [],
    },
    "team_trend": {
        "desc": "특정 팀 하나의 연도별 성적 추이와 로스터 변화 (팀이 지목된 경우)",
        "needs": ["team"],
    },
    "team_ranking": {
        "desc": "전체 팀 순위 비교. 어느 팀이 성적이 좋은지, 팀별 승률 줄세우기",
        "needs": [],
    },
}

POSITIONS = {
    "top": ["top", "탑", "탑라이너"],
    "jng": ["jng", "jungle", "정글", "정글러"],
    "mid": ["mid", "미드", "미드라이너"],
    "bot": ["bot", "adc", "원딜", "바텀"],
    "sup": ["sup", "support", "서폿", "서포터"],
}


# ---------------------------------------------------------------
# State
# ---------------------------------------------------------------

class AnalysisState(TypedDict):
    user_input: str          # 사용자 질문
    next_agent: str          # supervisor가 고른 노드
    intent: str              # analyze 노드가 고른 분석 종류
    entities: dict           # 파이썬이 추출한 팀/선수/챔피언/연도
    stats_text: str          # pandas가 계산한 결과 (LLM 입력용)
    profile_result: str      # 품질 진단 서술
    analysis_result: str     # 지표 분석 서술
    report_result: str       # 리포트 서술
    history: list            # 이번 세션에 누적된 분석 결과
    chart_spec: dict         # 차트 생성에 필요한 정보
    final_answer: str        # 최종 답변


# ---------------------------------------------------------------
# 데이터 (모듈 로드 시 1회)
# ---------------------------------------------------------------

_DATA = None


def get_data() -> dict:
    global _DATA
    if _DATA is None:
        _DATA = A.load_data()
    return _DATA


# ---------------------------------------------------------------
# 엔티티 추출 (LLM이 아니라 파이썬이 한다)
# ---------------------------------------------------------------

# 분석 관련 일반 단어. 이것들만으로 이뤄진 질문에는 선수·챔피언 이름이 없다.
COMMON_WORDS = {
    "데이터", "상태", "어때", "어떄", "분석", "해줘", "알려줘", "보여줘", "정리",
    "리포트", "요약", "품질", "진단", "지표", "승률", "승패", "경기", "전적",
    "상대", "비교", "추이", "변화", "저평가", "과대평가", "챔피언", "선수", "팀",
    "밴픽", "진영", "픽순서", "블루", "레드", "선픽", "후픽", "포지션", "년",
    "이번", "올해", "작년", "전체", "기간", "무엇", "뭐가", "있어", "있나",
    "궁금", "설명", "정도", "가장", "제일", "많이", "높은", "낮은", "순위",
    "그리고", "이랑", "하고", "에서", "으로", "인가", "인지", "부탁",
}


def _maybe_has_name(text: str) -> bool:
    """선수·챔피언 이름이 들어 있을 가능성이 있는지 판단

    한글 질문마다 변환 LLM을 부르면 낭비이므로,
    분석 일반어가 아닌 한글 토큰이 있을 때만 호출한다.
    """
    tokens = re.findall(r"[가-힣]{2,}", text)
    if not tokens:
        return False
    for t in tokens:
        if t in COMMON_WORDS:
            continue
        if any(t.startswith(w) or w.startswith(t) for w in COMMON_WORDS):
            continue
        return True
    return False


def _llm_transliterate(text: str, player_names: list, champ_names: list) -> dict:
    """한글 표기를 데이터에 있는 영문 표기로 바꾼다.

    후보 목록을 함께 주고 그 안에서만 고르게 한다.
    목록 없이 추측하게 하면 덜 알려진 선수명에서 실패한다.
    반환값은 다시 목록과 대조해 검증한다.
    """
    prompt = f"""문장에서 리그오브레전드 프로 선수명과 챔피언명을 찾아,
아래 목록에 있는 정확한 표기로 바꿔줘.

문장: {text}

[선수 목록]
{", ".join(sorted(player_names))}

[챔피언 목록]
{", ".join(sorted(champ_names))}

규칙:
- 한글 표기를 목록 안의 표기로 매칭한다.
  예: 시우 -> Siwoo, 쇼메이커 -> ShowMaker, 나르 -> Gnar, 아지르 -> Azir
- 반드시 위 목록에 있는 문자열을 그대로 쓴다. 목록에 없는 이름을 만들지 않는다.
- 문장에 없거나 목록에서 찾을 수 없으면 null을 넣는다.
- 팀 이름은 대상이 아니다.

아래 형식의 JSON만 출력. 설명이나 코드블록 없이 JSON만.
{{"player": "목록의 선수명 또는 null", "champion": "목록의 챔피언명 또는 null"}}"""

    try:
        raw = llm.invoke(prompt).content.strip()
        raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
        data = json.loads(raw)
        return {"player": data.get("player"), "champion": data.get("champion")}
    except Exception as e:
        print(f"[transliterate 실패] {e}")
        return {"player": None, "champion": None}


def _verify_name(candidate, valid_names) -> str | None:
    """LLM이 내놓은 이름을 실제 데이터 목록과 대조한다."""
    if not candidate or str(candidate).lower() == "null":
        return None
    key = A._norm(candidate)
    for n in valid_names:
        if A._norm(n) == key:
            return n
    matches = [n for n in valid_names if key and key in A._norm(n)]
    return matches[0] if len(matches) == 1 else None


def parse_entities(text: str) -> dict:
    """질문에서 연도, 팀, 선수, 챔피언, 포지션을 뽑는다.

    LLM에게 맡기면 표기 변형("젠지", "GenG")에서 실수하고
    존재하지 않는 이름을 만들어내기도 하므로 문자열 매칭으로 처리한다.
    """
    data = get_data()
    players, teams = data["players"], data["teams"]

    ent = {"year": None, "teams": [], "player": None,
           "champion": None, "position": None}

    # 연도 (올해/작년 같은 상대 표현은 데이터 최신 연도 기준)
    latest = int(teams["year"].dropna().max())
    ent["year"] = A.extract_year(text, latest=latest)

    # 팀 (등장 순서 유지)
    valid_teams = list(teams["teamname"].dropna().unique())
    found, seen = [], set()
    norm_text = A._norm(text)
    for official, aliases in A.TEAM_ALIASES.items():
        for cand in [official] + aliases:
            c = A._norm(cand)
            if len(c) >= 2 and c in norm_text and official not in seen:
                found.append((norm_text.index(c), official))
                seen.add(official)
                break
    ent["teams"] = [t for _, t in sorted(found)]

    # 선수 (긴 이름부터 매칭해 부분 일치 오류를 줄인다)
    names = sorted(players["playername"].dropna().unique(), key=len, reverse=True)
    for n in names:
        if len(str(n)) >= 3 and A._norm(n) in norm_text:
            ent["player"] = n
            break

    # 챔피언
    champs = sorted(players["champion"].dropna().unique(), key=len, reverse=True)
    for c in champs:
        if len(str(c)) >= 3 and A._norm(c) in norm_text:
            ent["champion"] = c
            break

    # 포지션
    for pos, aliases in POSITIONS.items():
        if any(A._norm(a) in norm_text for a in aliases):
            ent["position"] = pos
            break

    # 한글 표기 보정
    # 데이터는 선수·챔피언이 영문으로 저장돼 있어 한글 질문은 위 매칭이 실패한다.
    # 한글이 섞여 있고 아직 못 찾은 항목이 있으면 LLM 변환을 거친 뒤 목록으로 검증한다.
    if _maybe_has_name(text) and not (ent["player"] and ent["champion"]):
        guess = _llm_transliterate(text, list(names), list(champs))
        if not ent["player"]:
            ent["player"] = _verify_name(guess.get("player"), names)
        if not ent["champion"]:
            ent["champion"] = _verify_name(guess.get("champion"), champs)
        ent["transliterated"] = True
        # LLM이 이름을 지목했으나 목록 대조에서 탈락한 경우를 기록해 둔다.
        # 이 사실을 모르고 다른 분석으로 넘어가면 엉뚱한 결과에 잘못된 라벨이 붙는다.
        ent["unresolved"] = [
            f"선수:{guess['player']}" if guess.get("player") and not ent["player"] else None,
            f"챔피언:{guess['champion']}" if guess.get("champion") and not ent["champion"] else None,
        ]
        ent["unresolved"] = [u for u in ent["unresolved"] if u]

    return ent


# ---------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------

# 리포트 요청임을 확정하는 키워드.
# LLM이 "몇승 몇패야?" 같은 조회 질문을 리포트로 오분류하는 일이 잦아
# 키워드가 없으면 report로 보내지 않는다.
REPORT_KEYWORDS = [
    "정리", "리포트", "보고서", "요약", "종합", "지금까지",
    "마무리", "총정리", "브리핑",
]

# 이 에이전트가 다루는 범위
SCOPE_NOTE = """이 에이전트는 LCK 경기 데이터(2024~2026)만 분석합니다.

답할 수 있는 것:
- 데이터 품질 진단 (결측, 이상치, 표본 구조)
- 승패에 영향을 주는 지표 분석
- 팀 순위, 특정 팀의 연도별 추이, 두 팀의 상대전적
- 선수 지표, 포지션별 비교, 선수별 챔피언 전적
- 챔피언 픽/밴/승률, 저평가·과대평가 챔피언 탐지
- 진영과 픽 순서 조합별 승률
- 위 분석들을 모은 리포트 초안

답할 수 없는 것:
- 경기 일정, 중계, 순위표 등 실시간 정보
- 데이터에 없는 연도나 리그
- 선수 평판, 경기 외적 사정, 향후 결과 예측
- LCK와 무관한 주제"""


def supervisor_node(state: AnalysisState) -> AnalysisState:
    """질문 유형을 네 갈래로 나눈다"""
    prompt = f"""사용자 질문: {state['user_input']}

이 시스템은 LCK 프로 경기 데이터 분석 도구다.
아래 중 가장 적합한 하나를 골라 이름만 소문자로 출력해.

- profile: 데이터 자체가 어떤지, 품질은 괜찮은지, 무슨 컬럼이 있는지 묻는 경우
- analyze: 팀·선수·챔피언·지표·밴픽 등 구체적인 분석을 요청하는 경우
- report: 앞서 수행한 분석들을 모아 정리하거나 리포트로 만들어 달라는 경우
- out_of_scope: LCK 경기 데이터로 답할 수 없는 질문
  (날씨, 인사말, 잡담, 다른 게임, 실시간 일정, 미래 예측 등)

반드시 profile, analyze, report, out_of_scope 중 하나만 출력."""

    choice = llm.invoke(prompt).content.strip().lower()
    choice = re.sub(r"[^a-z_]", "", choice)
    if choice not in ("profile", "analyze", "report", "out_of_scope"):
        print(f"[경고] supervisor 이상 응답: {choice!r} -> analyze로 대체")
        choice = "analyze"

    ent = parse_entities(state["user_input"])
    has_target = bool(ent["teams"] or ent["player"] or ent["champion"]
                      or ent["position"])
    text = state["user_input"]

    # 특정 대상이 지목됐는데 profile로 가면 엉뚱한 답이 나온다.
    # (예: "나르는 어때?" -> 데이터 품질 진단)
    if choice == "profile" and has_target:
        print("[supervisor] 대상이 지목되어 analyze로 전환")
        choice = "analyze"

    # 리포트는 키워드가 있을 때만 인정한다.
    # (예: "디플러스기아 올해 몇승 몇패야?" -> 조회 질문인데 report로 가던 문제)
    if choice == "report" and not any(k in text for k in REPORT_KEYWORDS):
        print("[supervisor] 리포트 키워드가 없어 analyze로 전환")
        choice = "analyze"

    # 반대로 키워드가 뚜렷하면 report로 올린다.
    if choice == "analyze" and not has_target \
            and any(k in text for k in REPORT_KEYWORDS):
        print("[supervisor] 리포트 키워드 감지 -> report로 전환")
        choice = "report"

    # 실제 팀·선수·챔피언이 잡혔다면 범위 밖일 리 없다.
    if choice == "out_of_scope" and has_target:
        print("[supervisor] 대상이 지목되어 analyze로 전환")
        choice = "analyze"

    state["next_agent"] = choice
    state["entities"] = ent
    print(f"[supervisor] {choice} / entities={ent}")
    return state


def route_agent(state: AnalysisState) -> str:
    return state["next_agent"]


# ---------------------------------------------------------------
# profile 노드
# ---------------------------------------------------------------

def profile_node(state: AnalysisState) -> AnalysisState:
    data = get_data()
    teams, players = data["teams"], data["players"]

    t = A.profile_data(teams, "teams", counterpart=players)
    p = A.profile_data(players, "players", counterpart=teams)
    stats = "=== 팀 데이터 ===\n" + t["text"] + "\n\n=== 선수 데이터 ===\n" + p["text"]
    state["stats_text"] = stats

    prompt = f"""{SYSTEM_RULES}

사용자 질문: {state['user_input']}

[분석 결과 - pandas가 계산한 값]
{stats}

위 진단 결과를 사용자가 이해하기 쉽게 정리해줘.
특히 이 데이터를 분석할 때 주의해야 할 구조적 특성을 짚어줘.
숫자는 위에 있는 값을 그대로 쓰고 새로 만들지 마."""

    state["profile_result"] = llm.invoke(prompt).content
    state.setdefault("history", []).append(
        {"type": "profile", "question": state["user_input"], "stats": stats}
    )
    return state


# ---------------------------------------------------------------
# analyze 노드
# ---------------------------------------------------------------

def pick_analysis(state: AnalysisState) -> str:
    """메뉴에서 분석 종류를 고른다. 엔티티로 먼저 좁히고 애매하면 LLM에게 묻는다."""
    ent = state["entities"]
    text = state["user_input"]
    norm = A._norm(text)

    # 엔티티만으로 확정 가능한 경우는 LLM을 부르지 않는다
    if ent["player"] and ent["champion"]:
        return "player_champion"
    if len(ent["teams"]) >= 2:
        return "head_to_head"
    if ent["champion"] and not ent["player"] and not ent["teams"]:
        return "champion_detail"

    menu = "\n".join(f"- {k}: {v['desc']}" for k, v in ANALYSIS_MENU.items())
    prompt = f"""사용자 질문: {text}
추출된 정보: 팀={ent['teams']}, 선수={ent['player']}, 챔피언={ent['champion']}, 포지션={ent['position']}, 연도={ent['year']}

아래 분석 중 가장 적합한 하나를 골라 키 이름만 출력해.
{menu}

반드시 키 이름 하나만 출력."""

    choice = llm.invoke(prompt).content.strip().lower()
    choice = re.sub(r"[^a-z_]", "", choice)
    if choice not in ANALYSIS_MENU:
        print(f"[경고] 분석 선택 이상: {choice!r} -> win_factor로 대체")
        choice = "win_factor"
    return choice


def run_analysis(intent: str, ent: dict) -> tuple:
    """선택된 분석을 실행하고 (텍스트, 차트종류, 결과dict) 반환"""
    data = get_data()
    teams, players, champs = data["teams"], data["players"], data["champs"]
    year = ent.get("year")

    if intent == "quality":
        r = A.profile_data(teams, "teams", counterpart=players)
        return r["text"], None, r

    if intent == "win_factor":
        corr = A.correlate(teams, [
            "golddiffat15", "xpdiffat15", "csdiffat15",
            "dragons", "barons", "towers", "visionscore",
            "firstblood", "firstdragon", "firsttower",
        ], year=year)
        parts = [corr["text"]]
        for col in ["firstdragon", "firsttower", "firstblood"]:
            parts.append(A.compare_groups(teams, col, "result", year=year)["text"])
        return "\n\n".join(parts), "corr_bar", corr

    if intent == "champion":
        r = A.find_outlier_champs(champs, players, teams, year=year)
        return r["text"], "champ_scatter", r

    if intent == "champion_detail":
        if not ent.get("champion"):
            return ("[조회 실패] 질문에서 챔피언명을 찾지 못했습니다. "
                    "영문 표기로 다시 알려주세요. 예: Aphelios, Azir. "
                    "추측해서 답하면 안 됩니다."), None, None
        r = A.champion_stats(champs, players, teams, ent["champion"], year=year)
        return r["text"], None, r

    if intent == "player":
        # 질문에 챔피언이 언급됐는데 식별하지 못한 채로 선수 전체 지표를 주면,
        # LLM이 그 결과에 챔피언 이름을 붙여 잘못된 답을 만든다.
        unresolved_champ = [u for u in ent.get("unresolved", [])
                            if u.startswith("챔피언:")]
        if unresolved_champ and ent.get("player"):
            return (f"[조회 실패] 질문에 챔피언이 언급된 것으로 보이나 "
                    f"데이터에서 찾지 못했습니다 ({', '.join(unresolved_champ)}). "
                    "챔피언별 전적을 낼 수 없습니다. "
                    "영문 표기로 다시 알려주세요. "
                    "이 선수의 전체 지표를 챔피언별 성적으로 제시하면 안 됩니다."), None, None

        if ent.get("player"):
            r = A.player_stats(players, player=ent["player"], year=year)
            return r["text"], None, r
        if ent.get("position"):
            r = A.player_stats(players, position=ent["position"], year=year)
            return r["text"], None, r
        return ("선수명 또는 포지션을 함께 알려주세요. "
                "예: '쇼메이커 지표 보여줘', '미드 선수 비교해줘'"), None, None

    if intent == "player_champion":
        if not ent.get("player"):
            return ("[조회 실패] 질문에서 선수명을 찾지 못했습니다. "
                    "데이터에 등록된 표기와 일치하지 않았을 수 있습니다. "
                    "영문 활동명으로 다시 알려주세요. 예: ShowMaker, Faker, Chovy. "
                    "이 경우 승률이나 전적을 추측해서 답하면 안 됩니다."), None, None
        r = A.player_champion(players, ent["player"],
                              champion=ent.get("champion"), year=year)
        return r["text"], None, r

    if intent == "head_to_head":
        if len(ent.get("teams", [])) < 2:
            return "비교할 두 팀을 알려주세요. 예: 'T1 대 젠지 상대전적'", None, None
        r = A.head_to_head(teams, ent["teams"][0], ent["teams"][1], year=year)
        return r["text"], None, r

    if intent == "draft":
        r = A.draft_order(teams, year=year)
        return r["text"], "combo_bar", r

    if intent == "team_ranking":
        r = A.team_ranking(teams, year=year)
        return r["text"], "team_ranking", r

    if intent == "team_trend":
        if not ent.get("teams"):
            # 특정 팀이 없으면 전체 순위가 더 나은 답이다
            r = A.team_ranking(teams, year=year)
            return r["text"], "team_ranking", r
        r = A.team_trend(teams, players, ent["teams"][0], year=year)
        # 특정 연도만 본 경우 연도별 추이 차트는 의미가 없다
        return r["text"], (None if year else "team_trend"), r

    return "해당 분석을 찾지 못했습니다.", None, None


def analyze_node(state: AnalysisState) -> AnalysisState:
    intent = pick_analysis(state)
    state["intent"] = intent
    print(f"[analyze] intent={intent}")

    stats, chart, raw = run_analysis(intent, state["entities"])
    state["stats_text"] = stats
    if chart and raw:
        state["chart_spec"] = {"kind": chart, "data": raw}

    prompt = f"""{SYSTEM_RULES}

사용자 질문: {state['user_input']}

[분석 결과 - pandas가 계산한 값]
{stats}

위 수치를 근거로 질문에 답해줘.
결과에 포함된 [서술 지침], [해석 지침], [해석 제약], 주의 문구를 반드시 반영할 것.
표본이 부족하다고 표시됐으면 단정적인 결론을 내지 말 것."""

    state["analysis_result"] = llm.invoke(prompt).content
    state.setdefault("history", []).append(
        {"type": "analysis", "intent": intent,
         "question": state["user_input"], "stats": stats}
    )
    return state


# ---------------------------------------------------------------
# report 노드
# ---------------------------------------------------------------

def report_node(state: AnalysisState) -> AnalysisState:
    history = state.get("history", [])
    if not history:
        state["report_result"] = (
            "아직 수행한 분석이 없습니다. "
            "먼저 데이터 진단이나 지표 분석을 요청해 주세요. "
            "예: '이 데이터 상태 어때?', '승패에 영향 주는 지표 알려줘'"
        )
        return state

    blocks = []
    for i, h in enumerate(history, 1):
        blocks.append(f"--- 분석 {i} ({h['type']}) ---\n"
                      f"질문: {h['question']}\n{h['stats']}")
    joined = "\n\n".join(blocks)

    prompt = f"""{SYSTEM_RULES}

사용자 요청: {state['user_input']}

아래는 이번 세션에서 수행한 분석 {len(history)}건의 계산 결과다.

{joined}

이 결과들만으로 분석 리포트 초안을 작성해줘.

구성:
1. 분석 개요 - 무엇을 어떤 데이터로 분석했는가
2. 주요 발견 - 수치와 함께
3. 해석의 한계 - 각 분석에 붙은 주의사항을 종합
4. 추가로 확인해볼 사항

규칙:
- 위에 없는 분석 결과를 지어내지 말 것
- 수행하지 않은 분석을 했다고 쓰지 말 것
- 한계 항목을 생략하지 말 것"""

    state["report_result"] = llm.invoke(prompt).content
    return state


# ---------------------------------------------------------------
# final 노드
# ---------------------------------------------------------------

def out_of_scope_node(state: AnalysisState) -> AnalysisState:
    """범위 밖 질문. LLM을 부르지 않고 고정 안내를 준다.

    LLM에게 맡기면 데이터와 무관한 질문에도 그럴듯한 분석을 붙이려 한다.
    """
    state["analysis_result"] = (
        "이 질문은 LCK 경기 데이터로 답할 수 있는 범위를 벗어납니다.\n\n"
        + SCOPE_NOTE
    )
    return state


def final_node(state: AnalysisState) -> AnalysisState:
    parts = []
    if state.get("profile_result"):
        parts.append(state["profile_result"])
    if state.get("analysis_result"):
        parts.append(state["analysis_result"])
    if state.get("report_result"):
        parts.append(state["report_result"])
    state["final_answer"] = "\n\n".join(parts) if parts else "처리된 결과가 없습니다."
    return state


# ---------------------------------------------------------------
# 그래프
# ---------------------------------------------------------------

graph = StateGraph(AnalysisState)
graph.add_node("supervisor", supervisor_node)
graph.add_node("profile", profile_node)
graph.add_node("analyze", analyze_node)
graph.add_node("report", report_node)
graph.add_node("out_of_scope", out_of_scope_node)
graph.add_node("final", final_node)

graph.set_entry_point("supervisor")
graph.add_conditional_edges("supervisor", route_agent, {
    "profile": "profile",
    "analyze": "analyze",
    "report": "report",
    "out_of_scope": "out_of_scope",
})
graph.add_edge("profile", "final")
graph.add_edge("analyze", "final")
graph.add_edge("report", "final")
graph.add_edge("out_of_scope", "final")
graph.add_edge("final", END)

app_graph = graph.compile()


def run_analyst(user_input: str, history: list | None = None) -> dict:
    """외부(app.py)에서 호출하는 진입점

    history를 넘기면 이전 분석 결과를 이어받는다.
    State는 호출마다 새로 만들어지므로 세션 기억은 밖에서 보관해야 한다.
    """
    initial: AnalysisState = {
        "user_input": user_input,
        "next_agent": "",
        "intent": "",
        "entities": {},
        "stats_text": "",
        "profile_result": "",
        "analysis_result": "",
        "report_result": "",
        "history": list(history) if history else [],
        "chart_spec": {},
        "final_answer": "",
    }
    result = app_graph.invoke(initial)
    return {
        "answer": result["final_answer"],
        "history": result.get("history", []),
        "chart_spec": result.get("chart_spec", {}),
        "intent": result.get("intent", ""),
        "entities": result.get("entities", {}),
        "stats_text": result.get("stats_text", ""),
    }


if __name__ == "__main__":
    tests = [
        "이 데이터 상태 어때?",
        "T1이랑 젠지 2026년 상대전적 어때?",
        "쇼메이커 아지르 승률 알려줘",
        "저평가된 챔피언 있어?",
        "지금까지 분석한 거 리포트로 정리해줘",
    ]
    hist = []
    for q in tests:
        print("\n" + "=" * 66)
        print(f"Q: {q}")
        print("=" * 66)
        r = run_analyst(q, hist)
        hist = r["history"]
        print(r["answer"])