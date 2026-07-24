"""LCK 분석 함수 - 1단계 + 2단계

계산은 전부 여기서 한다. LLM은 이 함수들이 반환한 숫자를 해석만 한다.
각 함수는 dict를 반환하며, "text" 키에 LLM 프롬프트에 그대로 넣을 요약이 들어있다.

[1단계] profile_data / compare_groups / correlate
[2단계] player_stats / player_champion / head_to_head / draft_order / team_trend
[3단계] find_outlier_champs / make_chart

단독 테스트:
    python analysis.py
"""

import re
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"

# ---------------------------------------------------------------
# 표본 등급 기준
# ---------------------------------------------------------------

GRADE_SOLID = 30    # 통계적으로 의미 있음
GRADE_REF = 10      # 참고 수준
GRADE_WEAK = 5      # 숫자만 제시, 해석 금지

# ---------------------------------------------------------------
# 지표 분류
# ---------------------------------------------------------------

# 경기 종료 시점 누적값. 승패와 거의 동어반복이라 예측 지표로 쓸 수 없다.
LEAKAGE_COLS = {
    "towers", "opp_towers", "inhibitors", "opp_inhibitors",
    "dragons", "opp_dragons", "barons", "opp_barons",
    "elders", "opp_elders", "heralds", "void_grubs", "atakhans",
    "totalgold", "earnedgold", "teamkills", "teamdeaths",
    "kills", "deaths", "assists", "damagetochampions",
    "turretplates", "gspd",
}

# 15분 시점에 확정되는 값. 예측 지표로 쓸 수 있다.
EARLY_COLS = {
    "golddiffat15", "xpdiffat15", "csdiffat15",
    "goldat15", "xpat15", "csat15",
    "killsat15", "deathsat15", "assistsat15",
}

# 사건 발생 여부
EVENT_COLS = {
    "firstblood", "firstdragon", "firstherald", "firstbaron",
    "firsttower", "firstmidtower", "firsttothreetowers",
}

BINARY_HINT_COLS = EVENT_COLS | {
    "result", "playoffs", "firstbloodkill",
    "firstbloodassist", "firstbloodvictim",
}

# ---------------------------------------------------------------
# 팀명 별칭
# ---------------------------------------------------------------
# 사용자가 "젠지", "geng", "gen g" 어떻게 쓰든 정식 명칭으로 바꾼다.
# LLM에게 팀명 추출을 시키면 표기 변형에서 실수하므로 파이썬이 처리한다.

TEAM_ALIASES = {
    "T1": ["t1", "티원", "티투", "skt", "sktelecom", "sk"],
    "Gen.G": ["geng", "gen", "젠지", "겐지", "genggaming"],
    "Dplus Kia": ["dplus", "dpluskia", "dk", "디플러스", "디케이", "담원", "damwon"],
    "Hanwha Life Esports": ["hle", "hanwha", "한화", "한화생명", "hanwhalife"],
    "KT Rolster": ["kt", "ktrolster", "케이티", "케티"],
    "Kiwoom DRX": ["drx", "kiwoomdrx", "디알엑스", "키움", "kiwoom"],
    "Nongshim RedForce": ["ns", "nsrf", "nongshim", "농심", "레드포스", "nongshimredforce"],
    "BNK FEARX": ["bnk", "fearx", "bnkfearx", "피어엑스", "비엔케이"],
    "HANJIN BRION": ["bro", "brion", "hanjin", "브리온", "한진", "hanjinbrion", "okbrion"],
    "DN SOOPers": ["dn", "soopers", "soop", "dnsoopers", "숲", "디엔", "수퍼스"],
}


def _norm(s: str) -> str:
    """비교용 정규화: 소문자 + 공백/점/하이픈 제거"""
    return re.sub(r"[\s.\-_]", "", str(s)).lower()


def resolve_team(name: str, valid_names: list | None = None) -> str | None:
    """입력 문자열을 정식 팀명으로 변환. 못 찾으면 None"""
    if not name:
        return None
    key = _norm(name)

    # 정식 명칭 직접 매칭
    for official in TEAM_ALIASES:
        if _norm(official) == key:
            return official

    # 별칭 매칭
    for official, aliases in TEAM_ALIASES.items():
        if key in [_norm(a) for a in aliases]:
            return official

    # 부분 문자열 매칭 (마지막 수단)
    for official, aliases in TEAM_ALIASES.items():
        for cand in [official] + aliases:
            c = _norm(cand)
            if len(c) >= 2 and (c in key or key in c):
                return official

    # 데이터에 있는 실제 이름과 대조
    if valid_names:
        for v in valid_names:
            if _norm(v) == key:
                return v
    return None


def resolve_player(name: str, players: pd.DataFrame) -> str | None:
    """선수명을 데이터상 표기로 변환. 대소문자 무시, 부분 일치 허용"""
    if not name:
        return None
    key = _norm(name)
    names = players["playername"].dropna().unique()

    for n in names:
        if _norm(n) == key:
            return n
    matches = [n for n in names if key in _norm(n)]
    if len(matches) == 1:
        return matches[0]
    return None


RELATIVE_YEAR = {
    0: ["올해", "금년", "이번시즌", "이번 시즌", "올시즌", "올 시즌", "현재시즌"],
    -1: ["작년", "지난해", "지난시즌", "지난 시즌", "전시즌"],
    -2: ["재작년", "재작", "2년전", "2년 전"],
}


def extract_year(text: str, latest: int | None = None) -> int | None:
    """문장에서 연도를 뽑아낸다.

    '올해', '작년' 같은 상대 표현은 데이터의 최신 연도를 기준으로 환산한다.
    실제 달력 연도가 아니라 데이터 기준이어야 결과와 어긋나지 않는다.
    """
    m = re.search(r"(20\d{2})", str(text))
    if m:
        return int(m.group(1))

    if latest is None:
        return None

    norm = _norm(text)
    for offset, words in RELATIVE_YEAR.items():
        if any(_norm(w) in norm for w in words):
            return latest + offset
    return None


# ---------------------------------------------------------------
# 로드
# ---------------------------------------------------------------

def load_data() -> dict:
    files = {
        "players": "lck_players.csv",
        "teams": "lck_teams.csv",
        "champs": "champ_stats.csv",
    }
    out = {}
    for key, name in files.items():
        path = DATA_DIR / name
        if not path.exists():
            raise FileNotFoundError(
                f"{path} 가 없습니다. prep_lol.py를 먼저 실행하세요."
            )
        df = pd.read_csv(path, low_memory=False)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        out[key] = df
    return out


# ---------------------------------------------------------------
# 표본 등급
# ---------------------------------------------------------------

def grade_sample(n: int) -> dict:
    if n >= GRADE_SOLID:
        return {
            "n": n, "grade": "표본충분",
            "instruction": "통계적으로 의미 있는 차이로 서술해도 된다. "
                           "단 상관을 인과로 단정하지는 말 것.",
        }
    if n >= GRADE_REF:
        return {
            "n": n, "grade": "표본참고수준-단정금지",
            "instruction": "경향은 보이나 단정할 수 없는 수준이다. "
                           "'참고 수준'임을 반드시 밝힐 것.",
        }
    if n >= GRADE_WEAK:
        return {
            "n": n, "grade": "표본부족-해석금지",
            "instruction": "숫자만 제시하고 해석하지 말 것. "
                           "표본이 부족해 판단할 수 없다고 명시할 것.",
        }
    return {
        "n": n, "grade": "표본매우부족-비율산출금지",
        "instruction": f"표본이 {n}건뿐이다. 비율·승률을 계산해서 제시하지 말 것. "
                       "전적만 나열하고 분석이 불가능하다고 밝힐 것.",
    }


def swing_range(wins: int, n: int) -> str:
    """승률이 1경기 결과에 얼마나 흔들리는지"""
    if n == 0:
        return "표본 없음"
    cur = wins / n * 100
    if_lose = wins / (n + 1) * 100
    if_win = (wins + 1) / (n + 1) * 100
    return (f"현재 {cur:.1f}% / 1패 추가 시 {if_lose:.1f}% / "
            f"1승 추가 시 {if_win:.1f}%")


def _filter_year(df: pd.DataFrame, year: int | None) -> pd.DataFrame:
    return df if year is None else df[df["year"] == year]


def _period(year: int | None) -> str:
    return f"{year}년" if year else "전체 기간(2024~2026)"


# ===============================================================
# 1단계
# ===============================================================

def profile_data(df: pd.DataFrame, kind: str = "teams",
                 counterpart: pd.DataFrame | None = None) -> dict:
    """데이터 품질 진단"""
    result = {"kind": kind, "shape": {"rows": len(df), "cols": df.shape[1]}}
    lines = [f"[규모] {len(df):,}행 x {df.shape[1]}열 ({kind})"]

    if "year" in df.columns and "gameid" in df.columns:
        by_year = df.groupby("year")["gameid"].nunique().to_dict()
        result["games_by_year"] = by_year
        lines.append("[연도별 경기 수] " +
                     ", ".join(f"{y}년 {n}경기" for y, n in sorted(by_year.items())))

    # ---------- 결측 3분류 ----------
    na = df.isna().sum()
    na = na[na > 0]
    structural, empty, real = {}, [], {}
    other = "players" if kind == "teams" else "teams"

    for col, cnt in na.items():
        rate = cnt / len(df) * 100
        if rate < 100.0:
            real[col] = {"count": int(cnt), "rate": round(rate, 1)}
            continue
        if counterpart is not None and col in counterpart.columns \
                and counterpart[col].notna().any():
            structural[col] = other
        else:
            empty.append(col)

    result["missing_structural"] = structural
    result["missing_empty"] = empty
    result["missing_real"] = real

    if structural:
        names = sorted(structural)
        lines.append(
            f"[구조적 결측] {len(structural)}개 컬럼이 100% 비어 있으나 "
            f"{other} 쪽에는 값이 존재한다. {other} 전용 컬럼이므로 품질 문제가 아니다: "
            f"{', '.join(names[:8])}"
            + (f" 외 {len(names) - 8}개" if len(names) > 8 else "")
        )
    if empty:
        lines.append(
            f"[빈 컬럼] {len(empty)}개 컬럼이 양쪽 모두 값이 없다. 분석에서 제외할 것: "
            f"{', '.join(sorted(empty)[:8])}"
            + (f" 외 {len(empty) - 8}개" if len(empty) > 8 else "")
        )

    # ---------- 부분 결측 + 시간 의존 검사 ----------
    if real:
        top = sorted(real.items(), key=lambda x: -x[1]["rate"])[:5]
        lines.append("[부분 결측] " +
                     ", ".join(f"{c} {v['rate']}%" for c, v in top))

        time_dep = _check_time_dependent_missing(df, list(real))
        result["missing_time_dependent"] = time_dep
        for col, info in time_dep.items():
            lines.append(
                f"[시간 의존 결측] {col}은 {info['absent_years']}년에 100% 비어 있고 "
                f"{info['present_years']}년에는 값이 있다. "
                "해당 시기에 존재하지 않던 항목으로 보인다. "
                "전체 기간 집계에 그대로 넣으면 실제보다 적은 기간을 분석하게 된다."
            )

        biased = _check_biased_missing(df, list(real))
        result["missing_biased"] = biased

        # 같은 결측 패턴은 원인이 같으므로 묶어서 한 줄로 보고한다
        grouped = {}
        for col, info in biased.items():
            key = (info["missing_mean"], info["present_mean"])
            grouped.setdefault(key, []).append(col)

        for (m_mean, p_mean), cols in grouped.items():
            shown = ", ".join(sorted(cols)[:4])
            more = f" 외 {len(cols) - 4}개" if len(cols) > 4 else ""
            lines.append(
                f"[편향 결측] {len(cols)}개 컬럼({shown}{more})이 동일한 결측 패턴을 보인다. "
                f"결측 행의 평균 경기시간 {m_mean}분, 비결측 행 {p_mean}분. "
                "결측이 무작위가 아니라 짧게 끝난 경기에 몰려 있다. "
                "해당 시점 이전에 종료된 경기라 값이 없는 것이므로, "
                "결측 행을 제거하면 접전 경기만 남아 표본이 편향된다."
            )
    else:
        lines.append("[부분 결측] 없음")

    # ---------- 이진 컬럼 ----------
    binaries, events = [], []
    for col in df.columns:
        if col not in BINARY_HINT_COLS:
            continue
        s = df[col].dropna()
        if s.empty:
            continue
        if set(pd.unique(s)) <= {0, 1, 0.0, 1.0}:
            item = {"column": col, "mean": round(float(s.mean()), 3)}
            binaries.append(item)
            if col in EVENT_COLS:
                events.append(item)

    result["binary_columns"] = binaries
    if binaries:
        lines.append(
            f"[자료형 주의] 0/1 이진 컬럼 {len(binaries)}개가 숫자형으로 저장돼 있다. "
            "연속형 지표처럼 평균·표준편차를 계산하면 오해를 부른다."
        )
    if events:
        label = "선취 비율" if kind == "teams" else "관여율(선수 기준)"
        sample = ", ".join(f"{e['column']} {e['mean'] * 100:.1f}%" for e in events[:6])
        lines.append(f"[{label}] {sample}")

    # ---------- 클래스 균형 ----------
    if "result" in df.columns:
        wins = int(df["result"].sum())
        result["class_balance"] = {"win": wins, "loss": len(df) - wins}
        if kind == "teams":
            lines.append(
                f"[클래스 균형] {wins}승 {len(df) - wins}패. "
                "한 경기의 두 행이 승/패로 짝을 이루므로 50:50은 구조적 결과다. "
                "이를 '팀 전체 승률'로 해석하면 안 된다."
            )

    if kind == "teams":
        lines.append(
            "[표본 독립성] 같은 gameid의 두 행은 서로 독립이 아니다. "
            "한 팀이 이기면 상대는 반드시 진다. "
            "상관분석·가설검정 결과가 실제보다 부풀려질 수 있으므로 해석에 주의."
        )

    # ---------- 이상치 ----------
    if "gamelength_min" in df.columns:
        o = _iqr_outliers(df, "gamelength_min")
        result["outliers"] = o
        parts = [f"[이상치] gamelength_min 중앙값 {o['median']}분, "
                 f"정상범위 {o['low_bound']}~{o['high_bound']}분"]
        parts.append(
            f"하위 이상치 {o['low_count']}건(최단 {o['min']}분, 조기 항복 추정)"
            if o["low_count"] else f"하위 이상치 없음(최단 {o['min']}분)"
        )
        parts.append(
            f"상위 이상치 {o['high_count']}건(최장 {o['max']}분, 장기전)"
            if o["high_count"] else f"상위 이상치 없음(최장 {o['max']}분)"
        )
        lines.append(". ".join(parts) + ".")

    # ---------- 표본 불균형 ----------
    if "teamname" in df.columns:
        counts = df["teamname"].value_counts()
        result["team_samples"] = counts.to_dict()
        lines.append(
            f"[팀별 표본] 최다 {counts.index[0]} {counts.iloc[0]}행 / "
            f"최소 {counts.index[-1]} {counts.iloc[-1]}행 "
            f"(격차 {counts.iloc[0] - counts.iloc[-1]}행). "
            "팀별 비교 시 표본 수 차이를 감안해야 한다."
        )

    result["text"] = "\n".join(lines)
    return result


def _check_time_dependent_missing(df: pd.DataFrame, cols: list) -> dict:
    """특정 연도에만 결측인 컬럼을 찾는다 (해당 시기에 없던 항목)"""
    if "year" not in df.columns:
        return {}
    out = {}
    for col in cols:
        rate = df.groupby("year")[col].apply(lambda s: s.isna().mean())
        absent = [int(y) for y, r in rate.items() if r >= 0.999]
        present = [int(y) for y, r in rate.items() if r <= 0.5]
        if absent and present:
            out[col] = {"absent_years": absent, "present_years": present}
    return out


def _check_biased_missing(df: pd.DataFrame, cols: list,
                          ref: str = "gamelength_min") -> dict:
    """결측 여부가 다른 변수와 관련 있는지 검사 (무작위 결측인지 확인)"""
    if ref not in df.columns:
        return {}
    out = {}
    for col in cols:
        mask = df[col].isna()
        if mask.sum() < GRADE_WEAK or (~mask).sum() < GRADE_WEAK:
            continue
        m_mean = df.loc[mask, ref].mean()
        p_mean = df.loc[~mask, ref].mean()
        if pd.isna(m_mean) or pd.isna(p_mean):
            continue
        if abs(m_mean - p_mean) >= 2.0:   # 2분 이상 차이면 편향으로 본다
            out[col] = {
                "missing_mean": round(float(m_mean), 1),
                "present_mean": round(float(p_mean), 1),
            }
    return out


def _iqr_outliers(df: pd.DataFrame, col: str) -> dict:
    s = df[col].dropna()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return {
        "column": col,
        "low_bound": round(float(low), 1),
        "high_bound": round(float(high), 1),
        "low_count": int((s < low).sum()),
        "high_count": int((s > high).sum()),
        "min": round(float(s.min()), 1),
        "max": round(float(s.max()), 1),
        "median": round(float(s.median()), 1),
    }


def compare_groups(df: pd.DataFrame, by: str, metric: str,
                   year: int | None = None) -> dict:
    """기준 컬럼으로 나눈 그룹별 지표 비교"""
    for col in (by, metric):
        if col not in df.columns:
            return {"error": f"'{col}' 컬럼 없음",
                    "text": f"'{col}' 컬럼이 데이터에 없어 비교할 수 없습니다."}

    d = _filter_year(df, year)
    if d.empty:
        return {"error": "no data", "text": f"{year}년 데이터가 없습니다."}

    g = d.groupby(by)[metric].agg(["count", "mean", "median", "std"]).round(3)

    groups = []
    for key, row in g.iterrows():
        groups.append({
            "group": key,
            "n": int(row["count"]),
            "mean": float(row["mean"]),
            "median": float(row["median"]),
            "std": None if pd.isna(row["std"]) else float(row["std"]),
            "sample": grade_sample(int(row["count"])),
        })

    min_n = min((x["n"] for x in groups), default=0)
    out = {"by": by, "metric": metric, "year": year, "groups": groups,
           "overall_sample": grade_sample(min_n)}

    lines = [f"[그룹 비교] 기준={by} 지표={metric} 기간={_period(year)}"]
    for x in groups:
        lines.append(
            f"  {by}={x['group']}: n={x['n']:,}, 평균={x['mean']}, 중앙값={x['median']}"
        )
    if len(groups) == 2:
        diff = groups[1]["mean"] - groups[0]["mean"]
        out["diff"] = round(diff, 3)
        lines.append(f"  차이(뒤-앞): {diff:+.3f}")

    if by in EVENT_COLS and metric == "result":
        lines.append(
            f"  주의: {by}는 승패와 같은 경기 안에서 발생한 사건이다. "
            "선취한 팀이 이길 확률이 높다는 사실을 보여줄 뿐, "
            "선취가 승리를 만들었다는 근거는 되지 않는다. "
            "이미 유리했던 팀이 선취했을 가능성도 동일하게 성립한다."
        )

    lines.append(f"[표본 등급] 최소 그룹 {min_n:,}건 — {out['overall_sample']['grade']}")
    lines.append(f"[서술 지침] {out['overall_sample']['instruction']}")
    out["text"] = "\n".join(lines)
    return out


def classify_metric(col: str) -> str:
    if col in LEAKAGE_COLS:
        return "종료시점"
    if col in EARLY_COLS:
        return "15분시점"
    if col in EVENT_COLS:
        return "사건발생"
    return "기타"


def correlate(df: pd.DataFrame, metrics: list, target: str = "result",
              year: int | None = None) -> dict:
    """지표들과 target 간 상관계수. 결과 누수 지표를 분리해서 보고한다."""
    d = _filter_year(df, year)
    if d.empty:
        return {"error": "no data", "text": f"{year}년 데이터가 없습니다."}
    if target not in d.columns:
        return {"error": f"'{target}' 없음", "text": f"'{target}' 컬럼이 없습니다."}

    valid = [m for m in metrics if m in d.columns]
    missing = [m for m in metrics if m not in d.columns]

    rows = []
    for m in valid:
        sub = d[[m, target]].dropna()
        if len(sub) < GRADE_WEAK:
            rows.append({"metric": m, "corr": None, "n": len(sub),
                         "timing": classify_metric(m)})
            continue
        r = float(sub[m].corr(sub[target]))
        rows.append({
            "metric": m, "corr": round(r, 3), "abs": round(abs(r), 3),
            "n": len(sub), "strength": _corr_strength(abs(r)),
            "timing": classify_metric(m),
        })

    rows.sort(key=lambda x: -(x.get("abs") or 0))
    predictive = [r for r in rows if r["timing"] != "종료시점"]
    leaked = [r for r in rows if r["timing"] == "종료시점"]

    out = {"target": target, "year": year, "results": rows,
           "predictive": predictive, "leakage": leaked, "missing": missing}

    lines = [f"[상관분석] 대상={target} 기간={_period(year)}"]

    if predictive:
        lines.append("\n[예측 지표] 경기 도중 시점에 확정되는 값. 해석 가능.")
        lines += [_corr_line(r) for r in predictive]

    if leaked:
        lines.append("\n[결과 누수 지표] 경기 종료 시점 누적값. 예측 지표로 쓸 수 없음.")
        lines += [_corr_line(r) for r in leaked]
        top = leaked[0]
        if top["corr"] is not None:
            lines.append(
                f"  경고: {top['metric']}의 상관 {top['corr']:+.3f}은 "
                "승리의 원인이 아니라 승리를 다르게 센 값이다. "
                "타워를 모두 부수면 그 자체가 승리이기 때문이다. "
                "이런 지표로 '무엇이 승리를 만드는가'를 설명하면 동어반복이 된다."
            )

    if missing:
        lines.append(f"\n[누락] 데이터에 없는 컬럼: {', '.join(missing)}")

    lines.append(
        "\n[해석 지침] 상관계수는 인과관계를 뜻하지 않는다. "
        "또한 같은 경기의 두 팀 행이 짝을 이루어 서로 독립이 아니므로, "
        "상관의 크기가 실제보다 부풀려져 있을 수 있다. "
        "이 두 한계를 반드시 함께 서술할 것."
    )
    out["text"] = "\n".join(lines)
    return out


def _corr_line(r: dict) -> str:
    if r["corr"] is None:
        return f"  {r['metric']}: 표본 부족(n={r['n']})"
    return (f"  {r['metric']}: r={r['corr']:+.3f} "
            f"({r['strength']}, n={r['n']:,}, {r['timing']})")


def _corr_strength(a: float) -> str:
    if a >= 0.7:
        return "매우 강함"
    if a >= 0.5:
        return "강함"
    if a >= 0.3:
        return "보통"
    if a >= 0.1:
        return "약함"
    return "거의 없음"


# ===============================================================
# 2단계
# ===============================================================

PLAYER_METRICS = [
    "kills", "deaths", "assists", "dpm", "damageshare",
    "earnedgoldshare", "cspm", "visionscore", "vspm",
    "golddiffat15", "xpdiffat15", "csdiffat15",
]


def player_stats(players: pd.DataFrame, player: str | None = None,
                 position: str | None = None, year: int | None = None,
                 top: int = 10) -> dict:
    """선수 단위 지표

    player 지정: 해당 선수 상세
    position 지정: 해당 포지션 선수들 비교
    """
    d = _filter_year(players, year)
    if d.empty:
        return {"error": "no data", "text": f"{year}년 데이터가 없습니다."}

    # ---------- 개별 선수 ----------
    if player:
        name = resolve_player(player, d)
        if not name:
            return {"error": "not found",
                    "text": f"'{player}' 선수를 찾을 수 없습니다. 표기를 확인해 주세요."}

        sub = d[d["playername"] == name]
        n = len(sub)
        wins = int(sub["result"].sum())
        sample = grade_sample(n)

        pos = sub["position"].mode()
        pos = pos.iloc[0] if not pos.empty else "?"
        teams_played = sorted(sub["teamname"].dropna().unique())

        metrics = {}
        for m in PLAYER_METRICS:
            if m in sub.columns and sub[m].notna().any():
                metrics[m] = round(float(sub[m].mean()), 3)

        # 같은 포지션 평균과 비교
        peer = d[d["position"] == pos]
        comparison = {}
        for m, v in metrics.items():
            if m in peer.columns and peer[m].notna().any():
                pm = float(peer[m].mean())
                comparison[m] = {"player": v, "position_avg": round(pm, 3),
                                 "diff": round(v - pm, 3)}

        out = {"player": name, "position": pos, "teams": teams_played,
               "year": year, "games": n, "wins": wins,
               "winrate": round(wins / n * 100, 1) if n else None,
               "metrics": metrics, "vs_position_avg": comparison,
               "sample": sample}

        lines = [
            f"[선수 전체 지표] {name} ({pos}) 기간={_period(year)}",
            "  범위: 이 선수가 뛴 모든 경기. 챔피언 구분 없음.",
            "  * 이 수치를 특정 챔피언의 성적으로 제시하면 안 된다. "
            "챔피언별 전적은 별도 조회가 필요하다.",
            f"  소속: {', '.join(teams_played)}",
            f"  {n}경기 {wins}승 {n - wins}패 (승률 {out['winrate']}%)",
        ]
        lines.append("  * 선수 승률은 소속 팀 전력에 크게 좌우되므로 개인 기량 지표가 아니다.")
        lines.append(f"  [같은 포지션 평균 대비] (표본 {len(peer):,}행)")
        has_diff = False
        for m, c in comparison.items():
            sign = "+" if c["diff"] >= 0 else ""
            lines.append(
                f"    {m}: {c['player']} (평균 {c['position_avg']}, {sign}{c['diff']})"
            )
            if m.endswith("diffat15") or m.endswith("diffat25"):
                has_diff = True
        if has_diff:
            lines.append(
                "    * diff 계열(golddiffat15 등)의 전체 평균은 구조적으로 0이다. "
                "한 팀이 +면 상대 팀이 정확히 같은 크기로 -가 되기 때문이다. "
                "따라서 이 값은 '평균보다 잘했다'가 아니라 "
                "'상대 라이너보다 얼마나 앞섰는가'로 읽어야 한다."
            )
        lines.append(f"[표본 등급] {n}경기 — {sample['grade']}")
        lines.append(f"[서술 지침] {sample['instruction']}")
        out["text"] = "\n".join(lines)
        return out

    # ---------- 포지션별 비교 ----------
    if position:
        sub = d[d["position"].str.lower() == position.lower()]
        if sub.empty:
            valid = sorted(d["position"].dropna().unique())
            return {"error": "not found",
                    "text": f"'{position}' 포지션이 없습니다. 가능한 값: {valid}"}

        g = sub.groupby("playername").agg(
            games=("result", "count"), wins=("result", "sum"),
            dpm=("dpm", "mean"), damageshare=("damageshare", "mean"),
            golddiffat15=("golddiffat15", "mean"),
            visionscore=("visionscore", "mean"),
        ).round(3)
        g = g[g["games"] >= GRADE_REF].sort_values("games", ascending=False)

        rows = []
        for name, r in g.head(top).iterrows():
            rows.append({
                "player": name, "games": int(r["games"]), "wins": int(r["wins"]),
                "winrate": round(r["wins"] / r["games"] * 100, 1),
                "dpm": r["dpm"], "damageshare": r["damageshare"],
                "golddiffat15": r["golddiffat15"],
            })

        out = {"position": position, "year": year, "players": rows,
               "filtered_min_games": GRADE_REF}
        lines = [f"[포지션 비교] {position} 기간={_period(year)} "
                 f"({GRADE_REF}경기 이상만, 출전 순 {len(rows)}명)",
                 "  범위: 해당 포지션 선수들의 전체 경기. 챔피언 구분 없음."]
        for r in rows:
            lines.append(
                f"  {r['player']:<12} {r['games']:>3}경기 승률 {r['winrate']:>5}% "
                f"dpm {r['dpm']:>7} 딜지분 {r['damageshare']} "
                f"15분골드차 {r['golddiffat15']:+.0f}"
            )
        lines.append(
            "  * 표본 확보를 위해 10경기 미만 선수는 제외했다. "
            "제외된 선수가 있을 수 있으므로 '전체 순위'로 읽으면 안 된다."
        )
        lines.append(
            "  * 승률은 소속 팀 전력에 좌우되므로 선수 기량 순위가 아니다. "
            "강팀 소속이면 개인 지표가 평범해도 승률이 높게 나온다. "
            "개인 비교에는 dpm, 딜지분, 15분 격차처럼 팀 성적과 덜 얽힌 지표를 쓸 것."
        )
        out["text"] = "\n".join(lines)
        return out

    return {"error": "no target",
            "text": "선수명 또는 포지션 중 하나를 지정해야 합니다."}


def player_champion(players: pd.DataFrame, player: str,
                    champion: str | None = None,
                    year: int | None = None) -> dict:
    """선수 x 챔피언 전적. 표본이 작아 변동폭을 함께 제시한다."""
    d = _filter_year(players, year)
    name = resolve_player(player, d)
    if not name:
        return {"error": "not found",
                "text": f"'{player}' 선수를 찾을 수 없습니다."}

    sub = d[d["playername"] == name]

    # ---------- 특정 챔피언 ----------
    if champion:
        key = _norm(champion)
        champs = sub["champion"].dropna().unique()
        matched = [c for c in champs if _norm(c) == key]
        if not matched:
            matched = [c for c in champs if key in _norm(c)]
        if not matched:
            return {"error": "not found",
                    "text": f"{name} 선수가 '{champion}'을 플레이한 기록이 없습니다."}

        c = matched[0]
        cs = sub[sub["champion"] == c]
        n, wins = len(cs), int(cs["result"].sum())
        sample = grade_sample(n)

        out = {"player": name, "champion": c, "year": year,
               "games": n, "wins": wins, "losses": n - wins,
               "winrate": round(wins / n * 100, 1) if n else None,
               "swing": swing_range(wins, n), "sample": sample}

        lines = [f"[선수 x 챔피언] {name} — {c} 기간={_period(year)}",
                 f"  {n}경기 {wins}승 {n - wins}패"]
        if n >= GRADE_WEAK:
            lines.append(f"  승률 {out['winrate']}%")
            lines.append(f"  변동폭: {out['swing']}")
        else:
            lines.append("  표본이 매우 적어 승률을 산출하지 않는다.")

        for m in ["kills", "deaths", "assists", "dpm", "damageshare"]:
            if m in cs.columns and cs[m].notna().any():
                out.setdefault("metrics", {})[m] = round(float(cs[m].mean()), 2)
        if out.get("metrics"):
            lines.append("  " + ", ".join(f"{k} {v}" for k, v in out["metrics"].items()))

        lines.append(f"[표본 등급] {n}경기 — {sample['grade']}")
        lines.append(f"[서술 지침] {sample['instruction']}")
        out["text"] = "\n".join(lines)
        return out

    # ---------- 챔피언 목록 ----------
    g = sub.groupby("champion")["result"].agg(["count", "sum"])
    g = g.sort_values("count", ascending=False)

    rows = []
    for c, r in g.iterrows():
        n, w = int(r["count"]), int(r["sum"])
        rows.append({
            "champion": c, "games": n, "wins": w,
            "winrate": round(w / n * 100, 1) if n >= GRADE_WEAK else None,
            "grade": grade_sample(n)["grade"],
            "swing": swing_range(w, n) if n >= GRADE_WEAK else None,
        })

    usable = [r for r in rows if r["games"] >= GRADE_WEAK]
    out = {"player": name, "year": year, "champions": rows,
           "total_champions": len(rows), "usable": len(usable)}

    lines = [f"[선수 챔피언 목록] {name} 기간={_period(year)}",
             f"  총 {len(rows)}종 사용, 그 중 {GRADE_WEAK}경기 이상은 {len(usable)}종"]
    for r in rows[:15]:
        if r["winrate"] is None:
            lines.append(f"  {r['champion']:<12} {r['games']:>2}경기 {r['wins']}승 "
                         f"— 표본 부족, 승률 산출 안 함")
        elif not r["grade"].startswith("표본충분"):
            lines.append(f"  {r['champion']:<12} {r['games']:>2}경기 {r['wins']}승 "
                         f"승률 {r['winrate']:>5}% ({r['grade']}) — {r['swing']}")
        else:
            lines.append(f"  {r['champion']:<12} {r['games']:>2}경기 {r['wins']}승 "
                         f"승률 {r['winrate']:>5}% ({r['grade']})")
    lines.append(
        f"  * {GRADE_WEAK}경기 미만은 1경기 결과로 승률이 크게 흔들려 승률을 계산하지 않았다."
    )
    out["text"] = "\n".join(lines)
    return out


H2H_STYLE_METRICS = ["golddiffat15", "xpdiffat15", "csdiffat15",
                     "firstblood", "firstdragon", "firsttower", "gamelength_min"]


def head_to_head(teams: pd.DataFrame, team_a: str, team_b: str,
                 year: int | None = None) -> dict:
    """두 팀의 상대전적 + 경기 양상

    연도를 지정해도 전체 기간 누적을 함께 반환한다.
    한 시즌만 보면 표본이 8경기 안팎이라 우열을 논하기 어렵기 때문이다.
    """
    names = list(teams["teamname"].dropna().unique())
    a = resolve_team(team_a, names)
    b = resolve_team(team_b, names)
    if not a or not b:
        bad = team_a if not a else team_b
        return {"error": "not found",
                "text": f"'{bad}' 팀을 찾을 수 없습니다. 가능한 팀: {sorted(names)}"}
    if a == b:
        return {"error": "same team", "text": "서로 다른 두 팀을 지정해 주세요."}

    # 두 팀이 맞붙은 gameid 찾기
    ga = set(teams.loc[teams["teamname"] == a, "gameid"])
    gb = set(teams.loc[teams["teamname"] == b, "gameid"])
    common = ga & gb
    if not common:
        return {"error": "no match",
                "text": f"{a}와 {b}의 맞대결 기록이 없습니다."}

    h2h = teams[teams["gameid"].isin(common)]
    rows_a = h2h[h2h["teamname"] == a]

    def tally(df_a: pd.DataFrame) -> dict:
        n = len(df_a)
        w = int(df_a["result"].sum())
        return {"games": n, "a_wins": w, "b_wins": n - w}

    total = tally(rows_a)
    by_year = {}
    for y, sub in rows_a.groupby("year"):
        by_year[int(y)] = tally(sub)

    out = {"team_a": a, "team_b": b, "year": year,
           "total": total, "by_year": by_year,
           "sample_total": grade_sample(total["games"])}

    lines = [f"[상대전적] {a} vs {b}"]

    # 지정 연도
    if year is not None:
        cur = by_year.get(year)
        if not cur:
            lines.append(f"  {year}년 맞대결 기록 없음")
        else:
            s = grade_sample(cur["games"])
            out["current"] = cur
            out["sample_current"] = s
            lead = a if cur["a_wins"] > cur["b_wins"] else (
                b if cur["b_wins"] > cur["a_wins"] else None)
            lines.append(
                f"  [{year}] {a} {cur['a_wins']}승 {cur['b_wins']}패 "
                f"({cur['games']}경기)"
            )
            if lead:
                lines.append(f"    전적상 {lead}가 앞선다.")
            else:
                lines.append("    전적상 동률이다.")
            lines.append(f"    표본 {cur['games']}경기 — {s['grade']}. {s['instruction']}")

    # 누적
    lines.append(
        f"  [3년 누적] {a} {total['a_wins']}승 {total['b_wins']}패 "
        f"({total['games']}경기)"
    )
    lines.append("  [연도별] " + " / ".join(
        f"{y} {v['a_wins']}-{v['b_wins']}" for y, v in sorted(by_year.items())
    ))

    # 경기 양상
    scope = rows_a if year is None else rows_a[rows_a["year"] == year]
    if not scope.empty:
        style = {}
        for m in H2H_STYLE_METRICS:
            if m in scope.columns and scope[m].notna().any():
                style[m] = round(float(scope[m].mean()), 3)
        out["style"] = style
        lines.append(f"  [경기 양상 - {_period(year)}, {a} 기준]")
        for m, v in style.items():
            if m in EVENT_COLS:
                lines.append(f"    {m} 선취율 {v * 100:.1f}%")
            elif m == "gamelength_min":
                lines.append(f"    평균 경기시간 {v:.1f}분")
            else:
                lines.append(f"    {m} 평균 {v:+.0f}")
        lines.append(
            "    * 승패는 경기 수만큼의 표본이지만 골드·경험치 격차는 "
            "각 경기가 연속값을 제공하므로 표본이 적을 때 더 안정적인 지표다."
        )

    lines.append(
        "  * 3년 누적 전적은 그 사이 로스터가 여러 차례 바뀐 결과다. "
        "현재 두 팀의 전력 차이로 해석하면 안 된다."
    )
    out["text"] = "\n".join(lines)
    return out


def _detect_pick_rule(d: pd.DataFrame, fp_col: str) -> dict:
    """진영과 선픽이 붙어 있는지(구 규칙) 분리돼 있는지(신 규칙) 판정

    2026 Cup부터 팀이 진영과 선픽 중 하나를 고르게 규칙이 바뀌었다.
    규칙 시행 시점을 코드에 박아두지 않고 데이터로 판정한다.
    나중에 규칙이 또 바뀌어도 데이터가 알려주게 하기 위해서다.
    """
    sub = d[["side", fp_col]].dropna()
    if sub.empty:
        return {"rule": "unknown", "overlap": None}
    ct = pd.crosstab(sub["side"], sub[fp_col])
    overlap = float((ct.max(axis=1) / ct.sum(axis=1)).min())
    return {
        "rule": "결합" if overlap >= 0.99 else "분리",
        "overlap": round(overlap, 4),
        "crosstab": ct,
    }


def _resolve_firstpick_meaning(teams: pd.DataFrame, fp_col: str) -> dict:
    """firstPick=1이 선픽인지 후픽인지 판정

    구 규칙 시즌에는 블루 진영이 항상 먼저 픽했다.
    그 시즌에서 블루와 붙어 있는 값이 곧 '선픽'이다.
    """
    for y in sorted(teams["year"].dropna().unique()):
        sub = teams[teams["year"] == y]
        info = _detect_pick_rule(sub, fp_col)
        if info["rule"] != "결합":
            continue
        ct = info["crosstab"]
        if "Blue" not in ct.index:
            continue
        first_val = int(ct.loc["Blue"].idxmax())
        return {"first_value": first_val, "basis_year": int(y),
                "confident": True}
    return {"first_value": 1, "basis_year": None, "confident": False}


def _combo_table(d: pd.DataFrame, fp_col: str, first_val: int) -> dict:
    """진영 x 선픽 2x2 조합별 승률"""
    out = {}
    for (side, fp), g in d.groupby(["side", fp_col]):
        n, w = len(g), int(g["result"].sum())
        label = "선픽" if int(fp) == first_val else "후픽"
        out[(side, label)] = {
            "games": n, "wins": w,
            "winrate": round(w / n * 100, 1) if n else None,
            "sample": grade_sample(n),
        }
    return out


def _team_strength(teams: pd.DataFrame) -> pd.Series:
    """팀-연도별 시즌 승률. 선택 편향 확인에 쓴다."""
    g = teams.groupby(["year", "teamname"])["result"].agg(["count", "sum"])
    return (g["sum"] / g["count"] * 100).round(1)


def draft_order(teams: pd.DataFrame, year: int | None = None) -> dict:
    """진영 x 픽순서 조합별 승률

    2026 Cup부터 선택권을 가진 팀이 {1픽, 2픽, 블루, 레드} 중 하나를 고르고
    나머지를 상대가 정하는 방식으로 바뀌었다.
    데이터에는 최종 조합만 남고 누가 무엇을 선택했는지는 기록되지 않는다.
    따라서 조합별 승률은 사실이지만, 팀의 선호나 판단의 근거로 쓸 수 없다.
    """
    d = _filter_year(teams, year)
    if d.empty:
        return {"error": "no data", "text": f"{year}년 데이터가 없습니다."}

    fp_col = next((c for c in d.columns if c.lower() == "firstpick"), None)
    out = {"year": year}
    lines = [f"[진영 x 픽순서] 기간={_period(year)}"]

    # ---------- 진영별 ----------
    if "side" in d.columns:
        g = d.groupby("side")["result"].agg(["count", "sum"])
        sides = {}
        for s, r in g.iterrows():
            n, w = int(r["count"]), int(r["sum"])
            sides[s] = {"games": n, "wins": w, "winrate": round(w / n * 100, 1)}
            lines.append(f"  {s}: {n:,}경기 {w}승 {n - w}패, 승률 {sides[s]['winrate']}%")
        out["by_side"] = sides

    if not fp_col or "side" not in d.columns:
        lines.append("  픽순서 컬럼이 없어 조합별 분석은 하지 않는다.")
        out["text"] = "\n".join(lines)
        return out

    # ---------- firstPick 의미 판정 ----------
    meaning = _resolve_firstpick_meaning(teams, fp_col)
    first_val = meaning["first_value"]
    out["firstpick_meaning"] = meaning
    if meaning["confident"]:
        lines.append(
            f"  [컬럼 해석] {meaning['basis_year']}년에는 블루가 항상 먼저 픽했고 "
            f"그 해 블루와 일치한 값이 {fp_col}={first_val}이므로, "
            f"{fp_col}={first_val}을 선픽으로 본다."
        )

    # ---------- 규칙 판정 ----------
    rule_by_year = {}
    for y in sorted(d["year"].dropna().unique()):
        info = _detect_pick_rule(d[d["year"] == y], fp_col)
        rule_by_year[int(y)] = {"rule": info["rule"], "overlap": info["overlap"]}
    out["rule_by_year"] = rule_by_year

    joined = [y for y, i in rule_by_year.items() if i["rule"] == "결합"]
    split_y = [y for y, i in rule_by_year.items() if i["rule"] == "분리"]

    lines.append("  [진영-픽순서 결합 여부]")
    for y, i in rule_by_year.items():
        tag = "결합" if i["rule"] == "결합" else "분리"
        lines.append(f"    {y}: {tag} (일치율 {i['overlap']:.1%})")
    if joined:
        lines.append(
            f"    {joined}년은 블루가 곧 선픽이라 조합이 두 가지뿐이다. "
            "진영 효과와 선픽 효과를 분리할 수 없다."
        )
    if joined and split_y:
        lines.append(
            f"    {joined}년과 {split_y}년은 드래프트 규칙이 다르므로 "
            "합쳐서 집계하면 서로 다른 제도의 결과가 섞인다."
        )

    # ---------- 4조합 승률 ----------
    target = split_y if split_y else sorted(rule_by_year)
    dd = d[d["year"].isin(target)]
    combo = _combo_table(dd, fp_col, first_val)
    out["combo"] = {f"{k[0]}+{k[1]}": v for k, v in combo.items()}

    order = [("Blue", "선픽"), ("Blue", "후픽"), ("Red", "선픽"), ("Red", "후픽")]
    lines.append(f"  [조합별 승률] 대상 {target}년, {len(dd) // 2}경기")
    for key in order:
        v = combo.get(key)
        if not v:
            lines.append(f"    {key[0]}+{key[1]}: 해당 조합 없음")
            continue
        lines.append(
            f"    {key[0]}+{key[1]}: {v['games']:,}경기 "
            f"{v['wins']}승 {v['games'] - v['wins']}패, 승률 {v['winrate']}% "
            f"({v['sample']['grade']})"
        )

    def wr(side, label):
        v = combo.get((side, label))
        return v["winrate"] if v and v["winrate"] is not None else None

    bf, ba = wr("Blue", "선픽"), wr("Blue", "후픽")
    rf, ra = wr("Red", "선픽"), wr("Red", "후픽")

    effects = {}
    if bf is not None and rf is not None:
        effects["진영 차이 (선픽 조건에서)"] = round(bf - rf, 1)
    if ba is not None and ra is not None:
        effects["진영 차이 (후픽 조건에서)"] = round(ba - ra, 1)
    if bf is not None and ba is not None:
        effects["픽순서 차이 (블루 조건에서)"] = round(bf - ba, 1)
    if rf is not None and ra is not None:
        effects["픽순서 차이 (레드 조건에서)"] = round(rf - ra, 1)
    if effects:
        out["effects"] = effects
        lines.append("  [차이 요약] 한쪽 조건을 고정했을 때의 승률 차이(%p)")
        for k, v in effects.items():
            lines.append(f"    {k}: {v:+.1f}%p")

    if len(split_y) > 0:
        lines.append(
            "    * 대각선 두 칸의 승률 합은 100%다. "
            "블루+선픽 팀의 상대는 반드시 레드+후픽이기 때문이다. "
            "네 칸으로 보이지만 독립된 값은 두 개뿐이다."
        )

    # ---------- split별 ----------
    if "split" in dd.columns and dd["split"].nunique() > 1:
        lines.append("  [split별 조합 분포]")
        sp_out = {}
        for sp, g in dd.groupby("split"):
            c = _combo_table(g, fp_col, first_val)
            parts = []
            for key in order:
                v = c.get(key)
                if v:
                    parts.append(f"{key[0]}+{key[1]} {v['games']}건({v['winrate']}%)")
            sp_out[str(sp)] = {f"{k[0]}+{k[1]}": v for k, v in c.items()}
            lines.append(f"    {sp}: " + ", ".join(parts))
        out["by_split"] = sp_out

    # ---------- 해석 제약 ----------
    lines.append(
        "  [해석 제약] 선택권을 가진 팀이 1픽·2픽·블루·레드 중 하나를 고르고 "
        "나머지를 상대가 정한다. 데이터에는 최종 조합만 남고 "
        "누가 어떤 선택을 했는지, 애초에 누가 선택권을 가졌는지는 기록되지 않는다. "
        "따라서 위 승률로 '팀들이 블루를 선호한다', '블루를 고르는 것이 이득이다' 같은 "
        "선택에 관한 서술을 해서는 안 된다. "
        "조합별 승률이 이러했다는 사실까지만 말할 수 있다."
    )

    # ---------- 선택 편향 ----------
    strength = _team_strength(teams)
    rows = []
    for (side, fp), g in dd.groupby(["side", fp_col]):
        label = "선픽" if int(fp) == first_val else "후픽"
        vals = [strength.get((int(y), t)) for y, t in zip(g["year"], g["teamname"])]
        vals = [v for v in vals if v is not None and not pd.isna(v)]
        if vals:
            rows.append({"combo": f"{side}+{label}",
                         "avg_team_winrate": round(sum(vals) / len(vals), 1),
                         "n": len(vals)})
    if rows:
        out["selection_bias"] = rows
        lines.append("  [조합별 팀 전력] 각 조합에 앉은 팀들의 그해 시즌 승률 평균")
        for r in sorted(rows, key=lambda x: -x["avg_team_winrate"]):
            lines.append(f"    {r['combo']}: {r['avg_team_winrate']}% (n={r['n']:,})")
        spread = max(r["avg_team_winrate"] for r in rows) - \
            min(r["avg_team_winrate"] for r in rows)
        if spread >= 3.0:
            lines.append(
                f"    조합별 팀 전력이 최대 {spread:.1f}%p 차이 난다. "
                "조합별 승률에는 조합 자체의 영향과 그 자리에 앉은 팀의 전력이 섞여 있다."
            )

    out["text"] = "\n".join(lines)
    return out


def team_ranking(teams: pd.DataFrame, year: int | None = None) -> dict:
    """팀별 성적 순위

    '가장 잘하는 팀'은 평가가 섞인 표현이므로,
    데이터가 답할 수 있는 범위(승률과 경기 지표)로만 제시한다.
    """
    d = _filter_year(teams, year)
    if d.empty:
        return {"error": "no data", "text": f"{year}년 데이터가 없습니다."}

    agg = {"games": ("result", "count"), "wins": ("result", "sum")}
    for m in ["golddiffat15", "xpdiffat15", "gamelength_min",
              "firstdragon", "firsttower", "firstblood"]:
        if m in d.columns:
            agg[m] = (m, "mean")

    g = d.groupby("teamname").agg(**agg)
    g["losses"] = g["games"] - g["wins"]
    g["winrate"] = (g["wins"] / g["games"] * 100).round(1)
    g = g.sort_values("winrate", ascending=False)

    rows = []
    for name, r in g.iterrows():
        row = {
            "team": name, "games": int(r["games"]),
            "wins": int(r["wins"]), "losses": int(r["losses"]),
            "winrate": float(r["winrate"]),
            "sample": grade_sample(int(r["games"])),
        }
        for m in ["golddiffat15", "xpdiffat15", "gamelength_min",
                  "firstdragon", "firsttower", "firstblood"]:
            if m in g.columns:
                row[m] = round(float(r[m]), 3)
        rows.append(row)

    out = {"year": year, "teams": rows, "n_teams": len(rows)}

    lines = [f"[팀 순위] 기간={_period(year)} ({len(rows)}팀)"]
    for i, r in enumerate(rows, 1):
        extra = ""
        if "golddiffat15" in r:
            extra = f"  15분골드차 {r['golddiffat15']:+.0f}"
        if "firsttower" in r:
            extra += f"  첫타워선취 {r['firsttower'] * 100:.0f}%"
        lines.append(
            f"  {i}. {r['team']:<20} {r['games']:>3}경기 "
            f"{r['wins']:>3}승 {r['losses']:>3}패  승률 {r['winrate']:>5.1f}%{extra}"
        )

    gap = rows[0]["winrate"] - rows[-1]["winrate"]
    span = max(r["games"] for r in rows) - min(r["games"] for r in rows)
    out["winrate_gap"] = round(gap, 1)
    out["games_gap"] = span

    lines.append(
        f"  [범위] 1위 {rows[0]['team']} {rows[0]['winrate']}% ~ "
        f"최하위 {rows[-1]['team']} {rows[-1]['winrate']}% (격차 {gap:.1f}%p)"
    )
    lines.append(
        f"  [표본 불균형] 팀별 경기 수가 최대 {span}경기 차이 난다. "
        "플레이오프 진출 팀이 더 많은 경기를 치르기 때문이며, "
        "이 자체가 성적을 반영한다."
    )
    lines.append(
        "  [해석 지침] 승률은 그 기간의 성적일 뿐 '가장 잘하는 팀'이라는 평가가 아니다. "
        "상대 전력, 리그 포맷, 로스터 변경이 모두 승률에 섞여 있다. "
        "여러 해를 합쳐 볼 경우 각 시즌의 팀 구성이 다르다는 점도 함께 밝힐 것. "
        "순위를 단정적인 실력 서열로 서술하지 말 것."
    )

    out["text"] = "\n".join(lines)
    return out


def _starters(players: pd.DataFrame, team: str, year: int) -> dict:
    """해당 연도 팀의 포지션별 주전(최다 출전) 선수"""
    sub = players[(players["teamname"] == team) & (players["year"] == year)]
    out = {}
    for pos, g in sub.groupby("position"):
        counts = g["playername"].value_counts()
        if not counts.empty:
            out[pos] = {"player": counts.index[0], "games": int(counts.iloc[0])}
    return out


def team_trend(teams: pd.DataFrame, players: pd.DataFrame, team: str,
               year: int | None = None) -> dict:
    """팀의 연도별 성적 + 로스터 연속성

    같은 팀명이라도 선수가 바뀌면 사실상 다른 팀이다.
    성적 변화를 '성장'으로 서술하지 않도록 로스터 유지율을 함께 계산한다.
    """
    names = list(teams["teamname"].dropna().unique())
    t = resolve_team(team, names)
    if not t:
        return {"error": "not found",
                "text": f"'{team}' 팀을 찾을 수 없습니다. 가능한 팀: {sorted(names)}"}

    sub = teams[teams["teamname"] == t]
    if sub.empty:
        return {"error": "no data", "text": f"{t}의 경기 기록이 없습니다."}

    # 연도가 지정되면 그 해만 본다.
    # 지정 없이 물었을 때만 3년 추이와 로스터 연속성을 함께 보여준다.
    if year is not None:
        one = sub[sub["year"] == year]
        if one.empty:
            avail = sorted(int(y) for y in sub["year"].dropna().unique())
            return {"error": "no data",
                    "text": f"{t}의 {year}년 기록이 없습니다. "
                            f"데이터에 있는 연도: {avail}"}
        n, w = len(one), int(one["result"].sum())
        res = {"team": t, "year": year, "games": n, "wins": w,
               "losses": n - w, "winrate": round(w / n * 100, 1)}
        lines = [f"[팀 성적] {t} {year}년",
                 f"  {n}경기 {w}승 {n - w}패 (승률 {res['winrate']}%)",
                 "  * 이 수치는 이미 계산된 값이다. 다시 계산하지 말 것.",
                 f"  * 요청한 기간은 {year}년이다. 다른 연도 수치를 섞어서 답하지 말 것."]
        for m in ["golddiffat15", "xpdiffat15", "gamelength_min",
                  "firstdragon", "firsttower", "firstblood"]:
            if m in one.columns and one[m].notna().any():
                v = float(one[m].mean())
                res[m] = round(v, 3)
                if m in EVENT_COLS:
                    lines.append(f"  {m} 선취율 {v * 100:.1f}%")
                elif m == "gamelength_min":
                    lines.append(f"  평균 경기시간 {v:.1f}분")
                else:
                    lines.append(f"  {m} 평균 {v:+.0f}")

        rs = _starters(players, t, year)
        if rs:
            res["roster"] = rs
            lines.append("  주전: " + ", ".join(
                f"{p}({v['player']})" for p, v in sorted(rs.items())))

        splits = one["split"].dropna().unique() if "split" in one.columns else []
        if len(splits):
            lines.append(f"  포함 구간: {sorted(splits)}")
            lines.append("  * 시즌이 진행 중이면 최종 성적이 아니다.")

        res["sample"] = grade_sample(n)
        lines.append(f"[표본 등급] {n}경기 — {res['sample']['grade']}")
        res["text"] = "\n".join(lines)
        return res

    trend = {}
    for y, g in sub.groupby("year"):
        n, w = len(g), int(g["result"].sum())
        row = {"games": n, "wins": w, "losses": n - w,
               "winrate": round(w / n * 100, 1)}
        for m in ["golddiffat15", "gamelength_min", "firstdragon", "firsttower"]:
            if m in g.columns and g[m].notna().any():
                row[m] = round(float(g[m].mean()), 3)
        trend[int(y)] = row

    years = sorted(trend)
    rosters = {y: _starters(players, t, y) for y in years}

    continuity = {}
    for prev, cur in zip(years, years[1:]):
        p = {v["player"] for v in rosters[prev].values()}
        c = {v["player"] for v in rosters[cur].values()}
        kept = p & c
        continuity[f"{prev}->{cur}"] = {
            "kept": sorted(kept), "kept_n": len(kept),
            "total": len(c),
            "rate": round(len(kept) / len(c) * 100, 1) if c else None,
        }

    # 처음과 마지막 비교
    span = None
    if len(years) >= 2:
        first, last = years[0], years[-1]
        p = {v["player"] for v in rosters[first].values()}
        c = {v["player"] for v in rosters[last].values()}
        kept = p & c
        span = {"from": first, "to": last, "kept": sorted(kept),
                "kept_n": len(kept), "total": len(c),
                "rate": round(len(kept) / len(c) * 100, 1) if c else None}

    out = {"team": t, "trend": trend, "rosters": rosters,
           "continuity": continuity, "span": span}

    total_g = int(sub.shape[0])
    total_w = int(sub["result"].sum())
    out["total"] = {"games": total_g, "wins": total_w,
                    "losses": total_g - total_w,
                    "winrate": round(total_w / total_g * 100, 1) if total_g else None}

    lines = [f"[팀 추이] {t}"]
    lines.append(
        f"  [전체 합계] {total_g}경기 {total_w}승 {total_g - total_w}패 "
        f"(승률 {out['total']['winrate']}%) — 이 합계는 이미 계산된 값이다. "
        "연도별 수치를 더해서 다시 계산하지 말 것."
    )
    for y in years:
        r = trend[y]
        extra = ""
        if "golddiffat15" in r:
            extra = f"  15분골드차 {r['golddiffat15']:+.0f}"
        lines.append(
            f"  {y}: {r['games']}경기 {r['wins']}승 {r['losses']}패 "
            f"(승률 {r['winrate']}%){extra}"
        )

    lines.append("  [주전 로스터]")
    for y in years:
        rs = rosters[y]
        if rs:
            names_str = ", ".join(
                f"{p}({v['player']})" for p, v in sorted(rs.items())
            )
            lines.append(f"    {y}: {names_str}")

    if continuity:
        lines.append("  [로스터 연속성]")
        for k, v in continuity.items():
            kept_str = ", ".join(v["kept"]) if v["kept"] else "없음"
            lines.append(f"    {k}: {v['kept_n']}/{v['total']}명 유지 "
                         f"({v['rate']}%) — {kept_str}")

    if span:
        lines.append(
            f"  [{span['from']} -> {span['to']}] 주전 {span['total']}명 중 "
            f"{span['kept_n']}명 유지 ({span['rate']}%)"
        )
        if span["rate"] is not None and span["rate"] < 50:
            lines.append(
                "    * 로스터의 절반 이상이 교체됐다. 이 기간의 성적 변화를 "
                "'기존 선수단의 성장'으로 서술하면 안 된다. "
                "선수 구성 변경의 결과로 보는 편이 타당하다."
            )
        else:
            lines.append(
                "    * 주전 상당수가 유지됐으므로 성적 변화를 팀 차원의 변화로 "
                "볼 여지가 있다. 다만 상대 팀 전력 변화도 함께 작용한다."
            )

    lines.append(
        "  * 팀명은 데이터상 현재 기준으로 통일돼 있어, 과거 시즌의 실제 팀명과 "
        "다를 수 있다. 승률은 그해 리그 포맷과 상대 전력에 좌우된다."
    )
    out["text"] = "\n".join(lines)
    return out


# ===============================================================
# 3단계
# ===============================================================

def find_outlier_champs(champs: pd.DataFrame, players: pd.DataFrame,
                        teams: pd.DataFrame, year: int | None = None,
                        min_games: int = GRADE_REF, top: int = 8) -> dict:
    """저평가 / 과대평가 챔피언 탐지 (팀 전력 보정)

    챔피언 승률에는 그 챔피언을 고른 팀의 전력이 섞여 있다.
    강팀이 자주 고른 챔피언은 챔피언 자체와 무관하게 승률이 높게 나온다.

    그래서 기대 승률을 이렇게 잡는다.
        기대 승률 = 그 챔피언이 픽된 경기에서, 픽한 팀의 그해 시즌 승률 평균
        잔차       = 실제 승률 - 기대 승률

    잔차가 크게 +면 팀 전력으로 설명되는 것 이상으로 성과가 좋았다는 뜻이다.

    밴픽률 기준 회귀도 함께 계산해 참고용으로 보고한다.
    (LCK 데이터에서는 밴픽률과 승률의 상관이 매우 약해 기준으로 쓰기 어렵다.)
    """
    import numpy as np

    pdf = _filter_year(players, year)
    tdf = _filter_year(teams, year)
    if pdf.empty:
        return {"error": "no data", "text": f"{year}년 선수 데이터가 없습니다."}

    # 팀-연도별 시즌 승률
    strength = _team_strength(teams)

    pdf = pdf.copy()
    pdf["team_strength"] = [
        strength.get((int(y), t)) for y, t in zip(pdf["year"], pdf["teamname"])
    ]
    pdf = pdf.dropna(subset=["team_strength", "champion", "result"])

    g = pdf.groupby("champion").agg(
        picks=("result", "count"),
        wins=("result", "sum"),
        expected=("team_strength", "mean"),
    )
    g = g[g["picks"] >= min_games].copy()
    if len(g) < 5:
        return {"error": "too few",
                "text": f"{min_games}경기 이상 챔피언이 {len(g)}종뿐입니다."}

    g["winrate"] = (g["wins"] / g["picks"] * 100).round(1)
    g["expected"] = g["expected"].round(1)
    g["residual"] = (g["winrate"] - g["expected"]).round(1)
    g["swing"] = [swing_range(int(w), int(n))
                  for w, n in zip(g["wins"], g["picks"])]

    # 밴픽률 정보 붙이기 (참고용)
    cs = champs if year is None else champs[champs["year"] == year]
    bp = cs.set_index("champion")["banpick_rate"].to_dict()
    g["banpick_rate"] = [bp.get(c) for c in g.index]

    under = g.nlargest(top, "residual")
    over = g.nsmallest(top, "residual")

    def pack(df):
        rows = []
        for c, r in df.iterrows():
            rows.append({
                "champion": c, "picks": int(r["picks"]), "wins": int(r["wins"]),
                "winrate": float(r["winrate"]),
                "expected_winrate": float(r["expected"]),
                "residual": float(r["residual"]),
                "banpick_rate": (None if pd.isna(r["banpick_rate"])
                                 else float(r["banpick_rate"])),
                "swing": r["swing"],
                "sample": grade_sample(int(r["picks"])),
            })
        return rows

    out = {
        "year": year, "min_games": min_games, "n_champions": len(g),
        "baseline": "픽한 팀의 시즌 승률 평균",
        "undervalued": pack(under),
        "overvalued": pack(over),
    }

    # 참고: 밴픽률 기준 회귀
    ref = g.dropna(subset=["banpick_rate"])
    if len(ref) >= 10:
        x = ref["banpick_rate"].to_numpy(dtype=float)
        y = ref["winrate"].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        r = float(np.corrcoef(x, y)[0, 1])
        out["banpick_ref"] = {
            "slope": round(float(slope), 4),
            "intercept": round(float(intercept), 2),
            "corr": round(r, 3), "r_squared": round(r ** 2, 3),
        }

    lines = [
        f"[저평가/과대평가 탐지] 기간={_period(year)} "
        f"({min_games}픽 이상 {len(g)}종)",
        "  기준: 기대 승률 = 그 챔피언을 픽한 경기에서 픽한 팀의 시즌 승률 평균",
        "        잔차 = 실제 승률 - 기대 승률",
        "  즉 '강팀이 골라서 이긴 것'을 걷어낸 뒤 남는 성과를 본다.",
    ]

    if "banpick_ref" in out:
        b = out["banpick_ref"]
        lines.append(
            f"  [참고] 밴픽률과 승률의 상관 r={b['corr']:+.3f} (R2={b['r_squared']:.3f}). "
            + ("관계가 매우 약해 밴픽률은 기대 승률의 기준으로 쓸 수 없다. "
               "많이 밴픽되는 챔피언이 곧 강한 챔피언은 아니라는 뜻이다."
               if abs(b["corr"]) < 0.3 else "")
        )

    def dump(title, rows):
        lines.append(title)
        for r_ in rows:
            bpr = f"{r_['banpick_rate']:.1f}%" if r_["banpick_rate"] is not None else "-"
            lines.append(
                f"    {r_['champion']:<14} {r_['picks']:>3}픽  "
                f"승률 {r_['winrate']:>5.1f}%  기대 {r_['expected_winrate']:>5.1f}%  "
                f"잔차 {r_['residual']:+6.1f}%p  (밴픽 {bpr}, {r_['sample']['grade']})"
            )

    dump("  [저평가 후보] 팀 전력으로 설명되는 것보다 성과가 좋음", out["undervalued"])
    dump("  [과대평가 후보] 팀 전력에 비해 성과가 나쁨", out["overvalued"])

    weak = [r_ for r_ in out["undervalued"] + out["overvalued"]
            if not r_["sample"]["grade"].startswith("표본충분")]
    if weak:
        names = ", ".join(r_["champion"] for r_ in weak)
        lines.append(
            f"  [표본 주의] 다음 챔피언은 표본이 충분하지 않다: {names}. "
            "이들에 대해 '저평가되었다', '강하다' 같은 단정적 서술을 하지 말 것. "
            "아래 변동폭을 보면 승률이 몇 경기로 얼마나 흔들리는지 알 수 있다."
        )
        for r_ in weak[:5]:
            lines.append(f"    {r_['champion']}: {r_['swing']}")

    lines.append(
        "  [해석 지침] 잔차는 챔피언이 강하다는 증명이 아니다. "
        "밴을 뚫고 픽된 경기만 집계되는 선택 편향이 남아 있고, "
        "챔피언 성과에는 함께 고른 조합과 상대 밴픽 구도가 섞여 있다. "
        "또한 팀 전력을 시즌 승률 하나로만 보정했으므로 보정이 완전하지 않다. "
        "'추가로 살펴볼 후보' 수준으로만 서술할 것."
    )

    out["text"] = "\n".join(lines)
    return out


def champion_stats(champs: pd.DataFrame, players: pd.DataFrame,
                   teams: pd.DataFrame, champion: str,
                   year: int | None = None) -> dict:
    """특정 챔피언의 픽/밴/승률과 팀 전력 대비 성과"""
    pdf = _filter_year(players, year)
    names = pdf["champion"].dropna().unique()

    key = _norm(champion)
    matched = [c for c in names if _norm(c) == key]
    if not matched:
        matched = [c for c in names if key and key in _norm(c)]
    if not matched:
        return {"error": "not found",
                "text": f"'{champion}'은 {_period(year)} LCK 데이터에 픽 기록이 없습니다. "
                        "밴만 되었거나 한 번도 등장하지 않았을 수 있습니다."}

    c = matched[0]
    sub = pdf[pdf["champion"] == c]
    n = len(sub)
    wins = int(sub["result"].sum())
    sample = grade_sample(n)

    # 팀 전력 대비
    strength = _team_strength(teams)
    exp_vals = [strength.get((int(y), t)) for y, t in
                zip(sub["year"], sub["teamname"])]
    exp_vals = [v for v in exp_vals if v is not None and not pd.isna(v)]
    expected = round(sum(exp_vals) / len(exp_vals), 1) if exp_vals else None

    out = {
        "champion": c, "year": year, "picks": n, "wins": wins,
        "losses": n - wins,
        "winrate": round(wins / n * 100, 1) if n else None,
        "expected_winrate": expected,
        "residual": (round(wins / n * 100 - expected, 1)
                     if n and expected is not None else None),
        "swing": swing_range(wins, n),
        "sample": sample,
    }

    # 밴픽률
    cs = champs if year is None else champs[champs["year"] == year]
    row = cs[cs["champion"] == c]
    if not row.empty:
        r = row.iloc[-1]
        out["pick_rate"] = float(r["pick_rate"])
        out["ban_rate"] = float(r["ban_rate"])
        out["banpick_rate"] = float(r["banpick_rate"])

    # 포지션 / 주요 사용 선수
    pos = sub["position"].value_counts()
    out["positions"] = pos.head(3).to_dict()
    top_players = sub.groupby("playername")["result"].agg(["count", "sum"])
    top_players = top_players.nlargest(5, "count")
    out["top_players"] = [
        {"player": p, "games": int(v["count"]), "wins": int(v["sum"])}
        for p, v in top_players.iterrows()
    ]

    lines = [f"[챔피언] {c} 기간={_period(year)}",
             f"  {n}픽 {wins}승 {n - wins}패"]
    if n >= GRADE_WEAK:
        lines.append(f"  승률 {out['winrate']}%  (변동폭: {out['swing']})")
    else:
        lines.append("  표본이 매우 적어 승률을 산출하지 않는다.")

    if "banpick_rate" in out:
        lines.append(
            f"  픽률 {out['pick_rate']}%  밴률 {out['ban_rate']}%  "
            f"밴픽률 {out['banpick_rate']}% (합집합 기준)"
        )
    if expected is not None:
        lines.append(
            f"  기대 승률 {expected}% (이 챔피언을 픽한 팀들의 시즌 승률 평균), "
            f"잔차 {out['residual']:+.1f}%p"
        )
    if out["positions"]:
        lines.append("  주 포지션: " +
                     ", ".join(f"{k} {v}회" for k, v in out["positions"].items()))
    if out["top_players"]:
        lines.append("  많이 쓴 선수: " +
                     ", ".join(f"{p['player']} {p['games']}픽({p['wins']}승)"
                               for p in out["top_players"]))

    lines.append(f"[표본 등급] {n}픽 — {sample['grade']}")
    lines.append(f"[서술 지침] {sample['instruction']}")
    lines.append(
        "  * 밴률이 높다고 강한 챔피언은 아니다. 밴은 상성과 위협도 반영한다. "
        "또한 승률에는 이 챔피언을 고른 팀의 전력과 조합 구성이 섞여 있다."
    )

    out["text"] = "\n".join(lines)
    return out


# ---------------------------------------------------------------
# 차트
# ---------------------------------------------------------------

_FONT_READY = False


def _setup_font() -> str:
    """한글 폰트 설정. 없으면 영문 라벨로 대체하도록 알린다."""
    global _FONT_READY
    import matplotlib
    from matplotlib import font_manager

    if _FONT_READY:
        return matplotlib.rcParams["font.family"][0]

    matplotlib.rcParams["axes.unicode_minus"] = False
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in ["Malgun Gothic", "AppleGothic", "NanumGothic",
                 "Noto Sans CJK KR", "Noto Sans KR"]:
        if name in installed:
            matplotlib.rcParams["font.family"] = name
            _FONT_READY = True
            return name

    _FONT_READY = True
    return ""


def make_chart(kind: str, data: dict | pd.DataFrame | None = None, **kwargs):
    """차트 생성. matplotlib Figure를 반환한다.

    kind:
      "champ_scatter" — 밴픽률 x 승률 산점도 + 회귀선 (data=find_outlier_champs 결과)
      "corr_bar"      — 지표별 상관계수 막대 (data=correlate 결과)
      "team_trend"    — 팀 연도별 승률 (data=team_trend 결과)
      "group_bar"     — 그룹별 평균 막대 (data=compare_groups 결과)
      "combo_bar"     — 진영x픽순서 조합별 승률 (data=draft_order 결과)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    font = _setup_font()
    ko = bool(font)

    def L(korean: str, english: str) -> str:
        return korean if ko else english

    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (8, 5)))

    if kind == "champ_scatter":
        rows = (data.get("undervalued", []) + data.get("overvalued", []))
        champs = kwargs.get("champs")
        year = data.get("year")
        mg = data.get("min_games", GRADE_REF)

        pool = kwargs.get("all_rows")
        if pool is None:
            pool = rows

        xs = [r["expected_winrate"] for r in pool]
        ys = [r["winrate"] for r in pool]
        cols = ["#5B8DEF" if r["residual"] >= 0 else "#D97757" for r in pool]
        ax.scatter(xs, ys, s=40, alpha=0.7, color=cols, edgecolor="none")

        lo = min(xs + ys) - 3
        hi = max(xs + ys) + 3
        ax.plot([lo, hi], [lo, hi], color="#999999", linewidth=1.0, linestyle="--")

        for r in pool:
            ax.annotate(r["champion"], (r["expected_winrate"], r["winrate"]),
                        fontsize=8, xytext=(4, 3), textcoords="offset points")

        ax.set_xlabel(L("기대 승률 (픽한 팀의 시즌 승률 평균, %)",
                        "Expected win rate (%)"))
        ax.set_ylabel(L("실제 승률 (%)", "Actual win rate (%)"))
        ax.set_title(L(f"팀 전력 대비 챔피언 성과 ({mg}픽 이상)",
                       f"Champion performance vs team strength (min {mg})"))

    elif kind == "corr_bar":
        rows = [r for r in data["results"] if r.get("corr") is not None]
        rows = rows[::-1]
        names = [r["metric"] for r in rows]
        vals = [r["corr"] for r in rows]
        colors = ["#D97757" if r["timing"] == "종료시점" else "#5B8DEF"
                  for r in rows]
        ax.barh(names, vals, color=colors)
        ax.axvline(0, color="#999999", linewidth=0.8)
        ax.set_xlabel(L("상관계수", "Correlation"))
        ax.set_title(L("승패와의 상관 (주황=결과 누수 지표)",
                       "Correlation with result (orange = leakage)"))

    elif kind == "team_ranking":
        rows = data["teams"]
        names = [r["team"] for r in rows][::-1]
        vals = [r["winrate"] for r in rows][::-1]
        ax.barh(names, vals, color="#5B8DEF")
        ax.axvline(50, color="#CCCCCC", linewidth=0.8)
        for i, v in enumerate(vals):
            ax.text(v + 0.5, i, f"{v}%", va="center", fontsize=8)
        ax.set_xlabel(L("승률 (%)", "Win rate (%)"))
        ax.set_xlim(0, max(vals) * 1.15)
        period = f"{data['year']}년" if data.get("year") else L("2024~2026", "2024-2026")
        ax.set_title(L(f"팀별 승률 ({period})", f"Win rate by team ({period})"))

    elif kind == "team_trend":
        trend = data["trend"]
        years = sorted(trend)
        vals = [trend[y]["winrate"] for y in years]
        ax.bar([str(y) for y in years], vals, color="#5B8DEF", width=0.55)
        ax.axhline(50, color="#CCCCCC", linewidth=0.8)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.8, f"{v}%", ha="center", fontsize=9)
        ax.set_ylabel(L("승률 (%)", "Win rate (%)"))
        ax.set_ylim(0, max(vals) * 1.25)
        ax.set_title(f"{data['team']} " + L("연도별 승률", "win rate by year"))

    elif kind == "group_bar":
        groups = data["groups"]
        names = [str(g["group"]) for g in groups]
        vals = [g["mean"] for g in groups]
        ax.bar(names, vals, color="#5B8DEF", width=0.5)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_xlabel(data["by"])
        ax.set_ylabel(data["metric"])
        ax.set_title(f"{data['by']} - {data['metric']}")

    elif kind == "combo_bar":
        combo = data.get("combo", {})
        order = ["Blue+선픽", "Blue+후픽", "Red+선픽", "Red+후픽"]
        names, vals, ns = [], [], []
        for k in order:
            v = combo.get(k)
            if v and v.get("winrate") is not None:
                names.append(k if ko else k.replace("선픽", "1st").replace("후픽", "2nd"))
                vals.append(v["winrate"])
                ns.append(v["games"])
        colors = ["#5B8DEF" if n.startswith("Blue") else "#D97757" for n in order[:len(names)]]
        ax.bar(names, vals, color=colors, width=0.55)
        ax.axhline(50, color="#CCCCCC", linewidth=0.8)
        for i, (v, n) in enumerate(zip(vals, ns)):
            ax.text(i, v + 0.7, f"{v}%\n(n={n})", ha="center", fontsize=8)
        ax.set_ylabel(L("승률 (%)", "Win rate (%)"))
        ax.set_ylim(0, max(vals) * 1.3)
        ax.set_title(L("진영 x 픽순서 조합별 승률",
                       "Win rate by side x pick order"))

    else:
        plt.close(fig)
        raise ValueError(f"알 수 없는 차트 종류: {kind}")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


# ===============================================================
# 단독 테스트
# ===============================================================

if __name__ == "__main__":
    data = load_data()
    teams, players = data["teams"], data["players"]

    def show(title, res):
        print("\n" + "=" * 66)
        print(title)
        print("=" * 66)
        print(res["text"])

    # 1단계
    show("1. profile_data (teams)",
         profile_data(teams, "teams", counterpart=players))
    show("2. compare_groups — 드래곤 선취별 승률",
         compare_groups(teams, "firstdragon", "result"))
    show("3. correlate — 승패와 주요 지표",
         correlate(teams, ["golddiffat15", "xpdiffat15", "csdiffat15",
                           "dragons", "barons", "towers", "visionscore",
                           "firstblood", "firstdragon", "firsttower"]))

    # 2단계
    show("4. player_stats — 미드 포지션 비교 (2026)",
         player_stats(players, position="mid", year=2026))

    top_mid = players[(players["position"] == "mid") & (players["year"] == 2026)]
    if not top_mid.empty:
        sample_player = top_mid["playername"].value_counts().index[0]
        show(f"5. player_stats — {sample_player} 상세",
             player_stats(players, player=sample_player))
        show(f"6. player_champion — {sample_player} 챔피언 목록",
             player_champion(players, sample_player, year=2026))

    show("7. head_to_head — T1 vs 젠지 (2026)",
         head_to_head(teams, "T1", "젠지", year=2026))
    show("8. draft_order — 전체 기간 (규칙 변경 탐지)",
         draft_order(teams))
    show("8-1. draft_order — 2026년",
         draft_order(teams, year=2026))
    show("9. team_trend — 디플러스 기아",
         team_trend(teams, players, "디플러스"))

    champs = data["champs"]
    outlier = find_outlier_champs(champs, players, teams, year=2026)
    show("10. find_outlier_champs — 2026 저평가/과대평가", outlier)

    # 차트 저장 테스트
    from pathlib import Path as _P
    outdir = _P(__file__).parent / "charts"
    outdir.mkdir(exist_ok=True)
    made = []
    try:
        f1 = make_chart("champ_scatter", outlier, champs=champs)
        f1.savefig(outdir / "champ_scatter.png", dpi=120); made.append("champ_scatter.png")

        f2 = make_chart("corr_bar", correlate(teams, [
            "golddiffat15", "xpdiffat15", "csdiffat15",
            "dragons", "barons", "towers", "visionscore",
            "firstblood", "firstdragon", "firsttower"]))
        f2.savefig(outdir / "corr_bar.png", dpi=120); made.append("corr_bar.png")

        f3 = make_chart("team_trend", team_trend(teams, players, "디플러스"))
        f3.savefig(outdir / "team_trend.png", dpi=120); made.append("team_trend.png")

        f4 = make_chart("combo_bar", draft_order(teams, year=2026))
        f4.savefig(outdir / "combo_bar.png", dpi=120); made.append("combo_bar.png")
    except Exception as e:
        print(f"\n[차트 오류] {e}")

    print("\n" + "=" * 66)
    print("11. make_chart")
    print("=" * 66)
    print(f"  폰트: {_setup_font() or '한글 폰트 없음 -> 영문 라벨 사용'}")
    print(f"  저장 위치: {outdir}")
    for m in made:
        print(f"    {m}")