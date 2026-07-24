"""LCK 3년치(2024 + 2025 + 2026) 데이터 병합 전처리

[사용법 - 2단계]

1단계) 진단 모드 - 팀명/컬럼 차이를 먼저 확인한다
    python prep_lol.py --check

2단계) 실제 생성
    python prep_lol.py

    -> lol_analyst/data/ 안에
       lck_players.csv / lck_teams.csv / champ_stats.csv 생성 (year 컬럼 포함)
"""

import sys
from itertools import combinations
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------
# 설정
# ---------------------------------------------------------------

BASE = Path(__file__).parent

SOURCES = {
    2024: BASE / "2024_LoL_esports_match_data_from_OraclesElixir.csv",
    2025: BASE / "2025_LoL_esports_match_data_from_OraclesElixir.csv",
    2026: BASE / "2026_LoL_esports_match_data_from_OraclesElixir.csv",
}

OUT_DIR = BASE / "lol_analyst" / "data"

LEAGUE = "LCK"

# 표본 등급 기준 (analysis.py에서도 동일하게 사용)
MIN_GAMES_CHAMP = 10        # 챔피언 승률
MIN_GAMES_PLAYER_CHAMP = 5  # 선수 x 챔피언

# 밴 컬럼. 접두사로 잡으면 다른 컬럼이 딸려올 수 있어 명시적으로 지정한다.
BAN_COLS = ["ban1", "ban2", "ban3", "ban4", "ban5"]

# ---------------------------------------------------------------
# 팀명 매핑
# ---------------------------------------------------------------
# 2024~2026 LCK는 Oracle's Elixir가 팀명을 현재 기준으로 정리해둬서
# 매핑이 불필요한 것으로 확인됨. 다른 리그/연도를 추가하면 --check로 재확인할 것.
TEAM_NAME_MAP = {
}

PLAYER_NAME_MAP = {
}


# ---------------------------------------------------------------
# 로드
# ---------------------------------------------------------------

def load_year(year: int, path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[오류] 파일 없음: {path}")
        print("       SOURCES의 파일명을 실제 파일과 맞춰주세요.")
        sys.exit(1)

    df = pd.read_csv(path, low_memory=False)

    if "league" not in df.columns:
        print(f"[오류] {path.name} 에 'league' 컬럼이 없습니다.")
        sys.exit(1)

    df = df[df["league"] == LEAGUE].copy()
    if df.empty:
        print(f"[경고] {year}: LCK 데이터가 0행입니다. 리그명 표기를 확인하세요.")

    df["year"] = year
    print(f"  {year}: {len(df):,}행 x {df.shape[1]}열 (LCK만)")
    return df


def load_all() -> dict:
    print("[로드]")
    return {y: load_year(y, p) for y, p in sorted(SOURCES.items())}


# ---------------------------------------------------------------
# 1단계: 진단
# ---------------------------------------------------------------

def diagnose(frames: dict) -> None:
    years = sorted(frames)

    print("\n" + "=" * 64)
    print("컬럼 차이")
    print("=" * 64)

    col_sets = {y: set(frames[y].columns) for y in years}
    common = set.intersection(*col_sets.values())
    union = set.union(*col_sets.values())

    print(f"전체 등장 컬럼: {len(union)}개")
    print(f"모든 연도 공통: {len(common)}개  <- 병합 후 남는 컬럼")

    partial = sorted(union - common)
    if partial:
        print(f"\n일부 연도에만 있는 컬럼 {len(partial)}개:")
        print("  " + "컬럼명".ljust(28) + "".join(str(y).rjust(8) for y in years))
        print("  " + "-" * (28 + 8 * len(years)))
        for c in partial:
            marks = "".join(("O" if c in col_sets[y] else "-").rjust(8) for y in years)
            print("  " + c.ljust(28) + marks)
        print("\n-> 위 컬럼들은 병합 시 제외됩니다.")

    if len(years) > 2:
        print("\n연도 쌍별 공통 컬럼 수:")
        for a, b in combinations(years, 2):
            print(f"  {a} & {b}: {len(col_sets[a] & col_sets[b])}개")

    print("\n" + "=" * 64)
    print("팀명 비교")
    print("=" * 64)

    team_sets = {y: set(frames[y]["teamname"].dropna().unique()) for y in years}
    all_teams = sorted(set.union(*team_sets.values()))

    print("\n  " + "팀명".ljust(28) + "".join(str(y).rjust(8) for y in years))
    print("  " + "-" * (28 + 8 * len(years)))
    for t in all_teams:
        marks = "".join(("O" if t in team_sets[y] else "-").rjust(8) for y in years)
        print("  " + t.ljust(28) + marks)

    latest = years[-1]
    stale = sorted(set.union(*[team_sets[y] for y in years[:-1]]) - team_sets[latest])

    if stale:
        print(f"\n-> {latest}에 없는 팀명 {len(stale)}개.")
        print("\nTEAM_NAME_MAP = {")
        for t in stale:
            print(f'    "{t}": "",')
        print("}")
        print(f"\n// {latest} 현재 팀 목록: {sorted(team_sets[latest])}")
    else:
        print("\n-> 모든 팀명이 전 연도에 걸쳐 동일합니다. TEAM_NAME_MAP 불필요.")

    print("\n" + "=" * 64)
    print("경기 수 / 포맷")
    print("=" * 64)
    for y in years:
        n_game = frames[y]["gameid"].nunique()
        n_team = len(team_sets[y])
        splits = sorted(frames[y]["split"].dropna().unique()) if "split" in frames[y] else []
        print(f"  {y}: {n_game:>4}경기  {n_team}팀  split={splits}")

    print("\n" + "=" * 64)
    print("participantid 분포")
    print("=" * 64)
    for y in years:
        vals = sorted(int(v) for v in frames[y]["participantid"].dropna().unique())
        print(f"  {y}: {vals}")

    print("\n진단 완료.")


# ---------------------------------------------------------------
# 2단계: 병합 및 생성
# ---------------------------------------------------------------

def merge_frames(frames: dict) -> pd.DataFrame:
    """모든 연도 공통 컬럼만 남기고 세로로 합침"""
    col_sets = [set(df.columns) for df in frames.values()]
    common = set.intersection(*col_sets)

    latest = frames[max(frames)]
    ordered = [c for c in latest.columns if c in common]

    for y, df in frames.items():
        dropped = sorted(set(df.columns) - common)
        if dropped:
            print(f"  [{y}] 제외 {len(dropped)}개: {dropped}")

    merged = pd.concat([df[ordered] for df in frames.values()], ignore_index=True)
    print(f"  병합 결과: {len(merged):,}행 x {merged.shape[1]}열")
    return merged


def apply_name_maps(df: pd.DataFrame) -> pd.DataFrame:
    if TEAM_NAME_MAP:
        before = df["teamname"].nunique()
        df["teamname"] = df["teamname"].replace(TEAM_NAME_MAP)
        print(f"  팀명 통합: {before}개 -> {df['teamname'].nunique()}개")
    if PLAYER_NAME_MAP:
        df["playername"] = df["playername"].replace(PLAYER_NAME_MAP)
    return df


def split_players_teams(df: pd.DataFrame):
    """participantid 기준으로 선수행(1~10) / 팀행(100,200) 분리"""
    pid = pd.to_numeric(df["participantid"], errors="coerce")
    players = df[pid.between(1, 10)].copy()
    teams = df[pid.isin([100, 200])].copy()
    return players, teams


def add_derived(players: pd.DataFrame, teams: pd.DataFrame):
    """분석에 쓸 파생 컬럼"""
    for d in (players, teams):
        if "gamelength" in d.columns:
            d["gamelength_min"] = (d["gamelength"] / 60).round(2)
        if "date" in d.columns:
            d["date"] = pd.to_datetime(d["date"], errors="coerce")
    return players, teams


def build_champ_stats(players: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """연도별 챔피언 픽/밴/승률 집계

    밴픽률은 합집합으로 계산한다.
    픽률 + 밴률로 단순 합산하면 같은 경기가 중복 계산됨.
    """
    rows = []

    ban_cols = [c for c in BAN_COLS if c in teams.columns]
    if not ban_cols:
        print("  [경고] 밴 컬럼을 찾지 못했습니다. 밴률이 0으로 집계됩니다.")

    for year, tdf in teams.groupby("year"):
        total_games = tdf["gameid"].nunique()
        pdf = players[players["year"] == year]

        # 픽: 해당 챔피언이 등장한 경기 집합
        pick_games = pdf.groupby("champion")["gameid"].apply(set)

        # 밴: 밴 컬럼을 세로로 펼쳐서 경기 집합
        # value_name을 'champion'으로 두면 기존 컬럼과 충돌하므로 다른 이름을 쓴다.
        if ban_cols:
            ban_long = tdf.melt(
                id_vars=["gameid"],
                value_vars=ban_cols,
                var_name="ban_slot",
                value_name="banned_champ",
            ).dropna(subset=["banned_champ"])
            ban_games = ban_long.groupby("banned_champ")["gameid"].apply(set)
        else:
            ban_games = pd.Series(dtype=object)

        wins = pdf.groupby("champion")["result"].agg(["sum", "count"])

        champs = set(pick_games.index) | set(ban_games.index)
        for c in champs:
            pg = pick_games.get(c, set())
            bg = ban_games.get(c, set())
            picks = int(wins.loc[c, "count"]) if c in wins.index else 0
            win = int(wins.loc[c, "sum"]) if c in wins.index else 0

            rows.append({
                "year": year,
                "champion": c,
                "picks": picks,
                "pick_games": len(pg),
                "ban_games": len(bg),
                "pick_rate": round(len(pg) / total_games * 100, 1),
                "ban_rate": round(len(bg) / total_games * 100, 1),
                "banpick_rate": round(len(pg | bg) / total_games * 100, 1),
                "wins": win,
                "winrate": round(win / picks * 100, 1) if picks else None,
                "reliable": picks >= MIN_GAMES_CHAMP,
            })

    out = pd.DataFrame(rows)
    return out.sort_values(["year", "banpick_rate"], ascending=[True, False])


def main() -> None:
    frames = load_all()

    print("\n[병합]")
    merged = merge_frames(frames)
    merged = apply_name_maps(merged)

    players, teams = split_players_teams(merged)
    players, teams = add_derived(players, teams)

    print("\n[집계]")
    champ_stats = build_champ_stats(players, teams)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    players.to_csv(OUT_DIR / "lck_players.csv", index=False, encoding="utf-8-sig")
    teams.to_csv(OUT_DIR / "lck_teams.csv", index=False, encoding="utf-8-sig")
    champ_stats.to_csv(OUT_DIR / "champ_stats.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 64)
    print("완료")
    print("=" * 64)
    print(f"lck_players.csv : {players.shape[0]:>7,}행 x {players.shape[1]}열")
    print(f"lck_teams.csv   : {teams.shape[0]:>7,}행 x {teams.shape[1]}열")
    print(f"champ_stats.csv : {champ_stats.shape[0]:>7,}행 x {champ_stats.shape[1]}열")

    print("\n[연도별 경기 수]")
    print(teams.groupby("year")["gameid"].nunique().to_string())

    print("\n[결측치 상위 5개 컬럼 - teams]")
    na = teams.isna().sum().sort_values(ascending=False).head(5)
    if na.empty or na.iloc[0] == 0:
        print("  없음")
    else:
        print(na.to_string())

    print("\n[표본 확인] 선수 x 챔피언 조합")
    pc = players.groupby(["playername", "champion"]).size()
    print(f"  전체 조합    : {len(pc):,}개")
    print(f"  5경기 이상   : {(pc >= MIN_GAMES_PLAYER_CHAMP).sum():,}개 "
          f"({(pc >= MIN_GAMES_PLAYER_CHAMP).mean() * 100:.1f}%)")
    print(f"  10경기 이상  : {(pc >= 10).sum():,}개")

    print("\n[표본 확인] 팀 간 상대전적")
    pair = teams.groupby("gameid")["teamname"].apply(
        lambda s: tuple(sorted(s.dropna().unique()))
    )
    pair = pair[pair.apply(len) == 2]
    print(f"  매치업 수    : {pair.nunique():,}쌍")
    print(f"  쌍당 평균    : {len(pair) / max(pair.nunique(), 1):.1f}경기")

    print("\n[밴픽률 TOP 5 - 최신 연도]")
    latest_year = champ_stats["year"].max()
    top = champ_stats[champ_stats["year"] == latest_year].head(5)
    for _, r in top.iterrows():
        wr = f"{r['winrate']}%" if pd.notna(r["winrate"]) else "-"
        print(f"  {r['champion']:<14} 밴픽 {r['banpick_rate']:>5}%  "
              f"승률 {wr:>6}  ({r['picks']}픽)")

    print(f"\n저장 위치: {OUT_DIR}")


if __name__ == "__main__":
    if "--check" in sys.argv:
        diagnose(load_all())
    else:
        main()