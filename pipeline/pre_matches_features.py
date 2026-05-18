from __future__ import annotations

import ast
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

try:
    from config import PROCESSED_DIR as _PROCESSED_DIR, RAW_DIR as _RAW_DIR
except Exception:
    _PROCESSED_DIR = ROOT / "data" / "processed"
    _RAW_DIR = ROOT / "data" / "raw"

PROCESSED_DIR = Path(_PROCESSED_DIR)
RAW_DIR = Path(_RAW_DIR)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = PROCESSED_DIR / "pre_match_clean.csv"

K_ELO = 20.0
FORM_WINDOW = 5


def norm_text(x: Any) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)) or pd.isna(x):
        return ""
    return str(x).strip()


def norm_lower(x: Any) -> str:
    return norm_text(x).lower()


def season_to_int(value: Any):
    text = norm_text(value)
    if not text:
        return pd.NA
    m = re.search(r"\d{4}", text)
    return int(m.group()) if m else pd.NA


def parse_teams(row: dict[str, Any]) -> tuple[str | None, str | None]:
    for a, b in (("team1", "team2"), ("team_1", "team_2"), ("home_team", "away_team")):
        if a in row and b in row:
            t1, t2 = row.get(a), row.get(b)
            if pd.notna(t1) and pd.notna(t2):
                return norm_text(t1), norm_text(t2)

    teams = row.get("teams")
    if teams is None or pd.isna(teams):
        return None, None

    text = norm_text(teams)
    if not text:
        return None, None

    try:
        if text.startswith("[") or text.startswith("("):
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple)) and len(parsed) >= 2:
                return norm_text(parsed[0]), norm_text(parsed[1])
    except Exception:
        pass

    for sep in ("|", ";", "/", ","):
        if sep in text:
            parts = [p.strip().strip("'\"") for p in text.split(sep) if p.strip()]
            if len(parts) >= 2:
                return parts[0], parts[1]

    return None, None


def result_is_valid(winner: Any) -> bool:
    w = norm_lower(winner)
    return w not in {"", "nan", "none", "draw", "no result", "tie", "abandoned", "cancelled"}


def team1_bats_first(toss_winner: Any, toss_decision: Any, team1: Any) -> float:
    tw, td, t1 = norm_lower(toss_winner), norm_lower(toss_decision), norm_lower(team1)
    if not tw or not td or not t1:
        return np.nan

    if td == "bat":
        return 1.0 if tw == t1 else 0.0
    if td == "field":
        return 0.0 if tw == t1 else 1.0
    return np.nan


def safe_mean(values, default=0.5) -> float:
    clean = [v for v in values if v is not None and not pd.isna(v)]
    return float(np.mean(clean)) if clean else default


def load_metadata_from_parquet() -> pd.DataFrame | None:
    parquet_path = PROCESSED_DIR / "matches.parquet"
    if not parquet_path.exists():
        return None

    df = pd.read_parquet(parquet_path)
    needed = ["match_id", "team1", "team2"]
    if not all(c in df.columns for c in needed):
        return None

    out = df.copy()
    if "winner" not in out.columns and "match_winner" in out.columns:
        out["winner"] = out["match_winner"]
    if "start_date" not in out.columns and "date" in out.columns:
        out["start_date"] = out["date"]

    cols = [
        c
        for c in [
            "match_id",
            "season",
            "start_date",
            "venue",
            "city",
            "team1",
            "team2",
            "winner",
            "toss_winner",
            "toss_decision",
        ]
        if c in out.columns
    ]
    out = out[cols].drop_duplicates("match_id")
    out["start_date"] = pd.to_datetime(out["start_date"], errors="coerce")
    out["season"] = out["season"].map(season_to_int)
    return out


def load_metadata() -> pd.DataFrame:
    from_parquet = load_metadata_from_parquet()
    if from_parquet is not None and len(from_parquet):
        return from_parquet

    rows: list[dict[str, Any]] = []

    for fp in sorted(RAW_DIR.glob("*_info.csv")):
        try:
            raw = pd.read_csv(fp, low_memory=False, nrows=1)
        except Exception:
            continue

        if raw.empty:
            continue

        row = raw.iloc[0].to_dict()
        row["match_id"] = norm_text(Path(fp).stem.replace("_info", ""))

        team1, team2 = parse_teams(row)
        if not team1 or not team2:
            continue

        row["team1"] = team1
        row["team2"] = team2
        row["season"] = season_to_int(row.get("season"))
        row["start_date"] = row.get("date", row.get("start_date"))
        rows.append(row)

    if rows:
        out = pd.DataFrame(rows)
        out["start_date"] = pd.to_datetime(out["start_date"], errors="coerce")
        out["season"] = out["season"].map(season_to_int)
        return out

    # Fallback only if raw info files are not available.
    fallback = PROCESSED_DIR / "match_features.csv"
    if fallback.exists():
        df = pd.read_csv(fallback, low_memory=False)
        needed = [c for c in ["match_id", "season", "start_date", "venue", "team1", "team2", "winner", "toss_winner", "toss_decision"] if c in df.columns]
        if needed:
            out = df.drop_duplicates("match_id")[needed].copy()
            out["season"] = out["season"].map(season_to_int)
            out["start_date"] = pd.to_datetime(out["start_date"], errors="coerce")
            return out

    raise FileNotFoundError("No raw *_info.csv files found and no fallback match_features.csv available.")


def build_pre_match_features(meta: pd.DataFrame) -> pd.DataFrame:
    meta = meta.copy()
    meta = meta.sort_values(["start_date", "match_id"], kind="stable").reset_index(drop=True)

    elo = defaultdict(lambda: 1500.0)
    recent_results = defaultdict(lambda: deque(maxlen=FORM_WINDOW))
    h2h_results = defaultdict(list)
    venue_team_results = defaultdict(list)
    venue_bat_first_results = defaultdict(list)

    records: list[dict[str, Any]] = []

    for row in meta.itertuples(index=False):
        d = row._asdict()

        match_id = norm_text(d.get("match_id"))
        team1 = norm_text(d.get("team1"))
        team2 = norm_text(d.get("team2"))
        venue = norm_text(d.get("venue"))
        winner = d.get("winner")
        toss_winner = d.get("toss_winner")
        toss_decision = d.get("toss_decision")

        t1_elo = float(elo[team1])
        t2_elo = float(elo[team2])
        t1_form = safe_mean(recent_results[team1], default=0.5)
        t2_form = safe_mean(recent_results[team2], default=0.5)
        h2h_key = (team1, team2)
        venue_team_key = (venue, team1)

        head_to_head = safe_mean(h2h_results[h2h_key], default=0.5)
        venue_t1_wr = safe_mean(venue_team_results[venue_team_key], default=0.5)
        venue_bat_first_wr = safe_mean(venue_bat_first_results[venue], default=0.5)

        toss_is_team1 = np.nan
        if norm_text(toss_winner):
            toss_is_team1 = int(norm_lower(toss_winner) == norm_lower(team1))

        toss_decision_enc = -1
        td = norm_lower(toss_decision)
        if td == "bat":
            toss_decision_enc = 1
        elif td == "field":
            toss_decision_enc = 0

        valid_result = result_is_valid(winner)
        team1_won = np.nan
        if valid_result and norm_text(winner) and team1:
            team1_won = int(norm_lower(winner) == norm_lower(team1))

        record = {
            "match_id": match_id,
            "season": season_to_int(d.get("season")),
            "start_date": pd.to_datetime(d.get("start_date"), errors="coerce"),
            "venue": venue,
            "team1": team1,
            "team2": team2,
            "winner": norm_text(winner),
            "toss_winner": norm_text(toss_winner),
            "toss_decision": norm_text(toss_decision),
            "toss_is_team1": toss_is_team1,
            "toss_decision_enc": toss_decision_enc,
            "team1_elo": t1_elo,
            "team2_elo": t2_elo,
            "elo_diff": t1_elo - t2_elo,
            "team1_form": t1_form,
            "team2_form": t2_form,
            "form_diff": t1_form - t2_form,
            "head_to_head": head_to_head,
            "venue_t1_wr": venue_t1_wr,
            "venue_bat_first_wr": venue_bat_first_wr,
            "team1_won": team1_won,
        }
        records.append(record)

        if not valid_result:
            continue

        outcome = int(team1_won)

        # Update rolling results after the match.
        recent_results[team1].append(outcome)
        recent_results[team2].append(1 - outcome)
        h2h_results[h2h_key].append(outcome)
        venue_team_results[venue_team_key].append(outcome)
        venue_team_results[(venue, team2)].append(1 - outcome)

        t1_bf = team1_bats_first(toss_winner, toss_decision, team1)
        if not pd.isna(t1_bf):
            batting_first_win = outcome if int(t1_bf) == 1 else (1 - outcome)
            venue_bat_first_results[venue].append(batting_first_win)

        # ELO update
        exp1 = 1.0 / (1.0 + 10.0 ** ((t2_elo - t1_elo) / 400.0))
        elo[team1] = t1_elo + K_ELO * (outcome - exp1)
        elo[team2] = t2_elo + K_ELO * ((1 - outcome) - (1 - exp1))

    out = pd.DataFrame(records)
    out["season"] = out["season"].map(season_to_int).astype("Int64")
    out["start_date"] = pd.to_datetime(out["start_date"], errors="coerce")

    numeric_cols = [
        "toss_is_team1",
        "toss_decision_enc",
        "team1_elo",
        "team2_elo",
        "elo_diff",
        "team1_form",
        "team2_form",
        "form_diff",
        "head_to_head",
        "venue_t1_wr",
        "venue_bat_first_wr",
        "team1_won",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out.to_csv(OUT_PATH, index=False)
    print(f"Saved: {OUT_PATH}")
    print(f"Shape: {out.shape}")
    print(out.head())

    return out


def main():
    print("Loading raw match info...")
    meta = load_metadata()
    print(f"Loaded {len(meta)} matches across {meta['season'].nunique()} seasons")
    build_pre_match_features(meta)


if __name__ == "__main__":
    main()