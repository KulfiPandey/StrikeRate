"""
Leak-free player / team quality features.

Per-player strike rate and economy are computed per match, then rolling
means use ONLY prior matches (shift + rolling). Team quality for a match is
the mean of participating players' pre-match rolling values — never same-match
ball-by-ball totals.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

PREMATCH_PATH = ROOT / "data" / "processed" / "pre_match_clean.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "pre_match_with_player_quality.csv"

ROLL_WINDOW = 10
MIN_PERIODS = 3

INVALID_WINNERS = {
    "",
    "nan",
    "none",
    "draw",
    "no result",
    "tie",
    "abandoned",
    "cancelled",
}


def _rolling_prior(series: pd.Series) -> pd.Series:
    return series.shift(1).rolling(ROLL_WINDOW, min_periods=MIN_PERIODS).mean()


def main() -> None:
    from pipeline.data_store import load_deliveries, load_matches

    print("Loading deliveries, matches, and pre-match table...")

    deliveries = load_deliveries()
    matches = load_matches()
    prematch = pd.read_csv(PREMATCH_PATH, low_memory=False)

    prematch["match_id"] = prematch["match_id"].astype(str)
    deliveries["match_id"] = deliveries["match_id"].astype(str)

    schedule = prematch[["match_id", "start_date", "team1", "team2"]].copy()
    schedule["start_date"] = pd.to_datetime(schedule["start_date"], errors="coerce")

    print(f"  {len(deliveries):,} deliveries")
    print(f"  {len(matches):,} matches")
    print(f"  {len(prematch):,} pre-match rows")

    # --- per-player batting (this match only; rolling applied later) ---
    batter_team = (
        deliveries.groupby(["match_id", "batter"], as_index=False)["batting_team"]
        .first()
        .rename(columns={"batter": "player", "batting_team": "team"})
    )

    bat_match = (
        deliveries.groupby(["match_id", "batter"], as_index=False)
        .agg(runs=("batsman_runs", "sum"), balls=("ball", "count"))
        .rename(columns={"batter": "player"})
    )
    bat_match["strike_rate"] = np.where(
        bat_match["balls"] > 0,
        bat_match["runs"] / bat_match["balls"] * 100.0,
        np.nan,
    )
    bat_match = bat_match.merge(batter_team, on=["match_id", "player"], how="left")
    bat_match = bat_match.merge(schedule[["match_id", "start_date"]], on="match_id", how="left")
    bat_match = bat_match.sort_values(["player", "start_date", "match_id"], kind="stable")
    bat_match["bat_quality_pre"] = bat_match.groupby("player", sort=False)["strike_rate"].transform(
        _rolling_prior
    )

    team_bat = (
        bat_match.groupby(["match_id", "team"], as_index=False)["bat_quality_pre"]
        .mean()
        .rename(columns={"team": "batting_team", "bat_quality_pre": "team_batting_quality"})
    )

    # --- per-player bowling ---
    bowler_team = (
        deliveries.groupby(["match_id", "bowler"], as_index=False)["bowling_team"]
        .first()
        .rename(columns={"bowler": "player", "bowling_team": "team"})
    )

    bowl_match = (
        deliveries.groupby(["match_id", "bowler"], as_index=False)
        .agg(
            runs_conceded=("total_runs", "sum"),
            balls=("ball", "count"),
        )
        .rename(columns={"bowler": "player"})
    )
    overs = bowl_match["balls"] / 6.0
    bowl_match["economy"] = np.where(overs > 0, bowl_match["runs_conceded"] / overs, np.nan)
    bowl_match = bowl_match.merge(bowler_team, on=["match_id", "player"], how="left")
    bowl_match = bowl_match.merge(schedule[["match_id", "start_date"]], on="match_id", how="left")
    bowl_match = bowl_match.sort_values(["player", "start_date", "match_id"], kind="stable")
    bowl_match["bowl_quality_pre"] = bowl_match.groupby("player", sort=False)["economy"].transform(
        _rolling_prior
    )

    team_bowl = (
        bowl_match.groupby(["match_id", "team"], as_index=False)["bowl_quality_pre"]
        .mean()
        .rename(columns={"team": "bowling_team", "bowl_quality_pre": "team_bowling_quality"})
    )

    # --- team-level rolling fallback (no XI; prior team innings only) ---
    team_inn_bat = (
        deliveries.groupby(["match_id", "batting_team"], as_index=False)
        .agg(runs=("total_runs", "sum"), balls=("ball", "count"))
    )
    team_inn_bat["inn_sr"] = np.where(
        team_inn_bat["balls"] > 0,
        team_inn_bat["runs"] / team_inn_bat["balls"] * 100.0,
        np.nan,
    )
    team_inn_bat = team_inn_bat.merge(schedule[["match_id", "start_date"]], on="match_id", how="left")
    team_inn_bat = team_inn_bat.sort_values(["batting_team", "start_date", "match_id"], kind="stable")
    team_inn_bat["team_bat_roll"] = team_inn_bat.groupby("batting_team", sort=False)["inn_sr"].transform(
        _rolling_prior
    )

    team_inn_bowl = (
        deliveries.groupby(["match_id", "bowling_team"], as_index=False)
        .agg(runs=("total_runs", "sum"), balls=("ball", "count"))
    )
    overs_t = team_inn_bowl["balls"] / 6.0
    team_inn_bowl["inn_eco"] = np.where(overs_t > 0, team_inn_bowl["runs"] / overs_t, np.nan)
    team_inn_bowl = team_inn_bowl.merge(schedule[["match_id", "start_date"]], on="match_id", how="left")
    team_inn_bowl = team_inn_bowl.sort_values(["bowling_team", "start_date", "match_id"], kind="stable")
    team_inn_bowl["team_bowl_roll"] = team_inn_bowl.groupby("bowling_team", sort=False)["inn_eco"].transform(
        _rolling_prior
    )

    df = prematch.copy()

    t1_bat = team_bat.rename(
        columns={"batting_team": "team1", "team_batting_quality": "team1_batting_quality"}
    )
    t2_bat = team_bat.rename(
        columns={"batting_team": "team2", "team_batting_quality": "team2_batting_quality"}
    )
    t1_bowl = team_bowl.rename(
        columns={"bowling_team": "team1", "team_bowling_quality": "team1_bowling_quality"}
    )
    t2_bowl = team_bowl.rename(
        columns={"bowling_team": "team2", "team_bowling_quality": "team2_bowling_quality"}
    )

    for frame, keys in (
        (t1_bat, ["match_id", "team1"]),
        (t2_bat, ["match_id", "team2"]),
        (t1_bowl, ["match_id", "team1"]),
        (t2_bowl, ["match_id", "team2"]),
    ):
        df = df.merge(frame, on=keys, how="left")

    t1_bat_fb = team_inn_bat.rename(
        columns={"batting_team": "team1", "team_bat_roll": "team1_bat_roll"}
    )[["match_id", "team1", "team1_bat_roll"]]
    t2_bat_fb = team_inn_bat.rename(
        columns={"batting_team": "team2", "team_bat_roll": "team2_bat_roll"}
    )[["match_id", "team2", "team2_bat_roll"]]
    t1_bowl_fb = team_inn_bowl.rename(
        columns={"bowling_team": "team1", "team_bowl_roll": "team1_bowl_roll"}
    )[["match_id", "team1", "team1_bowl_roll"]]
    t2_bowl_fb = team_inn_bowl.rename(
        columns={"bowling_team": "team2", "team_bowl_roll": "team2_bowl_roll"}
    )[["match_id", "team2", "team2_bowl_roll"]]

    for frame, keys in (
        (t1_bat_fb, ["match_id", "team1"]),
        (t2_bat_fb, ["match_id", "team2"]),
        (t1_bowl_fb, ["match_id", "team1"]),
        (t2_bowl_fb, ["match_id", "team2"]),
    ):
        df = df.merge(frame, on=keys, how="left")

    df["team1_batting_quality"] = df["team1_batting_quality"].fillna(df["team1_bat_roll"])
    df["team2_batting_quality"] = df["team2_batting_quality"].fillna(df["team2_bat_roll"])
    df["team1_bowling_quality"] = df["team1_bowling_quality"].fillna(df["team1_bowl_roll"])
    df["team2_bowling_quality"] = df["team2_bowling_quality"].fillna(df["team2_bowl_roll"])

    df = df.drop(
        columns=["team1_bat_roll", "team2_bat_roll", "team1_bowl_roll", "team2_bowl_roll"],
        errors="ignore",
    )

    # winner from official matches.csv
    if "match_id" not in matches.columns and "id" in matches.columns:
        winner_map = matches[["id", "winner"]].rename(columns={"id": "match_id"})
    else:
        winner_map = matches[["match_id", "winner"]].copy()
    winner_map["match_id"] = winner_map["match_id"].astype(str)

    if "winner" in df.columns:
        df = df.drop(columns=["winner"])

    df = df.merge(winner_map, on="match_id", how="left")

    wnorm = df["winner"].astype(str).str.strip().str.lower()
    df = df[~wnorm.isin(INVALID_WINNERS)].copy()

    quality_cols = [
        "team1_batting_quality",
        "team2_batting_quality",
        "team1_bowling_quality",
        "team2_bowling_quality",
    ]
    for col in quality_cols:
        df[col] = df[col].fillna(df[col].median())

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved -> {OUTPUT_PATH}")
    print(f"  Rows with player rolling bat quality: {bat_match['bat_quality_pre'].notna().sum():,}")
    print(f"  Rows after dropping invalid results: {len(df):,}")

    print("\nSample (recent):")
    show = [
        "team1",
        "team2",
        "season",
        "team1_batting_quality",
        "team2_batting_quality",
        "team1_bowling_quality",
        "team2_bowling_quality",
        "winner",
    ]
    print(df[show].tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
