"""
Rolling player-quality features from ball-by-ball data (no external XI / ESPN).

For each pre-match row, using only prior deliveries:
  - Core squad = players who appeared (batted or bowled) for that team in the
    last LOOKBACK_MATCHES completed fixtures.
  - Batting quality = mean strike rate (runs / legal balls) over that window.
  - Bowling quality = mean economy (runs per over on legal balls) over that window.

Output: data/processed/pre_match_with_player_quality.csv
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from config import PROCESSED_DIR
from pipeline.team_name_standardizer import standardize_team_name

LOOKBACK_MATCHES = 10
MIN_BALLS_BAT = 6
MIN_BALLS_BOWL = 6
DEFAULT_BAT_SR = 120.0
DEFAULT_BOWL_ECO = 8.5


def load_data():
    master = pd.read_csv(Path(PROCESSED_DIR) / "master_deliveries.csv", low_memory=False)
    master["start_date"] = pd.to_datetime(master["start_date"])
    pre = pd.read_csv(Path(PROCESSED_DIR) / "pre_match_clean.csv")
    pre["start_date"] = pd.to_datetime(pre["start_date"])
    return master, pre


def _legal_mask(df):
    wides = df["wides"].fillna(0)
    noballs = df["noballs"].fillna(0)
    return (wides == 0) & (noballs == 0)


def build_player_match_stats(master):
    """Per player per match: legal-ball batting and bowling totals."""
    m = master.copy()
    m["batting_team_std"] = m["batting_team"].map(standardize_team_name)
    m["bowling_team_std"] = m["bowling_team"].map(standardize_team_name)
    legal = _legal_mask(m)

    bat = (
        m.loc[legal]
        .groupby(["match_id", "batting_team_std", "striker"], as_index=False)
        .agg(bat_runs=("runs_off_bat", "sum"), bat_balls=("ball", "count"))
    )
    bat.rename(columns={"batting_team_std": "team", "striker": "player"}, inplace=True)

    bowl = (
        m.loc[legal]
        .groupby(["match_id", "bowling_team_std", "bowler"], as_index=False)
        .agg(bowl_runs=("runs_off_bat", "sum"), bowl_balls=("ball", "count"))
    )
    bowl.rename(columns={"bowling_team_std": "team", "bowler": "player"}, inplace=True)

    stats = bat.merge(bowl, on=["match_id", "team", "player"], how="outer")
    for col in ("bat_runs", "bat_balls", "bowl_runs", "bowl_balls"):
        stats[col] = stats[col].fillna(0)
    return stats


def _mean_bat_sr(sub):
    srs = []
    for _, g in sub.groupby("player"):
        balls = g["bat_balls"].sum()
        if balls < MIN_BALLS_BAT:
            continue
        srs.append((g["bat_runs"].sum() / balls) * 100)
    return float(np.mean(srs)) if srs else DEFAULT_BAT_SR


def _mean_bowl_economy(sub):
    ecos = []
    for _, g in sub.groupby("player"):
        balls = g["bowl_balls"].sum()
        if balls < MIN_BALLS_BOWL:
            continue
        ecos.append(g["bowl_runs"].sum() / (balls / 6))
    return float(np.mean(ecos)) if ecos else DEFAULT_BOWL_ECO


def add_player_quality_features(pre, player_stats):
    pre = pre.sort_values("start_date").reset_index(drop=True)
    pre["team1_std"] = pre["team1"].map(standardize_team_name)
    pre["team2_std"] = pre["team2"].map(standardize_team_name)

    team_history = defaultdict(list)  # team -> [(date, match_id), ...]
    stats_by_mid = {
        mid: grp for mid, grp in player_stats.groupby("match_id")
    }

    t1_bat, t2_bat, t1_bowl, t2_bowl = [], [], [], []

    for _, row in pre.iterrows():
        date = row["start_date"]
        t1, t2 = row["team1_std"], row["team2_std"]

        def team_metrics(team):
            past = [
                mid
                for d, mid in team_history[team]
                if d < date
            ][-LOOKBACK_MATCHES:]
            if not past:
                return DEFAULT_BAT_SR, DEFAULT_BOWL_ECO

            chunks = [stats_by_mid[mid] for mid in past if mid in stats_by_mid]
            if not chunks:
                return DEFAULT_BAT_SR, DEFAULT_BOWL_ECO

            sub = pd.concat(chunks, ignore_index=True)
            sub = sub[sub["team"] == team]
            if sub.empty:
                return DEFAULT_BAT_SR, DEFAULT_BOWL_ECO

            core = sub["player"].unique()
            sub = sub[sub["player"].isin(core)]
            return _mean_bat_sr(sub), _mean_bowl_economy(sub)

        b1, w1 = team_metrics(t1)
        b2, w2 = team_metrics(t2)
        t1_bat.append(b1)
        t2_bat.append(b2)
        t1_bowl.append(w1)
        t2_bowl.append(w2)

        mid = row["match_id"]
        team_history[t1].append((date, mid))
        team_history[t2].append((date, mid))

    pre["team1_batting_quality"] = t1_bat
    pre["team2_batting_quality"] = t2_bat
    pre["team1_bowling_quality"] = t1_bowl
    pre["team2_bowling_quality"] = t2_bowl
    pre["batting_quality_diff"] = pre["team1_batting_quality"] - pre["team2_batting_quality"]
    pre["bowling_quality_diff"] = pre["team2_bowling_quality"] - pre["team1_bowling_quality"]

    return pre


def main():
    print("Loading master deliveries and pre-match table...")
    master, pre = load_data()
    print(f"  {len(master):,} deliveries, {len(pre)} matches")

    print("Building per-player per-match stats...")
    player_stats = build_player_match_stats(master)
    print(f"  {len(player_stats):,} player-match rows")

    print(f"Adding rolling player quality (last {LOOKBACK_MATCHES} matches)...")
    enriched = add_player_quality_features(pre, player_stats)

    out = Path(PROCESSED_DIR) / "pre_match_with_player_quality.csv"
    enriched.to_csv(out, index=False)
    print(f"Saved -> {out}")

    cols = [
        "team1", "team2", "season",
        "team1_batting_quality", "team2_batting_quality",
        "team1_bowling_quality", "team2_bowling_quality",
    ]
    print("\nSample (last 5 matches):")
    print(enriched[cols].tail().to_string(index=False))

    for c in ("team1_batting_quality", "team1_bowling_quality"):
        print(f"  {c}: nunique={enriched[c].nunique()}, mean={enriched[c].mean():.2f}")


if __name__ == "__main__":
    main()
