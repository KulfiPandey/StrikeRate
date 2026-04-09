# pipeline/match_features.py
# ──────────────────────────────────────────────────────
# What this file does:
#   Aggregates ball-by-ball data into ONE ROW PER MATCH.
#   This is what the ML model needs — not 279k delivery rows,
#   but ~1175 match rows, each with features describing that match.
#
# Why one row per match?
#   ML models learn patterns. The pattern here is:
#   "Given these match conditions → who won?"
#   Each row = one training example for the model.
# ──────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import os
from config import PROCESSED_DIR

def build_match_dataset():
    print("Loading master deliveries...")
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "master_deliveries.csv"), low_memory=False)
    df['season'] = df['season'].astype(str).str.extract(r'(\d{4})')[0].astype(int)

    # ── 1. Match metadata (one row per match) ──
    info = (
        df.groupby("match_id")
        .first()
        .reset_index()[["match_id", "season", "start_date", "venue",
                        "toss_winner", "toss_decision", "match_winner"]]
    )

    # ── 2. Innings run totals ──
    innings_runs = (
        df.groupby(["match_id", "innings", "batting_team"])["runs_off_bat"]
        .sum()
        .reset_index()
    )

    inn1 = innings_runs[innings_runs["innings"] == 1].rename(
        columns={"batting_team": "team1", "runs_off_bat": "team1_runs"}
    ).drop(columns="innings")

    inn2 = innings_runs[innings_runs["innings"] == 2].rename(
        columns={"batting_team": "team2", "runs_off_bat": "team2_runs"}
    ).drop(columns="innings")

    matches = inn1.merge(inn2, on="match_id", how="inner")

    # ── 3. Phase-wise runs (the rich features) ──
    # Why phases? The model needs to know HOW runs were scored,
    # not just the total. A team scoring 80 in powerplay vs 40
    # tells a very different story.
    def phase_runs(innings_num, over_start, over_end, label):
        mask = (
            (df["innings"] == innings_num) &
            (df["ball"] >= over_start) &
            (df["ball"] < over_end)
        )
        return (
            df[mask].groupby("match_id")["runs_off_bat"]
            .sum().reset_index().rename(columns={"runs_off_bat": label})
        )

    phase_features = [
        phase_runs(1, 0, 6,  "inn1_pp_runs"),
        phase_runs(1, 6, 15, "inn1_mid_runs"),
        phase_runs(1, 15, 20,"inn1_death_runs"),
        phase_runs(2, 0, 6,  "inn2_pp_runs"),
        phase_runs(2, 6, 15, "inn2_mid_runs"),
        phase_runs(2, 15, 20,"inn2_death_runs"),
    ]

    for pf in phase_features:
        matches = matches.merge(pf, on="match_id", how="left")

    # ── 4. Wickets per innings ──
    wickets = (
        df[df["wicket_type"].notna()]
        .groupby(["match_id", "innings"])
        .size().reset_index(name="wickets")
    )
    w1 = wickets[wickets["innings"]==1][["match_id","wickets"]].rename(columns={"wickets":"team1_wickets"})
    w2 = wickets[wickets["innings"]==2][["match_id","wickets"]].rename(columns={"wickets":"team2_wickets"})
    matches = matches.merge(w1, on="match_id", how="left")
    matches = matches.merge(w2, on="match_id", how="left")

    # ── 5. Boundary % and dot ball % per innings ──
    df["is_boundary"] = (df["runs_off_bat"] >= 4).astype(int)
    df["is_dot"]      = ((df["runs_off_bat"] == 0) & df["wicket_type"].isna()).astype(int)

    for inn, label in [(1, "inn1"), (2, "inn2")]:
        sub = df[df["innings"] == inn].groupby("match_id").agg(
            **{f"{label}_boundary_pct": ("is_boundary", "mean"),
               f"{label}_dot_pct":      ("is_dot",      "mean")}
        ).reset_index()
        matches = matches.merge(sub, on="match_id", how="left")

    # ── 6. Merge metadata + encode ──
    matches = matches.merge(info, on="match_id", how="left")
    matches["start_date"] = pd.to_datetime(matches["start_date"])

    # Toss feature: did team1 win the toss?
    matches["team1_won_toss"] = (matches["toss_winner"] == matches["team1"]).astype(int)

    # Target: did team1 win?
    matches["team1_won"] = (matches["match_winner"] == matches["team1"]).astype(int)

    # Encode teams and venue as numbers for ML
    for col in ["team1", "team2", "venue"]:
        matches[col + "_enc"] = matches[col].astype("category").cat.codes

    # Fill missing phase values
    phase_cols = ["inn1_pp_runs","inn1_mid_runs","inn1_death_runs",
                  "inn2_pp_runs","inn2_mid_runs","inn2_death_runs"]
    matches[phase_cols] = matches[phase_cols].fillna(0)
    matches[["team1_wickets","team2_wickets"]] = matches[["team1_wickets","team2_wickets"]].fillna(10)

    # ── Save ──
    out_path = os.path.join(PROCESSED_DIR, "match_features.csv")
    matches.to_csv(out_path, index=False)

    print(f"✅ Saved: {out_path}")
    print(f"📊 Shape: {matches.shape}")
    print(f"📅 Seasons: {sorted(matches['season'].unique())}")
    print(f"🏆 Win rate balance: {matches['team1_won'].mean():.2%} team1 wins")
    print(f"\nSample:\n{matches[['match_id','season','team1','team2','team1_runs','team2_runs','team1_won']].head()}")

    return matches

if __name__ == "__main__":
    build_match_dataset()