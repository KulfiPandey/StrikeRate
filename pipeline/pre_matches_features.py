import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import PROCESSED_DIR


def load_base():
    master = pd.read_csv(Path(PROCESSED_DIR) / "master_deliveries.csv", low_memory=False)
    matches = pd.read_csv(Path(PROCESSED_DIR) / "match_features.csv")

    matches["start_date"] = pd.to_datetime(matches["start_date"])
    matches["season"] = matches["season"].astype(str).str[:4].astype(int)

    return master, matches


def build_team_results(master):
    inn = (
        master.groupby(["match_id", "innings"])
        .agg(
            batting_team=("batting_team", "first"),
            runs=("runs_off_bat", "sum"),
            balls=("ball", "count"),
            start_date=("start_date", "first"),
        )
        .reset_index()
    )

    inn["start_date"] = pd.to_datetime(inn["start_date"])

    records = []
    for mid, grp in inn.groupby("match_id"):
        i1 = grp[grp["innings"] == 1]
        i2 = grp[grp["innings"] == 2]

        if i1.empty or i2.empty:
            continue

        r1, r2 = i1["runs"].iloc[0], i2["runs"].iloc[0]
        t1, t2 = i1["batting_team"].iloc[0], i2["batting_team"].iloc[0]
        b1, b2 = i1["balls"].iloc[0], i2["balls"].iloc[0]
        date = i1["start_date"].iloc[0]

        winner = t1 if r1 > r2 else t2

        for team, runs, balls, won in [
            (t1, r1, b1, winner == t1),
            (t2, r2, b2, winner == t2),
        ]:
            records.append({
                "match_id": mid,
                "start_date": date,
                "team": team,
                "runs": runs,
                "balls": balls,
                "did_win": int(won),
                "batting_first": int(team == t1),
            })

    df = pd.DataFrame(records)
    df["sr"] = np.where(df["balls"] > 0, (df["runs"] / df["balls"]) * 100, np.nan)

    return df.sort_values("start_date").reset_index(drop=True)


def compute_elo(matches, k=20):
    matches = matches.sort_values("start_date").reset_index(drop=True)

    elo = {}
    t1_elo, t2_elo = [], []

    for _, row in matches.iterrows():
        t1, t2 = row["team1"], row["team2"]

        e1 = elo.get(t1, 1500)
        e2 = elo.get(t2, 1500)

        t1_elo.append(e1)
        t2_elo.append(e2)

        exp1 = 1 / (1 + 10 ** ((e2 - e1) / 400))
        res1 = row["team1_won"]

        elo[t1] = e1 + k * (res1 - exp1)
        elo[t2] = e2 + k * ((1 - res1) - (1 - exp1))

    matches["team1_elo"] = t1_elo
    matches["team2_elo"] = t2_elo

    return matches


def add_rolling_form(matches, team_results, n=5):
    matches = matches.sort_values("start_date").reset_index(drop=True)

    t1_wr, t2_wr, t1_sr, t2_sr = [], [], [], []

    for _, row in matches.iterrows():
        date = row["start_date"]

        for team, wr_list, sr_list in [
            (row["team1"], t1_wr, t1_sr),
            (row["team2"], t2_wr, t2_sr),
        ]:
            past = team_results[
                (team_results["team"] == team) &
                (team_results["start_date"] < date)
            ].tail(n)

            if len(past) > 0:
                wr_list.append(past["did_win"].mean())
                sr_list.append((past["runs"].sum() / past["balls"].sum()) * 100)
            else:
                wr_list.append(0.5)
                sr_list.append(120.0)

    matches["team1_form"] = t1_wr
    matches["team2_form"] = t2_wr
    matches["team1_bat_sr"] = t1_sr
    matches["team2_bat_sr"] = t2_sr

    return matches


def add_head_to_head(matches, team_results, n=5):
    matches = matches.sort_values("start_date").reset_index(drop=True)

    match_teams = team_results.groupby("match_id")["team"].apply(set).to_dict()

    pair_map = {}
    for mid, teams in match_teams.items():
        key = tuple(sorted(teams))
        pair_map.setdefault(key, []).append(mid)

    h2h = []

    for _, row in matches.iterrows():
        date = row["start_date"]
        t1, t2 = row["team1"], row["team2"]

        past_ids = pair_map.get(tuple(sorted([t1, t2])), [])

        past = team_results[
            (team_results["match_id"].isin(past_ids)) &
            (team_results["start_date"] < date) &
            (team_results["team"] == t1)
        ].tail(n)

        h2h.append(past["did_win"].mean() if len(past) > 0 else 0.5)

    matches["head_to_head"] = h2h

    return matches


def add_venue_features(matches, team_results):
    matches = matches.sort_values("start_date").reset_index(drop=True)

    venue_map = matches.set_index("match_id")["venue"].to_dict()
    team_results = team_results.copy()
    team_results["venue"] = team_results["match_id"].map(venue_map)

    venue_bf_wr, venue_t1_wr = [], []

    for _, row in matches.iterrows():
        date, venue = row["start_date"], row["venue"]
        t1 = row["team1"]

        past = team_results[
            (team_results["venue"] == venue) &
            (team_results["start_date"] < date)
        ]

        bf = past[past["batting_first"] == 1]
        venue_bf_wr.append(bf["did_win"].mean() if len(bf) > 0 else 0.5)

        t1_v = past[past["team"] == t1]
        venue_t1_wr.append(t1_v["did_win"].mean() if len(t1_v) > 0 else 0.5)

    matches["venue_bat_first_wr"] = venue_bf_wr
    matches["venue_t1_wr"] = venue_t1_wr

    return matches


def add_toss(matches):
    """
    Toss info is already in match_features.csv (from _info files).
    We just encode it.
    """
    if "toss_winner" not in matches.columns or "toss_decision" not in matches.columns:
        raise ValueError("toss_winner or toss_decision missing from match_features.csv")

    matches["toss_is_team1"] = (matches["toss_winner"] == matches["team1"]).astype(int)
    matches["toss_decision_enc"] = matches["toss_decision"].map({"bat": 0, "field": 1}).fillna(0).astype(int)

    return matches


def build_dataset():
    print("Loading...")
    master, matches = load_base()

    print("Building team results...")
    team_results = build_team_results(master)

    print("Computing ELO...")
    matches = compute_elo(matches)

    print("Adding rolling form...")
    matches = add_rolling_form(matches, team_results)

    print("Adding head-to-head...")
    matches = add_head_to_head(matches, team_results)

    print("Adding venue features...")
    matches = add_venue_features(matches, team_results)

    print("Adding toss...")
    matches = add_toss(matches)

    matches["elo_diff"] = matches["team1_elo"] - matches["team2_elo"]
    matches["form_diff"] = matches["team1_form"] - matches["team2_form"]
    matches["sr_diff"] = matches["team1_bat_sr"] - matches["team2_bat_sr"]

    out = matches[[
        "match_id", "season", "start_date", "team1", "team2", "venue",
        "team1_won",
        "team1_elo", "team2_elo", "elo_diff",
        "team1_form", "team2_form", "form_diff",
        "team1_bat_sr", "team2_bat_sr", "sr_diff",
        "head_to_head",
        "venue_bat_first_wr", "venue_t1_wr",
        "toss_is_team1", "toss_decision_enc"
    ]]

    out_path = Path(PROCESSED_DIR) / "pre_match_clean.csv"
    out.to_csv(out_path, index=False)
    print(f"✅ Saved: {out_path}")
    print(f"Shape: {out.shape}")

    return out


if __name__ == "__main__":
    df = build_dataset()
    print(df.head())