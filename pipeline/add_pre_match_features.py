# pipeline/add_pre_match_features.py (with team name standardization)
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import PROCESSED_DIR

def standardize_team_name(name):
    """Convert any team name variation to standard short form."""
    name = str(name).strip().lower()
    mapping = {
        'kolkata knight riders': 'KKR', 'kkr': 'KKR',
        'mumbai indians': 'MI', 'mi': 'MI',
        'chennai super kings': 'CSK', 'csk': 'CSK',
        'royal challengers bangalore': 'RCB', 'royal challengers': 'RCB', 'rcb': 'RCB',
        'rajasthan royals': 'RR', 'rr': 'RR',
        'punjab kings': 'PBKS', 'pbks': 'PBKS', 'kings xi punjab': 'KXIP', 'kxip': 'KXIP',
        'delhi capitals': 'DC', 'dc': 'DC', 'delhi daredevils': 'DD', 'dd': 'DD',
        'lucknow super giants': 'LSG', 'lsg': 'LSG',
        'gujarat titans': 'GT', 'gt': 'GT',
        'sunrisers hyderabad': 'SRH', 'srh': 'SRH',
        'deccan chargers': 'DEC', 'deccan': 'DEC',
        'kochi tuskers kerala': 'KTK', 'ktk': 'KTK',
        'pune warriors': 'PW', 'pw': 'PW',
        'gujarat lions': 'GL', 'gl': 'GL',
        'rising pune supergiant': 'RPS', 'rps': 'RPS',
    }
    if name in mapping:
        return mapping[name]
    for key, std in mapping.items():
        if key in name:
            return std
    return name.upper()

def load_data():
    master = pd.read_csv(Path(PROCESSED_DIR) / "master_deliveries.csv", low_memory=False)
    master["start_date"] = pd.to_datetime(master["start_date"])
    pre = pd.read_csv(Path(PROCESSED_DIR) / "pre_match_clean.csv")
    pre["start_date"] = pd.to_datetime(pre["start_date"])
    
    # Standardize team names in master
    master["batting_team_std"] = master["batting_team"].apply(standardize_team_name)
    master["bowling_team_std"] = master["bowling_team"].apply(standardize_team_name)
    master["striker_std"] = master["striker"]  # keep as is
    master["non_striker_std"] = master["non_striker"]
    
    # Standardize team names in pre (already in short form? but ensure)
    pre["team1_std"] = pre["team1"].apply(standardize_team_name)
    pre["team2_std"] = pre["team2"].apply(standardize_team_name)
    pre["venue_std"] = pre["venue"]  # venue no change
    
    return master, pre

def build_team_partnership_sr(master):
    master["pair"] = master["striker_std"] + "|" + master["non_striker_std"]
    
    partnership_stats = master.groupby(["match_id", "innings", "pair"]).agg(
        runs=("runs_off_bat", "sum"),
        balls=("ball", "count")
    ).reset_index()
    partnership_stats["sr"] = (partnership_stats["runs"] / partnership_stats["balls"]) * 100
    
    team_info = master[["match_id", "innings", "batting_team_std", "start_date"]].drop_duplicates()
    partnership_stats = partnership_stats.merge(team_info, on=["match_id", "innings"])
    
    team_match_avg_sr = partnership_stats.groupby(["match_id", "batting_team_std"]).agg(
        team_partnership_sr=("sr", "mean")
    ).reset_index()
    team_match_avg_sr.rename(columns={"batting_team_std": "team"}, inplace=True)
    
    return team_match_avg_sr

def build_team_matchup_advantage(master):
    master["legal_ball"] = ((master["wides"] == 0) & (master["noballs"] == 0)).astype(int)
    
    matchup_stats = master.groupby(["bowling_team_std", "batting_team_std"]).agg(
        total_runs=("runs_off_bat", "sum"),
        total_legal_balls=("legal_ball", "sum"),
        total_wickets=("wicket_type", lambda x: x.notna().sum())
    ).reset_index()
    
    mask = matchup_stats["total_legal_balls"] > 0
    matchup_stats["economy"] = np.where(
        mask,
        matchup_stats["total_runs"] / (matchup_stats["total_legal_balls"] / 6),
        8.5
    )
    matchup_stats["strike_rate"] = np.where(
        mask,
        (matchup_stats["total_runs"] / matchup_stats["total_legal_balls"]) * 100,
        120.0
    )
    matchup_stats["wicket_rate"] = np.where(
        mask,
        matchup_stats["total_wickets"] / matchup_stats["total_legal_balls"],
        0.02
    )
    matchup_stats.rename(columns={"bowling_team_std": "bowling_team", "batting_team_std": "batting_team"}, inplace=True)
    return matchup_stats[["bowling_team", "batting_team", "economy", "strike_rate", "wicket_rate"]]

def add_rolling_features(pre, master):
    team_match_sr = build_team_partnership_sr(master)
    matchup = build_team_matchup_advantage(master)
    
    pre = pre.sort_values("start_date").reset_index(drop=True)
    match_dates = master[["match_id", "start_date"]].drop_duplicates()
    
    t1_part_sr, t2_part_sr = [], []
    t1_vs_t2_econ, t1_vs_t2_sr = [], []
    
    for idx, row in pre.iterrows():
        date = row["start_date"]
        t1 = row["team1_std"]
        t2 = row["team2_std"]
        
        past_match_ids = match_dates[match_dates["start_date"] < date]["match_id"].unique()
        past_sr = team_match_sr[team_match_sr["match_id"].isin(past_match_ids)]
        
        t1_sr_vals = past_sr[past_sr["team"] == t1]["team_partnership_sr"]
        t2_sr_vals = past_sr[past_sr["team"] == t2]["team_partnership_sr"]
        t1_part_sr.append(t1_sr_vals.mean() if len(t1_sr_vals) > 0 else 120.0)
        t2_part_sr.append(t2_sr_vals.mean() if len(t2_sr_vals) > 0 else 120.0)
        
        econ_row = matchup[(matchup["bowling_team"] == t1) & (matchup["batting_team"] == t2)]
        t1_vs_t2_econ.append(econ_row["economy"].iloc[0] if len(econ_row) > 0 else 8.5)
        
        sr_row = matchup[(matchup["batting_team"] == t1) & (matchup["bowling_team"] == t2)]
        t1_vs_t2_sr.append(sr_row["strike_rate"].iloc[0] if len(sr_row) > 0 else 120.0)
    
    pre["team1_partnership_sr"] = t1_part_sr
    pre["team2_partnership_sr"] = t2_part_sr
    pre["team1_economy_vs_team2"] = t1_vs_t2_econ
    pre["team1_strike_rate_vs_team2"] = t1_vs_t2_sr
    
    return pre

def main():
    print("Loading data...")
    master, pre = load_data()
    print(f"Master: {len(master)} deliveries, Pre: {len(pre)} matches")
    
    print("Adding rolling partnership and matchup features...")
    pre_enhanced = add_rolling_features(pre, master)
    
    for col in ["team1_partnership_sr", "team2_partnership_sr", 
                "team1_economy_vs_team2", "team1_strike_rate_vs_team2"]:
        pre_enhanced[col] = pre_enhanced[col].replace([np.inf, -np.inf], np.nan)
        if "sr" in col or "strike_rate" in col:
            pre_enhanced[col] = pre_enhanced[col].fillna(120.0)
        else:
            pre_enhanced[col] = pre_enhanced[col].fillna(8.5)
    
    out_path = Path(PROCESSED_DIR) / "pre_match_with_matchups.csv"
    pre_enhanced.to_csv(out_path, index=False)
    print(f"✅ Saved to {out_path}")
    print("New columns added:", [c for c in pre_enhanced.columns if c not in pre.columns])
    sample_cols = ["team1", "team2", "team1_partnership_sr", "team1_economy_vs_team2", "team1_strike_rate_vs_team2"]
    print("\nSample (first 5 rows):")
    print(pre_enhanced[sample_cols].head())

if __name__ == "__main__":
    main()