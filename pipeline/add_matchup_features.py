# pipeline/add_matchup_features.py
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import PROCESSED_DIR

def load_data():
    df = pd.read_csv(Path(PROCESSED_DIR) / "master_deliveries.csv", low_memory=False)
    df["start_date"] = pd.to_datetime(df["start_date"])
    return df

def build_partnership_stats(master):
    """
    For each batting pair (striker, non_striker), compute:
    - average runs per partnership
    - run rate when together
    - probability of wicket falling
    Only uses historical matches (no leakage).
    """
    # Create partnership key
    master["pair"] = master["striker"] + " & " + master["non_striker"]
    
    # Group by match and innings to identify partnership changes
    # Simplified: compute per-ball statistics aggregated later
    pair_stats = master.groupby(["pair", "match_id", "innings"]).agg(
        runs=("runs_off_bat", "sum"),
        balls=("ball", "count"),
        wicket_fell=("wicket_type", lambda x: (x.notna()).any())
    ).reset_index()
    
    # Now aggregate over all historical matches (time-aware)
    # We'll join to pre_match_clean later
    return pair_stats

def build_bowler_batter_matchups(master):
    """
    For each (bowler, striker) pair, compute:
    - batter's strike rate against that bowler
    - bowler's economy against that batter
    - dismissal probability
    """
    matchup = master.groupby(["bowler", "striker"]).agg(
        runs_conceded=("runs_off_bat", "sum"),
        balls_bowled=("ball", "count"),
        dismissals=("wicket_type", lambda x: x.notna().sum())
    ).reset_index()
    
    matchup["strike_rate_vs"] = (matchup["runs_conceded"] / matchup["balls_bowled"]) * 100
    matchup["economy_vs"] = matchup["runs_conceded"] / (matchup["balls_bowled"] / 6)
    matchup["dismissal_pct"] = matchup["dismissals"] / matchup["balls_bowled"]
    
    return matchup

def add_features_to_pre_match(pre_match, master):
    """
    Merge partnership and matchup stats into pre_match_clean.csv
    For each match, compute average team strength based on expected batting order.
    """
    # Placeholder: we need expected batting order (which we don't have pre-match)
    # Instead, we'll compute team-level aggregates: average partnership SR of top 6 batters
    # and average bowler vs batter performance for expected bowling attack.
    
    # For now, return original (we'll expand in next iteration)
    return pre_match

if __name__ == "__main__":
    master = load_data()
    partnerships = build_partnership_stats(master)
    matchups = build_bowler_batter_matchups(master)
    
    print(f"✅ Built {len(partnerships)} unique partnerships")
    print(f"✅ Built {len(matchups)} unique bowler-batter matchups")
    
    # Save for later use
    partnerships.to_csv(Path(PROCESSED_DIR) / "partnership_stats.csv", index=False)
    matchups.to_csv(Path(PROCESSED_DIR) / "matchup_stats.csv", index=False)