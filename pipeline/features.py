# pipeline/features.py
# ──────────────────────────────────────────────────────
# What this file does:
#   Takes the raw 279k delivery dataset and engineers
#   meaningful features that the ML model can learn from.
#
# Think of it like this:
#   Raw data = every at-bat in cricket history
#   Features = "Kohli averages 142 SR in powerplay"
#   The model learns from features, not raw balls.
# ──────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import os
from config import PROCESSED_DIR

def load_master():
    path = os.path.join(PROCESSED_DIR, "master_deliveries.csv")
    df = pd.read_csv(path, low_memory=False)
    # Fix mixed types
    df['season'] = df['season'].astype(str)
    df['match_id'] = df['match_id'].astype(str)
    print(f"✅ Loaded {len(df):,} deliveries")
    return df

def add_match_phase(df):
    """
    What: Labels every delivery with its match phase.
    Why: Cricket has 3 distinct phases with different strategies.
         Powerplay (overs 1-6): fielding restrictions, aggressive batting
         Middle (overs 7-15): consolidation, building platform  
         Death (overs 16-20): maximum aggression, big hits
    """
    def get_phase(over):
        if over <= 6:
            return 'powerplay'
        elif over <= 15:
            return 'middle'
        else:
            return 'death'
    
    # Extract over number from ball column (e.g. 3.2 → over 3)
    df['over_number'] = df['ball'].apply(lambda x: int(str(x).split('.')[0]) + 1)
    df['phase'] = df['over_number'].apply(get_phase)
    print("✅ Match phases added (powerplay / middle / death)")
    return df

def compute_batter_stats(df):
    """
    What: Computes each batter's performance stats by phase.
    Why: A batter who averages 160 SR in death overs is far more
         valuable than one who averages 90. The model needs to know this.
    
    Output: One row per batter per phase with runs, balls, SR, dismissals.
    """
    # Total runs per batter per phase
    batter_stats = df.groupby(['striker', 'phase']).agg(
        runs_scored    = ('runs_off_bat', 'sum'),
        balls_faced    = ('runs_off_bat', 'count'),
        dismissals     = ('player_dismissed', lambda x: x.notna().sum())
    ).reset_index()

    # Strike rate = (runs / balls) * 100
    batter_stats['strike_rate'] = (
        batter_stats['runs_scored'] / batter_stats['balls_faced'] * 100
    ).round(2)

    # Average = runs per dismissal (how many runs before getting out)
    batter_stats['average'] = (
        batter_stats['runs_scored'] / batter_stats['dismissals'].replace(0, 1)
    ).round(2)

    print(f"✅ Batter stats computed: {len(batter_stats):,} batter-phase combinations")
    return batter_stats

def compute_bowler_stats(df):
    """
    What: Computes each bowler's economy and wicket rate by phase.
    Why: Economy rate tells you how many runs a bowler concedes per over.
         A bowler with 6.5 economy in death overs is elite.
    """
    bowler_stats = df.groupby(['bowler', 'phase']).agg(
        runs_conceded  = ('runs_off_bat', 'sum'),
        balls_bowled   = ('runs_off_bat', 'count'),
        wickets        = ('player_dismissed', lambda x: x.notna().sum())
    ).reset_index()

    # Economy = runs per over (every 6 balls)
    bowler_stats['economy'] = (
        bowler_stats['runs_conceded'] / bowler_stats['balls_bowled'] * 6
    ).round(2)

    # Wickets per match (roughly per 6 overs)
    bowler_stats['wicket_rate'] = (
        bowler_stats['wickets'] / bowler_stats['balls_bowled'] * 6
    ).round(3)

    print(f"✅ Bowler stats computed: {len(bowler_stats):,} bowler-phase combinations")
    return bowler_stats

def compute_venue_stats(df):
    """
    What: Computes win patterns at each venue.
    Why: Some grounds heavily favour chasing (batting second).
         The model needs to know venue context for predictions.
    """
    # Get one row per match (not per delivery)
    matches = df.drop_duplicates('match_id')[
        ['match_id', 'venue', 'toss_decision', 'match_winner', 'batting_team']
    ].copy()

    venue_stats = matches.groupby('venue').agg(
        total_matches  = ('match_id', 'count'),
        chased_count   = ('toss_decision', lambda x: (x == 'field').sum())
    ).reset_index()

    venue_stats['chase_preference'] = (
        venue_stats['chased_count'] / venue_stats['total_matches'] * 100
    ).round(1)

    print(f"✅ Venue stats computed: {len(venue_stats)} venues")
    return venue_stats

def compute_toss_impact(df):
    """
    What: Does winning the toss actually help you win the match?
    Why: Teams make toss decisions (bat/field) based on conditions.
         If toss winners win 60%+ of matches at a venue, that's
         a strong signal the model should factor in.
    """
    matches = df.drop_duplicates('match_id')[
        ['match_id', 'venue', 'toss_winner', 'toss_decision', 'match_winner']
    ].copy()

    matches['toss_won_match'] = (
        matches['toss_winner'] == matches['match_winner']
    ).astype(int)

    toss_impact = matches.groupby('venue').agg(
        total_matches     = ('match_id', 'count'),
        toss_winner_wins  = ('toss_won_match', 'sum')
    ).reset_index()

    toss_impact['toss_win_pct'] = (
        toss_impact['toss_winner_wins'] / toss_impact['total_matches'] * 100
    ).round(1)

    print(f"✅ Toss impact computed across {len(toss_impact)} venues")
    return toss_impact

def compute_season_trends(df):
    """
    What: How has scoring changed across IPL seasons?
    Why: IPL has evolved massively — pitches, rules, player quality.
         2024 scoring is very different from 2008.
         The model needs to weight recent seasons more.
    """
    season_stats = df.groupby('season').agg(
        total_runs    = ('runs_off_bat', 'sum'),
        total_balls   = ('runs_off_bat', 'count'),
        total_wickets = ('player_dismissed', lambda x: x.notna().sum()),
        total_matches = ('match_id', 'nunique')
    ).reset_index()

    season_stats['avg_run_rate'] = (
        season_stats['total_runs'] / season_stats['total_balls'] * 6
    ).round(2)

    season_stats['runs_per_match'] = (
        season_stats['total_runs'] / season_stats['total_matches']
    ).round(1)

    print(f"✅ Season trends computed across {season_stats['season'].nunique()} seasons")
    return season_stats

def save_features(batter_stats, bowler_stats, venue_stats, toss_impact, season_trends):
    batter_stats.to_csv(os.path.join(PROCESSED_DIR, "batter_features.csv"), index=False)
    bowler_stats.to_csv(os.path.join(PROCESSED_DIR, "bowler_features.csv"), index=False)
    venue_stats.to_csv(os.path.join(PROCESSED_DIR, "venue_features.csv"), index=False)
    toss_impact.to_csv(os.path.join(PROCESSED_DIR, "toss_impact.csv"), index=False)
    season_trends.to_csv(os.path.join(PROCESSED_DIR, "season_trends.csv"), index=False)
    print("💾 All feature files saved to data/processed/")

if __name__ == "__main__":
    df = load_master()
    df = add_match_phase(df)

    batter_stats  = compute_batter_stats(df)
    bowler_stats  = compute_bowler_stats(df)
    venue_stats   = compute_venue_stats(df)
    toss_impact   = compute_toss_impact(df)
    season_trends = compute_season_trends(df)

    save_features(batter_stats, bowler_stats, venue_stats, toss_impact, season_trends)

    print("\n🏏 Top 5 death over batters (min 50 balls):")
    death_batters = batter_stats[
        (batter_stats['phase'] == 'death') &
        (batter_stats['balls_faced'] > 50)
    ].sort_values('strike_rate', ascending=False)
    print(death_batters.head())

    print("\n📈 Season run rate trends:")
    print(season_trends[['season', 'avg_run_rate', 'runs_per_match']].to_string(index=False))