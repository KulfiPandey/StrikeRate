# pipeline/processor.py
# ──────────────────────────────────────────────────────
# What this file does:
#   Loops through ALL 2352 match files, parses each one,
#   merges ball-by-ball data with match metadata,
#   and saves one big clean master CSV.
#
# Why this matters:
#   Right now data is spread across 2352 files.
#   After this script runs, you have ONE dataframe
#   with every IPL delivery ever bowled + match context.
#   That's your foundation for every ML model we build.
# ──────────────────────────────────────────────────────

import os
import pandas as pd
from tqdm import tqdm
from pipeline.ingest import parse_info_file
from config import RAW_DIR, PROCESSED_DIR

def build_master_dataset():
    print("🔄 Building master dataset from all matches...")
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    all_deliveries = []
    match_files = [f for f in os.listdir(RAW_DIR) 
                   if f.endswith('.csv') and '_info' not in f
                   and f != 'ipl_csv2.zip']

    print(f"📁 Found {len(match_files)} match files to process")

    for match_file in tqdm(match_files, desc="Processing matches"):
        match_id = match_file.replace('.csv', '')
        ball_path = os.path.join(RAW_DIR, match_file)
        info_path = os.path.join(RAW_DIR, f"{match_id}_info.csv")

        if not os.path.exists(info_path):
            continue

        try:
            # Load ball by ball data
            df = pd.read_csv(ball_path)

            # Parse match metadata
            info = parse_info_file(info_path)

            # Attach metadata columns to every delivery row
            df['venue']          = info.get('venue', '')
            df['city']           = info.get('city', '')
            df['toss_winner']    = info.get('toss_winner', '')
            df['toss_decision']  = info.get('toss_decision', '')
            df['match_winner']   = info.get('winner', '')
            df['winner_runs']    = info.get('winner_runs', '')
            df['winner_wickets'] = info.get('winner_wickets', '')
            df['player_of_match']= info.get('player_of_match', '')

            all_deliveries.append(df)

        except Exception as e:
            print(f"⚠️ Skipped {match_file}: {e}")
            continue

    print("\n🔗 Merging all matches...")
    master_df = pd.concat(all_deliveries, ignore_index=True)

    output_path = os.path.join(PROCESSED_DIR, "master_deliveries.csv")
    master_df.to_csv(output_path, index=False)

    print(f"✅ Master dataset saved!")
    print(f"📊 Total deliveries: {len(master_df):,}")
    print(f"📅 Seasons covered: {sorted(master_df['season'].astype(str).unique())}")
    print(f"🏟️  Venues: {master_df['venue'].nunique()}")
    print(f"🏏 Unique batters: {master_df['striker'].nunique()}")
    print(f"🎳 Unique bowlers: {master_df['bowler'].nunique()}")
    print(f"💾 Saved to: {output_path}")

    return master_df

if __name__ == "__main__":
    df = build_master_dataset()