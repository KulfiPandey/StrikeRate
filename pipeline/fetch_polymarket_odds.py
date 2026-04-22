# pipeline/fetch_polymarket_odds.py (IMPROVED VERSION)
import asyncio
import aiohttp
import pandas as pd
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
import re

sys.path.append(str(Path(__file__).parent.parent))
from config import PROCESSED_DIR

GAMMA_API_URL = "https://gamma-api.polymarket.com"

async def fetch_all_markets(active: bool = True, limit: int = 500):
    url = f"{GAMMA_API_URL}/markets"
    params = {
        "limit": int(limit),
        "active": "true" if active else "false",
        "order": "volume",
        "ascending": "false",
    }
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as resp:
            if resp.status == 200:
                data = await resp.json()
                print(f"✅ Fetched {len(data)} markets.")
                return data
            print(f"❌ API Error: {resp.status}")
            return []

def extract_teams_from_question(question: str):
    # IPL team names (full and common abbreviations)
    team_pattern = r'(?i)(mumbai indians|royal challengers\s*bangalore|rcb|chennai super kings|csk|kolkata knight riders|kkr|rajasthan royals|rr|punjab kings|pbks|delhi capitals|dc|lucknow super giants|lsg|gujarat titans|gt|sunrisers hyderabad|srh)'
    matches = re.findall(team_pattern, question)
    # Remove duplicates and standardise
    unique = []
    for m in matches:
        m_clean = m.strip().upper()
        if m_clean not in unique:
            unique.append(m_clean)
    if len(unique) >= 2:
        return unique[0], unique[1]
    return None, None

async def fetch_polymarket_data():
    print("🚀 Starting Polymarket IPL fetch...")
    all_markets = await fetch_all_markets(active=True, limit=500)
    if not all_markets:
        return

    processed = []
    for market in all_markets:
        question = market.get('question', '')
        # Must contain IPL or cricket keywords AND team names
        if not ('IPL' in question or 'Indian Premier League' in question or 'vs' in question):
            continue
        
        team_a, team_b = extract_teams_from_question(question)
        if not team_a or not team_b:
            continue

        outcome_prices_str = market.get('outcomePrices', '["0.5","0.5"]')
        outcomes_str = market.get("outcomes", '["A","B"]')
        try:
            prices = json.loads(outcome_prices_str)
            prob_a = float(prices[0]) if len(prices) > 0 else 0.5
            prob_b = float(prices[1]) if len(prices) > 1 else 0.5
        except:
            prob_a = prob_b = 0.5

        try:
            outcomes = json.loads(outcomes_str)
            outcome_a = outcomes[0] if len(outcomes) > 0 else ""
            outcome_b = outcomes[1] if len(outcomes) > 1 else ""
        except Exception:
            outcome_a = outcome_b = ""

        # Prefer market-level game start time/end date if present
        end_date = market.get("endDate") or market.get("endDateIso") or ""
        game_start_time = market.get("gameStartTime") or ""
        event_start_time = ""
        events = market.get("events") or []
        if isinstance(events, list) and len(events) > 0 and isinstance(events[0], dict):
            event_start_time = events[0].get("startTime") or events[0].get("endDate") or ""

        processed.append({
            'market_id': market.get('id'),
            'question': question,
            'slug': market.get('slug'),
            'team_a': team_a,
            'team_b': team_b,
            'prob_a_wins': prob_a,
            'prob_b_wins': prob_b,
            "outcome_a": outcome_a,
            "outcome_b": outcome_b,
            'volume': float(market.get('volume', 0)),
            "active": bool(market.get("active", True)),
            "closed": bool(market.get("closed", False)),
            "end_date": end_date,
            "game_start_time": game_start_time,
            "event_start_time": event_start_time,
            'fetch_timestamp': datetime.now(timezone.utc).isoformat(),
        })

    if not processed:
        print("No usable IPL match markets found.")
        return

    df = pd.DataFrame(processed)
    output_path = Path(PROCESSED_DIR) / "polymarket_match_odds.csv"
    df.to_csv(output_path, index=False)

    print(f"💾 Saved {len(df)} IPL match markets to {output_path}")
    print("\n📊 Sample:")
    print(df[['question', 'team_a', 'team_b', 'prob_a_wins', 'prob_b_wins']].head())

if __name__ == "__main__":
    asyncio.run(fetch_polymarket_data())