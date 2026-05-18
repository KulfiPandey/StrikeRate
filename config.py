# StrikeRate - Project Configuration
# This file is the single source of truth for all settings

import os

# ── Paths ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR  = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# ── Data Source ────────────────────────────────────────
CRICSHEET_IPL_URL = "https://cricsheet.org/downloads/ipl_csv2.zip"
CRICSHEET_IPL_JSON_URL = "https://cricsheet.org/downloads/ipl_json.zip"

CRICSHEET_RAW_DIR = os.path.join(RAW_DIR, "cricsheet")
MATCHES_PARQUET = os.path.join(PROCESSED_DIR, "matches.parquet")
DELIVERIES_PARQUET = os.path.join(PROCESSED_DIR, "deliveries.parquet")
MATCH_PLAYERS_PARQUET = os.path.join(PROCESSED_DIR, "match_players.parquet")

# ── API Keys (we'll fill these later) ──────────────────
ANTHROPIC_API_KEY = ""
OPENWEATHER_API_KEY = ""