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

# ── API Keys (we'll fill these later) ──────────────────
ANTHROPIC_API_KEY = ""
OPENWEATHER_API_KEY = ""