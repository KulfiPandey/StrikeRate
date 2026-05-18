"""
Canonical processed data paths and loaders.

Prefer Parquet (from ingest_cricsheet); fall back to legacy Kaggle CSVs.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

try:
    from config import DATA_DIR, PROCESSED_DIR, RAW_DIR
except Exception:
    DATA_DIR = ROOT / "data"
    RAW_DIR = DATA_DIR / "raw"
    PROCESSED_DIR = DATA_DIR / "processed"

RAW_DIR = Path(RAW_DIR)
PROCESSED_DIR = Path(PROCESSED_DIR)
RAW_CRICSHEET_DIR = RAW_DIR / "cricsheet"

MATCHES_PARQUET = PROCESSED_DIR / "matches.parquet"
DELIVERIES_PARQUET = PROCESSED_DIR / "deliveries.parquet"
MATCH_PLAYERS_PARQUET = PROCESSED_DIR / "match_players.parquet"

# Legacy compatibility (optional export)
LEGACY_DELIVERIES_CSV = RAW_DIR / "deliveries.csv"
LEGACY_MATCHES_CSV = RAW_DIR / "matches.csv"


def matches_path() -> Path:
    return MATCHES_PARQUET


def deliveries_path() -> Path:
    return DELIVERIES_PARQUET


def load_matches() -> pd.DataFrame:
    if MATCHES_PARQUET.exists():
        return pd.read_parquet(MATCHES_PARQUET)
    if LEGACY_MATCHES_CSV.exists():
        return pd.read_csv(LEGACY_MATCHES_CSV, low_memory=False)
    raise FileNotFoundError(
        f"No matches dataset. Run: python -m pipeline.ingest_cricsheet --download"
    )


def load_deliveries(columns: list[str] | None = None) -> pd.DataFrame:
    if DELIVERIES_PARQUET.exists():
        df = pd.read_parquet(DELIVERIES_PARQUET, columns=columns)
        return df
    if LEGACY_DELIVERIES_CSV.exists():
        return pd.read_csv(LEGACY_DELIVERIES_CSV, usecols=columns, low_memory=False)
    raise FileNotFoundError(
        f"No deliveries dataset. Run: python -m pipeline.ingest_cricsheet --download"
    )


def export_kaggle_compat(
    matches: pd.DataFrame | None = None,
    deliveries: pd.DataFrame | None = None,
) -> tuple[Path, Path]:
    """Write temporary Kaggle-style CSVs for legacy scripts."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    matches = matches if matches is not None else load_matches()
    deliveries = deliveries if deliveries is not None else load_deliveries()

    m = matches.copy()
    if "id" not in m.columns:
        m["id"] = m["match_id"]
    if "date" not in m.columns and "start_date" in m.columns:
        m["date"] = m["start_date"]

    m_out = m[["id", "season", "date", "team1", "team2", "venue", "winner"]].copy()
    m_out.to_csv(LEGACY_MATCHES_CSV, index=False)

    d = deliveries.copy()
    if "inning" not in d.columns and "innings" in d.columns:
        d["inning"] = d["innings"]
    if "batsman_runs" not in d.columns and "runs_off_bat" in d.columns:
        d["batsman_runs"] = d["runs_off_bat"]
    if "batter" not in d.columns and "striker" in d.columns:
        d["batter"] = d["striker"]

    keep = [
        c
        for c in [
            "match_id",
            "inning",
            "batting_team",
            "bowling_team",
            "ball",
            "batter",
            "bowler",
            "batsman_runs",
            "total_runs",
            "is_wicket",
        ]
        if c in d.columns
    ]
    d[keep].to_csv(LEGACY_DELIVERIES_CSV, index=False)
    return LEGACY_MATCHES_CSV, LEGACY_DELIVERIES_CSV
