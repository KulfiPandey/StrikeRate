from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

try:
    from config import PROCESSED_DIR as _PROCESSED_DIR, MODELS_DIR as _MODELS_DIR
except Exception:
    _PROCESSED_DIR = ROOT / "data" / "processed"
    _MODELS_DIR = ROOT / "models"

PROCESSED_DIR = Path(_PROCESSED_DIR)
MODELS_DIR = Path(_MODELS_DIR)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "honest_model.joblib"
FEATURE_PATH = MODELS_DIR / "honest_feature_columns.json"
DEFAULT_INPUTS = [
    PROCESSED_DIR / "pre_match_with_player_quality.csv",
    PROCESSED_DIR / "pre_match_clean.csv",
    PROCESSED_DIR / "match_features.csv",
]

MARKET_COL_CANDIDATES = ["p_market", "polymarket_prob", "market_prob", "implied_prob"]


def load_input(path: str | None) -> pd.DataFrame:
    if path:
        return pd.read_csv(path, low_memory=False)

    for fp in DEFAULT_INPUTS:
        if fp.exists():
            return pd.read_csv(fp, low_memory=False)

    raise FileNotFoundError("No input CSV found. Run the pre-match pipeline first.")


def load_model_and_columns():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Run `python -m models.honest_predictor` first.")
    if not FEATURE_PATH.exists():
        raise FileNotFoundError(f"Feature list not found: {FEATURE_PATH}. Run `python -m models.honest_predictor` first.")

    model = joblib.load(MODEL_PATH)
    with open(FEATURE_PATH, "r", encoding="utf-8") as f:
        feature_columns = json.load(f)

    return model, feature_columns


def score_matches(df: pd.DataFrame) -> pd.DataFrame:
    from models.honest_predictor import build_matrix

    model, feature_columns = load_model_and_columns()
    X, _, meta, _ = build_matrix(df, fit_columns=feature_columns, require_target=False)

    p_model = model.predict_proba(X)[:, 1]
    scored = meta.copy()
    scored["p_model"] = p_model

    # Carry through market probabilities if present.
    for c in MARKET_COL_CANDIDATES:
        if c in df.columns:
            scored[c] = pd.to_numeric(df.loc[scored.index, c], errors="coerce")
            break

    return scored


def pick_market_col(df: pd.DataFrame) -> str | None:
    for c in MARKET_COL_CANDIDATES:
        if c in df.columns:
            return c
    return None


def build_edges(df: pd.DataFrame, threshold: float = 0.05) -> pd.DataFrame:
    market_col = pick_market_col(df)
    out = df.copy()

    if market_col is None:
        out["edge"] = np.nan
        out["recommendation"] = "NO MARKET"
        out["kelly_fraction"] = np.nan
        return out

    out[market_col] = pd.to_numeric(out[market_col], errors="coerce")
    out = out[out[market_col].notna()].copy()

    out["edge"] = out["p_model"] - out[market_col]
    out["abs_edge"] = out["edge"].abs()

    out["recommendation"] = "NO BET"
    out.loc[out["edge"] >= threshold, "recommendation"] = out.loc[out["edge"] >= threshold, "team1"].map(lambda x: f"VALUE {x}")
    out.loc[out["edge"] <= -threshold, "recommendation"] = out.loc[out["edge"] <= -threshold, "team2"].map(lambda x: f"VALUE {x}")

    # Simple Kelly on the recommended side using fair odds from the market probability.
    out["kelly_fraction"] = np.nan
    for idx, row in out.iterrows():
        if row["recommendation"] == "NO BET":
            continue

        if row["recommendation"].startswith("VALUE ") and row["recommendation"].endswith(str(row.get("team1", ""))):
            p = float(row["p_model"])
            market_p = float(row[market_col])
        else:
            p = 1.0 - float(row["p_model"])
            market_p = 1.0 - float(row[market_col])

        market_p = min(max(market_p, 1e-6), 1 - 1e-6)
        decimal_odds = 1.0 / market_p
        b = decimal_odds - 1.0
        q = 1.0 - p
        kelly = ((b * p) - q) / b if b > 0 else 0.0
        out.at[idx, "kelly_fraction"] = max(0.0, float(kelly))

    return out


def save_edge_log(df: pd.DataFrame, log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stamped = df.copy()
    stamped["scanned_at_utc"] = datetime.now(timezone.utc).isoformat()

    if log_path.exists():
        prev = pd.read_csv(log_path, low_memory=False)
        stamped = pd.concat([prev, stamped], ignore_index=True)

    stamped.to_csv(log_path, index=False)
    print(f"Saved edge log -> {log_path}")


def main():
    parser = argparse.ArgumentParser(description="StrikeRate edge scanner")
    parser.add_argument("--input", type=str, default=None, help="Input CSV with pre-match features")
    parser.add_argument("--threshold", type=float, default=0.05, help="Minimum absolute edge")
    parser.add_argument("--top", type=int, default=10, help="How many rows to show")
    parser.add_argument("--log", action="store_true", help="Append scan output to edge log CSV")
    parser.add_argument("--log-path", type=str, default=str(PROCESSED_DIR / "edge_log.csv"))
    args = parser.parse_args()

    df = load_input(args.input)
    scored = score_matches(df)
    edges = build_edges(scored, threshold=args.threshold)

    sort_col = "abs_edge" if "abs_edge" in edges.columns else "p_model"
    view_cols = [c for c in ["match_id", "season", "start_date", "team1", "team2", "venue", "p_model", "p_market", "edge", "recommendation", "kelly_fraction"] if c in edges.columns]
    ranked = edges.sort_values(sort_col, ascending=False).head(args.top)

    print("\n── Top scan results ──")
    print(ranked[view_cols].to_string(index=False))

    if args.log:
        save_edge_log(edges[view_cols], Path(args.log_path))


if __name__ == "__main__":
    main()