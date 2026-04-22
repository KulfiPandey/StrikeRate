from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import PROCESSED_DIR
from pipeline.team_name_standardizer import standardize_team_name

try:
    import joblib  # type: ignore
except Exception:  # pragma: no cover
    joblib = None


ODDS_PATH = Path(PROCESSED_DIR) / "polymarket_match_odds.csv"
DATA_PATH = Path(PROCESSED_DIR) / "pre_match_clean.csv"
MODEL_PATH = Path(PROCESSED_DIR) / "models" / "pre_match_model.joblib"


@dataclass(frozen=True)
class BetRow:
    market_id: str
    question: str
    team_a: str
    team_b: str
    p_a: float
    p_b: float
    model_p_a: float
    edge_a: float
    edge_b: float
    volume: float
    fetch_timestamp: str


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def load_odds(path: Path = ODDS_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing odds file: {path}. Run: python strike.py fetch-odds")
    df = pd.read_csv(path)
    # Standardize teams to the same short forms used in joins
    for c in ["team_a", "team_b"]:
        if c in df.columns:
            df[c] = df[c].astype(str).apply(standardize_team_name)
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
    return df


def load_pre_match(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}. Run: python strike.py pipeline")
    df = pd.read_csv(path)
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df = df.dropna(subset=["start_date"]).sort_values("start_date").reset_index(drop=True)
    df["team1_std"] = df["team1"].astype(str).apply(standardize_team_name)
    df["team2_std"] = df["team2"].astype(str).apply(standardize_team_name)
    return df


def _pick_best_match(pre: pd.DataFrame, ta: str, tb: str) -> pd.Series | None:
    """
    Map a market (team_a/team_b) to the *next* scheduled match in the dataset.
    Since your dataset is historical, we do a best-effort mapping by finding the
    most recent season's matchup and selecting the latest row.
    """
    m = pre[
        (
            ((pre["team1_std"] == ta) & (pre["team2_std"] == tb))
            | ((pre["team1_std"] == tb) & (pre["team2_std"] == ta))
        )
    ]
    if m.empty:
        return None
    return m.sort_values("start_date").iloc[-1]


def _pick_match_by_time(pre: pd.DataFrame, ta: str, tb: str, market_dt: pd.Timestamp | None) -> pd.Series | None:
    m = pre[
        (
            ((pre["team1_std"] == ta) & (pre["team2_std"] == tb))
            | ((pre["team1_std"] == tb) & (pre["team2_std"] == ta))
        )
    ].copy()
    if m.empty:
        return None
    if market_dt is None or pd.isna(market_dt):
        return m.sort_values("start_date").iloc[-1]
    # Choose closest match date to the market end time (usually match start/end day).
    # Ensure both sides are tz-aware (UTC) to avoid pandas tz mismatch errors.
    start_dt = pd.to_datetime(m["start_date"], utc=True, errors="coerce")
    m["dt_diff"] = (start_dt - market_dt).abs()
    return m.sort_values("dt_diff").iloc[0]


def load_model(path: Path = MODEL_PATH):
    if joblib is None:
        raise RuntimeError("joblib not available; cannot load model.")
    if not path.exists():
        raise FileNotFoundError(f"Missing model: {path}. Run: python -m models.pre_match_model")
    return joblib.load(path)


def model_prob_team1_win(model, row: pd.Series) -> float:
    cols = ["team1", "team2", "venue"] + [
        "team1_elo",
        "team2_elo",
        "elo_diff",
        "team1_form",
        "team2_form",
        "form_diff",
        "team1_bat_sr",
        "team2_bat_sr",
        "sr_diff",
        "head_to_head",
        "venue_bat_first_wr",
        "venue_t1_wr",
        "toss_is_team1",
        "toss_decision_enc",
    ]
    payload = {c: row.get(c) for c in cols}
    for c in ["team1", "team2", "venue"]:
        payload[c] = "UNKNOWN" if payload.get(c) is None or (isinstance(payload.get(c), float) and np.isnan(payload[c])) else str(payload[c])
    X = pd.DataFrame([payload])
    p = float(model.predict_proba(X)[:, 1][0])
    return float(np.clip(p, 1e-6, 1 - 1e-6))


def implied_odds_to_decimal(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    return 1.0 / p


def kelly_fraction(p: float, decimal_odds: float, kelly_scale: float = 0.25) -> float:
    """
    Kelly (fractional): f* = (bp - q)/b where b = odds-1.
    We scale it down (default 1/4 Kelly) for sanity.
    """
    b = decimal_odds - 1.0
    q = 1.0 - p
    f = (b * p - q) / b if b > 0 else 0.0
    f = max(0.0, f)
    return float(f * kelly_scale)


def build_value_table(odds: pd.DataFrame, pre: pd.DataFrame, model) -> pd.DataFrame:
    rows: list[BetRow] = []
    for _, m in odds.iterrows():
        ta = str(m.get("team_a", ""))
        tb = str(m.get("team_b", ""))
        if not ta or not tb or ta == tb:
            continue

        # Use the market timestamp to map to the correct match (not "latest historical matchup")
        market_dt = None
        for c in ["end_date", "event_start_time", "game_start_time"]:
            v = m.get(c)
            if isinstance(v, str) and v:
                market_dt = pd.to_datetime(v, utc=True, errors="coerce")
                if market_dt is not None and not pd.isna(market_dt):
                    break

        best = _pick_match_by_time(pre, ta, tb, market_dt)
        if best is None:
            continue

        # Model probability for team_a:
        # If team_a maps to team1 in the dataset row -> p(team_a)=p(team1_win)
        # Else -> p(team_a)=1-p(team1_win)
        p_team1 = model_prob_team1_win(model, best)
        team1_std = standardize_team_name(best.get("team1"))
        p_a = p_team1 if team1_std == ta else (1.0 - p_team1)

        pm_a = _safe_float(m.get("prob_a_wins", 0.5), 0.5)
        pm_b = _safe_float(m.get("prob_b_wins", 0.5), 0.5)

        # Edge = model - market implied prob (simple, interpretable)
        edge_a = p_a - pm_a
        edge_b = (1.0 - p_a) - pm_b

        rows.append(
            BetRow(
                market_id=str(m.get("market_id", "")),
                question=str(m.get("question", "")),
                team_a=ta,
                team_b=tb,
                p_a=pm_a,
                p_b=pm_b,
                model_p_a=float(p_a),
                edge_a=float(edge_a),
                edge_b=float(edge_b),
                volume=_safe_float(m.get("volume", 0.0), 0.0),
                fetch_timestamp=str(m.get("fetch_timestamp", "")),
            )
        )

    df = pd.DataFrame([r.__dict__ for r in rows])
    if df.empty:
        return df

    df["pm_a_decimal"] = df["p_a"].apply(implied_odds_to_decimal)
    df["pm_b_decimal"] = df["p_b"].apply(implied_odds_to_decimal)
    df["kelly_a"] = df.apply(lambda r: kelly_fraction(r["model_p_a"], r["pm_a_decimal"]), axis=1)
    df["kelly_b"] = df.apply(lambda r: kelly_fraction(1.0 - r["model_p_a"], r["pm_b_decimal"]), axis=1)

    df = df.sort_values(["edge_a", "volume"], ascending=[False, False]).reset_index(drop=True)
    return df


def main():
    odds = load_odds()
    pre = load_pre_match()
    model = load_model()

    table = build_value_table(odds, pre, model)
    out_path = Path(PROCESSED_DIR) / "value_bets.csv"
    table.to_csv(out_path, index=False)

    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_markets_in": int(len(odds)),
        "n_rows_out": int(len(table)),
        "odds_path": str(ODDS_PATH),
        "model_path": str(MODEL_PATH),
    }
    (Path(PROCESSED_DIR) / "value_bets_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved: {out_path}")
    if len(table):
        show = table[
            [
                "team_a",
                "team_b",
                "p_a",
                "model_p_a",
                "edge_a",
                "kelly_a",
                "volume",
            ]
        ].head(15)
        print("\nTop edges (team_a side):")
        print(show.to_string(index=False))
    else:
        print("No markets could be mapped to matches yet.")


if __name__ == "__main__":
    main()

