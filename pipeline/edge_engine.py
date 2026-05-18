"""
Canonical edge pipeline: calibrated pre-match model vs Polymarket implied probs.

Golden path:
  load odds -> score with pre_match_model.joblib -> edge = p_model - p_market -> log
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from config import PROCESSED_DIR
from pipeline.team_name_standardizer import standardize_team_name

try:
    import joblib  # type: ignore
except Exception:
    joblib = None

ROOT = Path(__file__).parent.parent
ODDS_PATH = Path(PROCESSED_DIR) / "polymarket_match_odds.csv"
DATA_PATH = Path(PROCESSED_DIR) / "pre_match_with_player_quality.csv"
MODEL_PATH = ROOT / "models" / "honest_model.joblib"
LOG_DIR = Path(PROCESSED_DIR) / "edge_log"
LOG_PATH = LOG_DIR / "edges.csv"

# Must match models/honest_predictor.py
FEATURE_COLS = [
    "team1_elo", "team2_elo", "elo_diff",
    "team1_form", "team2_form", "form_diff",
    "head_to_head", "venue_t1_wr", "venue_bat_first_wr",
    "toss_is_team1", "toss_decision_enc",
    "team1_batting_quality", "team2_batting_quality",
    "team1_bowling_quality", "team2_bowling_quality",
]


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def load_odds(path: Path = ODDS_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing odds: {path}. Run: python strike.py fetch-odds")
    df = pd.read_csv(path)
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
    df["match_id"] = df["match_id"].astype(str)
    return df


def ensure_calibrated_model() -> Any:
    """Load honest_model.joblib (train with models.honest_predictor if missing)."""
    if MODEL_PATH.exists() and joblib is not None:
        return joblib.load(MODEL_PATH)
    if joblib is None:
        raise RuntimeError("joblib required. pip install joblib")
    raise FileNotFoundError(
        f"Missing {MODEL_PATH}. Run: python -m models.honest_predictor"
    )


def load_calibrated_model() -> Any:
    if joblib is None:
        raise RuntimeError("joblib required. pip install joblib")
    if not MODEL_PATH.exists():
        return ensure_calibrated_model()
    return joblib.load(MODEL_PATH)


def model_prob_team1_win(model, row: pd.Series) -> float:
    payload = {c: pd.to_numeric(row.get(c), errors="coerce") for c in FEATURE_COLS}
    X = pd.DataFrame([payload])
    p = float(model.predict_proba(X)[:, 1][0])
    return float(np.clip(p, 1e-6, 1 - 1e-6))


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
    start_dt = pd.to_datetime(m["start_date"], utc=True, errors="coerce")
    m["dt_diff"] = (start_dt - market_dt).abs()
    return m.sort_values("dt_diff").iloc[0]


def _market_timestamp(row: pd.Series) -> pd.Timestamp | None:
    for c in ["end_date", "event_start_time", "game_start_time"]:
        v = row.get(c)
        if isinstance(v, str) and v:
            dt = pd.to_datetime(v, utc=True, errors="coerce")
            if dt is not None and not pd.isna(dt):
                return dt
    return None


def _signal(team: str, edge: float, min_edge: float) -> str:
    if edge >= min_edge:
        return f"VALUE {team}"
    if edge <= -min_edge:
        return f"FADE {team}"
    return "NEUTRAL"


def build_edge_table(
    odds: pd.DataFrame,
    pre: pd.DataFrame,
    model,
    min_edge: float = 0.0,
) -> pd.DataFrame:
    rows = []
    for _, m in odds.iterrows():
        ta = str(m.get("team_a", ""))
        tb = str(m.get("team_b", ""))
        if not ta or not tb or ta == tb:
            continue

        best = _pick_match_by_time(pre, ta, tb, _market_timestamp(m))
        if best is None:
            continue

        p_team1 = model_prob_team1_win(model, best)
        team1_std = standardize_team_name(best.get("team1"))
        p_model_a = p_team1 if team1_std == ta else (1.0 - p_team1)

        p_market_a = _safe_float(m.get("prob_a_wins"), 0.5)
        p_market_b = _safe_float(m.get("prob_b_wins"), 0.5)
        edge_a = p_model_a - p_market_a
        edge_b = (1.0 - p_model_a) - p_market_b

        rows.append({
            "market_id": str(m.get("market_id", "")),
            "question": str(m.get("question", "")),
            "team_a": ta,
            "team_b": tb,
            "match_label": f"{ta} vs {tb}",
            "mapped_match_id": str(best.get("match_id", "")),
            "p_model_a": p_model_a,
            "p_market_a": p_market_a,
            "p_market_b": p_market_b,
            "edge_a": edge_a,
            "edge_b": edge_b,
            "signal": _signal(ta, edge_a, min_edge),
            "volume": _safe_float(m.get("volume"), 0.0),
            "fetch_timestamp": str(m.get("fetch_timestamp", "")),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["abs_edge"] = df[["edge_a", "edge_b"]].abs().max(axis=1)
    if min_edge > 0:
        df = df[df["abs_edge"] >= min_edge].copy()
    return df.sort_values("abs_edge", ascending=False).reset_index(drop=True)


def append_edge_log(scan_df: pd.DataFrame, scan_ts: str | None = None) -> Path:
    """Append scan rows to edge_log/edges.csv for forward testing / CLV."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = scan_ts or datetime.now(timezone.utc).isoformat()

    log = scan_df.copy()
    log.insert(0, "scan_timestamp", ts)
    log["result"] = ""  # fill after match: team_a_won / team_b_won / void

    cols = [
        "scan_timestamp", "market_id", "match_label", "team_a", "team_b",
        "mapped_match_id", "p_model_a", "p_market_a", "edge_a", "signal",
        "volume", "question", "result",
    ]
    cols = [c for c in cols if c in log.columns]
    log = log[cols]

    if LOG_PATH.exists():
        log.to_csv(LOG_PATH, mode="a", header=False, index=False)
    else:
        log.to_csv(LOG_PATH, index=False)
    return LOG_PATH


def format_scan_table(df: pd.DataFrame, min_edge: float) -> str:
    w = 52
    lines = [
        "=" * w,
        " StrikeRate - Market Edge Scanner",
        "=" * w,
        f" {'MATCH':<14} {'MODEL':>7} {'MARKET':>7} {'EDGE':>7}  SIGNAL",
        "-" * w,
    ]
    if df.empty:
        lines.append(" No markets above edge threshold.")
        lines.append(f" (min |edge| = {min_edge:.0%})")
    else:
        for _, r in df.iterrows():
            match = str(r["match_label"])[:14]
            model_p = f"{r['p_model_a']:.0%}"
            mkt_p = f"{r['p_market_a']:.0%}"
            edge = f"{r['edge_a']:+.0%}"
            sig = str(r["signal"])
            lines.append(f" {match:<14} {model_p:>7} {mkt_p:>7} {edge:>7}  {sig}")
    lines.append("=" * w)
    return "\n".join(lines)


def fetch_odds_subprocess() -> None:
    py = sys.executable
    script = ROOT / "pipeline" / "fetch_polymarket_odds.py"
    subprocess.check_call([py, str(script)], cwd=str(ROOT))


def run_scan(
    *,
    fetch: bool = True,
    min_edge: float = 0.05,
    train_if_missing: bool = True,
    log: bool = True,
    save_csv: bool = True,
) -> pd.DataFrame:
    if fetch:
        print("Fetching Polymarket odds...")
        fetch_odds_subprocess()

    if not MODEL_PATH.exists():
        if not train_if_missing:
            raise FileNotFoundError(
                f"Missing {MODEL_PATH}. Run: python strike.py train-prematch"
            )
        model = ensure_calibrated_model()
    else:
        model = load_calibrated_model()

    odds = load_odds()
    pre = load_pre_match()
    table = build_edge_table(odds, pre, model, min_edge=min_edge)

    scan_ts = datetime.now(timezone.utc).isoformat()
    print(format_scan_table(table, min_edge))
    print(f"\nMarkets scanned: {len(odds)}  |  Edges flagged: {len(table)}")
    print(f"Signal source: {MODEL_PATH.name} (honest pre-match)")

    if save_csv and len(table):
        out = Path(PROCESSED_DIR) / "value_bets.csv"
        table.to_csv(out, index=False)
        print(f"Saved: {out}")

    if log and len(table):
        path = append_edge_log(table, scan_ts)
        print(f"Appended edge log: {path}")

    meta = {
        "scan_timestamp": scan_ts,
        "n_markets": len(odds),
        "n_edges": len(table),
        "min_edge": min_edge,
        "model_path": str(MODEL_PATH),
    }
    (Path(PROCESSED_DIR) / "scan_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return table
