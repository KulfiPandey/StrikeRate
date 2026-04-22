from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import PROCESSED_DIR


PRED_PATH = Path(PROCESSED_DIR) / "models" / "pre_match_walkforward_predictions.csv"
FEATURE_PATH = Path(PROCESSED_DIR) / "pre_match_clean.csv"
OUT_DIR = Path(PROCESSED_DIR) / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class BacktestConfig:
    starting_bankroll: float = 1000.0
    margin: float = 0.05  # synthetic bookmaker overround
    min_edge: float = 0.02  # only bet if model has >= this edge
    kelly_scale: float = 0.05  # conservative fractional kelly
    max_bet_frac: float = 0.02  # cap per-bet risk


def load_preds(path: Path = PRED_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing predictions: {path}. Run: python strike.py train-prematch")
    df = pd.read_csv(path)
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df = df.dropna(subset=["start_date"]).sort_values("start_date").reset_index(drop=True)
    df["y_true"] = df["y_true"].astype(int)
    df["p_team1_win"] = pd.to_numeric(df["p_team1_win"], errors="coerce").clip(1e-6, 1 - 1e-6)
    return df


def attach_features(preds: pd.DataFrame) -> pd.DataFrame:
    """
    Join in extra match context so our "synthetic book" can be different from the model.
    We use a simple ELO-only bookmaker line to create a realistic-ish baseline.
    """
    if not FEATURE_PATH.exists():
        return preds.assign(elo_diff=0.0)
    f = pd.read_csv(FEATURE_PATH, usecols=["match_id", "elo_diff"])
    preds = preds.copy()
    preds["match_id"] = preds["match_id"].astype(str)
    f["match_id"] = f["match_id"].astype(str)
    out = preds.merge(f, on="match_id", how="left")
    out["elo_diff"] = pd.to_numeric(out["elo_diff"], errors="coerce").fillna(0.0)
    return out


def _apply_margin(p: float, margin: float) -> float:
    # Simple symmetric overround: push probs toward 0.5 then renormalize.
    # Not a perfect book model, but good enough for ROI simulations without real closing lines.
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    q = 1.0 - p
    p_m = p * (1.0 + margin)
    q_m = q * (1.0 + margin)
    s = p_m + q_m
    return float(np.clip(p_m / s, 1e-6, 1 - 1e-6))


def elo_book_prob_team1(elo_diff: float) -> float:
    """
    Convert ELO difference to win probability (standard chess-style mapping).
    """
    return float(1.0 / (1.0 + 10.0 ** (-float(elo_diff) / 400.0)))


def decimal_odds_from_prob(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    return 1.0 / p


def kelly_fraction(p: float, decimal_odds: float) -> float:
    b = decimal_odds - 1.0
    q = 1.0 - p
    if b <= 0:
        return 0.0
    return max(0.0, (b * p - q) / b)


def simulate(df: pd.DataFrame, cfg: BacktestConfig) -> tuple[pd.DataFrame, dict]:
    bankroll = cfg.starting_bankroll
    peak = bankroll
    rows = []

    for _, r in df.iterrows():
        p = float(r["p_team1_win"])
        y = int(r["y_true"])

        # Synthetic bookmaker line (ELO-only) with margin
        p0 = elo_book_prob_team1(float(r.get("elo_diff", 0.0)))
        p_book_t1 = _apply_margin(p0, cfg.margin)
        p_book_t2 = _apply_margin(1.0 - p0, cfg.margin)
        o1 = decimal_odds_from_prob(p_book_t1)
        o2 = decimal_odds_from_prob(p_book_t2)

        # Choose side with bigger edge vs book
        edge_t1 = p - p_book_t1
        edge_t2 = (1.0 - p) - p_book_t2

        side = "team1" if edge_t1 >= edge_t2 else "team2"
        edge = edge_t1 if side == "team1" else edge_t2

        stake = 0.0
        pnl = 0.0

        if edge >= cfg.min_edge:
            if side == "team1":
                frac = kelly_fraction(p, o1) * cfg.kelly_scale
                frac = min(frac, cfg.max_bet_frac)
                stake = bankroll * frac
                win = y == 1
                pnl = stake * (o1 - 1.0) if win else -stake
            else:
                p2 = 1.0 - p
                frac = kelly_fraction(p2, o2) * cfg.kelly_scale
                frac = min(frac, cfg.max_bet_frac)
                stake = bankroll * frac
                win = y == 0
                pnl = stake * (o2 - 1.0) if win else -stake

        bankroll = bankroll + pnl
        peak = max(peak, bankroll)
        dd = (peak - bankroll) / peak if peak > 0 else 0.0

        rows.append(
            {
                "start_date": r["start_date"],
                "season": int(r["season"]),
                "match_id": str(r["match_id"]),
                "team1": str(r["team1"]),
                "team2": str(r["team2"]),
                "y_true": y,
                "p_team1_win": p,
                "book_p_team1": p_book_t1,
                "book_odds_team1": o1,
                "book_odds_team2": o2,
                "side": side,
                "edge": float(edge),
                "stake": float(stake),
                "pnl": float(pnl),
                "bankroll": float(bankroll),
                "drawdown": float(dd),
            }
        )

    bt = pd.DataFrame(rows)
    n_bets = int((bt["stake"] > 0).sum())
    roi = (bankroll - cfg.starting_bankroll) / cfg.starting_bankroll if cfg.starting_bankroll > 0 else 0.0
    hit = float((bt.loc[bt["stake"] > 0, "pnl"] > 0).mean()) if n_bets else float("nan")
    max_dd = float(bt["drawdown"].max()) if len(bt) else 0.0

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "starting_bankroll": cfg.starting_bankroll,
        "ending_bankroll": float(bankroll),
        "roi": float(roi),
        "n_bets": n_bets,
        "hit_rate": hit,
        "max_drawdown": max_dd,
        "cfg": cfg.__dict__,
    }
    return bt, summary


def calibration_bins(df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    d = df.copy()
    d["bin"] = pd.cut(d["p_team1_win"], bins=np.linspace(0, 1, n_bins + 1), include_lowest=True)
    out = (
        d.groupby("bin")
        .agg(
            n=("y_true", "count"),
            avg_pred=("p_team1_win", "mean"),
            avg_true=("y_true", "mean"),
        )
        .reset_index()
    )
    return out


def plot_calibration(cal: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect")
    ax.plot(cal["avg_pred"], cal["avg_true"], marker="o", linewidth=2.5, color="#2d6a4f", label="model")
    for _, r in cal.iterrows():
        ax.annotate(str(int(r["n"])), (r["avg_pred"], r["avg_true"]), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_title("Calibration (walk-forward predictions)")
    ax.set_xlabel("Average predicted P(team1 win)")
    ax.set_ylabel("Empirical win rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    out = OUT_DIR / "calibration_curve.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_bankroll(bt: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(bt["start_date"], bt["bankroll"], linewidth=2.0, color="#1d3557")
    ax.set_title("Backtest bankroll (synthetic book)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Bankroll")
    ax.grid(alpha=0.3)
    out = OUT_DIR / "backtest_bankroll.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main():
    df = attach_features(load_preds())
    cfg = BacktestConfig()
    bt, summary = simulate(df, cfg)

    bt_path = OUT_DIR / "backtest_trades.csv"
    bt.to_csv(bt_path, index=False)

    cal = calibration_bins(df, n_bins=10)
    cal_path = OUT_DIR / "calibration_bins.csv"
    cal.to_csv(cal_path, index=False)

    curve = plot_calibration(cal)
    bankroll_plot = plot_bankroll(bt)

    summary["artifacts"] = {
        "trades_csv": str(bt_path),
        "calibration_bins_csv": str(cal_path),
        "calibration_plot": str(curve),
        "bankroll_plot": str(bankroll_plot),
    }
    (OUT_DIR / "backtest_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Backtest summary:")
    for k in ["roi", "ending_bankroll", "n_bets", "hit_rate", "max_drawdown"]:
        print(f"{k:>14}: {summary[k]}")
    print(f"\nSaved: {OUT_DIR}")


if __name__ == "__main__":
    main()

