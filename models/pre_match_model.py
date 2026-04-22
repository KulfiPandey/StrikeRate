from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import PROCESSED_DIR

try:
    import joblib  # type: ignore
except Exception:  # pragma: no cover
    joblib = None


DATA_PATH = Path(PROCESSED_DIR) / "pre_match_clean.csv"
MODEL_DIR = Path(PROCESSED_DIR) / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


NUMERIC_COLS = [
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

CATEGORICAL_COLS = ["team1", "team2", "venue"]


@dataclass(frozen=True)
class EvalRow:
    start_date: str
    season: int
    match_id: str
    team1: str
    team2: str
    y_true: int
    p_team1_win: float
    y_pred: int


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}. Run: python strike.py pipeline")
    df = pd.read_csv(path)
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df = df.dropna(subset=["start_date"]).sort_values("start_date").reset_index(drop=True)
    df["season"] = df["season"].astype(int)
    df["match_id"] = df["match_id"].astype(str)
    df["team1_won"] = df["team1_won"].astype(int)
    for c in CATEGORICAL_COLS:
        if c in df.columns:
            df[c] = df[c].fillna("UNKNOWN").astype(str)
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def make_pipeline() -> Pipeline:
    numeric = NUMERIC_COLS
    categorical = CATEGORICAL_COLS

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
            ("num", "passthrough", numeric),
        ],
        remainder="drop",
    )

    # Strong baseline for tabular, fast, and robust to non-linearities.
    base = HistGradientBoostingClassifier(
        random_state=42,
        max_depth=4,
        learning_rate=0.08,
        max_iter=600,
    )

    # Calibrate probabilities using time-series CV (important for odds/value betting).
    tscv = TimeSeriesSplit(n_splits=6)
    calibrated = CalibratedClassifierCV(base, method="isotonic", cv=tscv)

    return Pipeline([("pre", pre), ("model", calibrated)])


def _metrics(y_true: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    y_pred = (p >= 0.5).astype(int)
    out: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "brier": float(brier_score_loss(y_true, p)),
        "logloss": float(log_loss(y_true, np.c_[1 - p, p])),
    }
    try:
        out["auc"] = float(roc_auc_score(y_true, p))
    except Exception:
        out["auc"] = float("nan")
    out["baseline_acc"] = float(max(y_true.mean(), 1 - y_true.mean()))
    return out


def walk_forward_eval(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Walk-forward: for each season, train on all earlier seasons; predict that season.
    Produces a leakage-safe probability series you can use for backtests.
    """
    seasons = sorted(df["season"].unique())
    eval_rows: list[EvalRow] = []

    for s in seasons[2:]:
        train = df[df["season"] < s].copy()
        test = df[df["season"] == s].copy()
        if len(test) < 5 or len(train) < 50:
            continue

        pipe = make_pipeline()
        X_tr = train[CATEGORICAL_COLS + NUMERIC_COLS]
        y_tr = train["team1_won"].to_numpy()
        X_te = test[CATEGORICAL_COLS + NUMERIC_COLS]
        y_te = test["team1_won"].to_numpy()

        pipe.fit(X_tr, y_tr)
        p = pipe.predict_proba(X_te)[:, 1]
        y_pred = (p >= 0.5).astype(int)

        for i in range(len(test)):
            r = test.iloc[i]
            eval_rows.append(
                EvalRow(
                    start_date=str(r["start_date"].date()),
                    season=int(r["season"]),
                    match_id=str(r["match_id"]),
                    team1=str(r["team1"]),
                    team2=str(r["team2"]),
                    y_true=int(y_te[i]),
                    p_team1_win=float(p[i]),
                    y_pred=int(y_pred[i]),
                )
            )

    pred_df = pd.DataFrame([r.__dict__ for r in eval_rows])
    metrics = _metrics(pred_df["y_true"].to_numpy(), pred_df["p_team1_win"].to_numpy()) if len(pred_df) else {}
    return pred_df, metrics


def train_final(df: pd.DataFrame, train_until_season: int | None = None) -> Pipeline:
    """
    Train a deployable model on all data up to `train_until_season` (default: max season).
    """
    if train_until_season is None:
        train_until_season = int(df["season"].max())
    train = df[df["season"] <= train_until_season].copy()
    pipe = make_pipeline()
    X = train[CATEGORICAL_COLS + NUMERIC_COLS]
    y = train["team1_won"].to_numpy()
    pipe.fit(X, y)
    return pipe


def save_artifacts(model: Pipeline, eval_preds: pd.DataFrame, metrics: dict[str, Any]) -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {
        "created_at": ts,
        "dataset": str(DATA_PATH),
        "n_eval": int(len(eval_preds)),
        "metrics": metrics,
        "numeric_cols": NUMERIC_COLS,
        "categorical_cols": CATEGORICAL_COLS,
    }
    (MODEL_DIR / "pre_match_metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    eval_preds.to_csv(MODEL_DIR / "pre_match_walkforward_predictions.csv", index=False)

    if joblib is None:
        print("joblib not available; skipping model save.")
        return
    joblib.dump(model, MODEL_DIR / "pre_match_model.joblib")


def main():
    df = load_data()
    preds, metrics = walk_forward_eval(df)
    print("\nPre-match model (walk-forward, calibrated)")
    if metrics:
        for k in ["accuracy", "baseline_acc", "auc", "brier", "logloss"]:
            if k in metrics:
                print(f"{k:>12}: {metrics[k]:.4f}" if isinstance(metrics[k], float) else f"{k:>12}: {metrics[k]}")
    else:
        print("Not enough data to evaluate.")

    model = train_final(df)
    save_artifacts(model, preds, metrics)
    print(f"\nSaved artifacts to: {MODEL_DIR}")


if __name__ == "__main__":
    main()

