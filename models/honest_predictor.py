"""
models/honest_predictor.py
============================
Walk-forward IPL match predictor.

Changes from original:
  - XGBoost instead of GradientBoostingClassifier (faster, better missing-value handling)
  - OrdinalEncoder with handle_unknown for venues (no silent -1 crash)
  - Brier score + log loss alongside accuracy (honest probabilistic evaluation)
  - Player quality features from player_features.py
  - predict_proba output saved for backtest calibration

Run order:
    python -m pipeline.pre_matches_features
    python -m pipeline.player_features
    python -m models.honest_predictor
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report,
    brier_score_loss, log_loss
)
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import PROCESSED_DIR


FEATURES = [
    # Venue
    "venue_enc",
    # ELO (fixed: season-reset)
    "team1_elo", "team2_elo", "elo_diff",
    # Form (fixed: within-season)
    "team1_form", "team2_form", "form_diff",
    # H2H (fixed: last 3 seasons only)
    "head_to_head",
    # Venue win rates
    "venue_bat_first_wr", "venue_t1_wr",
    # Toss
    "toss_is_team1", "toss_decision_enc",
    # Rolling team batting quality (from player_features.py)
    "t1_bat_sr", "t2_bat_sr", "bat_sr_diff",
    "t1_pp_bat_sr", "t2_pp_bat_sr", "pp_bat_sr_diff",
    # Rolling team bowling quality
    "t1_bowl_bowl_eco", "t2_bowl_bowl_eco", "bowl_eco_diff",
    "t1_bowl_death_eco", "t2_bowl_death_eco", "death_eco_diff",
    "t1_bowl_wicket_rate", "t2_bowl_wicket_rate",
]


def load() -> pd.DataFrame:
    enriched = Path(PROCESSED_DIR) / "pre_match_with_players.csv"
    base = Path(PROCESSED_DIR) / "pre_match_clean.csv"

    if enriched.exists():
        df = pd.read_csv(enriched)
        print(f"Loaded {enriched.name} — {len(df)} matches, {df.shape[1]} columns")
    else:
        print("pre_match_with_players.csv not found. Run: python -m pipeline.player_features")
        print("Falling back to pre_match_clean.csv\n")
        df = pd.read_csv(base)
        player_cols = [
            "t1_bat_sr", "t2_bat_sr", "bat_sr_diff",
            "t1_pp_bat_sr", "t2_pp_bat_sr", "pp_bat_sr_diff",
            "t1_bowl_bowl_eco", "t2_bowl_bowl_eco", "bowl_eco_diff",
            "t1_bowl_death_eco", "t2_bowl_death_eco", "death_eco_diff",
            "t1_bowl_wicket_rate", "t2_bowl_wicket_rate",
        ]
        for c in player_cols:
            df[c] = 0.0

    df["start_date"] = pd.to_datetime(df["start_date"])
    df["season"] = df["season"].astype(int)
    df = df.sort_values("start_date").reset_index(drop=True)

    missing = [f for f in FEATURES if f not in df.columns and f != "venue_enc"]
    if missing:
        print(f"Missing columns (filling with 0): {missing}")
        for c in missing:
            df[c] = 0.0

    return df


def encode_venue(train, test):
    train, test = train.copy(), test.copy()
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    train["venue_enc"] = enc.fit_transform(train[["venue"]]).astype(int)
    test["venue_enc"] = enc.transform(test[["venue"]]).astype(int)
    return train, test


def get_features(df):
    return [f for f in FEATURES if f in df.columns]


def train_xgb(X_tr, y_tr):
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    model.fit(X_tr, y_tr)
    return model


def print_metrics(y_true, preds, probs):
    acc = accuracy_score(y_true, preds)
    base = max(y_true.mean(), 1 - y_true.mean())
    brier = brier_score_loss(y_true, probs)
    brier_base = brier_score_loss(y_true, np.full(len(y_true), y_true.mean()))
    ll = log_loss(y_true, probs)
    print(f"  Accuracy  : {acc:.2%}  (baseline {base:.2%}, lift {acc - base:+.2%})")
    print(f"  Brier     : {brier:.4f}  (baseline {brier_base:.4f}, lift {brier_base - brier:+.4f})")
    print(f"  Log-loss  : {ll:.4f}")


def rolling_window(df):
    seasons = sorted(df["season"].unique())
    print("\n── Rolling window ─────────────────────────────────────────────────────")
    lifts = []
    for i in range(3, len(seasons)):
        train = df[df["season"] < seasons[i]].copy()
        test = df[df["season"] == seasons[i]].copy()
        if len(test) < 5:
            continue
        train, test = encode_venue(train, test)
        feats = get_features(train)
        med = train[feats].median()
        model = train_xgb(train[feats].fillna(med), train["team1_won"])
        probs = model.predict_proba(test[feats].fillna(med))[:, 1]
        preds = (probs >= 0.5).astype(int)
        acc = accuracy_score(test["team1_won"], preds)
        base = max(test["team1_won"].mean(), 1 - test["team1_won"].mean())
        lift = acc - base
        lifts.append(lift)
        marker = "✅" if lift > 0 else "❌"
        print(f"  {seasons[i]} → acc {acc:.2%}  base {base:.2%}  lift {lift:+.2%} {marker}")
    wins = sum(1 for l in lifts if l > 0)
    print(f"\n  Seasons beating baseline: {wins}/{len(lifts)}  avg lift: {np.mean(lifts):+.2%}")


def run_final(df, lo, hi, label):
    train = df[(df["season"] >= lo) & (df["season"] <= hi)].copy() if lo else df[df["season"] <= hi].copy()
    test = df[df["season"] >= 2023].copy()
    train, test = encode_venue(train, test)
    feats = get_features(train)
    med = train[feats].median()
    X_tr = train[feats].fillna(med)
    X_te = test[feats].fillna(med)
    model = train_xgb(X_tr, train["team1_won"])
    probs = model.predict_proba(X_te)[:, 1]
    preds = (probs >= 0.5).astype(int)
    print_metrics(test["team1_won"], preds, probs)
    print(classification_report(test["team1_won"], preds, digits=2))
    imp = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)
    print("  Feature importances:")
    for feat, score in imp.head(15).items():
        bar = "█" * int(score * 50)
        print(f"    {feat:<30} {score:.4f}  {bar}")
    if label == "A":
        out = test[["match_id", "season", "start_date", "team1", "team2", "team1_won"]].copy()
        out["p_team1_win"] = probs
        out["y_true"] = test["team1_won"].values
        out_path = Path(PROCESSED_DIR) / "models" / "pre_match_walkforward_predictions.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False)
        print(f"\n  Predictions saved -> {out_path.name}")
    return model


def run_logistic(df):
    train = df[(df["season"] >= 2019) & (df["season"] <= 2022)].copy()
    test = df[df["season"] >= 2023].copy()
    train, test = encode_venue(train, test)
    feats = get_features(train)
    med = train[feats].median()
    X_tr = train[feats].fillna(med)
    X_te = test[feats].fillna(med)
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_tr_sc, train["team1_won"])
    probs = lr.predict_proba(X_te_sc)[:, 1]
    preds = lr.predict(X_te_sc)
    print_metrics(test["team1_won"], preds, probs)


if __name__ == "__main__":
    df = load()
    rolling_window(df)

    print("\n── Model A: train 2019-2022, test 2023+ ───────────────────────────────")
    run_final(df, lo=2019, hi=2022, label="A")

    print("\n── Model B: train all pre-2023, test 2023+ ────────────────────────────")
    run_final(df, lo=None, hi=2022, label="B")

    print("\n── Logistic regression sanity check ───────────────────────────────────")
    run_logistic(df)