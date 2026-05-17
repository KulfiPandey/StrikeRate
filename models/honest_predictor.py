import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import PROCESSED_DIR


# Team-level aggregates (ELO, form, venue, toss) — ceiling ~50% alone.
TEAM_FEATURES = [
    "venue_enc",
    "team1_elo", "team2_elo", "elo_diff",
    "team1_form", "team2_form", "form_diff",
    "head_to_head",
    "venue_bat_first_wr", "venue_t1_wr",
    "toss_is_team1", "toss_decision_enc",
]

# Rolling player quality from ball-by-ball (pipeline/player_features.py).
PLAYER_FEATURES = [
    "team1_batting_quality", "team2_batting_quality",
    "team1_bowling_quality", "team2_bowling_quality",
    "batting_quality_diff", "bowling_quality_diff",
]

FEATURES = TEAM_FEATURES + PLAYER_FEATURES

PLAYER_DEFAULTS = {
    "team1_batting_quality": 120.0,
    "team2_batting_quality": 120.0,
    "team1_bowling_quality": 8.5,
    "team2_bowling_quality": 8.5,
    "batting_quality_diff": 0.0,
    "bowling_quality_diff": 0.0,
}


def _fill_player_defaults(df):
    for col, val in PLAYER_DEFAULTS.items():
        if col not in df.columns:
            df[col] = val
    return df


def load():
    """
    Prefer pre_match_with_player_quality.csv (pipeline/player_features.py).
    Falls back to pre_match_clean.csv with neutral player-quality defaults.
    """
    player_path = Path(PROCESSED_DIR) / "pre_match_with_player_quality.csv"
    base_path = Path(PROCESSED_DIR) / "pre_match_clean.csv"

    if player_path.exists():
        df = pd.read_csv(player_path)
        print(f"Loaded {player_path.name} — {len(df)} matches")
    else:
        print("WARNING: pre_match_with_player_quality.csv not found. Run:")
        print("    python -m pipeline.player_features")
        print("Using pre_match_clean.csv with default player-quality values.\n")
        df = pd.read_csv(base_path)
        df = _fill_player_defaults(df)

    df = _fill_player_defaults(df)
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["season"] = df["season"].astype(int)
    return df.sort_values("start_date").reset_index(drop=True)


def encode_venue(train, test):
    train = train.copy()
    test = test.copy()
    le = LabelEncoder()
    le.fit(train["venue"])
    train["venue_enc"] = le.transform(train["venue"])
    mapping = {k: v for k, v in zip(le.classes_, le.transform(le.classes_))}
    test["venue_enc"] = test["venue"].map(mapping).fillna(-1).astype(int)
    return train, test


def train_gbm(X_tr, y_tr, n_estimators=200):
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=10,
        random_state=42,
    )
    model.fit(X_tr, y_tr)
    return model


def _prepare_xy(train, test, features):
    train, test = encode_venue(train, test)
    med = train[features].median()
    X_tr = train[features].fillna(med)
    X_te = test[features].fillna(med)
    return X_tr, X_te, train, test


def evaluate(df):
    seasons = sorted(df["season"].unique())

    print("\n-- Rolling window (train all prior -> test one season) [team + player] --")
    _rolling_by_season(df, FEATURES, seasons)

    print("\n-- A/B on 2023+ holdout: team-only vs team + player quality --")
    _run_final(df, lo=2019, hi=2022, features=TEAM_FEATURES, label="team-only")
    _run_final(df, lo=2019, hi=2022, features=FEATURES, label="team + player")

    print("\n-- Final model: train 2019-2022, test 2023+ [team + player] --")
    model_a = _run_final(df, lo=2019, hi=2022, features=FEATURES, label=None)

    print("\n-- Final model: train all pre-2023, test 2023+ [team + player] --")
    _run_final(df, lo=None, hi=2022, features=FEATURES, label=None)

    print("\n-- Logistic regression sanity check (train 2019-2022, test 2023+) --")
    _run_logistic(df, features=FEATURES)

    return model_a


def _rolling_by_season(df, features, seasons):
    for i in range(3, len(seasons)):
        train = df[df["season"] < seasons[i]].copy()
        test = df[df["season"] == seasons[i]].copy()
        if len(test) < 5:
            continue

        X_tr, X_te, _, test = _prepare_xy(train, test, features)
        model = train_gbm(X_tr, train["team1_won"], n_estimators=100)
        acc = accuracy_score(test["team1_won"], model.predict(X_te))
        base = max(test["team1_won"].mean(), 1 - test["team1_won"].mean())
        marker = "[+]" if acc > base else "[-]"
        print(f"  {seasons[i]} -> {acc:.2%}  (baseline {base:.2%}) {marker}")


def _run_final(df, lo, hi, features, label=None):
    if lo is None:
        train = df[df["season"] <= hi].copy()
    else:
        train = df[(df["season"] >= lo) & (df["season"] <= hi)].copy()
    test = df[df["season"] >= 2023].copy()

    X_tr, X_te, train, test = _prepare_xy(train, test, features)
    model = train_gbm(X_tr, train["team1_won"])
    preds = model.predict(X_te)
    acc = accuracy_score(test["team1_won"], preds)
    base = max(test["team1_won"].mean(), 1 - test["team1_won"].mean())

    title = f"  [{label}]" if label else ""
    print(f"{title}")
    print(f"  Accuracy  : {acc:.2%}")
    print(f"  Baseline  : {base:.2%}")
    print(f"  Lift      : {acc - base:+.2%}")

    if label is None:
        print(classification_report(test["team1_won"], preds, digits=2))
        imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
        print("  Feature importances:")
        for feat, score in imp.items():
            bar = "#" * int(score * 40)
            print(f"    {feat:<28} {score:.4f}  {bar}")

    return model


def _run_logistic(df, features):
    train = df[(df["season"] >= 2019) & (df["season"] <= 2022)].copy()
    test = df[df["season"] >= 2023].copy()

    X_tr, X_te, train, test = _prepare_xy(train, test, features)
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)

    lr = LogisticRegression(max_iter=500, random_state=42)
    lr.fit(X_tr_sc, train["team1_won"])
    preds = lr.predict(X_te_sc)
    acc = accuracy_score(test["team1_won"], preds)
    base = max(test["team1_won"].mean(), 1 - test["team1_won"].mean())
    print(f"  Accuracy  : {acc:.2%}  (baseline {base:.2%})")


if __name__ == "__main__":
    df = load()
    evaluate(df)
