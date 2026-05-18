from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "pre_match_with_player_quality.csv"
MODEL_PATH = ROOT / "models" / "honest_model.joblib"
FEATURE_PATH = ROOT / "models" / "honest_feature_columns.json"

FEATURES = [
    "team1_elo",
    "team2_elo",
    "elo_diff",
    "team1_form",
    "team2_form",
    "form_diff",
    "head_to_head",
    "venue_t1_wr",
    "venue_bat_first_wr",
    "toss_is_team1",
    "toss_decision_enc",
    "team1_batting_quality",
    "team2_batting_quality",
    "team1_bowling_quality",
    "team2_bowling_quality",
]

INVALID_WINNERS = {
    "",
    "nan",
    "none",
    "draw",
    "no result",
    "tie",
    "abandoned",
    "cancelled",
}


def main() -> None:
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded -> {DATA_PATH.name} ({len(df):,} rows)")

    wnorm = df["winner"].astype(str).str.strip().str.lower()
    df = df[~wnorm.isin(INVALID_WINNERS)].copy()
    df["target"] = (df["winner"].astype(str).str.strip() == df["team1"].astype(str).str.strip()).astype(int)

    df = df.dropna(subset=FEATURES)
    train = df[df["season"] < 2023]
    test = df[df["season"] >= 2023]

    X_train = train[FEATURES]
    y_train = train["target"]
    X_test = test[FEATURES]
    y_test = test["target"]

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=5,
        random_state=42,
    )
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(y_test, preds)
    baseline = max(y_test.mean(), 1 - y_test.mean())
    brier = brier_score_loss(y_test, probs)
    ll = log_loss(y_test, probs)

    print("\n-- Final evaluation (season >= 2023 holdout) --")
    print(f"Accuracy  : {acc:.2%}")
    print(f"Baseline  : {baseline:.2%}")
    print(f"Lift      : {(acc - baseline):+.2%}")
    print(f"Brier     : {brier:.4f}  (lower is better)")
    print(f"Log loss  : {ll:.4f}  (lower is better)")

    joblib.dump(model, MODEL_PATH)
    FEATURE_PATH.write_text(json.dumps(FEATURES, indent=2), encoding="utf-8")

    print("\nSaved model.")
    print(MODEL_PATH)
    print(f"Saved features -> {FEATURE_PATH}")


if __name__ == "__main__":
    main()
