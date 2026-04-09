import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import PROCESSED_DIR

def load_and_prepare():
    df = pd.read_csv(Path(PROCESSED_DIR) / "match_features.csv")

    # Merge toss info from master deliveries
    master = pd.read_csv(Path(PROCESSED_DIR) / "master_deliveries.csv", low_memory=False)
    toss = (
        master.groupby("match_id")
        .first()
        .reset_index()[["match_id", "toss_winner", "toss_decision"]]
    )
    df = df.merge(toss, on="match_id", how="left")

    # Encode categorical columns
    le = LabelEncoder()
    for col in ["team1", "team2", "venue", "toss_winner", "toss_decision"]:
        if col in df.columns:
            df[col + "_enc"] = le.fit_transform(df[col].astype(str))

    # Toss advantage — did the toss winner bat first?
    df["toss_winner_is_team1"] = (df["toss_winner"] == df["team1"]).astype(int)

    return df

def build_features(df):
    feature_cols = [
        "team1_enc", "team2_enc", "venue_enc",
        "toss_winner_is_team1", "toss_decision_enc",
        "inn1_pp_runs", "inn1_mid_runs", "inn1_death_runs",
        "inn2_pp_runs", "inn2_mid_runs", "inn2_death_runs",
    ]
    # Only keep cols that exist
    feature_cols = [c for c in feature_cols if c in df.columns]
    return feature_cols

def time_split_evaluate(df, feature_cols):
    """
    Rolling window evaluation — train on past, predict next season.
    Mirrors how a real prediction system works.
    """
    seasons = sorted(df["season"].unique())
    results = []

    print("\n── Rolling window evaluation ──")
    print(f"{'Train up to':<15} {'Test season':<15} {'Matches':<10} {'Accuracy'}")
    print("─" * 55)

    # Need at least 3 seasons to train meaningfully
    for i in range(3, len(seasons)):
        train_seasons = seasons[:i]
        test_season   = seasons[i]

        train = df[df["season"].isin(train_seasons)]
        test  = df[df["season"] == test_season]

        if len(test) < 5:
            continue

        X_train = train[feature_cols].fillna(0)
        y_train = train["team1_won"]
        X_test  = test[feature_cols].fillna(0)
        y_test  = test["team1_won"]

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        results.append({
            "train_up_to": train_seasons[-1],
            "test_season": test_season,
            "n_matches":   len(test),
            "accuracy":    round(acc, 3)
        })
        print(f"{train_seasons[-1]:<15} {test_season:<15} {len(test):<10} {acc:.1%}")

    return results

def train_final_model(df, feature_cols):
    """
    Final model — train on everything up to 2023, evaluate on 2024+
    """
    train = df[df["season"] <= 2023]
    test  = df[df["season"] >= 2024]

    X_train = train[feature_cols].fillna(0)
    y_train = train["team1_won"]
    X_test  = test[feature_cols].fillna(0)
    y_test  = test["team1_won"]

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(f"\n── Final model (train ≤2023, test ≥2024) ──")
    print(f"Test matches: {len(test)}")
    print(f"Accuracy: {accuracy_score(y_test, preds):.1%}")
    print(f"\nClassification report:\n{classification_report(y_test, preds)}")

    # Feature importance
    importances = pd.Series(
        model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)
    print(f"\nFeature importances:\n{importances}")

    return model, importances

if __name__ == "__main__":
    print("Loading data...")
    df = load_and_prepare()
    print(f"Dataset: {len(df)} matches, {df['season'].nunique()} seasons")

    feature_cols = build_features(df)
    print(f"Features: {feature_cols}")

    results = time_split_evaluate(df, feature_cols)
    model, importances = train_final_model(df, feature_cols)