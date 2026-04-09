import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import PROCESSED_DIR


FEATURES = [
    "team1_enc","team2_enc","venue_enc",
    "team1_elo","team2_elo","elo_diff",
    "team1_form","team2_form","form_diff",
    "team1_bat_sr","team2_bat_sr","sr_diff",
    "head_to_head",
    "venue_bat_first_wr","venue_t1_wr",
    "toss_is_team1","toss_decision_enc"
]


def load():
    df = pd.read_csv(Path(PROCESSED_DIR) / "pre_match_clean.csv")
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["season"] = df["season"].astype(int)
    df = df.sort_values("start_date").reset_index(drop=True)
    return df


def encode(train, test):
    le_team = LabelEncoder()
    le_venue = LabelEncoder()

    le_team.fit(pd.concat([train["team1"], train["team2"]]))
    le_venue.fit(train["venue"])

    train["team1_enc"] = le_team.transform(train["team1"])
    train["team2_enc"] = le_team.transform(train["team2"])
    train["venue_enc"] = le_venue.transform(train["venue"])

    mapping_team = {k: v for k, v in zip(le_team.classes_, le_team.transform(le_team.classes_))}
    mapping_venue = {k: v for k, v in zip(le_venue.classes_, le_venue.transform(le_venue.classes_))}

    test["team1_enc"] = test["team1"].map(mapping_team).fillna(-1).astype(int)
    test["team2_enc"] = test["team2"].map(mapping_team).fillna(-1).astype(int)
    test["venue_enc"] = test["venue"].map(mapping_venue).fillna(-1).astype(int)

    return train, test


def evaluate(df):
    seasons = sorted(df["season"].unique())

    print("\n── Rolling Window ──")
    for i in range(3, len(seasons)):
        train = df[df["season"] < seasons[i]].copy()
        test = df[df["season"] == seasons[i]].copy()

        if len(test) < 5:
            continue

        train, test = encode(train, test)

        med = train[FEATURES].median()

        X_tr = train[FEATURES].fillna(med)
        X_te = test[FEATURES].fillna(med)

        y_tr = train["team1_won"]
        y_te = test["team1_won"]

        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        model.fit(X_tr, y_tr)

        acc = accuracy_score(y_te, model.predict(X_te))
        print(f"{seasons[i]} → {acc:.2%}")

    # Final
    train = df[df["season"] <= 2022].copy()
    test = df[df["season"] >= 2023].copy()

    train, test = encode(train, test)

    med = train[FEATURES].median()

    X_tr = train[FEATURES].fillna(med)
    X_te = test[FEATURES].fillna(med)

    y_tr = train["team1_won"]
    y_te = test["team1_won"]

    model = GradientBoostingClassifier(n_estimators=200, random_state=42)
    model.fit(X_tr, y_tr)

    preds = model.predict(X_te)
    acc = accuracy_score(y_te, preds)
    base = max(y_te.mean(), 1 - y_te.mean())

    print("\n── FINAL MODEL ──")
    print(f"Accuracy: {acc:.2%}")
    print(f"Baseline: {base:.2%}")
    print(f"Lift: +{(acc-base):.2%}")

    print(classification_report(y_te, preds))

    imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("\nTop Features:")
    print(imp.head(10))

    return model


if __name__ == "__main__":
    df = load()
    model = evaluate(df)