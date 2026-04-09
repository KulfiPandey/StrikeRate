# models/predictor.py
# ──────────────────────────────────────────────────────
# What this file does:
#   Trains a Random Forest model to predict match outcomes.
#   Uses rolling window validation (ChatGPT's suggestion) —
#   train on past seasons, predict the next one.
#
# What is a Random Forest?
#   Imagine asking 100 different people to predict a match.
#   Each person looks at slightly different stats.
#   The majority vote becomes the final prediction.
#   That's a Random Forest — 100 decision trees, majority wins.
#
# Why rolling window instead of random split?
#   Cricket evolved massively (6.78 → 8.79 run rate).
#   A model trained on 2024 data shouldn't "know" about
#   2009 conditions. We always predict forward in time.
# ──────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from config import PROCESSED_DIR

def load_data():
    path = os.path.join(PROCESSED_DIR, "match_features.csv")
    df = pd.read_csv(path)
    df['season'] = df['season'].astype(int)
    print(f"✅ Loaded {len(df)} matches across {df['season'].nunique()} seasons")
    return df

def get_features(df):
    """
    These are the columns we feed to the model.
    We exclude: match_id, team names (strings), dates, 
    and the target (team1_won) itself.
    We use: encoded teams, venue, phase runs, wickets, boundaries.
    """
    feature_cols = [
        "team1_enc", "team2_enc", "venue_enc",
        "team1_won_toss",
        "inn1_pp_runs", "inn1_mid_runs", "inn1_death_runs",
        "inn2_pp_runs", "inn2_mid_runs", "inn2_death_runs",
        "team1_wickets", "team2_wickets",
        "inn1_boundary_pct", "inn1_dot_pct",
        "inn2_boundary_pct", "inn2_dot_pct",
    ]
    return feature_cols

def rolling_window_validation(df):
    """
    Rolling window validation — the right way to validate time-series data.
    
    For each season from 2015 onwards:
      - Train on ALL seasons before it
      - Predict THAT season
      - Record accuracy
    
    This tells us: "How would our model have performed if deployed live?"
    """
    feature_cols = get_features(df)
    seasons = sorted(df['season'].unique())
    test_seasons = [s for s in seasons if s >= 2015]

    results = []
    print("\n📊 Rolling Window Validation:")
    print(f"{'Season':<10} {'Train Size':<12} {'Test Size':<10} {'Accuracy':<10}")
    print("-" * 45)

    for test_season in test_seasons:
        train = df[df['season'] < test_season]
        test  = df[df['season'] == test_season]

        if len(train) < 50 or len(test) < 5:
            continue

        X_train = train[feature_cols].fillna(0)
        y_train = train['team1_won']
        X_test  = test[feature_cols].fillna(0)
        y_test  = test['team1_won']

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        results.append({
            'season': test_season,
            'train_size': len(train),
            'test_size': len(test),
            'accuracy': acc
        })
        print(f"{test_season:<10} {len(train):<12} {len(test):<10} {acc:.1%}")

    return pd.DataFrame(results)

def train_final_model(df):
    """
    Train on ALL data up to 2024, evaluate on 2025-2026.
    This is the model we'll use for actual predictions.
    """
    feature_cols = get_features(df)

    train = df[df['season'] <= 2024]
    test  = df[df['season'] >= 2025]

    X_train = train[feature_cols].fillna(0)
    y_train = train['team1_won']
    X_test  = test[feature_cols].fillna(0)
    y_test  = test['team1_won']

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"\n🏆 Final Model (trained up to 2024, tested on 2025-2026):")
    print(f"   Accuracy: {acc:.1%}")
    print(f"\n{classification_report(y_test, preds, target_names=['team2 wins','team1 wins'])}")

    return model, feature_cols

def plot_feature_importance(model, feature_cols):
    """
    Feature importance = how much each feature helped the model decide.
    The higher the bar, the more that feature matters.
    """
    importance = pd.Series(model.feature_importances_, index=feature_cols)
    importance = importance.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['#e63946' if v > importance.median() else '#457b9d' 
              for v in importance]
    importance.plot(kind='barh', ax=ax, color=colors)
    ax.set_title('Feature Importance — What Drives Match Outcomes?',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Importance Score')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PROCESSED_DIR, 'chart_feature_importance.png'), dpi=150)
    plt.show()
    print("✅ Feature importance chart saved!")

def plot_rolling_accuracy(results_df):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(results_df['season'], results_df['accuracy']*100,
            marker='o', linewidth=2.5, color='#2d6a4f', markersize=8)
    ax.fill_between(results_df['season'], results_df['accuracy']*100,
                    alpha=0.15, color='#2d6a4f')
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Random baseline (50%)')
    ax.set_title('Model Accuracy Over Time (Rolling Window Validation)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Season (predicted)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(40, 80)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PROCESSED_DIR, 'chart_rolling_accuracy.png'), dpi=150)
    plt.show()
    print("✅ Rolling accuracy chart saved!")

if __name__ == "__main__":
    df = load_data()
    results = rolling_window_validation(df)
    model, feature_cols = train_final_model(df)
    plot_feature_importance(model, feature_cols)
    plot_rolling_accuracy(results)
    print("\n🏏 Phase 4 complete — ML model trained and evaluated!")