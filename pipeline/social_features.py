from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import PROCESSED_DIR
from pipeline.team_name_standardizer import standardize_team_name


TEAM_KEYWORDS = {
    # canonical -> keywords to match in text (lowercased)
    "MI": ["mumbai indians", "mi"],
    "CSK": ["chennai super kings", "csk"],
    "RCB": ["royal challengers bangalore", "royal challengers", "rcb"],
    "KKR": ["kolkata knight riders", "kkr"],
    "RR": ["rajasthan royals", "rr"],
    "PBKS": ["punjab kings", "pbks", "kings xi punjab", "kxip"],
    "DC": ["delhi capitals", "dc", "delhi daredevils", "dd"],
    "SRH": ["sunrisers hyderabad", "srh"],
    "GT": ["gujarat titans", "gt"],
    "LSG": ["lucknow super giants", "lsg"],
    # defunct (kept for completeness)
    "DEC": ["deccan chargers", "deccan"],
    "KTK": ["kochi tuskers kerala", "ktk"],
    "PW": ["pune warriors", "pw"],
    "GL": ["gujarat lions", "gl"],
    "RPS": ["rising pune supergiant", "rps"],
}


def _text_has_any(text: str, keywords: list[str]) -> bool:
    t = (text or "").lower()
    return any(k in t for k in keywords)


def build_team_daily_buzz(posts: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw X posts into per-day, per-team "buzz" features:
    - x_mentions: count of posts mentioning the team
    - x_engagement: simple engagement proxy (likes + retweets + replies + quotes)
    """
    if posts.empty:
        return pd.DataFrame(columns=["date", "team", "x_mentions", "x_engagement"])

    df = posts.copy()
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["created_at"])
    df["date"] = df["created_at"].dt.date.astype(str)

    # engagement (safe even if cols missing)
    for c in ["like_count", "retweet_count", "reply_count", "quote_count"]:
        if c not in df.columns:
            df[c] = 0
    df["x_engagement"] = (
        pd.to_numeric(df["like_count"], errors="coerce").fillna(0)
        + pd.to_numeric(df["retweet_count"], errors="coerce").fillna(0)
        + pd.to_numeric(df["reply_count"], errors="coerce").fillna(0)
        + pd.to_numeric(df["quote_count"], errors="coerce").fillna(0)
    )

    rows = []
    for team, keywords in TEAM_KEYWORDS.items():
        mask = df["text"].astype(str).apply(lambda t: _text_has_any(t, keywords))
        sub = df[mask]
        if sub.empty:
            continue
        agg = (
            sub.groupby("date")
            .agg(x_mentions=("text", "count"), x_engagement=("x_engagement", "sum"))
            .reset_index()
        )
        agg["team"] = team
        rows.append(agg)

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["date", "team", "x_mentions", "x_engagement"])
    out["team"] = out["team"].apply(standardize_team_name)
    return out


def merge_buzz_into_pre_match(
    pre_match: pd.DataFrame,
    buzz_daily: pd.DataFrame,
) -> pd.DataFrame:
    df = pre_match.copy()
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["date"] = df["start_date"].dt.date.astype(str)

    df["team1_std"] = df["team1"].apply(standardize_team_name)
    df["team2_std"] = df["team2"].apply(standardize_team_name)

    b = buzz_daily.copy()
    b["date"] = b["date"].astype(str)

    b1 = b.rename(
        columns={
            "team": "team1_std",
            "x_mentions": "x_team1_mentions",
            "x_engagement": "x_team1_engagement",
        }
    )
    b2 = b.rename(
        columns={
            "team": "team2_std",
            "x_mentions": "x_team2_mentions",
            "x_engagement": "x_team2_engagement",
        }
    )

    df = df.merge(b1[["date", "team1_std", "x_team1_mentions", "x_team1_engagement"]], on=["date", "team1_std"], how="left")
    df = df.merge(b2[["date", "team2_std", "x_team2_mentions", "x_team2_engagement"]], on=["date", "team2_std"], how="left")

    for c, fill in [
        ("x_team1_mentions", 0),
        ("x_team2_mentions", 0),
        ("x_team1_engagement", 0),
        ("x_team2_engagement", 0),
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(fill)

    df["x_mentions_diff"] = df["x_team1_mentions"] - df["x_team2_mentions"]
    df["x_engagement_diff"] = df["x_team1_engagement"] - df["x_team2_engagement"]
    return df.drop(columns=["date"])


def main():
    posts_path = Path(PROCESSED_DIR) / "x_posts.csv"
    pre_path = Path(PROCESSED_DIR) / "pre_match_clean.csv"

    if not posts_path.exists():
        raise FileNotFoundError(f"Missing {posts_path}. Run X scrape first.")
    if not pre_path.exists():
        raise FileNotFoundError(f"Missing {pre_path}. Build pre-match dataset first.")

    posts = pd.read_csv(posts_path)
    pre = pd.read_csv(pre_path)

    buzz = build_team_daily_buzz(posts)
    buzz_out = Path(PROCESSED_DIR) / "x_team_daily_buzz.csv"
    buzz.to_csv(buzz_out, index=False)

    merged = merge_buzz_into_pre_match(pre, buzz)
    out_path = Path(PROCESSED_DIR) / "pre_match_with_x.csv"
    merged.to_csv(out_path, index=False)

    print(f"Saved buzz: {buzz_out}")
    print(f"Saved merged dataset: {out_path}")
    print("New columns:", [c for c in merged.columns if c not in pre.columns])


if __name__ == "__main__":
    main()

