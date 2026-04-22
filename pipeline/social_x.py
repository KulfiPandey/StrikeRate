from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from config import PROCESSED_DIR


@dataclass(frozen=True)
class XPost:
    created_at: datetime
    text: str
    author: str | None
    url: str | None
    like_count: int | None
    retweet_count: int | None
    reply_count: int | None
    quote_count: int | None


def _try_import_snscrape():
    try:
        import snscrape.modules.twitter as sntwitter  # type: ignore
    except Exception:
        return None
    return sntwitter


def scrape_x_posts_snscrape(
    query: str,
    limit: int = 500,
) -> pd.DataFrame:
    """
    No-token scraper (best-effort):
      pip install snscrape

    Notes:
    - snscrape can break when X changes; treat as best-effort enrichment.
    - We only store the fields we actually use downstream.
    """
    sntwitter = _try_import_snscrape()
    if sntwitter is None:
        raise RuntimeError(
            "snscrape is not installed or failed to import. "
            "Install it with: pip install snscrape"
        )

    rows: list[dict] = []
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
        if i >= limit:
            break
        rows.append(
            {
                "created_at": pd.to_datetime(tweet.date).tz_convert("UTC") if getattr(tweet, "date", None) else None,
                "text": getattr(tweet, "rawContent", None) or getattr(tweet, "content", None) or "",
                "author": getattr(getattr(tweet, "user", None), "username", None),
                "url": getattr(tweet, "url", None),
                "like_count": getattr(tweet, "likeCount", None),
                "retweet_count": getattr(tweet, "retweetCount", None),
                "reply_count": getattr(tweet, "replyCount", None),
                "quote_count": getattr(tweet, "quoteCount", None),
                "query": query,
                "scraped_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    df = pd.DataFrame(rows)
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    return df


def save_posts(df: pd.DataFrame, out_path: Path | None = None) -> Path:
    out_path = out_path or (Path(PROCESSED_DIR) / "x_posts.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path

