#!/usr/bin/env python
"""
StrikeRate golden path: fetch Polymarket odds, score with calibrated pre-match model, rank edges.

  python strike_scan.py
  python strike_scan.py --min-edge 0.08 --no-fetch
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pipeline.edge_engine import run_scan


def main() -> None:
    p = argparse.ArgumentParser(
        description="Scan Polymarket IPL markets for disagreement vs calibrated pre-match model.",
    )
    p.add_argument(
        "--min-edge",
        type=float,
        default=0.05,
        help="Minimum |model - market| to display (default 0.05 = 5%%).",
    )
    p.add_argument(
        "--no-fetch",
        action="store_true",
        help="Use existing polymarket_match_odds.csv (skip live fetch).",
    )
    p.add_argument(
        "--no-log",
        action="store_true",
        help="Do not append to data/processed/edge_log/edges.csv.",
    )
    p.add_argument(
        "--no-train",
        action="store_true",
        help="Fail if pre_match_model.joblib is missing instead of training.",
    )
    args = p.parse_args()

    run_scan(
        fetch=not args.no_fetch,
        min_edge=args.min_edge,
        train_if_missing=not args.no_train,
        log=not args.no_log,
    )


if __name__ == "__main__":
    main()
