from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parent


def _preferred_python() -> str:
    """
    Prefer the repo-local venv interpreter when present so commands work even if the
    caller's global Python doesn't have deps installed.
    """
    venv_py_win = ROOT / ".venv" / "Scripts" / "python.exe"
    if venv_py_win.exists():
        return str(venv_py_win)

    venv_py_posix = ROOT / ".venv" / "bin" / "python"
    if venv_py_posix.exists():
        return str(venv_py_posix)

    return sys.executable


def _run(module_or_path: str, args: list[str] | None = None) -> None:
    args = args or []
    cmd = [_preferred_python(), module_or_path, *args]
    print(f"\n$ {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=str(ROOT))


def cmd_pipeline(_: argparse.Namespace) -> None:
    _run(str(ROOT / "pipeline" / "processor.py"))
    _run(str(ROOT / "pipeline" / "match_features.py"))
    _run(str(ROOT / "pipeline" / "pre_matches_features.py"))


def cmd_train_honest(_: argparse.Namespace) -> None:
    _run(str(ROOT / "models" / "honest_predictor.py"))


def cmd_train_match(_: argparse.Namespace) -> None:
    _run(str(ROOT / "models" / "match_predictor.py"))


def cmd_train_prematch(_: argparse.Namespace) -> None:
    _run(str(ROOT / "models" / "pre_match_model.py"))


def cmd_fetch_odds(_: argparse.Namespace) -> None:
    _run(str(ROOT / "pipeline" / "fetch_polymarket_odds.py"))


def cmd_fetch_espn(_: argparse.Namespace) -> None:
    _run(str(ROOT / "pipeline" / "espn_scraper.py"))


def cmd_x_scrape(ns: argparse.Namespace) -> None:
    from pipeline.social_x import save_posts, scrape_x_posts_snscrape

    df = scrape_x_posts_snscrape(query=ns.query, limit=ns.limit)
    out = save_posts(df)
    print(f"Saved: {out} ({len(df)} posts)")


def cmd_x_features(_: argparse.Namespace) -> None:
    _run(str(ROOT / "pipeline" / "social_features.py"))


def cmd_value_bets(_: argparse.Namespace) -> None:
    _run(str(ROOT / "pipeline" / "value_bets.py"))


def cmd_backtest(_: argparse.Namespace) -> None:
    _run(str(ROOT / "models" / "backtest.py"))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="strike",
        description="StrikeRate: IPL data pipeline + models + enrichment (ESPN/Polymarket/X).",
    )
    sub = p.add_subparsers(required=True)

    s = sub.add_parser("pipeline", help="Build processed datasets (master/match/pre-match).")
    s.set_defaults(func=cmd_pipeline)

    s = sub.add_parser("train-honest", help="Train/evaluate the honest pre-match model.")
    s.set_defaults(func=cmd_train_honest)

    s = sub.add_parser("train-match", help="Train/evaluate match outcome model (match_features).")
    s.set_defaults(func=cmd_train_match)

    s = sub.add_parser("train-prematch", help="Train calibrated pre-match model (saves artifacts).")
    s.set_defaults(func=cmd_train_prematch)

    s = sub.add_parser("fetch-odds", help="Fetch Polymarket odds into data/processed.")
    s.set_defaults(func=cmd_fetch_odds)

    s = sub.add_parser("fetch-espn", help="Scrape ESPN stats into data/processed.")
    s.set_defaults(func=cmd_fetch_espn)

    s = sub.add_parser("x-scrape", help="Scrape X/Twitter posts (requires snscrape).")
    s.add_argument("--query", required=True, help='Search query, e.g. "IPL MI vs CSK since:2026-04-01 until:2026-04-22"')
    s.add_argument("--limit", type=int, default=500, help="Max posts to fetch.")
    s.set_defaults(func=cmd_x_scrape)

    s = sub.add_parser("x-features", help="Build X buzz features and merge into pre-match dataset.")
    s.set_defaults(func=cmd_x_features)

    s = sub.add_parser("value-bets", help="Merge model probs with Polymarket odds and rank edges.")
    s.set_defaults(func=cmd_value_bets)

    s = sub.add_parser("backtest", help="Run ROI backtest + calibration plots (synthetic book).")
    s.set_defaults(func=cmd_backtest)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    ns = parser.parse_args(argv)
    ns.func(ns)


if __name__ == "__main__":
    main()

