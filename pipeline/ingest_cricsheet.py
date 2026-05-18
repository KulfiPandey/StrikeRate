"""
Ingest Cricsheet JSON (IPL) into canonical Parquet tables.

Raw layout (after download):
  data/raw/cricsheet/{season_year}/*.json

Processed:
  data/processed/matches.parquet
  data/processed/deliveries.parquet
  data/processed/match_players.parquet

Legacy Kaggle-style CSVs are optional (--compat).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import requests
from tqdm import tqdm

from pipeline.data_store import (
    DELIVERIES_PARQUET,
    MATCH_PLAYERS_PARQUET,
    MATCHES_PARQUET,
    PROCESSED_DIR,
    RAW_CRICSHEET_DIR,
    export_kaggle_compat,
)

ROOT = Path(__file__).resolve().parents[1]

try:
    from config import CRICSHEET_IPL_JSON_URL
except Exception:
    CRICSHEET_IPL_JSON_URL = "https://cricsheet.org/downloads/ipl_json.zip"

T20_PP_END_OVER = 5
T20_DEATH_START_OVER = 15


def season_year(season: Any) -> str:
    text = str(season or "").strip()
    m = re.search(r"\d{4}", text)
    return m.group() if m else "unknown"


def parse_outcome(outcome: dict[str, Any]) -> dict[str, Any]:
    if not outcome:
        return {
            "winner": None,
            "outcome_type": "unknown",
            "outcome_method": None,
            "margin_runs": None,
            "margin_wickets": None,
            "eliminator": None,
        }

    result = outcome.get("result")
    winner = outcome.get("winner")
    eliminator = outcome.get("eliminator") or outcome.get("bowl_out")
    method = outcome.get("method")

    if winner:
        outcome_type = "winner"
        effective_winner = winner
    elif eliminator:
        outcome_type = "tie"
        effective_winner = eliminator
    elif result:
        outcome_type = str(result).strip().lower()
        effective_winner = None
    else:
        outcome_type = "unknown"
        effective_winner = None

    by = outcome.get("by") or {}
    return {
        "winner": effective_winner,
        "outcome_type": outcome_type,
        "outcome_method": method,
        "margin_runs": by.get("runs"),
        "margin_wickets": by.get("wickets"),
        "eliminator": eliminator,
    }


def delivery_ball_id(over_num: int, delivery_idx: int) -> float:
    return round(over_num + (delivery_idx + 1) / 10.0, 1)


def in_powerplay(ball: float, powerplays: list[dict[str, Any]] | None) -> tuple[bool, str | None]:
    if not powerplays:
        return False, None
    for pp in powerplays:
        try:
            start = float(pp.get("from", -1))
            end = float(pp.get("to", -1))
        except (TypeError, ValueError):
            continue
        if start <= ball <= end:
            return True, pp.get("type")
    return False, None


def t20_phase(over_num: int) -> str:
    if over_num <= T20_PP_END_OVER:
        return "powerplay"
    if over_num < T20_DEATH_START_OVER:
        return "middle"
    return "death"


def parse_deliveries(
    match_id: str,
    info: dict[str, Any],
    innings_list: list[dict[str, Any]],
    meta: dict[str, Any],
) -> list[dict[str, Any]]:
    teams = info.get("teams") or []
    if len(teams) != 2:
        return []

    rows: list[dict[str, Any]] = []
    base = {
        "match_id": match_id,
        "season": meta.get("season"),
        "start_date": meta.get("start_date"),
        "venue": meta.get("venue"),
        "city": meta.get("city"),
        "toss_winner": meta.get("toss_winner"),
        "toss_decision": meta.get("toss_decision"),
        "match_winner": meta.get("winner"),
    }

    for inn_idx, innings in enumerate(innings_list, start=1):
        batting_team = innings.get("team")
        if not batting_team:
            continue
        bowling_team = teams[1] if teams[0] == batting_team else teams[0]
        powerplays = innings.get("powerplays")
        is_super_over = bool(innings.get("super_over"))

        for over_block in innings.get("overs") or []:
            over_num = int(over_block.get("over", 0))
            for d_idx, delivery in enumerate(over_block.get("deliveries") or []):
                ball = delivery_ball_id(over_num, d_idx)
                runs = delivery.get("runs") or {}
                extras_obj = delivery.get("extras") or {}
                wickets = delivery.get("wickets") or []

                bye = int(extras_obj.get("byes") or 0)
                legbye = int(extras_obj.get("legbyes") or 0)
                wide = int(extras_obj.get("wides") or 0)
                noball = int(extras_obj.get("noballs") or 0)
                penalty = int(extras_obj.get("penalty") or 0)

                pp_flag, pp_type = in_powerplay(ball, powerplays)

                row = {
                    **base,
                    "innings": inn_idx,
                    "batting_team": batting_team,
                    "bowling_team": bowling_team,
                    "over": over_num,
                    "ball": ball,
                    "batter": delivery.get("batter"),
                    "non_striker": delivery.get("non_striker"),
                    "bowler": delivery.get("bowler"),
                    "batsman_runs": int(runs.get("batter") or 0),
                    "extras_runs": int(runs.get("extras") or 0),
                    "total_runs": int(runs.get("total") or 0),
                    "bye_runs": bye,
                    "legbye_runs": legbye,
                    "wide_runs": wide,
                    "noball_runs": noball,
                    "penalty_runs": penalty,
                    "is_wicket": int(len(wickets) > 0),
                    "wicket_kind": wickets[0].get("kind") if wickets else None,
                    "player_dismissed": wickets[0].get("player_out") if wickets else None,
                    "fielders": ",".join(
                        f.get("name", "") if isinstance(f, dict) else str(f)
                        for w in wickets
                        for f in (w.get("fielders") or [])
                    )
                    or None,
                    "is_super_over": int(is_super_over),
                    "is_powerplay": int(pp_flag),
                    "powerplay_type": pp_type,
                    "match_phase": t20_phase(over_num),
                }
                rows.append(row)
    return rows


def parse_match_json(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)

    info = doc.get("info") or {}
    teams = info.get("teams") or []
    if len(teams) != 2:
        raise ValueError(f"expected 2 teams, got {teams}")

    match_id = path.stem
    dates = info.get("dates") or []
    start_date = dates[0] if dates else None
    toss = info.get("toss") or {}
    outcome = parse_outcome(info.get("outcome") or {})
    event = info.get("event") or {}
    pom = info.get("player_of_match") or []

    match_row = {
        "match_id": match_id,
        "season": info.get("season"),
        "season_year": season_year(info.get("season")),
        "start_date": start_date,
        "team1": teams[0],
        "team2": teams[1],
        "venue": info.get("venue"),
        "city": info.get("city"),
        "toss_winner": toss.get("winner"),
        "toss_decision": toss.get("decision"),
        "toss_uncontested": int(bool(toss.get("uncontested"))),
        "winner": outcome["winner"],
        "outcome_type": outcome["outcome_type"],
        "outcome_method": outcome["outcome_method"],
        "margin_runs": outcome["margin_runs"],
        "margin_wickets": outcome["margin_wickets"],
        "eliminator": outcome["eliminator"],
        "player_of_match": pom[0] if pom else None,
        "match_type": info.get("match_type"),
        "gender": info.get("gender"),
        "event_name": event.get("name"),
        "match_number": event.get("match_number"),
        "balls_per_over": info.get("balls_per_over", 6),
        "overs_limit": info.get("overs"),
        "data_version": (doc.get("meta") or {}).get("data_version"),
    }

    players_rows: list[dict[str, Any]] = []
    players_map = info.get("players") or {}
    for team, plist in players_map.items():
        for player in plist or []:
            players_rows.append(
                {
                    "match_id": match_id,
                    "team": team,
                    "player": player,
                    "season": info.get("season"),
                    "start_date": start_date,
                }
            )

    delivery_rows = parse_deliveries(
        match_id,
        info,
        doc.get("innings") or [],
        {
            "season": info.get("season"),
            "start_date": start_date,
            "venue": info.get("venue"),
            "city": info.get("city"),
            "toss_winner": toss.get("winner"),
            "toss_decision": toss.get("decision"),
            "winner": outcome["winner"],
        },
    )
    return match_row, delivery_rows, players_rows


def iter_json_files(raw_dir: Path) -> Iterator[Path]:
    for path in sorted(raw_dir.rglob("*.json")):
        if path.name.startswith("."):
            continue
        yield path


def download_and_stage(raw_dir: Path = RAW_CRICSHEET_DIR) -> int:
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "ipl_json.zip"
    staging = raw_dir / "_staging"

    print(f"Downloading {CRICSHEET_IPL_JSON_URL} ...")
    resp = requests.get(CRICSHEET_IPL_JSON_URL, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length") or 0)

    with open(zip_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc="download") as bar:
        for chunk in resp.iter_content(chunk_size=65536):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))

    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    print("Extracting zip...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(staging)

    json_files = list(staging.rglob("*.json"))
    print(f"Organizing {len(json_files)} JSON files by season...")
    for jpath in tqdm(json_files, desc="stage"):
        try:
            with open(jpath, encoding="utf-8") as f:
                season = season_year((json.load(f).get("info") or {}).get("season"))
        except Exception:
            season = "unknown"
        dest_dir = raw_dir / season
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / jpath.name
        if not dest.exists():
            shutil.copy2(jpath, dest)

    shutil.rmtree(staging)
    print(f"Staged under {raw_dir}")
    return len(json_files)


def ingest(
    raw_dir: Path = RAW_CRICSHEET_DIR,
    *,
    compat_csv: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    files = list(iter_json_files(raw_dir))
    if not files:
        raise FileNotFoundError(
            f"No JSON under {raw_dir}. Run with --download first."
        )

    match_rows: list[dict[str, Any]] = []
    delivery_rows: list[dict[str, Any]] = []
    player_rows: list[dict[str, Any]] = []
    errors: list[str] = []

    for path in tqdm(files, desc="ingest"):
        try:
            m, d, p = parse_match_json(path)
            match_rows.append(m)
            delivery_rows.extend(d)
            player_rows.extend(p)
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")

    if errors:
        print(f"Skipped {len(errors)} files (first: {errors[0]})")

    matches = pd.DataFrame(match_rows)
    deliveries = pd.DataFrame(delivery_rows)
    players = pd.DataFrame(player_rows)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for frame in (matches, deliveries, players):
        if "season" in frame.columns:
            frame["season"] = frame["season"].astype(str)
        if "match_number" in frame.columns:
            frame["match_number"] = pd.to_numeric(frame["match_number"], errors="coerce")

    matches["start_date"] = pd.to_datetime(matches["start_date"], errors="coerce")
    deliveries["start_date"] = pd.to_datetime(deliveries["start_date"], errors="coerce")
    players["start_date"] = pd.to_datetime(players["start_date"], errors="coerce")

    matches = matches.sort_values(["start_date", "match_id"]).reset_index(drop=True)
    deliveries = deliveries.sort_values(["match_id", "innings", "ball"]).reset_index(drop=True)

    matches.to_parquet(MATCHES_PARQUET, index=False)
    deliveries.to_parquet(DELIVERIES_PARQUET, index=False)
    players.to_parquet(MATCH_PLAYERS_PARQUET, index=False)

    print(f"Saved {MATCHES_PARQUET} ({len(matches):,} matches)")
    print(f"Saved {DELIVERIES_PARQUET} ({len(deliveries):,} deliveries)")
    print(f"Saved {MATCH_PLAYERS_PARQUET} ({len(players):,} player rows)")

    if compat_csv:
        m_csv, d_csv = export_kaggle_compat(matches, deliveries)
        print(f"Compat CSV: {m_csv.name}, {d_csv.name}")

    return matches, deliveries, players


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Cricsheet IPL JSON to Parquet")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download ipl_json.zip and stage under data/raw/cricsheet/{year}/",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_CRICSHEET_DIR,
        help="Root directory containing season subfolders of JSON",
    )
    parser.add_argument(
        "--no-compat",
        action="store_true",
        help="Skip writing legacy deliveries.csv / matches.csv",
    )
    args = parser.parse_args()

    if args.download:
        download_and_stage(args.raw_dir)

    ingest(args.raw_dir, compat_csv=not args.no_compat)


if __name__ == "__main__":
    main()
