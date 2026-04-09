import pandas as pd
import requests
import time
from pathlib import Path
import sys
from bs4 import BeautifulSoup
sys.path.append(str(Path(__file__).parent.parent))
from config import PROCESSED_DIR

BASE_URL = "https://stats.espncricinfo.com/ci/engine/stats/index.html"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

def fetch_page(stat_type, page):
    params = f"class=6;competition=117;page={page};template=results;type={stat_type}"
    url = f"{BASE_URL}?{params}"
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    return resp.text

def parse_page(html):
    soup = BeautifulSoup(html, "lxml")
    for table in soup.find_all("table", class_="engineTable"):
        headers = [th.get_text(strip=True) for th in table.find_all("tr")[0].find_all(["th", "td"])]
        if "Player" not in headers:
            continue
        rows = []
        for tr in table.find_all("tr")[1:]:
            cells = [td.get_text(strip=True) for td in tr.find_all(["th", "td"])]
            if cells:
                rows.append(cells)
        if not rows:
            return None
        n = len(headers)
        rows = [r[:n] if len(r) >= n else r + [""] * (n - len(r)) for r in rows]
        return pd.DataFrame(rows, columns=headers)
    return None

def scrape(stat_type, max_pages=15):
    print(f"\nScraping {stat_type} stats...")
    all_dfs = []
    for page in range(1, max_pages + 1):
        print(f"  Page {page}...", end=" ", flush=True)
        try:
            html = fetch_page(stat_type, page)
            df = parse_page(html)
            if df is None or len(df) == 0:
                print("empty — stopping")
                break
            all_dfs.append(df)
            print(f"{len(df)} rows")
            time.sleep(2)
        except Exception as e:
            print(f"error: {e} — stopping")
            break
    if not all_dfs:
        return None
    return pd.concat(all_dfs, ignore_index=True)

def clean(df, stat_type):
    df.columns = [str(c).strip() for c in df.columns]

    # Drop unnamed/empty columns
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed|^$")]

    # Rename columns
    rename_map = {
        "batting": {
            "Player": "player", "Span": "span", "Mat": "matches",
            "Inns": "innings", "NO": "not_outs", "Runs": "runs",
            "HS": "highest_score", "Ave": "batting_avg", "BF": "balls_faced",
            "SR": "batting_sr", "100": "hundreds", "50": "fifties",
            "0": "ducks", "4s": "fours", "6s": "sixes",
        },
        "bowling": {
            "Player": "player", "Span": "span", "Mat": "matches",
            "Inns": "innings", "Overs": "overs", "Balls": "balls",
            "Runs": "runs_conceded", "Wkts": "wickets", "BBI": "bbi",
            "Ave": "bowling_avg", "Econ": "economy", "SR": "bowling_sr",
            "4": "four_wickets", "5": "five_wickets",
        },
    }
    df = df.rename(columns={k: v for k, v in rename_map[stat_type].items() if k in df.columns})

    # Drop sub-rows (team name rows — always start with '(' after stripping)
    df["player"] = df["player"].astype(str).str.strip()
    df = df[~df["player"].str.startswith("(")]
    df = df[df["player"] != ""]
    df = df[df["player"] != "nan"]
    df = df.dropna(subset=["player"])
    df = df.drop_duplicates(subset=["player"])
    df = df.reset_index(drop=True)

    # Convert numeric columns
    num_cols = {
        "batting":  ["matches", "innings", "runs", "batting_avg", "batting_sr",
                     "balls_faced", "hundreds", "fifties", "ducks", "fours", "sixes"],
        "bowling":  ["matches", "wickets", "economy", "bowling_avg", "bowling_sr",
                     "overs", "balls", "runs_conceded"],
    }
    for col in num_cols[stat_type]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

if __name__ == "__main__":
    bat_path  = Path(PROCESSED_DIR) / "espn_batting_stats.csv"
    bowl_path = Path(PROCESSED_DIR) / "espn_bowling_stats.csv"

    # ── Batting ──
    existing_bat = pd.read_csv(bat_path) if bat_path.exists() else None
    if existing_bat is not None and len(existing_bat) >= 500:
        print(f"Batting already good ({len(existing_bat)} players) — skipping")
    else:
        df = scrape("batting", max_pages=15)
        if df is not None:
            df = clean(df, "batting")
            df.to_csv(bat_path, index=False)
            print(f"\nSaved batting: {len(df)} players")
            print(df[["player", "matches", "runs", "batting_avg"]].head(10).to_string())

    time.sleep(5)

    # ── Bowling ──
    existing_bowl = pd.read_csv(bowl_path) if bowl_path.exists() else None
    if existing_bowl is not None and len(existing_bowl) >= 500:
        print(f"Bowling already good ({len(existing_bowl)} players) — skipping")
    else:
        df = scrape("bowling", max_pages=15)
        if df is not None:
            df = clean(df, "bowling")
            df.to_csv(bowl_path, index=False)
            print(f"\nSaved bowling: {len(df)} players")
            print(df[["player", "matches", "wickets", "economy", "bowling_avg"]].head(10).to_string())