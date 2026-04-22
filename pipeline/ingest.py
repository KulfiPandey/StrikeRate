# pipeline/ingest.py
# ──────────────────────────────────────────────────────
# What this file does:
#   Downloads the full IPL ball-by-ball dataset from Cricsheet,
#   unzips it, and saves the raw CSV files into data/raw/
#
# Why Cricsheet?
#   It's the best free source for ball-by-ball cricket data.
#   Every IPL match since 2008, every delivery recorded.
# ──────────────────────────────────────────────────────

import os
import json
import csv
import requests
import zipfile
from tqdm import tqdm
from config import CRICSHEET_IPL_URL, RAW_DIR

def download_ipl_data():
    print("📥 Starting IPL data download from Cricsheet...")
    os.makedirs(RAW_DIR, exist_ok=True)
    zip_path = os.path.join(RAW_DIR, "ipl_csv2.zip")

    response = requests.get(CRICSHEET_IPL_URL, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(zip_path, 'wb') as f, tqdm(
        desc="Downloading",
        total=total_size,
        unit='B',
        unit_scale=True
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

    print("✅ Download complete. Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(RAW_DIR)

    print(f"✅ Extracted to {RAW_DIR}")
    print(f"📁 Files downloaded: {len(os.listdir(RAW_DIR))}")

def parse_info_file(filepath):
    """
    Parses a Cricsheet _info.csv file into a clean dictionary.
    Why custom? Because the file has variable columns per row,
    so pandas can't read it directly.
    """
    info = {}
    players = {}

    # Use the csv module so quoted commas don't break parsing
    # (e.g. venue names or cities containing commas).
    with open(filepath, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for parts in reader:
            if len(parts) < 2:
                continue

            key = parts[1].strip()

            if key == "team" and len(parts) >= 3:
                info.setdefault("teams", []).append(parts[2].strip())
            elif key == "player" and len(parts) >= 4:
                team = parts[2].strip()
                player = parts[3].strip()
                if team and player:
                    players.setdefault(team, []).append(player)
            elif key == "registry":
                continue
            elif len(parts) >= 3:
                # Some keys may have extra columns; we take the first value column.
                info[key] = parts[2].strip()

    info['players'] = players
    return info

# ── Only runs when you execute this file directly ────
if __name__ == "__main__":
    test_info = parse_info_file("data/raw/1082591_info.csv")
    print(json.dumps(test_info, indent=2))