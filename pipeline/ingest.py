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

    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 2:
                continue
            key = parts[1]

            if key == 'team':
                info.setdefault('teams', []).append(parts[2])
            elif key == 'player':
                team = parts[2]
                player = parts[3]
                players.setdefault(team, []).append(player)
            elif key == 'registry':
                pass
            elif len(parts) == 3:
                info[key] = parts[2]

    info['players'] = players
    return info

# ── Only runs when you execute this file directly ────
if __name__ == "__main__":
    test_info = parse_info_file("data/raw/1082591_info.csv")
    print(json.dumps(test_info, indent=2))