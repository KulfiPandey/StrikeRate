# StrikeRate

IPL analytics: calibrated pre-match probabilities vs Polymarket — find where the market disagrees with historical signals.

## Quickstart (Windows / PowerShell)

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run the core pipeline (if you already have `data/raw` checked in, you can skip the download step):

```powershell
python pipeline\processor.py
python pipeline\match_features.py
python pipeline\pre_matches_features.py
```

Train/evaluate a model:

```powershell
python models\honest_predictor.py
```

Train the calibrated pre-match model (recommended for odds work):

```powershell
python strike.py train-prematch
```

**Golden path — market edge scanner:**

```powershell
python strike.py train-prematch   # once: calibrated pre_match_model.joblib
python strike_scan.py             # fetch odds -> score -> rank edges -> log
# or: python strike.py scan
```

Optional flags: `--min-edge 0.08`, `--no-fetch` (use cached odds), `--no-log`.

Outputs: `data/processed/value_bets.csv`, `data/processed/edge_log/edges.csv` (append-only forward log).

Run ROI backtest + calibration plots:

```powershell
python strike.py backtest
```

## Data layout

- `data/raw/`: Cricsheet match CSVs + `_info.csv` files
- `data/processed/`: derived datasets (`master_deliveries.csv`, `match_features.csv`, `pre_match_clean.csv`, etc.)

## Notes

- `pipeline/ingest.py` includes a downloader for Cricsheet IPL data.
- ESPN scraping (`pipeline/espn_scraper.py`) is best-effort and may break if ESPN changes HTML.
- Polymarket odds (`pipeline/fetch_polymarket_odds.py`) requires live network access.

