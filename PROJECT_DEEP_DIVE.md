# StrikeRate — Deep Dive (Architecture + ML + Betting)

This document explains the project in depth: what each dataset is, how the pipeline works, how the models are evaluated safely (no leakage), and how the odds/value-bets/backtest pieces connect.

> TL;DR: The core deliverable is a **pre-match win probability model** that is evaluated with **walk-forward validation**, calibrated for probability quality, and can be compared against **market-implied probabilities** (Polymarket) to compute edges and run ROI simulations.

---

## 1) Repository layout

- `config.py`
  - Central path + constants (raw/processed directories, Cricsheet URL).
- `data/raw/`
  - Cricsheet match files: one `match_id.csv` (ball-by-ball) + `match_id_info.csv` (metadata).
- `data/processed/`
  - “Derived truth” datasets produced by the pipeline.
- `pipeline/`
  - ETL / scraping / feature builders.
- `models/`
  - Training, evaluation, and backtesting.
- `strike.py`
  - CLI that runs the whole system using the repo venv interpreter when available.
- `run.ps1`
  - PowerShell helper that creates venv + installs deps + runs steps.

---

## 2) Data sources

### 2.1 Cricsheet IPL ball-by-ball (primary)

- URL in `config.py`: `https://cricsheet.org/downloads/ipl_csv2.zip`
- Provides:
  - Ball-by-ball `runs_off_bat`, dismissals, innings, players, etc.
  - `_info.csv` metadata: venue, toss winner/decision, match winner, etc.

#### Why `_info.csv` parsing is tricky
Cricsheet `_info.csv` lines can contain **quoted commas** (venues/cities). This repo uses Python’s `csv.reader` (not `split(',')`) so parsing doesn’t silently corrupt fields.

### 2.2 ESPN stats (optional enrichment)
`pipeline/espn_scraper.py` scrapes aggregated batting/bowling tables. This is best-effort: ESPN HTML changes can break scrapers.

### 2.3 Polymarket odds (optional enrichment)
`pipeline/fetch_polymarket_odds.py` calls the Gamma API and extracts IPL markets + implied probabilities.

Important stored fields:
- Teams inferred from question text (`team_a`, `team_b`)
- Implied probabilities (`prob_a_wins`, `prob_b_wins`)
- Event timing metadata:
  - `end_date`, `game_start_time`, `event_start_time`

### 2.4 X/Twitter (optional enrichment)
`pipeline/social_x.py` (snscrape-based) can fetch posts for a query and store them as `data/processed/x_posts.csv`.
`pipeline/social_features.py` converts posts into daily team “buzz” features and merges them into a pre-match dataset.

Note: snscrape is “best-effort”; X changes can break scraping.

---

## 3) Core datasets (processed)

### 3.1 `master_deliveries.csv`
Built by `pipeline/processor.py`.

Shape: one row per delivery, with match metadata repeated on each row.
Used for:
- Building match-level aggregates
- Building team form / ELO

### 3.2 `match_features.csv`
Built by `pipeline/match_features.py`.

Shape: one row per match with:
- Teams, venue, toss info
- Phase runs (powerplay/middle/death) for both innings
- Wickets, dot%, boundary%

Used for “match-outcome” models in `models/predictor.py` or `models/match_predictor.py`.

### 3.3 `pre_match_clean.csv`
Built by `pipeline/pre_matches_features.py`.

Shape: one row per match with **strictly pre-match** features (no target leakage).
Includes:
- ELO ratings at match time
- Rolling form (win rate last N matches)
- Head-to-head recent win rate
- Venue aggregates
- Toss encoding (toss winner/decision from info file)

This is the main dataset for the pre-match probability model.

### 3.4 `data/processed/models/pre_match_walkforward_predictions.csv`
Built by `models/pre_match_model.py`.

Shape: one row per match, containing:
- `p_team1_win` predicted for that match
- Only produced via **walk-forward** evaluation (train on past seasons → predict next season)

This file is the safest starting point for any ROI backtest or calibration analysis.

---

## 4) Team name normalization

Team strings can vary (full name, abbreviations, historical names). We normalize with:
- `pipeline/team_name_standardizer.py`

Important: defunct teams are kept distinct (e.g. Deccan Chargers → `DEC`) to avoid silently merging different franchises.

---

## 5) Modeling philosophy (what “correct” means here)

For betting / odds use-cases, the goal isn’t just accuracy; it’s **probability quality**:

- **Calibration**: when the model says 0.60, it should win ~60% of the time.
- **Log loss / Brier** matter more than accuracy.
- Validation must be **time-aware** (no mixing future seasons into training).

---

## 6) The calibrated pre-match model (`models/pre_match_model.py`)

### 6.1 Inputs
From `pre_match_clean.csv`:

- Categorical:
  - `team1`, `team2`, `venue` (one-hot encoded)
- Numeric:
  - ELO, form, SR, venue stats, head-to-head, toss encodings

### 6.2 Validation (walk-forward)
For each season \(s\):
- Train on all seasons \< \(s\)
- Predict season \(s\)

This produces true “as-if-deployed” predictions without leakage.

### 6.3 Probability calibration
We use isotonic calibration via `CalibratedClassifierCV` with `TimeSeriesSplit` internally, so predicted probabilities are smoother and more usable for odds comparisons.

### 6.4 Outputs
Saved to `data/processed/models/`:
- `pre_match_model.joblib` (final trained model)
- `pre_match_metrics.json` (walk-forward metrics)
- `pre_match_walkforward_predictions.csv` (per-match preds for analysis/backtest)

Run:

```powershell
python strike.py train-prematch
```

---

## 7) Proper market → match mapping (`pipeline/value_bets.py`)

### The problem
If you just map odds to “the last time these teams played historically”, you’ll attach the wrong match context to the odds.

### The fix
We store market timing metadata from Polymarket, and map by:
- matching teams (`team_a`/`team_b`) to a matchup in `pre_match_clean.csv`
- selecting the row whose `start_date` is **closest** to the market timestamp (UTC-safe)

This is still “best-effort” because:
- Your dataset is historical (Cricsheet) rather than a true future fixture list.
- For perfect mapping to upcoming matches, we’d ingest an authoritative fixtures feed.

---

## 8) Value bets table (`pipeline/value_bets.py`)

Given:
- Polymarket implied probabilities \(p_{mkt}\)
- Model probability \(p_{model}\)

We compute:
- **edge** = \(p_{model} - p_{mkt}\)
- **implied decimal odds** = \(1 / p_{mkt}\)
- **fractional Kelly sizing** (conservative)

Run:

```powershell
python strike.py fetch-odds
python strike.py value-bets
```

Outputs:
- `data/processed/value_bets.csv`

---

## 9) ROI backtest + calibration plots (`models/backtest.py`)

### 9.1 Why synthetic odds?
We do not yet have historical closing lines for IPL markets stored over time. Without real lines, we can still:
- test bankroll mechanics
- compare model probabilities vs a baseline “book” proxy

### 9.2 Book proxy used
We use an **ELO-only bookmaker probability** + a margin (overround), then convert to decimal odds.

This creates a “reasonable” opponent to the model and avoids the degenerate case where book odds equal model odds.

### 9.3 Betting rule
For each match:
- compute edge vs book
- bet the side with bigger edge if `edge >= min_edge`
- size stake via fractional Kelly with caps (`kelly_scale`, `max_bet_frac`)

### 9.4 Outputs
Saved to `data/processed/models/`:
- `backtest_trades.csv`
- `backtest_summary.json`
- `calibration_bins.csv`
- `calibration_curve.png`
- `backtest_bankroll.png`

Run:

```powershell
python strike.py backtest
```

---

## 10) CLI commands (single entrypoint)

```powershell
python strike.py pipeline
python strike.py train-prematch
python strike.py fetch-odds
python strike.py value-bets
python strike.py backtest
```

Important: `strike.py` will automatically use `.\.venv\Scripts\python.exe` if present, so you won’t get “missing numpy/pandas” errors when calling it with a global Python.

---

## 11) What “world-class predictor” would require next

This repo is now a solid foundation. To push it toward “best-in-world”, you’d add:

### 11.1 Real historical odds (critical)
- Snapshot Polymarket markets periodically and store:
  - timestamped prices
  - matched fixture (teams/date)
- Backtest using true pre-game odds instead of synthetic ELO-book.

### 11.2 Feature upgrades (highest impact)
- **Player availability**: injuries, rest, lineup news (API or curated sources)
- **Rest days / travel**: days since last match; travel distance; venue changes
- **Venue dynamics**: chase advantage by season and month; dew proxies
- **Recency decay**: exponential weighting so latest season signals dominate

### 11.3 Model upgrades
- Add **CatBoost/LightGBM** as optional model backends.
- Use probability calibration (keep it).
- Track:
  - log loss, Brier, calibration error, profitability metrics

### 11.4 Production polish
- Automated runs (daily/cron)
- Data validation checks (schema, missing columns, drift)
- Unit tests for parsing + key transforms

---

## 12) Known limitations (honest)

- `value_bets.csv` mapping is improved but still best-effort without a fixtures feed.
- X/Twitter scraping is unstable by nature.
- ROI backtest uses synthetic odds until you store real historical odds snapshots.

