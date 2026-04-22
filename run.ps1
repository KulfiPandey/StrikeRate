param(
  [ValidateSet("download","build-master","match-features","pre-match","train-honest","train-match","all")]
  [string]$Step = "all"
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"

function Ensure-Venv {
  if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    Write-Host "Creating venv at .\.venv ..."
    python -m venv .venv
  }
  . .\.venv\Scripts\Activate.ps1
  python -m pip install --upgrade pip
  pip install -r requirements.txt
}

Ensure-Venv

switch ($Step) {
  "download"      { python pipeline\ingest.py; break }
  "build-master"  { python pipeline\processor.py; break }
  "match-features"{ python pipeline\match_features.py; break }
  "pre-match"     { python pipeline\pre_matches_features.py; break }
  "train-honest"  { python models\honest_predictor.py; break }
  "train-match"   { python models\match_predictor.py; break }
  "all" {
    python pipeline\processor.py
    python pipeline\match_features.py
    python pipeline\pre_matches_features.py
    python models\honest_predictor.py
    break
  }
}

