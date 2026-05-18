"""Legacy entrypoint — delegates to edge_engine (canonical scan pipeline)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from pipeline.edge_engine import run_scan


def main():
    run_scan(fetch=False, min_edge=0.0, log=True)


if __name__ == "__main__":
    main()
