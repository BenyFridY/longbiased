"""Shared pytest setup for the longbiased-beny test suite.

Puts the repo root AND scripts/production on sys.path so tests can import both
`scripts.production.*` (used by generate_signal/bootstrap/build_features) and the
bare `config`/`generate_signal` modules (used by walkforward_backtest.py).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "scripts" / "production")):
    if p not in sys.path:
        sys.path.insert(0, p)


def repo_root() -> Path:
    return ROOT
