"""
Central configuration for whale_signals.

All tuneable constants and environment-variable reads live here.
Import this module anywhere you need a threshold, path, or API credential.

Reads .env automatically via python-dotenv — never hardcode credentials.
"""

import os
from pathlib import Path

from dotenv import load_dotenv  # reads key=value pairs from .env into os.environ

load_dotenv()  # must be called before any os.getenv() calls below

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT_DIR: Path = Path(__file__).parent  # the repo root (where this file lives)

RAW_DATA_DIR: Path = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR: Path = ROOT_DIR / "data" / "processed"

# ---------------------------------------------------------------------------
# Dune Analytics API
# ---------------------------------------------------------------------------

DUNE_BASE_URL: str = "https://api.dune.com/api/v1"

# os.getenv() returns None if the variable is missing (unlike os.environ[] which raises).
# We use empty-string defaults here so that importing config in tests does not crash
# even when .env is absent. Functions that actually call the API validate these at
# call time and raise clearly if they are blank.
DUNE_API_KEY: str = os.getenv("DUNE_API_KEY", "")
DUNE_QUERY_ID: int = int(os.getenv("DUNE_QUERY_ID", "0"))

# ---------------------------------------------------------------------------
# Whale filter parameters
# ---------------------------------------------------------------------------

# Minimum USD value (at transaction time) to count a transaction as a "whale" event.
# Configurable so we can easily test $100k, $500k, $1M thresholds later.
WHALE_USD_THRESHOLD: float = float(os.getenv("WHALE_USD_THRESHOLD", "1_000_000"))

# Conservative ETH pre-filter passed to Dune SQL to reduce rows scanned before
# the ETH/USD price join resolves the exact USD value.
#
# Reasoning: ETH's lowest price in our target window (2023-2024) was ~$1,200.
# At $1,200/ETH, $1M USD = 833 ETH. Setting the pre-filter at 200 ETH captures
# everything that could possibly be above $1M at any point in the window, with
# some false positives that the USD filter then removes. Erring lower is safe —
# a higher pre-filter risks silently dropping valid whale transactions.
WHALE_ETH_PREFILTER: float = 200.0

# ---------------------------------------------------------------------------
# Fetch window
# ---------------------------------------------------------------------------

# Two full calendar years: one bear-ish period (2023) + one bull period (2024).
# Having both regimes is important for walk-forward validation robustness.
# Stored as strings because they are passed directly to the Dune SQL as parameters.
FETCH_START_DATE: str = "2023-01-01"
FETCH_END_DATE: str = "2025-01-01"