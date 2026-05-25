"""
MEV (Maximum Extractable Value) bot filter.

MEV bots execute large transactions for automated profit extraction —
sandwich attacks, DEX arbitrage, liquidations. They appear in our data
as "whale" transactions but carry no directional price signal.

This module flags MEV candidates rather than deleting them, so that
Phase 4 can run sensitivity analyses with and without these rows.

Two flagging mechanisms:
  1. Known address list  — precise, but the list is never exhaustive
                           (bots deploy new contracts constantly)
  2. Gas price heuristic — catches unknown bots, but has false positives
                           (legitimate urgent transactions also pay high gas)

To extend the known address list, the best source is Dune's Flashbots tables:
  SELECT DISTINCT sandwicher_eoa
  FROM flashbots.sandwiched_swaps
Export and append to data/reference/mev_addresses.csv.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

import config

MEV_ADDRESSES_PATH: Path = config.ROOT_DIR / "data" / "reference" / "mev_addresses.csv"

# Default gas price percentile threshold for the heuristic flag.
# Transactions above this percentile AND calling a contract are flagged.
# 99.0 means only the top 1% of gas payers are considered suspicious.
# Lower values increase recall (catch more bots) but reduce precision (more false positives).
DEFAULT_GAS_PERCENTILE: float = 99.0


def load_mev_addresses(path: Optional[Path] = None) -> set[str]:
    """
    Load the curated MEV bot address list and return as a set of normalised
    addresses (lowercase, no 0x prefix) for fast membership testing.

    Parameters
    ----------
    path : Path, optional
        Defaults to data/reference/mev_addresses.csv.

    Returns
    -------
    set[str]
        Normalised addresses of known MEV bots.
    """
    csv_path = path or MEV_ADDRESSES_PATH

    df = pd.read_csv(csv_path, dtype=str)

    if "address" not in df.columns:
        raise ValueError(
            "mev_addresses.csv must have an 'address' column. "
            f"Found: {list(df.columns)}"
        )

    # Normalise to lowercase without 0x prefix — matches Dune output format
    addresses = (
        df["address"]
        .str.lower()
        .str.replace(r"^0x", "", regex=True)
    )
    return set(addresses)


def flag_mev_candidates(
    df: pd.DataFrame,
    mev_addresses: Optional[set[str]] = None,
    gas_percentile: float = DEFAULT_GAS_PERCENTILE,
) -> pd.DataFrame:
    """
    Add is_mev_candidate and mev_flag_reason columns to the whale DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Output of enrich_with_labels() or fetch_whale_transactions().
    mev_addresses : set[str], optional
        Set of known MEV bot addresses (normalised). Loaded from CSV if not provided.
    gas_percentile : float
        Gas price percentile above which a contract-calling transaction is
        considered a potential MEV candidate. Default 99.0 (top 1%).

    Returns
    -------
    pd.DataFrame
        df with two additional columns:
          is_mev_candidate  bool    True if either flag fires
          mev_flag_reason   str     'known_address', 'high_gas_heuristic',
                                    'both', or 'none'
    """
    if mev_addresses is None:
        mev_addresses = load_mev_addresses()

    df = df.copy()

    # --- Flag 1: known bot address ---
    # Series.isin() returns a boolean Series: True where the value is in the set.
    is_known_bot = df["from_address"].isin(mev_addresses)

    # --- Flag 2: gas price heuristic ---
    # Series.quantile(q) returns the value at the q-th quantile (0.0–1.0).
    # e.g. quantile(0.99) returns the value such that 99% of values are below it.
    gas_threshold = df["gas_price_gwei"].quantile(gas_percentile / 100.0)

    # Both conditions must hold: very high gas AND calling a contract.
    # A plain ETH transfer at high gas is suspicious but less diagnostic than
    # a high-gas contract call, which is the canonical MEV pattern.
    is_high_gas_contract = (df["gas_price_gwei"] > gas_threshold) & df["is_contract_call"]

    # --- Combine and label the reason ---
    df["is_mev_candidate"] = is_known_bot | is_high_gas_contract

    # Track which flag(s) fired for each row — useful for sensitivity analysis.
    # Start with 'none' and overwrite in order of specificity.
    df["mev_flag_reason"] = "none"
    df.loc[is_high_gas_contract, "mev_flag_reason"] = "high_gas_heuristic"
    df.loc[is_known_bot, "mev_flag_reason"] = "known_address"
    # If both fired, the known-address flag is more definitive — override.
    df.loc[is_known_bot & is_high_gas_contract, "mev_flag_reason"] = "both"

    return df


def summarise_mev_flags(df: pd.DataFrame) -> None:
    """
    Print a breakdown of MEV candidate flags.

    Call after flag_mev_candidates() to understand how many transactions
    are flagged and by which mechanism.
    """
    total = len(df)
    flagged = df["is_mev_candidate"].sum()

    print(f"Total transactions:   {total:,}")
    print(f"MEV candidates:       {flagged:,} ({flagged / total:.1%})")
    print()
    print("Flag reason breakdown:")
    print(df["mev_flag_reason"].value_counts().to_string())
