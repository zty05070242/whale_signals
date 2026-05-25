"""
Address label enricher.

Loads a curated CSV of known Ethereum addresses (exchanges, DeFi protocols)
and joins those labels onto a whale transaction DataFrame.

The output adds four columns:
  from_label     — human-readable name of the sending address, or 'unknown'
  from_category  — 'exchange', 'defi', or 'unknown'
  to_label       — human-readable name of the receiving address, or 'unknown'
  to_category    — 'exchange', 'defi', or 'unknown'

These four columns are the primary inputs to the Phase 2 transaction classifier.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

import config

LABELS_PATH: Path = config.ROOT_DIR / "data" / "reference" / "address_labels.csv"

# Sentinel value for addresses not found in the label table.
# Using a string rather than NaN makes downstream groupby and filtering simpler.
UNKNOWN = "unknown"


def load_labels(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the curated address label CSV and normalise the address column.

    The CSV stores addresses in conventional 0x-prefixed checksummed format.
    Dune outputs addresses as lowercase hex without the 0x prefix. This
    function normalises to lowercase-no-0x so the join works without any
    transformation on the whale DataFrame side.

    Parameters
    ----------
    path : Path, optional
        Path to the CSV. Defaults to data/reference/address_labels.csv.

    Returns
    -------
    pd.DataFrame
        Columns: address (normalised), label (str), category (str).
    """
    csv_path = path or LABELS_PATH

    # pd.read_csv loads the file into a DataFrame with column names from the header row
    labels = pd.read_csv(csv_path, dtype=str)  # dtype=str prevents any type coercion

    _validate_label_schema(labels)

    # Normalise addresses: strip leading '0x' if present, then lowercase.
    # str.lower() lowercases the whole string.
    # str.replace with regex=True removes the '0x' prefix if present.
    labels["address"] = (
        labels["address"]
        .str.lower()
        .str.replace(r"^0x", "", regex=True)
    )

    return labels


def enrich_with_labels(
    df: pd.DataFrame,
    labels: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Add from_label, from_category, to_label, to_category columns to the
    whale transaction DataFrame.

    Addresses not found in the label table receive 'unknown' for both
    label and category columns.

    Parameters
    ----------
    df : pd.DataFrame
        Output of fetch_whale_transactions() or _parse_types().
    labels : pd.DataFrame, optional
        Output of load_labels(). Loaded from default path if not provided.

    Returns
    -------
    pd.DataFrame
        df with four additional columns. Row count is unchanged.
    """
    if labels is None:
        labels = load_labels()

    df = df.copy()  # do not mutate the input

    # --- Enrich the 'from' side ---
    # pd.merge with how='left' keeps all rows from the left DataFrame (df).
    # Rows in df whose from_address is not in labels get NaN for label/category.
    # We rename the label columns before merging to avoid collisions with the
    # second merge on the 'to' side.
    from_labels = labels.rename(
        columns={"address": "from_address", "label": "from_label", "category": "from_category"}
    )
    df = df.merge(from_labels, on="from_address", how="left")

    # --- Enrich the 'to' side ---
    to_labels = labels.rename(
        columns={"address": "to_address", "label": "to_label", "category": "to_category"}
    )
    df = df.merge(to_labels, on="to_address", how="left")

    # --- Fill unknown ---
    # fillna replaces NaN with the UNKNOWN sentinel string.
    # Addresses not in the label table produce NaN from the left join above.
    for col in ("from_label", "from_category", "to_label", "to_category"):
        df[col] = df[col].fillna(UNKNOWN)

    return df


def summarise_label_coverage(df: pd.DataFrame) -> None:
    """
    Print a quick breakdown of label coverage for both sides of transactions.

    Useful for a sanity check after enrichment — tells you what fraction of
    whale transaction endpoints are recognised vs unknown.
    """
    total = len(df)
    from_known = (df["from_category"] != UNKNOWN).sum()
    to_known = (df["to_category"] != UNKNOWN).sum()

    print(f"Total transactions: {total:,}")
    print(f"From-address labelled: {from_known:,} ({from_known / total:.1%})")
    print(f"To-address labelled:   {to_known:,} ({to_known / total:.1%})")
    print()

    # value_counts() counts how many times each unique category appears
    print("From-category breakdown:")
    print(df["from_category"].value_counts().to_string())
    print()
    print("To-category breakdown:")
    print(df["to_category"].value_counts().to_string())


def _validate_label_schema(labels: pd.DataFrame) -> None:
    """Raise if the CSV is missing required columns."""
    required = {"address", "label", "category"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(
            f"address_labels.csv is missing columns: {missing}. "
            "Expected headers: address, label, category."
        )
