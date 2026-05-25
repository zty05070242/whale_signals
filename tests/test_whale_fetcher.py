"""
Tests for src/data/whale_fetcher.py.

Focus: _validate_schema and _parse_types are pure functions (input DataFrame
→ output DataFrame) and can be tested without any mocking.

fetch_whale_transactions() is tested with a mocked run_query() call.
"""

import pytest
from unittest.mock import patch
from pathlib import Path

import pandas as pd
import numpy as np

import config
config.DUNE_API_KEY = "test-key"
config.DUNE_QUERY_ID = 1

from src.data.whale_fetcher import (
    _validate_schema,
    _parse_types,
    fetch_whale_transactions,
    save_whale_transactions,
    EXPECTED_COLUMNS,
)


# ---------------------------------------------------------------------------
# Fixtures — reusable test data
# ---------------------------------------------------------------------------

def make_raw_dune_row(**overrides) -> dict:
    """
    Return a dict representing one row as Dune CSV would return it
    (all values are strings, matching what pd.read_csv produces from the
    /results/csv endpoint).
    """
    defaults = {
        "timestamp_utc": "2023-06-15 10:23:01+00:00",
        "block_number": "17551234",
        "tx_hash": "0xabc123",
        "from_address": "0xfrom",
        "to_address": "0xto",
        "eth_value": "847.3",
        "usd_value": "1524140.0",
        "eth_usd_price": "1799.8",
        "gas_price_gwei": "42.1",
        "gas_used": "21000",
        "tx_fee_eth": "0.000885",
        "is_contract_call": "False",
    }
    return {**defaults, **overrides}


def make_raw_df(**overrides) -> pd.DataFrame:
    """Return a single-row DataFrame with all expected columns."""
    return pd.DataFrame([make_raw_dune_row(**overrides)])


# ---------------------------------------------------------------------------
# _validate_schema
# ---------------------------------------------------------------------------

class TestValidateSchema:
    def test_passes_with_all_expected_columns(self):
        df = make_raw_df()
        result = _validate_schema(df)
        # All expected columns should be present in the result
        assert EXPECTED_COLUMNS.issubset(set(result.columns))

    def test_raises_on_missing_column(self):
        df = make_raw_df()
        df = df.drop(columns=["tx_hash"])  # remove a required column

        with pytest.raises(ValueError, match="tx_hash"):
            _validate_schema(df)

    def test_drops_extra_columns_from_dune(self):
        df = make_raw_df()
        df["dune_internal_column"] = "noise"  # simulate an unexpected extra column

        result = _validate_schema(df)
        assert "dune_internal_column" not in result.columns

    def test_raises_on_completely_empty_dataframe(self):
        # An empty DataFrame with no columns at all should fail validation
        with pytest.raises(ValueError):
            _validate_schema(pd.DataFrame())


# ---------------------------------------------------------------------------
# _parse_types
# ---------------------------------------------------------------------------

class TestParseTypes:
    def test_timestamp_is_utc_aware(self):
        df = _parse_types(make_raw_df())
        ts = df["timestamp_utc"].iloc[0]
        # pd.Timestamp tzinfo should be UTC, not None
        assert ts.tzinfo is not None
        assert str(ts.tzinfo) in ("UTC", "pytz.UTC", "datetime.timezone.utc")

    def test_numeric_columns_are_float(self):
        df = _parse_types(make_raw_df())
        for col in ("eth_value", "usd_value", "eth_usd_price", "gas_price_gwei", "tx_fee_eth"):
            assert pd.api.types.is_float_dtype(df[col]), f"{col} should be float"

    def test_integer_columns_are_int(self):
        df = _parse_types(make_raw_df())
        for col in ("block_number", "gas_used"):
            assert pd.api.types.is_integer_dtype(df[col]), f"{col} should be integer"

    def test_is_contract_call_false(self):
        df = _parse_types(make_raw_df(is_contract_call="False"))
        # pandas stores booleans as numpy.bool_, not Python bool — use == not is
        assert df["is_contract_call"].iloc[0] == False

    def test_is_contract_call_true(self):
        df = _parse_types(make_raw_df(is_contract_call="True"))
        assert df["is_contract_call"].iloc[0] == True

    def test_bad_numeric_becomes_nan_not_exception(self):
        # If Dune returns a corrupted value, _parse_types should produce NaN,
        # not crash. Upstream validation can then decide how to handle NaN rows.
        df = _parse_types(make_raw_df(eth_value="NOT_A_NUMBER"))
        assert pd.isna(df["eth_value"].iloc[0])

    def test_does_not_mutate_input_dataframe(self):
        # _parse_types makes a copy internally. Verify the input is unchanged.
        raw = make_raw_df()
        original_dtype = raw["eth_value"].dtype
        _parse_types(raw)
        assert raw["eth_value"].dtype == original_dtype  # still a string/object dtype


# ---------------------------------------------------------------------------
# fetch_whale_transactions (integration-level, mocked run_query)
# ---------------------------------------------------------------------------

class TestFetchWhaleTransactions:
    def test_returns_typed_dataframe(self):
        raw_dune_output = make_raw_df()  # simulates what run_query() returns

        with patch("src.data.whale_fetcher.run_query", return_value=raw_dune_output):
            df = fetch_whale_transactions()

        # After fetching, timestamp_utc should be a proper datetime, not a string
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp_utc"])

    def test_raises_when_query_id_not_set(self):
        original = config.DUNE_QUERY_ID
        config.DUNE_QUERY_ID = 0
        try:
            with pytest.raises(ValueError, match="DUNE_QUERY_ID"):
                fetch_whale_transactions()
        finally:
            config.DUNE_QUERY_ID = original


# ---------------------------------------------------------------------------
# save_whale_transactions
# ---------------------------------------------------------------------------

class TestSaveWhaleTransactions:
    def test_saves_csv_to_specified_path(self, tmp_path: Path):
        # pytest's tmp_path fixture provides a temporary directory that is
        # automatically cleaned up after the test — no need to delete manually
        df = _parse_types(make_raw_df())
        output_file = tmp_path / "test_output.csv"

        returned_path = save_whale_transactions(df, output_path=output_file)

        assert returned_path == output_file
        assert output_file.exists()

    def test_saved_csv_is_readable(self, tmp_path: Path):
        df = _parse_types(make_raw_df())
        output_file = tmp_path / "output.csv"
        save_whale_transactions(df, output_path=output_file)

        # Read it back and confirm we can reconstruct the DataFrame
        reloaded = pd.read_csv(output_file)
        assert len(reloaded) == len(df)
        assert "tx_hash" in reloaded.columns

    def test_creates_parent_directories_if_missing(self, tmp_path: Path):
        df = _parse_types(make_raw_df())
        nested_path = tmp_path / "nested" / "deep" / "output.csv"
        # The parent directories do not exist yet
        assert not nested_path.parent.exists()

        save_whale_transactions(df, output_path=nested_path)
        assert nested_path.exists()