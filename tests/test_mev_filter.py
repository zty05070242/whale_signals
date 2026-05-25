"""
Tests for src/data/mev_filter.py.
"""

import pytest
import pandas as pd
from pathlib import Path

from src.data.mev_filter import (
    load_mev_addresses,
    flag_mev_candidates,
    DEFAULT_GAS_PERCENTILE,
)

# A fake known bot address in normalised format (no 0x, lowercase)
BOT_ADDRESS = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
NORMAL_ADDRESS = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_mev_csv(tmp_path: Path, addresses: list[str]) -> Path:
    df = pd.DataFrame({"address": addresses, "label": ["bot"] * len(addresses)})
    path = tmp_path / "mev.csv"
    df.to_csv(path, index=False)
    return path


def make_df(
    from_address: str = NORMAL_ADDRESS,
    gas_price_gwei: float = 20.0,
    is_contract_call: bool = False,
    n_rows: int = 1,
) -> pd.DataFrame:
    return pd.DataFrame({
        "from_address": [from_address] * n_rows,
        "gas_price_gwei": [gas_price_gwei] * n_rows,
        "is_contract_call": [is_contract_call] * n_rows,
    })


# ---------------------------------------------------------------------------
# load_mev_addresses
# ---------------------------------------------------------------------------

class TestLoadMevAddresses:
    def test_returns_a_set(self, tmp_path: Path):
        path = make_mev_csv(tmp_path, ["0xABCD"])
        result = load_mev_addresses(path)
        assert isinstance(result, set)

    def test_strips_0x_and_lowercases(self, tmp_path: Path):
        path = make_mev_csv(tmp_path, ["0xABCDEF"])
        result = load_mev_addresses(path)
        assert "abcdef" in result

    def test_raises_on_missing_address_column(self, tmp_path: Path):
        df = pd.DataFrame({"wrong_col": ["0xABCD"]})
        path = tmp_path / "bad.csv"
        df.to_csv(path, index=False)
        with pytest.raises(ValueError, match="address"):
            load_mev_addresses(path)


# ---------------------------------------------------------------------------
# flag_mev_candidates
# ---------------------------------------------------------------------------

class TestFlagMevCandidates:
    def test_known_bot_address_is_flagged(self):
        df = make_df(from_address=BOT_ADDRESS)
        result = flag_mev_candidates(df, mev_addresses={BOT_ADDRESS})
        assert result["is_mev_candidate"].iloc[0] == True
        assert result["mev_flag_reason"].iloc[0] == "known_address"

    def test_normal_address_is_not_flagged(self):
        df = make_df(from_address=NORMAL_ADDRESS, gas_price_gwei=20.0)
        result = flag_mev_candidates(df, mev_addresses={BOT_ADDRESS})
        assert result["is_mev_candidate"].iloc[0] == False
        assert result["mev_flag_reason"].iloc[0] == "none"

    def test_high_gas_contract_call_is_flagged(self):
        # Build a DataFrame where one row has extreme gas — above the 99th percentile.
        # To guarantee it exceeds the threshold, give 99 normal rows and 1 extreme row.
        n_normal = 99
        normal_rows = make_df(gas_price_gwei=10.0, is_contract_call=True, n_rows=n_normal)
        extreme_row = make_df(gas_price_gwei=10_000.0, is_contract_call=True)
        df = pd.concat([normal_rows, extreme_row], ignore_index=True)

        result = flag_mev_candidates(df, mev_addresses=set(), gas_percentile=99.0)

        # Only the last row (extreme gas) should be flagged
        assert result["is_mev_candidate"].iloc[-1] == True
        assert result["mev_flag_reason"].iloc[-1] == "high_gas_heuristic"
        # Normal rows should not be flagged
        assert result["is_mev_candidate"].iloc[:-1].sum() == 0

    def test_high_gas_plain_transfer_is_not_flagged(self):
        # High gas but is_contract_call=False — should NOT trigger the heuristic.
        # Rationale: plain ETH transfers at high gas are unusual but not diagnostic of MEV.
        n_normal = 99
        normal_rows = make_df(gas_price_gwei=10.0, is_contract_call=False, n_rows=n_normal)
        high_gas_transfer = make_df(gas_price_gwei=10_000.0, is_contract_call=False)
        df = pd.concat([normal_rows, high_gas_transfer], ignore_index=True)

        result = flag_mev_candidates(df, mev_addresses=set(), gas_percentile=99.0)
        assert result["is_mev_candidate"].iloc[-1] == False

    def test_both_flags_fire_reason_is_both(self):
        n_normal = 99
        normal_rows = make_df(gas_price_gwei=10.0, is_contract_call=True, n_rows=n_normal)
        bot_extreme = make_df(
            from_address=BOT_ADDRESS, gas_price_gwei=10_000.0, is_contract_call=True
        )
        df = pd.concat([normal_rows, bot_extreme], ignore_index=True)

        result = flag_mev_candidates(df, mev_addresses={BOT_ADDRESS}, gas_percentile=99.0)
        assert result["mev_flag_reason"].iloc[-1] == "both"

    def test_adds_exactly_two_columns(self):
        df = make_df()
        original_cols = set(df.columns)
        result = flag_mev_candidates(df, mev_addresses=set())
        new_cols = set(result.columns) - original_cols
        assert new_cols == {"is_mev_candidate", "mev_flag_reason"}

    def test_row_count_unchanged(self):
        df = make_df(n_rows=50)
        result = flag_mev_candidates(df, mev_addresses=set())
        assert len(result) == 50

    def test_does_not_mutate_input(self):
        df = make_df()
        original_cols = list(df.columns)
        flag_mev_candidates(df, mev_addresses=set())
        assert list(df.columns) == original_cols
