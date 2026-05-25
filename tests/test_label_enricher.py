"""
Tests for src/data/label_enricher.py.

All tests use in-memory DataFrames — no file I/O except where we
explicitly test load_labels() using a temporary CSV.
"""

import pytest
import pandas as pd
from pathlib import Path

from src.data.label_enricher import (
    load_labels,
    enrich_with_labels,
    summarise_label_coverage,
    UNKNOWN,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_label_csv(tmp_path: Path, rows: list[dict]) -> Path:
    """Write a minimal labels CSV to a temp file and return its path."""
    df = pd.DataFrame(rows)
    path = tmp_path / "labels.csv"
    df.to_csv(path, index=False)
    return path


def make_whale_df(from_addresses: list[str], to_addresses: list[str]) -> pd.DataFrame:
    """Return a minimal whale transaction DataFrame for testing."""
    assert len(from_addresses) == len(to_addresses)
    return pd.DataFrame({
        "from_address": from_addresses,
        "to_address": to_addresses,
        "usd_value": [1_500_000.0] * len(from_addresses),
    })


# ---------------------------------------------------------------------------
# load_labels
# ---------------------------------------------------------------------------

class TestLoadLabels:
    def test_strips_0x_prefix(self, tmp_path: Path):
        path = make_label_csv(tmp_path, [
            {"address": "0xAbCd1234", "label": "Test Exchange", "category": "exchange"},
        ])
        labels = load_labels(path)
        # After normalisation the address should have no 0x and be lowercase
        assert labels["address"].iloc[0] == "abcd1234"

    def test_lowercases_address(self, tmp_path: Path):
        path = make_label_csv(tmp_path, [
            {"address": "ABCD1234", "label": "X", "category": "exchange"},
        ])
        labels = load_labels(path)
        assert labels["address"].iloc[0] == "abcd1234"

    def test_raises_on_missing_column(self, tmp_path: Path):
        # CSV missing the 'category' column
        df = pd.DataFrame([{"address": "0x1234", "label": "X"}])
        path = tmp_path / "bad.csv"
        df.to_csv(path, index=False)

        with pytest.raises(ValueError, match="category"):
            load_labels(path)

    def test_returns_expected_columns(self, tmp_path: Path):
        path = make_label_csv(tmp_path, [
            {"address": "0x1234", "label": "Exchange A", "category": "exchange"},
        ])
        labels = load_labels(path)
        assert set(labels.columns) == {"address", "label", "category"}


# ---------------------------------------------------------------------------
# enrich_with_labels
# ---------------------------------------------------------------------------

class TestEnrichWithLabels:
    def _make_labels(self) -> pd.DataFrame:
        """Minimal in-memory label table — no file I/O needed."""
        return pd.DataFrame({
            "address": [
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",  # exchange address
                "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",    # defi address
            ],
            "label": ["Big Exchange", "DeFi Protocol"],
            "category": ["exchange", "defi"],
        })

    def test_known_from_address_gets_label(self):
        labels = self._make_labels()
        df = make_whale_df(
            from_addresses=["aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"],
            to_addresses=["cccccccccccccccccccccccccccccccccccccccc"],
        )
        result = enrich_with_labels(df, labels)
        assert result["from_label"].iloc[0] == "Big Exchange"
        assert result["from_category"].iloc[0] == "exchange"

    def test_known_to_address_gets_label(self):
        labels = self._make_labels()
        df = make_whale_df(
            from_addresses=["cccccccccccccccccccccccccccccccccccccccc"],
            to_addresses=["bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"],
        )
        result = enrich_with_labels(df, labels)
        assert result["to_label"].iloc[0] == "DeFi Protocol"
        assert result["to_category"].iloc[0] == "defi"

    def test_unknown_address_gets_unknown_sentinel(self):
        labels = self._make_labels()
        df = make_whale_df(
            from_addresses=["dddddddddddddddddddddddddddddddddddddddd"],
            to_addresses=["eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"],
        )
        result = enrich_with_labels(df, labels)
        assert result["from_label"].iloc[0] == UNKNOWN
        assert result["from_category"].iloc[0] == UNKNOWN
        assert result["to_label"].iloc[0] == UNKNOWN
        assert result["to_category"].iloc[0] == UNKNOWN

    def test_adds_exactly_four_columns(self):
        labels = self._make_labels()
        df = make_whale_df(["aaa" * 14], ["bbb" * 13 + "bb"])
        original_cols = set(df.columns)
        result = enrich_with_labels(df, labels)
        new_cols = set(result.columns) - original_cols
        assert new_cols == {"from_label", "from_category", "to_label", "to_category"}

    def test_row_count_unchanged(self):
        labels = self._make_labels()
        df = make_whale_df(
            from_addresses=["aaa" * 14, "bbb" * 13 + "bb"],
            to_addresses=["ccc" * 14, "ddd" * 13 + "dd"],
        )
        result = enrich_with_labels(df, labels)
        # Left join must never duplicate or drop rows
        assert len(result) == len(df)

    def test_does_not_mutate_input(self):
        labels = self._make_labels()
        df = make_whale_df(["aaa" * 14], ["bbb" * 13 + "bb"])
        original_cols = list(df.columns)
        enrich_with_labels(df, labels)
        assert list(df.columns) == original_cols  # input unchanged


# ---------------------------------------------------------------------------
# summarise_label_coverage  (smoke test — just confirm it does not crash)
# ---------------------------------------------------------------------------

class TestSummariseLabelCoverage:
    def test_runs_without_error(self, capsys: pytest.CaptureFixture):
        labels = pd.DataFrame({
            "address": ["aaaa"],
            "label": ["Exchange A"],
            "category": ["exchange"],
        })
        df = make_whale_df(["aaaa"], ["bbbb"])
        enriched = enrich_with_labels(df, labels)
        summarise_label_coverage(enriched)  # should not raise
        output = capsys.readouterr().out
        assert "Total transactions" in output
