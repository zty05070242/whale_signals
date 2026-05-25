"""
Generic Dune Analytics API client.

Dune's API works in three steps:
  1. POST to execute a saved query  → returns an execution_id
  2. GET status until state == COMPLETED (poll in a loop)
  3. GET results/csv to download the data

This module handles all three steps. It has no whale-specific logic —
it is reusable for any Dune query in the project.

Dune API reference: https://docs.dune.com/api-reference/executions/endpoint/execute-query
"""

import time
from io import StringIO
from typing import Optional

import pandas as pd
import requests

import config


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _auth_header() -> dict[str, str]:
    """Return the HTTP header Dune requires on every authenticated request."""
    if not config.DUNE_API_KEY:
        raise ValueError(
            "DUNE_API_KEY is not set. Add it to your .env file."
        )
    return {"X-Dune-API-Key": config.DUNE_API_KEY}


def _infer_dune_param_type(value: object) -> str:
    """
    Map a Python value to one of Dune's three parameter type strings.

    Dune requires each parameter to be tagged with a type so it can
    validate and substitute them safely in the SQL template.
    """
    if isinstance(value, bool):
        # bool must come before int because in Python bool is a subclass of int
        return "enum"
    if isinstance(value, (int, float)):
        return "number"
    return "text"  # covers strings, dates formatted as "YYYY-MM-DD", etc.


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def execute_query(
    query_id: int,
    parameters: Optional[dict] = None,
) -> str:
    """
    Tell Dune to start executing a saved query.

    Parameters
    ----------
    query_id : int
        The integer ID in the Dune query URL
        (e.g. dune.com/queries/3456789 → query_id=3456789).
    parameters : dict, optional
        Maps parameter name to value. Names must match those defined in the
        Dune query UI. Values are cast to strings before sending.

    Returns
    -------
    str
        The execution_id string, used to poll status and fetch results.
    """
    url = f"{config.DUNE_BASE_URL}/query/{query_id}/execute"

    body: dict = {}
    if parameters:
        # Dune expects a list of dicts, one per parameter
        body["query_parameters"] = [
            {
                "name": name,
                "type": _infer_dune_param_type(value),
                "value": str(value),  # Dune always expects a string-encoded value
            }
            for name, value in parameters.items()
        ]

    response = requests.post(url, json=body, headers=_auth_header(), timeout=30)
    # raise_for_status() converts HTTP 4xx/5xx responses into Python exceptions,
    # so callers do not need to inspect the status code themselves
    response.raise_for_status()

    return response.json()["execution_id"]


def poll_until_complete(
    execution_id: str,
    poll_interval_seconds: int = 5,
    timeout_seconds: int = 600,
) -> None:
    """
    Repeatedly check Dune's status endpoint until the query finishes.

    Think of this as waiting for a render to complete in audio software —
    you poll a progress indicator and block until the job is done.

    Parameters
    ----------
    execution_id : str
        Returned by execute_query().
    poll_interval_seconds : int
        Seconds to wait between status checks. 5s is a reasonable default
        for queries that typically take 30-120 seconds on Dune free tier.
    timeout_seconds : int
        Maximum total wait time before giving up. Default 10 minutes.

    Raises
    ------
    RuntimeError
        If Dune reports the query failed.
    TimeoutError
        If the query does not complete within timeout_seconds.
    """
    url = f"{config.DUNE_BASE_URL}/execution/{execution_id}/status"
    elapsed = 0

    while elapsed < timeout_seconds:
        response = requests.get(url, headers=_auth_header(), timeout=30)
        response.raise_for_status()

        state: str = response.json()["state"]

        if state == "QUERY_STATE_COMPLETED":
            return

        if state == "QUERY_STATE_FAILED":
            raise RuntimeError(
                f"Dune execution '{execution_id}' failed. "
                "Check the Dune UI for error details."
            )

        # Any other state (PENDING, EXECUTING) — wait and try again
        print(f"  Dune status: {state} ({elapsed}s elapsed)...")
        time.sleep(poll_interval_seconds)
        elapsed += poll_interval_seconds

    raise TimeoutError(
        f"Dune execution '{execution_id}' did not complete within {timeout_seconds}s. "
        "The query may be too large for the free tier — consider narrowing the date range."
    )


def fetch_results_csv(execution_id: str) -> pd.DataFrame:
    """
    Download the results of a completed Dune execution as a DataFrame.

    Uses the /results/csv endpoint, which streams a plain CSV text response.
    This is simpler and more memory-efficient than the JSON results endpoint
    for large result sets.

    Parameters
    ----------
    execution_id : str

    Returns
    -------
    pd.DataFrame
        All columns as strings — type parsing is done by the caller.
    """
    url = f"{config.DUNE_BASE_URL}/execution/{execution_id}/results/csv"
    response = requests.get(url, headers=_auth_header(), timeout=120)
    response.raise_for_status()

    # StringIO wraps the raw CSV text string so pd.read_csv can treat it
    # as a file-like object — avoids writing to disk first
    return pd.read_csv(StringIO(response.text))


def run_query(
    query_id: int,
    parameters: Optional[dict] = None,
) -> pd.DataFrame:
    """
    High-level convenience function: execute → poll → fetch in one call.

    Parameters
    ----------
    query_id : int
        Dune query ID.
    parameters : dict, optional
        Query parameters to pass. See execute_query() for format.

    Returns
    -------
    pd.DataFrame
        Raw results with all columns as strings.
    """
    print(f"Submitting Dune query {query_id} with parameters: {parameters}")
    execution_id = execute_query(query_id, parameters)

    print(f"Execution started (id={execution_id}). Polling for completion...")
    poll_until_complete(execution_id)

    print("Query complete. Downloading results...")
    df = fetch_results_csv(execution_id)
    print(f"Downloaded {len(df):,} rows, {len(df.columns)} columns.")

    return df