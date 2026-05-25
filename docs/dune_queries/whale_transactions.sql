-- =============================================================================
-- Whale Transaction Fetcher
-- =============================================================================
-- Purpose: extract all successful ETH transfers above a USD threshold, with
--          the ETH/USD price at transaction time attached.
--
-- Paste this into a new Dune query (dune.com → New Query). Then define these
-- four parameters in the Dune UI (Query → Parameters panel):
--
--   Name                  Type     Default
--   ------------------    ------   ----------
--   start_date            text     2023-01-01
--   end_date              text     2025-01-01
--   whale_usd_threshold   number   1000000
--   eth_prefilter         number   200
--
-- After saving, note the query ID from the URL and add it to .env as DUNE_QUERY_ID.
-- =============================================================================

SELECT
    -- block_time is a UTC timestamp in Dune — this becomes our primary time index
    t.block_time                                                            AS timestamp_utc,

    t.block_number,

    -- to_hex() converts the binary hash to a 0x-prefixed hex string
    -- lower() normalises to lowercase so address comparisons are consistent
    lower(to_hex(t.hash))                                                   AS tx_hash,

    lower(to_hex(t."from"))                                                 AS from_address,

    -- "to" and "from" are reserved words in SQL, hence the double-quotes
    lower(to_hex(t.to))                                                     AS to_address,

    -- Dune stores value in wei (1 ETH = 1e18 wei). CAST to DOUBLE before dividing
    -- to avoid integer overflow — uint256 values exceed standard 64-bit integers.
    CAST(t.value AS DOUBLE) / 1e18                                          AS eth_value,

    -- USD value at transaction time, computed via the price join below
    (CAST(t.value AS DOUBLE) / 1e18) * p.price                             AS usd_value,

    -- Keep the price used for conversion so we can audit the calculation later
    p.price                                                                 AS eth_usd_price,

    -- gas_price is also in wei; dividing by 1e9 gives Gwei (the conventional unit)
    CAST(t.gas_price AS DOUBLE) / 1e9                                       AS gas_price_gwei,

    t.gas_used,

    -- Transaction fee paid by the sender: gas_price * gas_used, converted to ETH
    (CAST(t.gas_price AS DOUBLE) * CAST(t.gas_used AS DOUBLE)) / 1e18      AS tx_fee_eth,

    -- Heuristic for contract interaction: plain ETH transfers have empty calldata (0x).
    -- Any non-empty data field means the transaction is calling a smart contract function.
    -- Not perfect — someone could send a plain transfer with junk data — but correct
    -- for ~99% of cases and sufficient for Phase 2 feature engineering.
    CASE
        WHEN bytearray_length(t.data) > 0 THEN true
        ELSE false
    END                                                                     AS is_contract_call

FROM ethereum.transactions t

-- prices.usd contains minute-level price data for ERC-20 tokens.
-- Native ETH has no direct entry; WETH (Wrapped ETH, 0xc02a...) is used as the
-- price proxy. WETH always trades 1:1 with ETH by construction.
-- We join on the minute containing each transaction's block time.
-- date_trunc('minute', ...) rounds the timestamp down to the nearest whole minute.
-- LEFT JOIN means: if a price row is missing for a given minute (rare gaps in
-- price data), the transaction row is still returned with NULL price values,
-- and the WHERE clause below then excludes it.
LEFT JOIN prices.usd p
    ON  p.blockchain        = 'ethereum'
    -- Dune has no native ETH price entry; WETH (Wrapped ETH) is the proxy.
    -- WETH always trades 1:1 with ETH — it is ETH locked in an ERC-20 wrapper.
    AND p.contract_address  = 0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2
    AND p.minute            = date_trunc('minute', t.block_time)

WHERE
    -- Time window bounds — substituted at query execution time by Dune parameters.
    -- TIMESTAMP '' is TrinoSQL syntax for a literal timestamp constant.
    t.block_time >= TIMESTAMP '{{start_date}}'
    AND t.block_time <  TIMESTAMP '{{end_date}}'

    -- Only successful transactions. Failed transactions still pay gas fees and
    -- appear on-chain, but they transfer no value, so they are not whale signals.
    AND t.success = true

    -- Exclude contract creation transactions: these have a NULL "to" address
    -- because the destination contract does not exist yet at submission time.
    AND t.to IS NOT NULL

    -- ETH pre-filter: reduces rows scanned before the expensive price join.
    -- At $1,200/ETH (2023 floor), $1M = 833 ETH. 200 ETH is a conservative
    -- lower bound — the exact USD filter below removes false positives.
    AND CAST(t.value AS DOUBLE) / 1e18 > {{eth_prefilter}}

    -- Discard rows where the price join found no match (data gap in prices.usd).
    -- A null price means we cannot determine USD value — safer to exclude.
    AND p.price IS NOT NULL

    -- Final USD threshold — the authoritative filter.
    AND (CAST(t.value AS DOUBLE) / 1e18) * p.price > {{whale_usd_threshold}}

-- Ascending order so the CSV output is chronological, matching how the
-- walk-forward backtest will consume it.
ORDER BY t.block_time ASC