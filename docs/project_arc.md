# Full Project Arc

## Research Question

Do large on-chain Ethereum transactions (>$1M) predict short-term ETH price
movements, and does social media sentiment moderate this relationship?

## Phase Overview

Each phase feeds into the next. Understanding the full arc is essential before
touching any single phase.

### Phase 1 -- Whale Data Pipeline (COMPLETE)

Extract large ETH transactions from the blockchain via Dune Analytics. Enrich
with wallet labels (exchange, DeFi, unknown). Flag MEV bots.

- Output: `data/processed/whale_txs.csv` with labelled, cleaned transactions.
- Key columns: `timestamp_utc`, `from_address`, `to_address`, `from_category`,
  `to_category`, `eth_value`, `usd_value`, `gas_price_gwei`, `gas_used`,
  `is_contract_call`, `is_mev_candidate`, `mev_flag_reason`

Key decisions documented in `docs/design_notes.md`:
- Dune Analytics over Etherscan (chain-wide SQL vs per-address API)
- WETH as ETH price proxy (no native ETH in Dune's prices.usd table)
- Internal transactions out of scope (ethereum.traces not queried)
- MEV candidates flagged, not deleted (for Phase 4 sensitivity analysis)

### Phase 2 -- Transaction Classification (NEXT)

The raw labels (exchange_deposit, exchange_withdrawal, defi_interaction,
wallet_to_wallet) are derived mechanically from address labels. The classifier
learns to generalise this to unlabelled (unknown->unknown) transactions using
features like gas price, transaction size, sender history, time of day.

- Output: each transaction gets a predicted category + probability score.
- Why ML here: ~30-40% of whale transactions are unknown->unknown. Rule-based
  labelling cannot handle these. The classifier extends coverage.
- Key files to build: `src/features/feature_engineer.py`,
  `src/models/transaction_classifier.py`

Pre-coding analysis completed (2026-05-26):
- Labelling strategy: rule-based from from_category/to_category.
  Exchange->exchange folded into exchange_deposit (to-side priority).
- Features: log_usd_value, gas_price_gwei, log_gas_used, is_contract_call,
  hour_of_day, day_of_week, eth_usd_price, sender_prior_tx_count.
- Look-ahead risks identified: sender history must use only prior rows;
  gas price normalisation must avoid full-dataset statistics; classifier
  must be retrainable on temporal subsets for Phase 4 walk-forward.

### Phase 3 -- Sentiment Pipeline

Reddit (r/CryptoCurrency, r/Bitcoin, r/Ethereum) and crypto news headlines
(CryptoPanic) scored hourly using VADER sentiment. Aggregated to match the
hourly timestamp of whale transactions.

- Output: hourly sentiment scores aligned to whale transaction timestamps.
- Why this matters: whale movements in isolation are noisy. If a whale deposits
  to an exchange during strongly negative sentiment, the selling pressure signal
  is stronger than the same transaction during positive sentiment.

### Phase 4 -- Price Impact Prediction (THE PAYOFF)

All prior phases combine. Features fed into the final model:
- Whale transaction category (from Phase 2 classifier)
- Transaction size and gas (from Phase 1)
- Hourly sentiment score (from Phase 3)
- Recent price features (rolling returns, volatility)

Target variable: ETH price direction at t+24h (binary: up or down).
Walk-forward validation: model trained only on data available before each
prediction. No look-ahead.

- Output: directional accuracy, edge over random baseline, P&L simulation.

### Phase 5 -- Evaluation and Write-up

Honest reporting of findings. Where does the signal exist? Which transaction
categories drive it? Does sentiment improve or not? What are the limitations?

- Output: `results/charts/`, `docs/findings.md`, final README.

## Where ML Appears

| Phase   | ML Component              | Type                                          |
|---------|---------------------------|-----------------------------------------------|
| Phase 2 | Transaction classifier    | Supervised classification (Random Forest)     |
| Phase 3 | Sentiment scoring         | Pre-trained NLP (VADER, optional FinBERT)     |
| Phase 4 | Price impact predictor    | Supervised classification (XGBoost/LogReg)    |

Phase 2 ML is in service of Phase 4. The classifier is not the end goal -- it
is a feature engineering step that makes Phase 4 possible by extending label
coverage to unknown->unknown transactions.

## Session Handoff Notes

Update this section at the end of each working session.

**Last session: 2026-05-26**
- Phase 1 fully complete (48 tests passing).
- Phase 2 pre-coding analysis done: labelling strategy, feature design,
  look-ahead risks all documented above.
- Next step: build rule-based labeller in `src/features/feature_engineer.py`.
