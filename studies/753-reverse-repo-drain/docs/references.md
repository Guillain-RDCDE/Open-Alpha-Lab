# References & literature map — Study 753 (Reverse-Repo-Drain)

## The claim under test

- **The ON RRP facility as a liquidity gauge.** The Federal Reserve's **Overnight Reverse
  Repo (ON RRP) facility** lets money-market funds and other counterparties park cash at the
  Fed overnight at an administered rate. Balances ran from near zero in early 2021 to an
  all-time peak of **~$2.554 trillion on 2022-12-30**, then drained back toward the facility's
  structural floor through 2023-2025. Daily amounts are published by the **NY Fed**
  ("Reverse Repo Operations") and mirrored on FRED as **`RRPONTSYD`** (Overnight Reverse
  Repurchase Agreements: Treasury Securities Sold by the Fed).
- **The folklore (steelmanned).** Popularised across liquidity-plumbing commentary — the
  Zoltan Pozsar *Global Money Notes* lineage and its FinTwit descendants — the reading is:
  the RRP is a "cash parking lot," and when it **drains**, that cash is flowing *out* of the
  Fed and *into* risk assets, adding market liquidity. So a draining RRP marks a **risk-on**
  regime (be long equities) and a filling RRP marks liquidity leaving markets (be cautious).
  The RRP balance is routinely charted against the S&P 500 as a "hidden liquidity tell."

## Why we ship a hardcoded proxy — and how

- **Not on yfinance.** The ON RRP balance is a Fed operating-desk series, not a tradable
  price, so there is no yfinance handle. Per the desk's house rule for non-yfinance macro
  series, we ship a **small, clearly-labelled, hardcoded monthly series** — end-of-month ON
  RRP levels in USD billions, transcribed from the public FRED `RRPONTSYD` / NY Fed prints and
  rounded (quarter-end window-dressing spikes smoothed to round marks). It is named a **proxy**
  everywhere (`reverse_repo_drain/data.py::RRP_BILLIONS`). The same labelled-proxy pattern is
  used by studies **358-watch-index** and **708-eurovision-effect** on this bench.
- **The verdict does not hinge on the marks.** The proxy carries the one feature the claim
  rests on — the 2021 fill, the ~$2.55T Dec-2022 peak, and the 2023-25 drain — and the
  conclusion turns on the *shape of that single episode*, not on any decimal place.

## Why a draining RRP need not predict returns — the economics & statistics

- **The drain is a plumbing identity, not a demand signal.** By the Fed's balance-sheet
  identity, ON RRP take-up rises and falls with the supply of competing safe assets and the
  size of reserves: the 2023-25 drain coincided with **quantitative tightening** and a **flood
  of Treasury-bill issuance** after the mid-2023 debt-ceiling resolution, which pulled
  money-fund cash out of the RRP and into bills (see Afonso, Cipriani, La Spada et al., NY Fed
  *Liberty Street Economics*, 2022-2024, on ON RRP take-up drivers; Fed H.4.1 factors-supplying
  -reserves framework). The level is an *accounting residual* of policy and issuance, not a
  forward risk-appetite gauge.
- **The n=1 problem.** The facility's entire meaningful history is **one** fill-then-drain
  cycle. It straddles the 2022 bear market (RRP *rising* to its peak) and the 2023-24 bull (RRP
  *draining*) — so a naive "drain = up" reading is a single macro coincidence. The 2021 ramp
  (RRP *filling* alongside a roaring bull) directly contradicts it. This is the classic
  common-driver / small-sample confound, not a directional edge.
- **Regime inference on few, long regimes.** With one fill and one drain, an i.i.d. label
  shuffle would wildly understate the null variance, so we test the drain-minus-fill spread
  with a **circular block bootstrap** of the regime labels (Politis & Romano, 1994,
  *The Stationary Bootstrap*, JASA; Künsch, 1989, on block bootstrap) alongside a **Welch
  two-sample t** (Welch, 1947). US equities rise in most months, so the honest object is the
  *excess* over the unconditional base rate (Kahneman & Tversky, 1973, on base-rate neglect).
- **Selection on a famous chart.** A liquidity series charted ex-post against the index it is
  supposed to lead is selected on its best-looking episode; Harvey, Liu & Zhu (2016),
  *…and the Cross-Section of Expected Returns* (RFS), is the multiple-testing caution.

## Method lineage (the desk's shared engine)

- **Regime split + Welch t.** [`strategy.regime_returns`](../reverse_repo_drain/strategy.py)
  and [`strategy.welch_t`](../reverse_repo_drain/strategy.py) — next-month SPY returns split by
  the drain regime, drain-vs-fill Welch *t*, one-month execution lag.
- **Block-bootstrap placebo.**
  [`strategy.block_bootstrap_pvalue`](../reverse_repo_drain/strategy.py) — a 20,000-draw
  block-resampled null for the regime spread that respects the long, few RRP regimes.
- **Timing backtest, net of costs.**
  [`strategy.timing_backtest`](../reverse_repo_drain/strategy.py) — long/flat (or long/short)
  SPY held when the RRP is draining, one-month lag, one-way cost per switch, raced against
  buy-and-hold on a Sharpe basis (SPY total-return, labelled).
- **Deterministic synthetic control.**
  [`data.synthetic`](../reverse_repo_drain/data.py) plants a known drain edge; the offline core
  runs with no network. It confirms the inference recovers a planted edge **and** does not
  manufacture significance when the true edge is zero.

## Data sources used here

- **yfinance** daily adjusted SPY closes, cached under `_cache/spy.csv`, resampled month-end
  (total-return). **Hardcoded ON RRP proxy** (FRED `RRPONTSYD` / NY Fed), 2021-01 → 2025-07.
  Window 2021-01-31 → 2025-07-31 (4.5 years, 55 month-ends). All headline numbers are pinned in
  [`docs/results.md`](results.md) and reproduced by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **Macro-nowcasting & regime gauges** on this bench — the ISM-PMI regime (384), the economic
  surprise index (387), jobless-claims momentum (385) — share the lesson: a plausible macro
  relationship rarely survives as a *tradable* monthly timing rule once you charge it the
  opportunity cost of sitting in cash through an up-drifting market. This one fails a step
  earlier: on a single-episode sample it never even clears significance.
