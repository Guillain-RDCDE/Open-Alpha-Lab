# References & literature map — Study 534 (Revenue-Surprise-Drift)

## The claim under test

- **The thesis.** Narasimhan Jegadeesh & Joshua Livnat, *Revenue surprises and stock returns*
  (2006, Journal of Accounting & Economics 41(1–2), 147–171). They show that the post-earnings
  drift is driven not only by **earnings (EPS) surprises** but also by **revenue (sales)
  surprises**, and — the central claim — that the standardized unexpected revenue (SUR) carries
  information about future returns **incremental to** the standardized unexpected earnings (SUE).
  A hedge portfolio long high-SUR / short low-SUR firms earned a significant drift over the
  ~60 trading days following the announcement, even after controlling for the SUE drift.
- **Why revenue at all.** Jegadeesh & Livnat (and Swaminathan & Weintrop 2008,
  *The information content of earnings, revenues, and expenses*, JAR) argue revenue is a
  *higher-quality*, harder-to-manage signal of the *persistence* of an earnings surprise:
  a beat driven by rising sales is more durable than one driven by cost cuts or one-offs — so the
  market underreacts to the revenue component specifically.

## The parent anomaly (post-earnings-announcement drift)

- **Ball & Brown (1968)**, *An Empirical Evaluation of Accounting Income Numbers* (JAR) — first
  documented prices keep adjusting *after* an announcement.
- **Bernard & Thomas (1989, 1990)** established **SUE** drift: top-SUE-decile firms outperform
  the bottom for ~60 trading days. Eugene Fama (1998, JFE) called PEAD the "granddaddy of
  anomalies." Our study 363 (EPS-PEAD) rebuilds the **earnings**-surprise version; this study is
  the **revenue**-surprise sibling, deliberately distinct.

## How we measure the revenue surprise (SUR)

- **Seasonal random walk.** Following the standard de-seasonalisation for quarterly sales, the
  *unexpected* revenue is `u_q = Rev_q − Rev_{q−4}` (this quarter minus the same quarter a year
  ago). **SUR** standardizes it by the trailing volatility of those seasonal differences known at
  q: `SUR_q = u_q / std(prior u)`. This is the direct revenue analogue of academic SUE and the
  expectation model Jegadeesh-Livnat use (a seasonally-differenced random walk with drift).
- **No look-ahead / timing.** We anchor each event at the **10-Q/10-K filing date** reported by
  EDGAR (the date the number became public), take the first trading session on/after it, then
  **enter one day later** and hold — so the drift is strictly *after* the revenue figure is
  disclosed (the standard event-study convention).

## Why a flat result here is the expected outcome

- **Limits to arbitrage / liquidity.** Chordia, Goyal, Sadka, Sadka & Shivakumar (2009,
  *Liquidity and the post-earnings-announcement drift*, FAJ) show the whole drift family
  concentrates in **small, illiquid** names and shrinks among liquid large-caps — exactly the
  conservative universe we use by construction.
- **Post-publication decay.** McLean & Pontiff (2016, *Does academic research destroy stock
  return predictability?*, JF) and Harvey, Liu & Zhu (2016, *…and the Cross-Section of Expected
  Returns*, RFS) document that published anomalies weaken sharply out of sample. Jegadeesh-Livnat
  (2006) is 20 years old; a null on a 2013–2026 large-cap survivor sample is consistent with both
  liquidity limits and decay.

## Why a high *t* would still need a placebo + clustering check

- **One-sample / Welch t** (Welch, 1947) for the long-short mean against zero. Filings cluster in
  **seasons**, so naive *t*-stats overstate significance; we add a **label-shuffle placebo**
  (Fisher randomization; Efron & Tibshirani, *An Introduction to the Bootstrap*, 1993) and a
  **within-quarter block placebo** that respects the clustering. Here both confirm the absence of
  signal rather than rescue it.

## Method lineage (the desk's shared engine)

- **Quintile long-short + one-sample t.** [`strategy.long_short_drift`](../revenue_drift/strategy.py)
  / [`strategy.ttest_vs_zero`](../revenue_drift/strategy.py).
- **Label-shuffle & block placebo.** [`strategy.placebo_pvalue`](../revenue_drift/strategy.py)
  and [`strategy.block_placebo_pvalue`](../revenue_drift/strategy.py).
- **Incremental-to-EPS test.** [`strategy.incremental_to_eps`](../revenue_drift/strategy.py) —
  the SUR long-short within EPS-sign strata (the Jegadeesh-Livnat incremental claim), using the
  reported EPS surprise from study 363's cache.
- **Deterministic synthetic control.** [`data.synthetic_rev`](../revenue_drift/data.py) plants a
  known post-event drift proportional to the surprise; with the edge set to zero the inference
  must NOT manufacture significance.

## Data sources used here

- **yfinance** daily adjusted closes for a fixed 30-name large-cap basket.
- **EDGAR** `companyconcept` XBRL API (`data.sec.gov`) — frame-tagged quarterly revenue
  (`RevenueFromContractWithCustomerExcludingAssessedTax`, falling back to `Revenues` /
  `SalesRevenueNet`), with the 10-Q/10-K filing date. Cached under `_cache/rev_prices.csv` and
  `_cache/rev_events.csv`. All headline numbers are pinned in [`docs/results.md`](results.md) and
  reproduced by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **[363 — PEAD-Drift](../363-pead-drift)** — the **EPS**-surprise sibling, which *does* clear
  the bar (Signal Real / Tradability Fragile). The contrast is the point: on the same kind of
  basket the **earnings** surprise drifts but the **revenue** surprise (here) does not — the
  Jegadeesh-Livnat incremental claim does not survive on this conservative survivor universe.
