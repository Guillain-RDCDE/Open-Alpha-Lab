# References & literature map — Study 268 (Sahm-Rule)

## The rule under test

**Sahm, C. (2019).** "Direct Stimulus Payments to Individuals." In *Recession
Ready: Fiscal Policies to Stabilize the American Economy* (Boushey, Nunn &
Shambaugh, eds.), The Hamilton Project / Brookings, pp. 67–92. The original
statement of the **Sahm Rule**: a recession has begun when the 3-month moving
average of the national unemployment rate rises 0.50 percentage points or more
above its minimum over the previous 12 months. Crucially, Sahm designed it as a
**real-time recession identifier to trigger automatic fiscal stabilizers** —
*not* as a market-timing or asset-allocation signal. The Federal Reserve Bank of
St. Louis publishes the live series as the **Sahm Recession Indicator** (FRED:
`SAHMREALTIME`, `SAHMCURRENT`).

## Why a great recession indicator can be a poor sell button

- **Lead/lag mismatch.** Equity prices are forward-looking and tend to *lead* the
  business cycle, typically bottoming *during* a recession. The unemployment rate
  is a classic *lagging* indicator that often keeps rising after the recession
  trough and the stock-market bottom. A rule built on unemployment therefore
  triggers *after* much of the equity decline has occurred. The post-trigger
  window is disproportionately the recovery.

- **The wrong null.** Equities have a strong upward drift; the S&P is positive in
  the large majority of rolling 12-month windows. Any "sell" signal must beat
  *staying invested*, i.e. the unconditional forward return — not a 0% bar. Tested
  against the correct baseline, the post-trigger forward return is *above* average.

- **Tiny n.** There are only ~12 distinct Sahm onsets in 65 years. With ~20%
  annual volatility on forward returns, the minimum detectable mean shift at
  conventional power is enormous (~15pp/event). No macro-timing edge — positive or
  negative — can be resolved at this sample size.

- **Real-time vintage / revisions.** The live Sahm value uses real-time
  unemployment data, which is revised. The St. Louis Fed maintains both a
  `SAHMREALTIME` (vintage) and a `SAHMCURRENT` (revised) version precisely because
  they differ. Backtests on the *final* series (as here) modestly overstate
  timeliness; the live signal triggers a touch later. Named on the Signal axis.

## Recessions vs bear markets — the empirical backdrop

- **NBER Business Cycle Dating Committee.** The official US recession chronology.
  The committee dates recessions with a lag of 6–18 months, again underscoring
  that recession *dating* and market *timing* are different problems.

- **Estrella, A. & Mishkin, F. S. (1998).** "Predicting U.S. Recessions: Financial
  Variables as Leading Indicators." *Review of Economics and Statistics*, 80(1),
  45–61. The yield-curve term spread *leads* recessions by ~12 months — the
  contrast that explains why inversion-based timing has at least a chance where a
  lagging unemployment rule does not.

- **Siegel, J. J. (1991).** "Does It Pay Stock Investors to Forecast the Business
  Cycle?" *Journal of Portfolio Management*, 18(1), 27–34. Even with *perfect*
  recession foresight, the transaction-cost and out-of-market drag erode most of
  the benefit — the classic statement of why business-cycle timing underwhelms.

## Method lineage

- **Sahm value.** `ma3 = unrate.rolling(3).mean()`; `gap = ma3 - ma3.rolling(12).min()`;
  trigger when `gap >= 0.50`. Onset = first month of each new trigger episode.
- **Event study.** Forward H-month price return from an entry month set one month
  after the onset (execution lag), compared to the unconditional forward-return
  distribution. Welch t-test for the event-vs-all mean difference.
- **HAC / Newey-West t.** Overlapping forward returns are serially correlated;
  we report a Bartlett-kernel HAC t on the (event − baseline) series. A REAL
  signal would need a robust HAC |t| ≥ 2 — here it is +0.55 (and the wrong sign).
- **Block permutation.** Keep the event count fixed; draw that many random entry
  months 5,000 times; the one-sided p is the fraction of random draws whose mean
  is ≤ the observed event mean (a "trigger → low returns" test).
- **Timing overlay.** Long/flat binary overlay, one-way costs on NAV, CAGR / vol /
  Sharpe / max-drawdown vs buy-and-hold; HAC t on the monthly active return.

## Data sources

- **Unemployment rate.** U.S. Bureau of Labor Statistics, civilian unemployment
  rate, 16+, seasonally adjusted, series `LNS14000000` (FRED `UNRATE`). Monthly,
  1959–2025, hardcoded in `data.py` (final prints, as-of 2026-06-17).
- **S&P 500.** ^GSPC daily close (price-only, split-adjusted) from the repo-level
  cache `_cache/^GSPC_split_only.parquet`. Price return only — no dividends — which
  *understates* buy-and-hold and so flatters the timing overlay.

## Related desk studies

- The yield-curve inversion teardown (a *leading* recession signal — the natural
  contrast to this *lagging* one).
- The unemployment-trend / 12-month-MA equity-timing rules (same macro-timing
  family).
- Study 158 — Super-Bowl: the same tiny-n + correct-baseline methodology applied
  to a folklore indicator.
