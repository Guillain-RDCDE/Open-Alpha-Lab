# References & literature map — Study 930 (When-Issued Window)

## The claim under test

- **The when-issued window.** A US spin-off has a dead zone. Between the announcement of a
  separation and the moment the child trades regular way, the child exists as a
  *when-issued* claim — settled only if and when the distribution completes — while the
  parent trades as a bundle of "stub plus entitlement". Practitioner folklore attaches an
  edge to each side of that zone: the parent re-rates upward on the announcement as the
  conglomerate discount is unwound; the child is dumped by index funds and mandate-bound
  holders the moment it lands, so it is cheap for a few days; and the two halves, once
  separately priced, are together worth more than the pre-spin whole. This study measures
  all three on the same table.
- **The steelman.** Every leg is a mechanical, dated, publicly observable window with a
  named entry and exit — no forecasting, no parameter to tune. If any of the three stories
  is true it should show up as a cross-sectional alpha versus the index over exactly those
  dates.

## The forced-seller literature (the child's first days)

- **Cusatis, Miles & Woolridge (1993),** *Restructuring through spinoffs: The stock market
  evidence*, Journal of Financial Economics 33 — the founding result: spin-off children
  and their parents beat matched benchmarks over the following three years.
- **Greenblatt (1997),** *You Can Be a Stock Market Genius*, chapters on spin-offs — the
  popular statement of the mechanism: institutions receive shares they neither chose nor
  can hold (wrong index, wrong size, wrong mandate) and sell them regardless of value,
  so the child is temporarily cheap. Note the claim is about *forced selling pressure*,
  which is a short-horizon prediction even where the tested horizon is long.
- **McConnell & Ovtchinnikov (2004),** *Predictability of long-term spinoff returns*,
  Journal of Investment Management — re-examines Cusatis et al. and finds the long-horizon
  excess return is driven by a small number of observations and is sensitive to the
  benchmark. The same warning applies with force at n = 26.
- **Chemmanur, Krishnan & Nandy (2014),** *The effects of corporate spin-offs on
  productivity*, Journal of Corporate Finance — the fundamental (rather than
  flow-of-funds) channel for any post-spin drift.
- **Our result reverses the sign at short horizon.** The child under-performs by −4.5% over
  five regular-way sessions and −6.1% by session three. That is consistent not with forced
  *selling* into the first days but with forced *buying* at the distribution — index funds
  that track the parent's benchmark must hold the child on the day it is added, which
  pushes the print up, and the pressure unwinds over the following sessions.

## Why forced buying, not forced selling, is the better prior here

- **Shleifer (1986),** *Do demand curves for stocks slope down?*, Journal of Finance — the
  original index-inclusion price-pressure result: mechanical, non-informational demand moves
  prices.
- **Harris & Gurel (1986),** *Price and volume effects associated with changes in the S&P
  500 list*, Journal of Finance — and the modern re-examination, **Wurgler & Zhuravskaya
  (2002)**, *Does arbitrage flatten demand curves for stocks?*, Journal of Business. The
  common finding is a pop on the effective date followed by partial or complete reversal —
  precisely the three-session trough and month-long fill-in this study measures.
- **Petajisto (2011),** *The index premium and its hidden cost for index funds*, Journal of
  Empirical Finance — quantifies the cost index funds pay for trading on the effective
  date. A spin-off distribution is an index event by construction (the child enters the
  parent's index on day one), so the same mechanics apply.
- **The when-issued market itself.** Our regular-way anchor sweep shows the decline is
  already under way in the when-issued sessions *before* the anchor (−7.8% at a one-session
  earlier start), which is what you would expect if the pre-distribution when-issued price
  is set by a thin, one-sided market rather than by the eventual regular-way clearing price.

## The parent side of the window

- **Vijh (1994),** *The spinoff and merger ex-date effects*, Journal of Finance — documents
  abnormal returns concentrated around the ex-distribution date rather than the
  announcement, and is the closest antecedent to this study's dating discipline.
- **Veld & Veld-Merkoulova (2009),** *Value creation through spin-offs: A review of the
  empirical evidence*, International Journal of Management Reviews — surveys the
  announcement-effect literature: the parent's re-rating is concentrated in the **first
  days after the announcement**, not spread across the year-long wait we measure here.
  Our null on the announcement-to-distribution leg (−1.45% gross, *t* = −0.26) is consistent
  with that: the news is priced quickly and the long wait is just beta plus noise.

## Related desk studies (dedup)

- **[Study 239 — Spinoffs](../../239-spinoffs/)**: the *long-horizon* child drift — buy the
  child and hold it for **6 / 12 / 18 / 24 months** versus SPY, testing Cusatis-Greenblatt on
  its own terms (Weak / Fragile: 6m and 12m alpha barely clear |*t*| = 2 on 14 curated
  events). Study 930 tests the **opposite end of the clock**: the announcement-to-
  distribution wait and the first **5 / 10 / 21 sessions** of regular-way trading, plus the
  parent and the parent+child combination, which Study 239 never looks at. The two studies
  do not overlap in window, in leg, or in event table (26 events here, 14 there), and they
  disagree in sign at their respective horizons — which is itself the finding.
- **[Study 452 — Spinning Top](../../452-spinning-top/)**: a candlestick pattern; shares
  only the word.
- **Method lineage** (shared across the desk): Newey & West (1987), *A Simple,
  Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance
  Matrix*, Econometrica → [`strategy.newey_west_t`](../when_issued/strategy.py) and
  [`quantlab.analytics.mean_tstat_hac`](../../../quantlab/analytics.py). Efron (1979),
  *Bootstrap Methods*, Annals of Statistics, and Politis & Romano (1994), *The Stationary
  Bootstrap*, JASA → [`strategy.bootstrap_mean_ci`](../when_issued/strategy.py) and
  [`strategy.block_bootstrap_mean_ci`](../when_issued/strategy.py). Wilson (1927),
  *Probable Inference*, JASA → the hit-rate intervals. **Westfall & Young (1993),**
  *Resampling-Based Multiple Testing*, Wiley → [`strategy.family_wise_p`](../when_issued/strategy.py),
  the max-|*t*| bootstrap that prices all five pre-specified legs at once instead of
  quoting the best of five as if it had been the only one tested.

## Data sources & the non-tape inputs

- **Prices.** Daily **total-return** closes via `yfinance` (`auto_adjust=True`) for 26
  parents, 26 children, **SPY** (benchmark), **BIL** (cash), plus **IWM** and **MDY** for the
  size-matched benchmark cross-check. Cached as parquet in the shared desk cache
  `studies/_cache`. Yahoo's adjustment handles splits and cash dividends but treats a
  spin-off *distribution* inconsistently, which is why no window in this study crosses the
  distribution date.
- **The event table is an ASSUMPTION.** Parent/child pairs, announcement dates, first
  regular-way sessions and distribution ratios are hand-compiled from Form 10 / 8-K filings
  and the financial press. All three date-and-ratio inputs are swept in `docs/results.md`;
  the regular-way anchor is the one that matters and is treated as such.
- **Borrow rates are an ASSUMPTION.** SPY borrow (40 bp/yr) on the short index leg and the
  child's borrow on the harvesting trade (swept 0 → 100 %/yr) are not observable on this
  tape and are stated as assumptions, not measured.
- **As-of 2026-06-30.** The partial current month is dropped so the sample never creeps.
