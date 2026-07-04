# References — Study 598 (Cederburg "100% Equities for Life")

## The claim's source

- **Anarkulova, A., Cederburg, S. & O'Doherty, M. (2023).** *Beyond the Status Quo: A Critical
  Assessment of Lifecycle Investment Advice.* SSRN working paper 4590406.
  <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4590406>.
  The tested claim: on a block-bootstrap of long-run developed-market returns (38 countries),
  a lifecycle held **100% in stocks — half domestic, half international — for life** beats
  balanced stock/bond strategies and target-date glidepaths on retirement wealth, bequest AND
  ruin probability (their headline: ~8.2% ruin for a balanced strategy vs ~3.5% for the
  50/50 domestic/international all-equity mix). Widely reported as "the case against
  target-date funds".
- **Anarkulova, A., Cederburg, S. & O'Doherty, M. (2022).** *Stocks for the Long Run?
  Evidence from a Broad Sample of Developed Markets.* Journal of Financial Economics 143(1).
  The underlying 38-country bootstrap dataset and method (10-year block resampling of
  long-horizon real returns) that the 2023 lifecycle paper builds on.

## The adversarial literature

- **Dimson, E., Marsh, P. & Staunton, M.** *Global Investment Returns Yearbook* (UBS/Credit
  Suisse, annual). <https://www.ubs.com/global/en/investment-bank/global-markets/global-investment-returns-yearbook.html>.
  The long-run world-ex-US real equity numbers (geometric ≈ 4.3%/yr, vol ≈ 17%, correlation
  with US ≈ 0.6) that calibrate our pre-EFA international leg — and the reason the
  international half of the paper's prescription drags on a US-based tape.
- **Siegel, J. (1994–2022).** *Stocks for the Long Run.* The domestic ancestor of the claim —
  torn down separately in [study 151](../../151-stocks-for-long-run/).
- **Pfau, W. & Kitces, M. (2014).** *Reducing Retirement Risk with a Rising Equity Glide Path.*
  Journal of Financial Planning 27(1) — the glidepath (TDF-style) advice the paper attacks;
  the decumulation half is torn down in [study 596](../../596-bond-tent-glidepath/).
- **Bengen, W. (1994).** *Determining Withdrawal Rates Using Historical Data.* Journal of
  Financial Planning 7(4) — the 4%-rule mechanics our retirement phase follows.

## Sibling studies on this desk (the dedup guard)

- [Study 151 — Stocks-For-Long-Run](../../151-stocks-for-long-run/): tested the **horizon
  claim** — does the equity premium reliably materialise over 20–30 years? This study tests
  something different: the **lifecycle allocation horse race** from the 2023 Cederburg paper —
  savings phase + retirement phase, terminal wealth *and* ruin probability, 100%-equity
  (50/50 domestic/international) vs 60/40 vs a TDF glidepath.
- [Study 596 — Bond-Tent-Glidepath](../../596-bond-tent-glidepath/): the retirement-phase
  glidepath shape at a matched equity budget. Here the glidepath (TDF) is one *contender* in a
  whole-lifecycle race, not the object under test.
- [Study 172 — Hundred-Minus-Age](../../172-hundred-minus-age/) and
  [Study 173 — Four-Percent-Rule](../../173-four-percent-rule/): the accumulation glidepath and
  the withdrawal-rate rule this study inherits its mechanics from.

## Data

- **Shiller, R.** *Irrational Exuberance* long-run US dataset (S&P composite, dividends, CPI,
  10-year yield), monthly 1871+. <http://www.econ.yale.edu/~shiller/data.htm>. Fetched via the
  GitHub raw mirror <https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv>,
  cached at `_cache/shiller_sp500.parquet` (cache-first).
- **iShares MSCI EAFE ETF (EFA)** — yfinance auto-adjusted (total-return) monthly closes,
  2001-08 onward, cached at `_cache/efa_monthly.csv`; deflated by realised US CPI from the
  Shiller panel. The **only market data** in the international leg; everything before 2001-09
  is a literature-calibrated simulation (DMS parameters, fixed seed 598) and is labeled as
  such everywhere it appears.
- Bond returns: first-order 10-year approximation `y_{t-1}/12 − D·Δy` with modified duration
  D = 7, CPI-deflated (same construction as study 596; stated as a decision).

## Method

- **Newey, W. & West, K. (1987).** HAC t on overlapping-cohort differences, bandwidth forced
  to the full 840-month overlap.
- **Politis, D. & Romano, J. (1992).** Circular block bootstrap — 120-month blocks, both for
  the Cederburg-style resampled lifetimes and for the outer tape-level CIs.
