# References -- Study 112 (Move-Index)

## The claim's source

- **ICE BofA MOVE Index** — Merrill Lynch Option Volatility Estimate, a measure of the implied
  volatility of 1-month Treasury options (weighted across maturities 2, 5, 10, 30 years). Often
  described as "the VIX for bonds." See: <https://fred.stlouisfed.org/series/MOVE> and the ICE
  BofA documentation; the series trades as ``^MOVE`` on Yahoo Finance from ~2003.

- **Gross, B. (Bill Gross)** — popularised MOVE as a macro risk gauge in PIMCO commentaries
  (2000s–2010s); the narrative that "when MOVE rises, cross-asset risk rises" became embedded in
  sell-side fixed-income research.

## The underlying effect

- **Brunnermeier, M. K., & Pedersen, L. H. (2009).** "Market Liquidity and Funding Liquidity."
  *Review of Financial Studies*, 22(6), 2201–2238. The theoretical basis for cross-asset contagion:
  funding-constrained intermediaries de-risk across bond and equity markets simultaneously, so bond
  vol and equity vol can rise together — but the causal direction is unclear.

- **Adrian, T., & Brunnermeier, M. K. (2016).** "CoVaR." *American Economic Review*, 106(7),
  1705–1741. Systemic risk co-movement across assets: bond and equity markets share common
  risk factors, especially in tail events, which underpins the cross-asset MOVE narrative.

## The VIX as a benchmark

- **Whaley, R. E. (2000).** "The Investor Fear Gauge." *Journal of Portfolio Management*, 26(3),
  12–17. The original VIX concept — a real-time measure of the market's expectation of near-term
  volatility in the S&P 500. Provides the natural benchmark for MOVE: if VIX already captures
  equity-market fear, does MOVE add incremental information?

- **CBOE (2019).** *VIX White Paper.* Chicago Board Options Exchange. Official methodology for
  VIX calculation. Available at: <https://www.cboe.com/tradable_products/vix/>.

## Method lineage

- **Newey, W. K., & West, K. D. (1987).** "A Simple, Positive Semi-Definite, Heteroskedasticity
  and Autocorrelation Consistent Covariance Matrix." *Econometrica*, 55(3), 703–708. The HAC
  t-statistic used throughout this study for inference on overlapping forward returns.

- **McLean, R. D., & Pontiff, J. (2016).** "Does Academic Research Destroy Stock Return
  Predictability?" *Journal of Finance*, 71(1), 5–32. The post-publication decay problem: anomalies
  documented in the literature tend to weaken after publication due to arbitrage.

## Related desk studies

- **Study 86 (Tail-Radar)** — tests the CBOE SKEW index as a crash predictor for SPY; same
  methodology (quintile sorts, HAC inference, VIX regression), same NONE verdict. MOVE is the
  bond-market analogue of SKEW.

- **Study 68 (All-Weather)** and **Study 16 (Storm-Shy)** — cross-asset allocation studies where
  bond-market signals (rates, spreads) are tested as equity timing gauges; similar "coincident not
  leading" failures documented.

## Data sources

- ``^MOVE`` daily close from Yahoo Finance (``yfinance``), available from ~2003-01-01 onward.
- ``^VIX`` daily close from Yahoo Finance, available from 1990-01-02 onward.
- ``SPY`` daily close (total-return adjusted) from Yahoo Finance, available from 1993-01-29 onward.
- All cached under ``_cache/daily_move_vix_spy.parquet``; fingerprint ``74dbe2ac989f`` (2026-06-13).
