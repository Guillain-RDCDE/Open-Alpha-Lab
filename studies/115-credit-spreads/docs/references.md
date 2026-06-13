# References & literature map — Study 115 (Credit-Spreads)

## The claim under test

- **"Credit leads equities."** The high-yield (HY) credit-risk premium is a widely-cited
  leading indicator of equity risk: when HY spreads widen, equity stress tends to follow.
  The practical version: monitor HYG underperformance vs Treasuries (IEF) or vs
  investment-grade bonds (LQD); go defensive or short equities when the spread widens
  above a threshold.  We steelman this as: *the rolling ETF-based credit-spread proxy
  (HYG-IEF or HYG/LQD 20-day return differential) predicts negative forward SPY returns
  in stress regimes, more so than the unconditional equity drift.*

## Why the claim is plausible — the real effect it leans on

- **Credit as a leading risk indicator.** Gilchrist & Zakrajsek (2012), *Credit Spreads
  and Business Cycle Fluctuations* (American Economic Review), document that the excess
  bond premium (EBP) — the idiosyncratic component of credit spreads after controlling
  for default risk — is a powerful predictor of economic activity and equity returns,
  Granger-causing both GDP growth and equity returns 1-6 quarters ahead.  This is the
  canonical academic version of the claim.
- **High-yield as equity risk barometer.** Fama & French (1989), *Business Conditions and
  Expected Returns on Stocks and Bonds* (Journal of Financial Economics), show that
  default-risk spreads predict both stock and bond returns, especially over longer
  horizons.  The credit spread is a "risk appetite" gauge.
- **Credit-equity co-movement.** Collin-Dufresne, Goldstein & Martin (2001), *The
  Determinants of Credit Spread Changes* (Journal of Finance), show that credit-spread
  innovations are driven largely by the same systematic equity-risk factors — meaning
  credit and equity are co-integrating in stress periods, not leading one another.  This
  is the key tension: credit may co-move with equity rather than *lead* it, especially
  at short horizons and when using ETF proxies.
- **ETF proxies vs. OAS spreads.** The "true" credit-spread signal is the Option-Adjusted
  Spread (OAS) from FRED/ICE BofA indices (e.g. BAMLH0A0HYM2OAS), which measures the
  yield premium over a matched Treasury.  Our ETF-price proxy (HYG total-return
  underperformance vs IEF) is noisier because it also reflects duration risk, flow
  dynamics, and ETF-specific premiums/discounts.  Bai, Bali & Wen (2019), *Common Risk
  Factors in the Cross-Section of Corporate Bond Returns* (Journal of Financial
  Economics), discuss the gap between the credit factor in bond ETFs and the underlying
  index.

## Why the claim is hard to trade

- **Coincident vs. leading.** Kwan (1996), *Firm-specific Information and the Correlation
  Between Individual Stocks and Bonds* (Journal of Financial Economics), and Blanco,
  Brennan & Marsh (2005), *An Empirical Analysis of the Dynamic Relation between
  Investment-Grade Bonds and Credit Default Swaps* (Journal of Finance), show that
  credit-market price discovery is largely contemporaneous with equities, not a reliable
  lead.  HYG is itself bought/sold by equity-risk-aware investors; widening HYG spreads
  happen in the same crisis window as equity sell-offs, not reliably before them.
- **Short ETF history and the ETF-OAS gap.** HYG launched in 2007 (our usable sample
  begins 2010); the 2008 crisis is excluded.  ETF flows during stress events can drive
  premiums/discounts that temporarily distort the spread proxy away from the true
  underlying credit-risk measure.
- **Slow-moving signal.** A 20-day rolling window smooths out the very events where credit
  leads matter most (sharp dislocations), leaving a slow-moving signal that by the time
  it flags "stress," equity markets have already repriced.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  [`strategy.regime_stats`](../credit_spreads/strategy.py) and
  [`quantlab.analytics.mean_tstat_hac`](../../../quantlab/analytics.py).
- **Regime-conditional forward-return framework.** Koijen, Lustig & Van Nieuwerburgh
  (2017), *The Cross-Section and Time Series of Stock and Bond Returns* (Journal of
  Monetary Economics), use a related split of economic regimes to examine predictability.
- **Rolling-median signal.** Uses an expanding/rolling median rather than a fixed
  threshold to avoid look-ahead in regime classification — each observation is assigned
  to stress/calm based only on data available at that date.
- **Reproducibility stamp.** [`quantlab/`](../../../quantlab/) — content fingerprints in
  [`docs/results.md`](results.md).

## Data sources used here

- **Yahoo Finance daily adjusted-close prices** (via `yfinance`): HYG, LQD, IEF, SPY,
  2010-01-04 to 2026-06-12 (approximately 4,136 trading days).  **FRED OAS data was not
  available in this sandbox** (endpoint timeout / rate limiting); we use ETF price
  returns as the credit-spread proxy and explicitly state the limitation.  The ETF proxy
  is economically motivated but noisier than level-spread data.

## Related desk studies

- **[Study 68 — All-Weather](../../68-all-weather/)**: regime-based allocation using
  macro signals — the same conditional-return framework applied to multi-asset portfolios.
- **[Study 86 — Tail-Radar](../../86-tail-radar/)**: VIX as an equity risk signal — the
  volatility index is closely correlated with credit spreads as a stress indicator; a
  natural companion test.
- **[Study 85 — Dr-Copper](../../85-dr-copper/)**: copper/gold ratio as a macro
  leading indicator — the cross-asset ratio predictability family.
- **[Study 16 — Storm-Shy](../../16-storm-shy/)**: equity timing around macro regimes;
  another "defensive signal → reduce equity" framework.
