# References — Study 246 (Defensive-Sectors Leadership)

## The folk claim

The "defensive-sector leadership" signal extends the single-utility-canary narrative
(Study 131) by combining consumer staples (XLP) and utilities (XLU) relative to SPY.
The intuition: when *both* bond-like defensive sectors simultaneously outperform the
broad market on a relative-strength basis, the message is stronger — institutional money
is rotating out of risk assets on multiple fronts, and a drawdown is imminent.

## Academic and practitioner foundations

**Fama, E. F., & French, K. R. (1993).** Common risk factors in the returns on stocks and bonds.
*Journal of Financial Economics*, 33(1), 3–56.
The foundational framework for understanding sector return differences as risk factor
exposures (market beta, earnings stability) rather than predictive signals. Utilities and
staples are "boring" because they have stable cash flows, not because investors can time
broad market cycles using their relative strength.

**Hong, H., Torous, W., & Valkanov, R. (2007).** Do industries lead stock markets?
*Journal of Financial Economics*, 83(2), 367–396.
A formal test of sector lead-lag relationships with the aggregate market. Some industries
(petroleum, financials) contain modest leading information, but utilities and consumer
staples are among the *weaker* predictors of aggregate market returns.

**Moskowitz, T. J., & Grinblatt, M. (1999).** Do industries explain momentum?
*Journal of Finance*, 54(4), 1249–1290.
Documents industry momentum effects operating at 3-12 month horizons. The short 20-day
RS momentum used by the folk canary signal is a different — and less robust — phenomenon.

**Asness, C. S., Moskowitz, T. J., & Pedersen, L. H. (2013).** Value and momentum everywhere.
*Journal of Finance*, 68(3), 929–985.
Momentum is a documented anomaly across many asset classes, but at multi-month horizons
and requiring careful statistical verification. Short-term sector RS signals lack the
evidence base of longer-horizon momentum strategies.

**Rapach, D. E., Strauss, J. K., & Zhou, G. (2010).** Out-of-sample equity premium prediction:
Combination forecasts and links to the real economy.
*Review of Financial Studies*, 23(2), 821–862.
Demonstrates that macro-linked variables can predict market returns out-of-sample at
monthly horizons with R² of ~1%, but requires formal out-of-sample testing — the standard
the folk canary signal skips. Combining predictors can help (consistent with combining
XLP+XLU), but the documented effect sizes are very small.

**Baker, M., & Wurgler, J. (2007).** Investor sentiment in the stock market.
*Journal of Economic Perspectives*, 21(2), 129–152.
Frames defensive-sector outperformance as a sentiment indicator. The relationship between
sentiment and subsequent returns is imprecise at actionable short horizons.

**Newey, W. K., & West, K. D. (1987).** A simple, positive semi-definite, heteroskedasticity
and autocorrelation consistent covariance matrix.
*Econometrica*, 55(3), 703–708.
The HAC variance estimator used to compute all t-statistics in this study, essential for
overlapping return series and autocorrelated signals.

## Related Open-Alpha-Lab studies

- **Study 131 — Utilities-Canary:** Tests the XLU/SPY single-sector version of this
  signal. Found *t* = +1.58 at the 21-day horizon (Weak/Mirage verdict). The combined
  XLP+XLU version in this study finds *t* = +0.79 — weaker, not stronger.
- **Study 111 — VIX Term Structure:** Tests a different risk-off timing signal (VIX
  futures term structure slope) that is also largely coincident with stress.

## Data sources

- **Yahoo Finance** (via `yfinance`): XLP, XLU, and SPY daily adjusted closes 1998-12-22
  onward. XLP (Consumer Staples Select Sector SPDR Fund) and XLU (Utilities Select Sector
  SPDR Fund) both launched 1998-12-22; data available from inception. Auto-adjusted prices
  account for dividends and splits.
- SPDR sector ETF inception dates: [https://www.sectorspdr.com/sectorspdr/](https://www.sectorspdr.com/sectorspdr/)

## Why the result is NONE / MIRAGE

The combined defensive signal fails on two levels:

1. **Wrong direction at short horizons.** The 1-day Q1−Q5 spread is *negative* (−1.56 bps),
   meaning defensive-alert days earned *more* than calm days over the next day on average.
   This directly contradicts the folk claim.

2. **Non-monotone at all horizons.** The quintile pattern at 1-day, 5-day, and 21-day all
   show Q5 reverting above Q3/Q4, breaking the monotone descent the folk signal requires.
   Adding XLP to the XLU canary reduces the (already weak) directional consistency of the
   single-sector signal.

The regime-Sharpe split (0.52 calm vs 0.38 alert) is real but smaller than Study 131 and
is driven by the well-known volatility clustering in stress regimes, not by a forward-looking
return differential. The mean return in the calm vs alert regimes (+3.36 vs +3.27 bps/day)
is indistinguishable from zero difference.
