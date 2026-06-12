# References — Study 71 (Ambush)

## The ingredients (and the bench studies that certified-then-buried them)

- Connors, L. & Alvarez, C. (2009). *High Probability ETF Trading*. TradingMarkets. — the
  IBS / low-close bounce rule. Torn down gross-real, net-dead in
  [study 19 — Rubber-Band](../../19-rubber-band/).
- Kakushadze, Z. & Serur, J.A. (2018). *151 Trading Strategies*. §4.4 ETF mean reversion.
  [SSRN 3247865](https://ssrn.com/abstract=3247865).
- Lakonishok, J. & Smidt, S. (1988). "Are Seasonal Anomalies Real? A Ninety-Year
  Perspective." *Review of Financial Studies* 1(4). — the turn-of-the-month window;
  McConnell, J.J. & Xu, W. (2008). "Equity Returns at the Turn of the Month." *Financial
  Analysts Journal* 64(2). Torn down in [study 42 — Last-Call](../../42-last-call/).
- Whaley, R.E. (2000). "The Investor Fear Gauge." *Journal of Portfolio Management* 26(3).
  — VIX stress and forward equity returns. Torn down in
  [study 03 — Fear-Gauge](../../03-fear-gauge/).
- The red-open/red-day continuation lore — torn down at intraday horizon in
  [study 13 — Crimson-Hour](../../13-crimson-hour/).
- Nagel, S. (2012). "Evaporating Liquidity." *Review of Financial Studies* 25(7). — the
  common mechanism steelmanned here: short-horizon reversal as compensation for
  liquidity provision, strongest under stress.

## The overlay

- Moreira, A. & Muir, T. (2017). "Volatility-Managed Portfolios." *Journal of Finance*
  72(4). — the vol-targeting layer, certified on this bench in
  [study 16 — Storm-Shy](../../16-storm-shy/).
- [Study 38 — Chorus](../../38-chorus/) — the anti-pattern this study inverts: *averaging*
  weak signals into a composite forecast adds nothing; *gating on their coincidence* is
  the alternative under test.

## Method

- White, H. (2000). "A Reality Check for Data Snooping." *Econometrica* 68(5).
- Politis, D.N. & Romano, J.P. (1994). "The Stationary Bootstrap." *JASA* 89(428).
- Newey, W.K. & West, K.D. (1987). "A Simple, Positive Semi-definite, Heteroskedasticity
  and Autocorrelation Consistent Covariance Matrix." *Econometrica* 55(3).
- House protocol: [`METHODOLOGY.md`](../../../METHODOLOGY.md); pre-registration:
  [`preregistration.md`](preregistration.md); engine: [`quantlab/`](../../../quantlab/).

## Data

- SPY split-only daily OHLC and ^VIX daily closes via Yahoo! Finance (yfinance), pinned
  as-of 2026-06-01 and fingerprinted in [`results.md`](results.md); ^IRX 13-week T-bill
  as the per-day cash rate (shared cache with study 42).
