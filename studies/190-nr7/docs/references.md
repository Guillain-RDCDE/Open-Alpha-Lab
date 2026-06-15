# References & literature map — Study 190 (NR7)

## The claim under test

- **Tony Crabel (1990).** *Day Trading With Short-Term Price Patterns and Opening Range
  Breakout.* Traders Press. The original source for the NR7 pattern: a day whose
  high-minus-low range is the *narrowest* of the last 7 days signals "volatility contraction
  that precedes expansion." Crabel tested it on futures contracts in the 1980s and found
  modest next-day range expansion and a breakout edge (buy a break above the NR7 high, sell a
  break below the NR7 low). The desk steelmans this as two separate, jointly testable
  hypotheses: (H1) range expansion and (H2) directional breakout edge vs a random-day baseline.

## Why the volatility-clustering claim has coherent theoretical backing

- **Mandelbrot (1963).** *The Variation of Certain Speculative Prices.* Journal of Business.
  Documents that large changes in prices tend to be followed by large changes, and small by
  small — the original "volatility clustering" observation.
- **Engle (1982).** *Autoregressive Conditional Heteroskedasticity with Estimates of the
  Variance of United Kingdom Inflation.* Econometrica, 50(4), 987–1007.  The ARCH model
  formalises volatility clustering: conditional variance depends on lagged squared returns.
  NR7 is an informal, non-parametric attempt to detect a low-volatility regime that should
  revert to the mean — but ARCH implies the reversion is gradual and noisy, not a clean
  one-day expansion.
- **Bollerslev (1986).** *Generalised Autoregressive Conditional Heteroskedasticity.*
  Journal of Econometrics, 31(3), 307–327. GARCH(1,1) extends ARCH; the estimated
  persistence parameter close to 1 in equity returns implies that low-vol days tend to
  *persist*, not expand — consistent with our finding (ratio = 0.92 after NR7).
- **Schwert (1989).** *Why Does Stock Market Volatility Change Over Time?* Journal of Finance,
  44(5), 1115–1153. Documents regime-like volatility behaviour at low frequencies; within a
  regime, short-horizon reversals of compressed vol are weak.

## On breakout patterns and technical analysis

- **Brock, Lakonishok & LeBaron (1992).** *Simple Technical Trading Rules and the Stochastic
  Properties of Stock Returns.* Journal of Finance, 47(5), 1731–1764. Found predictive power
  in moving-average and trading-range breakout rules on DJIA 1897–1986; however, data-snooping
  and costs substantially erode the edge in most replications.
- **Park & Irwin (2007).** *What Do We Know About the Profitability of Technical Analysis?*
  Journal of Economic Surveys, 21(4), 786–826.  A comprehensive meta-study: the positive
  evidence for technical rules is heavily concentrated in early samples and largely evaporates
  out-of-sample, after data-snooping correction (White 2000 Reality Check), and once costs are
  applied at realistic turnover.
- **Faber (2010).** *Relative Strength Strategies for Investing.* Mebane Faber Research.
  Argues that simple momentum/breakout rules have persistent cross-asset efficacy; notable
  counter-evidence is the lack of persistence within a single instrument at short horizons.
- **Jegadeesh & Titman (1993).** *Returns to Buying Winners and Selling Losers: Implications
  for Stock Market Efficiency.* Journal of Finance, 48(1), 65–91. Momentum at the 3–12 month
  horizon; at very short (1-day) horizons, the dominant effect is usually *reversal*, not
  continuation — inconsistent with a one-day breakout edge.

## Costs, capacity, and the realistic hurdle

- **Novy-Marx & Velikov (2016).** *A Taxonomy of Anomalies and Their Trading Costs.* Review
  of Financial Studies, 29(1), 104–147. Finds that most published anomalies are strongly
  attenuated or eliminated once realistic transaction costs are applied, particularly for
  high-turnover rules. NR7's ~39 signals/year on a single instrument is moderate turnover,
  but at bid-ask levels on ETFs the break-even is thin.
- **Korajczyk & Sadka (2004).** *Are Momentum Profits Robust to Trading Costs?* Journal of
  Finance, 59(3), 1039–1082. Even for better-documented effects, cost sensitivity is high
  at round-trip costs of 5–10 bps; the NR7 gap vs the random-day control (+9.8 bps) sits
  exactly in this fragile zone.

## Survivorship and look-ahead in stock universes

- **Kothari, Shanken & Sloan (1995).** *Another Look at the Cross-Section of Expected Returns.*
  Journal of Finance, 50(1), 185–224. Documents survivorship bias inflating backtests on
  single-stock universes; TSLA and NVDA enter the test universe only after their most dramatic
  growth phases — the pooled result is skewed toward recent highly-volatile, positive-momentum
  names. Named explicitly on the Signal axis in this study.
- **Lo & MacKinlay (1990).** *Data-Snooping Biases in Tests of Financial Asset Pricing Models.*
  Review of Financial Studies, 3(3), 431–467. NR7 is one of many chart-pattern variants; the
  lookback of 7 is Crabel's choice, but a researcher free to choose 5, 7, 10, etc. inflates
  the effective significance threshold.

## Method lineage (the desk's shared engine)

- **Newey & West (1987).** *A Simple, Positive Semi-Definite, Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix.* Econometrica — the HAC t-stat used in
  `strategy.summarize` and `strategy.summarize_expansion`.
- **Politis & Romano (1994).** *The Stationary Bootstrap.* JASA — the block-bootstrap Sharpe
  CI in `quantlab.stats.sharpe_ci_bootstrap`.
- **Lo (2002).** *The Statistics of Sharpe Ratios.* Financial Analysts Journal —
  `quantlab.analytics.sharpe_with_se`.

## Related desk studies

- **[Study 72 — Loaded-Dice](../../72-loaded-dice/)**: the intraday SMA(5/10) crossover scalp,
  the same "random-direction control" methodology applied to 5-minute bars — also NONE/MIRAGE.
- **[Study 127 — Williams-R](../../127-williams-r/)**: another short-term overbought/oversold
  indicator applied to daily bars with the same random-baseline discipline.
- **[Study 21 — Fools-Gold](../../21-fools-gold/)**: the 50/200 daily golden cross — moving-
  average family, daily bars, similar verdict.
- **[Study 76 — Rice-Paper](../../76-rice-paper/)**: candlestick patterns on daily bars — the
  closest structural analogue (visual pattern → one-day forward return).
