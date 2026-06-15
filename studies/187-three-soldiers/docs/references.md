# References & literature map — Study 187 (Three-Soldiers)

## The claim under test

- **The folk recipe.** Three White Soldiers and Three Black Crows are among the most widely
  taught Japanese candlestick patterns, appearing in virtually every candlestick textbook and
  trading education resource.  The claim: *"Three consecutive strong bullish (or bearish)
  candles, each opening within the prior candle's body and closing near its high (or low),
  signal a powerful continuation of the prevailing trend — or the start of a new one after a
  consolidation."*  Sources: Nison (1991), Morris (2006), Bulkowski (2008).  We steelman it
  as: *the pattern's directional signal carries statistically meaningful forward return
  information that exceeds a random-day baseline in the claimed direction.*

## The canonical sources for the patterns

- **Nison, S. (1991).** *Japanese Candlestick Charting Techniques.* New York Institute of
  Finance.  The book that introduced Japanese candlestick patterns to Western audiences.
  Describes Three White Soldiers as a "very bullish signal" when each white candle opens
  within the prior body and closes near the high, implying a strong trend continuation.
  Three Black Crows are the mirror image bearish signal.

- **Morris, G. (2006).** *Candlestick Charting Explained: Timeless Techniques for Trading
  Stocks and Futures.* McGraw-Hill.  Formalises the pattern rules used in this study and
  provides frequency analysis; notes that both patterns are rare and warns against over-
  reliance on isolated candlestick signals.

- **Bulkowski, T. (2008).** *Encyclopedia of Candlestick Charts.* Wiley Trading.  The most
  rigorous empirical treatment of candlestick patterns.  Bulkowski finds that Three White
  Soldiers has a bullish breakout rate of ~82% but *average* forward returns that are
  unremarkable, a pattern that aligns with our finding that the raw forward return eventually
  turns positive (upward drift) but the *excess* over a baseline does not.

## Why the claim is coherent but probably wrong

- **Short-term momentum vs reversion.**  The pattern bets on continuation (momentum), but
  at the 1-5 day horizon, mean reversion tends to dominate in equities.  A three-day
  run-up already represents an above-average return, and markets tend to pull back.  This
  is the mechanism behind our finding that 3WS fires near exhaustion and 3BC fires near
  short-term bottoms — both patterns trigger *after* a local extreme, not at the start of
  a new trend.

- **Jegadeesh & Titman (1993).** *Returns to Buying Winners and Selling Losers:
  Implications for Stock Market Efficiency.* Journal of Finance, 48(1), 65-91.  Momentum
  works at 3-12 month horizons, not 1-5 day horizons.  At very short horizons Jegadeesh
  (1990) shows reversal.

- **Jegadeesh, N. (1990).** *Evidence of Predictable Behavior of Security Returns.*
  Journal of Finance, 45(3), 881-898.  Documents 1-month reversal in US stocks —
  consistent with the pattern triggering near exhaustion and the next bar reversing.

- **Lo, A., Mamaysky, H., & Wang, J. (2000).** *Foundations of Technical Analysis:
  Computational Algorithms, Statistical Inference, and Empirical Implementation.* Journal
  of Finance, 55(4), 1705-1765.  A landmark attempt to quantify technical patterns; finds
  some patterns carry information but the effect is weak and inconsistent across time.

## Empirical literature on candlestick patterns

- **Lim, C. M., & Loh, W. L. (2014).** Revisiting calendar anomalies in Asian stock
  markets using a stochastic dominance approach.  Notes the challenge of generalising
  textbook pattern rules to gapped-open modern markets where the strict "open within prior
  body" condition is rarely satisfied, consistent with this study's relaxed detector.

- **Marshall, B. R., Young, M. R., & Rose, L. C. (2006).** *Candlestick Technical
  Trading Strategies: Can They Create Value for Investors?* Journal of Banking and Finance,
  30(8), 2303-2323.  Tests candlestick patterns on US stocks and finds minimal predictive
  ability after adjusting for risk.  Their results on bullish/bearish three-bar patterns
  are consistent with this study's finding of no significant excess return.

- **Goo, Y. J., Chen, D. H., & Chang, Y. W. (2007).** The application of Japanese
  candlestick trading strategies in Taiwan.  Investment Management and Financial
  Innovations.  Finds some in-sample predictability for candlestick patterns in Taiwan
  but warns of data-snooping bias and lack of out-of-sample robustness — a common finding
  in the chart-pattern literature.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix.* Econometrica, 55(3),
  703-708.  Used in `strategy.summarize_pattern`.

- **Multiple comparisons / Bonferroni correction.**  Two patterns × three horizons = six
  t-stats; Bonferroni threshold |t| ≥ 2.65 for α = 5%.  See Romano & Wolf (2005),
  *Stepwise Multiple Testing as Formalized Data Snooping.* Econometrica.

- **Random-day control.**  The unconditional baseline used throughout this study.  A
  pattern adds real signal only if it beats a random-day entry in the same claimed
  direction.  This is the standard false-discovery discipline described in White (2000),
  *A Reality Check for Data Snooping.* Econometrica, 68(5), 1097-1126.

## Related desk studies

- **[Study 76 — Rice-Paper](../../76-rice-paper/)**: five single-bar Japanese candlestick
  reversal patterns (bullish engulfing, bearish engulfing, hammer, shooting star, doji)
  on the same basket — a closely related family of pattern tests, also finding no robust
  edge.

- **[Study 17 — Glass-Ceiling](../../17-glass-ceiling/)**: multi-bar chart-pattern
  detection using a barrier backtest and random-day control — the same protocol applied to
  a breakout family.

- **[Study 127 — Williams-R](../../127-williams-r/)**: overbought/oversold technical
  indicator on the same basket, using the same random-baseline framework.
