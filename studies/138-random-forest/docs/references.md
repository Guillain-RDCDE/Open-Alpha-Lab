# References — Study 138 (Random-Forest)

## Core papers

1. **Breiman, L. (2001).** "Random Forests." *Machine Learning* 45(1), 5-32.
   The original Random Forest paper. The ensemble method that is the subject of this study.

2. **Harvey, C. R., Liu, Y., & Zhu, H. (2016).** "... and the Cross-Section of Expected Returns."
   *Review of Financial Studies* 29(1), 5-68.
   The "t > 3.0" bar for claiming a factor is real; motivates the study's shuffle-control
   falsification approach. Most ML-discovered factors do not survive this standard.

3. **Aronson, D. R. (2007).** *Evidence-Based Technical Analysis.* Wiley.
   Systematic treatment of why standard technical indicators (RSI, momentum, lagged returns)
   fail significance tests once multiple comparisons are accounted for.

4. **Bailey, D. H., Borwein, J., Lopez de Prado, M., & Zhu, Q. J. (2014).**
   "The Probability of Backtest Overfitting." *Journal of Computational Finance* 20(4), 39-70.
   Establishes that a large enough parameter grid will always find an in-sample fit that
   overfits a random walk — the motivation for the out-of-sample walk-forward protocol.

5. **Lopez de Prado, M. (2018).** *Advances in Financial Machine Learning.* Wiley.
   Chapters 7-11 detail purged walk-forward cross-validation and the combinatorial purged CV
   method. Our simpler rolling walk-forward is a deliberate conservative baseline.

6. **Gu, S., Kelly, B., & Xiu, D. (2020).** "Empirical Asset Pricing via Machine Learning."
   *Review of Financial Studies* 33(5), 2223-2273.
   Found that tree ensembles have the best predictive R^2 out-of-sample across a large
   feature set — but with 900+ features and monthly rebalancing, not daily single-ticker RF.
   This study tests the accessible retail version of the same idea.

7. **Krauss, C., Do, X. A., & Huck, N. (2017).** "Deep Neural Networks, Gradient-Boosted Trees,
   Random Forests: Statistical Arbitrage on the S&P 500." *European Journal of Operational
   Research* 259(2), 689-702.
   Found daily RF accuracy ~55-56% on S&P 500 in-sample, but decay post-2010 and negligible
   net-of-costs returns — consistent with this study's walk-forward results.

8. **McLean, R. D., & Pontiff, J. (2016).** "Does Academic Research Destroy Stock Return
   Predictability?" *Journal of Finance* 71(1), 5-32.
   Post-publication decay: once a pattern is documented, arbitrageurs erode it. RF-discovered
   patterns face the same decay, especially short-horizon technical ones.

## Related studies in this repo

- **Study 39 — Black-Box:** Neural network (MLP) on crypto daily returns; same walk-forward /
  shuffled-label protocol on a different model family and asset class.
- **Study 12 — Paper-Prophet:** Facebook Prophet time-series forecasting on equities; in-sample
  forecast vs out-of-sample reality.
