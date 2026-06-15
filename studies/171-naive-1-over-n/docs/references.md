# References & literature map — Study 171 (Naive-1-Over-N)

## The claim under test

The central result of DeMiguel, Garlappi & Uppal (2009): *"the 1/N equal-weight rule
cannot be consistently beaten by any of 14 optimisation-based portfolio strategies
out of sample, across seven empirical datasets."* The argument is that the theoretical
advantage of mean-variance optimisation is offset — often dominated — by estimation
error from finite historical samples. This is not a fringe claim: it is a highly cited
result in the *Review of Financial Studies* that reframed how practitioners think about
portfolio construction. The question here is whether it holds on a post-2018 sector ETF
universe, and whether the Markowitz optimiser (the canonical villain) is truly a mirage.

## The seminal paper

- **DeMiguel, V., Garlappi, L., & Uppal, R. (2009).** *Optimal versus naive
  diversification: how inefficient is the 1/N portfolio strategy?* Review of Financial
  Studies, 22(5), 1915–1953. The canonical source for the 1/N result. Tests 14
  optimisation strategies against equal-weight across seven datasets (US industry
  portfolios, international indices, individual stocks) over rolling 60- and 120-month
  windows; 1/N wins on out-of-sample Sharpe in most cases. The "menu of models"
  includes minimum-variance, mean-variance (various risk-aversion levels), Bayes-Stein
  shrinkage, and the "optimal" (sample mean-variance) portfolio.

## Why Markowitz optimisation fails out of sample — the estimation-error story

- **Michaud, R. O. (1989).** *The Markowitz optimization enigma: is optimized optimal?*
  Financial Analysts Journal, 45(1), 31–42. Documents that sample mean-variance
  frontiers are typically "error maximisers": small errors in estimated expected returns
  are amplified by the optimiser into extreme, unstable weights.
- **Best, M. J., & Grauer, R. R. (1991).** *On the sensitivity of mean-variance-
  efficient portfolios to changes in asset means.* Review of Financial Studies, 4(2),
  315–342. Shows that tiny changes in estimated means produce large weight shifts —
  the core instability that estimation error exploits.
- **Merton, R. C. (1980).** *On estimating the expected return on the market: an
  exploratory investigation.* Journal of Financial Economics, 8(4), 323–361. Expected
  returns are notoriously hard to estimate from historical data; the signal-to-noise
  ratio is extremely low at typical sample sizes.

## The 1/N defence — why naive is not stupid

- **Stein, C. (1956).** *Inadmissibility of the usual estimator for the mean of a
  multivariate normal distribution.* Proceedings of the Third Berkeley Symposium on
  Mathematical Statistics and Probability. The theoretical root: the sample mean is
  inadmissible for three or more parameters — shrinkage (James-Stein) dominates it,
  and 1/N can be seen as an extreme shrinkage of the weight vector.
- **Tu, J., & Zhou, G. (2011).** *Markowitz meets Talmud: a combination of
  sophisticated and naive diversification strategies.* Journal of Financial Economics,
  99(1), 204–215. Shows that a blend of 1/N and the sample mean-variance portfolio
  outperforms both in some settings — the optimum is often closer to 1/N than to the
  efficient frontier.
- **Pflug, G. C., Pichler, A., & Wozabal, D. (2012).** *The 1/N investment strategy
  is optimal under high model ambiguity.* Journal of Banking & Finance, 36(2), 410–417.
  Formal result: when model uncertainty is high, 1/N is minimax-optimal.

## Related strands in the portfolio-construction literature

- **Black, F., & Litterman, R. (1992).** *Global portfolio optimization.* Financial
  Analysts Journal, 48(5), 28–43. The classic shrinkage approach: blend the optimiser's
  output with an equilibrium prior to reduce estimation error. A middle path between 1/N
  and full MVO.
- **Ledoit, O., & Wolf, M. (2004).** *Honey, I shrunk the sample covariance matrix.*
  Journal of Portfolio Management, 30(4), 110–119. Shrinkage estimator for the
  covariance matrix that reduces estimation error substantially; helps the optimiser
  but does not fully close the gap with 1/N in most OOS tests.
- **Kan, R., & Zhou, G. (2007).** *Optimal portfolio choice with parameter uncertainty.*
  Review of Financial Studies, 20(6), 1681–1706. Three-fund separation under estimation
  uncertainty: the optimal portfolio adds a hedging term for uncertainty that shrinks
  toward the riskless asset.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  [`strategy.summarize`](../naive_1_over_n/strategy.py) and
  [`quantlab.analytics.mean_tstat_hac`](../../../quantlab/analytics.py).
- **Annualised Sharpe and SE.** Lo, A. W. (2002), *The Statistics of Sharpe Ratios*
  (Financial Analysts Journal) — [`quantlab.analytics.sharpe_with_se`](../../../quantlab/analytics.py).

## Data sources used here

- **Yahoo Finance adjusted daily close prices** (via `yfinance`), eleven SPDR sector
  ETFs (XLB, XLC, XLE, XLF, XLI, XLK, XLP, XLRE, XLU, XLV, XLY). Universe start:
  2018-07-01 (first full month after XLC launched). Full adjusted total-return prices,
  so no dividend correction is required. Every headline pinned to an `as_of` date and
  a content fingerprint (`docs/results.md`).

## Related desk studies

- **[Study 102 — Free-Rebalance](../../102-free-rebalance/)**: rebalanced vs drift —
  the rebalancing bonus question (diversification return); distinct from this study's
  1/N-vs-optimiser framing.
- **[Study 144 — Permanent-Portfolio](../../144-permanent-portfolio/)**: Harry Browne's
  25/25/25/25 — another example of a naive fixed-weight portfolio tested honestly.
- **[Study 68 — All-Weather](../../68-all-weather/)**: risk-parity allocation — yet
  another "optimiser" variant; risk-parity is closer to min-variance than to 1/N.
- **[Study 97 — Balancing-Act](../../97-balancing-act/)**: mean-variance on a multi-
  asset basket — a related construction question.
