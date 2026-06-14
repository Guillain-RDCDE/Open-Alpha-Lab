# References & literature map — Study 141 (Turnover-Anomaly)

## The claim under test

- **Datar, V.T., Naik, N.Y., & Radcliffe, R. (1998).** *Liquidity and stock returns: An
  alternative test.* Journal of Financial Markets, 1(2), 203–219. The canonical paper:
  stocks with **higher share turnover** (trading volume / shares outstanding) earn **lower
  subsequent returns**, after controlling for size, book-to-market, and momentum. The authors
  propose turnover as a proxy for illiquidity and divergence of opinion; both predict
  underperformance. Effect documented on NYSE/AMEX universe 1963–1991.

## Why the idea is almost coherent

- **Amihud, Y., & Mendelson, H. (1986).** *Asset pricing and the bid-ask spread.*
  Journal of Financial Economics, 17(2), 223–249. The liquidity-premium argument: investors
  demand compensation for illiquidity. Under this view, *low*-turnover stocks carry an
  illiquidity premium and should outperform. Datar-Naik-Radcliffe tests the same idea with
  turnover as the measure.

- **Miller, E.M. (1977).** *Risk, uncertainty, and divergence of opinion.* Journal of
  Finance, 32(4), 1151–1168. High turnover reflects disagreement among investors; with short-
  sale constraints, prices reflect the optimistic investor's view, biasing high-turnover stocks
  toward overvaluation and subsequent underperformance. A key theoretical pillar of the claim.

- **Barber, B.M., & Odean, T. (2008).** *All that glitters: The effect of attention and news
  on the buying behavior of individual and institutional investors.* Review of Financial
  Studies, 21(2), 785–818. High-turnover / high-attention stocks attract retail buyers who
  drive prices temporarily above fundamental value, predicting lower future returns.

## Counter-evidence and complications

- **Lee, C.M.C., & Swaminathan, B. (2000).** *Price momentum and trading volume.* Journal of
  Finance, 55(5), 2017–2069. Turnover interacts with price momentum: high-turnover, recent-
  winner stocks earn the *highest* near-term momentum returns, reversing only over longer
  horizons. This creates a sign ambiguity that depends on the measurement period.

- **Brennan, M.J., Chordia, T., & Subrahmanyam, A. (1998).** *Alternative factor specifications,
  security characteristics, and the cross-section of expected stock returns.* Journal of
  Financial Economics, 49(3), 345–373. Documents volume / turnover as cross-sectional
  predictors, with sign and magnitude sensitive to controlling variables.

- **McLean, R.D., & Pontiff, J. (2016).** *Does academic publication destroy stock return
  predictability?* Journal of Finance, 71(1), 5–32. Cross-sectional predictors (including
  liquidity proxies) systematically decay post-publication; the post-1998 effect may be
  attenuated or reversed.

## Why this study reverses the claim (survivorship bias)

- **Shumway, T. (1997).** *The delisting bias in CRSP's Nasdaq data and its implications for
  the size effect.* Journal of Finance, 52(1), 361–382. Studies using surviving firms only
  systematically overstate returns of higher-risk portfolios. High-turnover firms that failed
  are excluded from the EDGAR panel, producing an upward-biased return for the high-turnover
  quintile in our sample.

- **Kothari, S.P., Shanken, J., & Sloan, R.G. (1995).** *Another look at the cross-section of
  expected returns.* Journal of Finance, 50(1), 185–224. Survivorship bias in COMPUSTAT
  data is estimated to inflate anomaly returns by several percent per year — consistent with
  the magnitude of the reversal observed here.

## Method lineage (the desk's shared engine)

- **Newey, W.K., & West, K.D. (1987).** *A simple, positive semi-definite, heteroskedasticity
  and autocorrelation consistent covariance matrix.* Econometrica, 55(3), 703–708. The HAC
  standard error used in `strategy.summarize` for the t-stat on annual hedge returns.

- **Fama, E.F., & French, K.R. (1992).** *The cross-section of expected stock returns.*
  Journal of Finance, 47(2), 427–465. The standard portfolio-sort methodology (quintile
  sorts, equal-weight returns, annual rebalancing) adopted here.

## Related desk studies

- **[Study 121 — Magic-Formula](../../121-magic-formula/)**: the same EDGAR fundamentals +
  yfinance price panel, the same quintile-sort methodology. Survivorship-biased results
  treated as upper bounds.
- **[Study 65 — Scorecard](../../65-scorecard/)**: another EDGAR-based factor study with the
  same survival-bias caveat and comparison to a shuffled-label null.
- **[Study 52 — Smoke-Screen](../../52-smoke-screen/)**: the clearest desk case of
  survivorship bias dominating a fundamental signal — the cautionary model for this study.
