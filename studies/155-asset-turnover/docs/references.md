# References — Study 155 (Asset-Turnover)

## Primary literature

**DuPont decomposition and the return on equity identity**

- Bliss, R. T., & Rosen, R. J. (2001). "CEO Compensation and Bank Mergers." *Journal of Financial
  Economics*, 61(1), 107–138.  Early application of the DuPont framework to decompose performance
  drivers into margin, turnover, and leverage.

- Nissim, D., & Penman, S. H. (2001). "Ratio Analysis and Equity Valuation: From Research to
  Practice." *Review of Accounting Studies*, 6(1), 109–154.  The reference decomposition of ROE
  into profitability × asset turnover × leverage, used in fundamental equity analysis.

**Asset turnover as a factor**

- Novy-Marx, R. (2013). "The Other Side of Value: The Gross Profitability Premium." *Journal of
  Financial Economics*, 108(1), 1–28.  Gross profit / assets predicts returns; closely related to
  AT in that both reward revenue generation relative to assets.  The paper discusses why
  sales-based metrics carry return predictive power.

- Asness, C., Frazzini, A., & Pedersen, L. H. (2019). "Quality Minus Junk." *Review of Accounting
  Studies*, 24(1), 34–112.  Defines a quality factor that includes profitability, growth, safety,
  and payout — asset efficiency (related to AT) is one sub-component.  Demonstrates that composite
  quality factors are more robust than individual components.

- Greenblatt, J. (2005). *The Little Book That Beats the Market.* Wiley.  Popularised the combined
  Return on Capital (closely related to AT) + Earnings Yield rank as a stock selection heuristic.
  Study 121 (Magic-Formula) tests this approach on the same EDGAR cache.

**Factor redundancy and independence tests**

- Harvey, C. R., Liu, Y., & Zhu, H. (2016). "... and the Cross-Section of Expected Returns."
  *Review of Financial Studies*, 29(1), 5–68.  Establishes the multiple-testing problem in factor
  discovery and the inflated *t*-stat bar (≥ 3.0) needed for new factors to be taken seriously.
  Directly relevant to evaluating a HAC *t* of 1.96.

- Hou, K., Xue, C., & Zhang, L. (2020). "Replicating Anomalies." *Review of Financial Studies*,
  33(5), 2019–2133.  Systematic replication of 447 anomalies; find that many fail to survive
  value-weighted portfolios, micro-cap exclusions, or survival-bias correction.

**Survivorship bias**

- Brown, S. J., Goetzmann, W., Ibbotson, R. G., & Ross, S. A. (1992). "Survivorship Bias in
  Performance Studies." *Review of Financial Studies*, 5(4), 553–580.  Classic reference
  quantifying how survivorship bias overstates measured performance.

- Elton, E. J., Gruber, M. J., & Blake, C. R. (1996). "Survivorship Bias and Mutual Fund
  Performance." *Review of Financial Studies*, 9(4), 1097–1120.  Demonstrates the magnitude of
  survivorship distortion (several percentage points per year in some settings).

## Methodology

- Newey, W. K., & West, K. D. (1987). "A Simple, Positive Semi-Definite, Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix." *Econometrica*, 55(3), 703–708.  The HAC standard
  error used for inference on the annual hedge series.

- Politis, D. N., & Romano, J. P. (1994). "The Stationary Bootstrap." *Journal of the American
  Statistical Association*, 89(428), 1303–1313.  Block-bootstrap methodology used in Study 155's
  companion notebooks for Sharpe ratio confidence intervals.

## Data sources

- **EDGAR** (via desk shared cache): Annual 10-K fundamentals for current S&P 500 members.
  Concepts used: `Revenues`, `Assets`, `NetIncomeLoss`.  Survivorship-biased (current members only).
  URL: https://www.sec.gov/cgi-bin/browse-edgar

- **Annual returns panel** (`_edgar_yrret.parquet`): Pre-computed calendar-year returns for the
  same ticker universe, drawn from Yahoo Finance via `yfinance`.
