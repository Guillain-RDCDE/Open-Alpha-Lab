# References & literature map — Study 159 (Presidential-Party)

## The claim under test

- **The canonical source.** Santa-Clara, P. & Valkanov, R. (2003). *'The Presidential
  Puzzle: Political Cycles and the Stock Market'*. **Journal of Finance, 58(5)**, 1841-1872.
  The paper that made the effect famous: using Shiller data from 1927-1998, they find
  D presidents outperform R presidents by roughly 9 percentage points per year in *excess*
  real returns (over T-bill), with an even larger nominal gap. They explicitly state the
  mechanism is "not explained by business conditions or risk" — a claim the subsequent
  literature has contested aggressively. [doi:10.1111/1540-6261.00600]

## The mechanism debate — luck vs policy

- **Blinder, A.S. & Watson, M.W. (2016).** *'Presidents and the U.S. Economy: An
  Econometric Exploration'*. **American Economic Review, 106(4)**, 1015-1045. The most
  rigorous counter-analysis: they find that ~75% of the Democratic outperformance is
  explained by *lucky* factors outside presidential control — oil price shocks,
  productivity surprises, international growth, and consumer confidence. The remainder
  (25%) is genuinely unexplained. Their conclusion: presidents get too much credit (and
  blame) for macroeconomic outcomes. [doi:10.1257/aer.20140913]

- **Belo, F., Gala, V.D. & Li, J. (2013).** *'Government Spending, Political Cycles,
  and the Cross-Section of Stock Returns'*. **Journal of Financial Economics, 107(2)**,
  305-324. A cross-sectional study: government procurement-heavy sectors outperform under
  Democrats, consistent with fiscal-policy rather than monetary-policy mechanisms.
  [doi:10.1016/j.jfineco.2012.08.009]

- **Kelly, B., Pástor, L. & Veronesi, P. (2016).** *'The Price of Political Uncertainty:
  Theory and Evidence from the Option Market'*. **Journal of Finance, 71(5)**, 2417-2480.
  Documents that political uncertainty (measured via option implied volatility around
  elections) is priced, consistent with higher risk premia under one party being a
  compensation for uncertainty rather than a free alpha. [doi:10.1111/jofi.12406]

## The small-n critique

- **Pástor, L. & Veronesi, P. (2020).** *'Political Cycles and Stock Returns'*.
  **Journal of Political Economy, 128(11)**, 4011-4045. A structural model showing that
  higher Democratic equity returns are consistent with rational risk pricing: Democrat
  presidents are elected when risk aversion is high (after market downturns), so the
  premium represents compensation for political risk, not alpha.

- **Knight, B. (2006).** *'Are Policy Platforms Capitalized into Equity Prices?
  Evidence from the Bush/Gore 2000 Presidential Election'*. **Journal of Public
  Economics, 90(4-5)**, 751-773. Shows that industry-level returns in the 2000 election
  moved *before* the outcome was known, consistent with markets pricing party policy in
  the election period, not the administration period.

## The year-of-term companion (distinct from party effect)

- **Hirsch, Y. (1968, 1986, annually).** *Stock Trader's Almanac*. The original
  popularisation of the "presidential election cycle" — specifically that year 3 (pre-
  election) is the bonanza and year 1 (post-election) is the lean year. This is distinct
  from the party claim tested here; Study 81 (Four-Year-Itch) in this desk covers it.

- **Huang, R.D. (1985).** *'Common Stock Returns and Presidential Elections'*.
  **Financial Analysts Journal, 41(2)**, 58-61. Early empirical work on the cycle,
  finding year-3 and year-4 pre-election years outperform year 1 and 2. The party-gap
  effect is a *separate* dimension.

## Method lineage (the desk's shared engine)

- **Newey, W.K. & West, K.D. (1987).** *'A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix'*.
  **Econometrica, 55(3)**, 703-708. The HAC estimator used in
  [`strategy._hac_tstat`](../presidential_party/strategy.py).

- **Politis, D.N. & Romano, J.P. (1994).** *'The Stationary Bootstrap'*.
  **Journal of the American Statistical Association, 89(428)**, 1303-1313. The circular
  block bootstrap methodology underlying
  [`quantlab.stats.sharpe_ci_bootstrap`](../../../quantlab/stats.py).

## Data sources used here

- **Shiller, R.J.** *S&P 500 monthly data*, available at
  [http://www.econ.yale.edu/~shiller/data.htm](http://www.econ.yale.edu/~shiller/data.htm)
  (reproduced from *Irrational Exuberance*, 3rd ed., Princeton University Press, 2015).
  Covers 1871-2023. The 'Real Price' column uses the CPI deflator; our 1927-2023 window
  covers the modern equity market era. **Pre-staged at `_cache/shiller_sp500.parquet`**.

- **U.S. National Archives & Wikipedia.** Presidential start/end dates and party labels
  used in the hardcoded `PRESIDENTS` table in `data.py`. All dates are publicly verifiable
  against the National Archives record of presidential inaugurations.

## Related desk studies

- **[Study 81 — Four-Year-Itch](../../81-four-year-itch/)**: the year-of-term cycle
  (year 3 bonanza) — a *different* political calendar claim, distinct from party identity.
  Tested on the same ^GSPC daily tape; also suffers the small-n problem (~24 terms vs
  17 here), but the year-3 hypothesis has more observations per "signal" cell.

- **[Study 80 — Cold-Open](../../80-cold-open/)**: the January Effect, another seasonal
  / calendar anomaly where the raw gap is real in the data but statististically fragile
  once the sample is broken into sub-periods.

- **[Study 136 — Mark-Twain](../../136-mark-twain/)**: 'Sell in May and go away',
  another famous seasonal folklore claim — same playful-tone treatment, similar small-n
  verdict on the mechanism.
