# References & literature map — Study 531 (Enterprise-Multiple)

## The claim under test

The **enterprise multiple**, EV/EBITDA, is the practitioner's value yardstick: enterprise
value (market cap + total debt − cash) divided by EBITDA (operating earnings before interest,
tax, depreciation and amortisation). A *low* multiple means the whole firm — equity plus net
debt — is cheap relative to the cash-generating operations it owns. The testable form is a
standard value sort: long the low-multiple (cheap) names, short the high-multiple (expensive)
names, hold, rebalance. Loughran & Wellman argue it is a *better* value signal than
book-to-market because EV neutralises capital-structure differences and EBITDA is closer to
operating cash flow than net income. We test the most literal monthly long-short form on a
large-cap survivor basket.

## The canonical paper

- **Loughran, T. & Wellman, J. (2011)**, *New Evidence on the Relation Between the Enterprise
  Multiple and Average Stock Returns* (Journal of Financial and Quantitative Analysis 46(6)).
  The paper this study replicates. Documents that the enterprise multiple (EV/EBITDA) is a
  robust cross-sectional return predictor in the US, that the spread is economically large,
  and — importantly — that it is **concentrated in smaller, less-liquid names** and weakens in
  the largest deciles. Frames EV/EBITDA as a proxy for an unlevered cash-flow yield.

## The broader value-factor lineage

- **Fama, E. & French, K. (1992)**, *The Cross-Section of Expected Stock Returns* (Journal of
  Finance). The foundational book-to-market value evidence; the enterprise multiple is the
  capital-structure-adjusted cousin of B/M.
- **Lakonishok, Shleifer & Vishny (1994)**, *Contrarian Investment, Extrapolation, and Risk*
  (Journal of Finance). Cash-flow/price, earnings/price and book/market all generate a value
  premium; EV/EBITDA generalises the cash-flow/price idea to the whole capital structure.
- **Gray, W. & Vogel, J. (2012)**, *Analyzing Valuation Measures: A Performance Horse Race over
  the Past 40 Years* (Journal of Portfolio Management). Races EV/EBITDA against B/M, E/P, FCF/EV
  and others; EV/EBITDA is among the best-performing single value metrics in their sample —
  direct motivation for testing it on its own.
- **Asness, Frazzini & Pedersen (2019)**, *Quality Minus Junk* (Review of Accounting Studies).
  Places cheap-multiple value alongside quality; relevant to whether a raw EV/EBITDA sort needs
  a quality overlay to be tradable.

## Why it might be gone (in large caps)

- **McLean, R. & Pontiff, J. (2016)**, *Does Academic Publication Destroy Stock Return
  Predictability?* (Journal of Finance). ~half of published anomalies decay materially after
  publication; EV/EBITDA value has been a known practitioner staple for decades.
- **Chordia, Subrahmanyam & Tong (2014)**, *Have Capital Market Anomalies Attenuated in the
  Recent Era of High Liquidity and Trading Activity?* (Journal of Accounting and Economics).
  Fundamental-factor returns in large, liquid stocks have eroded as quant AUM and liquidity
  rose — consistent with the NONE we find in a large-cap-only basket.
- **Loughran & Wellman (2011) themselves** report the premium is weakest in the largest,
  most-liquid names — exactly the universe this survivor basket samples. Finding little here is
  in-sample with their own caveat, not a contradiction of it.

## Method lineage (the desk's shared engine)

- **Newey, W. & West, K. (1987)**, *A Simple, Positive Semi-Definite, Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix* (Econometrica). The HAC t-stat in
  `strategy.summary`.
- **Spearman, C. (1904)**, *The Proof and Measurement of Association between Two Things*
  (American Journal of Psychology). The rank IC in `strategy.monthly_ic` (`scipy.stats.spearmanr`).
- **Label-shuffle permutation test.** The within-month signal shuffle in `strategy.placebo_null`
  is a standard non-parametric null for cross-sectional sorts: it preserves marginals while
  destroying the signal→return alignment.

## Related desk studies

- **[Study 530 — Book-to-Market-Value](../530-book-to-market-value/)**: the classic Fama-French
  B/M value sort. EV/EBITDA differs by adding net debt to the numerator and using operating
  earnings, not book equity, as the anchor.
- **[Study 124 — Cash-Flow-Yield](../124-cash-flow-yield/)**: OCF/market-cap as a value signal.
  EV/EBITDA is the *enterprise-level* cash-flow-yield cousin — same cheapness intuition, a
  capital-structure-neutral denominator (EBITDA over EV instead of OCF over market cap).
- **[Study 238 — Betting-Against-Beta](../238-betting-against-beta/)** and
  **[Study 330 — Low-Volatility-Anomaly](../330-low-volatility-anomaly/)**: the same
  large-cap-survivor cross-sectional sort machinery applied to risk rather than value factors.
