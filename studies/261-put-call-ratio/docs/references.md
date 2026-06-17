# References & literature map -- Study 261 (Put-Call-Ratio)

## The canonical claim

- **CBOE / market lore.** The put/call ratio (puts traded / calls traded) is the
  textbook contrarian "fear gauge by flow". The standard rule: a high ratio means
  the crowd is buying protection in panic -- selling is exhausted, so it is a
  contrarian *buy*; a very low ratio (complacency/greed) is a *sell*. The CBOE
  began disseminating the daily total put/call ratio in 2003; the long-run mean
  sits near ~0.92 with spikes above ~1.2 in genuine panics (Oct-2008, Dec-2018,
  Mar-2020).

## Academic evidence on option-flow sentiment

- **Pan, J. & Poteshman, A. M. (2006).** *The Information in Option Volume for
  Future Stock Prices.* Review of Financial Studies, 19(3), 871--908.
  Finds that a high put/call *volume* ratio predicts *lower* future returns at the
  single-stock level -- the *opposite* sign of the contrarian-bottom folklore.
  Informed traders buying puts presage declines, not rebounds. The signal is also
  short-lived (days, not the month horizon tested here).

- **Blau, B. M., Nguyen, N. & Whitby, R. J. (2014).** *The information content of
  option ratios.* Journal of Banking & Finance, 43, 179--187.
  Examines put/call and option-to-stock volume ratios; the predictive content is
  concentrated in the cross-section of individual names and is largely arbitraged
  at the index level.

- **Bandopadhyaya, A. & Jones, A. L. (2008).** *Measures of Investor Sentiment.*
  Journal of Business & Economics Research, 6(8).
  Treats the put/call ratio as one of several sentiment proxies; documents weak,
  unstable, and regime-dependent links to index returns.

- **Baker, M. & Wurgler, J. (2006).** *Investor Sentiment and the Cross-Section of
  Stock Returns.* Journal of Finance, 61(4), 1645--1680.
  The reference work on sentiment-and-returns: sentiment proxies (the put/call
  ratio among them) matter mostly for hard-to-arbitrage small/speculative stocks,
  not for timing the broad index -- consistent with the null we find on ^GSPC.

## Why the contrarian timing rule fails here

- **Look-ahead and base rates.** The S&P rises in ~63% of months unconditionally;
  any rule that is in-market only ~26% of the time forfeits most of that drift.
  The honest benchmark is always-invested buy-and-hold, not cash.
- **Few independent extremes.** A 23-year monthly sample contains only a handful
  of genuinely independent "extreme-fear" episodes (clustered in 2008, 2011, 2018,
  2020, 2022). The effective n on the tail is tiny -- inference is fragile.
- **Wrong sign at the stock level.** Pan-Poteshman's result (high put volume ->
  lower returns) means the contrarian story already contradicts the micro
  evidence; at the index level we find no reliable effect of either sign.

## Method lineage

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive
  Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance
  Matrix* (Econometrica).
- **Predictive-regression caveats.** Stambaugh (1999), *Predictive Regressions*
  (JFE): persistent predictors (the put/call ratio is mildly persistent) bias
  small-sample slope t-stats -- another reason to treat any thin edge sceptically.

## Related desk studies

- **[Study 158 -- Super-Bowl](../../158-super-bowl/)**: the base-rate trap for a
  binary up/down market predictor -- the same "70% of years are up anyway" logic
  applies to "in-market only when fearful".
- **Other sentiment / fear gauges** in the desk (VIX term-structure, MOVE) test
  adjacent "buy the fear" claims; this study is the option-flow cousin.
