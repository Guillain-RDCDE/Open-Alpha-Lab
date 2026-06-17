# References & literature map -- Study 259 (News-Tone)

## The canonical claim: news sentiment predicts returns

- **Tetlock, P. C. (2007).** *Giving Content to Investor Sentiment: The Role of
  Media in the Stock Market.* Journal of Finance, 62(3), 1139--1168.
  The founding paper: high media pessimism (a tone index built from the WSJ
  "Abreast of the Market" column) predicts downward pressure on prices followed
  by reversion to fundamentals. The predictability is *short-lived* (days) and
  the effect is small -- a key caveat often dropped when the idea is sold.

- **Tetlock, P. C., Saar-Tsechansky, M. & Macskassy, S. (2008).** *More Than
  Words: Quantifying Language to Measure Firms' Fundamentals.* Journal of
  Finance, 63(3), 1437--1467. Firm-level negative-word counts forecast earnings
  and (weakly) next-day returns, but the return effect largely reverses.

- **Garcia, D. (2013).** *Sentiment During Recessions.* Journal of Finance,
  68(3), 1267--1300. News tone predicts returns mainly in recessions; the
  effect is concentrated and time-varying -- not a stable, always-on signal.

## Why aggregate next-day tone signals usually fail

- **Loughran, T. & McDonald, B. (2011).** *When Is a Liability Not a Liability?
  Textual Analysis, Dictionaries, and 10-Ks.* Journal of Finance, 66(1), 35--65.
  Shows that naive sentiment dictionaries mislabel finance text and that
  measured "tone" is dominated by contemporaneous, not predictive, content.

- **Baker, S. R., Bloom, N. & Davis, S. J. (2016).** *Measuring Economic Policy
  Uncertainty.* Quarterly Journal of Economics, 131(4), 1593--1636. The EPU
  index (a headline-based macro-mood proxy in the spirit of our tone series) is
  strongly *contemporaneous* with markets; its forward predictive content for
  daily index returns is weak.

- **McLean, R. D. & Pontiff, J. (2016).** *Does Academic Research Destroy Stock
  Return Predictability?* Journal of Finance, 71(1), 5--32. Anomaly returns decay
  ~58% post-publication; widely-publicised sentiment effects are prime
  candidates for arbitrage away at the index level.

## The methodological trap this study is built to expose

- **Look-ahead / hindsight bias in curated indicators.** A sentiment series
  hand-labelled *after* the fact (or aggregated to a low frequency that
  straddles the very returns it "predicts") leaks the outcome. The within-month
  demeaning placebo here isolates genuine day-to-day information from the
  month-level drift -- a direct analogue of the in-sample/out-of-sample split.
  See **Lo & MacKinlay (1990)**, *Data-Snooping Biases in Tests of Financial
  Asset Pricing Models* (Review of Financial Studies, 3(3), 431--467).

- **Contemporaneous vs predictive correlation.** Bad-news days are down days by
  construction; the same-day correlation is mechanical. Only the strictly
  forward (next-day) test, formed on information available at the prior close,
  is tradable. This distinction is the spine of the study.

## Related desk studies

- **[Study 158 -- Super-Bowl](../../158-super-bowl/)**: another hardcoded-table
  folklore signal where the honest base-rate / placebo kills an apparently
  strong indicator.
- **[Study 223 -- Same-Month-Seasonality](../../223-same-month-seasonality/)**:
  the synthetic-panel + cached-proxy template this study mirrors.

## Method lineage

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive
  Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance
  Matrix* (Econometrica). Used for both the regression slope and the strategy
  mean t-stat.
