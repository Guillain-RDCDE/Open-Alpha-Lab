# Sources & literature map — Study 963 (The Half Day)

The half-session is a niche corner of a very old literature: **calendar effects**. What
follows is what is actually established, what is contested, and where this study sits.

## The exchange's own calendar

- **NYSE, "Holidays & Trading Hours"** — the exchange publishes the early closes each year
  (1:00 p.m. ET close, 1:15 p.m. for bonds). The three standing ones are the session before
  Independence Day, the Friday after Thanksgiving, and December 24 when it falls on a
  trading day. Occasional one-off early closes (a state funeral, weather) are not on any
  rule, which is exactly why this study confirms every candidate against the volume tape
  and publishes the thin sessions the rule failed to propose.
- **Nasdaq trading calendar** — the same schedule; the ETFs studied here trade on both.

## The holiday effect — the parent claim

- **Lakonishok, J. & Smidt, S. (1988), "Are Seasonal Anomalies Real? A Ninety-Year
  Perspective", *Review of Financial Studies* 1(4), 403-425.** The canonical long-sample
  survey: the pre-holiday day showed a positive mean return over ninety years of the Dow.
  This is the strongest version of the claim the present study inherits.
- **Ariel, R. A. (1990), "High Stock Returns Before Holidays", *Journal of Finance* 45(5),
  1611-1626.** Pre-holiday returns reported at many times the ordinary daily mean. Note the
  unit: Ariel's event is the *full* session before a market holiday — the half day is a
  different object, and the two are often conflated in retellings.
- **Kim, C.-W. & Park, J. (1994), "Holiday Effects and Stock Returns: Further Evidence",
  *Journal of Financial and Quantitative Analysis* 29(1), 145-157.** International evidence;
  the effect appears outside the US too, which cuts against a purely US-institutional story.
- **Vergin, R. C. & McGinnis, J. (1999), "Revisiting the Holiday Effect: Is It On Holiday?",
  *Applied Financial Economics* 9(5), 477-482.** The post-publication decay: the pre-holiday
  premium largely disappears for large firms after the effect became well known. The single
  most important reference for reading any modern calendar result.

## Why calendar findings deserve hostile treatment

- **Sullivan, R., Timmermann, A. & White, H. (2001), "Dangers of Data Mining: The Case of
  Calendar Effects in Stock Returns", *Journal of Econometrics* 105(1), 249-286.** The
  definitive warning: once the *universe of calendar rules searched* is accounted for with
  White's Reality Check, the classic calendar effects stop being significant. This study's
  45-cell multiplicity table is a small, explicit version of that discipline.
- **Harvey, C. R., Liu, Y. & Zhu, H. (2016), "... and the Cross-Section of Expected Returns",
  *Review of Financial Studies* 29(1), 5-68.** The *t* > 3 argument for a literature that has
  run thousands of tests. With ~3 events a year, no half-day study will ever reach it.
- **Newey, W. K. & West, K. D. (1987), "A Simple, Positive Semi-Definite, Heteroskedasticity
  and Autocorrelation Consistent Covariance Matrix", *Econometrica* 55(3), 703-708.** The
  standard errors used throughout (`quantlab.analytics.mean_tstat_hac`).
- **Politis, D. N. & Romano, J. P. (1994), "The Stationary Bootstrap", *JASA* 89(428),
  1303-1313.** The resampling philosophy behind the event bootstrap here.

## Thin markets and what they do to prices

- **Amihud, Y. (2002), "Illiquidity and Stock Returns", *Journal of Financial Markets* 5(1),
  31-56.** Illiquidity is priced; the half day is a natural experiment in temporary
  illiquidity, which is why volume — not return — is the variable this study can actually
  measure with confidence.
- **Barclay, M. J., Litzenberger, R. H. & Warner, J. B. (1990), "Private Information, Trading
  Volume, and Stock-Return Variances", *Review of Financial Studies* 3(2), 233-253.** Return
  variance per unit of *time* falls when the exchange is closed; a 2.5-hour session should
  therefore carry roughly the variance of 2.5 hours, not of a day — the prediction tested in
  the notebook's variance section.
- **French, K. R. & Roll, R. (1986), "Stock Return Variances: The Arrival of Information and
  the Reaction of Traders", *Journal of Financial Economics* 17(1), 5-26.** The classic
  trading-time versus calendar-time result; the 1968 Wednesday closures are the closest
  natural experiment to a half day in the literature.

## Neighbours on this desk

Studies **95-holiday-cheer**, **79-sleigh-ride**, **194-turkey**, **780-long-weekend-drift**,
**89-turn-of-the-month**, **290-september-effect** and **346-multiple-testing** are the
surrounding calendar work; **90-weekend** and **788-overnight-intraday-tug-of-war** carry the
overnight/intraday decomposition reused here.
