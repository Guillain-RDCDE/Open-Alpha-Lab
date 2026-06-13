# References & literature map - Study 95 (Holiday-Cheer)

## The claim under test

The **pre-holiday effect**: stocks reliably drift up on the trading day *before* a market
holiday, earning a return several times the normal daily average. The strong, sold-at-
full-strength version is a tradable rule - *"buy the day before every holiday."* It is one
of the oldest and most-cited calendar anomalies in the academic record.

- Ariel, R. A. (1990), *"High Stock Returns before Holidays: Existence and Evidence on
  Possible Causes,"* **Journal of Finance** 45(5), 1611-1626. The canonical paper: the
  pre-holiday day earned 9-14x the mean return over 1963-1982; ~1/3 to 1/2 of all market
  gains in the sample fell on the eight pre-holiday days a year.
- Lakonishok, J. & Smidt, S. (1988), *"Are Seasonal Anomalies Real? A Ninety-Year
  Perspective,"* **Review of Financial Studies** 1(4), 403-425. Documents the pre-holiday
  effect (among others) on 90 years of the Dow and stresses the data-snooping problem for
  calendar anomalies.
- Fields, M. J. (1934), *"Security Prices and Stock Exchange Holidays in Relation to Short
  Selling,"* **Journal of Business** - one of the earliest notes that pre-holiday returns
  are unusually high.

## Why the steelman is almost coherent

- The pattern is **real and large in the older data** - multiple independent studies on
  different indices and decades find the same pre-holiday premium, so it is not a single
  data-mined fluke.
- Plausible micro-structure / behavioural stories exist: pre-holiday sessions are thin and
  often short, short-sellers were thought to close positions before a non-trading stretch
  (Fields 1934), and a "holiday mood" / inventory-management bid has been proposed.

## Why it is likely to fail *as stated* ("buy the day before every holiday")

- **The documented post-1990 decay.** After Ariel (1990) the effect shrank markedly - a
  recurring finding that calendar anomalies attenuate once published and arbitraged.
  Surveys and replications (e.g. Schwert, *Anomalies and Market Efficiency*, Handbook of
  the Economics of Finance, 2003) note exactly this fade for seasonal effects.
- **Capacity is the killer.** There are only ~8-9 pre-holiday days a year. Even a large
  per-day edge captures too few days to build a competitive standalone return; a book that
  sits in cash 96% of the time cannot keep up with buy-and-hold.
- **Price-only vs total-return.** The long sample available is a price index (^GSPC, no
  dividends). The magnitude of any "edge" must be read against the right benchmark and
  labelled price-only - which this study does.

## Method lineage

- **Holiday-from-gaps derivation.** Rather than load an external holiday calendar, we read
  the market's own trading index: a weekday absent from the trading dates (and not a
  weekend) is a market holiday, and the prior trading day is "pre-holiday." This needs no
  third-party calendar, cannot drift out of date, and is exactly the set the folklore
  concerns. (It also captures rare unscheduled closes - which genuinely are days the market
  was shut.)
- **Newey-West HAC standard errors** for the mean of an autocorrelated return series:
  Newey, W. K. & West, K. D. (1987), *"A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix,"* **Econometrica**
  55(3), 703-708.
- **Wilson score interval** for the win-rate proportions: Wilson, E. B. (1927), *"Probable
  Inference, the Law of Succession, and Statistical Inference,"* JASA 22, 209-212.
- **Circular block bootstrap** for the test of the *difference* of gaps across the
  pre/post-1990 split, preserving the short-horizon autocorrelation of daily returns.

## Data sources used

- **SPY**, daily, **total-return adjusted** via `quantlab.data` (Yahoo Finance), cached to
  parquet - the fair, dividend-inclusive tape for the tradability question, from 1993.
- **^GSPC**, daily, **price-only** (split-adjusted, no dividends) via `quantlab.data`,
  back to 1950 - the long sample the pre/post-1990 decay test requires. Labelled
  price-only wherever quoted.

## Related desk studies

- [Study 91 - Death-Cross](../../91-death-cross/) - the "real risk reduction vs beats the
  market" split, and the desk's HAC + matched-control pattern.
- Other calendar-anomaly teardowns on the bench (turn-of-month, sell-in-May) share the
  capacity-and-decay verdict pattern this study lands on.
