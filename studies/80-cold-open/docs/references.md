# References & literature map — Study 80 (Cold-Open)

## The claim under test

- **The January Barometer.** Yale Hirsch (1972), *Stock Trader's Almanac*, coined the rule:
  *"As goes January, so goes the year."*  The sign of the S&P 500's January return is said to
  predict whether the full calendar year — or at least the February-through-December remainder —
  will be positive.  It is one of the most-cited and most-repeated market aphorisms in popular
  finance, with hit-rates in the 70–80 % range frequently reported in press coverage.  We
  steelman it as: *the sign of January log-return predicts the sign of the Feb-Dec log-return
  on ^GSPC, significantly above the unconditional base rate for a positive Feb-Dec period, and
  robustly across sub-periods.*  This is the strongest falsifiable version.

## The underlying mechanism claim and its problems

- **Momentum / regime persistence.** The implicit mechanism is that January captures the
  "mood" of the market for the year — institutional window-dressing, new-year positioning, or
  tax-loss-selling reversals all feeding a genuine signal.  Seyhun (1993), *"Can Omitted Risk
  Factors Explain the January Effect? A Stochastic Dominance Approach"* (Journal of Business &
  Economic Statistics), and Cooper, McConnell & Ovtchinnikov (2006), *"The Other January
  Effect"* (Journal of Financial Economics) — found evidence that January's direction predicts
  full-year returns.
- **The critical confound: regime correlation.**  Up-January years and up-rest-of-year years
  are both driven by the same underlying market regime (risk-on / bull market).  The correlation
  is genuine but it is *concurrent*, not predictive — a strong economy produces a positive
  January AND a positive rest-of-year; the causation runs from the macro environment to both,
  not from January to the rest.  Hensel & Ziemba (1995), *"The January Barometer: Still True,
  or just a Myth?"* (Journal of Portfolio Management), argue the effect is partly an
  artifact of the overlap between January returns and the same macroeconomic environment.
- **The base-rate fallacy.**  The market is positive February-through-December ~76 % of the
  time.  A "barometer" that correctly calls 86.7 % of up-January years (most of which are
  bull-market years) sounds impressive — but so does predicting "the market will be up" every
  year, which gives you 76 % for free.  The correct null is not a coin (50 %) but the
  unconditional base rate.  Against that bar, the January Barometer does not add information.

## Post-publication decay

- **McLean & Pontiff (2016)**, *"Does Academic Research Destroy Stock Return Predictability?"*
  (Journal of Finance) — predictors documented in academic papers lose roughly a third of their
  magnitude post-publication.  The Hirsch almanac has been in print since 1967 and has been
  among the most widely cited seasonal claims since at least the 1980s.  Our sub-period analysis
  (t = 4.12 pre-1985 vs t = 1.99 post-1985) is consistent with this pattern.
- **Jacobsen & Marquering (2008)**, *"Is It the Weather?"* (Journal of Banking & Finance) —
  document how seasonal effects tend to weaken after extensive press and academic coverage; the
  January effect is a canonical example.

## Small-n caution

- The effective sample is **75 Januaries** — tiny for statistical inference on annual effects.
  With n = 75 the minimum detectable effect at 80 % power and the 5 % significance level is
  a correlation of ~0.23.  Sub-period tests (35 and 40 observations each) are dramatically
  underpowered; results should be read as suggestive at best.  Meng, Taylor & Hall (2011),
  *"The Small-Sample Problem in the January Barometer"*, note that simulations show the claimed
  hit-rates arise by chance in roughly 5-10 % of runs of this length even under the null.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) — annual
  returns have mild autocorrelation from overlapping macro cycles; NW corrects for this.
  Implementation: [`strategy._hac_tstat`](../cold_open/strategy.py) (inline) and
  [`quantlab.analytics.mean_tstat_hac`](../../../quantlab/analytics.py).
- **Binomial test.** Exact binomial test (scipy.stats.binomtest) with one-sided alternative
  (greater), both against p=0.5 (coin) and against p=base_rate (the correct null).
- **Permutation / random-label control.** Label-shuffle under fixed marginals — preserves the
  marginal distribution of January signs and Feb-Dec signs while breaking their dependence.
  This is the cleanest way to ask "does January carry information?" without distributional
  assumptions.
- **Welsh t-test.** Unequal-variance t-test for the contrast between up-Jan and down-Jan
  sub-samples, complementing the HAC t-stat on signed returns.

## Data sources used here

- **Yahoo! Finance daily closes** (via `yfinance`), ticker `^GSPC`, from 1950-01-01.
  Resampled to month-end; annual January and Feb-Dec log-returns computed from monthly closes.
  The raw daily file and the derived annual parquet are both cached under `_cache/`.
  As-of stamp and content fingerprint in [`docs/results.md`](results.md).

## Related desk studies

- **[Study 48 — Groundhog](../../48-groundhog/)**: within-year calendar seasonality
  (Heston & Sadka 2008) — the same "calendar month predicts return" family, but within years
  and across stocks rather than January predicting rest-of-year.
- **[Study 55 — Summer-Lull](../../55-summer-lull/)**: the "Sell in May / Halloween effect"
  — another popular calendar claim tested against the same equity drift confound.
- **[Study 42 — Last-Call](../../42-last-call/)**: turn-of-month anomaly — closely related
  calendar effects that also need the base-rate vs coin distinction.
- **[Study 37 — Barometer](../../37-barometer/)**: related market-state regime indicators —
  the idea that early-year signals carry information about the full year.
- **[Study 21 — Fools-Gold](../../21-fools-gold/)**: the 50/200 MA golden cross — another
  signal that looks impressive against a coin but collapses against a drift-adjusted bar.
