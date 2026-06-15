# References & literature map — Study 167 (Hindenburg-Omen)

## The claim under test

The Hindenburg Omen was developed by Jim Miekka (c. 1995) and popularised by market commentators
in the early 2000s.  The recipe: when, on the same trading day, the fraction of NYSE issues making
new 52-week highs *and* new 52-week lows both exceed ~2.2% of total issues, with the NYSE
Composite above its 50-day moving average and the McClellan Oscillator negative, a stock-market
crash is "imminent" within 40 trading days.  The underlying intuition is that a bifurcated market
— strong internals coexisting with weak ones — is internally incoherent and historically precedes
corrections.  The claim is explicit in practitioners' notes as: *"every stock market crash since
1987 was preceded by a Hindenburg Omen."*

## Why the claim is almost coherent

- **McClellan, S. & McClellan, T.** (1970, and ongoing at mcoscillator.com) — the McClellan
  Oscillator (advances minus declines, exponentially smoothed) is a well-established breadth tool.
  A negative reading does carry mild near-term bearish information in some sub-periods, though not
  at the magnitude the Omen literature implies.
- **Breadth divergences as a warning sign.**  DeStefano, M. (2004), *Stock Returns and the Business
  Cycle* (Financial Review, 39(4)), documents that breadth indicators carry information about
  future market conditions — but at the aggregate level and over multi-month horizons, not the 40
  trading-day window the Omen specifies.
- **New highs and new lows together.**  The logical core — simultaneous elevated highs and lows
  implies a split market — is grounded in the advance/decline literature.  Zweig, M. (1986),
  *Winning on Wall Street* (Warner Books), uses breadth bifurcation as one of several warning
  signals, though never as a single crash predictor.

## The critique — a false-alarm machine

- **Hulbert, M.** (2010, 2013 — MarketWatch).  The most-cited practitioner takedown: tracks every
  Hindenburg Omen signal from 1985 and finds that a crash (≥5% loss) in the next 40 days followed
  only ~25% of signals.  Because the market falls 5% within 40 days roughly 40% of the time
  unconditionally, the omen is *anti-predictive* on this measure.
- **Steenbarger, B.** (2010 — TraderFeed blog).  Examines the October 2010 cluster of signals that
  generated enormous press coverage; the expected crash did not materialise.  Argues the signal is
  a coincidence detector in a low-base-rate environment: a 1-in-10 event can still "predict"
  crashes if crashes happen often enough anyway.
- **Birinyi Associates** (various years).  Tracks the omen in real time and repeatedly finds it
  fires during bear-market *recoveries* (many simultaneous highs and lows) as often as during
  pre-crash environments, undermining the directional interpretation.

## Multiple-comparisons and small-n context

- **Harvey, C., Liu, Y. & Zhu, H.** (2016), *… and the Cross-Section of Expected Returns*
  (Review of Financial Studies, 29(1)).  The case for aggressive Bonferroni / FDR correction
  on stock-market predictors: with 12 parameter combinations (4 horizons × 3 thresholds) tested
  here, the Bonferroni threshold for a 5% family-wise error rate is p < 0.0042 per test.  All
  our raw p-values exceed 0.50.
- **McLean, R.D. & Pontiff, J.** (2016), *Does Academic Research Destroy Stock Return
  Predictability?* (Journal of Finance, 71(1)).  Once a signal is published and widely known,
  any residual edge tends to decay — but for the Hindenburg Omen the decay is irrelevant because
  the effect was never statistically present to begin with (n ≈ 30 clusters in 20 years).
- **Lo, A.W.** (2002), *The Statistics of Sharpe Ratios* (Financial Analysts Journal, 58(4)).
  With n ≈ 30 independent observations, even a Sharpe of 0.5 on the signal arm carries a standard
  error of ~0.19, making it indistinguishable from zero.

## Breadth & market-timing context

- **Zweig, M.** (1986), *Winning on Wall Street* (Warner Books) — the breadth-thrust / advance-
  decline literature that the Omen loosely draws on.
- **Fosback, N.** (1976), *Stock Market Logic* (The Institute for Econometric Research) — early
  compilation of breadth indicators; most carry far less forecasting power at short horizons than
  claimed.
- **Brown, S.J., Goetzmann, W.N. & Kumar, A.** (1998), *The Dow Theory: William Peter Hamilton's
  Track Record Reconsidered* (Journal of Finance, 53(4)).  Technical-analysis rules can show
  in-sample fit that disappears out-of-sample; the Hindenburg Omen is a textbook example.

## Survivorship bias in constituent panels

- **Shumway, T. & Warther, V.** (1999), *The Delisting Bias in CRSP's Nasdaq Data and Its
  Implications for the Size Effect* (Journal of Finance, 54(6)).  Delisted / removed stocks
  are disproportionately weak performers; excluding them systematically understates the fraction
  of new 52-week lows during crashes.  Our panel uses current S&P 500 membership
  (``allow_survivorship_bias=True``), introducing this bias; it works *against* the signal (fewer
  lows measured → fewer Hindenburg days during crashes → if anything biases toward a finding).

## Method lineage (the desk's shared engine)

- **Newey, W. & West, K.** (1987), *A Simple, Positive Semi-Definite, Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix* (Econometrica, 55(3)) — the HAC t-stat in
  `strategy._hac_tstat` and `quantlab.analytics.mean_tstat_hac`.
- **Welch, B.L.** (1947), *The Generalization of Student's Problem When Several Different
  Population Variances are Involved* (Biometrika, 34(1)) — the Welch two-sample t-test used
  for signal-vs-base and crash-rate comparisons.
- **Bonferroni, C.E.** (1936) / **Holm, S.** (1979) — the family-wise error rate correction
  applied to the 12-hypothesis grid (4 horizons × 3 thresholds).

## Related desk studies

- **[Study 80 — Cold-Open](../../80-cold-open/)**: first-day-of-month breadth effect — breadth
  indicators at a known seasonal window rather than a conditional signal.
- **[Study 81 — Four-Year-Itch](../../81-four-year-itch/)**: another calendar/predictive rule with
  the same structural small-n problem (~24 non-overlapping observations).
- **[Study 83 — Half-Life](../../83-half-life/)**: a rigorous teardown of a signal with an even
  smaller effective n — the same small-sample reckoning applied here.
- **[Study 76 — Rice-Paper](../../76-rice-paper/)**: Bonferroni/permutation correction when testing
  many sub-hypotheses — the multiple-comparisons discipline this study applies.
