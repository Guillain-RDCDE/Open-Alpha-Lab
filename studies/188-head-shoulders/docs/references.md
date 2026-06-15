# References & literature map — Study 188 (Head-Shoulders)

## The claim under test

- **The folk recipe.** Virtually every technical analysis textbook devotes a chapter to
  the head-and-shoulders (H&S) top: *"The most reliable reversal pattern in technical
  analysis. When price forms a left shoulder, a higher head, and a right shoulder of
  similar height — with a roughly horizontal neckline connecting the troughs — and then
  breaks below the neckline with conviction, a sustained decline follows, with a measured
  move equal to the distance from the head to the neckline."* The steelmanned version:
  **the confirmed neckline break carries directional information that exceeds the
  unconditional base-rate of next-day and next-week returns.**

## The seminal tests — and what they actually found

- **Lo, Mamaysky & Wang (2000)**, *Foundations of Technical Analysis: Computational
  Algorithms, Statistical Inference, and Empirical Implementation*, **Journal of Finance
  55(4)**, pp. 1705–1770. The canonical academic attempt to formalise chart-pattern
  detection using kernel-regression smoothing.  They find some statistically significant
  conditional return differences — but the effect sizes are small, the methodology
  criticised for data-snooping, and the patterns they detect are not the strict five-point
  structural definition this study uses.

- **Bulkowski (2005)**, *Encyclopedia of Chart Patterns* (2nd ed.), Wiley. The most
  comprehensive empirical survey of chart patterns by a practitioner.  Bulkowski finds
  H&S tops "break out downward 96% of the time" — but this is measured *after* confirming
  a break below the neckline, which by construction is a downward move.  His "failure
  rate" and "average decline" statistics do not control for the unconditional distribution
  of returns and are therefore not evidence of an excess edge.

- **Osler & Chang (1995)**, *Head and Shoulders: Not Just a Flaky Pattern*, Federal
  Reserve Bank of New York Staff Report No. 4. Finds H&S patterns are more common than
  would be expected by chance in foreign exchange markets, and that the directional
  prediction holds better in FX than in equities.  However, the sample period is short
  and the FX result has not been reliably replicated in major equity indices.

- **Jegadeesh (2000)**, Discussion of Lo, Mamaysky & Wang (2000), **Journal of Finance**
  55(4). Points out the multiple-comparisons problem inherent in testing many chart
  patterns on the same tape and the difficulty of achieving out-of-sample statistical
  power.

## Why the pattern looks reliable but isn't — selection and publication bias

- **Brock, Lakonishok & LeBaron (1992)**, *Simple Technical Trading Rules and the
  Stochastic Properties of Stock Returns*, **Journal of Finance 47(5)**. Found Dow
  theory rules with apparent predictive power — but Park & Irwin (2007) show this
  largely evaporates out of sample.  The same problem plagues chart patterns: the rules
  are identified on the same history that generated the pattern.

- **Sullivan, Timmermann & White (1999)**, *Data-Snooping, Technical Trading Rule
  Performance, and the Bootstrap*, **Journal of Finance 54(5)**. Applies a Reality Check
  bootstrap to a universe of technical rules and finds that most apparent out-performance
  disappears once the search across rules is accounted for.  H&S is one of thousands of
  candidate patterns — Bonferroni-style correction is the minimum honesty requirement.

- **Subjective recognition problem.** Human traders identify H&S patterns subjectively,
  with hindsight.  A strict algorithmic implementation (five-point structure, shoulder
  symmetry, confirmed neckline break) produces far fewer signals than subjective
  identification — our study finds only 3 H&S top confirmations per 10 tickers over 20
  years.  The literature's "H&S studies" often use loose definitions that fire
  substantially more often and therefore carry different (and harder to evaluate) evidence.

## The rarity problem — why inference fails

- **The n problem.** With 3 confirmed H&S tops across 200 ticker-years, the study is
  statistically underpowered by construction.  This is itself informative: if strict
  detection is necessary to avoid selection bias, then the pattern is too rare to be
  tradable.  If detection is loosened to increase n, selection bias re-enters.
  There is no free lunch between rarity and selection bias.

- **The measured move.** The traditional "measured-move target" (project the head-to-
  neckline distance below the break) is a price target, not a return expectation.  Our
  forward-return test bypasses this framing and asks the simpler question: does the
  direction after the break exceed random?  It does not, on our tape.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*, **Econometrica**
  55(3) — `strategy._hac_tstat` and `quantlab.analytics.mean_tstat_hac`.

- **Local-extrema detection.** Jones (1993), *Automated Identification of Chart Patterns
  in Financial Series*, and the `scipy.signal.find_peaks` implementation — our
  `strategy._local_peaks` / `strategy._local_troughs` wrappers.

- **Bonferroni correction.** Standard multiple-comparisons adjustment; see Hochberg &
  Tamhane (1987), *Multiple Comparison Procedures*, Wiley.  Here: 8 tests (2 pattern
  types × 4 horizons), threshold |t| ~ 2.58.

## Data sources used here

- **Yahoo! Finance daily bars** (via `yfinance`), adjusted close, 2005–2026.  Tickers:
  SPY, QQQ, AAPL, MSFT, AMZN, GOOGL, META, NVDA, TSLA, JPM.  All fingerprinted in
  `docs/results.md`.

## Related desk studies

- **[Study 76 — Rice-Paper](../../76-rice-paper/)**: five Japanese candlestick patterns
  on the same daily tape — the same random-baseline methodology, same conclusion (NONE).
- **[Study 72 — Loaded-Dice](../../72-loaded-dice/)**: the SMA(5/10) crossover — the
  same "famous pattern, honest test, no edge" story at 5-minute fidelity.
- **[Study 21 — Fools-Gold](../../21-fools-gold/)**: the 50/200 golden cross — the same
  "looks like a pattern, isn't reliable" family.
- **[Study 17 — Glass-Ceiling](../../17-glass-ceiling/)**: resistance breakouts — the
  closest structural cousin of the H&S neckline break.
