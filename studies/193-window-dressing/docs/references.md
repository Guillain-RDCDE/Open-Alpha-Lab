# References & literature map — Study 193 (Window-Dressing)

## The claim under test

**Carhart, Kaniel, Musto & Reed (2002).** *Leaning for the Tape: Evidence of Gaming
Behavior in Equity Mutual Funds*, Journal of Finance 57(2), 661–693. The foundational
paper. Using quarterly holdings data on US equity mutual funds (1985–1994), they document
that fund managers inflate the prices of their existing holdings in the last few days of
each calendar quarter — especially the stocks already held at large weights — and that
these price inflations systematically reverse in the first few days of the next quarter.
The mechanism: funds submit aggressive buy orders into quarter-end to paint a flattering
performance picture ("leaning for the tape"), then the demand evaporates and prices revert.

## The mechanism and related evidence

- **Haugen & Lakonishok (1988).** *The Incredible January Effect*, Dow Jones-Irwin. Early
  documentation of return patterns around calendar boundaries; the January effect has since
  been attributed partly to window-dressing-like behaviour (selling losers in December,
  rotating in January). The January pattern has largely decayed since the 1990s.

- **Musto (1997).** *Portfolio Disclosures and Year-End Price Shifts*, Journal of Finance
  52(4), 1563–1588. Shows that stocks held more commonly in funds experience positive
  abnormal returns at year-end and negative abnormal returns at the turn of the year — a
  disclosure-driven price cycle. Predecessor to Carhart et al.

- **Gallagher, Gardner & Swan (2009).** *Portfolio Pumping: An Examination of Investment
  Manager Quarter-End Trading and Impact on Performance*, Pacific-Basin Finance Journal
  17(1), 1–27. Replicates the Carhart finding using Australian equity fund data, showing
  that portfolio pumping is not limited to the US.

- **Bhattacharyya & Nanda (2013).** *Portfolio Pumping, Trading Activity and Fund
  Performance*, Review of Finance 17(3), 885–919. Documents the persistence of pumping
  behaviour among US domestic equity funds and its correlation with career concerns.

## Why the effect has likely decayed or was never broad-based

- **Shleifer & Vishny (1997).** *The Limits of Arbitrage*, Journal of Finance 52(1),
  35–55. Quarter-end calendar trades are extremely well-known; arbitrageurs who front-run
  the seasonal demand would eliminate the premium over time.

- **Sias & Starks (1997).** *Institutions and Individuals at the Turn of the Year*,
  Journal of Finance 52(4), 1543–1562. Shows that institutional-driven January return
  patterns became less exploitable once they became widely documented.

- **Market structure change.** Electronic limit-order books, faster price discovery, and
  the proliferation of ETFs (SPY launched 1993) make it harder to move prices at
  quarter-end without immediate and large arbitrage pressure. The Carhart 1985–1994 sample
  predates these structural shifts.

## Related desk studies

- **[Study 82 — Witching-Hour](../../82-witching-hour/)**: quarterly triple-witching
  expiration effects — same calendar boundary, options/futures channel rather than fund
  flows.
- **[Study 48 — Groundhog](../../48-groundhog/)**: a calendar-superstition study; the
  same "is the calendar label informative?" protocol used here.
- **[Study 136 — Mark-Twain](../../136-mark-twain/)**: the October seasonality claim —
  calendar months, same inference approach.
- **[Study 163 — Friday-13th](../../163-friday-13th/)**: superstition-driven calendar
  effect, same Welch + Bonferroni protocol.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica).
  Used in `strategy._hac_tstat`.
- **Welch t-test (unequal variances).** Welch (1947), *The Generalization of 'Student's'
  Problem When Several Different Population Variances are Involved*, Biometrika 34(1/2).
- **Bonferroni correction.** Bonferroni (1935); applied here for the window-size sweep
  (10 tests) to prevent data-snooping artefacts from an optimised window choice.
- **Data source.** SPY total-return daily prices from the shared repo cache
  (`_cache/SPY_total_return.parquet`, 1993–2026), via `yfinance`.
