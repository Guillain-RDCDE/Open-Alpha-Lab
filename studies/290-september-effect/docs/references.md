# References & literature map — Study 290 (September-Effect)

## The claim under test

The "September effect" is the empirical regularity that September has the lowest
(often the only negative) long-run average return among calendar months for the
S&P 500 and many other equity indices. Unlike the Super Bowl indicator, it is
not obviously a data-snooping artefact — September weakness shows up across
decades and across countries — which is exactly what makes the "documented vs
tradable" distinction worth drawing carefully.

## Core literature

- **Bouman, S. & Jacobsen, B. (2002).** "The Halloween Indicator, 'Sell in May
  and Go Away': Another Puzzle." *American Economic Review*, 92(5), 1618–1635.
  The canonical study of the May–October seasonal of which the September effect
  is a part: returns from November–April systematically exceed May–October
  across 36 of 37 markets. September is the weak heart of the "bad" half-year.

- **Jacobsen, B. & Marquering, W. (2008).** "Is it the weather?" *Journal of
  Banking & Finance*, 32(4), 526–540, and the Kamstra–Kramer–Levi SAD line of
  work — proposed (and contested) behavioural channels for autumn weakness
  (seasonal-affective risk aversion, tax-loss and window-dressing flows around
  the September quarter-end).

- **Hirsch, Y. (annual).** *Stock Trader's Almanac.* The popular home of the
  "September is the worst month" / "Sell in May" folklore; the source most
  retail commentary cites each Labor Day.

- **Kamstra, M., Kramer, L. & Levi, M. (2003).** "Winter Blues: A SAD Stock
  Market Cycle." *American Economic Review*, 93(1), 324–343. A behavioural
  mechanism for a fall/winter return pattern — relevant to *why* September might
  be soft, while remaining controversial.

## Why "documented" is not "tradable"

- **The base-rate trap.** The S&P rises in ~63% of all months. "September is
  down" must be read against a market that is up most of the time; the right
  comparison is September vs the pooled other-month mean, not against zero.

- **Serial correlation and HAC.** Monthly equity returns are mildly
  autocorrelated, which inflates a naïve t-stat. The Newey-West HAC t-stat
  (Bartlett kernel, ⌊4(n/100)^{2/9}⌋ lags) is the honest test; on the 1950–2025
  S&P it gives **−1.93** for the September dummy — below the |t| ≥ 2 bar.

- **Small effective n.** One month a year means only 76 Septembers in 76 years.
  At ~3.7% monthly vol the minimum detectable gap (80% power) is ≈ 0.92pp/mo;
  the observed −0.79pp/mo gap sits right at the edge.

- **Non-persistence.** The S&P September drag is concentrated in 1986–2005 and
  is mildly *positive* before and after — consistent with a few clustered autumn
  crashes (1987, 2001, 2008) rather than a structural seasonal.

- **Multiple testing.** "Worst month" is selected ex-post from twelve candidates;
  finding *some* month with a sub-zero mean is nearly guaranteed by chance. In
  this window the worst month is actually October, not September.

- **Sullivan, Timmermann & White (1999).** "Data-Snooping, Technical Trading
  Rule Performance, and the Bootstrap." *Journal of Finance*, 54(5), 1647–1691.
  The standard treatment of how data-mining inflates apparent calendar effects;
  the appropriate benchmark accounts for the number of seasonals implicitly
  searched.

## Method lineage

- **Newey-West (HAC) t-test.** Heteroskedasticity- and autocorrelation-
  consistent standard error of the September-dummy coefficient — the headline
  significance test, robust to serial correlation in monthly returns.
- **Permutation test.** Shuffle which months carry the September label 10,000
  times; one-sided p = fraction of shuffles with mean ≤ the observed September
  mean. Assumes i.i.d. months, so it is *less* conservative than HAC.
- **Welch t-test.** `scipy.stats.ttest_ind(equal_var=False)` on September vs
  pooled other-month returns.
- **Power calculation.** Two-sample minimum detectable effect at 80% power,
  α = 0.05, using the September monthly volatility.

## Data sources

- **Shiller S&P 500 monthly dataset.** Robert Shiller's long-run US equity data,
  staged at `_cache/shiller_sp500.parquet`. We use month-over-month nominal
  **price** returns for 1950–2025 (912 months, 76 Septembers). Price-only — no
  dividends, consistent with the way the folklore is usually stated.
- **^GSPC (live fall-back).** `fetch_sp500_monthly(fetch=True)` lazily pulls
  ^GSPC monthly closes via yfinance and caches them; used only to refresh the
  tape, never in CI.

## Related desk studies

- The broader autumn / "Sell in May" / October-crash family (the seasonal whole
  of which September is one month) — see the desk's October-effect and
  Halloween-indicator teardowns.
- **Super-Bowl (Study 158)** — the same base-rate-trap and tiny-n machinery
  applied to a folklore indicator with no academic support, for contrast.
