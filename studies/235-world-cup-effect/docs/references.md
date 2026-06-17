# References & literature map — Study 235 (World-Cup-Effect)

## The claim under test

**Edmans, A., Garcia, D. & Norli, O. (2007).** "Sports Sentiment and Stock
Returns." *The Journal of Finance*, 62(4), 1967–1998.
The foundational paper. Edmans et al. test football (soccer) World Cup
elimination on **next-day returns in the losing country**, using a sample
of 39 countries and 1,162 elimination matches (1974–2004). They find a
statistically significant negative return (−0.38% next-day abnormal return)
following World Cup eliminations, consistent with investor mood affecting
asset prices. Crucially, the paper tests *country-specific* effects on
*elimination days*, not the global S&P 500 during the full tournament.
Extrapolating to the S&P 500 across the full tournament window — the
"World Cup effect" as commonly cited in financial media — is a much weaker
and untested claim.

## Extensions and replications

**Berument, M. H., Ceylan, N. B. & Ogut-Eker, G. (2009).** "Soccer, stock
returns and fanaticism: Evidence from Turkey." *The Social Science Journal*,
46(3), 794–801.
Country-specific study extending Edmans et al. to Turkey; finds similar
negative same-day returns following elimination.

**Saraç, M. & Zeren, F. (2013).** "The Effect of Soccer Performance on
Stock Return: New Evidence from Quantile Cointegration Approach." *International
Journal of Finance & Economics*, 18(4), 389–399.
Broader cross-country replication; effect varies substantially by country and
is not stable over time.

**Gerlach, J. R. (2011).** "International Sports and Investor Sentiment: Does
It Pay to Watch the Game?" *Applied Financial Economics*, 21(21), 1601–1610.
Tests sports sentiment across multiple sports; finds effects are largely
concentrated in small, illiquid markets where retail sentiment can more easily
move prices. The S&P 500 — the world's most liquid market — is the hardest
case for a sentiment channel.

## Why the S&P 500 global version is expected to be weak

**Hirshleifer, D. & Shumway, T. (2003).** "Good Day Sunshine: Stock Returns
and the Weather." *The Journal of Finance*, 58(3), 1009–1032.
The same mood/sentiment mechanism as Edmans et al., applied to weather.
Effect is small, hard to trade, and likely smaller in the world's most
institutional market.

**Kamstra, M. J., Kramer, L. A. & Levi, M. D. (2003).** "Winter Blues: A
SAD (Seasonal Affective Disorder) Study of Stock Market Returns." *American
Economic Review*, 93(1), 324–343.
Seasonal mood effects; similar mechanism. The effect on US equities, while
statistically detectable, is small and not robust to transaction costs.

**Harvey, C. R., Liu, Y. & Zhu, H. (2016).** "... and the Cross-Section of
Expected Returns." *Review of Financial Studies*, 29(1), 5–68.
Documents that hundreds of "significant" predictors are spurious; the
appropriate t-stat hurdle for a new anomaly is ~3.0, not 2.0. Our per-edition
t = -2.52 sits below this elevated bar.

## The confound problem

The most negative World Cup windows overlap with documented macro crises:
- **1950 (−88 bps/day):** The Korean War started June 25, 1950 — one day after
  the World Cup opened. The S&P 500 dropped ~15% in six weeks entirely due to
  the war, not football attention.
- **1974 (−59 bps/day):** The 1973–74 oil crisis and bear market. The WC ran
  June 13 – July 7, 1974, during a steep recession.
- **2002 (−34 bps/day):** The dot-com bust and Enron/WorldCom accounting
  scandals were at their nadir in summer 2002.

These are not sentiment effects from World Cup football — they are coincidences
between the tournament calendar and known macro downturns. This confounding
makes the S&P 500 "World Cup effect" essentially uninterpretable as a pure
sentiment signal.

## Method lineage

- **Welch t-test.** `scipy.stats.ttest_ind(equal_var=False)` for comparing
  WC-window daily returns vs control windows.
- **One-sample t-test.** `scipy.stats.ttest_1samp` comparing per-edition mean
  returns to the unconditional daily mean — the correct inference unit is the
  19 independent tournaments, not the ~340 pooled daily observations.
- **Permutation test.** Shuffle WC-day labels 10,000 times; record the
  empirical distribution of WC-minus-non-WC mean returns.
- **Binomial test.** `scipy.stats.binomtest` to test whether more than 50% of
  the 19 WC windows have negative mean returns.
- **Matched control windows.** Same-duration window 2 years prior to each WC,
  same calendar month, to control for seasonal effects.

## Data sources

- **S&P 500 daily returns.** Yahoo Finance via `yfinance` (^GSPC), adjusted
  close, 1950–2023. Cached at `studies/235-world-cup-effect/_cache/sp500_daily.parquet`.
- **World Cup windows.** Hardcoded in `data.py`. Sources: FIFA official records
  at https://www.fifa.com/tournaments/mens/worldcup, Wikipedia "FIFA World Cup",
  and supplementary tables in Edmans et al. (2007).

## Related desk studies

- **[Study 158 — Super-Bowl](../../158-super-bowl/)**: Same analytical framework —
  annual sporting event, hardcoded event table, Shiller S&P data. A direct
  methodological sibling.
- **[Study 165 — Rosh-Hashanah](../../165-rosh-hashanah/)**: Religious calendar
  anomaly — same family of sentiment/attention calendar effects.
- **[Study 164 — Mercury-Retrograde](../../164-mercury-retrograde/)**: Celestial
  calendar anomaly — the purest example of a confounded, spurious correlation
  in this desk's collection.
