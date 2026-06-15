# References — Study 182 (Vortex-Indicator)

## Primary source

**Botes, E. & Siepman, D. (2010).** "The Vortex Indicator." *Technical Analysis of Stocks
& Commodities*, January 2010, pp. 20–25.  The originating paper: defines VI+/VI- from
vortex movement (|high−prior low| and |low−prior high|) normalised by true range, proposes
the 14-period rolling sum, and illustrates the crossover signal on commodity and equity
charts.  Freely available via the TASC subscriber archive.

## Related technical indicator literature

**Wilder, J.W. (1978).** *New Concepts in Technical Trading Systems*. Trend Research.
ISBN 978-0894590276.  Introduces the True Range, Average True Range, the Directional
Movement Index (+DI/−DI), and ADX — the direct conceptual ancestor of the Vortex
Indicator; VI+ and VI- are kinematically related to +DM and −DM but with different
normalisation.

**Kaufman, P.J. (2013).** *Trading Systems and Methods*, 5th ed. Wiley.  Chapter 23
surveys volatility-adjusted indicators including ATR-normalised momentum measures; Section
23.4 is directly relevant to interpreting VI+/VI-.

**Aronson, D.R. (2006).** *Evidence-Based Technical Analysis*. Wiley.  Chapters 6–10 lay
out the multiple-comparisons problem for technical rules: when many parameters are swept
(here, five hold horizons), the naive *p*-value understates the false-discovery rate by a
factor proportional to the number of tests.  Motivates the Bonferroni correction applied
in this study.

**Lo, A.W., Mamaysky, H. & Wang, J. (2000).** "Foundations of Technical Analysis:
Computational Algorithms, Statistical Inference, and Empirical Implementation." *Journal
of Finance* 55(4): 1705–1765.  Provides the methodological template for testing whether
technical patterns carry statistically significant information beyond a random-walk
baseline.

**Sullivan, R., Timmermann, A. & White, H. (1999).** "Data-Snooping, Technical Trading
Rule Performance, and the Bootstrap." *Journal of Finance* 54(5): 1647–1691.  The
canonical reference for why a rule that looks best from a sweep of many parameters
requires a multiple-comparisons correction; the White Reality Check underlies the
desk-wide inference bar.

## Related studies in this desk

- **Study 72 — Loaded-Dice:** SMA(5/10) crossover on 5-minute bars. Same crossover
  family on intraday data; verdict Signal=NONE, Tradability=MIRAGE.
- **Study 78 — Crossed-Wires:** MACD crossover on daily bars. Signal=NONE/WEAK.
- **Study 91 — Death-Cross:** 50/200-day SMA golden/death cross on SPY. A slower
  crossover cousin; verdict Signal=NONE on the gross level.
- **Study 106 — Supertrend:** ATR-band flip entries, daily; Signal=WEAK, Mirage.
  Supertrend uses the same ATR normalisation idea as the Vortex denominator (TR sum).
- **Study 127 — Williams-R:** Oversold/overbought oscillator, daily; Signal=WEAK, Mirage.
  Complements this study: Williams %R tests mean-reversion, Vortex tests trend-following.
