# References — Study 166 (First-Five-Days)

## Primary claim

**Hirsch, Yale** (1972). *Stock Trader's Almanac*. Old Farmer's Almanac / Hirsch Organization.
The original source of the "As goes January, so goes the year" family of calendar indicators.
The First-Five-Days Early Warning System is a compressed version: if the S&P 500 is up over the
first five trading days of January, the year is supposed to be good.  Hirsch updated the
scorecard annually in subsequent editions.

**Hirsch, Jeffrey A. & Hirsch, Yale** (2010). *Stock Trader's Almanac 2010* (and subsequent
annual editions).  The FFD indicator is presented as having ~80% accuracy going back to 1950,
a claim that relies on the coin-null rather than the base-rate-adjusted null.

## Academic literature on the January Barometer family

**Cooper, Michael J., John J. McConnell, and Alexei V. Ovtchinnikov** (2006).
"The Other January Effect."  *Journal of Financial Economics* 82(2): 315–341.
Shows that the *level* of January's return (not just the sign) predicts the rest-of-year
return, and that the effect is stronger than the sign-only version.  One of the more serious
academic treatments of the Hirsch effect.

**Hensel, Chris R., and William T. Ziemba** (1995). "The January Barometer: What Is It and How
Does It Work?"  *Journal of Portfolio Management* 21(2): 20–28.  Documents the empirical
regularity and discusses its potential sources, including regime effects and seasonal patterns.

**Brown, Philip, Donald B. Keim, Allan W. Kleidon, and Terry A. Marsh** (1983).
"Stock Return Seasonalities and the Tax-Loss Selling Hypothesis."  *Journal of Financial
Economics* 12(1): 105–127.  Context on January anomalies more broadly; helpful for
understanding the regime-correlation mechanism behind the barometer family.

**Thaler, Richard H., and Werner F. M. De Bondt** (1985). "Does the Stock Market Overreact?"
*Journal of Finance* 40(3): 793–805.  Background on the behavioural/seasonal literature
that the January Barometer is embedded in.

## Closely related studies in this repo

**Study 80 — Cold-Open:** The full-January Barometer (sign of January → sign of Feb-Dec).
A larger sample with no mechanical overlap confound.  Verdict: MIXED (real contrast, but fails
base-rate test and fades post-1985).  The FFD effect is weaker than Cold-Open on the
directional test because the first-five-days have less information than the whole month.

**Study 48 — Groundhog:** Another beloved seasonal indicator (groundhog shadow → market direction).
Zero overlap with the period being predicted; much smaller effective n; cleaner causal logic.

## Inference methodology

**Newey, Whitney K., and Kenneth D. West** (1987). "A Simple, Positive Semi-Definite,
Heteroskedasticity and Autocorrelation Consistent Covariance Matrix."  *Econometrica* 55(3):
703–708.  The HAC t-statistic used throughout.

**Romano, Joseph P., and Michael Wolf** (2005). "Stepwise Multiple Testing as Formalized Data
Snooping."  *Econometrica* 73(4): 1237–1282.  Background on why permutation tests are the
right tool for data-snooped calendar rules.

## Why the signal is NONE despite a large t-stat

The full-year HAC t = 4.87 looks compelling.  Three confounds each account for part of it:

1. **Mechanical overlap.** The F5D return is inside the full-year return.  Even a completely
   random market would show some correlation between the sign of an embedded sub-period and
   the full period's sign.  The Cold-Open study correctly separates the signal period (January)
   from the outcome period (Feb-Dec) to eliminate this.

2. **Regime correlation.** Up-F5D years concentrate in bull-market regimes; down-F5D years
   concentrate in weak regimes.  Both the F5D return and the Dec year-end return load on the
   same latent regime variable.  This is the "mirror of the equity premium" argument.

3. **Small n and post-publication decay.** n = 76 is tiny.  The hit-rate drops from 81% pre-1985
   to 60% post-1985, consistent with either noise in the discovery sample or gradual arbitrage.
   Post-1985, the directional call is not significantly different from a coin (p = 0.13).
