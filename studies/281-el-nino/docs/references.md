# References & literature map -- Study 281 (El-Nino)

## The claim under test

The folklore: the **El Nino-Southern Oscillation (ENSO)** -- a warming (El Nino) or
cooling (La Nina) of the equatorial Pacific -- shifts global weather enough to move
commodity prices and even equities. "El Nino summers are bullish," droughts spike
crop and energy prices, and because NOAA forecasts the phase months ahead, you can
supposedly position for it. We test whether the ENSO phase (NOAA's Oceanic Nino
Index) predicts S&P 500 and crude-oil calendar-year returns, on a one-year execution
lag so the trade is actually actionable.

## The weather mechanism is genuine -- the price effect is the question

- **Cashin, P., Mohaddes, K. & Raissi, M. (2017).** "Fair weather or foul? The
  macroeconomic effects of El Nino." *Journal of International Economics*, 106, 37-54.
  A careful GVAR study finding real, heterogeneous El Nino effects: it raises
  inflation and energy/commodity prices in many economies while modestly lifting US
  and some others. The effect on the *real economy and spot commodities* is real --
  which is exactly why a tradable equity edge should be arbitraged away.

- **Brunner, A. D. (2002).** "El Nino and World Primary Commodity Prices: Warm Water
  or Hot Air?" *Review of Economics and Statistics*, 84(1), 176-183. Finds ENSO
  explains roughly a fifth of the variance in real commodity-price inflation and a
  smaller share of CPI -- a real but bounded macro effect, far smaller than the
  folklore implies for asset returns.

- **Ubilava, D. (2018).** "The role of El Nino Southern Oscillation in commodity
  price movement and predictability." *American Journal of Agricultural Economics*,
  100(1), 239-263. Documents nonlinear, regime-dependent ENSO effects on
  agricultural commodity prices -- again, in spot/physical markets, not necessarily
  in tradable, forward-looking ETF returns.

## Why a real weather effect need not be a tradable price signal

- **The base-rate / drift trap.** Equities drift up ~7%/yr and rise in ~73% of
  calendar years unconditionally. An "El-Nino-is-bullish" rule inherits that drift
  for free; the correct null is the *unconditional* mean, and the correct test is
  whether El Nino years beat La Nina years by more than sampling noise (a Welch t
  or permutation test), not whether El Nino years are merely positive.

- **Forecastability kills the edge.** NOAA's Climate Prediction Center publishes
  ENSO probabilities openly, months in advance. If a forecastable ocean state
  predicted next-year equity returns, the edge would be competed away. This is the
  efficient-markets reason to expect the equity signal to be null even when the
  weather effect is real.

- **Tiny n.** ~75 ENSO years since 1950, split into ~25 El Nino / ~25 La Nina /
  ~25 Neutral. With ~17% equity volatility the minimum detectable El-La mean gap at
  80% power is ~7%/yr -- larger than any plausible weather premium. The oil tape
  (USO, 2006+) has only ~19 years, hopelessly underpowered.

- **Spot vs tradable.** A spike in physical crop or crude prices does not map
  cleanly to a tradable ETF return: roll yield, storage, and the fact that forecast
  ENSO news is already in the curve all dilute the effect. We test the tradable USO
  ETF and find the gap is small and *wrong-signed*.

## Related desk studies (same shape: real-world curiosity, null on the tape)

- **[Study 158 -- Super-Bowl](../../158-super-bowl/)**: a famous sports indicator
  that evaporates against the honest base rate -- the canonical sibling teardown.
- **[Study 223 -- Same-Month-Seasonality](../../223-same-month-seasonality/)**: the
  synthetic-panel + cached-real-series pattern this study mirrors.

## Method lineage

- **Welch t-test.** `scipy.stats.ttest_ind(equal_var=False)` on per-year returns
  for El Nino vs La Nina (unequal group sizes, unequal variances).
- **One-way ANOVA.** `scipy.stats.f_oneway` across the three phases -- the omnibus
  test for "does phase matter at all".
- **Permutation test.** Shuffle the phase labels 10,000 times; record the empirical
  distribution of `|mean(El Nino) - mean(La Nina)|`; the p-value is the fraction of
  shuffles whose gap equals or exceeds the observed one.
- **Newey-West HAC t.** A heteroskedasticity-and-autocorrelation-consistent t-stat
  (lag 1) on the long-short (El Nino long, La Nina short) annual return series -- the
  desk's bar for a REAL signal is a robust HAC |t| >= 2.
- **Bonferroni.** Multiply p-values by the number of tapes tested (equity, oil,
  lag-0 robustness) -- a nod to the multiple-comparisons problem.

## Data sources

- **NOAA Climate Prediction Center, ONI v5.** Oceanic Nino Index, the official
  3-month-running-mean SST anomaly in the Nino-3.4 region. We hardcode the NDJ
  (Nov-Dec-Jan) seasonal value for each winter 1950-2024 in `data.py`. Source:
  https://origin.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php
- **^GSPC daily close** (Yahoo, 1927+; we use 1950+), repo-level
  `_cache/^GSPC_split_only.parquet`. December/December calendar-year price returns.
- **USO (United States Oil Fund)**, repo-level `_cache/USO_split_only.parquet`
  (2006+), a tradable crude-oil proxy for the commodity-channel test. Price return
  only; roll/storage drag is real and named in the teardown.

## Honesty caveats

- **Price-only, not total return** on both tapes (no dividends/roll-adjusted total
  return) -- consistent with a directional-signal test; named here so it is not
  silently assumed.
- **One-year execution lag** is enforced in the join (NDJ phase of year Y -> return
  year Y+1) so there is no look-ahead; lag-0 is shown only as a degrees-of-freedom
  illustration.
- **Survivorship** is not a single-name concern for the index tape (^GSPC is the
  index), but the USO tape is one fund post-2006 and the result is a small-sample
  upper bound, not a live edge.
