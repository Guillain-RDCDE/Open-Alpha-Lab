# References & literature map — Study 175 (Crypto-Weekend)

## The claim under test

- **The folk recipe.** Bitcoin trading forums, crypto-Twitter, and retail-broker educational
  content routinely state that BTC weekends are "different" — either systematically pumped by
  retail FOMO ("weekend pump") or systematically dumped by thin professional liquidity
  ("weekend dump"). The microstructure story: traditional banking rails (ACH, Fedwire) close on
  weekends, stablecoin minting slows, institutional desks reduce staffing, and retail activity
  dominates, supposedly creating a detectable directional anomaly and elevated volatility.
  We test the strongest version: *is there a statistically significant mean-return premium (either
  sign) and/or a volatility premium on BTC weekend days vs weekdays?*

## The empirical literature on crypto calendar effects

- **Caporale, Gil-Alana & Plastun (2019)**, *Searching for Inefficiencies in Exchange Rate Data*
  (Cogent Economics & Finance) — early work finding calendar anomalies in BTC returns, including
  a day-of-week effect. A cautionary tale in small samples and lack of multiple-comparisons
  correction; their window (2013–2016) contains fewer than 200 weekend days.
- **Aharon & Qadan (2019)**, *Bitcoin and the day-of-the-week effect* (Finance Research Letters) —
  find significant Monday returns in BTC (consistent with our finding that Monday has the highest
  raw mean at +0.42% / day) but attribute it to retail weekend news consumption materialising at
  the Monday open, not to a structural weekend-session anomaly.
- **Baur, Cahill, Godfrey & Liu (2019)**, *Bitcoin Time-of-Day, Day-of-Week and Month-of-Year
  Effects in Returns and Trading Volume* (Finance Research Letters) — document that BTC volume and
  volatility are *lower* on weekends, consistent with our Levene result (weekend daily vol = 51.6%
  annualised vs 72.2% on weekdays). They do not find a significant directional effect.
- **Kurihara & Fukushima (2017)**, *The market efficiency of Bitcoin: A weekly anomaly perspective*
  (Journal of Applied Finance & Banking) — find weak day-of-week effects in BTC but note power
  is extremely low in early windows.
- **Ante, Fiedler & Strehle (2021)**, *The influence of stablecoin issuances on cryptocurrency
  markets* (Finance Research Letters) — document that Tether (USDT) minting is lower on weekends
  and correlates with reduced buy-pressure, a plausible mechanism for a weekend *dump* rather than
  pump, but the empirical effect is economically small and fades post-2020.

## The microstructure mechanism — and why it should already be dead

- **Auer, Cornelli & Frost (2022)**, *Banking in the shadow of Bitcoin? The institutional adoption
  of cryptocurrencies* (BIS Working Paper No. 1013) — institutional adoption post-2020 implies
  professional market-makers are active 24/7, flattening weekend microstructure quirks.
- **Federal Reserve FedNow Service press release (2023-07-20)** — the July 2023 launch of
  24/7 instant settlement began eroding the "banking closed on weekends" friction for stablecoin
  rails. Our pre/post-2023 regime split finds no change in the (already absent) weekend premium,
  suggesting the mechanism was never strong enough to measure.
- **Silvergate Bank closure (March 2023)** and **Signature Bank failure (March 2023)** — the
  loss of 24/7 crypto-friendly banking rails briefly disrupted stablecoin settlement on weekends,
  which is the sub-sample we test as "post-2023". The effect remains absent.

## The inference tools used

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  [`strategy._hac_tstat`](../crypto_weekend/strategy.py) and
  [`quantlab.analytics.mean_tstat_hac`](../../../quantlab/analytics.py).
- **Multiple-comparisons (Bonferroni).** Three simultaneous tests (return mean, volatility,
  timing rule) → adjusted critical value |t| ≥ 2.39 (α = 0.05/3, two-sided). See also
  Harvey, Liu & Zhu (2016), *… and the Cross-Section of Expected Returns* (Review of Financial
  Studies) on the elevated discovery bar in multiple-testing financial research.
- **Levene-style variance test.** HAC t on the squared deviations — a heteroskedasticity-robust
  analogue of the classical F-test for equal variances; Brown & Forsythe (1974), *Robust Tests for
  the Equality of Variances* (JASQ).

## Related desk studies

- **[Study 133 — Crypto-Seasonality](../../133-crypto-seasonality/)** — the sister study testing
  monthly seasonality (Uptober, Rektember) in BTC; same data source, complementary calendar
  decomposition. Verdict: NONE / MIRAGE.
- **[Study 48 — Groundhog](../../48-groundhog/)** and
  **[Study 80 — Cold-Open](../../80-cold-open/)** — day-of-year and calendar-window claims in
  equities, tested with the same Bonferroni-corrected rigour and found NONE.
- **[Study 81 — Four-Year-Itch](../../81-four-year-itch/)** — small-n calendar anomaly (the Bitcoin
  halving cycle), with a teardown of the tiny-effective-n problem. The halving cycle's n=4 is even
  thinner than our weekend sub-samples.
- **[Study 136 — Mark-Twain](../../136-mark-twain/)** — the "sell in May" and "October Effect"
  seasonal lore in equities; structurally identical study design (calendar label + HAC + Bonferroni).
