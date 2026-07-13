# References & literature map — Study 762 (Vegas-Gaming-Win)

## The claim under test

- **Strip GGR as the sector's fundamental pulse.** The trading folklore, repeated across
  gaming-sector sell-side notes and financial-media "casino watch" segments, is that Las
  Vegas Strip **gross gaming revenue** (GGR) is *the* top-down read on the casino business —
  so when the monthly Strip GGR run-rate **accelerates**, the casino operators (MGM, Caesars,
  Las Vegas Sands, Wynn, Boyd, Penn) are about to run. We test the strongest form: *does a
  Strip-GGR momentum uptick lead the casino equities cleanly enough to trade?*
- **The data series.** Nevada Gaming Control Board, *Nevada Gaming Revenue Report* — the
  monthly "**Las Vegas Strip**" gross-gaming-revenue line (the Board's Tax & License Division
  publishes it ~5 weeks after each reference month). The University of Nevada, Las Vegas
  **Center for Gaming Research** maintains the widely-cited historical Strip GGR tables built
  from these filings. Our monthly tape is a **hardcoded approximate reconstruction** whose
  annual sums match the published Strip totals (the machine-readable NGCB PDFs are not
  fetchable in this build).

## Why the GGR tape isn't fetched live here — and what we do

- **NGCB filings are PDFs.** The Board's monthly revenue reports are released as PDF tables,
  not a CSV/JSON endpoint, and are not reachable from this build's network sandbox. Following
  the desk convention for small, public, well-documented series that resist live fetching —
  **Study 385 (Jobless-Claims)** hardcodes a snapshot of FRED `IC4WSA`, **Study 358
  (Watch-Index)** and **Study 708 (Eurovision-Effect)** hardcode a labelled proxy series — we
  hardcode a **monthly reconstruction** of the Strip GGR line, calibrated so its annual sums
  track the published totals, with the 2020 COVID closure represented faithfully. It is a
  *labelled reconstruction*, not the settled month-by-month print; the caveat is named on the
  Signal axis.
- **Equities.** Casino-operator daily adjusted closes via **yfinance** (no key), month-end
  sampled, equal-weighted into a basket, total-return adjusted — labelled as such.

## Why "leading" is the crux — markets discount public data

- **Semi-strong efficiency.** Fama (1970), *Efficient Capital Markets: A Review of Theory and
  Empirical Work* (Journal of Finance) — under semi-strong efficiency, publicly-released
  fundamentals are impounded into prices quickly; a widely-watched, lagged monthly revenue
  print is the archetype of information the liquid equities have already discounted. A GGR
  report five weeks stale cannot lead a stock that trades on next-quarter expectations.
- **Prices lead fundamentals.** The equity market is itself a leading indicator (a stock-price
  aggregate is a Conference Board LEI component); casino share prices move on forward booking
  trends, room rates and macro before the backward-looking GGR tally is tabulated. So a GGR
  series that co-moves with the sector can *lag* the equities that lead it. We isolate this
  with an explicit **lead/lag cross-correlation**.
- **Predictive regressions and small-sample caution.** Welch & Goyal (2008), *A Comprehensive
  Look at the Empirical Performance of Equity Premium Prediction* (Review of Financial
  Studies) — most fundamentals that look predictive in-sample fail out-of-sample; the bar for
  a tradable single-series signal is high, and Goyal–Welch's caution applies directly to a
  celebrated "leading" sector gauge.

## Why the inference is small-sample / placebo-based

- **Welch two-sample t.** Welch (1947), *The generalization of "Student's" problem when
  several different population variances are involved* (Biometrika) — unequal-variance test of
  the RISING-set forward mean against the unconditional mean.
- **Randomization / placebo null.** Because regime months are autocorrelated and the effective
  sample is small, we resample random same-size month sets and ask how often chance is as
  bullish as the RISING set (Fisher's randomization logic; Efron & Tibshirani, *An
  Introduction to the Bootstrap*, 1993).
- **One coincident shock dominates.** The 2020 COVID closure (Strip GGR ≈ $0 in April–May
  2020) and the 2021–22 rebound is one enormous coincident event; we report results with and
  without it so the verdict doesn't ride on a single episode.

## Method lineage (this study's engine)

- **Signal + inference.** [`strategy.ggr_momentum`](../vegas_gaming_win/strategy.py) (TTM
  run-rate momentum, deseasonalised), [`strategy.summarize`](../vegas_gaming_win/strategy.py)
  (Welch *t* + placebo *p*), [`strategy.lead_lag`](../vegas_gaming_win/strategy.py) (the
  leading-signal test), [`strategy.timing_overlay`](../vegas_gaming_win/strategy.py)
  (own-when-rising, one-month lag, one-way costs).
- **Deterministic synthetic control.**
  [`data.synthetic_ggr`](../vegas_gaming_win/data.py) plants a known GGR→returns link;
  `edge = 0` must not manufacture significance, a large `edge` must light up the test.

## Data sources used here

- **NGCB Las Vegas Strip GGR** (hardcoded monthly reconstruction, US$ millions) + **yfinance**
  casino basket daily adjusted close, 2002-02 → 2025-06, cached under
  `_cache/casino_prices.csv`. All headline numbers are pinned in
  [`docs/results.md`](results.md) and reproduced by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **[Study 385 — Jobless-Claims-Momentum](../385-jobless-claims-momentum/)**: the closest
  sibling — a famous "leading" macro print tested the same way (hardcoded snapshot + lead/lag +
  timing overlay), and found to be a coincident-to-lagging echo.
- **[Study 384 — ISM-PMI-Regime](../384-ism-pmi-regime/)** and
  **[Study 387 — Economic-Surprise-Index](../387-economic-surprise-index/)**: companion
  macro-nowcasting teardowns asking whether a celebrated diffusion/surprise gauge actually
  times equities.
