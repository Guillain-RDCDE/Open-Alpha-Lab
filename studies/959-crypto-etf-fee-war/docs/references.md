# References & literature map — Study 959 (Crypto Fee War)

## The claim under test

- **The fee-war thesis.** In January 2024 the SEC approved eleven US spot-bitcoin ETFs on
  the same day. Ten began trading on 2024-01-11 holding the identical asset, with the
  identical strike time, under near-identical prospectuses — and with a fee dispersion no
  other ETF category has ever launched with: **19 bp (Franklin) to 150 bp (Grayscale)**, a
  nearly eightfold spread, on top of a wave of introductory waivers that took several
  launches to **zero** for six to twelve months. The folk claim is the obvious one: the
  cheapest wrapper should track best, the ranking of published fees should reproduce itself
  in realised tracking difference, and the waiver expiries should be visible as steps in the
  tape. This study tests all three on the tape.
- **The steelman.** Unlike almost every claim on this desk, this one has a *mechanism that
  cannot fail*: a sponsor fee is contractually accrued daily out of NAV. There is no
  forecast, no behaviour, no crowding. If the tape is measured properly the fee must appear.
  The interesting question is therefore not *whether* but **how much of it a public daily
  tape can resolve** — which turns the study into an exercise in measurement floors.

## Tracking difference: what it is, and how it is usually measured wrong

- **Tracking difference vs tracking error.** *Difference* is the mean gap between fund and
  benchmark return (a level, dominated by fee and drag); *error* is the standard deviation
  of that gap (a dispersion). The trade press routinely reports one and names the other.
  See Elton, Gruber, Comer & Li (2002), *Spiders: Where Are the Bugs?*, Journal of Business
  — the founding decomposition of an ETF's return gap into fee, cash drag, and dividend
  timing, on the original SPDR.
- **Gastineau (2004), *The Benchmark Index ETF Performance Problem*, Journal of Portfolio
  Management** — why index-tracking vehicles under-perform their benchmarks by more than
  their fee, and why the residual is mechanical rather than skill.
- **Petajisto (2017), *Inefficiencies in the Pricing of Exchange-Traded Funds*, Financial
  Analysts Journal** — the premium/discount process around NAV. This is the AR(1) term the
  synthetic generator plants: it is what makes an *endpoint* tracking-difference estimate
  unreliable, because it contaminates both anchors.
- **Ben-David, Franzoni & Moussawi (2017), *Exchange-Traded Funds*, Annual Review of
  Financial Economics** — the arbitrage mechanism that keeps a wrapper on its asset, and its
  limits when creation/redemption is impaired (the GBTC-before-conversion case).

## Why the fee ranking is measurable at the top and not inside the pack

- **The clock stub is the whole problem.** Bitcoin trades continuously; the ETFs strike at
  16:00 New York. The fund-minus-spot difference therefore inherits the overnight and
  weekend move — ~135 bp of daily standard deviation on this tape — while the
  fund-minus-fund difference inherits almost none (~9 bp). Our Study **175-crypto-weekend**
  measures that same 24/7-versus-exchange-hours mismatch from the other side, as a
  return effect rather than a measurement nuisance.
- **Detection floors in fund comparison.** Study **913-tracking-difference-persistence**
  established the desk's version of this argument on S&P 500 trackers: a 6.45 bp fee spread
  cannot be read through a 10.7 bp/yr measurement floor. Study 959 is the same argument at a
  different scale — the *within-tier* spread here (6 bp) is again below the floor (66 bp/yr,
  higher because the sample is 29 months rather than 14 years), while the *across-tier*
  spread (130 bp) towers over it.
- **Fee dispersion in identical products.** Hortaçsu & Syverson (2004), *Product
  Differentiation, Search Costs, and Competition in the Mutual Fund Industry*, Quarterly
  Journal of Economics — the canonical demonstration that S&P 500 index funds charging
  wildly different fees for an identical product is an equilibrium, not an anomaly. It is
  the reason GBTC can still charge 150 bp: search costs, inertia, and — here — an embedded
  capital-gains lock-in the study prices explicitly.

## The GBTC case specifically

- **Grayscale's conversion.** GBTC traded as a closed-end trust from 2013 and converted to
  an ETF on 2024-01-11, the cohort's first day. Its closing discount that afternoon is worth
  ~250 bp on its own — which is exactly why this study's *endpoint* estimator puts GBTC's
  leak at −79 bp/yr, roughly half the truth, and why the monthly estimator is the headline.
- **Study 618-gbtc-premium-cycle** is the desk's study of the pre-conversion premium and
  discount cycle itself, as a tradable dislocation. Study 959 begins the day that cycle
  *ended* and asks a different question: what does the surviving wrapper leak, now that
  arbitrage works and only the fee is left?
- **The Mini Trust as a control.** Grayscale launched the Bitcoin Mini Trust (ticker BTC)
  on 2024-07-31 at 15 bp, spun out of GBTC's own assets. Same sponsor, same coin, same
  custodian, same strike — 135 bp of fee difference and nothing else. It is the cleanest
  natural control the desk has had for a fee effect.

## Related desk studies (dedup)

- **[Study 913 — Tracking-Difference Persistence](../../913-tracking-difference-persistence/)**:
  does last year's TD *rank* predict next year's, among S&P 500 trackers, over 14 years. That
  study is about **persistence** (a memory question) on a mature, low-dispersion category.
  Study 959 asks about **level** (a delivery question) on the newest, widest-dispersion
  category on the tape, with a purpose-built waiver event study 913 has no analogue of.
- **[Study 958 — Spot-BTC ETF Basis](../../958-spot-btc-etf-basis/)**: the *premium/discount*
  of the same wrappers against spot — the high-frequency dislocation term. Study 959 treats
  that term as the nuisance to be differenced away, and measures the slow contractual drift
  underneath it.
- **[Study 618 — GBTC Premium Cycle](../../618-gbtc-premium-cycle/)**: the pre-conversion
  closed-end discount. Ends where 959 begins.
- **[Study 378 — ETF NAV Premium](../../378-etf-nav-premium/)**: premium/discount as a
  general cross-ETF signal, not a fee measurement.
- **[Study 624 — Buffer ETF Cost](../../624-buffer-etf-cost/)** and
  **[Study 942 — Inverse ETF Structural Loss](../../942-inverse-etf-structural-loss/)**:
  structural leaks in *derivative* wrappers (option cost, volatility drag). Study 959's
  wrapper holds the asset outright, so the only leak is the fee — which is why the answer
  here is so much sharper than in either of those.
- **[Study 956 — ADR Custody Fee Drag](../../956-adr-custody-fee-drag/)**: the same
  "published fee, does it show up?" question in a different wrapper (depositary receipts).
- **[Study 210 — Crypto Trend](../../210-crypto-trend/)**, **[Study 632 — Crypto XS
  Momentum](../../632-crypto-xs-momentum/)**, **[Study 133 — Crypto Seasonality](../../133-crypto-seasonality/)**:
  return-predicting rules on the coin. Study 959 is deliberately orthogonal — it holds
  bitcoin exposure constant and measures only the wrapper.

## Method lineage

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*, Econometrica —
  [`strategy.newey_west_t`](../fee_war/strategy.py) and
  [`quantlab.analytics.mean_tstat_hac`](../../../quantlab/analytics.py).
- **Moving-block bootstrap.** Künsch (1989), *The Jackknife and the Bootstrap for General
  Stationary Observations*, Annals of Statistics; Politis & Romano (1994), *The Stationary
  Bootstrap*, JASA — [`strategy.block_bootstrap_ci`](../fee_war/strategy.py).
- **Permutation test for a rank statistic.** Fisher (1935), *The Design of Experiments*;
  Spearman (1904), *The Proof and Measurement of Association Between Two Things*, American
  Journal of Psychology — [`strategy.rank_test`](../fee_war/strategy.py), enumerated exactly
  for ≤ 8 funds and sampled above that, and reporting the **attainable** 5% critical value so
  a tied fee sheet cannot be mistaken for an insignificant result.
- **Reproducibility stamp.** [`quantlab.repro`](../../../quantlab/repro.py) — as-of slicing
  and content fingerprinting, so a re-run either matches byte for byte or says so loudly.

## Data sources

- **IBIT, FBTC, ARKB, BITB, HODL, BRRR, BTCO, EZBC, BTCW, GBTC** (the 2024-01-11 cohort),
  **BTC** (Grayscale Mini Trust, 2024-07-31), **BTC-USD** (24/7 spot) and **BIL** (cash leg)
  — daily closes via `yfinance` (`auto_adjust=True`), 2023-12-01 → 2026-06-30. None of the
  wrappers distributes, so adjusted = price for all of them; BTC-USD is **price-only** by
  nature; BIL is genuine total return and is used only for the excess-of-cash race.
- **DEFI (Hashdex)** is excluded and named in `data.EXCLUDED`: a bitcoin *futures* fund
  until its 2024-03-27 spot conversion.
- **Fee and waiver schedules** are **not tape data**. They are issuer disclosure recorded at
  build time, live in `data.FEE_BPS` / `data.WAIVER`, are labelled ASSUMPTION everywhere they
  are used, and enter only the rank test and the waiver event study — never the headline
  spread, which is pure tape.
- **As-of 2026-06-30**, the last complete calendar month; the partial current month is
  dropped so the non-overlapping monthly estimator never eats a stub.
