# References & literature map — Study 334 (ARK-Innovation)

## The claim under test

- **ARKK as the retail-innovation-momentum trade.** ARK Invest's *ARK Innovation ETF*
  (ARKK), managed by Cathie Wood, was the most-hyped active ETF of 2020-21: a concentrated,
  high-beta basket of "disruptive innovation" names that rose ~+150% in 2020 and drew
  enormous retail and adviser flows. The bull case stated at full strength: innovation is a
  durable momentum theme, and a star manager riding it compounds wealth far faster than the
  index. We test that two ways — *(a)* is there a tradable trend / momentum / mean-reversion
  signal in ARKK's own price, and *(b)* did the people who actually bought it make money
  (the dollar-weighted return), or did the flows arrive at the top?

## The headline mechanism — the investor-return (behaviour) gap

- **Morningstar, *Mind the Gap* (annual series; Amy C. Arnott).** The canonical measurement
  of the gap between a fund's *time-weighted* (buy-and-hold) return and its *dollar-weighted*
  (investor) return. Morningstar's analyses of ARK funds (2021-2023) repeatedly singled out
  ARKK as a worst-case: despite a positive posted return over its big years, the average
  dollar earned far less, because assets peaked near the 2021 top. The widely-cited estimate
  is that ARKK *destroyed* on the order of **$7–10 billion** of investor wealth on a
  dollar-weighted basis over 2014-2023.
- **Dichev, I. (2007), *What Are Stock Investors' Actual Historical Returns? Evidence from
  Dollar-Weighted Returns* (American Economic Review).** The foundational method: the
  money-weighted IRR of a fund's actual cashflows is the right measure of investor
  experience, and it is systematically *below* the time-weighted return because flows chase
  performance. This study's `money_weighted_irr` / `behaviour_gap` implement exactly his
  construction.
- **Friesen, G. & Sapp, T. (2007), *Mutual Fund Flows and Investor Returns* (Journal of
  Banking & Finance).** Quantifies the performance-chasing timing penalty across mutual
  funds — the general phenomenon ARKK is an extreme instance of.

## Performance-chasing flows and the smart-money / dumb-money divide

- **Frazzini, A. & Lamont, O. (2008), *Dumb Money: Mutual Fund Flows and the Cross-Section
  of Stock Returns* (Journal of Financial Economics).** Retail flows predict *lower* future
  returns — money arrives after the run-up. The behaviour engine in `synthetic_hype_cycle`
  (flows chase the trailing 12-month return) is a direct caricature of this.
- **Ben-David, Franzoni, Kim & Moussawi (2023), *Competition for Attention in the ETF
  Space* (Review of Financial Studies).** Specialised/thematic ETFs (ARKK is the archetype)
  launch and gather assets near a hype peak and then systematically underperform — the
  buy-the-top machine at the product level.

## Trend / momentum / mean-reversion — the overlay tools

- **Moskowitz, Ooi & Pedersen (2012), *Time Series Momentum* (Journal of Financial
  Economics).** The basis for the `ts_momentum_signal` overlay (long when trailing return is
  positive). We find ARKK's daily series carries no tradable persistence (HAC *t* = +1.36).
- **Faber, M. (2007), *A Quantitative Approach to Tactical Asset Allocation* (Journal of
  Wealth Management).** The moving-average-timing rule behind `ma_crossover_signal`; on ARKK
  the 50/200 crossover clears the bar (*t* = +2.09) but only as crash avoidance.
- **Jegadeesh, N. (1990), *Evidence of Predictable Behavior of Security Returns* (Journal of
  Finance).** Short-horizon mean reversion — the effect `mean_reversion_signal` would
  harvest if present; on ARKK it is not (*t* = +0.33).

## Method lineage (the desk's shared engine)

- **HAC / Newey-West *t*-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  [`strategy.hac_tstat`](../ark_innovation/strategy.py).
- **Circular block bootstrap.** Politis & Romano (1992) / Künsch (1989) — block resampling
  preserves the volatility clustering i.i.d. bootstrap destroys —
  [`strategy.block_bootstrap_ci`](../ark_innovation/strategy.py).
- **Excess-of-cash Sharpe race.** A long/flat overlay sits in cash part-time, so it is
  compared excess-to-excess against fully-invested benchmarks (`strategy.excess_sharpe`).

## Data sources used here

- **Yahoo! Finance daily bars** (via `yfinance`), auto-adjusted (total-return proxy):
  ARKK, QQQ, XLK, since ARKK's 2014-10-31 inception. ARKK's **shares-outstanding / AUM
  history is not on the public daily feed**, so the *real* dollar-weighted gap is quoted
  from the Morningstar literature above and the mechanism is demonstrated on the
  deterministic [`data.synthetic_hype_cycle`](../ark_innovation/data.py) generator. The
  offline reproducible core and test-suite run entirely on that generator, never the
  network. All real headline numbers are pinned with an as-of date and content fingerprint
  (see [`docs/results.md`](results.md)).

## Related desk studies

- **[Study 04 — Social-Oracle](../../04-social-oracle/)** and **[Study 254 — WSB-mentions](../../254-wsb-mentions/)**:
  the retail-hype / crowd family ARKK belongs to.
- **[Study 213 — Meme-stocks](../../213-meme-stocks/)**: the same buy-the-top dynamic in
  individual names; ARKK is the diversified fund version.
- **[Study 144 — Permanent-Portfolio](../../144-permanent-portfolio/)** and
  **[Study 97 — Balancing-Act](../../97-balancing-act/)**: the time-weighted vs investor-
  experience contrast on diversified allocations, the calm counterpoint to ARKK's hype cycle.
