# References & literature map — Study 323 (BTC-Halving)

## The claim under test

- **The four-year halving cycle.** The most durable piece of crypto folklore: Bitcoin's
  price moves in a clean ~4-year cycle locked to the protocol's block-subsidy **halving**
  (every 210,000 blocks, ~every four years, the per-block reward halves). The popular
  narrative — repeated across PlanB's "stock-to-flow" essays, countless YouTube "cycle"
  charts, and exchange research notes — holds that the market **bottoms a little before**
  each halving, **tops ~12-18 months after** it, then crashes into the next bottom. The
  implied free lunch: the halving schedule is *known years in advance*, so the calendar
  alone would have timed the tops and bottoms.

- **Stock-to-flow (the supply-shock rationale).** "PlanB" (pseudonymous), *Modeling
  Bitcoin Value with Scarcity* (2019) and *Bitcoin Stock-to-Flow Cross Asset Model*
  (2020). The mechanism behind the cycle story: each halving cuts new supply, so (the
  argument goes) a fixed demand meets a supply shock and price must rise. The model was
  enormously influential and has since broken down badly out of sample.

## Why the mechanism is weaker than it sounds

- **The halving is fully anticipated.** Efficient-markets logic (Fama 1970, *Efficient
  Capital Markets*) says a supply change known years in advance should already be in the
  price; a *scheduled* event cannot be a tradable surprise. Any post-halving drift must
  come from something other than the supply cut being "news."
- **Stock-to-flow critiques.** Several analyses (e.g. the widely circulated 2021–2022
  rebuttals, and the model's own author conceding the floor models had failed) show the
  S2F relationship is a spurious regression of two trending series — exactly the trap
  Granger & Newbold (1974), *Spurious Regressions in Econometrics* (Journal of
  Econometrics), warned about. Two things that both go up over time correlate by
  construction.
- **Diminishing returns across cycles.** Each cycle's amplitude has shrunk (2013 ≫ 2017 >
  2021 > 2025-to-date), consistent with a maturing, more-arbitraged asset — and with the
  "cycle" being a small-sample artefact rather than a structural law.

## The small-sample / clustering problem (why we can't stamp REAL)

- **Effective n is the number of cycles, not the number of days.** Returns within a
  post-halving window are one autocorrelated path. Treating ~1000 daily observations from
  three halvings as independent inflates any *t*-stat enormously; the honest unit of
  observation is the **cycle**, of which Yahoo's tape has **three** (2016 partial, 2020,
  2024). This is the cluster-robust-inference point of Cameron, Gelbach & Miller (2011),
  *Robust Inference With Multiway Clustering* (Journal of Business & Economic Statistics):
  with a handful of clusters, no standard error is trustworthy.
- **Multiple testing / cycle-mining.** Harvey, Liu & Zhu (2016), *…and the Cross-Section of
  Expected Returns* (Review of Financial Studies). "Bottom before, top after" is one of
  many phase stories one could draw through four price paths; fitting a cycle to four
  realisations is the definition of over-determination.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  used in [`strategy.hac_tstat`](../btc_halving/strategy.py), with the explicit caveat
  above that it cannot fix a three-cluster sample.
- **Circular block bootstrap.** Politis & Romano (1992/1994), the stationary/circular
  block bootstrap — preserves the volatility clustering an i.i.d. resample would destroy.
  In [`strategy.block_bootstrap_ci`](../btc_halving/strategy.py).
- **Synthetic positive control.** A deterministic GBM tape with a tunable halving-locked
  sinusoid ([`data.synthetic_daily`](../btc_halving/data.py)) — proves the engine detects a
  planted cycle and finds nothing in the null, the standard machinery proof.

## Data sources used here

- **Yahoo! Finance `BTC-USD` daily bars** (via `yfinance`), auto-adjusted close. History
  begins **2014-09-17** — the central data limitation of this study. Halving dates are the
  four public protocol facts, hardcoded in [`data.HALVINGS`](../btc_halving/data.py). All
  headline numbers are pinned with an as-of date and content fingerprint (see
  [`docs/results.md`](results.md)). The offline reproducible core and test-suite run on the
  deterministic synthetic generator, never the network.

## Related desk studies

- **[Study 117 — Pi-Cycle-Top](../../117-pi-cycle-top/)**: the "Pi Cycle Top" moving-average
  cross sold as a Bitcoin top-caller — the same "the calendar/indicator prints the top"
  family, also a small-sample mirage.
- **[Study 174 — Bitcoin-Rainbow](../../174-bitcoin-rainbow/)**: the rainbow log-regression
  band, another visually compelling BTC "cycle" overlay fit to one price path.
- **[Study 292 — Bitcoin-Hashrate](../../292-bitcoin-hashrate/)** & **[Study 293 — MVRV](../../293-mvrv-ratio/)**:
  on-chain "this predicts BTC price" theses, each pinned against buy-and-hold and each
  failing to beat it — the same single-trending-asset trap this study runs into.
- **[Study 133 — Crypto-Seasonality](../../133-crypto-seasonality/)**: calendar effects in
  crypto returns; the halving cycle is the grandest calendar story of all.
