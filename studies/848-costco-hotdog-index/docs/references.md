# References & literature map — Study 848 ("Costco Hot-Dog Index")

## The claim under test

- **The pitch.** A recurring finance-media / retail-investor idea that the **$1.50 Costco
  hot-dog-and-soda combo**, nominally unchanged since ~**1985**, is a folk "anti-inflation"
  icon — and that Costco's (`COST`) **pricing power** and **membership-fee** model let its
  *stock* outrun inflation and the broad market "regardless." The testable versions:
  (H₀) the combo's *real* price has collapsed with CPI (a mechanical identity — quantify
  it); (H₁) COST total return beats CPI and SPY over the sample (descriptive); (H₂) COST's
  return has a **beta to inflation surprises** that is robustly non-zero and **differs from
  a consumer-staples basket** — the tradable "is it a distinctive inflation hedge?" claim.
- **The economics behind the steelman.** Costco's profit leans on recurring **membership
  fees** (a near-annuity, largely CPI-linked at renewal) and on razor-thin merchandise
  margins where a handful of "loss-leader" signposts (the hot dog, the rotisserie chicken)
  are deliberately frozen for brand equity. See Costco Wholesale 10-K / investor filings
  (SEC EDGAR, CIK 0000909832): https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000909832 .
  The frozen $1.50 combo is confirmed in decades of public reporting and management
  commentary (famously, CFO Richard Galanti's "$1.50 … forever" remarks).

## The CPI series (the real deflator)

- **FRED `CPIAUCSL` — Consumer Price Index for All Urban Consumers: All Items in U.S. City
  Average, seasonally adjusted, index 1982-84 = 100** (= BLS series `CUSR0000SA0`). The
  standard headline CPI level. https://fred.stlouisfed.org/series/CPIAUCSL
- **BLS public API** — the source of record, fetched directly. `hotdog_index.data.fetch_cpi`
  pulls `CUSR0000SA0` from `https://api.bls.gov/publicAPI/v2/timeseries/data/` (no key,
  ≤10-year windows) and caches the real monthly levels (2000-01 → 2026-06). FRED's own
  `fredgraph.csv` endpoint is unreachable from this build host, so BLS is the primary feed;
  the same real values are embedded in `data.py` as an offline fallback so the machinery
  tests never need the network. BLS CPI home: https://www.bls.gov/cpi/
- The 1985 anchor `CPI_1985 = 107.6` is the CPIAUCSL **1985 annual average** (the year the
  $1.50 combo was priced) — confirmed against the BLS 1985 monthly prints.
- **One interpolated point.** The SA series' **2025-10** print was not published (the 2025
  federal government-shutdown / data-collection disruption); it is linearly interpolated
  between the published 2025-09 and 2025-11 levels so the YoY differencing sees a contiguous
  monthly grid. This single point is the only value not taken directly from BLS.
- Load-bearing public facts: the multi-decade **level path** (169.3 in 2000-01 → 332.6 in
  2026-06, +96%) and the **2021-23 surge** (YoY inflation peaked **+9.0%** in 2022-06, the
  highest since 1981).

> **Transparency.** `hotdog_index.data.load_cpi` returns the **real** FRED/BLS `CPIAUCSL`
> series (BLS `CUSR0000SA0`), read from the cached BLS pull (or the embedded real values
> offline). Only the 2025-10 point is interpolated, and it is disclosed above; every other
> value is a BLS observation. The inflation-surprise signal is built on ΔYoY (the
> "inflation accelerating?" innovation).

## The real tape

- **Costco Wholesale (`COST`, NASDAQ).** The hot-dog company. Month-end total-return
  (`auto_adjust=True`) close via yfinance. A **hindsight-selected single name** — its
  historical win is descriptive, not a forward promise; the survivorship caveat travels
  with every published number.
- **`SPY`** — SPDR S&P 500 ETF, the "you'd have done better / worse" market benchmark.
- **`XLP`** — Consumer Staples Select Sector SPDR, the "does *pricing power* beat a boring
  staples basket?" control for the inflation-beta comparison.

## Why the folk hedge is the wrong default — the finance

- **Nominal-rigidity vs the Fisher/real-price identity.** A nominally frozen price has a
  *real* price that decays mechanically as `1/CPI` (Fisher 1930, *The Theory of Interest*).
  The combo's real erosion is arithmetic, not an anomaly — and carries no information about
  the equity that sells it.
- **Stocks and inflation surprises.** The classic result (Fama & Schwert 1977, *Asset
  returns and inflation*; Bodie 1976) is that equities are a **poor short-run inflation
  hedge** — nominal stock returns tend to correlate *negatively* with inflation surprises,
  not positively. A single retailer's frozen loss-leader does not overturn that; our
  near-zero COST beta is the expected outcome.
- **Survivorship / hindsight selection.** Choosing COST *because* it is the famous
  hot-dog stock that thrived conditions on the outcome (the desk's recurring lesson; cf.
  Brown, Goetzmann, Ibbotson & Ross 1992 on survivorship). Its 14.6× real multiple is a
  fact about the past of one winner, not a tradable inflation rule.
- **Spurious "it went up, so it hedges" reasoning.** Conflating a high total return with a
  causal inflation-protection mechanism is the level-vs-relationship trap; the honest test
  is the *beta to inflation surprises* and whether it beats a staples control — both null.
- **HAC inference.** Newey & West (1987). Inflation shocks persist for months, so naive
  standard errors overstate significance; every *t* here is Newey-West (Bartlett, 6-lag),
  cross-checked with a block-rotation placebo and a two-era split.

## Method lineage (the desk's shared engine)

- **Inference primitives.** `one_sample_t`, `welch_t`, `newey_west_t`, `wilson_interval`
  and an OLS+HAC slope regression (`hac_regression`) — the canonical house kit
  ([`hotdog_index/strategy.py`](../hotdog_index/strategy.py)), matching
  [803-realized-skewness-reversal](../../803-realized-skewness-reversal/).
- **Real-price identity + total-return race.** `real_price` / `erosion_stats` /
  `total_return_race` — the mechanical deflation and the descriptive COST/SPY/CPI race.
- **Signal.** `inflation_beta` (contemporaneous & predictive) and `beta_gap`
  (COST − staples / − market), with a block-rotation `placebo_pvalue` and an `era_cut`.
  `REAL` needs HAC |t| ≥ 2 on the contemporaneous COST beta **and** a distinctive gap vs
  staples **and** sub-era robustness — none is met.
- **Timer + cost realism.** One-month execution lag; excess-of-cash Sharpe; one-way × NAV
  per switch ([`strategy.timer_stats`](../hotdog_index/strategy.py)).
- **Deterministic synthetic control.** A fixed-seed planted-inflation-β world
  ([`data.synthetic_world`](../hotdog_index/data.py)), averaged over ≥20 seeds
  (`synthetic_mean_t`), proving the engine recovers a real inflation beta — no network.
- **Reproducibility.** As-of slice + content fingerprint, pinned in
  [`docs/results.md`](results.md) and reproduced by [`examples/verify.py`](../examples/verify.py).

## Related desk studies (dedup — how 848 differs)

- **[Study 215 — Big-Mac PPP](../../215-big-mac-ppp/)**: the Big-Mac *purchasing-power-parity*
  currency signal (a cross-country FX story). 848 is a **single frozen US price vs CPI and a
  single stock**, not a cross-country PPP test.
- **[Study 725 — Eggflation](../../725-eggflation/)**: trading the avian-flu egg *spike* via
  Cal-Maine. 848 is the mirror image — a *frozen* price and a **pricing-power / inflation-
  hedge** claim about the retailer, not a commodity-spike momentum trade.
- **[Study 726 — Chicken-Wing Index](../../726-chicken-wing-index/)**: a commodity-price
  folklore signal. 848 is about **CPI deflation of a nominal icon and equity inflation-beta**,
  not a wing-price trade.
- **[Study 266 — Misery Index](../../266-misery-index/)**: CPI *inflation + unemployment* as a
  macro return signal. 848 shares the cited-public-record CPI pattern but asks a **firm-level
  inflation-hedge** question about COST, not a macro misery timing rule.
