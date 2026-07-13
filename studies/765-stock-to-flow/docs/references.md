# References & literature map — Study 765 (Stock-to-Flow)

## The claim under test

- **The folklore.** "Stock-to-Flow" (S2F) — PlanB (pseudonymous, @100trillionUSD), *"Modeling
  Bitcoin's Value with Scarcity"*, Medium, 2019-03-22
  (https://medium.com/@100trillionUSD/modeling-bitcoins-value-with-scarcity-91fa0fc03e25). The
  mechanism as the believers state it: a commodity's value is driven by scarcity, quantified as
  the **stock-to-flow ratio** `SF = existing stock / annual new production`. Gold and silver have
  high SF; Bitcoin's SF doubles at each halving. PlanB regressed `ln(market value)` on `ln(SF)`
  across monthly Bitcoin data (and gold/silver anchors) and reported an in-sample **R² ≈ 0.95**,
  with a fitted power law `market_value = exp(a)·SF^b` (b ≈ 3.3 in his headline chart). The model
  implied a ~$100k BTC after the 2020 halving and famously higher targets thereafter.
- **The follow-up that raised the stakes.** PlanB, *"Bitcoin Stock-to-Flow Cross Asset (S2FX)
  Model"* (2020), extended the fit across asset classes and produced the widely-cited **$288k**
  average-price prediction for the 2024 cycle. These concrete, dated, six-figure forecasts are
  what made S2F the most famous — and, after 2022, the most cited-as-busted — valuation model in
  crypto.
- **The economic logic, steelmanned.** Bitcoin issuance is *deterministic consensus law*: the
  block subsidy is 50 BTC and halves every 210,000 blocks, so the future flow is knowable years
  ahead. If scarcity really priced the asset, a model built purely on that schedule would be an
  extraordinary thing — a valuation you could compute from first principles with no market data.
  That is the version we test at full strength.
- **What we are NOT testing.** Whether the halving *event* moves price on a calendar
  (that is a supply-shock timing question). This study tests the literal S2F **level model** —
  the power-law fit and its out-of-sample predictions — and whether the model's valuation gap is
  a tradable signal.

## The academic rebuttal we lean on

- **Marcel Burger / "The Stock-to-Flow model is a spurious regression."** The central critique
  (widely circulated 2021–2022, e.g. https://medium.com/coinmonks/critique-of-the-stock-to-flow-cross-asset-model-32ebb01b6b7c
  and follow-ups): both `ln(price)` and `ln(SF)` are **non-stationary, trending** (integrated)
  series, and `ln(SF)` is nearly a deterministic function of time. Regressing one trending series
  on another manufactures a high R² and significant *t* even when there is no relationship —
  Granger & Newbold's classic spurious-regression result. Our §1 reproduces this directly:
  `corr(ln SF, time) = 0.96`, and `ln(price) ~ time` scores R² = 0.876 vs the S2F fit's 0.880.
- **Granger, C. W. J. & Newbold, P. (1974), "Spurious Regressions in Econometrics",** *Journal of
  Econometrics* 2(2):111–120 — the foundational result that trending I(1) series yield inflated
  R² and *t* under OLS. The reason a beautiful S2F fit certifies nothing.
- **Phillips, P. C. B. (1986), "Understanding Spurious Regressions in Econometrics",** *Journal of
  Econometrics* 33(3):311–340 — the asymptotic theory of why the *t*-statistic diverges under
  non-stationarity; the formal backing for treating the in-sample fit as uninformative.

## Shared-method citations (the desk's standard machinery)

- **Newey, W. K. & West, K. D. (1987), "A Simple, Positive Semi-Definite, Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix",** *Econometrica* 55(3):703–708 — the HAC
  standard errors used for the residual→forward-return regression (overlapping windows induce
  exactly the serial correlation this corrects; we use lag = 1.5 × horizon).
- **White, H. (2000), "A Reality Check for Data Snooping",** *Econometrica* 68(5):1097–1126 —
  the spirit of the matched-exposure random-timing placebo: a signal must beat the distribution
  of random rules at the same exposure, not just an arbitrary baseline.
- **Bitcoin issuance schedule** — Bitcoin Core consensus rules (`GetBlockSubsidy`, 210,000-block
  halving interval); halving block heights 0 / 210,000 / 420,000 / 630,000 / 840,000 and their
  historical dates are public record. The S2F curve in [`data.py`](../stock_to_flow/data.py) is
  reconstructed from these, not estimated.

## Data sources

- **BTC-USD daily close** — yfinance (no key), cached under `_cache/s2f_btc_usd.csv`,
  2014-09-17 → 2026-06-30. Price-only == total-return for BTC (no dividends).
- **S2F curve** — reconstructed deterministically from the issuance schedule (above); stock is
  exact at every halving, interpolated by date within epochs; flow is the annualised current
  issuance. All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [323-btc-halving](../../323-btc-halving/) — tests the **halving calendar** as a price-timing
  event (a supply-shock *date*), not a valuation level model. S2F uses the same issuance schedule
  but as a continuous scarcity *ratio* fed into a power-law price model — a different claim.
- [663-hash-ribbons](../../663-hash-ribbons/) — a miner-capitulation **on-chain buy signal**;
  event study, not a valuation model.
- [293-mvrv-ratio](../../293-mvrv-ratio/) — an on-chain **valuation ratio** (market vs realised
  value) as a mean-reversion timing signal; the closest cousin, but MVRV is a market-derived
  metric whereas S2F is derived purely from the deterministic supply schedule.
- [221-mayer-multiple](../../221-mayer-multiple/) — a **price/200-day-SMA** valuation band; price
  only, no issuance.
- [210-crypto-trend](../../210-crypto-trend/) — 200-day price SMA trend-following; no valuation
  model.

None of the siblings test PlanB's literal `ln(price) ~ ln(SF)` power-law level model, its
out-of-sample divergence, or the spurious-regression critique — this study's own axis.
