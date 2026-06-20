# References & literature map — Study 341 (MLP-Pipelines)

## The claim under test

- **"Bond-like income from the toll-roads of energy."** The marketing pitch behind the MLP /
  midstream-income category — Alerian's **AMLP** (Alerian MLP ETF, the category bellwether),
  Global X's **MLPA** (MLP & Energy Infrastructure), and the leveraged InfraCap **AMZA**. The
  funds advertise high, steady **distribution yields** (typically 7–8%, AMZA much higher) as
  fee-based "income" that is insensitive to the oil price, and are sold to income investors and
  retirees as a high-yield bond substitute. The testable hypothesis: an MLP fund delivers a
  bond-like income stream (low energy beta, preserved NAV, comparable risk) plus a fat
  distribution. We test it with an energy-beta regression (HAC *t*), a SPY/XLE total-return race,
  and a distribution-vs-NAV decomposition.

## MLPs / midstream — the established theory and prior evidence

- **The C-corporation wrapper drag.** An ETF of MLPs that holds >25% MLPs must be structured as a
  taxable C-corp and accrues a **deferred tax liability** on unrealised gains, which is netted out
  of NAV. In the 2014–16 energy bust AMLP's accrued tax asset/liability whipsawed and the fund
  took a large NAV write-down; the structure mechanically caps long-run NAV growth relative to the
  underlying index. (See SS&C/ALPS AMLP prospectus tax disclosures; Morningstar, *The Hidden Cost
  of MLP ETFs*, on the C-corp tax drag.)
- **Distribution coverage and return of capital.** A large share of MLP fund distributions is
  classified as **return of capital** for tax purposes; when distribution coverage falls below 1×
  (as it did broadly in 2015–2020), the payout is financed by shrinking NAV rather than by
  distributable cash flow. The decomposition total return = price (NAV) return + distribution
  yield makes this visible — a negative price CAGR under a high distribution means the *entire*
  distribution is return of capital. This is the central, distinct object of this study.
- **MLPs as an energy/oil factor.** Empirically, midstream MLP returns load heavily on the energy
  sector and on crude oil despite the "fee-based, volume-not-price" narrative — distribution
  sustainability, counterparty risk and sentiment all co-move with the oil cycle. AMLP fell ~50%
  in 2014–16 and ~60% in March 2020 alongside the oil collapse. (See Alerian index methodology;
  energy-sector beta is documented across the midstream literature.)

## The income/"yield" illusion — distribution vs total return

- **The dividend/"income" fallacy.** Hartzmark & Solomon (2019), *The Dividend Disconnect*
  (Journal of Finance) — investors treat distributions as separate "income" rather than as
  self-financed sales of principal, the behavioural reason high-distribution products (MLP funds,
  covered-call ETFs, high-dividend tilts) are appealing despite being NAV-neutral at best. Mirrors
  this desk's [Study 57 — Yield-Trap](../../57-yield-trap/), [Study 337 — Covered-Call-ETF]
  (../../337-covered-call-etf/) and [Study 143 — Dividend-Capture](../../143-dividend-capture/).
- **Miller & Modigliani (1961)** — dividend (distribution) policy is irrelevant to value in
  frictionless markets; a payout is a portfolio rebalance, not a return. The MLP "yield" inherits
  the same logic with a structural NAV bleed bolted on top.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) — applied to
  the energy-beta regression slope and the monthly return spread in
  [`strategy.energy_beta`](../mlp_pipelines/strategy.py) and `strategy.race`.
- **Circular block bootstrap.** Politis & Romano (1992) — block resampling preserves the serial
  dependence that i.i.d. resampling destroys; used for the beta CI and the spread's 95% CI.
- **Up/down capture.** The standard Morningstar capture-ratio construction, computed on aligned
  monthly returns conditioned on the energy factor's sign.

## Data sources used here

- **Yahoo! Finance monthly bars** (via `yfinance`). The regression and race use `auto_adjust=True`
  (total return, distributions reinvested, fund fees inside); the income-illusion split uses the
  split-only close + the dividend stream so the price (NAV) leg can be separated from the
  distribution. Tickers: AMLP, MLPA, AMZA, SPY, XLE, USO. As-of **2026-05-31**, partial current
  month dropped, content-fingerprinted (see [`docs/results.md`](results.md)). The offline
  reproducible core and the test-suite run entirely on the deterministic
  [`data.synthetic_mlp`](../mlp_pipelines/data.py) replicator, never the network.

## Related desk studies

- **[Study 337 — Covered-Call-ETF](../../337-covered-call-etf/)** — the *option-overlay* version of
  the same return-of-capital income illusion (JEPI/QYLD). **This study is the distinct asset-class
  angle:** the *MLP/midstream* income illusion, where the third axis is **energy beta** (the income
  is a leveraged oil bet), not capped upside.
- **[Study 338 — Preferred-Stocks](../../338-preferred-stocks/)** — another "bond-like" high-yield
  sleeve that carries equity-grade crash risk; same "the safety is a costume" lesson, different
  instrument.
- **[Study 57 — Yield-Trap](../../57-yield-trap/)** — high-dividend stocks on total return; same
  "the yield is not free" lesson on single names.
