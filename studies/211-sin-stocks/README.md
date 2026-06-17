# Study 211 — Sin-Stocks

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Excess return **−1.40%/yr**, HAC *t* = **−0.46**; OLS alpha **+0.61%/yr**, *t* = **+0.20**. The basket earns equity returns (raw *t* = +2.06) but there is **no evidence of a neglect premium over the market** — the claimed effect has t far below the |t| ≥ 2 bar. Bootstrap excess-Sharpe CI [−0.539, +0.339], 68% of resamples negative. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | SIN_EW CAGR **+10.60%** vs SPY **+12.17%**; Sharpe **0.481** vs **0.579**. Total return since 2008: **527%** vs **710%**. LVS casino stock: −98.3% peak-to-trough, 58.5% vol, +0.62%/yr CAGR — drags down the whole basket. |
| **Neglect premium (Hong-Kacperczyk)?** | ![Mixed](https://img.shields.io/badge/Neglect_premium%3F-Mixed-8b949e?style=flat-square) | Outperformed 2009–2017 (value era, +3–6%/yr excess); catastrophic in 2018–2021 growth era (−17%/yr); huge positive 2022, fully reversed 2023. Regime-dependent, not a durable anomaly post-publication. |

> **In one sentence:** the six-stock sin basket (tobacco, alcohol, gambling, defense) returned 527% vs the market's 710% since 2008, with a lower Sharpe, worse max drawdown, and no statistically detectable neglect premium (excess return HAC t = −0.46) — the Hong-Kacperczyk effect was real in 1965–2006 but does not hold in this post-publication sample.

## What we tested

Do vice stocks (tobacco: MO, PM; alcohol: STZ; gambling: LVS; defense: LMT, RTX) beat the
market (SPY) and the ESG counter-portfolio (DSI) on a risk-adjusted basis?  We test an
equal-weight basket of six sin tickers vs SPY and DSI over 2008–2026 (n = 4,589 daily
total-return closes), computing CAGR, Sharpe, OLS alpha/beta, per-ticker attribution,
crash-episode protection, and sub-period breakdown.  The "neglect premium" hypothesis (Hong
& Kacperczyk 2009) predicts a positive HAC t-stat on the excess return; we find t = −0.46.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | cumulative wealth chart, ticker attribution, crash episodes, sub-period breakdown in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC t-stat, bootstrap Sharpe CI, OLS alpha/beta, ESG comparison, regime decomposition, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`sin_stocks/`](sin_stocks/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
