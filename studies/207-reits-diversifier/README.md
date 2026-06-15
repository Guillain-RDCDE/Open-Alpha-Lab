# Study 207 — REITs-Diversifier

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Mixed](https://img.shields.io/badge/Mixed-dab617?style=flat-square) | VNQ earns +7.3% CAGR (genuine income), but adding it to a 60/40 *lowers* the Sharpe by 0.117 and deepens max drawdown by 18 pp; HAC *t* = +1.40 on the CAGR diff, not clearing the bar. |
| **Tradability** — does it survive costs, capacity, scale? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | Sleeve beats 60/40 CAGR by only +0.7 pp on far more vol; bootstrap CI overwhelmingly favours 60/40 (sleeve wins 6% of resamples). Tradable as income; not as diversifier. |
| **Inflation hedge?** | ![Busted](https://img.shields.io/badge/Inflation_hedge%3F-Busted-8b949e?style=flat-square) | In four high-CPI years VNQ averaged **−2.7%** vs SPY **−0.0%**; interest-rate sensitivity dominates the real-asset story. VNQ amplified equity losses in GFC (−69%), COVID (−42%), and 2022 (−32%). |

> **In one sentence:** REITs (VNQ) are leveraged equity-beta dressed up as diversifiers — 0.75 correlated with SPY full-sample, deeper drawdowns than SPY in every major crisis, and the worst performer in the high-inflation years where the hedge story should shine.

## What we tested

A staple of the asset-allocation playbook: *"Add a REIT sleeve to your 60/40 — real estate has low correlation with stocks and bonds, hedges inflation via rents, and produces stable income through the cycle."* We take all three claims literally and run them: (1) does 60/20/20 SPY/VNQ/TLT beat 60/40 on risk-adjusted metrics? (2) does VNQ outperform in high-CPI years? (3) does VNQ cushion equity crashes? Using VNQ daily total-return prices from 2004-11-18 through 2026-06-11 (~21.5 years), with CPI regimes from the Shiller dataset and a deterministic synthetic control to confirm the engine can detect genuine diversification when it is planted.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the three claims, the equity-beta reality, crash behaviour, and the inflation-hedge failure in plain English |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | portfolio stats table, rolling VNQ–SPY correlation, crisis-episode breakdown, CPI-regime returns, HAC t-stats, bootstrap Sharpe CI, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`reits_diversifier/`](reits_diversifier/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
