# Study 144 — Permanent-Portfolio

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Real (risk-adjusted)](https://img.shields.io/badge/Signal-Real_(risk--adjusted)-2ea44f?style=flat-square) | PP Sharpe **0.775** vs 0.527 (SPY) and 0.651 (60/40); bootstrap 95% CI [−0.13, +0.62], PP wins 90% of resamples; 5/7 equity crashes showed genuine cross-asset cushioning. CI spans zero — directionally robust, not certified at 5%. |
| **Tradability** — does it survive costs, capacity, scale? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | 4 liquid ETFs, annual rebalance, <1 bp CAGR cost impact. Structurally fragile: the 2004-2021 bond+gold bull inflates the Sharpe, and 2022 showed correlated inflation failure (bonds **and** gold failed simultaneously). Forward Sharpe advantage is uncertain. |
| **Returns vs SPY** | ![Negative](https://img.shields.io/badge/Returns_vs_SPY%3F-Negative-8b949e?style=flat-square) | PP CAGR **7.4%** vs **10.9%** (SPY); −3.5 pp/yr, HAC t = −1.66. The return forfeiture is the cost of the hedge. Accepts this knowingly. |

> **In one sentence:** Harry Browne's 25/25/25/25 blend genuinely reduces drawdowns (−18% vs −55% for SPY) and improves Sharpe (0.78 vs 0.53), but it leans on a secular bond+gold bull and can fail in a correlated inflation shock — REAL on risk, FRAGILE going forward.

## What we tested

A famous recipe from 1987: divide your savings equally between stocks (SPY), long Treasuries (TLT), gold (GLD), and short-term Treasuries / cash (SHY). Rebalance once per year. One of the four legs is supposed to thrive in every macroeconomic regime: stocks in prosperity, long bonds in deflation, gold in inflation and crisis, and cash in recession. We run the exact recipe over 21 years (2004-2026, constrained by GLD's inception) and pin it against 100% SPY (the return ceiling) and 60/40 SPY/TLT (the mainstream alternative). We compute Sharpe ratios excess-of-SHY so all arms are compared fairly, run circular block-bootstrap CIs, examine every SPY crash episode individually, and confirm the engine on a synthetic positive control with a planted regime cycle. Distinct from Study 68 (All-Weather, risk parity) and Study 97 (Balancing-Act, 60/40 deep-dive).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the equity curve, the crash cushioning table, the return trade-off chart, year-by-year comparison in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | bootstrap Sharpe CIs, HAC t-stats on annual diffs, regime crash table, synthetic positive control with tunable cycle strength |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`permanent_portfolio/`](permanent_portfolio/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
