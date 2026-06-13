# Study 97 — Balancing-Act 🏦

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the risk-adjusted improvement real? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | Yes: 60/40 lifts the excess-of-cash Sharpe to **0.655** (vs **0.541** for 100% stocks), halves volatility (**10.2%** vs 18.8%) and drawdown (**−29.8%** vs −55.2%). The Sharpe edge **+0.115** survives HAC *t* = **−2.28** and a block-bootstrap CI **[+0.022, +0.207]** that excludes zero. |
| **Tradability** — can you actually run it? | ![Investable](https://img.shields.io/badge/Investable-2ea44f?style=flat-square) | Two cheap ETFs (SPY + IEF), annual rebalance — a sensible default core. *With caveats:* it gave up **2.6 pts/yr** of CAGR (8.63% vs 11.20%), and much of the Sharpe edge is the historic **bond bull market** — beta you were paid for, not a free structural law. |
| **"Bonds cushion *every* equity crash"?** | ![Not supported](https://img.shields.io/badge/Not_supported-8b949e?style=flat-square) | Bonds rose in **6 of 8** equity crashes — but **failed in 2022**: as stocks fell ~24%, IEF fell **−15%** and TLT **−31%**. The cushion is regime-dependent (rolling stock/bond corr ranges −0.83 to +0.67), not a law. |

> **In one sentence:** the fixed 60/40 genuinely lowered volatility and improved risk-adjusted return over 2002–2021 — a *real* and *investable* default — but the "bonds cushion every crash" selling point is **not supported** (it broke in 2022), and much of the Sharpe edge is the disinflation-era bond bull, not a free structural law.

## What we tested

The classic **60/40 stock/bond portfolio**, steelmanned: *"the sensible default — most of the stock return with much less risk, a better risk-adjusted outcome than 100% stocks, and bonds cushion every equity crash."* We build the **fixed** 60% SPY / 40% IEF blend (total return), **rebalanced annually**, charge 2 bps on rebalance turnover, and race it against 100% stocks and a 60/40 (SPY/TLT) variant over **2002–2026** (the Treasury-ETF-bounded window). Sharpes are **excess-of-cash** (SHY proxy). We put the Sharpe *difference* through a HAC *t* and a circular block bootstrap, track the **rolling stock/bond correlation** across regimes, and stress the cushion claim on every >10% equity drawdown and specifically on **2022**. The offline control is a two-asset world with a tunable correlation (negatively-correlated legs → the blend beats the best leg; positively-correlated → it doesn't). **Distinct from [Study 68 (All-Weather)](../../68-all-weather/)**, which is *volatility*-weighted risk parity — this is the *fixed*-weight 60/40.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a smoother ride is real, what you give up for it, and the year (2022) the cushion failed |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | excess-of-cash Sharpe, HAC *t* + block-bootstrap on the Sharpe difference, rolling correlation, the bond-bull / beta read |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run (2002–2026, joint fp `75b675e02dc5`): [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`balancing_act/`](balancing_act/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
