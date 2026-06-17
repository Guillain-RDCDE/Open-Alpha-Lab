# Study 227 — Natgas-Winter

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does UNG earn a winter premium? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | No. The regression slope on the winter dummy is **−0.035 (t = −1.96)** — the wrong sign. Winter months average **−3.56%/month** versus **−0.04%/month** in summer. Winter is the worst season, not the best. |
| **Tradability** — does buying Oct-Mar beat holding year-round? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The seasonal strategy earns **Sharpe −0.60** against buy-and-hold **Sharpe −0.45** (CAGR −24.5% vs −28.2%) — concentrating in winter months amplifies the loss. Both strategies are ruinous; the seasonal is worse. |
| **"Winter is a tailwind for natgas longs"?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | Busted in both sub-periods: 2007–2015 (t = −0.99, wrong sign) and 2016–2026 (t = −1.72, wrong sign). The contango roll-drag more than swamps any spot-price seasonality — in every sub-period. |

> **In one sentence:** the winter-natgas-spike trade is the commodity market's widow-maker — contango roll drag (up to −3%/month in winter) more than cancels any heating-demand tailwind, and the seasonal long destroys capital faster than simply holding UNG year-round.

## What we tested

A genuine cross-asset folk belief: **buy UNG (United States Natural Gas Fund) in October, sell in
March** — capture the winter heating spike. The winter demand peak (Dec-Feb) is real in physical
markets. But UNG is a front-month futures roll, not a spot holding. When the natural-gas curve is in
contango (the standard condition), each monthly roll sells the cheap front contract and buys the
expensive second — a guaranteed drag. We test the seasonal long on **every complete calendar month of
UNG's history** (2007-05 → 2026-05, 229 months, hole-free): a winter-dummy regression (is the
Oct-Mar slope positive and significant?), a strategy vs buy-and-hold race, and a sub-period decay split.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why heating demand doesn't translate into tradable returns — and what the widow-maker really means |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the winter-dummy regression, the seasonal vs buy-and-hold race, sub-period slope splits |

The fingerprinted real-data run (UNG, 2007–2026, fp `68346ee6801f`) is in [docs/results.md](docs/results.md).
Reproduce via [examples/verify.py](examples/verify.py) (`--fetch` to download); the offline machinery proof
runs on the synthetic world in [natgas_winter/data.py](natgas_winter/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
