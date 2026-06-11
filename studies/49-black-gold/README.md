# Study 49 — Black-Gold 🛢️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does oil forecast next month's stocks? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | No. Regressing equity returns on last calendar month's oil gives slope **−0.001 (t = −0.03)** — as close to exactly zero as a regression gets. |
| **Tradability** — does an oil-timing rule help? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The rule (hold equities after oil falls, T-bill in cash) earns **Sharpe 0.32** vs buy-and-hold's **0.37** (excess of T-bill, both legs) — and gives up 1.6%/yr of compounding by sitting out 56% of the time for no edge. |
| **"Replicates out of sample"?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | Insignificant in *both* 2000–2008 (t = −0.9) and 2009-on (t = +0.2, sign flipped). The 1973–2003 result did not carry into the tradable era. |

> **In one sentence:** the striking claim that the oil price forecasts the stock market doesn't survive a single month into the era you'd actually trade it — the predictive slope is exactly zero, and a timing rule built on it trails buy-and-hold even with its cash leg paid the T-bill.

## What we tested

A genuinely intriguing cross-asset claim: **oil-price changes predict next month's equity returns** (Driesprong, Jacobsen & Maat 2008; [paperswithbacktest](https://github.com/paperswithbacktest/awesome-systematic-trading) lists it at Sharpe `0.599`), negatively — oil rises, stocks fall next month, a delayed reaction. We test it on **every calendar month of tradable oil-futures data** (WTI CL=F vs the S&P 500, 2000-09 → 2026-05, 309 months on a verified hole-free monthly grid built from daily closes): the predictive regression (with an analytic t-stat, calendar lag), a timing rule that holds equities after oil falls — T-bill (^IRX) credited in cash — versus buy-and-hold, and a pre/post-2008 decay split. The offline control is a synthetic world with a tunable (negative-sign) oil→equity link and a null.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a clever cross-asset story can be true once and dead forever after |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the predictive regression and its t-stat, the timing-vs-buy-and-hold race, the sub-period slopes |

The fingerprinted real-data run (CL=F + ^GSPC + ^IRX, 2000–2026, fp `5b97b7d871ee`) is in [docs/results.md](docs/results.md). Reproduce via [examples/verify.py](examples/verify.py) (`--fetch` to download); the offline machinery proof runs on the synthetic world in [black_gold/data.py](black_gold/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
