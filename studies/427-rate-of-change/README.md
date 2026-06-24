# Study 427 — Rate of Change ⏱️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does ROC add an edge? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | ROC's *own* excess Sharpe **0.59** (HAC *t* = **3.71**) clears t≥2 — but that's **beta** (buy-and-hold's *t* = **3.64**, same machinery). The decisive number, ROC **minus** buy-and-hold, is **−2.97%/yr at HAC *t* = −1.47** — a small, *insignificant loss*. The timing isn't pure noise (permutation *p* = **0.034**) but it shows up as risk-reduction, not return. Not REAL — it fails the only test that matters. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | ROC **lowers** CAGR (**8.8%** vs **10.9%**); its sole deliverable is drawdown (**−31%** vs **−55%**), which a 200-day SMA hands you identically. Costs are a red herring (~7 turns/yr, Sharpe barely moves to 10 bps) — there's simply no excess return to harvest or scale. |
| **"A better momentum tool"?** | ![Misattributed](https://img.shields.io/badge/Better_momentum_tool%3F-Misattributed-8b949e?style=flat-square) | ROC(126) (**0.59**) is a dead heat with a plain SMA(200) (**0.58**); both beat the higher-churn MACD/RSI. The benefit is the shared "stay-long-in-uptrends" filter — not the ROC formula. The credit is mis-assigned to the indicator. |

> **In one sentence:** the oldest momentum oscillator's headline *t* of 3.71 looks great only because it's long a rising market ~76% of the time — once you strip that beta out, ROC adds **−2.97%/yr at t = −1.47** over simply holding SPY, its lone real benefit (half the drawdown) is delivered identically by a 200-day moving average, and the failure is structural, not a cost problem.

## What we tested

A staple of every charting platform since the 1970s: *"Rate of Change is the purest momentum tool — go long when ROC (the percent price change over the last N days) is above zero, step aside to cash when it's below, and you ride the trend while dodging crashes."* We take it literally as a daily **long/flat** (and long/short) timing rule on **SPY total-return closes, 1993→2026** (33 years, fingerprint `f3fa058adfc8`), with one execution lag (signal at close *t*, return *t+1*) and costs one-way × NAV. The Signal axis is the HAC *t* on the **difference** ROC − buy-and-hold (the number that removes beta), backed by a 5,000-draw circular-permutation placebo on the timing; Tradability runs a cost sweep + break-even; and a head-to-head against **SMA(200), MACD, RSI** tests the implicit "ROC is better" claim instead of asserting it. A deterministic regime-switching synthetic control confirms the harness *can* detect a real timing edge — and that on SPY there is none.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what ROC is, why its "win" is really just owning stocks, the one thing it buys you (half the drawdown), and why a plain moving average ties it — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the beta-vs-alpha *t* split, the HAC *t* on ROC − buy-and-hold, a circular-permutation placebo, window robustness + cost sweep, the SMA/MACD/RSI race, and a regime-switching positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`rate_of_change/`](rate_of_change/). SPY closes are **total-return** (`auto_adjust=True`); Sharpe is excess-vs-excess, gross/net labeled. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
