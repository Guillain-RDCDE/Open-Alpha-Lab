# Study 110 — Faber-Timing

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | Max drawdown cut from **−55%** to **−22%** (−33 pp) across 33 years; timing Sharpe **+0.729** vs random-control **+0.304** (*t* = +1.75). On the risk dimension, the timing is real. |
| **Tradability** — does it survive costs, capacity, scale? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | Only ~6 switches/yr; costs barely matter. But CAGR lags BH (+9.2% vs +10.8%) and the Sharpe improvement vs BH is not statistically certified (*t* = −0.73 on return difference). Works as a drawdown shield, not as a return engine. |
| **Risk reduction or alpha?** | ![Confirmed](https://img.shields.io/badge/Risk_reduction%3F-Confirmed-8b949e?style=flat-square) | The Sharpe improves because vol falls more than CAGR. It is a smoother ride at a cost of some return — exactly what Faber claimed. |

> **In one sentence:** Faber's 200-day SMA rule is a genuine bear-market shield — it cuts SPY's max drawdown from −55% to −22% over 33 years in a way random timing cannot replicate — but it lags buy-and-hold on total return in bull decades, making it a risk-reduction tool, not an alpha generator.

## What we tested

The most-cited tactical allocation rule: hold the S&P 500 (SPY) when its price is *above* its
200-day (10-month) SMA; move entirely to T-bills when below. Faber's 2007 SSRN paper has over
one million downloads and has been replicated across five asset classes and a century of data.
We test it on SPY daily total-return closes 1993–2026 vs a buy-and-hold baseline and — critically —
vs a **random-timing control** matched to the same in-market fraction (75.3%) on random days. This
isolates *timing skill* from mere *exposure reduction*. We credit cash at a flat 4%/yr proxy,
net of 5 bps one-way transaction costs. Sub-period breakdown reveals the expected asymmetry:
the rule excels in bear decades (2000s: timing Sharpe +0.55 vs BH −0.04) and lags in pure
bull markets (pre-2000: timing Sharpe +1.11 vs BH +1.25).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the rule in plain language, the crash protection story, why CAGR lags and Sharpe leads, the random-control surprise |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC t-stats, Sharpe diff test, cost sweep, sub-period breakdown, synthetic regime control, the exposure-reduction vs timing-skill distinction |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`faber_timing/`](faber_timing/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
