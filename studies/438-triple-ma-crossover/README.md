# Study 438 — Triple MA Crossover 📉

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is there an edge? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The triple-MA (5/20/50) long/flat rule on 33.5y of ^GSPC earns a net excess **Sharpe of just +0.02** (HAC **t = 0.14** on the excess mean — far under the *t* ≥ 2 bar), and a circular-block permutation null can't tell it from luck (**p = 0.57**). Indistinguishable from noise. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Its Sharpe is **below buy-and-hold (+0.29)** *and* **below the simpler two-MA rule (+0.17)** — even at **0 bps**; it turns negative by 10 bps. The lower drawdown (−23% vs −57%) is pure exposure reduction (47% in-market), not timing. |
| **"Three beats two"?** | ![Busted](https://img.shields.io/badge/Three_beats_two%3F-Busted-8b949e?style=flat-square) | The third "confirmation" MA's marginal contribution is **negative**: HAC **t(triple − dual) = −1.46**, and the triple stack loses to its two-MA twin in **4/4** configs (incl. the famous 10/50/200). More confirmation → more lag → less return. |

> **In one sentence:** the celebrated triple moving-average "ribbon" — buy only when fast > medium > slow — has no edge on the S&P (net excess Sharpe +0.02, HAC *t* = 0.14, permutation *p* = 0.57), and the third "confirmation" average actively makes things worse: it loses to both buy-and-hold and the simpler two-MA cross it claims to beat, before costs and in every parameter set we tried.

## What we tested

We turn the triple-MA crossover into a daily long/flat timing rule on **^GSPC** (S&P 500, price-only): go long when **fast > medium > slow** (SMA 5/20/50), else hold cash. Because a part-time-in-cash rule must be judged **excess-of-cash**, we credit cash at a flat 3% and race **net excess Sharpe** against two benchmarks — buy-and-hold *and*, crucially, the **simpler two-MA cross** (5/50, same fast/slow legs) that the third MA claims to improve. The signal axis is a HAC *t* on the excess mean plus a circular-block permutation null; tradability charges one-way costs × NAV (one documented execution lag, signal-on-close-of-*t* held over *t+1*). We sweep four classic configs and the cost ladder, and confirm with a deterministic synthetic control that the engine *can* detect a planted trend — so the flat real-tape result is a genuine null, not a broken backtester.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a moving-average ribbon is, why "more confirmation" feels smart, and the one chart that shows three MAs finishing dead last behind two MAs and buy-and-hold — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the three-way excess-Sharpe race, HAC Sharpe-difference *t* (triple − BH, triple − dual), a circular-block permutation null, config + cost sweeps, and a synthetic faithful-engine / power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`triple_ma_crossover/`](triple_ma_crossover/). Data is **^GSPC price-only** daily closes (no dividends) — Sharpes are net, excess-of-cash, labelled throughout; single-index series, no survivorship issue. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
