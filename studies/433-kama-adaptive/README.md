# Study 433 — Kaufman Adaptive MA 📉

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does adapting add a timing edge? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | On SPY the KAMA-cross book's daily out-performance over a matched **SMA(30)** is **−1.39 bps/day at HAC *t* = −2.98** — significant on the **wrong** side. Five of six tapes are significantly behind the plain SMA (none clears +2); the position-shuffle placebo gives ***p* = 0.995**. The adaptation carries a reliable *dis*advantage. *(Survivorship: liquid surviving ETFs/large-caps — the bias favours holding, which is exactly what wins here.)* |
| **Tradability** — could you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | KAMA net Sharpe **+0.20** vs SMA **+0.53** vs buy-and-hold **+0.65** — it loses to *both*. And it **churns more** (turnover 1269 vs 765), so costs *widen* the gap (KS *t* = −2.87 gross → −3.41 at 5 bp). Nothing to trade. |
| **"Adapting to volatility beats a fixed SMA?"** | ![Busted](https://img.shields.io/badge/Beats_a_fixed_SMA%3F-Busted-8b949e?style=flat-square) | The adaptive smoothing makes the rule **worse** and **raises** turnover — the exact reverse of the "fewer whipsaws" pitch. A planted-edge synthetic control confirms the harness *can* find a KAMA win (KS *t* = +2.86) when trends/chop alternate sharply — the real market just isn't that tape. |

> **In one sentence:** Kaufman's Adaptive MA promises fewer whipsaws than a fixed moving average by tightening in trends and freezing in chop — but raced head-to-head against a plain SMA(30) on 33 years of SPY (and five other tapes), the adaptation is *significantly worse* (KAMA − SMA = −1.39 bps/day, *t* = −2.98), churns *more*, and loses to buy-and-hold; only a synthetic regime-switching tape, where the harness recovers a +2.86 *t*, shows when KAMA *would* help.

## What we tested

A staple of charting platforms and "smarter than a moving average" trading content: *"KAMA scales its smoothing to the Efficiency Ratio — net travel ÷ total path — so it acts like a fast EMA in clean trends and a slow EMA in chop. A price-cross timing rule on KAMA therefore beats the same rule on a fixed SMA: fewer whipsaws, faster entries."* We take the only fair version of that claim and race it: compute KAMA(10, 2, 30) on daily closes, go **long when close > KAMA / flat otherwise**, enter at the next day's return (one documented execution lag), NET of one-way costs × NAV turnover with shorts paying borrow — then pit it **head-to-head against the identical rule on a fixed SMA(30)** and against buy-and-hold, all **excess-of-cash**. Arbiters: a HAC *t* on the KAMA − SMA daily out-performance, a position-shuffle permutation placebo, a cost sweep, an ER/length robustness grid, and a long/short variant. A deterministic synthetic regime-switching tape with a planted edge is the positive control that proves the engine can detect a KAMA advantage when one exists.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what "adaptive" means, why the Efficiency Ratio idea sounds smart, the head-to-head equity curves vs a plain SMA and vs just holding, and why adapting *added* whipsaws here — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | KAMA(10,2,30) vs SMA(30) long/flat books, KAMA − SMA HAC *t*, the position-shuffle placebo, cost & parameter sweeps, the long/short variant, and the synthetic planted-edge positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`kama_adaptive/`](kama_adaptive/). Real data: Yahoo daily, `auto_adjust=True` (total-return-ish), as-of 2026-05-31. Basket is **survivors** — named on the Signal axis. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
