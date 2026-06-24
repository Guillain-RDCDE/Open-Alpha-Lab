# Study 418 — Money Flow Index 💵

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is there an edge? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | The MFI long/flat book is positive at **net excess Sharpe 0.46, HAC *t* = +2.72** (clears 2) — but it is long only ~42% of a market that drifted up, and a block-permutation placebo (*p* = 0.03) plus the buy-hold comparison expose the *t* as **recycled beta**, not a volume signal. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | **Buy-and-hold SPY (0.505) beats both timers at every cost level, including zero.** The rule converts beta into *less* beta; there is no residual alpha. |
| **"Beats the plain RSI?"** | ![Not supported](https://img.shields.io/badge/Beats_plain_RSI%3F-Not_supported-8b949e?style=flat-square) | The MFI edges the RSI by **+0.148 Sharpe**, but a 21-day block bootstrap on the daily difference puts that gap at **two-sided *p* = 0.28** — statistically invisible. |

> **In one sentence:** weighting the RSI by volume *feels* like adding information, but on 26 years of SPY the Money Flow Index beats the plain RSI by a statistically invisible +0.148 Sharpe (bootstrap *p* = 0.28), its own positive *t* = 2.72 is just being long a rising market ~42% of the time, and both oscillators lose to simply buying and holding the index — even at zero cost.

## What we tested

We build the 14-day **Money Flow Index** (typical-price × volume, up- vs down-day) and the 14-day **Wilder RSI** on SPY daily bars (2000–2026), then turn *both* into the **same** contrarian long/flat timing rule: indicator ≤ 30 → go long, ≥ 70 → step to cash, otherwise hold. The position is decided at each close and earns the *next* day's return (one execution lag); flat legs earn cash; costs are charged one-way × NAV on every flip; everything is measured **excess-of-cash** so the race is excess-vs-excess. We race the two indicators against each other and against buy-and-hold, test each book with a Newey-West HAC *t* and a block-permutation placebo, bootstrap the MFI-minus-RSI difference, sweep costs, and confirm with a synthetic positive control (a planted volume-keyed bounce) that the harness *can* see a volume edge when one is really there.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what the MFI is, why "volume confirms price" sounds smart, the three-horse race vs the RSI and buy-and-hold, and why the volume gap is noise — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the MFI/RSI construction, HAC *t* on each book, the block bootstrap on the head-to-head, the block-permutation placebo, the cost sweep, and the synthetic volume-edge control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`money_flow_index/`](money_flow_index/). SPY auto-adjusted daily (yfinance) — `close` is split/dividend-adjusted, `volume` is raw traded shares. Long/flat oscillator timer, one execution lag, excess-of-cash, one-way costs × NAV. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
