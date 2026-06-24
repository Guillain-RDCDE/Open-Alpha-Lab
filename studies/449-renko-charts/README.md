# Study 449 — Renko-Charts 🧱

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does brick-filtering add an edge? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The Renko-filtered 10/30 crossover and the **identical crossover on raw closes** are a dead heat: pooled incremental **delta Sharpe = -0.002**, sign-flipping across instruments, with a return-permutation placebo *p* = **0.495** (a coin flip). The crossover's own *t* = +3.43 is bull-market beta — it sits *below* buy-and-hold's +3.73, so there is no Renko alpha on this tape. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Even gross, Renko adds nothing: identical turnover (~7.8 trades/yr), identical Sharpe, and **both arms trail simply owning the basket** (Renko 0.683 vs buy-and-hold 0.713). No cost level and no instrument makes the brick the better choice. |
| **"Renko gives a cleaner, lower-noise signal"?** | ![Busted](https://img.shields.io/badge/Cleaner_signal%3F-Busted-8b949e?style=flat-square) | On daily bars the ATR-brick is ~one ATR wide, so the brick series tracks the close step-for-step: same trades, same Sharpe, same drawdowns as the raw crossover. The "noise filter" is a redraw of the same chart. |

> **In one sentence:** Renko charts promise a noise-filtered view that makes trend signals "cleaner," but on a six-ETF daily tape (2005–2026) a 10/30 moving-average crossover run on the ATR-Renko brick series is statistically indistinguishable from the same crossover on raw closes (delta Sharpe -0.002, placebo *p* = 0.50) — same turnover, same drawdowns — and both trail just buying and holding the basket; the brick is a redraw, not new information.

## What we tested

Renko proponents claim the brick chart strips out time and price-noise so trends and crossovers come through cleaner, with fewer whipsaws. We encode the tightest falsifiable version: build the **ATR-Renko** brick series (brick size = 1.0 × median ATR(14)) causally from daily closes, run a long/flat **10/30 simple-MA crossover** on the brick series, and pin it head-to-head against the **same crossover on raw closes** and against **buy-and-hold**, across SPY/QQQ/DIA/IWM/EFA/GLD (2005-01-03 → 2026-05-29, total-return, one execution lag, turnover one-way × NAV). The Signal axis tests the *incremental* Renko-over-raw Sharpe with a HAC *t* and a 500-draw return-permutation placebo; Tradability charges a cost sweep and counts turnover. A deterministic synthetic tape with planted return-persistence confirms the crossover engine *can* harvest a real trend — so finding nothing here is a true negative, not a broken harness.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a Renko chart is, why "cleaner chart" feels like an edge but isn't, the head-to-head vs the raw crossover and vs buy-and-hold, in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | ATR-brick construction, the 10/30 crossover head-to-head, HAC *t*, the return-permutation placebo, cost/turnover sweep, and the synthetic planted-trend positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`renko_charts/`](renko_charts/). Brick = 1.0 × median ATR(14); crossover = 10/30 SMA, long/flat. Adjusted close = total-return. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
