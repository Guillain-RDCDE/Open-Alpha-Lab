# Study 673 — T3 (Tillson)

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does T3 timing beat holding? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The T3(14, v=0.7) price-cross rule's daily **active spread vs buy-and-hold is −3.40 bps/day at HAC *t* = −4.03** on SPY (gross *t* = −3.31), negative on all five basket tapes (four of five individually *t* ≤ −2), in both sample halves, and across **every** volume factor v from 0.1 to 0.9. A position-shuffle permutation gives ***p* = 0.927** — the timing is *worse* than random. |
| **Tradability** — could you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Net Sharpe **0.308 vs buy-and-hold 0.647**; even **gross** it trails. There is no break-even cost, and the long/short version is outright destructive (Sharpe −0.29). |
| **"Genuinely lower-lag, cleaner crossovers"?** | ![Mixed](https://img.shields.io/badge/Genuinely_lower--lag%2C_cleaner_crossovers%3F-Mixed-8b949e?style=flat-square) | *Cleaner* holds: T3 fires **12–17% fewer** whipsaws than SMA/EMA of the same N, on all five tickers. *Lower-lag* is **busted**: T3 tracks price less tightly (1.88% vs 1.53%/1.33%) and is the **slowest** of the three to catch up after a deterministic step — the opposite of Tillson's own pitch at the same N. Neither half beats the "dumb" MAs it claims to (head-to-head *t* = 0.97 / 0.42). |

> **In one sentence:** T3's own headline claim splits in two — it really does whipsaw less than a plain SMA/EMA (12–17% fewer switches, on every basket ticker) but it does **not** react faster (it's the slowest of the three to catch a shock at the same N), and the resulting timing loses to buy-and-hold by 3.40 bps/day at *t* = −4.03, fails a permutation placebo (*p* = 0.927), and stays negative and significant across every "volume factor" from 0.1 to 0.9 — while a synthetic tape with a planted trend confirms the engine banks a real trend the moment one exists.

## What we tested

A staple of MT4/5 indicator packs and TradingView scripts since 1998: *"Tim Tillson's T3 — a six-times-smoothed, 'generalized DEMA' moving average tuned by a volume factor v — virtually eliminates lag while smoothing the data, so a crossover or slope rule built on it turns earlier **and** gives cleaner crossovers than a plain SMA/EMA."* We take it literally: implement `T3 = GD(GD(GD(price, v), v), v)` from Tillson's own 1998 formula (six stacked EMAs, no price recursion), turn price-vs-T3 (and, separately, T3's own slope) into a daily **long/flat** timing rule with one documented execution lag (position formed on the close of *t* earns *t+1*'s return), and race it **net of one-way costs × NAV** against **buy-and-hold** and against the obvious simpler benchmarks — an **SMA(14)** and an **EMA(14)** crossover, same nominal N — on SPY, QQQ, AAPL, MSFT and XLE (total-return daily bars, full history to 2026-06-30). We test the mechanism first (a deterministic step response and tracking distance), then the Signal axis with a Newey-West HAC *t* and a 2,000-draw position-shuffle permutation, then sweep Tillson's own "volume factor" v (0.1→0.9) for robustness. A deterministic synthetic tape with a *planted* trend is the positive control proving the harness banks an edge when one is there. **Dedup:** [672-mcginley-dynamic](../672-mcginley-dynamic/) (a *recursive* price-adaptive MA, opposite mechanism), [432-hull-moving-average](../432-hull-moving-average/) (the *opposite* failure mode: less lag, *more* whipsaws), [483-zlema](../483-zlema/) (a minimal single-term lag-cancellation EMA), [674-vidya](../674-vidya/) and [433-kama-adaptive](../433-kama-adaptive/) (state-driven adaptive-speed MAs, not T3's fixed-coefficient stack) — none tests T3's own six-stage nested construction or its counter-intuitive "smoother AND slower" failure mode.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what T3 is, why "six smoothed EMAs" sounds like a strict upgrade, why "smoother" and "less lag" turn out to be different promises, and why timing must be raced against just holding, in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the step-response/tracking-distance mechanism check, T3 vs SMA vs EMA vs buy&hold, the active-spread HAC *t*, the whipsaw count, the position-shuffle permutation, cost & per-instrument & in/out-of-sample sweeps, the T3-slope variant, the volume-factor v robustness sweep, and the planted-trend positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`t3_tillson/`](t3_tillson/). Daily bars are **total-return** (`auto_adjust=True`); all Sharpe figures are net and excess-of-cash. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
