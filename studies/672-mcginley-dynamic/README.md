# Study 672 — McGinley Dynamic 📉

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is there a real timing edge? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | McGinley Dynamic(14)'s daily active spread vs buy-and-hold is **−3.01 bps/day at HAC *t* = −3.55** on SPY (gross *t* = −2.98), negative on **all five** basket tapes and **both** sample halves. A position-shuffle permutation gives ***p* = 0.9895** — the timing is *worse* than random. |
| **Tradability** — could you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Net Sharpe **0.380 vs buy-and-hold 0.647** (gross 0.489 — still behind). No cost level tested rescues it, there's no break-even, and the long/short flip is worse (Sharpe **−0.178**). |
| **Cuts whipsaws & beats a plain MA?** | ![Mixed](https://img.shields.io/badge/Cuts_whipsaws_%26_beats_a_plain_MA%3F-Mixed-8b949e?style=flat-square) | It genuinely fires **30-33% fewer** position changes than SMA(14)/EMA(14) (25.5 vs 36.5/38.3 switches/yr) — but the stated *mechanism* runs backwards (it tracks price **more loosely**, 1.98% vs EMA's 1.33%, and reacts **slower** to a shock), and the fewer trades clear *t* ≥ 2 against SMA/EMA on only 3 of 10 basket comparisons. |

> **In one sentence:** John McGinley's "auto-adjusting" line really does cut whipsaws — 30-33% fewer position changes than a plain SMA/EMA of the same length, the opposite failure mode from its cousins the Hull MA and KAMA (which *add* whipsaws) — but it does so by tracking price *more loosely* and reacting *slower*, not faster, and turned into a timing rule it still loses to buy-and-hold by 3.0 bps/day at *t* = −3.55, loses to a random reshuffle of its own calls (*p* = 0.99), and its "better than a plain MA" edge never clears *t* = 2 outside three of ten head-to-head checks.

## What we tested

A staple of charting platforms and technical-analysis glossaries since the 1990s: *"McGinley's Dynamic line automatically adjusts its speed to the market — it hugs price far more closely than a fixed SMA or EMA of the same length, so its crossovers are cleaner and it never needs re-optimizing."* We take it at face value on **two levels**: first the literal mechanism (does the `(P/MD)^4` term actually make the line track price more tightly and react faster? — a deterministic step-response test and a real-tape tracking-distance check), then the trading claim (turn price-vs-line into a daily **long/flat** timing rule, net of one-way costs × NAV with one execution lag, and race it against the equivalent **SMA(14)** and **EMA(14)** rules and against **buy-and-hold** on a five-ticker liquid basket — SPY, QQQ, AAPL, MSFT, XLE, daily total-return bars to 2026-06-30). The Signal axis uses a Newey-West HAC *t* on the daily active spread and a 2,000-draw position-shuffle permutation; the third axis counts whipsaws and races McGinley head-to-head against the "dumb" MAs it claims to beat. A deterministic synthetic tape with a *planted* trend is the positive control proving the harness banks an edge when one exists. **Dedup:** siblings [91-death-cross](../91-death-cross/) (a fixed SMA cross, not a "smarter line"), [432-hull-moving-average](../432-hull-moving-average/) and [433-kama-adaptive](../433-kama-adaptive/) (adaptive MAs that *increase* whipsaws), [434-dema-tema](../434-dema-tema/) (lag-cancelling MAs, same story) and [437-triple-ma-crossover](../437-triple-ma-crossover/) (a rule-shape variant, not a line-shape one) — none tests McGinley's specific quartic-brake formula.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what "auto-adjusting" actually means, the step-response test that shows the mechanism runs backwards, the head-to-head race vs SMA/EMA and vs just holding, in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the tracking-distance and step-response mechanism checks, the HAC-*t* active-spread race, the McGinley-minus-SMA/EMA head-to-head, the permutation placebo, cost/parameter/split robustness, and the planted-trend synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`mcginley_dynamic/`](mcginley_dynamic/). Real tape is Yahoo daily, `auto_adjust=True` (total-return), as-of 2026-06-30. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
