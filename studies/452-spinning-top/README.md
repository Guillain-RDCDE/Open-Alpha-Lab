# Study 452 — Spinning-Top 🪀

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the indecision candle forecast? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The "buy after a spinning top" rule does **not** beat a drift-matched **random-entry** baseline once you average over which random dates you draw: the top-vs-random Welch *t* (seed-robust mean over 20 baseline seeds) is **−0.07 / +1.02 / +1.73 / −0.20** at 5/10/20/60 days — **never clears 2**. A single lucky seed hits **+3.38** at 20d, but the same horizon ranges down to **+0.67**: baseline-draw luck, not an edge. The huge one-sample *t*'s (20d **+8.35**) are **pure beta** — the upward drift every long entry inherits. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | No residual edge once the free drift is removed; costs only deepen the hole. You'd capture the same drift more cheaply by **holding the index**. Nothing to scale. |
| **"Does indecision forecast direction?"** | ![Busted](https://img.shields.io/badge/Indecision_forecasts_direction%3F-Busted-8b949e?style=flat-square) | Scramble the candle's wicks into nonsense (wick-scramble placebo) and the result barely moves: **~7%** of scrambled-shape candles match or beat the real spinning top (*p* = **0.066**, above the 0.05 bar). The small-body/balanced-wick *shape* carries no information — the rule is really just flagging high-range, two-sided days. |

> **In one sentence:** the spinning top looks meaningful because indices drift up — encode it mechanically (body < 25% of range, two long balanced wicks) and fire the "indecision resolves, buy it" rule on **3,006** candles across 5 indices over 21 years, and it **fails to robustly beat buying on random days** (seed-robust Welch *t* never ≥ 2; the lone >2 reading is a lucky baseline draw) while the geometry placebo leaves the result intact (*p* = 0.07): all tide, no tool.

## What we tested

We encode the tightest mechanical version a proponent would accept. A bar is a **spinning top** iff its real body is `< 25%` of the day's high-low range, **both** wicks are `≥ 25%` of the range, and the two wicks are comparable (the smaller is `≥ 50%` of the larger) — no eyeballing, all four prices known on the bar's close. A long fires on the **next close** (one documented lag), and we measure the forward 5/10/20/60-day return on SPY, QQQ, IWM, DIA and GLD (yfinance daily total-return OHLC, 2005→2026). The Signal axis is **spinning-top vs a drift-matched random-entry baseline** (a Welch *t*) — the only honest test on an upward-drifting tape — and because that baseline is itself a *draw* of dates, we **average the Welch *t* over 20 baseline seeds** so a lucky comparison set can't masquerade as an edge. We add a **wick-scramble geometry placebo** that destroys the spinning-top shape while keeping the price path and the wick marginal. Tradability charges costs on every entry. A deterministic synthetic control with a *planted* post-spinning-top move proves the detector is live (edge 0 → *t* = −1.45, no false positive; planted resolution → *t* = +7.38), so the flat real-tape result is a genuine "nothing there".

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a spinning top is, why a buy on a rising market always looks good, the top-vs-random race, and the wick scramble — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | mechanical candle geometry, one-sample HAC *t* vs the beta trap, the **seed-robust** random-entry Welch test, the wick-scramble placebo, per-ticker deltas, costs, and a synthetic planted-resolution control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`spinning_top/`](spinning_top/). Spinning tops are classified mechanically (body < 25% of range, both wicks ≥ 25%, balance ≥ 0.5) on the bar's close; entry is the next close (one lag). Basket is surviving liquid ETFs — but this is a single-instrument event study, so the random-entry baseline neutralizes the drift/survivorship. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
