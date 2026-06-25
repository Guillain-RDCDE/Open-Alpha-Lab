# Study 471 — QQE ⚡

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the band-cross pay? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The "buy the QQE band-cross" rule does **not** beat a **pooled** drift-matched **random-entry** baseline: cross − random = **+2.8 / +3.3 / −9.9 / +6.0 bps** at 5/10/20/60 days, and the cross-vs-random Welch *t* **never clears 2** (max **+0.28** at 5d, *p* = 0.78). The big one-sample *t*'s (20d **+7.01**, 60d **+9.05**) are **pure beta** — the upward drift every long-only entry inherits. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | No residual edge once the free drift is removed; costs only deepen the hole. You'd capture the same drift more cheaply by **holding the index**. Nothing to scale. |
| **"Does the QQE band-cross forecast?"** | ![Busted](https://img.shields.io/badge/Band--cross_forecasts%3F-Busted-8b949e?style=flat-square) | Scramble the price *timing* into a Fourier look-alike and the result barely moves: **31%** of nonsense-timing tapes match or beat the real one (*p* = **0.313**). The QQE geometry carries no information. |

> **In one sentence:** QQE looks like a momentum signal because indices drift up — encode it mechanically (causal smoothed-RSI band-cross, no look-ahead) and fire it 697 times across 5 indices over 21 years, and it **ties buying on random days** (and the timing placebo leaves it untouched, *p* = 0.31): all tide, no tool. The kicker — with **one** unlucky random seed the cross *appears* to win at Welch *t* ≈ 3; pool the seeds and that fake edge dies.

## What we tested

We encode the standard mechanical QQE a proponent would accept. A causal **Wilder RSI** (len 14) is EMA-smoothed (sf 5); an **ATR of the smoothed RSI** (× the Wilder factor 4.236) is laid out as a dual-band **trailing stop** that flips between a long band (below) and a short band (above) — exactly as the TradingView/MetaTrader scripts do. A long fires on the bar where the smoothed RSI **crosses above** the trailing stop, entered at the **next close** (one documented lag), and we measure the forward 5/10/20/60-day return on SPY, QQQ, IWM, DIA and GLD (yfinance daily total-return, 2005→2026). The Signal axis is **band-cross vs a drift-matched random-entry baseline** (a Welch *t*) — **pooled over 40 random seeds**, because a single noisy draw can fabricate a fake edge — plus a **phase-scramble placebo** (Fourier surrogate) that destroys the timing while keeping the spectrum/marginal. Tradability charges costs on every cross. A deterministic synthetic control with a *planted* post-cross continuation proves the detector is live (edge 0 → *t* < 0; planted continuation → *t* = +9.73), so the flat real-tape result is a genuine "nothing there".

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what QQE is, why a smoothed-RSI buy on a rising market always looks good, the cross-vs-random race, and the timing scramble — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | causal QQE bands, one-sample HAC *t* vs the beta trap, the **pooled** random-entry Welch test, the lucky-seed trap, the phase-scramble placebo, per-ticker deltas, costs, and a synthetic planted-continuation control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`qqe/`](qqe/). QQE is a causal Wilder-smoothed RSI with an ATR-of-RSI dual-band trailing stop; entry is the next close (one lag). Basket is surviving liquid ETFs — but this is a single-instrument momentum study, so the random-entry baseline neutralizes the drift/survivorship. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
