# Study 460 — Counterattack / Meeting Lines ⚔️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the meeting reverse the trend? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The bullish meeting-line buy does **not** beat a drift-matched **random-entry** baseline: meeting − random = **−68.6 / −61.5 / +13.4 / +146.6 bps** at 5/10/20/60 days, and the meeting-vs-random Welch *t* is **significantly negative** at the reversal horizon (**−3.01** at 5d, *p* = 0.003) and **never clears 2** (max **+1.99** at 60d, *p* = 0.047). The respectable one-sample *t*'s (20d **+3.50**, 60d **+4.20**) are **pure beta** — the upward drift every dip-buy inherits. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | No residual edge once the free drift is removed; at the short, supposedly-reversal horizon the pattern is a **net negative**, and costs only deepen the hole. You'd capture the same drift more cheaply by **holding the index**. Nothing to scale. |
| **"Does the equal-close meeting forecast?"** | ![Busted](https://img.shields.io/badge/Meeting_forecasts%3F-Busted-8b949e?style=flat-square) | Drop the equal-close condition but keep the down-leg + gap context, and the result barely moves: **62%** of no-meeting down-leg dip-buys match or beat the real meeting line (*p* = **0.623**). The defining meeting carries no information. |

> **In one sentence:** The counterattack line looks plausible because indices drift up — encode it mechanically (down leg, opposite-colour candles, gap-down open, closes within 15 bps) and fire the buy 280 times across 5 indices over 21 years, and it **loses to buying on random days** at the 1–2 week reversal horizon (Welch *t* = −3.01 at 5d), only catching up at 20–60 days where it is just inheriting the market's climb (and the meeting placebo leaves the result untouched, *p* = 0.62): all tide, no tool.

## What we tested

We encode the tightest mechanical version a proponent would accept. A **bullish meeting line** completes at bar *t* when there is a confirmed down leg (close[t-1] below close[t-10]), candle *t-1* is **black** (close < open), candle *t* is **white** (close > open) and **gaps down** on the open (open[t] < close[t-1]), and the two closes **meet** (|close[t] − close[t-1]| / close[t-1] ≤ 15 bps). A long fires on the meeting bar's close, entered at the **next close** (one documented lag), and we measure the forward 5/10/20/60-day return on SPY, QQQ, IWM, DIA and GLD (yfinance daily total-return, 2005→2026). The Signal axis is **meeting vs a drift-matched random-entry baseline** (a Welch *t*) — the only honest test on an upward-drifting tape — plus a **close-scramble placebo** that keeps the down-leg/gap context but drops the equal-close test. Tradability charges costs on every signal. A deterministic synthetic control with a *planted* meeting-bounce proves the detector is live (edge 0 → *t* = −0.64; planted bounce → *t* = +9.86, win 92%), so the flat real-tape result is a genuine "nothing there".

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a counterattack line is, why a dip-buy on a rising market always looks good, the meeting-vs-random race, and the meeting placebo — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | mechanical meeting lines, one-sample HAC *t* vs the beta trap, the random-entry Welch test, the close-scramble placebo, per-ticker deltas, costs, and a synthetic planted-bounce control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`counterattack_lines/`](counterattack_lines/). Pattern is read on the close of *t* (down leg, opposite-colour candles, gap-down open, equal close within 15 bps); entry is the next close (one lag). Basket is surviving liquid ETFs — but this is a single-instrument pattern study, so the random-entry baseline neutralizes the drift/survivorship. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
