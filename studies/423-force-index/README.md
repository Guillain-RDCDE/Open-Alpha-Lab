# Study 423 — Force Index

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the rule beat holding? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | FI(13) zero-cross long/flat scores net Sharpe **0.30** vs buy-and-hold's **0.64** on SPY (1993–2026), a **−3.45 bps/day** drag at HAC *t* = **−4.03**. Negative on **all six** ETF tapes (significantly on five). A sign-permutation placebo puts the real Sharpe gap at *p* = **0.82** — no timing skill beyond exposure. |
| **Tradability** — could you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Behind buy-and-hold **even gross** (Sharpe 0.34); the ~18 flips/yr of costs only widen the gap and **no positive break-even cost exists**. The long/short version is outright **negative** (−0.30). |
| **"Flags reversals"?** | ![Busted](https://img.shields.io/badge/Flags_reversals%3F-Busted-8b949e?style=flat-square) | A plain **SMA(50/200)** filter — same machinery, no volume term — scores **0.74**, beating *both* Force Index and buy-and-hold. The price×volume twist subtracts value; it adds churn, not foresight. |

> **In one sentence:** Elder's Force Index zero-cross is not a reversal detector — it is a noisy, lagging trend proxy that pulls you out of the market during the up-legs that pay, underperforming buy-and-hold on every one of six ETFs (SPY net Sharpe 0.30 vs 0.64, HAC *t* = −4.03), losing to a plain moving-average filter that *beats* the market, and going outright negative as a long/short.

## What we tested

Alexander Elder's *Force Index* (`(Close − Close_prev) × Volume`, EMA-smoothed) is one of the most-taught price-volume oscillators in retail trading (*Trading for a Living*, 1993). We take the folk rule literally: **long while the 13-day Force Index is above zero, flat while it's below**, entered with one execution lag, and race its **NET** Sharpe against **buy-and-hold** on an excess-of-cash to excess-of-cash basis — with a Newey–West HAC *t* on the daily return difference, a sign-permutation placebo, a one-way cost sweep, an FI(2)/FI(13)/FI(50) period sweep, a long/short variant, and the obvious simpler benchmarks (SMA 50/200, RSI 14) so the *"it's better"* claim is actually tested. A deterministic synthetic tape with a **planted signed-volume edge** is the positive control: the same engine recovers the edge when one exists (HAC *t* → +5.49), proving the verdict is about the market, not the method.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what the Force Index is, why "big move on big volume" sounds smart, the equity curve vs buy-and-hold, why a dumber moving-average filter wins, in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | FI(13) zero-cross long/flat & long/short, NET Sharpe race vs buy-and-hold excess-vs-excess, per-tape HAC *t*, period & cost sweeps, the SMA/RSI benchmark race, a sign-permutation placebo, and the planted-edge positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`force_index/`](force_index/). Close is Yahoo total-return-adjusted; race is NET of one-way costs × NAV with one-day execution lag, shorts paying borrow. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
