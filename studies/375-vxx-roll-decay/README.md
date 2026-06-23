# Study 375 — VXX-Roll-Decay 📉

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the roll-decay carry real? | ![Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square) | Contango is the usual state (**92%** of days), so each daily roll bleeds the long ETP into the short's pocket. The short-carry book earns **+0.171%/day**, HAC(21) **t = 2.51** (> 2), sign-shuffle placebo **p = 0.008** — a genuine premium, not a fluke. The ETP decayed **×3.4e-5** over 15 years. *(Vehicle is **VIXY**, the continuous-tape VXX-equivalent — named on the axis.)* |
| **Tradability** — can you deploy it? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | Net of borrow + costs the Sharpe is a tradable-looking **0.50** — but a constant-notional short carries skew **−1.65**, excess kurtosis **9.4**, a **−43%** single day (Aug-2024) and a **−92%** drawdown. Conditioning on contango lifts the Sharpe but still leaves a −67% drawdown; crisis borrow fees gut it. Survives on paper, **un-allocatable** at NAV scale. |
| **Free carry?** | ![Busted](https://img.shields.io/badge/Free_carry%3F-Busted-8b949e?style=flat-square) | The carry is real but it is **selling crash insurance**, not a free lunch. You collect a steady nickel in front of a steamroller — and the steamroller has run **three times in fifteen years** (Volmageddon 2018, COVID 2020, the 2024 unwind), each taking a third-to-half of the book in a single session. |

> **In one sentence:** shorting a VIX-futures ETP for its contango roll-decay is a *statistically real* carry (HAC t = 2.51, +35%/yr net), but it is the premium for selling crash insurance — a −43% day and a −92% drawdown make a constant-notional short un-allocatable, so it is real-as-a-signal and fragile-to-mirage as a trade: nickels in front of a steamroller.

## What we tested

A continuous, split-adjusted `VXX` series isn't on yfinance (the iPath ETN was redeemed and reissued in 2018, which would miss Volmageddon), so we short **VIXY** — the *same* short-dated constant-maturity VIX-futures object with an unbroken tape from 2011 — labelled a VXX-equivalent throughout. We hold a **constant-notional short**, rebalanced daily, and measure the carry against a **Newey-West (HAC) t-stat** and a **sign-shuffle placebo** null; then we charge it **short borrow + rebalancing costs** and confront the only thing that decides a short-vol book — the **crash tail** (skew, kurtosis, single-worst-day, drawdown). A deterministic synthetic control with a *planted carry knob* and *injected crashes* confirms the engine recovers a real carry, never invents one, and reproduces the steamroller by construction.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why the ETP melts down ×100,000, why shorting it is a steady carry — and why one bad day eats a quarter of a year, in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the short-carry book, HAC inference + a placebo null, costs with borrow, the skew/kurtosis/VaR/drawdown tail, contango conditioning, borrow sensitivity, and a synthetic carry-knob / crash control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`vxx_roll_decay/`](vxx_roll_decay/). The shorted vehicle is **VIXY**, an explicit **VXX-equivalent** (same constant-maturity VIX-futures index, longer tape), not VXX itself. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
