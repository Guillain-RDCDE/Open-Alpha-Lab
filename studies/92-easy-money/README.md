# Study 92 — Easy-Money 🎢

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the carry statistically real? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | VIXY decays at **−48.5%/yr** (a dollar at launch is worth **0.004¢** today). Shorting it 1× earns **+14.8%/yr**, Sharpe **0.57**, and the daily return clears the bar: HAC *t* = **+2.31**. The contango carry is genuine. |
| **Tradability** — does it survive costs, capacity, scale? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | Net of 5 bps + 300 bps/yr borrow the edge persists — but the payoff is **−1.66 skewed**, **+9.4 excess kurtosis**, worst day **−43%**, max drawdown **−92%**. Sized so the worst day costs ≤25% (**0.58×**) it *still* draws down **−72%** at the same Sharpe. One mis-sized spike (the leverage **XIV** ran in 2018) is terminal. |
| **Free money?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | A 0.57 Sharpe with −1.66 skew, fat tails and a −92% drawdown is **selling crash insurance**, not a free coupon. The carry is the *price* of the spikes you eventually pay for. |

> **In one sentence:** the VIX-contango carry is **real and large** — VIXY bleeds ~48%/yr and shorting it earns a HAC-significant ~15%/yr — but it is a **compensated risk premium, not free money**: a steady drip with a −43% worst day and a −92% drawdown that wiped out XIV in a single afternoon in 2018, tradable only at a tail-disciplined size so small the dream of "easy money" is **busted**.

## What we tested

The perennial retail trade, stated at full strength: *"VIX futures are almost always in contango, so a long-vol ETP like VIXY bleeds lower every single day — just short it (or hold an inverse like SVXY) and harvest the roll. It's near-free money, an almost risk-less carry."* We take it literally — short **VIXY** (total-return adjusted, so its **1:4 reverse splits** fold into one continuous NAV), 1× of capital, net of **5 bps** trade cost and **300 bps/yr** stock borrow, one-day execution lag — and confront it with the only thing that decides free-money-vs-risk-premium: **the tail**. We report the headline Sharpe *and* its skew, kurtosis, worst day and max drawdown, then solve the leverage that survives the worst historical day. A deterministic synthetic vol-ETP tape — a planted steady decay **plus** rare upward jump-spikes vs a no-carry/no-jump tape — is the positive control (the short earns a drip *and* shows a fat left tail there; on the null it earns nothing).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the relentless VIXY decay, the carry equity curve, the 2018/2020/2024 cliffs that erase years of gains, why "free money" is a trap |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC *t* on the carry, the return distribution's left tail (skew/kurtosis), the survivable-sizing maths, the risk-premium reading |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`easy_money/`](easy_money/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
