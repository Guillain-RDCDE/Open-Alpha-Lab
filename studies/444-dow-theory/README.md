# Study 444 — Dow Theory 🚂

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the confirmation predict? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | Timing the Industrials with the Transports-confirmation rule **underperforms** buy-and-hold: the active daily spread is **negative** (HAC *t* = **−1.94** latched, **−3.14** on the raw flag), and the direct content test of confirmed-vs-non-confirmed next-day returns is **−3.67%/yr (t = −0.41)** — never clearing the **t ≥ 2** bar in the believers' direction, unchanged in sign over **34 years** of index history. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The only thing it buys you is a smaller drawdown (**−22.5%** vs **−51.9%**) — pure de-risking from sitting in cash 38% of the time. It surrenders **4.2 points of CAGR** (5.56% vs 9.75%) for a Sharpe that is **no better** than buy-and-hold (**0.599** vs **0.612**). No residual edge to scale. |
| **Does the I/T confirmation work?** | ![Busted](https://img.shields.io/badge/I%2FT_confirmation%3F-Busted-8b949e?style=flat-square) | A **random** regime with the same on-fraction and persistence beats the real confirmation regime **~20% of the time** (placebo *p* = **0.197**). The agreement between the two averages carries no detectable information beyond *being in cash sometimes*. |

> **In one sentence:** Dow Theory's famous "the Transports must confirm the Industrials" rule, encoded as a mechanical higher-high/lower-low regime and traded against buy-and-hold, adds **negative** timing value (active HAC *t* = −1.94), matches buy-and-hold's Sharpe only by giving up 4.2 points of CAGR for a smaller drawdown — and a random cash-timing regime with the same persistence beats it 20% of the time, so the confirmation itself is doing none of the (drawdown-only) work.

## What we tested

Dow Theory is partly subjective, so we steelman it into the **tightest mechanical rule a proponent would accept**: the primary uptrend is *confirmed* when **both** the Industrials (`DIA`) and the Transports (`IYT`) close at or above their **63-day trailing high**, latched ON as a "primary trend" until **both** close below their 63-day trailing low. We then trade the believers' defensive reading — **long the Industrials while confirmed, cash otherwise** — with a one-day execution lag and 2 bps per flip, and race it excess-of-cash against buy-and-hold. The Signal axis charges a **HAC one-sample t** on the active daily spread plus a **content test** (confirmed-vs-non-confirmed forward returns) that strips the cash-drag; a **random-regime placebo** matched to the rule's on-fraction and persistence asks whether the *confirmation* matters or just *being in cash*. A deterministic synthetic control with a *planted* confirmation edge proves the harness can detect one when it's there. Robustness spans costs (0–10 bps), trailing-high windows (21–200 days), and the 34-year `^DJI`/`^DJT` price-only history.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what "the Transports must confirm" means, why it *feels* smart, the equity-curve race, why a smaller drawdown isn't an edge, and why a coin-flip cash rule does just as well — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | latched confirmation regime, active-spread HAC *t*, the cash-drag-stripped content test, a matched-persistence random-regime placebo, cost/lookback robustness, the 34-year index check, and a synthetic planted-edge power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`dow_theory/`](dow_theory/). Industrials = `DIA`, Transports = `IYT` (auto-adjusted ≈ total return); robustness on `^DJI`/`^DJT` (price-only). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
