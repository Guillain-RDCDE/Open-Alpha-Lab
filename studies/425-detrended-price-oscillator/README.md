# Study 425 — Detrended Price Oscillator 📉

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is there a tradable cycle? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | On SPY (1993–2026) the look-ahead-free DPO(20) long/flat rule nets **+0.43** Sharpe vs buy-and-hold's **+0.65** — it *gives up* **5.5%/yr** (daily excess-vs-excess HAC *t* = **−3.11**, significant but **wrong-signed**). Negative on all six tapes (SPY/QQQ/IWM/EFA/GLD/DBC). No positive edge exists. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | A structural loser **before costs** at every cost {0,1,2,5} bps and every window {10,20,30,50}. At ~10 trades/yr costs are not the cause — the rule is. No positive break-even cost. |
| **Isolates a tradable cycle?** | ![Busted](https://img.shields.io/badge/Isolates_a_tradable_cycle%3F-Busted-8b949e?style=flat-square) | Circular-shift permutation placebo *p* = **0.066** (real edge in the *left* tail) and no better than a plain SMA-cross (ΔSharpe −0.22 vs −0.13). A synthetic **planted cycle** proves the engine *would* find one (ΔSharpe +3.68) — the real market just has none. |

> **In one sentence:** the DPO does exactly what it promises — it strips the trend out of the price — and in a market that mostly trends *up*, trading the detrended residual is a machine for sitting out the trend: on six diverse tapes the long/flat rule underperforms buy-and-hold by 5–8%/yr (HAC *t* = −3.11 on SPY, the wrong way), its timing beats a random shift only ~7% of the time, and it is no better than an SMA crossover — while a synthetic planted cycle confirms the engine would light up if a tradable rhythm were actually there.

## What we tested

We compute the standard Detrended Price Oscillator, `DPO(t) = Close(t) − SMA₂₀(Close)(t)` (the **look-ahead-free** form — the textbook centered version peeks at its own future, which we demonstrate and refuse to trade), and turn it into a daily **long/flat** (and **long/short**) timing rule: long below the −1σ band, flat above the +1σ band, with a one-day execution lag. We race the rule's **net** Sharpe against buy-and-hold, both **excess of cash**, with a Newey–West HAC *t* on the daily difference, a 2,000-draw circular-shift permutation placebo on the position, a cost and period sweep, and a head-to-head against the obvious simpler benchmarks (SMA crossover, MACD) so the "detrending is special" claim is actually tested. A deterministic synthetic control with a *planted sinusoidal cycle* confirms the engine recovers a cycle edge when one exists — and reads ~zero when it doesn't. Data: Yahoo daily **adjusted** bars, full histories, as-of **2026-05-31**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what "detrending" means, why removing the trend fights the one thing that pays, the head-to-head vs buy-and-hold across six assets, and why the rule loses before costs — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | DPO(20) long/flat & long/short, excess-vs-excess Sharpe race, HAC edge *t*, circular-shift permutation placebo, cost/period sweeps, the centered-DPO look-ahead trap, the SMA/MACD benchmark race, and a synthetic planted-cycle positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`detrended_price_oscillator/`](detrended_price_oscillator/). Indicator is the look-ahead-free DPO; the centered/peeking variant is shown only to expose the trap. Basket is **liquid ETFs/indices** (no single-name survivorship). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
