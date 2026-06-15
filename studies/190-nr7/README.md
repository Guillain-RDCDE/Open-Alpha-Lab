# Study 190 — NR7

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Mixed](https://img.shields.io/badge/Mixed-dab617?style=flat-square) | Range-expansion claim **refuted** (ratio = 0.92, *t* = −11.4 — NR7 is followed by a *narrower* day, not a wider one). Breakout direction shows pooled gap vs random control of +9.8 bps gross, but collapses on SPY at any realistic cost (+0.4 bps at 5 bps round-trip, *t* = +0.14) and is not subsample-stable. |
| **Tradability** — does it survive costs, capacity, scale? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | Breakout survives costs only in volatile single stocks (TSLA, NVDA) that distort the pooled result; the liquid index proxy (SPY) has no tradable edge at realistic costs. No subsample stability. |
| **Volatility claim?** | ![Refuted](https://img.shields.io/badge/Volatility_claim%3F-Refuted-c0392b?style=flat-square) | The defining premise — that NR7 is a "coiled spring" that releases the next day — is strongly rejected on 26 years of daily data across all six tickers. |

> **In one sentence:** Crabel's NR7 "coiled spring" uncoils the wrong way — volatility stays contracted after the narrowest day — and any breakout directional edge on the liquid index (SPY) disappears at 5 bps round-trip cost, leaving a fragile single-stock anomaly not a tradable system.

## What we tested

Tony Crabel (1990) defined the **NR7**: a daily bar whose high-minus-low range is the
*narrowest* of the last 7 days, signalling volatility contraction that "precedes expansion".
The recipe: buy a break above the NR7 high on the next open, sell a break below the NR7 low.
We test two sub-claims on six daily tapes (SPY, QQQ, IWM, AAPL, TSLA, NVDA) over
2000–2026 (~26.5 years, 5,574 NR7 signals pooled):

1. **Range expansion** — next-day range is wider than the rolling baseline after an NR7.
   *Result: the opposite is true — range = 0.92× baseline, t = −11.4.*
2. **Directional breakout** — buying the NR7-high break (or selling the NR7-low break)
   beats an unconditional baseline and a random-day control.
   *Result: Mixed — pooled gross gap of +9.8 bps vs random days, but +0 bps on SPY at 5 bps cost.*

A deterministic synthetic tape with a tunable range-expansion knob serves as the positive
control, confirming the engine recovers an edge *when one is planted*.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the NR7 recipe, the coiled-spring test result, the breakout vs random-day comparison in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | per-ticker HAC *t*, range-expansion refutation, bootstrap Sharpe CI, cost sweep, subsample robustness check, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`nr7/`](nr7/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
