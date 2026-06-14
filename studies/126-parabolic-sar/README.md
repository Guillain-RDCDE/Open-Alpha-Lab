# Study 126 — Parabolic-SAR

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | Pooled gross HAC *t* = **+2.47** (marginal), but **zero** individual ticker clears \|*t*\| ≥ 2 (range: +0.19 to +1.75). Pooling amplifies noise; no single market confirms the edge. |
| **Tradability** — does it survive costs, capacity, scale? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | Gross only ~**+2.1%/yr** at ~22 flips/yr; 2 bps per round-trip drops the *t* below 2; 5 bps leaves ~+1%/yr. No realistic all-in cost makes this investable. |
| **Whipsaw tax?** | ![Confirmed](https://img.shields.io/badge/Whipsaw_tax-Confirmed-8b949e?style=flat-square) | SAR generates ~22 round-trips/yr; range-bound periods multiply false flips that eat the thin trend premium before costs even apply. |

> **In one sentence:** Wilder's Parabolic SAR earns a barely-significant pooled tilt on ten years of daily data (+9.3 bps/trade gross, HAC *t* = +2.47), but no single instrument confirms it, 2 bps of cost wipes the significance, and ~22 flips/year leave nothing investable.

## What we tested

Wilder (1978) introduced the Parabolic SAR as a "stop-and-reverse" system: the trailing stop accelerates each bar the extreme extends (AF 0.02 → 0.20), flipping from long to short — or short to long — the moment price pierces it. The bet is that the acceleration mechanic tracks real trends tightly enough that the flip direction consistently predicts the next move, net of the whipsaws a choppy market inflicts. We take the canonical settings (AF init 0.02, step 0.02, cap 0.20), run a flip-triggered barrier backtest with **symmetric ±1 ATR(20)** exits (the only direction-fair payoff) across six daily tapes (SPY, QQQ, IWM, GLD, TLT, EEM, ten years), pin it against a **random-direction control** on the same flip dates, and sweep round-trip costs at the rule's natural ~22-flip-per-year turnover.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the acceleration mechanic in plain language, the whipsaw tax, the fair bet vs a coin, why costs kill it |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | per-instrument HAC *t*, the pooling caveat, cost sweep, synthetic positive control, regime analysis |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`parabolic_sar/`](parabolic_sar/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
