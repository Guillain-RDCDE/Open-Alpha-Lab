# Study 78 — Crossed-Wires

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Cross gross **+12.65 bps/trade**, HAC *t* = **+0.94** — right sign, but well below the |*t*| ≥ 2 inference bar; only ~97 trades/instrument over 5 years, a chronically underpowered test. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | A WEAK signal that never clears inference is untradable at any cost level; low turnover (~19/yr) spares expenses but cannot rescue a noise estimate. |
| **Beats a coin?** | ![Not--Supported](https://img.shields.io/badge/Not_Supported-8b949e?style=flat-square) | Cross vs random delta = +14.67 bps — directionally positive but statistically silent (t = +0.94); the 95% win-rate is pure exit geometry (skew −7.0), not evidence of skill. |

> **In one sentence:** the MACD(12,26,9) daily signal-line crossover shows a ghost of a positive edge (+12.65 bps, HAC *t* = +0.94) that is impossible to distinguish from noise at ~97 trades per instrument — the indicator zoo's verdict: another lagging coin-flip, completing the SMA (#72), RSI (#15), and TSI (#08) desk teardowns.

## What we tested

A cornerstone of retail trading culture and algorithmic strategies alike: on the daily chart, when
the MACD line (EMA(12) − EMA(26)) crosses above its signal line (EMA(9) of MACD), go long; when
it crosses below, go short or flat — *"the indicator synthesises the medium-term trend and the
signal-line cross is a high-probability direction call."* We take that literally: run it as a
barrier backtest with **symmetric ±1 ATR** exits (the only direction-fair payoff) across six
liquid daily tapes (SPY, QQQ, IWM, AAPL, TSLA, NVDA, five years), pin it against a
**random-direction control** on identical entries, expose the fixed-tick win-rate trap (95% wins
at skew −7), and compare directly to Study 72 (SMA(5/10) on 5-minute bars, t = −1.12) —
completing the desk's EMA-crossover indicator zoo. A deterministic synthetic daily tape with
tunable AR(1) momentum serves as the positive control (the engine harvests momentum when it
exists; the real tape hasn't enough).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what MACD is, the win-rate trap, the fair bet vs a coin, why the low trade count matters |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | per-instrument HAC *t*, forward-return horizon sweep, barrier-ratio win-rate identity, cost sweep, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`crossed_wires/`](crossed_wires/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
