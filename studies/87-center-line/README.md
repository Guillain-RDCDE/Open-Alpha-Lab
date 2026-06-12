# Study 87 — Center-Line

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | VWAP-fade gross **−0.35 bps/trade**, HAC *t* = **−1.16**; no edge over a random-direction control (Δ = −0.09 bps), every instrument \|*t*\| ≤ 1.25. No threshold (0.5–2.0 ATR) unlocks a signal. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Negative gross at ~49 trades/day → a *significant loser* (*t* = −4.5 at 1 bp), ≈ **−166%/yr** net; no positive break-even cost. |
| **Beats a coin?** | ![Not_supported](https://img.shields.io/badge/Not_supported-8b949e?style=flat-square) | The VWAP is a centre of mass by construction; fading deviations from it does not beat a random-direction entry — the "gravity" is a mathematical identity, not a forecasting edge. |

> **In one sentence:** the VWAP-fade fires on ~63% of all intraday bars, earns −0.35 bps gross per trade and does not beat a coin, because the session VWAP is a volume-weighted average of price — mean-reversion toward it is an accounting identity, not a signal.

## What we tested

A pervasive practitioner claim: *"the session VWAP is the market's centre of gravity — whenever price stretches more than an ATR away, fade it back, collect the reversion."* We take that literally and ask whether the VWAP deviation carries **directional information**: we run the fade as a barrier backtest with **symmetric ±1 ATR** exits (the only direction-fair payoff) across eight liquid 5-minute tapes (SPY, QQQ, IWM, AAPL, TSLA, NVDA, ES, NQ, ~60 days), pin it against a **random-direction control** on identical entries, sweep the deviation threshold (0.5, 1.0, 2.0 ATR) to check whether a sharper filter helps, and sweep costs at the rule's natural turnover (~49 trades/day/ticker). A deterministic synthetic tape with tunable intra-session mean-reversion serves as the positive control.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the VWAP recipe, the "gravity" intuition, why it fires constantly, the fair bet vs a coin, why costs bury it |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | per-instrument HAC *t*, bootstrap Sharpe CI, the threshold sweep, the cost/turnover sweep, the synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`center_line/`](center_line/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
