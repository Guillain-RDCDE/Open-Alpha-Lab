# Study 128 — Keltner-Channel

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Breakout arm earns **+81 bps/trade** but a random same-date entry earns **+117 bps** — the filter makes it **35 bps worse**. Reversion arm beats random by only **+6 bps** (noise). No arm clears the inference bar vs its matched control; both positive gross means are captured equity drift. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | No channel-alpha above the unconditional drift exists to trade. Turnover is low (~6–23 entries/yr/ticker), so costs are not the primary concern — the absence of signal above the random baseline is. |
| **Breakout vs Reversion?** | ![Both_noise](https://img.shields.io/badge/Both_noise-8b949e?style=flat-square) | Two contradictory folk claims, one channel. The breakout framing (momentum) earns less than a coin-flip entry. The reversion framing earns only negligibly more. Both are resolved empirically: both are noise. |

> **In one sentence:** the Keltner Channel's two contradictory folk rules — trade the breakout above the upper band, and buy the dip below the lower band — both fail to add alpha over a random same-date entry once 21 years of equity drift are controlled for.

## What we tested

A widely-used volatility envelope built as EMA(20) ± 2 × ATR(10) on daily closes. Two competing
folk claims are made: (1) *"close above the upper band signals a strong trend — buy"* (the
breakout/momentum framing); (2) *"close below the lower band signals a stretched move — buy
the snap-back"* (the mean-reversion framing). Both cannot be right simultaneously. We implement
both as fixed 20-bar time-exit long trades, pin each against a **random-entry control** matched
in trade count (which strips out the instrument's unconditional drift), and run the comparison
across five liquid daily tapes (SPY, QQQ, IWM, GLD, EEM) over 21 years. A deterministic
synthetic tape with tunable AR(1) momentum serves as the positive control — confirming both
arms find their respective edges when planted structure exists, and that the real tape has neither.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the channel visualised, the two contradictory rules, the random-control reveal, why both arms look profitable (equity drift) |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | per-instrument HAC *t*-stats, signal-vs-random delta table, the contradiction formalised, the synthetic momentum sweep as positive control, cost sweep |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`keltner_channel/`](keltner_channel/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
