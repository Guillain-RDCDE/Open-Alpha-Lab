# Study 116 -- Power-Hour

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) -- see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** -- is the effect statistically real? | ![MIXED](https://img.shields.io/badge/MIXED-dab617?style=flat-square) | Follow-morning earns **-1.83 bps/session**, HAC *t* = **-3.62** (significantly *wrong* direction). Fade (contrarian) earns **+1.83 bps**, *t* = **+3.62** -- a real reversal effect, opposite the claim. |
| **Tradability** -- does it survive costs, capacity, scale? | ![FRAGILE](https://img.shields.io/badge/FRAGILE-dab617?style=flat-square) | Fade breakeven ~1 bps round-trip; survives at 0.5 bps (*t* = +2.63) but not at 1 bps (*t* = +1.64). Retail traders (>1 bps) cannot access it. |
| **Continuation claim?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | Pool corr(morning, last) = **-0.031** (IWM: **-0.086**, *t* = -2.31) -- the last bar reverses the morning, it does not follow it. |

> **In one sentence:** the "power hour" continuation story is directionally backwards -- the last regular-session bar on SPY, QQQ, and IWM *reverses* the morning's direction rather than following it, and the only real (but thin) edge is the contrarian fade that lives below the typical retail cost threshold.

## What we tested

A staple of day-trading content: *"The last hour (15:00-16:00 ET) is the power hour -- smart money and institutions drive the day's established trend into the close. Follow the morning's direction for the last hour."* We take that literally: compute `morning_ret` (09:30 open -> 14:30 close) and `last_ret` (15:30 open -> 15:30 close) from ~723 sessions of 1-hour SPY/QQQ/IWM bars, test the Pearson correlation between them, and pin the follow-morning signal against a **random-direction control** on the same sessions. The academic cousin (Gao et al. 2018 first-half-hour -> last-half-hour momentum on SPY) is noted but does not survive our 2023-2026 sample. A deterministic synthetic tape with a tunable `continuation` knob serves as the positive control.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the claim, the morning-vs-last correlation in plain language, the fair bet vs a coin, why the fade barely survives costs |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | per-instrument Pearson correlation, HAC t-stats on all three arms, the fade cost sweep, the synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`power_hour/`](power_hour/). **Not investment advice** -- research & education. See [LICENSE](../../LICENSE).*
