# Study 184 — Williams-Fractals

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Reversal gross **−4.09 bps/trade**, HAC *t* = **−1.26**; breakout **−0.06 bps**, *t* = **−0.01**. Both framings underperform a random-direction control; no instrument clears \|*t*\| ≥ 2. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Gross edge is already negative; at 5 bps a statistically significant loser (*t* = −2.80). Fires on ~27% of trading days — high turnover kills any residual hope. |
| **Beat a coin?** | ![No-8b949e](https://img.shields.io/badge/Beat_a_coin%3F-No-8b949e?style=flat-square) | The fractal signal performs *worse* than a random-direction entry on the same bars in both framings: the 5-bar pattern carries no directional information. |

> **In one sentence:** Bill Williams' 5-bar fractal marks a local swing extreme but carries no forward information — the reversal entry opens after any reversion has already occurred, and the breakout is a pure coin flip; both framings are statistically indistinguishable from noise and die quickly under costs.

## What we tested

Bill Williams defined fractals in *Trading Chaos* (1995): a **bearish fractal** is a daily bar whose high exceeds both neighbours two bars on each side; a **bullish fractal** has the lowest low in the 5-bar window. The pattern fires on ~27% of trading days, confirming its ubiquity. Two readings appear in retail literature: (1) **reversal** — fade the swing extreme, expecting a bounce; (2) **breakout** — buy when price closes above a prior bullish fractal high, sell when it closes below a bearish fractal low. We implement both with an honest 2-bar confirmation lag (the pattern is only *known* two bars after the centre bar), pin each against a **random-direction control** on identical entries, sweep hold periods and costs, and test over six liquid daily tapes (SPY, QQQ, IWM, AAPL, MSFT, GLD) covering 10 years.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a fractal is, the two recipes, the confirmation-lag trap, the honest coin-flip comparison |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | per-instrument HAC *t*, hold-period sweep, cost sweep, the synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`williams_fractals/`](williams_fractals/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
