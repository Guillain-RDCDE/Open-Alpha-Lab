# Study 107 — Stochastic-Oscillator

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Zone-filtered gross **−17 bps/trade**, HAC *t* = **−1.39**; coin beats the stochastic by 23 bps/trade, every instrument \|*t*\| < 2. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Negative gross expectancy before costs; no positive break-even cost exists; costs simply deepen an already-losing bet. |
| **Beats a coin?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | A random-direction entry on the same bars earns **+6.19 bps/trade** vs **−17.19 bps** for the stochastic — the famous zone filter actively *hurts*. |

> **In one sentence:** the Stochastic Oscillator's %K/%D crossover in oversold/overbought zones carries no directional information on daily equity data — over 10 years and six instruments it underperforms a coin by 23 bps per trade, and the negative gross means costs only make things worse.

## What we tested

George Lane's Stochastic Oscillator — the centerpiece of a generation of technical analysis books and charting courses — computes %K(14) as the close's position within the 14-day high-low range, then smooths it into %D(3). The folk rule: **buy when %K crosses above %D while both are below 20** (oversold), **sell when %K crosses below %D while both are above 80** (overbought). We take that literally and ask whether the zone-filtered cross actually predicts the next 5 days of price action. We run it as a fixed-horizon forward-return backtest across six liquid daily tapes (SPY, QQQ, IWM, AAPL, TSLA, NVDA, 10 years), pin it against a **random-direction control** on identical entry bars, test both the zone-filtered and zone-agnostic framings, and sweep hold periods (1/3/5/10 days) and costs. A deterministic synthetic tape with tunable mean-reversion serves as the positive control.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the recipe, what the stochastic is really measuring, the coin test in plain language, why the zone filter makes things worse |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | per-instrument HAC *t*, hold-period sweep, zone vs no-zone breakdown, cost sweep, the synthetic positive control, secular-uptrend bias dissection |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`stochastic_oscillator/`](stochastic_oscillator/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
