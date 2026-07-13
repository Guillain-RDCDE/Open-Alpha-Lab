# Study 770 — Concert-Economy 🎤

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | The "rally into festival season" is a statistical zero: 1-month run-up AR **+0.72%**, *t* = **+0.29**, placebo *p* = **0.52**; the 2-month window is *negative*; the jackknife *t* never leaves [−0.44, +0.65]. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Nothing to trade. The calendar-known run-up clears *t* ≥ 2 neither gross nor net at 5 or 10 bps — best case *t* = +0.29. |
| **Front-runs the season?** | ![Not supported](https://img.shields.io/badge/Not_supported-8b949e?style=flat-square) | Q3 really is ~37% of revenue, but the stock doesn't anticipate it. The one big move (+8.5% *in-season*) comes *after* the season starts, sits inside a random 4.5-month window's luck cloud (*p* = 0.16), and is mostly LYV's **1.35 beta** over a long horizon — beta, not front-running. |

> **In one sentence:** festival season is a real, huge, entirely predictable revenue event
> for Live Nation — and precisely because everyone can see the calendar, the stock does
> *not* rally into it, the run-up is indistinguishable from noise, and the only sizeable
> move is high-beta drift that happens *during* the season, not before.

## What we tested

The recurring "concert economy" trade — a fixture of retail finance every spring — says
you should buy Live Nation (`LYV`) *ahead of* festival season, because the market will
front-run the summer touring quarter (its biggest by far). The steelman is real: Live
Nation's own [10-Q filings](docs/references.md) put ~37% of annual revenue in Q3. We
hardcode all 20 Coachella editions 2006→2025 (2020–21 COVID-cancelled) using each year's
announced weekend-1 Friday — a *calendar-known, zero-look-ahead* anchor — and measure
`LYV − SPY` total-return abnormal returns over the 1- and 2-month run-up into the
festival, with a random-window placebo, a jackknife, a costed trade, and an in-season
"does it front-run the revenue?" check.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the claim, why the seasonality is genuinely real, the flat run-up, and why the one big number is a beta mirage |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the one-sample-*t* battery, the placebo, the jackknife, the in-season beta decomposition, the synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`concert_economy/`](concert_economy/). The Coachella calendar is hardcoded from
Wikipedia; the touring-revenue seasonality is a **labelled proxy** reconstructed from
Live Nation's SEC 10-K/10-Q filings (never a live tape). **Beta named** on the Signal
axis: LYV's 1.35 beta to SPY is why the raw in-season out-performance is not alpha.
**Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
