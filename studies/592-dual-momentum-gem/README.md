# Study 592 — Dual Momentum (GEM) 🌍

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — beats buy-and-hold with half the drawdown? | ![Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square) | *Real on the drawdown-halving · None on the beats-buy-and-hold.* Max DD **−20.7% vs SPY's −50.8%** (ratio 0.41 — better than half) is on the tape. The outperformance is not: active vs SPY **−1.59%/yr (HAC t = −0.64)** over 2002-2026, turning **significantly negative** ex-GFC (t = −2.02) and post-2013 (t = −2.96). No lookback in the 3/6/9/12 grid beats SPY; the 40-seed random-switching Welch t averages +0.26. No survivorship (index ETFs) — but the live-ETF tape starts 2002 and can't see the book's 1974-2011 backtest years. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Execution is trivially easy (3 mega-liquid ETFs, 1.47 switches/yr ≈ 3–16 bps/yr of cost, unlimited capacity) — and that's the trap: what deploys cheaply is **beta plus a 2008 story**. Since publication GEM has *cost* **−5.51%/yr vs SPY (t = −2.96)**, and its full-sample Sharpe (0.70) is matched by a no-timing **60/40 (0.72)**. The promised alpha is the mirage, not the plumbing. |
| **Decayed since publication (2013)?** | ![Confirmed](https://img.shields.io/badge/Decayed_since_2013%3F-Confirmed-8b949e?style=flat-square) | Pre-2013 active **+29.4 bps/mo** (t = +0.71, never significant) flips to **−45.9 bps/mo (t = −2.96)** after; the pre-vs-post difference clears the bar at **Welch t = +2.02**. Whipsawed in 2015-16, out for the 2020 v-recovery, lagging the 2023-25 bull. |

> **In one sentence:** on the live-ETF tape (2002-2026) Antonacci's GEM genuinely halves the drawdown (−21% vs −51%) — but it does **not** beat buy-and-hold (−1.6%/yr vs SPY, HAC t = −0.64; significantly negative once 2008 is excluded), its whole case is one great sidestep, and since the strategy was published it has underperformed SPY by 5.5%/yr — so **Mixed, and a tradable Mirage**.

## What we tested

The classic **Global Equities Momentum** decision tree on month-end data: if SPY's trailing
12-month return beats the cumulated T-bill return (^IRX, prior-month yield — no look-ahead),
hold the better of **SPY/EFA** by 12-month return; otherwise hold bonds (**AGG**, IEF-spliced
before Sept 2003). Signal from month-end closes, position earns the *following* month — exactly
one execution lag. We race GEM against SPY and a 60/40 (excess-vs-excess Sharpe, total-return
everywhere), put a **HAC t on the monthly active return**, run a **lookback placebo grid
(3/6/9/12)**, a **40-seed random-switching baseline** (GEM's own holdings, shuffled timing,
averaged Welch t), sub-period splits (is it all 2008?), and a 2013 publication-decay split.
A deterministic synthetic control (two-state Markov regimes, persistence knob) proves the
engine detects a planted trend (t = 2.86) and does not manufacture one under an i.i.d. null.
Distinct from [146-country-momentum](../146-country-momentum/) (cross-sectional country panel)
and [518-time-series-momentum](../518-time-series-momentum/) (futures TSM panel): GEM is the
**composite retail allocation strategy** built on top of those ideas. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what "dual momentum" means, the one chart where GEM shines (2008) and the decade where it doesn't, why "half the drawdown" is true and "beats the market" isn't — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC t on active returns, the lookback placebo grid, the 40-seed shuffled-timing baseline, sub-period + publication-decay splits, costs × switch turnover, and the regime-persistence synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`dual_momentum_gem/`](dual_momentum_gem/). The signal is Antonacci's official GEM tree (SPY-vs-T-bill absolute gate, SPY-vs-EFA relative pick, bonds otherwise); the myth-check is the 2013 publication split. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
