# Study 890 — Sector Risk-Parity ⚖️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does equal-*risk*-weighting the GICS sectors improve the excess-of-cash Sharpe (and drawdown) vs cap-weight SPY? | ![Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square) | The claim splits. The **risk reduction is real**: over 2007–2026 inverse-vol cut volatility to **17.8%** (SPY 19.9%) and max drawdown to **−49.6%** (SPY −55.2%), beating SPY by **+4.6 pp in 2008** and **+13.3 pp in 2022**. But the **excess-of-cash Sharpe advantage does not clear the bar**: it is **+0.002** on the nine-sector panel (paired-bootstrap 95% CI **[−0.10, +0.11]** straddles zero, NW *t* = −1.12), **−0.095** on the tech-led 2018– panel, and it *flips sign* across eras (**+0.064** in 2007–2015 → **−0.064** in 2016–2026) — the fingerprint of diversification, not alpha. A 20-seed synthetic control recovers a *planted* advantage cleanly (fires on **0/20** nulls). *Short history: XLC only from 2018-06 → the eleven-sector panel is ~8y — named here.* |
| **Tradability** — can you get paid for it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Costs are trivial (**1.4 bps/yr**, ~48% turnover × 3 bps on ultra-liquid ETFs) and capacity is effectively unlimited — but there is **no Sharpe edge to harvest**. Unlevered you simply earn *less* than SPY (**+11.2%** vs +12.3%/yr) for the smoother ride, and levering back to SPY's vol (just 1.12×) reproduces SPY (**Sharpe 0.552 vs 0.553**). The promised risk-adjusted pickup is a mirage; what remains is a lower-beta equity book. |

> **In one sentence:** equal-risk-weighting the eleven GICS sectors buys a **genuinely smoother
> ride** — lower vol, milder drawdowns, wins in every real bear year — but **no risk-adjusted
> edge over cap-weight SPY** (the Sharpe advantage is ~zero and flips sign by regime), and
> nothing bankable survives once you notice it just earns less than the index.

## What we tested

Risk parity applied **within equities**: weight the eleven SPDR Select-Sector ETFs by
**inverse volatility** or full **equal-risk-contribution** (ERC), rebalance **quarterly** with
realistic weight drift and a 3 bps one-way cost, and race the result **excess of BIL cash**
against **cap-weight SPY** on Sharpe, drawdown and a calendar-year table. Two real panels
(yfinance daily total-return): an **eleven-sector** headline (2018-06 → 2026-06-30, short because
XLC is young) and a longer **nine-sector** panel back to **2007-05**, with a paired
Sharpe-difference bootstrap, a Newey-West *t*, an era cut, a levered-to-SPY-vol timer, and a
20-seed synthetic positive control. Short history is named on the **Signal** axis. **Dedup:**
[68-all-weather](../68-all-weather/) does risk parity *across asset classes* (not within
equities); [225-sector-rotation](../225-sector-rotation/) and [28-carousel](../28-carousel/)
*time / rotate* into sectors on a signal (this study forecasts nothing, holding all eleven
always); [94-level-pegging](../94-level-pegging/) equalises *dollars* (1/N), whereas this study
equalises *risk*. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why cap-weight over-concentrates risk, why risk-parity smooths the ride — and why the Sharpe "advantage" flips sign every era |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the excess-vs-excess Sharpe race, the paired block-bootstrap CI, the Newey-West *t*, the era cut, the costed & levered timers, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`sector_rp/`](sector_rp/). Sector ETF total-return prices via yfinance, cached under
the study's own `_cache/`; every Sharpe is excess of BIL cash. **Not investment advice** —
research & education. See [LICENSE](../../LICENSE).*
