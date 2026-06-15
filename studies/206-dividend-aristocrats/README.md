# Study 206 — Dividend-Aristocrats

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Excess return **−3.48%/yr**, HAC *t* = **−1.63**; OLS alpha **−0.96%/yr**, *t* = **−0.45**. Bootstrap excess-Sharpe CI [−0.987, +0.088], 95% of resamples negative. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | NOBL net CAGR **+10.10%** vs SPY **+14.36%**; Sharpe **0.607** vs **0.782**. Total return since inception: **253%** vs **452%**. Random-mix control beats pure NOBL in 100% of seeds. |
| **Crash shield?** | ![Busted](https://img.shields.io/badge/Crash_shield%3F-Busted-8b949e?style=flat-square) | Negative protection in all 4 major drawdowns: −7.6 pp in COVID, −13.4 pp in 2022, −15.3 pp in Apr 2025. |

> **In one sentence:** the Dividend Aristocrats (NOBL ETF) delivered 253% vs the S&P 500's 452% since 2013, with a lower Sharpe despite lower vol, negative crash protection in every major bear market, and no statistically detectable quality alpha — a quality story that costs 0.35%/yr to hold and hasn't paid off.

## What we tested

The classic quality pitch: S&P 500 members with 25+ consecutive years of dividend growth (the Dividend Aristocrats, tracked by NOBL since Oct 2013) earn a quality/dividend-growth premium and protect capital in downturns. We test NOBL vs SPY on full live history (2013–2026, n = 3,188 daily total-return closes), computing CAGR, Sharpe, OLS alpha/beta, crash-episode protection, and a random-mix control. A deterministic synthetic tape with a tunable alpha serves as the positive control.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | cumulative wealth chart, crash-shield test, random-mix control in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC t-stat, bootstrap Sharpe CI, OLS alpha/beta, per-crash episode table, ER drag, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`dividend_aristocrats/`](dividend_aristocrats/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
