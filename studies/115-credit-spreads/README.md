# Study 115 — Credit-Spreads

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | HYG-IEF stress regime predicts lower 5-day SPY forward returns (−8.9 bps vs calm), but HAC *t* = **−0.71** — well below the |*t*| ≥ 2 bar.  HYG/LQD proxy has the wrong sign (+5.4 bps).  Literature support is real; the ETF-proxy signal is not. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The regime flip is a slow-moving binary flag on a noisy 20-day ETF-return proxy.  No profitable rule is demonstrated; both stress and calm regimes have positive expected SPY returns. |
| **Beats buy-and-hold?** | ![Not--Supported](https://img.shields.io/badge/Not--Supported-8b949e?style=flat-square) | SPY has positive expected returns in all regimes; the spread flag does not identify windows where defensive positioning is statistically warranted. |

> **In one sentence:** the ETF-based credit-spread proxy (HYG underperformance vs IEF/LQD) does point in the right direction but the signal is far too weak to clear the inference bar — credit and equity stress are predominantly coincident, not predictive, at this proxy resolution.

## What we tested

The "credit leads equities" narrative is one of the most-cited macro timing rules: when high-yield credit risk rises (HY spreads widen), equity stress follows.  We take this literally and build two ETF-based credit-spread proxies — HYG minus IEF (rolling 20-day return, capturing HY underperformance vs duration-matched Treasuries) and HYG minus LQD (stripping duration to isolate the credit-risk premium) — and test whether a widening-spread regime predicts lower SPY forward returns than a tightening regime.  The baseline is the unconditional SPY drift.  FRED OAS data is unavailable in this environment, so we use ETF price returns as the proxy and say so.  The test covers 2010-01-04 to 2026-06-12 (4,111 observations), spanning four major credit cycles.  A deterministic synthetic tape with a tunable credit-spread lead-lag serves as the positive control.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what credit spreads are, the ETF proxy trick, the regime-conditional test in plain language, why credit is coincident not predictive |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC t-stats, regime-conditional distributions, rolling-median threshold, forward-return horizon sweep, the ETF-vs-OAS gap, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`credit_spreads/`](credit_spreads/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
