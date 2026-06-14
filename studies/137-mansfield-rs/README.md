# Study 137 — Mansfield-RS

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Stage-2 entry excess return +29.4 bps/trade (13w), HAC *t* = **+0.72** — far below the inference bar; underperforms a random-entry control by **−88.7 bps/trade** across all hold periods and all 15 stocks. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Gross excess is not significant; costs do not rescue a signal that never had a t-stat above 1. Low turnover (~2.2 entries/stock/year) means costs aren't the killer — the signal is. |
| **Timing adds nothing?** | ![Confirmed](https://img.shields.io/badge/Timing_adds_nothing-Confirmed-8b949e?style=flat-square) | Stage-2 entries underperform random entries at every hold period (4w, 8w, 13w, 26w) and in-Stage-2 holding earns *less* next-week excess than out-of-Stage-2 (+5.7 vs +9.2 bps/week). |

> **In one sentence:** Weinstein's Stage-2 Mansfield-RS filter selects momentum stocks correctly but enters too late — the transition into Stage 2 arrives after the initial surge, leaving Stage-2 entries below both the statistical bar and the random-entry baseline.

## What we tested

Stan Weinstein's *Secrets for Profiting in Bull and Bear Markets* (1988) prescribes a
two-condition filter: buy stocks where (1) price is above a rising 30-week SMA, and (2)
Mansfield Relative Strength — the stock's price-to-30w-SMA ratio divided by the
benchmark's ratio — is positive.  Together, these conditions identify stocks in an
uptrend that are also outperforming the market on a trend-adjusted basis: Weinstein's
"Stage 2" leaders.  We test this steelmanned: Stage-2 *entry* signals (the transition
from Stage 1 to Stage 2) on a 15-stock large-cap basket, measuring 13-week forward
excess return vs SPY, compared against a random-entry control on the same stocks.  A
deterministic synthetic tape with tunable RS momentum serves as the positive control,
confirming the engine recovers an edge *when RS momentum is actually present*.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what Stage 2 means in plain English, the "late arrival" trap, fair comparison vs a random-entry coin |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | Mansfield RS formula, HAC t-stats, per-stock breakdown, hold-period sweep, synthetic positive control, survivorship-bias discussion |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`mansfield_rs/`](mansfield_rs/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
