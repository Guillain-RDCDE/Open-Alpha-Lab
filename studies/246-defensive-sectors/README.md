# Study 246 — Defensive-Sectors Leadership

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Q1−Q5 21-day spread **+43.6 bps/month** but HAC *t* = **+0.79** — far below the bar of 2; the 1-day spread is *negative* (wrong direction); non-monotone quintile pattern at every horizon. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Timing overlay (cash in top-20% alert periods) *lowers* Sharpe 0.432→0.430 even before costs; gross alpha = −0.77 bps/day (*t* = −1.04); 480 switches accelerate deterioration. No break-even cost exists. |
| **Coincident or forecast?** | ![Coincident](https://img.shields.io/badge/Coincident--not--forecast-8b949e?style=flat-square) | Combined XLP+XLU outperformance *co-occurs* with market stress but does not statistically *precede* weak SPY returns at any tested horizon. Adding XLP to the utilities canary (Study 131) weakens — not strengthens — the signal (*t* 0.79 vs 1.58). |

> **In one sentence:** the combined XLP+XLU/SPY defensive-leadership signal is weaker than the single-sector utilities canary, fails on direction at the 1-day horizon, produces a non-monotone quintile pattern, and destroys value even at zero trading cost.

## What we tested

The folk claim: when *both* consumer staples (XLP) and utilities (XLU) simultaneously
outperform SPY on a combined relative-strength basis (rising 20-day momentum of the
average XLP+XLU / SPY log-ratio), multiple defensive sectors are rotating together — a
stronger risk-off warning than a single-sector canary, and a reliable signal to reduce
equity exposure ahead of a drawdown. We test this literally: sort trading days by the
rolling percentile rank of the combined 20-day defensive RS momentum into five quintiles
(Q1 = SPY outperforming; Q5 = defensives strongly outperforming) and test whether there
is a **monotone descent** in 1-day, 5-day, and 21-day forward SPY returns from Q1 to Q5,
with HAC inference on the Q1−Q5 spread. We also run a binary timing overlay (go to cash
in the top-20% alert bucket) vs unconditional buy-and-hold, sweep costs at the rule's
natural turnover, and run a synthetic positive control to confirm the machinery can detect
a planted signal. Real tape: XLP, XLU, and SPY daily (1998-12-23 to 2026-06-16,
n = 6,911 days, ~27.5 years).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the folk recipe, the quintile results in plain language, why adding XLP weakens not strengthens the utility canary, the coincident vs forecast distinction |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC t-stats per quintile and horizon, the spread t-stat, regime Sharpe decomposition, cost sweep, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`defensive_sectors/`](defensive_sectors/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
