# Study 142 — Split-Drift

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Post-effective-date returns are flat to negative vs a matched baseline: 12m mean +5.1% (HAC *t* = +0.99, not significant); split events underperform the same-basket no-split baseline by −23.6% at 12m (HAC *t* = −4.56). No horizon clears \|*t*\| ≥ 2. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | ~2 large-cap events/year in a 30-stock basket; no significant gross edge; below-baseline returns at every horizon from 1m to 12m. |
| **Post-effective drift?** | ![Absent](https://img.shields.io/badge/Absent-8b949e?style=flat-square) | Split events trail the matched same-stock baseline at 1m (−3.6%), 3m (−5.3%), 6m (−12.4%), and 12m (−23.6%), all with \|*t*\| ≥ 2. The market prices splits before the ex-date. |

> **In one sentence:** the post-split drift documented by Ikenberry, Rankine & Stice (1996) was measured from the *announcement* date — by the time the effective (ex-split) date arrives the market has already reacted, and split stocks subsequently underperform their own same-basket baseline at every horizon tested.

## What we tested

The 1996 Ikenberry-Rankine-Stice finding that stocks drift up +7.9% in the year following a split **announcement** is one of the more credible event-study anomalies: splits are costly signals of management confidence in future earnings, and the paper argued the market under-reacts. We steelman this: using `yfinance` `.splits` on 30 large-cap US stocks (2000–2025), we extract 41 split events, compute buy-and-hold forward returns at 1/3/6/12-month horizons starting the day after the effective date, and compare to a matched no-split baseline (same tickers, non-split windows). A Newey-West HAC t-stat decides whether the difference is real. **One honest caveat is stated prominently:** yfinance gives the *effective* date, not the *announcement* date (which precedes it by 3–6 weeks). We test a strictly weaker window; the absence of post-effective drift is consistent with the original signal being fully priced by announcement.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the claim, the horizon sweep, why split stocks underperform their own baseline, the event-frequency problem, the effective-vs-announcement caveat |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | per-horizon HAC t-stats, split vs baseline comparison table, synthetic positive control, cost sweep, survivorship accounting |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`split_drift/`](split_drift/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
