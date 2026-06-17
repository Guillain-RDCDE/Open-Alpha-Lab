# Study 279 — Geomagnetic-Storms

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

**Do geomagnetic storms depress returns (Krivelyova-Robotti)?**

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Calm-minus-stormy gap **+0.78%/month** (~9%/yr) in the predicted direction, permutation p = **0.046**, with a published mood mechanism — but the honest Newey-West HAC t-stat is **1.88** (< 2.0) and stays under 2.0 in every sub-period. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The lagged, costed long-calm / short-storm overlay earns **+2.4%/yr net** (Sharpe 0.22) vs **+8.7%/yr** for buy-and-hold (Sharpe 0.49). No edge survives borrow and the opportunity cost of sitting out the drift. |

> **In one sentence:** geomagnetic storms really do precede lower average returns, in the right direction and with a plausible mood channel — but the effect lands just shy of the |t| ≥ 2 bar and offers nothing tradable over simply owning the index.

## What we tested

The Krivelyova & Robotti (2003) storm-return effect: returns following high geomagnetic
activity are lower than returns following calm periods, via a mood-misattribution channel.
We hardcode a monthly geomagnetic **Ap index** in `data.py` (reconstructed deterministically
from the 11-year solar cycle, with the famous great storms of 1989/2003/2024), classify each
month as **stormy** (top quintile of activity), **calm** (bottom quintile), or **normal**, and
join with ^GSPC monthly **price** returns (1932–2025, 1,128 months). We measure the
calm-minus-stormy mean-return gap with a **Newey-West HAC** t-stat — the honest standard error
given that both storms and returns cluster in time — cross-checked with a Welch t-test and a
10,000-shuffle permutation test, across the full sample and sub-periods. We then run the actual
**lagged, costed** long/short overlay (one-month execution lag, one-way costs × NAV, borrow on
shorts) against buy-and-hold, and confirm the machinery with a synthetic positive control.

**Honesty ledger:** one-month execution lag; one-way costs × NAV + short borrow; price-only
returns (no dividends — named on the Signal axis); in-sample storm/calm thresholds (a mild
look-ahead, named on the Signal axis).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the mood mechanism, the regime bar-chart, the "right sign, t = 1.9" story, the un-tradable overlay |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | Newey-West HAC t-stat, permutation distribution, sub-period robustness, lagged net P&L, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`geomagnetic_storms/`](geomagnetic_storms/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
