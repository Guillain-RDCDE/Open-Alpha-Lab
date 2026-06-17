# Study 284 — Equinox-Effect

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

**Do equinoxes and solstices mark turning points?**

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | All 393 events average **−3 bps** abnormal on the event trading day (event-day win-rate 49.4%, *below* the ~54% base rate), HAC t = **−0.54**, perm p = **0.61**; the most extreme slice (autumn equinox, −22 bps) is **below |t| = 2 even raw** and dies under Bonferroni (7 slices → 0.50). n ≈ 99 per season is too small to detect anything below ~20 bps. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | No tradable vehicle — a four-times-a-year, single-day overlay with one-way costs (shorts pay borrow) is dominated by passive buy-and-hold; there is no edge to monetise. |
| **Busted?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | Markets have been said to "turn" with the Sun's calendar since the Gann era, but on 98 years of ^GSPC the equinox/solstice day is indistinguishable from any other day. |

> **In one sentence:** measured honestly against the market's ~3 bps/day drift, the S&P 500 does nothing unusual on equinox or solstice days — the "astronomical turning point" is folklore (and the one almost-interesting slice is just September weakness in disguise).

## What we tested

We compute every equinox and solstice 1928–2026 from Meeus' algorithm (verified to
the minute against the USNO seasons table) and hardcode them in `data.py`, tagged by
`kind` (equinox/solstice) and `season`. We align ^GSPC daily returns into a symmetric
event window, define **event day 0** as the first trading day on/after the
astronomical instant (one built-in execution-lag day), and test the mean **abnormal**
event-day return (raw minus the full-sample daily drift) with a Newey-West (HAC)
t-stat and a 10,000-draw permutation test. We slice by kind and the four seasons and
Bonferroni-correct for the seven simultaneous tests. A synthetic positive control
confirms the engine finds a planted event drag; the real tape has none.

Survivorship: ^GSPC is the index itself (no constituent survivorship), but **price-only**
(no dividends) — the daily drift is the honest baseline.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the omen, the base-rate trap, the event-day chart, the season split, the verdict in plain English |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | event-study CAR, HAC t-stat, permutation distribution, the autumn multiple-comparisons mirage, the tiny-n power calc, the synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`equinox_effect/`](equinox_effect/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
