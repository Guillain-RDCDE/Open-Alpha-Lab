# Study 224 -- Monday Effect

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) -- see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** -- is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Monday is **+5.64 bps** (positive, not negative) and the Monday-vs-rest contrast is **+1.08 bps** (HAC *t* **+0.36**) -- wrong sign, nowhere near significance. Thursday is actually the weakest weekday (+1.46 bps). There is no negative-Monday signal on this tape. |
| **Tradability** -- does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | "Buy Monday only" earns **1.34% CAGR** vs **10.82%** for buy-and-hold -- a loss of **9.5 pts/yr** because it sits in cash 81% of the time and forfeits the equity premium. "Skip Monday" loses **3.5 pts/yr** (7.30% vs 10.82%). You can't skip your way to alpha on a day that isn't actually negative. |
| **Is Monday still the market's worst day, or did that ghost die decades ago?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | The ghost died before the SPY era began. Monday is **positive pre- and post-2000**, and the change test (HAC *t* **-0.77**) confirms nothing significant shifted. French's (1980) negative-Monday finding is a 1953-1977 pre-publication sample artefact; it is not on this tape. |

> **In one sentence:** The classic Monday Effect -- negative close-to-close Monday returns, documented by French (1980) on 1953-1977 data -- is simply **not present on the 1993-2026 SPY tape**: Monday is positive (+5.64 bps), the contrast is insignificant (HAC *t* +0.36), and every literal "avoid Monday" rule loses to buy-and-hold by 3-9 points a year because you forfeit the equity premium on days you sit out.

## What we tested

The desk-folklore version, stated at full strength: *"Mondays are negative -- the **Monday
Effect** (a.k.a. the Weekend Effect measured close-to-close) -- and avoiding Monday or
trading only on the best day beats buy-and-hold."* Kenneth French's landmark *Stock Returns
and the Weekend Effect* (JFE, 1980) found significantly **negative average Monday
close-to-close returns** on the S&P composite over 1953-1977, and Gibbons & Hess (1981)
confirmed it. The effect is widely reported to have **decayed or reversed after
publication**. We take it literally on **total-return SPY** (1993-2026): per-weekday mean
close-to-close returns with **HAC** *t*-stats, the **Monday-vs-rest** contrast, a
**pre-2000 vs post-2000** split *with a test of the change*, and two literal timers
("buy Monday only", "skip Monday") net of **1 bp**/switch. Day-of-week is calendar-known,
so there is **no execution lag**. A synthetic tape with a planted weekday bump is the
positive control; a zero-bump i.i.d. tape is the null.

**Difference from Study 90 (Weekend Effect):** that study used *overnight* (close-to-open)
returns to isolate the weekend gap. This study uses *close-to-close* returns -- matching
French's (1980) exact measurement convention -- and focuses on the Monday Effect specifically.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the claim, the bar-chart of weekday returns, why Monday isn't negative, and why "avoid Monday" quietly loses to just holding |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC *t* on each weekday and each contrast, the five-test selection problem, the pre/post-2000 difference test, the lost-beta arithmetic |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`monday_effect/`](monday_effect/). **Not investment advice** -- research & education. See [LICENSE](../../LICENSE).*
