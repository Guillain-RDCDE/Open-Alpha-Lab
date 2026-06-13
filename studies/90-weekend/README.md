# Study 90 — Weekend 📆

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Tuesday is the strongest single weekday (+7.17 bps, HAC *t* +2.64), but the *claim* — Tuesday **beats the other days** — is only +3.03 bps (HAC *t* **+1.00**), and the Monday claim is the wrong sign. With five weekday tests in play, no contrast survives honestly. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Every literal rule loses badly: "buy Tuesday" earns **8.5 pts/yr less** than buy-and-hold (2.36% vs 10.82%) — sitting in cash 79% of the time forfeits the equity premium — and "skip Monday" loses **3.5 pts/yr**. The weekday tilt is dwarfed by the lost beta. |
| **Weekend effect still there?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | The defining feature — a *negative* Monday — is **absent** across the whole SPY record: Monday is **+5.64 bps** (positive pre- *and* post-2000), and the change-across-2000 test is insignificant (HAC *t* −0.77). The famous effect lived in French's 1953–1977 sample; it is not in this tape. |

> **In one sentence:** Tuesday really is the best weekday, but only barely and not significantly versus the rest — the *negative-Monday* "weekend effect" is simply **not in the modern SPY tape**, and any literal "avoid Monday / buy Tuesday" rule loses to buy-and-hold by 3–8 points a year because you forfeit the equity premium on the days you sit out.

## What we tested

The desk-folklore version, stated at full strength: *"Mondays are negative — the **weekend
effect** — and Tuesday rebounds (**turnaround Tuesday**). Avoid Monday / buy Tuesday and you
beat buy-and-hold."* The effect is real in the founding literature — Kenneth French's
*Stock Returns and the Weekend Effect* (JFE, 1980) found significantly negative Mondays on
1953–1977 index data — but it is widely reported to have **decayed or reversed after
publication**. We take it literally on **total-return SPY** (1993–2026): per-weekday mean
returns with **HAC** *t*-stats, the **Monday-vs-rest** and **Tuesday-vs-rest** contrasts, a
**pre-2000 vs post-2000** split *with a test of the change*, and two literal timers ("buy
Tuesday", "skip Monday") net of **1 bp**/switch. Day-of-week is calendar-known, so there is
**no execution lag**. A deterministic synthetic tape with a *planted* weekday bump serves as
the positive control (the harness banks it); a knob-zero i.i.d. tape is the null (it finds
nothing).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the claim, the bar-chart of weekday returns, why Monday isn't negative, and why "buy Tuesday" quietly loses to just holding |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC *t* on each weekday and each contrast, the five-test selection problem, the pre/post-2000 difference test, the lost-beta arithmetic |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`weekend/`](weekend/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
