# Study 924 — First Cut

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the edge real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Buying TLT the day after the first cut of a cycle returns **+4.88%** net excess-of-cash over 12 months on **N = 4** events — *t* = **+0.55**, randomisation *p* = **0.365** against matched random start dates, and no horizon reaches *t* = 1.1. Both are generous: the 12-month windows overlap, so N = 4 is really **3 macro episodes**, and the iid placebo makes 0.365 a floor. The premise fails its own control: buying **any** cut paid *more* (**+5.50%**, N = 18 — the same 3 episodes, nominal *t* = +1.78 oversized, daily HAC *t* = +0.22). Era split flips sign (+19.4% pre-2020 / −9.7% since). |
| **Tradability** — is it bankable? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The conditional strategy earns **+0.47%/y** excess of cash over the full sample, CI [−2.05%, +3.02%] — **+2.62%/y** counting only the 18.7% of days it is invested — and *both* sit below the **+2.89%/y** from just holding TLT. The whole positive mean is one window (Aug-2019 → Jul-2020) that owes its return to a pandemic. Costs are irrelevant; the absence of an effect is not. |

> **In one sentence:** The "when the Fed starts cutting, buy duration" trade has fired four measurable times in twenty years, two of them lost money, the average is carried by a single pandemic window, and buying *every* cut beat buying the hand-picked *first* one — so the timing device adds nothing to simply owning duration, and four events could never have proved otherwise.

## What we tested

A **hardcoded list of first-cut-of-cycle FOMC dates** (2001-01-03, 2007-09-18, 2019-07-31,
2020-03-03, 2024-09-18 — an **ASSUMPTION**, the study's only non-tape input): buy **TLT**
at the **close of the session after** the announcement (one execution lag), hold **1 / 3 /
6 / 12 months**, score **excess of BIL** over the identical days, 5 bps one-way × NAV on
entry and exit. Controls: **all 31 cuts**, a **2,000-draw random-date placebo**, a daily
conditional leg with Newey-West *t* and block-bootstrap CI, an era split, cost and
**borrow** sweeps (the long-TLT/short-SHY curve version pays borrow), an **IEF** belly
cross-check and an **^IRX** proxy-cash cross-check. 2001 is unmeasurable — TLT lists from
2002-07-30. **Dedup:** distinct from **67-fed-drift** and **517-pre-fomc-drift** (equity
drift *before* announcements), **135-fomc-cycle** (fortnightly seasonality across all
meetings), **322-fomc-blackout** / **637-fomc-vol-crush** (communications window, implied
vol), **647-pboc-rrr-effect** (another central bank, equities), and **59-downhill** /
**581-term-premium** / **625-starting-yield-bond-decade** (*unconditional* or
yield-conditioned reasons to own duration — this one conditions on a policy event, and
loses to them).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | all four trades in full, the one window carrying the average, the control that breaks the premise, why we cannot even call it false |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | horizon sweep vs the all-cuts control, randomisation inference, the daily conditional leg's HAC *t* and bootstrap CI, cost/borrow sweeps, and a power calibration of the five-event test itself |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`first_cut/`](first_cut/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
