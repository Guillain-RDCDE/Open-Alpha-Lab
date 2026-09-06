# Study 985 — The Last Hike 🏛

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do markets behave differently after a tightening cycle ends? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Across the **5 tightening cycles** since the Fed began announcing its target in 1994, the S&P returned **+17.8%** in the twelve months after the cycle's true final hike, against an unconditional base rate of +10.1% — an excess of **+7.7%** (*t* = **+1.08**, 80% of cycles positive). With 5 events there is no *t*-statistic that can carry much weight, and the study says so rather than dressing eight observations as a finding. |
| **Tradability** — can a rule that does not know the future capture any of it? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | None of that is available at the time. Nobody knew 2023-07-26 was the last hike until months of not-hiking had passed. A live rule — declare the cycle over after **6 quiet months** — fired 7 times, of which **43% were false alarms** (another hike followed within two years), and earned an excess of **-1.1%** at twelve months (*t* = -0.19) against the hindsight version's +7.7%. The delay alone gave up a median **+11.4%** between the true last hike and the day the rule could act — 64% of the whole twelve-month move. |

> **In one sentence:** Buying the last hike returned +7.7% of excess over twelve months in hindsight and -1.1% for a rule that had to identify the event in real time — the gap is what the folklore is actually made of.

## What we tested

"Buy when the Fed stops hiking" is folklore that sounds like it should be true,
and the charts that support it are real. This study reconstructs every tightening cycle since
February 1994 — when the Fed began announcing its target, so the record is fact rather than
inference — from a **hard-coded** table of all 90-odd target changes, and measures what stocks,
long bonds, gold and small caps did over the following 3, 6, 12 and 24 months, always against an
**unconditional base rate** computed from every month in the sample (without which "stocks rose
14% in the year after the last hike" says nothing, because stocks rise about 9% in the year after
most dates).

Then it asks the question the folklore never does: **you did not know it was the last hike.**
The most generous live rule available — declare the cycle over once six months pass with no
further hike, using no forecast at all — is run over the same history, and it fires on pauses
that were not endings. Those false alarms are counted, the return given up while waiting is
measured cycle by cycle, and the identical event study is re-run on the dates a live investor
could actually have acted. A synthetic world with a *planted* post-cycle rally shows that even a
large, genuine effect loses most of itself to the recognition delay — a property of the problem,
not of the rule.
**Dedup:** distinct from **149-fed-day-drift** and **312-fomc-announcement-returns** (returns
around individual meetings), **088-yield-curve-inversion** and **507-rate-cuts-and-stocks** (the
easing side, and the curve as a recession signal), **625-macro-regime-switching** (a fitted
regime model rather than the published policy record) and **429-dont-fight-the-fed** (a
continuous rate-direction overlay, not a cycle-turning-point event study).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what the tape did after each of the five tightening cycles ended, and how much of that was available to anyone at the time |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | cycle reconstruction from the published policy record, base-rate-adjusted event returns, a live pause-recognition rule with its false alarms counted, convention sweeps, and a planted-rally control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`lasthike/`](lasthike/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
