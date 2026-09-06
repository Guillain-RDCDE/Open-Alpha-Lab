# Study 999 — The Break 🔗

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — can a change-point detector find real regime shifts in returns? | ![Partial](https://img.shields.io/badge/Partial-dab617?style=flat-square) | Against **planted** change points — the only kind whose dates are known — a variance CUSUM at threshold 5 found **100%** of them, with a median delay of **6 sessions** and 13.7 alarms a year. That delay is not a flaw in the algorithm: Wald's identity puts the unavoidable floor at about inf sessions for a shift this size, so the detector is running at **0.0× the information-theoretic limit**. The threshold is the only real dial, and it buys exactly one thing with another: dropping it to 2 cut the median delay to 3 sessions and raised the alarm rate to 38.2 a year. On the real tape the detector fired around the episodes everyone would name, but a **retrospective** method placed the same breaks 2 sessions earlier on average — which is the gap between what is knowable afterwards and what was knowable at the time. |
| **Tradability** — is the detection fast enough to act on? | ![Partial](https://img.shields.io/badge/Partial-dab617?style=flat-square) | Run live with no look-ahead, going to cash for 21 sessions after each alarm, the rule returned **+3.43%/yr at a Sharpe of 0.49** against buy-and-hold's +10.68% and 0.64, with a drawdown of -15% versus -55%. Now the number that explains it: the **identical rule given the break dates in advance** returned +12.01% at a Sharpe of 0.74. The gap between the live and hindsight versions — +0.25 of Sharpe — is not a failure of the strategy. It is the price of the 6-session delay, and no better detector removes more than a fraction of it. |

> **In one sentence:** A change-point detector finds 100% of planted regime shifts a median 6 sessions late — close to the theoretical floor — and that delay alone accounts for +0.25 of Sharpe against knowing the dates in advance.

## What we tested

Everyone agrees markets change regime. Far fewer people ask **how long it takes to
notice**. A change-point detector is a hypothesis test run repeatedly, so it needs evidence to
accumulate before it fires — which means every detection is late, and this study measures by how
much.

The grading is done where grading is possible: against change points **planted** at known dates,
so a detector can be scored on the two quantities that actually trade off — detection delay and
false-alarm rate. The result that reframes everything is section 2: Wald's identity puts a floor
on the delay of *any* sequential detector at roughly `threshold / (change size − drift)`, and
the CUSUM here runs within a small factor of it. **A detector that seems slow is usually
performing near the information-theoretic limit — the data is slow, not the code.** No amount of
algorithmic cleverness fixes that, which is the single most useful thing to know about the whole
field.

The other spine is the distinction between **retrospective** and **sequential** methods. Binary
segmentation, given the whole series, places breaks far more accurately than a live detector
can — because it sees the future. Papers routinely demonstrate the retrospective version and
imply the live one. Finally the cost is priced directly: a crude switching rule run live against
**the identical rule told the break dates in advance**, and the Sharpe gap between them is
attributable to the delay rather than to the strategy.
**Dedup:** distinct from **625-macro-regime-switching** (a fitted Markov-switching model rather
than sequential detection), **992-vol-clustering-halflife** (how long a regime lasts, not when it
starts), **990-var-breach-count** (risk-model calibration) and **985-last-hike-timing** (a
specific macro event with a published date rather than a statistical break).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | how long it really takes to notice a regime change, why that delay is mostly unavoidable, and what it costs |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | CUSUM and variance-CUSUM graded against planted breaks, the Wald delay bound, the delay/false-alarm curve, retrospective versus sequential detection, and a live rule priced against one told the answers |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`thebreak/`](thebreak/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
