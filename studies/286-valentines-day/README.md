# Study 286 — Valentines-Day

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

**Does the market love Valentine's Day?**

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Valentine session beats an average day by **+0.24 bps**; HAC t = **+0.02**, perm p = **0.98**, t vs unconditional = **+0.02**. Its up-rate (47%) is actually *below* the 53% base rate. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | No vehicle beyond "be long the S&P for one February day"; the ~4 bps gross "edge" is noise and one-way costs halve it. Passive buy-and-hold of that day is strictly better. |
| **Busted?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | The faint positive drift on Feb 14 is just the market's everyday drift — present on every session. The "good mood" is folklore. |

> **In one sentence:** tested against the honest baseline (the S&P drifts up ~3.6 bps on *any* day), the Valentine's Day session adds a quarter of a basis point — statistically indistinguishable from noise, and the day is actually *down* more often than an average day.

## What we tested

Calendar folklore says the market is "in love" on Feb 14 and rises. We run it as a
clean single-day **event study**: for each year 1950–2025 we isolate the first ^GSPC
session on or after Feb 14 (Feb 14 often falls on a weekend/holiday — the roll-forward
convention is hardcoded in `data.py`, 76 sessions), and test whether its
close-to-close return beats the **unconditional daily mean** — not zero. We report a
one-sample t-test (vs 0 and vs the baseline), a Newey-West HAC t on the excess return,
a 10,000-draw permutation test, the 1950–87 vs 1988–2025 split, and a gross/net
tradability column. A synthetic positive control confirms the engine fires when a
Valentine premium is planted; the real tape has none.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the claim, the daily-drift trap, the (backwards) up-rate, the answer in plain English |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | one-sample & HAC t, the permutation distribution, sub-periods, the n=76 power calc, the synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`valentines_day/`](valentines_day/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
