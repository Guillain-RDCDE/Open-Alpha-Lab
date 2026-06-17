# Study 285 — St-Patricks-Day

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

**Is there a St. Patrick's Day bump?**

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | St. Patrick session mean **+14.8 bps** but HAC t = **+1.13**; vs other March days Welch p = **0.30**. The whole effect lives in 1991–2026 (t = 3.27) and is *negative* before — a regime artifact, not a stable signal. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | One round-trip a year, ~13 bps net (HAC t = 0.98); dominated by simply holding the index, which earns the drift every day. |
| **Busted?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | The placebo (March 24) is flat, no March anchor day survives Bonferroni, and the bump exists only in the last 35 years. |

> **In one sentence:** the "wearin' o' the green" rally is folklore — a mildly positive day whose reputation rests entirely on one lucky modern regime and the fact that the market drifts up anyway.

## What we tested

An **event study** on ^GSPC daily returns (1928–2026, 99 St. Patrick sessions). The
St. Patrick's Day trading session is derived by date arithmetic (first trading day
on/after March 17) in `data.py` — not a hard-coded table. We put a Newey-West HAC
t-stat on the event-day mean, run a Welch contrast against all other days and against
other March days, add a **placebo** (March 24, one week later), Bonferroni-correct a
**day-of-March sweep** over five anchor days, split the sample into three regimes, and
charge a 2 bps round-trip on the "trade the bump" P&L. A synthetic positive control
confirms the machinery fires when an effect is planted; the real tape has none.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the claim, the drift baseline, the sub-period reveal, the placebo, in plain English |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC t-stat, Welch contrasts, Bonferroni sweep, sub-period split, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`st_patricks_day/`](st_patricks_day/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
