# Study 195 — Monthly-OpEx

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Monthly-only (non-quarterly) OpEx volume t = −0.65; return week t = −0.07.  Aggregate day volume (t = +4.44) is the Study-82 quarterly triple-witching result, not an incremental monthly cycle. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Long-OpEx-week rule: Sharpe 0.63 vs buy-and-hold 0.65, CAGR 10.6% vs 10.9%, HAC *t* = +1.86; a 0.5 bp round-trip erases it entirely. |
| **Quarterly effect?** | ![Already known](https://img.shields.io/badge/Already_known-8b949e?style=flat-square) | Quarterly triple-witching volume uplift confirmed (t = +3.70); it is the Study-82 finding and does not extend to the eight non-quarterly months. |

> **In one sentence:** the monthly options-expiration week carries no incremental signal beyond the known quarterly triple-witching — eight months of calendar noise surrounding four quarterly events, untradable even if real.

## What we tested

The third Friday of every calendar month is the standard equity option expiry day.  Traders claim the surrounding week sees elevated volume, range, and a directional drift from dealer delta-hedging and gamma pinning.  We tested all three on SPY daily bars 1993–2026 (n = 8,401 days, 393 OpEx events), with a critical decomposition: **quarterly months** (Mar/Jun/Sep/Dec — the known triple-witching) vs **monthly-only months** (the other eight).  We also tested a pre/post-2010 structural break for the 0DTE / weekly-options era.

The volume signal that does appear (t = +4.44 aggregate) belongs entirely to the quarterly sub-sample (t = +3.70 quarterly vs t = −0.65 monthly-only).  Range is actually *lower* on OpEx days (t = −2.41) — consistent with gamma pinning dampening moves near strikes, not amplifying them.  The week-level return is null (t = −0.07).  A long-OpEx-week calendar rule earns the same Sharpe as buy-and-hold per dollar of exposure.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | volume decomposition, range pinning effect, return null, tradability check in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC t-stats, quarterly/monthly-only decomposition, pre/post-2010 structural break, synthetic positive control, cost sensitivity |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`monthly_opex/`](monthly_opex/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
