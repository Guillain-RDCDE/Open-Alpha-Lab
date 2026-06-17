# Study 283 — Hurricane-Season

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

**Does Atlantic hurricane season sink stocks (or insurers)?**

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Market in-season − off-season spread **−0.47 bps/day** (HAC t = **−0.21**, perm p = **0.80**); insurer basket **−1.13 bps/day** (HAC t = **−0.35**); landfall event-window CAR **+0.25%** (t = **+0.33**). Nothing clears \|HAC t\| ≥ 2. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The only implementation is "sit out half the year," which earns **+4.4%/yr** vs buy-and-hold **+8.3%/yr** — you forgo the equity premium to dodge a drag that isn't there. |
| **Busted?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | Any in-season weakness coincides with the documented Sell-in-May/Halloween window, and the insurer sign is ambiguous anyway (catastrophes raise forward pricing — "gaining from loss"). |

> **In one sentence:** the Atlantic hurricane season covers half the calendar and moves neither the broad market nor a catastrophe-exposed insurer basket in any statistically detectable way — and the named-storm event windows are, if anything, mildly *positive* for insurers.

## What we tested

Two tapes, one calendar rule. We hardcode the hurricane-season window (Jun 1 – Nov 30)
and a table of 30 major US Atlantic-hurricane landfalls (1992–2024) in `data.py`, then
join them to daily **^GSPC** (price-only, repo cache) and an equal-weight **P&C-insurer
basket** (TRV, CB, ALL, PGR, AIG, HIG, CINF, WRB; total-return, study cache). We compare
in-season vs off-season daily returns with a Welch t, an honest **HAC (Newey-West) t**,
and a block-permutation test; run an **event study** of cumulative abnormal returns over
[-5, +20] days around each landfall (1-day execution lag); and benchmark a "sit out the
season" timing rule against buy-and-hold. A synthetic positive control confirms the engine
finds a planted −10 bps/day drag when one exists.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the claim, the half-the-calendar trap, the in-season/off-season bars, the timing-rule shortfall in plain English |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | Welch vs HAC t, the permutation distribution, the landfall event study, the power calculation, the synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`hurricane_season/`](hurricane_season/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
