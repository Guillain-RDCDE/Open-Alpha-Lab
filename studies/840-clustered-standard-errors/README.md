# Study 840 — Clustered Standard Errors 🧷

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is there a real edge to find? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | The panel is a **constructed null** — true slope **zero**, so there is nothing to find. Fama-MacBeth correctly declines to reject (false-positive rate **5.3%** ≈ nominal). The only thing that "finds significance" is a broken (or wrongly-clustered) standard error. |
| **Tradability** — could you trade it? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Nothing to harvest. The significance the naive *t* manufactures is a standard-error artefact, not a return; costed as a notional dollar-neutral long-short the null loses **−25.6%/yr**. |
| **Does cross-sectional dependence fake significance?** | ![Confirmed](https://img.shields.io/badge/Confirmed-8b949e?style=flat-square) | On a mean-zero panel with a common time effect the **naive OLS false-positive rate is 60%** (12× the 5% it advertises); the naive *t* is inflated by the closed-form Moulton factor **√13.25 ≈ 3.64×**; **one-way firm-clustering makes it worse (66%)** — it clusters on the wrong dimension; the no-dependence control is perfectly calibrated; and Fama-MacBeth restores ~nominal error while still catching a planted effect at **84% power**. |

> **In one sentence:** treat the 2,500 cross-sectionally-correlated observations of a panel with a common time shock as if they were independent and the ordinary *t*-statistic rejects a **true** null 60% of the time — inflated by exactly √(1 + (N−1)·ρ_x·ρ_e) — and clustering by *firm* only makes it worse, which is why a panel with a time effect needs Fama-MacBeth (or time / two-way clustering), and why a naked *t* > 2 from a pooled cross-sectional regression is worth almost nothing.

## What we tested

Petersen (2009) and Fama-MacBeth (1973): when a panel carries a **common time effect** — a shock
that hits every firm in a period at once (a market move, a macro surprise) — the residuals of a
pooled regression are **correlated across firms within that period**. The ordinary i.i.d. OLS
standard error, and one-way **firm** clustering, ignore that dependence and are far too small, so
the *t*-statistic **overstates significance**. We make the pitfall un-deniable by running it on a
panel we *built* to be empty: a mean-zero (β = 0) "noise predictor" `x` and an outcome `y`, both
loading on an independent shared per-period factor, simulated 2,000 times over 50 periods × 50
firms. We measure the **false-positive rate** of four standard errors on the *same* pooled slope
— naive OLS, firm-clustered, time-clustered, and **Fama-MacBeth** — recover the **inflation
factor** and match it to its closed form (the Moulton factor √(1 + (N−1)·ρ_x·ρ_e)), verify a
no-dependence control is calibrated, and confirm Fama-MacBeth still **fires on a planted effect**
(84% power). **Dedup:** this is the *cross-sectional-dependence* method demo — distinct from
[838 hac-necessity](../838-hac-necessity/) (autocorrelation *over time* in a single series,
Newey-West) and [346 multiple-testing](../346-multiple-testing/) (a haircut for *how many*
hypotheses you tried); together the three isolate the leading reasons a panel *t* > 2 lies.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why firms in the same month aren't really independent, how a shared shock fakes a "significant" predictor out of pure noise, and the one procedure (Fama-MacBeth) that fixes it — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the four standard errors for one pooled slope, the Monte-Carlo false-positive experiment, the Moulton √(1+(N−1)ρ_xρ_e) inflation identity, why firm-clustering fails, the no-dependence control, and the planted-effect power check |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run (config fp `a271c7ebce63`, null-panel fp `607a6862117f`, as-of 2026-06-30): [docs/results.md](docs/results.md).

---

*Engine: [`clustered_se/`](clustered_se/) — pure numpy / pandas / scipy, deterministic, offline. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
