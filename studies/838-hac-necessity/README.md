# Study 838 — HAC Necessity 📏

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is there a real edge to find? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | The series is a **constructed null** — true mean **zero**, so there is nothing to find. Told about the autocorrelation, the Newey-West *t* correctly declines to reject (false-positive rate **9.5%** ≈ nominal). The only thing that "finds significance" is a broken estimator. |
| **Tradability** — could you trade it? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Nothing to harvest. The significance the naive *t* manufactures is a standard-error artefact, not a return; costed as a notional long-short the null loses **−24.6%/yr**. |
| **Does ignoring autocorrelation fake significance?** | ![Confirmed](https://img.shields.io/badge/Confirmed-8b949e?style=flat-square) | On a mean-zero 21-day-overlap tape the **naive false-positive rate is 64%** (13× the 5% it advertises); the naive *t* is inflated by exactly **√21 ≈ 4.6×**; the inflation tracks √window and √((1+ρ)/(1−ρ)) to two figures; the i.i.d. control is perfectly calibrated; and Newey-West restores ~nominal error while still catching a planted effect at **89% power**. |

> **In one sentence:** treat 2,520 heavily-overlapping daily observations as if they were independent and the ordinary *t*-statistic rejects a **true** null 64% of the time — inflated by exactly the square root of the overlap length — which is why any strategy with a formation window needs the Newey-West (HAC) standard error, and why a naked *t* > 2 on a serially-correlated series is worth almost nothing.

## What we tested

Newey & West (1987) and Hansen & Hodrick (1980): when a return series is **serially correlated** — the unavoidable signature of a daily long-short with, say, a 21-day formation window — the ordinary i.i.d. OLS standard error is too small, so the *t*-statistic **overstates significance**. We make the pitfall un-deniable by running it on a tape we *built* to be empty: a mean-zero series carrying the autocorrelation of an overlapping window (an MA(20) process), simulated 600 times over 2,520 days. We measure the **false-positive rate** of the naive *t* versus the **Newey-West (HAC)** *t*, recover the **inflation factor** and match it to its closed form (√window for an overlap, √((1+ρ)/(1−ρ)) for AR(1)), verify a no-autocorrelation control is calibrated, and confirm the HAC machinery still **fires on a planted effect** (89% power). **Dedup:** this is the *inference-side* method demo — distinct from [346 multiple-testing](../../346-multiple-testing/) (which corrects for *how many* hypotheses you tried), [841 overlapping-returns](../../841-overlapping-returns/) (the same MA structure from the returns-construction side), and [348 curve-fitting](../../348-curve-fitting/) (a fake edge from fitting flexibility, not a mis-specified variance).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why overlapping days aren't really independent, how that fakes a "significant" result out of pure noise, and the one correction that fixes it — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the Monte-Carlo false-positive experiment, the √window / √((1+ρ)/(1−ρ)) inflation identities, Newey-West vs statsmodels, the automatic-bandwidth trap, and the planted-effect power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run (config fp `767e2ce61be1`, null-matrix fp `0c98419fb4d7`, as-of 2026-06-30): [docs/results.md](docs/results.md).

---

*Engine: [`hac_necessity/`](hac_necessity/) — pure numpy / pandas / scipy / statsmodels, deterministic, offline. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
