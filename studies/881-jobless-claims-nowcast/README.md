# Study 881 — Jobless-Claims Sector Rotation 📋

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does a 4-week claims uptick tilt the market to defensives over cyclicals? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The predictive slope of the **cyclical−defensive** forward spread on the 4-week claims change is **wrong-signed and "significant"** (**+0.0177**, Newey-West *t* = **+6.39**) — the claim needs a *negative* slope. And even that inverted significance is a **single-outlier (COVID-2020)** artefact: it collapses to *t* = **+1.49** ex-COVID, *t* = **+1.43** winsorised, a Spearman **ρ = +0.02 (p = 0.73)**, and *t* = **+1.41** in the pre-2020 era, with a borderline placebo (**p = 0.053**). A clean 20-seed synthetic control (recovers a *planted* rotation at *t* = −13.85, fires on **1/20** nulls) rules out a broken engine. Rising claims carry **no** robust cyclical-vs-defensive rotation information. *No survivorship in the outcome — the four SPDR ETFs trade continuously since 1998; the honest hazard is the 2020 outlier, named on the Signal axis.* |
| **Tradability** — can you get paid for the rotation? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The sign-correct long-short rotation earns only **+0.89%/yr gross** (*t* ≈ 0) and goes **−1.91%/yr net** once its **168** flips pay a 10 bp one-way cost. |

> **In one sentence:** the labour-nowcast "rising claims → rotate to defensives" rule
> **does not exist** on the sector tape — the only significant number is *wrong-signed*
> and entirely a 2020 outlier (Spearman ρ ≈ 0, p = 0.73), and the rotation earns nothing
> gross and less than nothing net, so the honest read is **claimed signal absent,
> paycheck a mirage**.

## What we tested

The nowcast folk-rule: the **4-week change in initial jobless claims** should drive a
**cyclical-vs-defensive sector rotation** — rising claims (labour softening) → tilt to
**defensives** (XLP staples, XLU utilities) over **cyclicals** (XLY discretionary, XLI
industrials), so the cyclical-minus-defensive forward spread should be *negative*. We run
it on a **monthly frame (1998-12-31 → 2026-06-30, 331 months)**: the FRED `IC4WSA`
4-week-MA claims level (a documented public snapshot — the FRED CSV host is unreachable in
this build) and the four SPDR sector ETFs + SPY (yfinance total-return, **fetched live &
cached**). The headline is a **predictive Newey-West regression** of the forward
equal-weight cyclical-minus-defensive spread on the 4-week claims change (signal through
month-end `t` → hold month `t+1`, one lag, zero look-ahead), with a COVID-sensitivity /
winsor / Spearman triple, a two-era cut, a 2,000-draw permutation placebo, a costed
long-short rotation timer, and a 20-seed synthetic positive control. The 2020 outlier is
named on the **Signal** axis. **Dedup:** [385-jobless-claims-momentum](../385-jobless-claims-momentum/)
uses the same claims tape but times the **whole market** (SPY), not a sector rotation;
[268-sahm-rule](../268-sahm-rule/) is an **unemployment-rate** recession *call*, not a
claims-change rotation; [626-unemployment-trend-timing](../626-unemployment-trend-timing/)
trend-times market exposure on the **unemployment rate**; [756-challenger-layoffs](../756-challenger-layoffs/)
uses **Challenger job-cut** announcements, a different feed and outcome. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a claims uptick *should* rotate the market to defensives — and why the one "significant" number is a single 2020 mirage |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the predictive NW slope, the COVID / winsor / Spearman triple, the two-era cut, the permutation placebo, the costed rotation timer, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`claims_nowcast/`](claims_nowcast/). Claims = FRED `IC4WSA` 4-week-MA (documented
public snapshot); sector tape pulled live from Yahoo (total-return, continuous since 1998).
**Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
