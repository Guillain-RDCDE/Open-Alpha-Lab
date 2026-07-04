# Study 596 — Bond Tent (Rising Equity Glidepath) ⛺

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the tent cut sequence-of-returns risk? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | On 1,470 monthly-start 30-year retirements (Shiller 1871–2023, real returns), the rising tent (30→70% equity) trails static 60/40 on **every** metric: success **94.0% vs 96.3%** at 4% (**60.4% vs 75.8%** at 5%), mean terminal wealth **1.47× vs 1.88×** (HAC *t* = **−6.11**, bootstrap CI excludes 0), SAFEMAX **3.57% vs 3.68%**. It even loses to its own **mirror image** (declining 70→30). Robust across four tent shapes, 0–25 bps costs, and high-CAPE retirements. Named caveat: the US tape is history's best equity market, and our bond leg is a CPI-deflated 10-year. |
| **Tradability** — should you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Perfectly implementable with two ETFs — and implementing it historically *bought less safety*: −0.42× terminal wealth and −2.3 pp success at 4% (−15.4 pp at 5%) vs a simpler static 60/40. There is no benefit to harvest. |
| **"Timing or just less equity?"** | ![Busted](https://img.shields.io/badge/Timing_or_less_equity%3F-Busted-8b949e?style=flat-square) | The decomposition `rising − 60/40 = shape + allocation` kills both channels: less equity is a certified cost (*t* = −5.92) and the **shape itself is negative at matched average equity** (*t* = −1.85; loses the pure-timing mirror race). The advertised mechanism exists only in i.i.d. Monte Carlo with a low equity premium (synthetic control: +6.0 pp, *t* +5.7 at 0 premium) — real history's decade-long inflation grinds, which crush bonds and stocks together, never paid it. |

> **In one sentence:** the Kitces-Pfau bond tent — retire bond-heavy, then re-raise equity — insures against a short 1929-style crash (that one cohort: **1.25× vs 0.39×** for 60/40) but on 152 years of US data the dominant retirement killer was the 15-year 1966–82 inflation grind, where the tent holds bonds through their worst decade and walks equity up *after* the recovery, so it trails static 60/40 everywhere — the "benefit" is neither sequence timing nor smart de-risking, it is a Monte-Carlo-only artefact.

## What we tested

We simulate 30-year retirements with 4% (and 4.5–5.5%) real withdrawals, annually rebalanced, on Shiller real stock and CPI-deflated 10-year bond returns: static **60/40**, **50/50**, **40/60**, the **rising tent** 30→70% equity, and its **declining mirror** 70→30% (same 50% average — the pure sequence-timing control). Inference: Newey-West HAC *t* on overlapping-cohort terminal-wealth differences (bandwidth = full 360-month overlap) plus circular block-bootstrap CIs (1,000 reps, 120-month blocks). The third axis decomposes the tent's headline gap into *shape at matched average equity* vs *just holding less equity*. A 20-world synthetic control proves the engine detects the tent's benefit where it truly exists (i.i.d., low premium — exactly Kitces-Pfau's laboratory) and finds nothing at the exact null. Cousin of [172 — Hundred-Minus-Age](../172-hundred-minus-age/) (accumulation glidepaths) and [173 — Four-Percent-Rule](../173-four-percent-rule/) (same tape, withdrawal rate itself); new here is the **decumulation glidepath shape**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a bond tent is, why 1929 made it famous, why 1966 breaks it, and what actually protects a retiree — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | cohort machinery, HAC/bootstrap inference, the shape-vs-allocation decomposition, high-CAPE conditioning, tent-shape robustness, and the premium-sweep synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`bond_tent_glidepath/`](bond_tent_glidepath/). Weights are a deterministic function of years-since-retirement (set at the end of the prior year — one clean lag); costs 10 bps one-way × traded value. Real (CPI-deflated) total returns throughout, labeled. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
