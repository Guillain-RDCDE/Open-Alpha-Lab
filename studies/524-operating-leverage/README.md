# Study 524 — Operating-Leverage

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## The claim

> *Do firms with a high fixed-cost (operating-leverage) structure earn a premium?*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | Long-high-OL / short-low-OL hedge **−2.14%/yr**, HAC *t* = **−0.39** (wrong sign); placebo p = **0.72** (a shuffled label beats it 72% of the time); high-OL beats only **54%** of random draws. The low and high OL terciles earn near-identical returns (+22.3% vs +20.1%/yr). Literature supports the premium on a broad universe, not on survivor large-caps. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The gross hedge is already negative; after 10 bps × turnover and a 50 bps short borrow it is **−2.84%/yr net**. Nothing to trade. |
| **Operating- vs financial-leverage channel?** | ![Busted](https://img.shields.io/badge/Channel-Busted-8b949e?style=flat-square) | The fixed-cost / operating-leverage channel is no more present than the balance-sheet financial-leverage channel of [Study 154](../154-leverage-anomaly/) — both vanish on the survivor large-cap panel. |

> **In one sentence:** Novy-Marx (2011) found that high-operating-leverage firms — those whose fixed operating costs make them a levered claim on demand — earn a premium on the broad US cross-section, but on a survivorship-biased large-cap basket the effect is absent and mildly inverted (hedge −2.14%/yr, *t* = −0.39, net −2.84%/yr), because deleting the high-fixed-cost firms that *failed* removes exactly the bad-state risk the premium is meant to pay for.

## What we tested

Novy-Marx (2011) argues that a firm's *fixed* operating costs act like leverage on the income
statement: a firm whose operating costs (COGS + SG&A) are large relative to its asset base
behaves like a levered claim on demand — small revenue swings produce large operating-profit
swings — so it carries more operating risk and earns a premium that overlaps the value effect.
This is the **operating-leverage value channel**, deliberately distinct from the *financial*
leverage (balance-sheet debt) anomaly of [Study 154](../154-leverage-anomaly/).

We compute **OL = (COGS + SG&A) / Total_Assets** from SEC EDGAR companyfacts (a ~13-year deep
fundamentals history, far longer than yfinance statements expose), sort a fixed large-cap
survivor basket (~38 names/year) into terciles each fiscal year, lag the signal by a
conservative report lag (12-month forward return beginning 4 months after fiscal year-end,
entered one trading day later — exactly one execution lag, no look-ahead), and test whether
the high-OL tercile beats the low-OL tercile. A label-shuffle placebo, a random-portfolio
null, explicit costs + short borrow, and a seed-robust synthetic positive control round out
the inference.

The basket is **survivorship-biased**: it covers only firms still large-cap in 2026. This
matters more than usual here because operating leverage is, by construction, a *risk* premium
— it is meant to pay in the bad states where a high-fixed-cost firm's demand collapses. Those
failures are exactly what a survivor basket deletes, so the surviving high-OL names are the
ones whose demand *didn't* collapse, biasing the test toward finding no premium. (Note
2024: the low-OL, capital-light megacaps returned +94.7% while the high-OL tercile lagged.)

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the operating-leverage recipe in plain English, why the survivor basket erases (and inverts) the sign, year-by-year results |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | tercile monotonicity (absent), HAC t-stats, label-shuffle placebo, random-portfolio null, costs, seed-robust synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`operating_leverage/`](operating_leverage/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
