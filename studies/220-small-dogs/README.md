# Study 220 — Small Dogs of the Dow

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Over **26 point-in-time years (2000–2025)** the Small Dogs beat the Dow by **+3.04 pts/yr** (CAGR 10.57% vs 7.54%) — but the annual excess has **HAC *t* = +1.82**, just below the |t| ≥ 2 bar (bootstrap 95% CI **[−0.27%, +5.63%]**, *p* = 0.07). The incremental test (Small Dogs vs full Dogs) clears at *t* = **+2.40**, but the price filter was discovered in-sample — a data-mining warning. The alpha *t* = +2.68 clears the bar but is driven by a low beta (0.72), not independent alpha. The **Foolish Four variant is flat noise** (HAC *t* = −0.09 vs Dow), busting the in-sample "skip-the-cheapest" quirk. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | A 5-name equal-weight basket of the *cheapest* blue chips carries outsized concentration and dividend-trap risk (Jan-2009: Citi at 23%, BofA at 21% — both about to cut). The low-price tilt is a crude small-cap proxy replicable far more cheaply with a small-cap-value ETF. Multiple-comparison penalty (3 strategy variants tested), in-sample origins of the price-filter rule, and no tax/capacity advantage over the full Dogs make the gap uninvestable. |
| **Myth check** — is the "Foolish Four" beaten-down-stock story coherent? | ![BUSTED](https://img.shields.io/badge/BUSTED-8b949e?style=flat-square) | The Foolish Four (skip the cheapest Dog, buy the next four) earned −0.11%/yr vs the Dow over 26 years; the Motley Fool retired it as a strategy in 2000 after it collapsed in live trading. The skip-the-cheapest rule had no theory behind it — it was the residue of curve-fitting on a short history. |

> Do the 5 cheapest Dogs (the Foolish Four / Small Dogs) beat the full ten?

> **In one sentence:** the Small Dogs show a tempting +3-point historical lead over 26 years — but it doesn't clear the significance bar, the Foolish Four variant is flat noise, the price filter was invented by data-mining the same window, and the "low-price tilt" you'd be buying is a crude small-cap proxy you can own far more cheaply.

## What we tested

The **Small Dogs of the Dow** (also called the "Low-5"): each January, of the ten
highest-trailing-yield Dow stocks (the Dogs), buy only the **five with the lowest absolute
share price** equal-weight, hold a year. The **Foolish Four**: same universe, but skip the
#1 cheapest and buy #2-#5 — a Motley Fool variant from 1996, since debunked as a pure
data-mining artefact (the Motley Fool itself retired it in 2000). We test all three variants
on the **same 26-year point-in-time tape** as Study 88, so the race is identical in every
detail except the sub-selection rule. We ask: (1) do the sub-strategies beat the Dow?
(2) do they beat the full Dogs? (3) can any lead survive a risk-adjustment and a
multiple-comparison correction appropriate for strategies discovered by in-sample optimisation?

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the three-basket race, the equity curves, the 2009 dividend-trap basket, the Foolish Four post-mortem |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC + block-bootstrap on three strategies simultaneously, alpha-vs-beta, multiple-comparison framing, why the price filter is a crude size-factor proxy |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`small_dogs/`](small_dogs/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
