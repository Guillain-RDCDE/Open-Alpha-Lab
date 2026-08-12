# Study 902 — Multi-Factor Composite 🧩

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the blend's edge real? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | The **diversification is real & mechanical**: the equal-weight sleeve carries lower vol (**14.1%** vs mean single sleeve 15.2%) and a higher excess Sharpe (**0.841** vs mean single 0.786), taming ~7 pp/yr of cross-sleeve dispersion. But the headline claim — *beat the market* — fails: excess-of-cash Sharpe advantage over SPY **−0.033** (active NW *t* **−0.73**), bootstrap CI **[−0.205, +0.077]** straddles zero, and it **flips sign across eras**. Flagship-survivor selection named. |
| **Tradability** — is it bankable? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | Costs are **not** the obstacle — the sleeve turns over 1.17% of NAV/mo (**~0.3 bps/yr**); gross and net Sharpe are identical to two decimals. The real, deliverable benefit (diversified factor-timing risk, one liquid ticket per sleeve) is trivially buyable — but what it buys is a **marginally-lower-Sharpe, lower-vol clone of SPY**, not a market-beating edge. Real but thin → Fragile. |

> **In one sentence:** an equal-weight live sleeve of the five flagship iShares single-factor
> ETFs (VLUE/QUAL/MTUM/USMV/SIZE), rebalanced monthly over 2013-08 → 2026-06, does exactly what
> diversification promises — it beats the *average* single factor and smooths the ride — but it
> does **not** beat the *market*: its excess-of-cash Sharpe (0.841) lands just short of SPY's
> (0.874), insignificantly and not era-robustly, and at near-zero cost — **a genuine diversifier,
> not an edge; Weak / Fragile**.

## What we tested

The equal-weight, monthly-rebalanced composite of **VLUE + QUAL + MTUM + USMV + SIZE** raced
against **SPY** on the **excess-of-cash Sharpe** (both legs minus the BIL T-bill ETF) over the
**155-month** window common to all five sleeves (2013-08 → 2026-06), as-of 2026-06-30. We report
the Sharpe race gross and **net** of the blend's rebalancing turnover, the HAC *t* on the active
return, a **paired moving-block bootstrap CI on the Sharpe advantage**, a two-era robustness cut,
a diversification decomposition (blend vs mean/best single sleeve; cross-sleeve annual
dispersion), an inverse-vol-weighting robustness alt, and a planted-edge synthetic control that
proves the machinery. Flagship-survivor selection is named on the Signal axis.
**Dedup:** [601-factor-etf-live-test](../601-factor-etf-live-test/) tested **each** ETF's
*exposure* vs SPY — this tests the **combined sleeve as one portfolio**;
[638-value-momentum-everywhere](../638-value-momentum-everywhere/) and
[242-quality-minus-junk](../242-quality-minus-junk/) are the **academic long-short** factors, and
[401-signal-stacking](../401-signal-stacking/) is the generic signal-stacking method demo — this
is its live, long-only, five-ETF realization.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a factor blend smooths the ride, the two halves of the pitch (diversify vs beat-the-market), why the sleeve beats the *average* factor but not SPY, and why it's not a cost story — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the excess-of-cash Sharpe race gross/net, the active-return NW *t*, the paired block-bootstrap advantage CI, the two-era cut, the diversification decomposition, the per-sleeve breakdown, the inverse-vol robustness, and the planted-edge synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`multi_factor/`](multi_factor/). The tested unit is a LIVE equal-weight ETF sleeve,
net of its rebalancing turnover; excess-vs-excess vs SPY, both minus BIL. **Not investment
advice** — research & education. See [LICENSE](../../LICENSE).*
