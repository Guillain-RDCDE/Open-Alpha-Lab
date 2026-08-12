# Study 882 — Gas-Price → Discretionary ⛽

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does a rising gas price forecast consumer-discretionary (XLY) lagging staples (XLP) next month — the "pump tax" rotation? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The predictive slope of the XLY−XLP forward one-month spread on the trailing one-month gasoline (RB=F) return is **−0.0171** (Newey-West *t* = **−0.71**, R² = **0.25%**) over 2005–2026 (256 months). The point estimate has the *right* (negative) sign and the tercile split leans the predicted way (fwd XLY−XLP +0.19% after cheap gas vs −0.08% after dear gas), but the slope is **statistically indistinguishable from zero**: p = **0.43** in a 2,000-draw placebo, and insignificant in both eras (*t* = −0.16 / −0.88). The parallel energy tilt (XLE−SPY) is flatter still (*t* = **−0.11**). A 20-seed synthetic control recovers a *planted* pump-tax slope cleanly (*t* = −5.92), so the flat real-tape result is a genuine null, not a broken engine. *Survivorship: RB=F/XLY/XLP/XLE/SPY are continuously-listed futures/ETFs — no delisting bias; RB=F is a front-month RBOB roll proxy for the pump price, named on the Signal axis.* |
| **Tradability** — can you get paid for it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The spread timer (`−sign(gas_ret)` of the XLY−XLP book) is a coin flip: hit rate **0.504**, gross +0.089%/mo, **net +0.028%/mo** at 1 bp and **−0.046%/mo** once you pay realistic two-leg costs (Sharpe ≈ 0). No paycheck. |

> **In one sentence:** the intuitive "pump tax" — a gas spike should rotate you out of
> discretionary into staples (and into energy) — is **directionally sensible but
> economically absent** on 2005–2026 US ETFs; the predictive slope is the right sign yet a
> flat, insignificant −0.0171 (NW *t* = −0.71), and no timer built on it clears costs, so the
> honest read is **claimed signal absent, paycheck a mirage**.

## What we tested

The pump-tax folklore: a rise in the **gasoline price this month** is a tax on the consumer's
wallet, so it should forecast **consumer-discretionary (XLY) underperforming staples (XLP)
next month** and a **tailwind for energy (XLE)**. We take **RB=F (RBOB gasoline futures) +
XLY + XLP + XLE + SPY, yfinance daily adjusted close, total-return, 2005-01-03 →
2026-06-30**: month-end resample → a single-regressor predictive regression of the XLY−XLP
**forward one-month** spread on the **trailing one-month** gas return (signal known at the
close of month `t`, held over month `t+1`, one documented lag, zero look-ahead), with a
Newey-West HAC *t* on the slope, its sign and R², a Welch tercile cross-check, a parallel
energy-tilt regression, a 2,000-permutation placebo, a two-era robustness cut, a costed
monthly spread timer, and a 20-seed synthetic positive control. Survivorship is **named on
the Signal axis** (continuously-listed futures/ETFs — no delisting bias; RB=F is a
front-month RBOB roll proxy). **Dedup:**
[825-oil-predicts-equities](../825-oil-predicts-equities/) forecasts the **aggregate
market** from **crude**, not a sector rotation from **gasoline**;
[245-oil-equity-correlation](../245-oil-equity-correlation/) is the **contemporaneous**
oil↔equity co-move, not a lagged forecast; [226-crude-seasonality](../226-crude-seasonality/)
is crude's **calendar** seasonality; [639-gasoline-rvp-seasonality](../639-gasoline-rvp-seasonality/)
is gasoline's **own** summer-blend seasonality, not gas as a cross-asset predictor. As-of
**2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a gas spike *should* rotate you out of discretionary into staples — and why on the real tape the slope is the right sign but a flat, insignificant zero |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the slope Newey-West *t*, the R², the tercile Welch check, the energy-tilt regression, the 2,000-permutation placebo, the two-era cut, the costed spread timer, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`gas_discretionary/`](gas_discretionary/). Real tape pulled from yfinance (RB=F +
XLY + XLP + XLE + SPY daily adjusted close) and cached under the study's own `_cache/`.
**Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
