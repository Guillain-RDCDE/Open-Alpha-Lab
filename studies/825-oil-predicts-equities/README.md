# Study 825 — Oil Predicts Equities 🛢️📉

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does this month's oil change forecast next month's stocks, *negatively* (Driesprong et al)? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The predictive slope of SPY's forward one-month return on the trailing one-month oil (USO) return is **+0.0136** (Newey-West *t* = **+0.33**, R² = **0.11%**) over 2006–2026 (241 months) — statistically **indistinguishable from zero**, and if anything the *wrong* (positive) sign versus the claimed negative one. Next-month SPY after oil rose (+0.83%) ≈ after oil fell (+0.95%). The observed slope is p = **0.59** in a 2,000-draw permutation placebo, and it **flips sign across eras** (*t* = +1.23 / −0.50). A 20-seed synthetic control recovers a *planted* negative slope cleanly (fires on **1/20** nulls, the nominal rate), so the flat real-tape result is a genuine null, not a broken engine. *Survivorship: USO/SPY are continuously-listed ETFs — no delisting bias; USO is a front-month roll proxy for oil, named on the Signal axis.* |
| **Tradability** — can you get paid for it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The long/short timer (`−sign(oil_ret)` of SPY next month) is a coin flip: net **+0.17%/mo**, *t* = **0.58**, hit rate **0.506**. The long/flat variant *looks* significant (*t* = 2.60) only because it inherits the equity risk premium — and it still earns a **lower Sharpe (0.58) than simply buying and holding SPY (0.77)**. The oil forecast subtracts value. |

> **In one sentence:** the famous Driesprong result — a rise in oil this month should
> forecast weaker stocks next month — **does not survive on 2006–2026 US ETFs**; the
> predictive slope is a flat, wrong-signed nothing (NW *t* = +0.33), and no timer built on
> it beats buy-and-hold, so the honest read is **claimed signal absent, paycheck a mirage**.

## What we tested

Driesprong, Jacobsen & Maat (2008), **"Striking Oil: Another Puzzle?"**: a change in the
oil price **this month** predicts the equity market return **next month**, *negatively* —
oil is a slow-diffusing macro shock the market prices in with a lag, so the predictive slope
should be **negative**. We take the self-contained monthly version on **USO (crude-oil ETF)
+ SPY (S&P 500 ETF), yfinance daily adjusted close, total-return, 2006-04-10 → 2026-06-30**:
month-end resample → a single-regressor predictive regression of SPY's **forward one-month**
return on the **trailing one-month** oil return (signal known at the close of month `t`, held
over month `t+1`, one documented lag, zero look-ahead), with a Newey-West HAC *t* on the
slope, its sign and R², a Welch tercile cross-check, a 2,000-permutation placebo, a two-era
robustness cut, a costed monthly timer benchmarked against buy-and-hold, and a 20-seed
synthetic positive control. Survivorship is **named on the Signal axis** (continuously-listed
ETFs — no delisting bias; USO is a front-month roll proxy). **Dedup:**
[245-oil-equity-correlation](../245-oil-equity-correlation/) tests the **contemporaneous**
same-period co-movement, not the **lagged** forecast; [226-crude-seasonality](../226-crude-seasonality/)
tests crude's **calendar** seasonality in the oil price itself, not oil→equity cross-prediction;
[85-dr-copper](../85-dr-copper/) uses **copper** as a *pro-cyclical* (positive) growth barometer,
a different commodity and the opposite predicted sign. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why an oil shock *should* forecast weaker stocks a month later — and why on the real tape the slope is a flat, wrong-signed zero |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the slope Newey-West *t*, the R², the tercile Welch check, the 2,000-permutation placebo, the two-era cut, the costed timer vs buy-and-hold, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`oil_equities/`](oil_equities/). Real tape pulled from yfinance (USO + SPY daily
adjusted close) and cached under the study's own `_cache/`. **Not investment advice** —
research & education. See [LICENSE](../../LICENSE).*
