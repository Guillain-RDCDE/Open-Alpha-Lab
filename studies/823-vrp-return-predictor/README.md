# Study 823 — Variance-Risk-Premium Return Predictor 🌩️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the VRP (implied − realized variance) predict the market's forward return (Bollerslev-Tauchen-Zhou)? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | The predictor's **shape replicates exactly**: the slope is **positive** at short horizons and the R² **peaks at the quarterly (3-month) horizon** (+2.12 slope, R² 1.07%), just as the paper claims — and it is **strongly significant in the original era** (1993–2009, HAC *t* = **+5.93**, R² 8.65%, bracketing BTZ's own 1990–2007 sample). But on the **full modern tape the Newey-West *t* is only +0.75** (< 2) and the edge **decays and inverts** after 2010 (*t* = −1.28). Right sign, right horizon, real in its time — not certifiable out-of-sample. *Risk-free leg proxied at 0 — named on the Signal axis; it shifts the intercept, not the slope.* |
| **Tradability** — can you get paid for it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | A VRP-conditioned long/flat market-timer **loses to buy-and-hold SPY** (Sharpe **0.55 vs 0.69**; mean **−0.33 %/month**, NW *t* = −2.21) while churning 5×/yr. |
| **In-sample vs out-of-sample** | ![Decayed](https://img.shields.io/badge/In--sample_vs_out--of--sample-Decayed-8b949e?style=flat-square) | A ~6-*t* edge in 1993–2009 collapses to an insignificant, wrong-signed −1.28 in 2010–2026 — a post-publication fade. |

> **In one sentence:** the celebrated variance-risk-premium return predictor **was real in
> its original era** (HAC *t* ≈ +6 in 1993–2009, with the exact quarterly-peaking R² shape the
> paper describes) but **decays to an insignificant +0.75 on the full 1993–2026 tape and
> inverts after 2010**, and no VRP-timer beats simply holding the index — so the honest read
> is **a genuine edge that faded, not a paycheck**.

## What we tested

Bollerslev, Tauchen & Zhou (2009), **"Expected Stock Returns and Variance Risk Premia"**:
the **variance risk premium** `VRP = IV − RV` (option-implied minus realized variance)
**predicts the aggregate market's forward excess return** — positive VRP → higher forward
return — with the predictive R² peaking at the quarterly horizon. We build the self-contained
monthly version on **SPY + ^VIX daily closes (yfinance, total-return SPY, 1993-01-29 →
2026-06-30, 400 month-ends)**: `IV = (VIX/100)²/12`, `RV` = trailing-21-day sum of daily
squared log returns, and regress the **forward 1- and 3-month SPY return** on `VRP_t` with a
**Newey-West slope *t***, a block-bootstrap placebo, a two-era cut, a costed long/flat timer,
and a 20-seed synthetic positive control. Point-in-time (VRP known at month-end `t`, held
`t → t+h`, one lag, zero look-ahead); the risk-free leg is proxied at 0 — named on the
**Signal** axis (it moves the intercept, not the slope). **Dedup:**
[130-vol-risk-premium](../130-vol-risk-premium/) tests whether the VRP **exists / is
positive** (its level, harvested by shorting variance), not whether it **predicts the
market's direction**; [111-vix-term-structure](../111-vix-term-structure/) tests the **shape**
of the VIX futures curve; [3-fear-gauge](../3-fear-gauge/) tests the VIX **level** as a
contrarian dip signal. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what the variance risk premium is, why a fat one *should* precede higher returns — and how a real 1990s–2000s edge quietly faded after everyone read about it |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the predictive-regression slope with its Newey-West *t*, the OLS-vs-HAC gap, the quarterly-peaking R², the two-era decay, the block-bootstrap placebo, the timer cost math, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`vrp_predictor/`](vrp_predictor/). Real tape: yfinance daily SPY + ^VIX, cached
under this study's own `_cache/`. **Not investment advice** — research & education. See
[LICENSE](../../LICENSE).*
