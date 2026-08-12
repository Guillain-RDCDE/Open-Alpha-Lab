# Study 891 — Insurance-Float-Engine 🛡️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does a P&C-insurer basket beat the market on the float? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | Over **2007-06 → 2026-06 (229 months)** KIE/IAK excess-of-cash Sharpe (**0.40 / 0.36**) *trails* SPY (**0.64**); the excess-vs-excess advantage is **−0.25 / −0.29** with the return difference statistically zero (**HAC *t* = −0.49 / −0.90**), and CAPM alpha is **−2.5 / −3.1 %/yr** (*t* < 1). The decisive two-factor test — insurer excess on [market, KBE−SPY bank spread] — collapses the alpha to **−0.11 %/yr (*t* = −0.04)** against a large financial-sector loading (**+0.36, *t* = 6.3**). The advantage is negative in every calm era; the bootstrap can't separate the insurer Sharpe from zero. The float is real economics but at the basket level it *is* sector beta. (Honest aside: insurers do out-Sharpe *banks* by +2.9 %/yr, but only *t* = 0.96 — a different claim.) |
| **Tradability** — anything to pocket over the market? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Nothing. The long-insurer / short-SPY isolation trade is **−1.4 %/yr gross, −1.9 %/yr net** (borrow + costs); a one-month-lag "own insurers when they've led" rotation nets **+9.5 %/yr, below always-SPY's +11.4 %/yr**. The apparent engine is financial-sector beta you can rent more cheaply — and at a shallower drawdown (SPY −51 % vs KIE −70 %) — by just owning the market. |

> **In one sentence:** Buffett compounded on insurance float, but a plain listed-insurer basket
> does *not* inherit it — KIE/IAK lagged the market on return, Sharpe, and drawdown over 19 years,
> earned a **negative** CAPM alpha, and the moment you add a single financial-sector factor the
> "float premium" vanishes to zero (*t* = −0.04): the basket *is* sector beta, not a structural edge.

## What we tested

A P&C insurer holds premiums as **float** before paying claims — near-zero-cost leverage, the
engine behind Berkshire. We test whether a liquid insurer basket turns that into a genuine
risk-adjusted edge over the market: **KIE** (SPDR S&P Insurance, equal-weight) and **IAK**
(iShares U.S. Insurance) vs **SPY**, both legs **excess-of-cash** (minus **BIL**), with **KBE**
(SPDR S&P Bank) as the financial-sector control. On yfinance total-return closes we run the
excess-vs-excess Sharpe race, a **Newey-West HAC *t*** on the return difference, a bootstrap
Sharpe CI, a CAPM and a two-factor (market + bank-spread) decomposition, an era cut, drawdowns
and a calendar-year table, plus a one-month-lag rotation and a costed long-insurer/short-market
isolation trade. A deterministic synthetic world with a **planted, tunable float edge** proves
the estimator recovers a real edge and that a zero-edge null cannot fire.
**Dedup:** distinct from [628-buffetts-alpha](../../628-buffetts-alpha/) (decomposes *Berkshire
itself*, not a listed-insurer basket), [51-blue-chip-quality](../../51-blue-chip-quality/) (a
cross-sectional gross-profitability factor across all sectors),
[246-defensive-sectors](../../246-defensive-sectors/) (the low-vol sector-rotation claim — and
insurers here are *not* defensive, β = 1.1, DD −70 %), and [340-bank-loans](../../340-bank-loans/)
(floating-rate bank-loan *credit*, not bank/insurer equity). As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what "float" is, why Buffett loved it, and why a basket of ordinary insurers doesn't turn it into free money — in plain language, with the race and the one chart that matters |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the excess-vs-excess Sharpe race, HAC *t* on the diff, bootstrap Sharpe CIs, the CAPM and two-factor decompositions that kill the alpha, era splits, the costed isolation/rotation trades, and the planted-edge synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`insurance_float/`](insurance_float/). Both legs are excess-of-cash (minus BIL); the
decisive number is the two-factor alpha (insurer excess on market + bank-spread) that collapses
to zero. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
