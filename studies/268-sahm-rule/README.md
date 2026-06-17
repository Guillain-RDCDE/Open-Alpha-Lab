# Study 268 — Sahm-Rule

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

**Is the Sahm recession trigger a usable sell button?**

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | After a trigger (1-month execution lag), the S&P returned **+11.9%** over the next year vs **+8.6%** unconditionally — *higher*, the wrong sign for a sell signal. Welch t = **+0.52**, HAC t = **+0.55**, perm p = **0.77**; n = 12 triggers. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | A 12-month-out overlay cut max drawdown (−52.6% → −38.5%) but lowered CAGR (7.5% → 6.5%) at the **same Sharpe**; active-return HAC t = **−1.01**. No usable sell button. |
| **Busted?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | Not because the rule is wrong — it nails every postwar recession — but because **identifying a recession is not timing the market**. Stocks lead the cycle; unemployment lags it. |

> **In one sentence:** the Sahm Rule is a first-rate recession thermometer and a poor market sell button — by the time joblessness has risen half a point, the decline is largely priced and the forward year is usually the recovery.

## What we tested

The Sahm Rule (Sahm 2019): a recession has begun when the 3-month-average
seasonally-adjusted unemployment rate rises ≥ 0.50pp above its trailing-12-month
minimum. We hardcode the BLS U-3 SA series (1959–2025) in `data.py`, compute the
12 trigger onsets, and run an **event study** on forward S&P 500 price returns with
a one-month execution lag — comparing the post-trigger mean to the **unconditional**
forward return (the correct baseline for a drifting-up asset). We report a Welch t,
a HAC/Newey-West t on the overlapping forward series, and a block-permutation test,
then build a net-of-cost long/flat **timing overlay** and pin it against
buy-and-hold. A synthetic positive control confirms the trigger machinery fires
when (and only when) a recession-sized shock is planted.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the trigger, the late-siren chart, why a recession dater isn't a market timer, in plain English |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | event study, block permutation, HAC t, net-of-cost overlay, horizon robustness, the n=12 power reckoning |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`sahm_rule/`](sahm_rule/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
