# Study 869 — 52-Week-High Breakout Drift 🚀

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — after a fresh 52-week-high *breakout*, does the name drift up or fade? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | The breakout name **does out-earn the rest** — the sign is **breakout momentum, not fade**: **+7.21 bps** forward-5-day, **+21.64 bps** forward-20-day (breakout book beats the rest at both horizons). But the honest, overlap-corrected **Newey-West *t* is only +1.14 / +1.20** — below the bar. The permutation placebo puts it at **p ≈ 0.08–0.10**, and the whole drift is a **2018-2026** phenomenon (2010-2017 is flat-to-negative, *t* = −0.24 / +0.57), so it does **not hold across eras**. A 20-seed synthetic control recovers a *planted* breakout drift cleanly (*t* = +9.62, fires on **1/20** nulls), so the weak real drift is a feature of the tape, not machinery. Directionally right, statistically absent. *Survivorship: current-membership mega-caps — survivors over-print new highs, so magnitudes are an upper bound.* |
| **Tradability** — can you get paid for it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Even at an optimistic **1 bp** one-way the net edge is insignificant (net **+2.52 bps**/5-day, *t* = +0.60; +14.90/20-day, *t* = +1.71, Sharpe ≈ 0.1); at a realistic **5 bps** it turns **negative** (−13.48 / −1.10 bps). No cost leaves a paycheck. |

> **In one sentence:** a fresh 52-week-high breakout *is* followed by a small drift **up**
> (momentum, not the resistance-fade), but on 50 liquid US mega-caps the effect is only +7–22
> bps, fails the HAC significance bar (*t* ≈ 1.2), lives entirely in 2018-2026, and dies at any
> realistic cost — so the honest read is **directionally there, statistically weak, paycheck a
> mirage**.

## What we tested

Trading-desk folklore ("buy new highs") versus the anchoring/resistance story: when a stock
**closes at a fresh 52-week high** — the discrete breakout *event* — does it **drift up** or
**fade**? We flag every new-52-week-high day point-in-time (`Close[t]` strictly tops the rolling
252-day maximum of *prior* closes, one shift, today excluded) on a **liquid 50-name US
cross-section (yfinance daily OHLC, total-return, 2010-01-04 → 2026-06-30)**, enter one day later
(`Close[t+1]`, one documented execution lag, zero look-ahead), and measure the forward **5- and
20-day** return of a long-just-broke-out book vs the rest — with a Newey-West *t* (lags = 2×
horizon, the forward windows overlap), a 1,000-permutation placebo, a two-era robustness cut, a
costed long-short timer, and a 20-seed synthetic positive control. The universe is a
**current-membership** survivor set (`quantlab.universe` opt-in guard) — named on the **Signal**
axis. **Dedup:** [236-fifty-two-week-high](../236-fifty-two-week-high/) tests George-Hwang
**nearness** (a continuous `close/high` *level*), not the discrete **breakout event**;
[202-fifty-two-week-low](../202-fifty-two-week-low/) is the symmetric **low**-side anchor;
[331-fifty-two-week-range](../331-fifty-two-week-range/) is position **within** the 52-week
*range*; [437-donchian-breakout](../437-donchian-breakout/) is the generic **Donchian** n-day
channel breakout, not the fixed 52-week-high event. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why breaking a 52-week high *should* mean something (momentum vs resistance), the forward-return comparison, and why the drift here is real in sign but too weak to trust |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the spread Newey-West *t* at both horizons, the pooled Welch book test, the 1,000-permutation placebo, the two-era cut, the cost math, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`breakout_high/`](breakout_high/). Cross-section pulled through the
`quantlab.universe` survivorship guard (current membership → magnitudes are an upper bound).
**Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
