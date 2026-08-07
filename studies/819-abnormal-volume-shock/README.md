# Study 819 — Abnormal-Volume Shock 📊📣

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do abnormally heavy names go on to *drift up* (Garfinkel-Sokobin)? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The attention/disagreement drift does **not** replicate as a tradable spread on 50 liquid US mega-caps. The specified long-high-abnormal-volume / short-low book earns **+0.55 bps/day** (Newey-West *t* = **+0.54**) — *correctly signed* but statistically indistinguishable from zero. The high-avol book (+7.44 bps) barely tops the low-avol book (+6.89 bps, Welch *t* = +0.21); the spread sits just **0.62σ** into a 1,000-permutation placebo (p = **0.27**) and **flips sign** across the two eras (−0.40 / +0.94). A clean synthetic control (planted *t* = **+21.6**, null ≈ N(0,1)) proves the flat result is the data, not the engine. The disagreement drift is an **earnings-window / small-cap** phenomenon, diluted to nothing on an all-days mega-cap panel. *Survivorship: current-membership mega-caps — magnitudes are an upper bound.* |
| **Tradability** — can you get paid for it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Even the right-signed **+0.55 bps/day** gross tilt is dwarfed by the **2.14 bps/day** round-trip friction at a mere 1 bp one-way — net **−1.59 bps/day** (*t* = −1.49), and −9.59 bps/day at 5 bps. No version of the book survives costs. |

> **In one sentence:** the celebrated abnormal-volume-as-disagreement drift is **in the right
> direction but statistically absent** on liquid US mega-caps (+0.55 bps/day, NW *t* = +0.54,
> sign-flipping across eras), and the tiny tilt is eaten many times over by trading costs — so
> the honest read is **claimed signal absent, paycheck a mirage**.

## What we tested

Garfinkel & Sokobin (2006), **"Volume, Opinion Divergence, and Returns"**: **unusual trading
volume** proxies attention / opinion divergence, so names printing abnormally high volume should
earn a *positive subsequent drift*. We take a self-contained daily version on a **liquid 50-name
US cross-section (yfinance daily OHLC + Volume, total-return, 2010-01-04 → 2026-06-30)**: each
name's **standardised abnormal volume** `(V − mean_60)/std_60` averaged over a 5-day formation
window, sorted point-in-time (signal known at the close of `t−1`, one shift, zero look-ahead),
long the top 30% (high abnormal volume) / short the bottom 30%, with a Newey-West *t* on the
daily spread, a 1,000-permutation placebo, a two-era robustness cut, a costed long-short timer,
and a 20-seed synthetic positive control. The universe is a **current-membership** survivor set
(`quantlab.universe` opt-in guard) — named on the **Signal** axis. **Dedup:**
[512-high-volume-return-premium](../512-high-volume-return-premium/) sorts on the **level** of
volume (a liquidity premium), not volume *relative to a name's own norm*;
[141-turnover-anomaly](../141-turnover-anomaly/) uses **share turnover**, a slow level, not a
short **shock**; [254-wsb-mentions](../254-wsb-mentions/) uses an **exogenous** social-media
attention feed, not this **endogenous** abnormal-volume proxy. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a burst of unexplained volume *should* mark attention/disagreement and a forward drift — and why on mega-caps nothing shows up |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the spread Newey-West *t*, the pooled Welch book test, the 1,000-permutation placebo, the two-era cut, the cost math, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`volume_shock/`](volume_shock/). Cross-section pulled through the
`quantlab.universe` survivorship guard (current membership → magnitudes are an upper bound).
**Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
