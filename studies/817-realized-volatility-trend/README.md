# Study 817 — Realized-Volatility Trend 📈📉

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does rising vol de-rate and falling vol re-rate (a *trend*, not the low-vol *level*)? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The vol *trend* carries **no** reliable cross-sectional signal on 50 liquid US mega-caps. The long-falling-vol / short-rising-vol spread is **+0.94 bps/day** (Newey-West *t* = **+0.86**) — the *claimed* sign, but statistically zero, only **+1.09σ** into a 1,000-permutation placebo (p = 0.12), weak in both eras (*t* = +1.15 / +0.24). Crucially it is **not additive**: it is near-orthogonal to the low-vol *level* sort (corr **+0.065**) yet its alpha net of that level is just **+1.12 bps/day (*t* = +1.02)**. A 20-seed synthetic control recovers a *planted* trend relation cleanly (*t* = +9.49, fires on **0/20** nulls), so the flat result is real, not machinery. *Survivorship: current-membership mega-caps — magnitudes are an upper bound.* |
| **Tradability** — can you get paid for it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The +0.94 bps/day gross edge is smaller than the **2.14 bps/day** round-trip friction at a mere 1 bp one-way, so the book is already net **−1.20 bps/day** at 1 bp and **−9.20 bps/day** at 5 bps. Nothing to trade. |

> **In one sentence:** vol *momentum* — long falling-vol, short rising-vol — sounds like a
> distinct edge from the low-vol *level* anomaly, and it *is* near-orthogonal to it, but on
> liquid US mega-caps it is **just as empty** (spread +0.94 bps/day, NW *t* = +0.86, adds
> nothing on top of the level), so the honest read is **claimed signal absent, paycheck a mirage**.

## What we tested

The claim: a name's **vol trend** = `(trailing 21d realized vol) / (trailing 63d realized vol)
- 1` predicts returns — rising-vol names keep de-rating, falling-vol names re-rate — a vol
*momentum* effect **distinct from the low-vol *level* anomaly** (study 330). We take the daily
cross-sectional version on a **liquid 50-name US cross-section (yfinance daily OHLC,
total-return, 2010-01-04 → 2026-06-30)**: each name's vol trend, sorted point-in-time (signal
known at the close of `t−1`, one shift, zero look-ahead), long the bottom 30% (most falling
vol) / short the top 30% (most rising vol), with a Newey-West *t* on the daily spread, a
1,000-permutation placebo, an **additivity regression against the low-vol level sort**, a
two-era robustness cut, a costed long-short timer, and a 20-seed synthetic positive control.
The universe is a **current-membership** survivor set (`quantlab.universe` opt-in guard) —
named on the **Signal** axis. **Dedup:** [330-low-volatility](../330-low-volatility/) is the
low-vol **level** anomaly (we sort on the *trend* and regress *out* the level — corr is only
+0.065); [501-idiosyncratic-volatility](../501-idiosyncratic-volatility/) is the **level** of
*residual* vol; [6-clockwork-volatility](../6-clockwork-volatility/) is the **calendar
seasonality** of aggregate vol, not a cross-sectional name-by-name trend. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why *accelerating* volatility might warn of lower returns — and why on mega-caps the trend says nothing |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the spread Newey-West *t*, the pooled Welch book test, the 1,000-permutation placebo, the level-vs-trend additivity regression, the two-era cut, the cost math, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`vol_trend/`](vol_trend/). Cross-section pulled through the `quantlab.universe`
survivorship guard (current membership → magnitudes are an upper bound). **Not investment
advice** — research & education. See [LICENSE](../../LICENSE).*
