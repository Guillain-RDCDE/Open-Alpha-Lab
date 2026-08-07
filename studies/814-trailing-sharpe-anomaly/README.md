# Study 814 — Trailing-Sharpe Anomaly 📐📈

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does long-high-Sharpe / short-low-Sharpe pay, and does risk-adjusting *beat* plain momentum? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The trailing-12-1-Sharpe sort earns **+1.29 bps/day** (Newey-West *t* = **+0.83**) on 50 liquid US mega-caps — right sign, **not significant**. And it does **not** beat what it is built from: the signal is **0.95 rank-correlated with plain 12-1 momentum** and carries no low-vol tilt (+0.09), so risk-adjusting adds nothing — the Sharpe book is a hair *weaker* than plain momentum (+1.59 bps, *t* +0.99), and **neither clears \|t\| ≥ 2**. It sits only ~1.25σ into a 1,000-permutation placebo, is flat in both eras, and a 20-seed synthetic control recovers a *planted* effect cleanly (fires on 1/20 nulls, the nominal 5%). Risk-adjusted momentum is just **momentum repackaged**. *Survivorship: current-membership mega-caps — magnitudes are an upper bound.* |
| **Tradability** — can you get paid for it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The gross edge is insignificant and already flips **net-negative at 1 bp** one-way (−0.84 bps/day; *t* −0.51), collapsing to −8.84 bps/day at 5 bps as the 2.14 bps/day round-trip friction eats it whole. |

> **In one sentence:** a trailing-Sharpe sort *is* plain 12-1 momentum wearing a
> risk-adjusted hat (0.95 rank-correlated, no low-vol content) — it earns an insignificant
> +1.29 bps/day, is a touch *weaker* than the momentum it repackages, and dies at the first
> basis point of cost, so **risk-adjusting buys nothing** here.

## What we tested

Risk-adjusted momentum (Rachev / Biglova et al; Jegadeesh-Titman 12-1 skeleton): rank a
liquid US cross-section on each name's **trailing 12-month Sharpe ratio** — mean ÷ std of
daily returns over the ~252-day formation window, **skipping the most recent month** — and
go **long the top 30% (high Sharpe), short the bottom 30% (low Sharpe)**, equal-weight. We
take the self-contained daily version on a **50-name US cross-section (yfinance daily OHLC,
total-return, 2010-01-04 → 2026-06-30)**, sorted point-in-time (signal known at the close of
`t−1`, one shift, zero look-ahead), with a Newey-West *t* on the daily spread, a
1,000-permutation placebo, a two-era cut, a costed long-short timer, a 20-seed synthetic
positive control — and, crucially, a **head-to-head against plain 12-1 momentum and a pure
low-vol sort** to answer whether risk-adjusting earns its keep. The universe is a
**current-membership** survivor set (`quantlab.universe` opt-in guard) — named on the
**Signal** axis. **Dedup:** [507-cross-sectional-momentum](../507-cross-sectional-momentum/)
is the **plain 12-1 momentum** we grade against (0.95 rank-correlated here);
[8-true-strength](../8-true-strength/) is a **trend/oscillator** indicator, not a moment
ratio; [330-low-volatility-anomaly](../330-low-volatility-anomaly/) sorts on the
**denominator alone** (volatility); [237-residual-momentum](../237-residual-momentum/) uses
**factor-residual** momentum, not total-return Sharpe. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why dividing momentum by its own volatility *sounds* smarter — and why, on mega-caps, it just re-picks the same winners |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the spread Newey-West *t*, the Sharpe-vs-momentum-vs-lowvol head-to-head and rank overlap, the 1,000-permutation placebo, the two-era cut, the cost math, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`trailing_sharpe/`](trailing_sharpe/). Cross-section pulled through the
`quantlab.universe` survivorship guard (current membership → magnitudes are an upper bound).
**Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
