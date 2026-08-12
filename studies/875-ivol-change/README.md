# Study 875 — Idiosyncratic-Vol Change 📈

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does *rising* idiosyncratic vol precede lower returns (a *change*, not the idio-vol *level*)? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The *change* in idiosyncratic (market-model residual) vol carries **no** reliable cross-sectional signal on 50 liquid US mega-caps. The long-falling / short-rising spread is **+0.87 bps/day** (Newey-West *t* = **+0.86**) — the *claimed* sign, but statistically zero, only **+0.99σ** into a 1,000-permutation placebo (p = 0.17). Decisively, it is **not robust across eras**: significant **+2.31** in 2010–2017, then **−0.63** and sign-flipped in 2018–2026 — a one-era artefact. It is also **not additive**: near-orthogonal to the idio-vol *level* sort (corr **+0.216**) yet its alpha net of that level is just **+1.47 bps/day (*t* = +1.52)**. A 20-seed synthetic control recovers a *planted* relation cleanly (*t* = +8.43, fires on **1/20** nulls — the nominal 5%), so the flat result is real, not machinery. *Survivorship: current-membership mega-caps — magnitudes are an upper bound.* |
| **Tradability** — can you get paid for it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The +0.87 bps/day gross edge is smaller than the **2.14 bps/day** round-trip friction at a mere 1 bp one-way, so the book is already net **−1.27 bps/day** at 1 bp and **−9.27 bps/day** at 5 bps. Nothing to trade. |

> **In one sentence:** a *rising* idiosyncratic vol sounds like a warning of lower returns —
> a distinct signal from the idio-vol *level* puzzle — and it *is* near-orthogonal to that
> level, but on liquid US mega-caps it is **just as empty** (spread +0.87 bps/day, NW *t* =
> +0.86, alive in one era only, adds nothing on top of the level), so the honest read is
> **claimed signal absent, paycheck a mirage**.

## What we tested

The claim: the **change** in a name's idiosyncratic (market-model residual) volatility —
`recent-21d residual vol − prior-21d residual vol` — predicts returns; **rising** idio-vol (a
deteriorating information environment / rising disagreement) precedes **lower** returns,
**falling** idio-vol re-rates. This is distinct from the idio-vol *level* puzzle. We take the
daily cross-sectional version on a **liquid 50-name US cross-section (yfinance daily OHLC,
total-return, 2010-01-04 → 2026-06-30)**: the market factor is the equal-weight
cross-sectional mean return, each name's residual vol is computed vectorised via
`var(r) − cov(r,mkt)²/var(mkt)`, and we sort point-in-time (delta-IVOL known at the close of
`t−1`, one shift, zero look-ahead) long the bottom 30% (most falling) / short the top 30%
(most rising), with a Newey-West *t* on the daily spread, a 1,000-permutation placebo, an
**additivity regression against the idio-vol level sort**, a two-era robustness cut, a costed
long-short timer, and a 20-seed synthetic positive control. The universe is a
**current-membership** survivor set (`quantlab.universe` opt-in guard) — named on the
**Signal** axis. **Dedup:** [501-idiosyncratic-volatility](../501-idiosyncratic-volatility/)
is the **level** of residual vol (we sort on the *change* and regress *out* the level — corr
is only +0.216); [817-realized-volatility-trend](../817-realized-volatility-trend/) is the
trend in **total** vol (we use the **residual** vol, stripping the common market move);
[330-low-volatility](../330-low-volatility/) is the low-**total**-vol *level* anomaly. As-of
**2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why *accelerating* idiosyncratic noise might warn of lower returns — and why on mega-caps the change says nothing |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the spread Newey-West *t*, the pooled Welch book test, the 1,000-permutation placebo, the level-vs-change additivity regression, the two-era cut, the cost math, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`ivol_change/`](ivol_change/). Cross-section pulled through the `quantlab.universe`
survivorship guard (current membership → magnitudes are an upper bound). **Not investment
advice** — research & education. See [LICENSE](../../LICENSE).*
