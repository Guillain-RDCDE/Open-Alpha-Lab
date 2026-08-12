# Study 903 — Sector-Neutral Low-Vol 🧮

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the low-vol edge survive once its defensive-**sector** bet is stripped? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | On 50 liquid US mega-caps the low-vol edge **fails to replicate — and inverts.** The sector-neutral long-low-vol / short-high-vol spread is **−3.53 bps/day** (Newey-West *t* = **−2.67**): the *wild* tech mega-caps *out-earned* the calm names (2010–2026). It is significant but *opposite in sign* to the claim, holds in both eras (*t* = −1.93 / −1.90), and sits ≈**3.55σ into the left tail** of a 1,000-permutation placebo. On the anomaly's own **Sharpe** axis the low-vol leg wins no advantage once sector-neutral (**1.05 vs 1.07**). What character the *raw* sort had was largely a **defensive-sector tilt** (its long book is **45% defensive** vs 20% in the universe; neutralising moves the spread only +1.20 bps). A synthetic control confirms the machinery — a pure sector premium fools the raw sort (*t* = +3.46) while the sector-neutral sort stays silent (**0/20**). *Survivorship: current-membership mega-caps — magnitudes are an upper bound.* |
| **Tradability** — can you get paid for it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The sector-neutral book loses money gross and net (**−5.67 bps/day** at 1 bp one-way, −13.67 at 5 bps). Even the data-mined *sign-flip* (long high-vol) earns only +3.53 bps/day gross, which the **2.14 bps/day** round-trip friction at a mere 1 bp already eats — a Mirage in either direction. |

> **In one sentence:** the low-volatility anomaly's naive sort really *is* largely a
> defensive-sector bet (long book 45% Staples + Health Care), but stripping that bet does **not**
> reveal a hidden stock-level edge — on liquid US mega-caps the sector-neutral low-vol spread is
> significantly *reversed* (the wild names out-earned, NW *t* = −2.67) with no Sharpe advantage
> and no version that survives costs, so the honest read is **claimed signal absent, paycheck a mirage**.

## What we tested

The **low-volatility anomaly** (Baker-Bradley-Wurgler 2011; Frazzini-Pedersen 2014): calm stocks
out-earn wild ones risk-adjusted, so a long-low-vol / short-high-vol book should earn a *positive*
spread. The critique: much of a naive low-vol sort is a **defensive-sector** allocation (long
utilities/staples, short tech/energy). We strip that out on a **liquid 50-name US cross-section
(yfinance daily OHLC, total-return, 2010-01-04 → 2026-06-30)**: rank each name's **trailing-63-day
volatility within its own GICS sector** (demean by the sector median), sort point-in-time (signal
known at the close of `t−1`, one shift, zero look-ahead), long the bottom 30% / short the top 30%,
and compare to the raw sort — with a Newey-West *t*, a per-leg Sharpe race, a defensive-tilt
diagnostic, a 1,000-permutation placebo, a two-era cut, a costed timer, and a 20-seed synthetic
positive control (including a sector-confound proof). The universe is a **current-membership**
survivor set (`quantlab.universe` opt-in guard) — named on the **Signal** axis. **Dedup:**
[330-low-volatility-anomaly](../330-low-volatility-anomaly/) is the un-controlled anomaly (SPLV vs
SPHB) this study neutralises; [501-idiosyncratic-volatility](../501-idiosyncratic-volatility/) sorts
on **factor-residual** vol, not total vol with the **sector** removed;
[58-bunker](../58-bunker/) is the **USMV** min-vol ETF held as one fund, not a cross-sectional sort;
[246-defensive-sectors](../246-defensive-sectors/) *trades* the sector rotation this study *removes*.
As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the sector bet made visible, whether the edge survives the strip, and a live proof the demean works |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the raw-vs-neutral spread *t*, the per-leg Sharpe race, the defensive-tilt diagnostic, the 1,000-permutation placebo, the two-era cut, the cost math, and the synthetic control + sector-confound proof |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`sn_lowvol/`](sn_lowvol/). Cross-section pulled through the `quantlab.universe`
survivorship guard (current membership → magnitudes are an upper bound). **Not investment advice**
— research & education. See [LICENSE](../../LICENSE).*
