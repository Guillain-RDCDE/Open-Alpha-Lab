# Study 831 — Gold Real-Yield Timing 🥇

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the real-yield *trend* time gold? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Sorting forward 21-day gold on the lagged real-yield-fall rank, the fastest-falling-yield quintile earned **+0.93%** vs **+1.16%** for the fastest-*rising* — a Q5−Q1 spread of **−0.23%** (faintly the **wrong** way), HAC *t* **−0.36**, placebo *p* **0.73**. Right-signed only at 63-126d and still sub-2 (peak *t* +1.29), tiny and unstable across sub-periods. The *trend* carries no forward edge. |
| **Tradability** — does the timer beat buy-and-hold gold? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Owning GLD when real yields are falling ties buy-and-hold on Sharpe (**0.565 vs 0.564**) only by sitting in cash 52% of the time; its *mean-return* spread is **−1.40 bps/day** (*t* −1.24) at **16.7** switches/yr, and at 5 bps/switch its Sharpe (**0.523**) drops *below* buy-and-hold. |
| **Gold ↔ real-yield inverse link** | ![Confirmed but untradable](https://img.shields.io/badge/Confirmed_but_untradable-8b949e?style=flat-square) | The *contemporaneous* corr(gold return, same-day real-yield change) is **−0.26** at HAC *t* **−9.67** — the famous inverse fact **is real and strong**. But it is same-day co-movement, not a lead: descriptive, not a timing edge. |

> **In one sentence:** gold genuinely co-moves inversely with real yields *contemporaneously* (corr −0.26, HAC *t* −9.67), but the *trend* of real yields does **not** predict forward gold (Q5−Q1 spread −0.23%, HAC *t* −0.36, placebo *p* 0.73), and a costed "own gold when real yields fall" timer ties buy-and-hold by cash-drag while losing on mean return after ~17 switches a year.

## What we tested

The textbook fact: gold has no coupon, so its appeal is inversely tied to the **real** yield on safe
assets (Erb & Harvey 2013; Baur & McDermott 2010; Barsky & Summers 1988). The tradeable twist: a "real
yields are falling → own gold" **timing rule** ought to beat buying and holding gold. Because the
official 10y TIPS real yield (FRED `DFII10`) isn't a Yahoo ticker, we build the cheapest honest proxy —
**TIP total return as an inverse real-yield gauge** (`ryfall = log(TIP_t) − log(TIP_{t−63})`, cross-
checked against a `TNX − breakeven` level from TIP-vs-IEF) — rank it out-of-sample, and test whether
the fastest-falling-real-yield days precede higher forward GLD returns. Tests: the **contemporaneous
inverse-link** cross-check (a HAC *t* on the same-day beta), a Q5−Q1 forward-return sort with a **HAC
(Newey-West) *t***, a **block-shuffle placebo** null, horizon / lookback / sub-period sweeps, a costed
timing overlay vs buy-and-hold GLD, and a deterministic, seed-robust **synthetic positive control** that
plants a *predictive* timing edge (while keeping the contemporaneous link on) and proves the engine
catches it. **Dedup:** distinct from [640 Gold-Overnight](../640-gold-overnight/) (an intraday/overnight
session split), [649 Gold-Seasonality](../649-gold-seasonality/) (a calendar signal), [381
TIPS-Breakeven](../381-tips-breakeven/) (trades the breakeven itself, not gold), and [580
Gold-Lease-Rate](../580-gold-lease-rate/) (a supply-side carry signal) — here the signal is the
*macro real-yield trend* and the traded asset is *gold*.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why gold "should" track real yields, the difference between a *same-day* fact and a *forecast*, and why the timing rule doesn't pay |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the inverse-link HAC *t*, the quintile sort, the placebo null, the horizon/lookback/sub-period sweeps, the timing-overlay cost maths, and the seed-robust synthetic positive control |

The fingerprinted real-data run (GLD + TIP + IEF + `^TNX`, 2004-11-19 → 2026-06-29, 5,434 days, tape fp
`4f27dc5f4b4f`) is in [docs/results.md](docs/results.md); the offline machinery proof runs on the
deterministic synthetic world in [`gold_real_yield/data.py`](gold_real_yield/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`gold_real_yield/`](gold_real_yield/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
