# Study 959 — Crypto Fee War ₿

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the edge real? | ![Real](https://img.shields.io/badge/Real-1e7f4f?style=flat-square) | The tape hands the fee back. Cheapest − priciest is **+145.8 bp/yr, HAC *t* = +8.43**, bootstrap CI **[+110.5, +187.0]** clear of zero, positive in **26/29** months and in **both** eras, replicated across **nine** independent cheap wrappers (*t* = +5.8 to +10.6), and it survives its own two anchors (trim the ends, or use all 618 sessions as anchors, and it returns **135–142 bp/yr** — so call it ~140, not 145.8). Grayscale's *own* 15 bp Mini Trust beats its 150 bp flagship by **+130.6 bp/yr** against a 135 bp fee gap — same sponsor, same coin, same custodian. But only the *coarse* ranking resolves: the pass-through slope of **−1.12** rides on **one anchor** (drop GBTC and R² falls 0.98 → **0.25**), inside the 19–25 bp tier the rank test is dead (*p* = 0.18–0.41 against a **66 bp/yr** detection floor), and the waiver expiries are **invisible** (no step clears \|*t*\| = 0.8, four of eight have the wrong sign). No survivorship filter — the cohort is the entire January-2024 batch, all still alive — but the sample is **29 months**, one bitcoin cycle, and the fee sheet is read with hindsight. |
| **Tradability** — is it bankable? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | It is a purchase decision, not a strategy. In the ownership race the 140 bp is worth **+0.010 of Sharpe** — invisible against 50% vol — and chasing last quarter's tracking winner *under-performs* owning the cheapest fund and never trading. The long/short is a 140 bp gross spread that **dies above ~135 bp/yr of borrow**, a number this study cannot observe on a permanently hard-to-borrow name. And the holder who actually owns the dear fund needs **7 to 14 years** for the spread to repay the capital-gains bill on switching. Buying today at 20 bp instead of 150 is free and permanent; everything past that is not. |

> **In one sentence:** The widest fee dispersion any ETF category has ever launched with is delivered in full where it is large — ~140 bp of GBTC's leak shows up with *t* = +8, on nine wrappers, in both eras, under every anchor we can test it against, and against Grayscale's own cheap twin — and is completely unreadable where it is small, because a 6 bp fee spread and a six-month waiver both sit far below what 29 months of a 50%-vol tape can resolve.

## What we tested

The ten US spot-bitcoin ETFs approved together and launched on the **same day**, 2024-01-11:
identical asset, identical 16:00 strike, fees from **19 bp to 150 bp**, several launched with
outright waivers. We measure each fund's realised tracking difference three ways (endpoint,
HAC trend slope, non-overlapping monthly), show that only the **fund-versus-fund** comparison
is usable — bitcoin's 24/7 tape against a 16:00 strike puts a **135 bp/day** clock stub into
every fund-versus-spot difference, **15×** the fund-versus-fund noise — then test whether the
fee ranking predicts the outcome ranking (Spearman with a permutation null, plus a
pass-through regression), put a HAC *t* and a block bootstrap on the cheapest-minus-priciest
spread, hunt the waiver expiries as event steps, and cost the three ways to act on it
(switch, taxable switch, long/short with swept borrow). One execution lag, round-trip costs
× NAV on anything that trades, an anchor-trim and all-session-anchor re-estimate of the
headline, an era cut, an out-of-cohort control, as-of 2026-06-30. **Fees and waiver dates are a
labelled ASSUMPTION** — issuer disclosure, not tape — and enter *only* the rank test; the
headline spread uses no fee input at all. **Dedup:** distinct from
**913-tracking-difference-persistence** (does last year's TD rank *persist*, on mature 3–9 bp
S&P 500 trackers — a memory question; here it is a *delivery* question on an 8× dispersion),
**958-spot-btc-etf-basis** (the premium/discount term, which this study differences away as
the nuisance), **618-gbtc-premium-cycle** (the pre-conversion closed-end discount, which ends
on this study's first day), **624-buffer-etf-cost** and **942-inverse-etf-structural-loss**
(leaks in *derivative* wrappers, not a contractual fee on an outright holding), and
**210-crypto-trend** / **632-crypto-xs-momentum** (rules on the coin; here the coin exposure
is held constant and only the wrapper varies).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why the same coin costs 19 bp or 150 bp, why measuring against bitcoin itself is hopeless, the Grayscale twin experiment, and the tax lock-in that lets 150 bp survive |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the three estimators and where the endpoint one lies, the measurement floor, the HAC/bootstrap spread, the rank test's attainable critical value, the waiver event study, the borrow sweep, the live synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`fee_war/`](fee_war/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
