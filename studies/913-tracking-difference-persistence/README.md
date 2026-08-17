# Study 913 — Tracking-Difference Persistence 🧾

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the edge real? | ![Mixed](https://img.shields.io/badge/Mixed-dab617?style=flat-square) | Among near-identical S&P 500 ETFs last year's tracking-difference rank predicts **nothing** (Spearman **+0.071**, *t* = +0.43, permutation *p* = 0.96): a 6.45 bp fee spread cannot be read through a **10.7 bp** measurement floor. The *fee* effect underneath is real — cheapest − flagship **+10.64 bp/yr (*t* = +4.85**, HAC +4.78, 11/13 yrs), **+5.21 (*t* = +2.68)** once the ex-post fee-sheet pick is removed; QQQM over QQQ **+8.55 bp/yr (*t* = +3.95**, 5/5 yrs). But the permutation test calls it a **published level, not a memory** — all-pairs rank correlation (+0.195) ≈ consecutive-pair (+0.253). Survivorship (eight funds that still exist) flatters it. |
| **Tradability** — is it bankable? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | What works is a purchase decision, not a strategy: buy the cheapest share class, never trade, collect 3–9 bp/yr with zero turnover. The rotation rule pays **+0.33 bp/yr (*t* = +0.10)** for **10 fund changes in 14 years**, goes negative above 1 bp of switching cost, and is invisible in Sharpe (0.8228 vs 0.8241). Moving an existing taxable holding never pays: a +50% embedded gain at 20% takes **155 years** to repay. |

> **In one sentence:** Tracking-difference ranks persist exactly as far as expense ratios do and not one basis point further — so the fund you should own is the one whose fee sheet you can read today, not the one that won last year's tape.

## What we tested

The fund-picker's folklore, at full strength: two funds hold the same index, one quietly
returns a few basis points more each year, so look up last year's **tracking difference** and
buy whoever won it — good tracking is a skill, and skills persist. Against it, the duller
rule: ignore the tape and buy the lowest published fee. We measure each fund's
complete-calendar-year total return minus its family's mean, test year-over-year Spearman
persistence against a **year-label permutation** null, then race five annually-rebalanced
rules with one execution lag, one-way costs × NAV, no short leg, an era cut, a cost sweep and
a capital-gains break-even. Families: **SPY/IVV/VOO** (2011–2025), the same plus NAV-priced
**VFIAX/FXAIX/SWPPX** (2012–2025), **QQQ/QQQM** (2021–2025); as-of 2026-06-30. Two named
caveats: **SPLG is unavailable** (the source serves one stale bar) so it is declared missing
rather than swapped, and expense ratios are a labelled **ASSUMPTION carrying hindsight** —
today's fee sheet applied to 2012 — which is swept, and re-run without any fee sheet at all.
**Dedup:** distinct from **613-currency-hedged-etf-carry** (same-holdings gap, but a
hundreds-of-bp rate identity), **378-etf-nav-premium** (price-vs-NAV dislocation, not the
annual return gap), **379-etf-lead-lag** (one ETF forecasting another's *return*),
**601-factor-etf-live-test** (varying the strategy inside the wrapper; here only the wrapper
varies) and **139-ai-powered-etf** (a ~70 bp active-fee gap, an order of magnitude coarser).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a tracking difference is, why 2014 and 2016 are data artefacts, the noise floor beating the fee, the tax arithmetic |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | permutation null vs the rank *t*, level-versus-memory, the hindsight-free control, both test units and why they disagree, era cut, cost sweep, the live synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`td_persist/`](td_persist/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
