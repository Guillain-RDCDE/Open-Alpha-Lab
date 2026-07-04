# Study 627 — 13F Cloning 🐘

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the 45-day-late Berkshire clone beat the market? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | On every scriptable month (52 original 13F-HRs, 2013→2026, 154 months) the clone **lost**: EW active **−4.86%/yr** vs SPY at **HAC *t* = −2.11** (significantly *negative*), VW −2.27%/yr (*t* = −1.07); negative in both halves and at top-5/10/15; **below all 200 random managers** drawn from its own 28-name universe. Survivorship (97.5% slot coverage — two delisted acquirees) and the post-2013-XML-only window are named; the claim's supporting literature lives on 1976–2006 tape. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Access is perfect — ten mega-caps, four trades a year, ~40%/yr one-way turnover, **8.7 bps/yr** cost drag at 10 bps one-way — and there is nothing to harvest. Costs are not the problem; the signal is. A flawlessly executable way to lag the index. |
| **"At least beats Berkshire itself (cash drag)?"** | ![Busted](https://img.shields.io/badge/Beats_Berkshire_itself%3F-Busted-8b949e?style=flat-square) | The fully-invested clone should out-run cash-dragging BRK — it didn't: EW clone 8.78%/yr vs BRK-B **12.43%/yr** (−3.40%/yr, *t* = −1.16), VW −0.81%/yr. Berkshire's whole beat its cloneable parts, and SPY beat both. |

> **In one sentence:** the most famous free-rider trade in investing — photocopy Berkshire's
> 13F 45 days late — is fully scriptable since 2013, costs almost nothing to run, and on those
> 154 months it lagged SPY by **−4.9%/yr (EW, HAC *t* = −2.11)**, did worse than **every one
> of 200 random managers** picking from the same names, and even lost to Berkshire itself — a
> perfectly tradable mirage.

## What we tested

We rebuild the claim literally. Every **original 13F-HR** Berkshire Hathaway filed on EDGAR in
the machine-readable era (periods 2013-06-30 → 2026-03-31; amendments excluded — the clone
only sees what was public) is parsed and aggregated by CUSIP; the clone buys the **top-10
holdings by reported value** at the close of the **first trading day after the filing date**
(the 45-day lag is inside the filing date — the study's single execution lag), equal-weight
and 13F-value-weight, weights drifting between filings. Total-return prices; races vs **SPY**
and vs **BRK-B**; monthly active returns and CAPM alpha get **Newey-West HAC *t***, Sharpe is
excess-vs-excess (^IRX). Costs are one-way bps × traded NAV. A 200-draw random-manager
placebo (same universe, calendar and lag) separates Berkshire's *ranking* from its *universe*,
and a deterministic synthetic control proves the harness recovers a planted manager alpha
through the same lagged-filing pipe and stays flat on a null. Two delisted acquirees (DIRECTV,
Activision) cost 2.5% of top-10 slots — named on the Signal axis. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a 13F actually is, how the free-rider trade works, the growth-of-$1 race, why the clone lost to random picks from its own names, and why even Berkshire beat it — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the clone engine (filing-date+1 rebalance, drifting weights), HAC active/alpha *t*-stats, sub-period + top-N robustness, the 200-seed random-manager placebo, costs × turnover, the BRK-B race, and the planted-alpha synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`thirteen_f_cloning/`](thirteen_f_cloning/). Siblings: [263-insider-buying](../263-insider-buying/)
is the **Form 4** (insider-trades) cousin — this is the **Form 13F** holdings-cloning claim,
Berkshire-specific because that is how the legend is told. **Not investment advice** — research
& education. See [LICENSE](../../LICENSE).*
