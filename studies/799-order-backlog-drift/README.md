# Study 799 — Order-Backlog Drift 📦

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does YoY growth in a firm's order backlog (RPO) lead its forward returns? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | The tercile long-short (long fast-RPO-growth names, short slow) is positively signed on **every** cut — gross **+100 bps/mo ≈ +12%/yr**, Sharpe **0.57**, hit 58%, and a pooled event-drift that rises monotonically to **+3.4%** at two quarters — but the decisive autocorrelation-robust statistic is **Newey-West *t* = +1.10**, *below* the desk's t ≥ 2 bar. Worse, the whole edge lives in the 2019-2021 SaaS melt-up (**+245 bps/mo**) and is dead since 2022 (**+17 bps/mo**, *t* = +0.17). Literature says the backlog lead is real; this thin post-ASC-606 tape **cannot certify the market underreacts to it.** |
| **Tradability** — can you get paid for it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Turnover is low (a quarterly signal), so costs barely bite — net of 20 bps + 100 bps borrow the point estimate is still +87 bps/mo (Sharpe 0.49). But the net **NW *t* = +0.96** is indistinguishable from zero, and the paper Sharpe lives entirely in one bull regime. An uncertified, regime-concentrated paper edge is not a paycheck. |
| **Does backlog actually lead sales?** | ![Confirmed](https://img.shields.io/badge/Backlog--leads--sales%3F-Confirmed-8b949e?style=flat-square) | Yes — unambiguously. This quarter's RPO growth predicts **next** quarter's revenue growth with slope +0.31 (*t* ≈ +9.9); the top RPO-growth tercile goes on to post **+20.8 pp** faster sales growth than the bottom. The fundamental lead is real; it is the *return* underreaction, not the accounting, that this sample can't stand up. |

> **In one sentence:** a swelling order backlog genuinely front-runs a software firm's
> revenue (the accounting lead is real, *t* ≈ 10) and ranking on it *looks* like a +12%/yr
> long-short — but the forward-return edge never clears the desk's robust-*t* bar
> (**NW *t* = 1.10**), lives entirely in the 2019-21 boom, and is dead since 2022, so the
> honest read is **a real fundamental lead the market doesn't obviously misprice** — WEAK
> signal, MIRAGE paycheck.

## What we tested

The investor folklore that *"order backlog is a leading indicator the market is slow to
price."* The modern, machine-readable proxy for backlog is the ASC-606 disclosure
**Remaining Performance Obligations (RPO)** — the dollar value of contracted revenue a firm
has *signed but not yet recognised*. We take a fixed basket of **38 RPO-disclosing
enterprise-software names**, rank them each month on **year-over-year RPO growth**
(point-in-time at the 10-Q/10-K filing date, zero look-ahead), and run the tercile long-short
net of realistic costs — decisive statistic the Newey-West *t*, cross-checked with a
534-style pooled event drift and a mechanism test (does RPO growth lead *sales*?). **Coverage
is thin and one-regime by construction: RPO did not exist before ASC-606 (2018), so the whole
study lives in a single ~7-year window (2019-06 → 2026-06, 85 months).** **Dedup:** the
sibling [798-deferred-revenue-signal](../798-deferred-revenue-signal/) ranks on the
*billed*-but-unrecognised **deferred-revenue** balance — RPO is the strictly larger,
more-forward *signed-backlog* line item (contracted, not-yet-billed *and* not-yet-recognised);
[199-sales-growth](../199-sales-growth/) ranks on the **realised** income-statement sales
number this study claims RPO *leads*; and [534-revenue-surprise-drift](../534-revenue-surprise-drift/)
is a revenue-*surprise* PEAD around the announcement, not a *level-growth* backlog sort. As-of
**2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what RPO/backlog is, why a swelling backlog *should* be leading information, why ranking on it looks like a +12%/yr trade — and why that edge is a 2019-21 bull-market ghost that doesn't survive honest scrutiny |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the calendar long-short + Newey-West *t*, the era split that kills it, the pooled event-drift placebo, the RPO→sales mechanism regression, the cost/borrow sweep, and the 20-seed synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`order_backlog/`](order_backlog/). RPO from SEC EDGAR `companyconcept`; prices from
yfinance. Survivorship is named on the Signal axis (current-survivors basket). **Not
investment advice** — research & education. See [LICENSE](../../LICENSE).*
