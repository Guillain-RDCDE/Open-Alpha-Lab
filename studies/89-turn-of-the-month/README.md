# Study 89 — Turn-of-the-Month 📅

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | On the SPY total-return tape (1993–) the TOM-minus-rest premium is **+3.6 bps/day** but **HAC *t* = +1.17** — below the bar. The long price-only `^GSPC` history (1950–) *does* clear it (*t* = **+4.97**), so the literature's effect is real over deep history; this modern, dividend-inclusive sample alone can't certify it. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | A TOM-only timer earns **2.20%/yr vs 10.82%** — **−8.6 pts/yr** — because it sits in cash ~81% of the time and forgoes the equity premium. Its *per-day-invested* Sharpe (0.895) beats buy-and-hold (0.646), but you can't bank that: 803 switch-costs and four invested days a month leave you far poorer. |
| **Almost all the gains?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | The window is **19.1%** of trading days and earns **32.9%** of SPY's total return — a real ~1.7x over-representation, but a long way from the folklore *"almost all."* |

> **In one sentence:** the turn-of-the-month really *is* over-represented — about a third of the market's return lands in a fifth of the days — but it's **not "almost all,"** the daily premium **doesn't clear the significance bar on the modern total-return tape**, and a **TOM-only timer trails buy-and-hold by ~8.6 points a year** because it spends the rest of the month in cash.

## What we tested

The calendar-anomaly classic, stated at full strength: *"almost all of the stock market's gains happen in a narrow window around the turn of the month — the last trading day plus the first three of the next. Be invested only then and you capture the market with a fraction of the risk."* (Ariel 1987; Lakonishok & Smidt 1988; McConnell & Xu 2008.) We take it literally — long SPY (total return) during the calendar-known `[-1, +3]` window, in cash (earning **0%**) otherwise. The window is knowable in advance, so there is **no execution lag** (a calendar rule is not a forecast). We charge **5 bps/switch** and pin it against buy-and-hold on both raw and per-day-invested terms, measure the share of total return earned inside the window, run a HAC *t* on the TOM-minus-rest daily difference, and run an early-vs-late decay test. A deterministic synthetic tape with a *planted* TOM bump is the positive control (the harness banks it at *t* ≈ 4; on a flat i.i.d. tape it correctly finds nothing).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the window, the cumulative-return split, the bar chart of return by day-of-month, why a TOM-only timer ends up far behind just holding |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC *t* on the TOM-minus-rest difference, total-return vs price-only samples, per-day-invested Sharpe, the block-bootstrap decay test |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`turn_of_the_month/`](turn_of_the_month/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
