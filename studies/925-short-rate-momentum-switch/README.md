# Study 925 — Front-End Trend

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the edge real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Excess-of-cash Sharpe advantage over static IEF is **−0.139** (HAC *t* = **−0.20**), paired bootstrap difference CI **[−0.50, +0.22]**. The sign flips across eras (−0.45 / +0.14) and **three times** across the lookback grid; the day-level spread is +1.09 bp/d at HAC *t* = +0.43. Against a **flip-free blend of identical average exposure** (constant 47.8% TLT / 52.2% bills) the rule is **+0.001 gross, −0.028 net** — the timing is worth nothing. Its one impressive result, beating a frequency-matched *random* control, is a **turnover artefact**: +0.27 net but **+0.06 gross** over 30 seeds (0/30 clear *t* ≥ 2), because the coin flips ~2,400 times and the rule 321. **Best counter-number, stated plainly:** the SHY 12-1 expression of the same bet beats *its* matched blend by +0.153 net / +0.158 gross (*t* = +1.92 / +1.97) — still under the bar, and +0.005 in the first half where it held duration 99% of days. No survivorship channel (five named legs, held throughout). |
| **Tradability** — is it bankable? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Worse on every axis than the static holding it replaces: lower excess Sharpe, **higher** vol (11.0% vs 7.0%), **deeper** drawdown (−28.2% vs −23.9%), at ~17 round trips a year. Negative gross (−0.11 at 0 bps) and more negative at every cost tested (−0.50 at 25 bps). It beats IEF in 11 of 20 calendar years for a mean gap of **−0.10 pp/yr** — a coin toss that costs you turnover. |

> **In one sentence:** Trend-following the 3-month bill yield to choose between long duration and cash buys you two genuinely memorable *timing* years — all of 2022 out of bonds, and the 37 late-2023 sessions in which TLT rose +14.7% — while every other big year it posts is simply 20-year duration beta it rode up (2011, 2014) or down (2009, 2015, 2021); over the full 2007-2026 tape it loses to owning IEF, ties a flip-free blend of its own average exposure gross, and its sign depends on a lookback nobody can justify.

## What we tested

Signal = the **3-month (63-day) change in `^IRX`**, the 13-week T-bill yield (a price-only
yield level, signal input only — never a return). Falling → hold **TLT**; rising → hold
**BIL**. Binary, unlevered, **one** execution lag, 2 bps one-way × NAV per switch (an
**ASSUMPTION**, swept 0 → 25 bps); no short leg, so no borrow. Raced **excess-of-cash**
against static **IEF**, static TLT, a frequency-matched random control and a **matched
constant-weight blend** (same average exposure, zero flips) over
SHY∩IEF∩TLT∩BIL∩^IRX 2007-05-30 → 2026-06-30 (total-return ETF closes, `auto_adjust=True`),
with a paired block bootstrap, an era cut, a lookback sweep, a HAC day-level conditional
test, a **seed-swept random control run gross and net**, the **full** 20-year calendar
table, and a **SHY 12-1 momentum** cross-check.
**Dedup:** distinct from **829-global-sovereign-bond-momentum** (cross-sectional *price*
trend across countries), **132-yield-curve-steepener** and **380-curve-roll-down** (curve
*level/slope*, not its change), **864-yield-curve-twist** (curvature, belly instrument),
**924-cut-cycle-duration-extension** (five hardcoded first-cut FOMC dates — this is the
always-on version with no assumed event list), and **826-treasury-duration-bab** (static,
no timing).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why the rule sounds obvious, the two years that sell it, the twenty-year scorecard behind them, the coin-flip trap in plain words, the made-up lookback |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | excess-of-cash race, HAC return-diff *t*, paired bootstrap on the Sharpe difference, matched constant-weight blend control, seed-swept random control gross vs net, HAC day-level conditional test, era and lookback sweeps, live synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`front_end_trend/`](front_end_trend/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
