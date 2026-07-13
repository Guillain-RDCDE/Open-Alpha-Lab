# Study 736 — Sportsbook-Playoffs 🎰

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do betting stocks rally *into* the season? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | DKNG's 10-day run-up into 12 flagship betting seasons is **−1.93%** (the *wrong* sign), one-sample *t* = **−0.55**, positive in only **5/12** (Wilson [19.3%, 68.0%]), random-calendar placebo **p = 0.698**. The basket (*t* = −1.19), the BETZ ETF (*t* = −0.74) and the market-adjusted variant (*t* = −0.41) all agree; neither NFL nor March Madness rescues it. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Buying the run-up produces no cost-surviving edge at any horizon (5/10/20 days); the one positive point estimate (5-day, +216 bps net) rests on *t* = 0.87, a 50% win rate over 12 events, and flips sign as the window widens. Nothing to charge costs against. |
| **Do betting stocks rally into the season?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | The betting *handle* seasonality is real (Sep→Jan NFL peak, March bump); the tradable stock-price *rally* folklore bolts onto it is not — the stocks, if anything, drift *down* into the games. |

> **In one sentence:** across DraftKings' full public operating history (2020→2026) and 12 flagship betting seasons — 6 NFL Wild-Card weekends and 6 March-Madness Round-of-64s — the "buy the sportsbooks ahead of the games" trade returns a *negative* 10-day run-up (**−1.93%**, *t* = −0.55, placebo *p* = 0.70), the basket and the BETZ ETF drift down too, and a costed timer finds no edge at any horizon: the handle really does spike, but a schedule everyone can read a year in advance is already in the price.

## What we tested

Every autumn and January, sell-side notes and betting-industry press repeat that
sportsbook / iGaming stocks are "seasonal" and rally *into* the NFL playoffs and March
Madness, as anticipation of record betting handle gets priced ahead of the games. The
*handle* seasonality is genuinely real (American Gaming Association and state-regulator
monthly releases — encoded here as a small **labelled proxy**). We test the *second*
leap — the tradable stock rally — on DraftKings (DKNG, from its 2020-04-24 SPAC-merger
close), a 5-name sportsbook basket (DKNG/PENN/CZR/MGM/RSI) and the BETZ pure-play ETF,
running an event study on the 10-session run-up before each of **12 hardcoded
first-game dates** (2021→2026) with one zero-look-ahead, calendar-known execution rule,
a right-tail random-calendar placebo, a Wilson hit-rate, a beta-adjusted cross-check
and a costed timer. **As-of 2026-06-30.**

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why the handle really does spike, why the stock rally still doesn't show up, and why "buy the sportsbooks before the games" loses its shine after costs |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the run-up event study, the right-tail placebo, the NFL-vs-NCAA split, the beta-adjusted and basket/ETF cross-checks, the look-elsewhere caveat on the two |*t*| ≥ 2 offsets, the costed timer, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`sportsbook_playoffs/`](sportsbook_playoffs/). The betting calendar is
hardcoded from public NFL/NCAA schedules; DKNG/PENN/CZR/MGM/RSI/BETZ/SPY are fetched via
yfinance (total-return). **DKNG floored at its 2020-04-24 SPAC-merger close** (the
earlier tape is the DEAC cash shell); **survivorship named** — the basket is the
current survivors of the 2021-22 betting-sector shakeout, a bias that points *for* the
rally, not against it. The handle-seasonality series is a **labelled proxy**, never
traded. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
