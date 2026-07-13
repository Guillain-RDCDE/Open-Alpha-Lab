# Study 740 — Drought-Monitor 🏜️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the ag complex jump when a worse Drought Monitor prints? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | Print-day ag-equity abnormal return **−24.28 bps** (*t* = **−0.84**), grain **−71.01 bps** (*t* = **−1.30**) — both the *wrong* sign, random-calendar placebo *p* = **0.837 / 0.958**. The drought-*regime* split is Welch *t* = **−0.11** (high-drought months slightly *behind* the market). |
| **Tradability** — could you "buy the drought"? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Long the ag basket at the print close nets, at best, **+67 bps** over 5 days — *t* = **+1.36**, short of the bar — and *decays* toward the always-hold baseline by 21 days. No horizon clears *t* ≥ 2 net of costs. |
| **Does a drought print move the ag complex?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | 21 escalations, an event study, a random-calendar placebo, a grain-vs-equity paired test and a regime split all agree: the weekly Monitor is old news to a market that prices the weather forecast in real time. |

> **In one sentence:** across 21 of the biggest US drought escalations since 2000, the day
> the Drought Monitor confirms the bad news the ag complex (Deere, Mosaic, ADM, the ag/grain
> ETFs) moves by a statistically-nothing **−24 bps vs the market** (*t* = −0.84, placebo
> *p* = 0.84), grain reacts if anything *less*, "buying the drought" never clears the bar
> net of costs, and high-drought months carry no forward ag edge — the weekly monitor is
> old news to a market that priced the forecast weeks earlier.

## What we tested

The tradable story writes itself: severe drought across the US crop belt means a smaller
harvest, pricier grain, and a tailwind for the ag names — so if the weekly
[US Drought Monitor](https://droughtmonitor.unl.edu) (published every Thursday since 2000)
prints a worsening picture, buy the drought. We steelman it on a hand-curated table of
**21 major US drought-intensification episodes, 2000→2025**, each dated to a representative
Thursday Monitor release, run an event study on an **ag-equity basket** (DE/MOS/ADM/MOO)
and a **grain basket** (DBA/CORN/WEAT), measured **abnormal of SPY**, around each print's
first tradable session (one execution lag: the Monitor is public that morning, so you enter
at the release-day close — zero look-ahead), with a random-calendar placebo, a grain-vs-
equity paired test, a costed "buy the drought" timer, and a drought-*regime* split on a
clearly-labelled monthly severity proxy. A synthetic tape with a *planted* print-day bump
is the positive control. **As-of 2026-06-30.**

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a drought *should* move the ag complex if the supply-shock story is right, what the tape actually shows on the print, and why "buy the drought" doesn't pay |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the event-study anatomy, the random-calendar placebo, the look-elsewhere caveat on the one nominally-significant offset, the grain-vs-equity paired test, the costed timer, the regime split, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`drought_monitor/`](drought_monitor/). The drought calendar is hand-curated from
US Drought Monitor archive reporting; the monthly D2+ severity series is a **labelled
proxy** (approximate, cited, used only for the regime test — never under a real-tape
banner); SPY / DE / MOS / ADM / MOO / DBA / CORN / WEAT are fetched via yfinance, grain-ETF
coverage named honestly (16 of 21 events fall in the DBA/CORN/WEAT era). **Not investment
advice** — research & education. See [LICENSE](../../LICENSE).*
