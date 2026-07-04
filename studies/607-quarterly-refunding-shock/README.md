# Study 607 — Quarterly Refunding Shock 🏛️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the QRA move the long end on announcement day? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | On the **69 FOMC-clean** QRA days (35% of the 106 QRA Wednesdays 2000→2026 are *also* FOMC statement days!), the 10Y moves **×0.90** of an ordinary day (3.86 vs 4.28 bps, Welch *t* = **−1.12**; placebo **p = 0.82**); the 30Y is ×1.10 (*t* = +0.73); signed day-0 drift nil. The lone window bump (day+2, *t* = +4.10) is the **jobs report** — 91% of those sessions are first-Fridays, and without them it's *t* = +0.01. No survivorship: full official record. |
| **Tradability** — can you trade TLT off the announcement? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Entry at the day-0 close (the one lag), 3-day hold: **−14.3 bps/event gross** (*t* = −0.89) unconditional, **−21.3 bps/event** riding the day-0 sign (*t* = −1.33) — negative *before* costs, −73 to −126 bps/yr after. Nothing exists to harvest. |
| **"2023 made QRA-day a macro event"?** | ![Busted](https://img.shields.io/badge/2023_regime_change%3F-Busted-8b949e?style=flat-square) | Vol-normalised, FOMC-clean 2023+ QRA days run **0.99×** their own era's noise vs 0.95× pre-2023 (Welch *t* = **+0.10**). Aug-2 2023 moved **+2.7 bps** (0.51× local noise, Fitch downgrade in the same tape); Nov-1 2023's −8.6 bps had a **same-day FOMC**. Two loud *weeks* with other catalysts — not a QRA-day regime. |

> **In one sentence:** across all 106 Quarterly Refunding Announcements since 2000 the 10-Year
> actually moves *slightly less* on QRA day than on an ordinary day (×0.90, Welch *t* = −1.12,
> placebo p = 0.82) — the storied 2023 "QRA shocks" dissolve into a Fitch downgrade, a same-day
> FOMC and the first-Friday jobs report — so **None, Mirage, and the 2023-regime story Busted**.

## What we tested

All **106 QRA statement dates 2000-02-02 → 2026-05-06**, hardcoded from the official
TreasuryDirect auction records (the mid-quarter refunding securities are announced *via* the
refunding statement, so their `announcementDate` **is** the QRA date; all 106 are Wednesdays,
08:30 ET), against ^TNX/^TYX daily yield changes and TLT (total-return). The catch the daily
bar demands: **35% of QRA days are also FOMC statement days** (both calendars pick
early-quarter Wednesdays — 2023-11-01 included), so the primary Welch *t* on day-0 |Δy| runs
on FOMC-clean QRA days vs an event-free baseline, backed by a 2,000-draw random-calendar
placebo, a [-1..+3] window profile with a **first-Friday (jobs-report) collision check** at
day+2, and a vol-normalised era split at 2023 (the claim's own break date; n = 14 since,
honest small-n). Tradability holds TLT from the day-0 close (one documented lag) for 3
sessions, gross and net at 2/5 bps one-way. A deterministic synthetic control (planted
event-day vol multiplier / signed mean vs the null) proves the machinery. Sibling
[603-treasury-auction-concession](../603-treasury-auction-concession/) tests the **auctions**
(execution of the supply); this study tests the **announcement** of the plan — different
events, opposite verdicts. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what the QRA is, why 2023 made it famous, why the famous days shrink on inspection (Fitch, FOMC, jobs Friday), and what "the announcement was already priced" means — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the FOMC decontamination, Welch day-0 tests + placebo, the [-1..+3] profile with the first-Friday collision, the vol-normalised era split, TLT costs, and the synthetic power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quarterly_refunding_shock/`](quarterly_refunding_shock/). Event table hardcoded
from TreasuryDirect (source in `data.py`); FOMC calendar shared with studies 517/602. TLT is
total-return; the one execution lag is the day-0 close. **Not investment advice** — research
& education. See [LICENSE](../../LICENSE).*
