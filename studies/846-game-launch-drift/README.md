# Study 846 — Blockbuster Game-Launch Drift 🕹️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the publisher (TTWO/EA/NTDOY/UBSFY) move around a blockbuster launch? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | Run-up a faint *dip* (2-week AR −1.58%, *t* = −1.26, Newey-West −1.63) — the **wrong sign** for "buy the hype" and \|*t*\| < 2. The headline **20-day post-launch drift is a clean zero** (−0.17%, *t* = −0.10, hit 16/33). Both inside the placebo cloud (right-tail p = 0.94 / 0.60); drift signs scatter across publishers (UBSFY −5.6% vs NTDOY +2.8%) and flip across eras (early +1.16% vs late −1.42%); jackknife *t* never leaves [−0.32, +1.11]. Survivors-only panel (TTWO/EA/NTDOY/UBSFY still listed). |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Nothing to bank: no significant edge gross or net (20-day drift −0.17% → −0.37% at 10 bps), "buy the hype" is a *losing* long, no post-launch momentum to ride, and timing a telegraphed AAA ship on a large-cap publisher is not a business. |
| **Does the market flinch at a game launch?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | ~33 marquee AAA launches, 2013→2024, a proper event study with a random-window placebo, per-publisher and sub-era splits, a jackknife and a synthetic control all agree: no detectable publisher move around the ship date. |

> **In one sentence:** across ~33 blockbuster AAA launches (2013→2024) TTWO / EA / NTDOY / UBSFY neither reliably rally into the ship date (a faint, insignificant *dip* if anything — the wrong way for "buy the hype") nor drift after it (a clean 20-day zero), so the finance version of "buy the hype into a game launch" is a bust.

## What we tested

The gaming/finance reflex **"buy the hype into a marquee game launch"** — then ride the
momentum or *sell the news* — transplanted onto the tape: does the *publisher's* stock earn
an abnormal return around a blockbuster's ship date and over the ~20-session drift that
follows? We hand-curate **37 marquee AAA launches** (2013→2024 — GTA V, RDR2, Zelda
BotW/TotK, Battlefield, Assassin's Creed, Diablo IV…), map each to the listed publisher it
most plausibly moves (**TTWO / EA / NTDOY / UBSFY**; **ATVI** kept for the record but
tape-less — Activision delisted 2023-10-13 after the Microsoft buyout, so its 4 launches are
excluded, leaving **33 resolvable**), anchor each to *its own* publisher vs **SPY**, and run
an event study — run-up and 20-day post-launch abnormal returns (one-sample *t* and a
Newey-West HAC *t*), a random-window placebo, per-publisher and sub-era splits, a
leave-one-out jackknife, and a costed leg — with a deterministic synthetic tape carrying a
*planted* launch drift as the positive control. **As-of 2026-06-30.** **Dedup:** distinct
from [844-madden-cover-curse](../844-madden-cover-curse/) (the Madden/NBA-2K "cover curse" on
EA + TTWO only, 2-week window), [774-nintendo-direct](../774-nintendo-direct/) (a Nintendo
Direct *broadcast*, not a launch), [771-box-office-bomb](../771-box-office-bomb/) (a film
*flop* shock on the studio) and [550-box-office-momentum](../550-box-office-momentum/) (film
box-office receipts momentum).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a blockbuster launch *should* move the publisher if "buy the hype" is right, the launch calendar, the CAR picture, and why the tape says no |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the one-sample and HAC *t*, per-publisher & sub-era splits, the random-window placebo, the jackknife, the costed leg, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`game_launch/`](game_launch/). The launch calendar is hand-curated from public
release records; TTWO, EA, NTDOY, UBSFY and SPY are fetched via yfinance (total-return); ATVI
is delisted (no tape). Survivorship named on the Signal axis (the four publishers still
listed); one documented execution lag (0 — the launch date is calendar-known months ahead).
**Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
