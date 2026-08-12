# Study 844 — Madden-Cover-Curse 🎮

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the publisher (EA/TTWO) move around a Madden/NBA 2K launch? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | Run-up a clean zero (2-week AR +0.05% / −0.17%, \|*t*\| < 0.25, hit 10/20). Post-launch drift, if anything, *negative* (1-week −1.21%) — but **not significant** (*t* = −1.89, Newey-West *t* = −1.56), the **wrong sign** for "buy the hype", placebo right-tail **p = 0.82**, and gone under EA-vs-TTWO (*t* −0.24) and early-vs-late (*t* −0.04) splits. Survivors-only panel (EA + TTWO both still listed). |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Nothing to bank: no significant edge gross or net (2-week drift −0.78% → −0.98% at 10 bps), "buy the hype" is a *losing* long, the mirror short is insignificant, and a two-week fade on a large-cap publisher around a telegraphed annual ship is not a business. |
| **Does the market flinch at a game launch?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | 20 Madden + NBA 2K launches, 2015→2024, a proper event study, a random-window placebo, per-publisher and sub-era splits and a synthetic control all agree: no detectable publisher move around the ship date. |

> **In one sentence:** across 20 *Madden* and *NBA 2K* launches (2015→2024) EA and TTWO neither reliably rally into the ship date (a clean zero) nor pop after it (a faint, insignificant *dip* if anything — the wrong way for "buy the hype"), so the finance version of the "Madden curse" is a bust.

## What we tested

The gaming **"Madden curse"** — the cover athlete gets hurt — transplanted onto the
tape: does the *publisher* (**EA** for *Madden*, **TTWO** for *NBA 2K*) earn an abnormal
return around the annual game launch? We hand-curate **20 launches** (Madden 16→25 +
NBA 2K16→25, real US street dates + cover athletes from Wikipedia / publisher press
releases), anchor each to *its own* publisher vs **SPY**, and run an event study — run-up
and post-launch abnormal returns (one-sample *t* and a Newey-West HAC *t*), a random-window
placebo, per-publisher and sub-era splits, a leave-one-out jackknife, and a costed leg —
with a deterministic synthetic tape carrying a *planted* launch drift as the positive
control. **As-of 2026-06-30.** **Dedup:** distinct from
[720-super-bowl-advertiser](../720-super-bowl-advertiser/) (advertisers, not publishers),
[774-nintendo-direct](../774-nintendo-direct/) (a Nintendo Direct broadcast, not a launch),
[550-box-office-momentum](../550-box-office-momentum/) (film receipts momentum) and
[846-game-launch-drift](../846-game-launch-drift/) (this is the Madden/2K "cover curse"
framing on EA + TTWO specifically).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a game launch *should* move the publisher if "buy the hype" is right, the launch calendar, the CAR picture, and why the tape says no |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the one-sample and HAC *t*, per-publisher & sub-era splits, the random-window placebo, the jackknife, the costed leg, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`madden_curse/`](madden_curse/). The launch calendar is hand-curated from public
release records; EA, TTWO and SPY are fetched via yfinance (total-return). Survivorship
named on the Signal axis (both publishers still listed); one documented execution lag (0 —
the launch date is calendar-known years ahead). **Not investment advice** — research &
education. See [LICENSE](../../LICENSE).*
