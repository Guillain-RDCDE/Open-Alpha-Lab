# Study 923 — The Cash Lag 💤

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do cash vehicles reprice at different speeds, and does that say which to hold? | ![Mixed](https://img.shields.io/badge/Mixed-dab617?style=flat-square) | The **lag is real and overwhelming**: realised durations of −0.001 (USFR), +0.072 (SGOV), +0.083 (BIL) and +0.177 yr (SHV), |*t*| up to **8.1**, ordered exactly by weighted-average maturity, with pass-through summing to 1 and the same ordering at |*t*| ≥ 2.8 in **both** eras — but it is `duration ≈ WAM/2`, bond arithmetic measured well, not a forecast. The **rotation is absent**: switching by rate direction earns **−6.9 bp/yr gross (HAC *t* = −0.29)**, bootstrap CI [−55, +35], no lookback in the pre-registered grid beats \|*t*\| = 0.53, and the gross sign flips between eras. Split further, the **timing on its own is −18.9 bp/yr (*t* = −0.97)** and the rule's only positive gross component is the passive 55/45 allocation it happens to stand in — which needs no forecast. |
| **Tradability** — is it bankable? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The whole prize — the dispersion between every cash vehicle — is ~16 bp/yr; one round trip costs 4 bps and the rule trades 27×/yr. Net **−115 bp/yr** at 2 bps, still −61 at 1 bp, and a **turnover-matched random placebo loses the same** (−88 to −121 across five seeds, bracketing the rule), so the loss is pure friction. The one bankable finding is a **fund swap, not a trade**: hold SGOV over BIL for **+12.3 bp/yr** (*t* = +3.98; both halves positive, the late one only +7.7 at *t* = +1.96) and never rebalance — one 4 bp round trip, paid back in **3.9 months**, against 27 a year. That arm was **selected ex post** from three static candidates and is defended by a published, ex-ante fee gap, not by its *t*. |

> **In one sentence:** Cash-vehicle yields genuinely do reprice at different speeds — the duration ordering is bond arithmetic and reads off the tape at *t* = 8 — but knowing precisely *how* each one lags tells you nothing about *which* to hold next, because the trailing direction of the bill rate does not forecast its next move; so the rotation earns **zero gross** and −115 bp/yr net, and the only honest way to bank the lag is to buy the shorter, cheaper fund once and stop trading.

## What we tested

For each of **BIL, SGOV, USFR, SHV** we build a *labelled proxy* for realised yield
(trailing 21-day total return, annualised) and (a) regress daily returns on daily **^IRX**
changes for a **window-free** effective duration, (b) run a distributed-lag pass-through
on lags 0-63d, and (c) sweep the proxy window 5/10/21/42d to strip the trailing-average
artefact out of the measured lag. Then the trade: sign of the 21-day ^IRX change through
day *t*, acted at *t+1* — rates up → USFR (zero duration), rates down → SHV (longest
ladder) — long-only, excess of BIL's own total return, 2 bps one-way (a **PROXY**, swept
0-10), against a turnover-matched random placebo, the reversed rule, both static arms, a
pre-registered four-window grid, a block bootstrap, an era cut and an SGOV cross-check.
BIL∩USFR∩SHV∩^IRX 2014-02-04 → 2026-06-30 (3,118 days); these are the *surviving*
large cash ETFs, so the measured dispersion is an upper bound. **Dedup:** distinct from
**921-bill-ladder-vs-etf** (a *simulated* ladder vs one ETF; we race the *listed* vehicles),
**922-frn-vs-fixed-front-end** (the same vehicles *held, never traded*),
**925-short-rate-momentum-switch** (the same ^IRX-direction signal, but cash against *long*
duration), **924-cut-cycle-duration-extension** (four hand-labelled cut events; ours is a
continuous daily signal), **892-corporate-bond-ladder** (ladder folklore in credit) and
**826-treasury-duration-bab** (levered, whole curve; ours moves 18 bps of duration).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why one cash fund is slower than another, what the lag actually buys you, why the switch never pays, and the fund swap that does |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | duration regressions, the distributed-lag profile, the window-artefact sweep, the lookback grid, cost sweep, era cut, bootstrap CI and the live three-world synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`cash_lag/`](cash_lag/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
